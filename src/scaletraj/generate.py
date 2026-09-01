"""ScaleTraj autoregressive generation."""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import random
from pathlib import Path

import torch

from .model import ScaleTraj
from .tokenizer import (
    DAY_TYPE_VOCAB,
    DURATION_BIN_EDGES_MIN,
    N_DURATION_BINS,
    SPECIAL_TOKENS,
    GridVocab,
    ScaleGMM,
)


def _gmm_scale_boundaries_km(scale_gmm: ScaleGMM) -> list[float]:
    """Compute GMM decision boundaries (km) between adjacent ordered scales.

    Returns N-1 boundaries for N scales, where each boundary is the log-space
    crossing point of adjacent Gaussian responsibilities.
    """
    order = scale_gmm.order
    mus = [scale_gmm.means_log[i] for i in order]
    vars_ = [scale_gmm.variances_log[i] for i in order]
    weights = [scale_gmm.weights[i] for i in order]
    boundaries = []
    for i in range(len(mus) - 1):
        m1, v1, w1 = mus[i], vars_[i], weights[i]
        m2, v2, w2 = mus[i + 1], vars_[i + 1], weights[i + 1]
        s1, s2 = max(v1, 1e-8) ** 0.5, max(v2, 1e-8) ** 0.5
        a = 1 / (2 * v1) - 1 / (2 * v2)
        b = m2 / v2 - m1 / v1
        c = m1**2 / (2 * v1) - m2**2 / (2 * v2) + math.log(s2 / s1) + math.log(max(w1, 1e-12) / max(w2, 1e-12))
        if abs(a) < 1e-10:
            x = -c / b
        else:
            disc = b**2 - 4 * a * c
            x1 = (-b + max(disc, 0) ** 0.5) / (2 * a)
            x2 = (-b - max(disc, 0) ** 0.5) / (2 * a)
            x = min((x1, x2), key=lambda v: abs(v - (m1 + m2) / 2))
        boundaries.append(math.exp(x))
    return boundaries

logger = logging.getLogger(__name__)

def load_model(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> tuple[ScaleTraj, dict[str, int], dict]:
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    user_to_persona = ckpt.get("user_to_persona", {})
    anchor_distributions = ckpt.get("anchor_distributions", {})

    cell_coords = ckpt.get("cell_coords")
    # Pass GMM params so log_cell_density buffer has correct size before state_dict load
    gmm_path = Path(checkpoint_path).parent / "scale_gmm.json"
    gmm_means_log = None
    gmm_variances_log = None
    if gmm_path.exists():
        with open(gmm_path) as gf:
            gmm_data = json.load(gf)
        order = gmm_data["order"]
        means = gmm_data["means_log"]
        vars_ = gmm_data["variances_log"]
        gmm_means_log = [means[i] for i in order]
        gmm_variances_log = [vars_[i] for i in order]
    model = ScaleTraj(
        n_cells=cfg["n_cells"],
        n_scales=cfg["n_scales"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        dropout=cfg.get("dropout", 0.2),
        cell_coords=cell_coords,
        n_train_users=cfg.get("n_train_users", 0),
        gmm_means_log=gmm_means_log,
        gmm_variances_log=gmm_variances_log,
        alpha_res_per_scale=cfg.get("alpha_res_per_scale", None),
        shared_residual=cfg.get("shared_residual", False),
    ).to(device)
    state_dict = ckpt["model_state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    logger.info(
        "Loaded ScaleTraj: %d params, val_loss=%.4f, epoch=%d, %d cells, %d scales",
        sum(p.numel() for p in model.parameters()),
        ckpt.get("val_loss", -1), ckpt.get("epoch", -1),
        cfg["n_cells"], cfg["n_scales"],
    )
    return model, user_to_persona, anchor_distributions

def _tokens_to_trajectory(
    tokens: list[tuple[int, int, int, int]],
    grid_vocab: GridVocab,
    scale_gmm: ScaleGMM,
    user_id: str,
    date_str: str,
    day_type: str,
    rng: random.Random,
    home_cell: int | None = None,
    work_cell: int | None = None,
) -> dict:
    """Convert generated tokens to a trajectory record (same format as train.jsonl)."""
    stays = []
    current_min = 0.0

    for _i, (cell_id, _scale_id, dur_id, arr_id) in enumerate(tokens):
        cell_id = min(cell_id, grid_vocab.n_cells - 1)
        lon, lat = grid_vocab.id_to_cell[cell_id]
        coord = [lon, lat]

        arrival_min = arr_id * 5.0 + 2.5
        if arrival_min < current_min:
            arrival_min = current_min

        dur_lo = DURATION_BIN_EDGES_MIN[min(dur_id, N_DURATION_BINS - 1)]
        dur_hi = DURATION_BIN_EDGES_MIN[min(dur_id + 1, len(DURATION_BIN_EDGES_MIN) - 1)]
        duration_min = (dur_lo + dur_hi) / 2.0
        if duration_min < 5:
            duration_min = 5.0

        end_min = min(arrival_min + duration_min, 1440.0)

        intent = "other"
        if home_cell is not None and cell_id == home_cell:
            intent = "home"
        elif work_cell is not None and cell_id == work_cell:
            intent = "work"

        stays.append({
            "start_min": round(arrival_min, 1),
            "end_min": round(end_min, 1),
            "coord": [round(coord[0], 6), round(coord[1], 6)],
            "intent_class": intent,
        })
        current_min = end_min

    return {
        "user_id": user_id,
        "date": date_str,
        "day_type": day_type,
        "stays": stays,
    }

def _sample_anchor(dist: dict[int, float], rng: random.Random) -> int | None:
    """Sample a cell_id from a probability distribution {cell_id: prob}."""
    if not dist:
        return None
    cells = list(dist.keys())
    probs = list(dist.values())
    r = rng.random()
    cumsum = 0.0
    for cell, prob in zip(cells, probs, strict=False):
        cumsum += prob
        if r <= cumsum:
            return cell
    return cells[-1]

def run_pipeline(
    model_path: str | Path,
    grid_vocab_path: str | Path,
    scale_gmm_path: str | Path,
    train_path: str | Path,
    out_path: Path,
    n_users: int = 100,
    n_days: int = 30,  # baseline-aligned (--ntrajs 30 per user)
    temperature: float = 1.0,  # baseline-aligned (implicit p sampling, no temperature)
    top_k: int = 0,
    seed: int = 42,
    user_list: list[str] | None = None,
    freq_alpha: float = 0.3,
    return_alpha: float = 0.3,
    arrival_mask: str = "hard",
    stagger_dates: bool = False,
    replay_user_codes: bool = False,
    cell_temperature: float | None = None,
    exclude_holidays: bool = False,
    base_date_str: str | None = None,
    calendar_day_type: bool = False,
    test_dates_map: dict[str, list[str]] | None = None,
    eos_threshold: float = 0.5,
    min_stops: int = 0,
) -> list[dict]:
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model, user_to_persona, anchor_distributions = load_model(model_path, device)
    grid_vocab = GridVocab.load(grid_vocab_path)
    scale_gmm = ScaleGMM.load(scale_gmm_path)

    cell_coords = grid_vocab.id_to_cell
    scale_gmm_params = {
        "means_log": scale_gmm.means_log[scale_gmm.order].tolist(),
        "variances_log": scale_gmm.variances_log[scale_gmm.order].tolist(),
    }

    cell_freq_log_global = None
    if freq_alpha > 0:
        counts = torch.tensor(grid_vocab.cell_counts, dtype=torch.float32)
        total = counts.sum()
        cell_freq_log_global = (counts / total).clamp(min=1e-8).log()
        logger.info("Frequency prior: alpha=%.2f, max_log=%.2f, min_log=%.2f",
                     freq_alpha, cell_freq_log_global.max().item(), cell_freq_log_global.min().item())

    user_to_idx = None
    user_idx_anchors: dict[int, tuple[int | None, int | None]] = {}
    if replay_user_codes:
        ckpt_for_meta = torch.load(model_path, map_location="cpu", weights_only=False)
        user_to_idx = ckpt_for_meta.get("user_to_idx", {})
        if user_to_idx:
            from .tokenizer import extract_user_anchors
            user_anchors_map = extract_user_anchors(str(train_path), grid_vocab)
            for uid, idx in user_to_idx.items():
                a = user_anchors_map.get(uid, {})
                user_idx_anchors[idx] = (a.get("home"), a.get("work"))
            n_anchored = sum(1 for h, _ in user_idx_anchors.values() if h is not None)
            logger.info("Replay anchors: %d/%d users have home cells", n_anchored, len(user_to_idx))

    day_types_pool = []
    with open(train_path) as f:
        for line in f:
            rec = json.loads(line)
            dt = rec.get("day_type", "workday")
            if exclude_holidays and dt == "holiday":
                continue
            day_types_pool.append(dt)
    if not day_types_pool:
        day_types_pool = ["workday"]
    if exclude_holidays:
        logger.info("Excluding holidays from day_type pool (%d entries)", len(day_types_pool))

    if test_dates_map is not None:
        # Test-aligned mode: user_list and per-user day counts come from test set
        user_list = list(test_dates_map.keys())
        actual_n_users = len(user_list)
        calendar_day_type = True  # force calendar-derived day_type
        logger.info("Test-aligned mode: %d users, total %d day-records",
                     actual_n_users, sum(len(v) for v in test_dates_map.values()))
    elif user_list is not None:
        actual_n_users = len(user_list)
    else:
        actual_n_users = n_users
        user_list = [f"gen_{i:04d}" for i in range(n_users)]

    persona_ids = list(user_to_persona.values())
    if not persona_ids:
        persona_ids = [0]

    if anchor_distributions:
        n_anchored = sum(1 for p, d in anchor_distributions.items() if d.get("home"))
        logger.info("Anchor distributions: %d personas with home/work priors", n_anchored)

    rng = random.Random(seed)
    date_rng = random.Random(seed + 7)
    if base_date_str:
        base_date = datetime.date.fromisoformat(base_date_str)
    else:
        base_date = datetime.date(2012, 3, 1)
    date_stagger_days = actual_n_users * 5 if stagger_dates else 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_trajectories: list[dict] = []
    with open(out_path, "w") as out_file:
        for u_idx, uid in enumerate(user_list):
            persona_id = user_to_persona.get(uid, rng.choice(persona_ids))
            prev_day_tokens: list[list[tuple[int, int, int, int]]] = []
            user_start_offset = date_rng.randint(0, date_stagger_days) if date_stagger_days > 0 else 0

            cell_freq_log = cell_freq_log_global

            user_home_cell = None
            user_work_cell = None
            if anchor_distributions:
                pdist = anchor_distributions.get(persona_id, {})
                user_home_cell = _sample_anchor(pdist.get("home", {}), rng)
                user_work_cell = _sample_anchor(pdist.get("work", {}), rng)

            train_uidx = None
            if replay_user_codes and model.user_codes is not None:
                train_uidx = rng.randint(0, model.n_train_users - 1)
                user_z = model.user_codes.weight[train_uidx].detach().to(device)
                if user_idx_anchors:
                    h_real, w_real = user_idx_anchors.get(train_uidx, (None, None))
                    if h_real is not None:
                        user_home_cell = h_real
                    if w_real is not None:
                        user_work_cell = w_real
            else:
                user_z = model.sample_persona_latent(persona_id, device=device)

            hist_tokens: list[tuple[int, int, int, int]] = []
            hist_day_boundaries: list[tuple[int, int]] = []

            user_day_count = (
                len(test_dates_map[uid]) if test_dates_map is not None else n_days
            )
            user_date_list = (
                sorted(test_dates_map[uid]) if test_dates_map is not None else None
            )
            for day_idx in range(user_day_count):
                if user_date_list is not None:
                    _d = datetime.date.fromisoformat(user_date_list[day_idx])
                else:
                    _d = base_date + datetime.timedelta(days=user_start_offset + day_idx)
                if calendar_day_type:
                    from chinese_calendar import get_holiday_detail
                    _is_off, _hname = get_holiday_detail(_d)
                    if not _is_off:
                        day_type = "workday"
                    elif _hname is not None and "holiday" in day_types_pool:
                        day_type = "holiday"
                    else:
                        day_type = "weekend"
                else:
                    day_type = rng.choice(day_types_pool)
                date_str = _d.isoformat()
                day_type_idx = DAY_TYPE_VOCAB.index(day_type) if day_type in DAY_TYPE_VOCAB else 0

                sep_cell = grid_vocab.n_cells + SPECIAL_TOKENS["[SEP]"]
                sep_scale = scale_gmm.n_components + SPECIAL_TOKENS["[SEP]"]
                context: list[tuple[int, int, int, int]] = []
                for prev_tokens in prev_day_tokens[-3:]:
                    context.extend(prev_tokens)
                    context.append((sep_cell, sep_scale, 0, 0))

                tokens = model.generate(
                    persona_id=persona_id,
                    day_type=day_type_idx,
                    n_stops=None,
                    max_len=30,
                    eos_threshold=eos_threshold,
                    min_stops=min_stops,
                    temperature=temperature,
                    top_k=top_k,
                    device=device,
                    context_tokens=context if context else None,
                    cell_coords=cell_coords,
                    scale_gmm_params=scale_gmm_params,
                    cell_freq_log=cell_freq_log,
                    freq_alpha=freq_alpha,
                    return_alpha=return_alpha,
                    arrival_mask=arrival_mask,
                    hist_tokens=hist_tokens if hist_tokens else None,
                    hist_day_boundaries=hist_day_boundaries if hist_day_boundaries else None,
                    cell_temperature=cell_temperature,
                    home_cell=user_home_cell,
                    work_cell=user_work_cell,
                    user_latent=user_z,
                )

                if tokens:
                    day_start = len(hist_tokens)
                    hist_tokens.extend(tokens)
                    hist_day_boundaries.append((day_start, len(hist_tokens)))
                    while len(hist_day_boundaries) > 30 or len(hist_tokens) > 200:
                        if not hist_day_boundaries:
                            break
                        d_start, d_end = hist_day_boundaries.pop(0)
                        n_drop = d_end - d_start
                        hist_tokens = hist_tokens[n_drop:]
                        hist_day_boundaries = [(s - n_drop, e - n_drop) for s, e in hist_day_boundaries]

                if not tokens:
                    tokens = [(0, 0, 8, 0)]

                prev_day_tokens.append(tokens)

                traj = _tokens_to_trajectory(
                    tokens, grid_vocab, scale_gmm,
                    uid, date_str, day_type, rng,
                    home_cell=user_home_cell,
                    work_cell=user_work_cell,
                )
                all_trajectories.append(traj)
                out_file.write(json.dumps(traj) + "\n")

            if (u_idx + 1) % 10 == 0 or u_idx == 0:
                logger.info("Progress: %d/%d users done", u_idx + 1, actual_n_users)

    logger.info("Generation complete: %d trajectories → %s", len(all_trajectories), out_path)
    return all_trajectories

def _run_shard(shard_args: tuple) -> Path:
    """Worker function: generate one user shard, return shard file path."""
    shard_id, shard_out, shard_test_dates, shard_user_list, kwargs = shard_args
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [W{shard_id}] %(message)s", datefmt="%H:%M:%S")
    run_pipeline(
        out_path=shard_out,
        user_list=shard_user_list,
        test_dates_map=shard_test_dates,
        **kwargs,
    )
    return shard_out

def _parallel_generate(
    n_workers: int,
    out_path: Path,
    test_dates_map: dict[str, list[str]],
    common_kwargs: dict,
) -> list[dict]:
    """Split test_dates_map across n_workers processes, merge results."""
    import multiprocessing as mp

    users = list(test_dates_map.keys())
    shards = [users[i::n_workers] for i in range(n_workers)]
    shard_dir = out_path.parent / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for i, user_chunk in enumerate(shards):
        if not user_chunk:
            continue
        shard_tdm = {u: test_dates_map[u] for u in user_chunk}
        shard_out = shard_dir / f"shard_{i}.jsonl"
        kw = dict(common_kwargs)
        kw["seed"] = common_kwargs["seed"] + i
        tasks.append((i, shard_out, shard_tdm, None, kw))

    print(f"[parallel] Spawning {len(tasks)} workers for {len(users)} users")
    with mp.Pool(len(tasks)) as pool:
        shard_paths = pool.map(_run_shard, tasks)

    all_trajs = []
    with open(out_path, "w") as fout:
        for sp in shard_paths:
            with open(sp) as fin:
                for line in fin:
                    fout.write(line)
                    all_trajs.append(json.loads(line))
    import shutil
    shutil.rmtree(shard_dir, ignore_errors=True)
    print(f"[parallel] Merged {len(all_trajs)} trajectories from {len(tasks)} shards")
    return all_trajs

def _parallel_generate_userlist(
    n_workers: int,
    out_path: Path,
    user_list: list[str],
    common_kwargs: dict,
) -> list[dict]:
    """Split user_list across n_workers processes, merge results."""
    import multiprocessing as mp

    shards = [user_list[i::n_workers] for i in range(n_workers)]
    shard_dir = out_path.parent / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for i, user_chunk in enumerate(shards):
        if not user_chunk:
            continue
        shard_out = shard_dir / f"shard_{i}.jsonl"
        kw = dict(common_kwargs)
        kw["seed"] = common_kwargs["seed"] + i
        tasks.append((i, shard_out, None, user_chunk, kw))

    print(f"[parallel] Spawning {len(tasks)} workers for {len(user_list)} users")
    with mp.Pool(len(tasks)) as pool:
        shard_paths = pool.map(_run_shard, tasks)

    all_trajs = []
    with open(out_path, "w") as fout:
        for sp in shard_paths:
            with open(sp) as fin:
                for line in fin:
                    fout.write(line)
                    all_trajs.append(json.loads(line))
    import shutil
    shutil.rmtree(shard_dir, ignore_errors=True)
    print(f"[parallel] Merged {len(all_trajs)} trajectories from {len(tasks)} shards")
    return all_trajs

def main():
    parser = argparse.ArgumentParser(description="ScaleTraj generation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--vocab", type=str, default=None)
    parser.add_argument("--gmm", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="geolife_beijing",
                        choices=["geolife_beijing", "tencent_beijing"])
    parser.add_argument("--n_users", type=int, default=None)
    parser.add_argument("--n_days", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--freq_alpha", type=float, default=0.3)
    parser.add_argument("--return_alpha", type=float, default=0.3)
    parser.add_argument("--arrival_mask", type=str, default="hard", choices=["hard", "soft", "none"])
    parser.add_argument("--stagger_dates", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--user_list", type=str, default=None)
    parser.add_argument("--replay_user_codes", action="store_true")
    parser.add_argument("--cell_temperature", type=float, default=0.3)
    parser.add_argument("--exclude_holidays", action="store_true")
    parser.add_argument("--base_date", type=str, default=None)
    parser.add_argument("--calendar_day_type", action="store_true")
    parser.add_argument("--align_test_dates", action="store_true")
    parser.add_argument("--eos_threshold", type=float, default=None)
    parser.add_argument("--min_stops", type=int, default=None)
    parser.add_argument("--gen_workers", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    ROOT = Path(__file__).resolve().parents[2]
    DATASETS = {
        "geolife_beijing": {
            "train": ROOT / "data/geolife_beijing/preprocessed/train.jsonl",
            "test": ROOT / "data/geolife_beijing/preprocessed/test.jsonl",
            "snap_grid_deg": 0.005,
            "n_users": 121,
            "n_days": 30,
            "min_stops": 2,
            "eos_threshold": 0.7,
        },
        "tencent_beijing": {
            "train": ROOT / "data/tencent_beijing/preprocessed/train.jsonl",
            "test": ROOT / "data/tencent_beijing/preprocessed/test.jsonl",
            "snap_grid_deg": 0.005,
            "n_users": 500,
            "n_days": 30,
            "min_stops": 1,
            "eos_threshold": 0.5,
        },
    }
    ds = DATASETS[args.dataset]

    model_dir = Path(args.model).parent
    vocab_path = args.vocab or str(model_dir / "grid_vocab.json")
    gmm_path = args.gmm or str(model_dir / "scale_gmm.json")

    n_users = args.n_users or ds["n_users"]
    n_days = args.n_days or ds["n_days"]
    eos_threshold = args.eos_threshold if args.eos_threshold is not None else ds.get("eos_threshold", 0.5)
    min_stops = args.min_stops if args.min_stops is not None else ds.get("min_stops", 0)
    print(f"[ds={args.dataset}] eos_threshold={eos_threshold}  min_stops={min_stops}")

    user_list_data = None
    if args.user_list:
        with open(args.user_list) as f:
            user_list_data = [line.strip() for line in f if line.strip()]

    test_dates_map = None
    if args.align_test_dates:
        from collections import defaultdict
        test_dates_map = defaultdict(list)
        with open(ds["test"]) as f:
            for line in f:
                rec = json.loads(line)
                test_dates_map[rec["user_id"]].append(rec["date"])
        test_dates_map = dict(test_dates_map)
        print(f"[align_test_dates] loaded {len(test_dates_map)} users, "
              f"{sum(len(v) for v in test_dates_map.values())} day-records from {ds['test']}")

    out_path = Path(args.output) / "trajectories.jsonl"

    common_kwargs = dict(
        model_path=args.model,
        grid_vocab_path=vocab_path,
        scale_gmm_path=gmm_path,
        train_path=str(ds["train"]),
        n_users=n_users,
        n_days=n_days,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        freq_alpha=args.freq_alpha,
        return_alpha=args.return_alpha,
        arrival_mask=args.arrival_mask,
        stagger_dates=args.stagger_dates,
        replay_user_codes=args.replay_user_codes,
        cell_temperature=args.cell_temperature,
        exclude_holidays=args.exclude_holidays,
        base_date_str=args.base_date,
        calendar_day_type=args.calendar_day_type,
        eos_threshold=eos_threshold,
        min_stops=min_stops,
    )

    gen_workers = getattr(args, "gen_workers", 1) or 1
    if gen_workers > 1 and test_dates_map is not None:
        trajectories = _parallel_generate(
            gen_workers, out_path, test_dates_map, common_kwargs,
        )
    elif gen_workers > 1 and user_list_data is not None:
        trajectories = _parallel_generate_userlist(
            gen_workers, out_path, user_list_data, common_kwargs,
        )
    else:
        trajectories = run_pipeline(
            out_path=out_path,
            user_list=user_list_data,
            test_dates_map=test_dates_map,
            **common_kwargs,
        )

    print(f"\nScaleTraj generation done: {len(trajectories)} trajectories → {out_path}")

    if args.evaluate:
        from src.evaluation.mobility_evaluator import (
            MobilityEvaluator,
            build_location_grid,
        )

        eval_dir = Path(args.output) / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        snap_grid = build_location_grid(str(ds["train"]))
        with open(ds["test"]) as f:
            real_trajs = [json.loads(line) for line in f]
        with open(out_path) as f:
            sim_trajs = [json.loads(line) for line in f]

        scale_gmm_eval = ScaleGMM.load(Path(args.model).parent / "scale_gmm.json")
        gmm_boundaries = _gmm_scale_boundaries_km(scale_gmm_eval)
        logger.info("Scale boundaries from GMM: %s km", [f"{b:.1f}" for b in gmm_boundaries])
        evaluator = MobilityEvaluator(output_dir=str(eval_dir), scale_boundaries_km=gmm_boundaries)
        metrics = evaluator.evaluate(
            real_trajs, sim_trajs,
            run_scaling=True, run_plots=False,
            snap_grid=snap_grid,
        )
        metrics_path = eval_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        js = metrics.get("jsd_scores", {})
        macro = metrics.get("macro", {})
        print(f"\n{'='*60}")
        print(f"  ScaleTraj Evaluation: {args.dataset}")
        print(f"{'='*60}")
        print(f"  SD={js.get('SD',0):.4f}  RG={js.get('RG',0):.4f}  "
              f"DTD={js.get('DTD',0):.4f}  DUL={js.get('DUL',0):.4f}")
        print(f"  SI={js.get('SI',0):.4f}  STAY={js.get('STAY',0):.4f}  "
              f"Visit={js.get('Visitation',0):.4f}  IRank={js.get('IRank',0):.4f}")
        print(f"  SFP={js.get('SFP', 0):.4f}")
        print(f"  Metrics saved: {metrics_path}")

if __name__ == "__main__":
    main()
