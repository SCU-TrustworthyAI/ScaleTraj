"""ScaleTraj training loop."""

from __future__ import annotations

import argparse
import json
import logging
import math
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from .data import TrajectoryDataset, collate_fn
from .model import ScaleTraj
from .tokenizer import (
    DURATION_BIN_EDGES_MIN,
    N_DURATION_BINS,
    build_anchor_distributions,
    build_grid_vocab,
    cluster_users,
    fit_scale_gmm,
    tokenize_dataset,
)

logger = logging.getLogger(__name__)


def _fit_global_powerlaw(train_path, x_min=2.0, dr0=1.5):
    """Fit truncated power-law on all displacements >= x_min km.

    Returns (beta, kappa, dr0).
    """
    import json as _json

    import numpy as np
    from scipy.optimize import minimize

    dists = []
    with open(train_path) as f:
        for line in f:
            rec = _json.loads(line)
            stays = rec.get("stays", [])
            for i in range(1, len(stays)):
                c1, c2 = stays[i-1]["coord"], stays[i]["coord"]
                dlat = np.radians(c2[1] - c1[1])
                dlon = np.radians(c2[0] - c1[0])
                a = (np.sin(dlat/2)**2 +
                     np.cos(np.radians(c1[1])) * np.cos(np.radians(c2[1])) *
                     np.sin(dlon/2)**2)
                d_km = 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                if d_km >= x_min:
                    dists.append(d_km)
    data = np.array(dists)

    def neg_ll(params):
        beta, log_kappa = params
        kappa = np.exp(log_kappa)
        if beta < 0.5 or beta > 5.0 or kappa < 1 or kappa > 1e5:
            return 1e15
        ll = -beta * np.sum(np.log(data + dr0)) - np.sum(data) / kappa
        x_grid = np.logspace(np.log10(x_min), np.log10(data.max() * 3), 5000)
        integrand = (x_grid + dr0)**(-beta) * np.exp(-x_grid / kappa)
        log_norm = np.log(np.trapezoid(integrand, x_grid) + 1e-30)
        return -(ll - len(data) * log_norm)

    best = None
    for b0 in [1.2, 1.5, 1.8, 2.2]:
        for lk0 in [2.0, 3.0, 4.0, 5.0]:
            res = minimize(neg_ll, [b0, lk0], method="Nelder-Mead",
                          options={"maxiter": 5000, "xatol": 1e-5})
            if best is None or res.fun < best.fun:
                best = res
    beta = best.x[0]
    kappa = np.exp(best.x[1])
    return beta, kappa, dr0


def _dur_bin_to_log_center(dur_bins: torch.Tensor) -> torch.Tensor:
    edges = DURATION_BIN_EDGES_MIN
    centers = []
    for i in range(N_DURATION_BINS):
        lo = max(edges[i], 1.0)
        hi = edges[min(i + 1, len(edges) - 1)]
        if hi <= lo:
            hi = lo + 10.0
        centers.append(math.log((lo + hi) / 2.0))
    centers_t = torch.tensor(centers, device=dur_bins.device, dtype=torch.float32)
    return centers_t[dur_bins.clamp(max=N_DURATION_BINS - 1)]


def _dist_consistency_loss(
    cell_logits: torch.Tensor,
    prev_cells: torch.Tensor,
    target_cells: torch.Tensor,
    pad_mask: torch.Tensor,
    log_dist: torch.Tensor,
    gmm_mu: torch.Tensor,
    gmm_var: torch.Tensor,
    target_scales: torch.Tensor,
    n_cells: int,
) -> torch.Tensor:
    """Expected log-distance under cell distribution should match scale's GMM center.

    Differentiable penalty: E_p[log_dist(prev, c)] should be close to μ_s for
    the assigned scale s. Penalizes the squared Mahalanobis distance.
    """
    cell_probs = F.softmax(cell_logits[:, :, :n_cells], dim=-1)
    prev_safe = prev_cells.clamp(0, n_cells - 1)
    scale_safe = target_scales.clamp(0, gmm_mu.size(0) - 1)
    prev_log_d = log_dist[prev_safe]
    expected_log_d = (cell_probs * prev_log_d).sum(dim=-1)
    target_mu = gmm_mu[scale_safe]
    target_var = gmm_var[scale_safe].clamp(min=0.01)
    mahal = (expected_log_d - target_mu) ** 2 / target_var
    is_eos = (target_cells >= n_cells).float()
    valid = (~pad_mask).float() * (1.0 - is_eos)
    is_special_prev = (prev_cells >= n_cells).float()
    valid = valid * (1.0 - is_special_prev)
    return (mahal * valid).sum() / valid.sum().clamp(min=1)


def compute_loss(
    logits: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    loss_mask: torch.Tensor,
    n_cells: int,
    n_scales: int,
    label_smoothing: float = 0.1,
    dur_gauss_weight: float = 0.5,
    activity_target: torch.Tensor | None = None,
    activity_weight: float = 0.2,
    daily_stats_target: torch.Tensor | None = None,
    daily_stats_weight: float = 0.1,
    kl_z: torch.Tensor | None = None,
    kl_weight: float = 0.05,
    lambda_ent: float = 0.0,
    day_weights: torch.Tensor | None = None,
    eos_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Multi-head loss: cell + scale + duration + arrival + EOS + activity + daily + KL.

    day_weights: (B, T) per-token weight = 1/n_target_tokens_in_this_day.
        When provided, each day contributes equally to the loss regardless
        of how many trips it contains. When None, uses uniform per-token
        weighting.
    """
    n_cell_out = logits["cell"].size(-1)
    n_scale_out = logits["scale"].size(-1)
    n_dur = logits["duration"].size(-1)
    n_arr = logits["arrival"].size(-1)

    is_eos = (targets["cell"] >= n_cells).float()

    valid = (
        loss_mask
        * (1.0 - is_eos)
    ).float()

    t_cell = targets["cell"].clamp(max=n_cells - 1)
    t_scale = targets["scale"].clamp(max=n_scales - 1)
    t_dur = targets["duration"].clamp(max=n_dur - 1)
    t_arr = targets["arrival"].clamp(max=n_arr - 1)

    losses = {}

    ce_cell = F.cross_entropy(
        logits["cell"].reshape(-1, n_cell_out), t_cell.reshape(-1),
        label_smoothing=label_smoothing, reduction="none",
    ).reshape(valid.shape)

    ce_scale = F.cross_entropy(
        logits["scale"].reshape(-1, n_scale_out), t_scale.reshape(-1),
        label_smoothing=label_smoothing, reduction="none",
    ).reshape(valid.shape)

    ce_dur = F.cross_entropy(
        logits["duration"].reshape(-1, n_dur), t_dur.reshape(-1),
        label_smoothing=label_smoothing, reduction="none",
    ).reshape(valid.shape)

    log_target = _dur_bin_to_log_center(t_dur)
    dur_mu = logits["dur_mu"]
    dur_logsig = logits["dur_logsig"].clamp(-4, 4)
    dur_var = (dur_logsig * 2).exp()
    gauss_nll = 0.5 * (
        (log_target - dur_mu) ** 2 / dur_var.clamp(min=1e-6)
        + dur_logsig * 2 + math.log(2 * math.pi)
    )

    ce_arr = F.cross_entropy(
        logits["arrival"].reshape(-1, n_arr), t_arr.reshape(-1),
        label_smoothing=label_smoothing, reduction="none",
    ).reshape(valid.shape)

    n_valid = valid.sum().clamp(min=1)

    losses["cell"] = (ce_cell * valid).sum() / n_valid
    losses["scale"] = (ce_scale * valid).sum() / n_valid
    losses["dur_cls"] = (ce_dur * valid).sum() / n_valid
    losses["dur_gauss"] = (gauss_nll * valid).sum() / n_valid * dur_gauss_weight
    losses["arrival"] = (ce_arr * valid).sum() / n_valid

    eos_loss = F.binary_cross_entropy_with_logits(
        logits["eos"], is_eos, reduction="none",
    )
    if day_weights is not None:
        eos_w = loss_mask.float() * day_weights
        eos_w_sum = eos_w.sum().clamp(min=1e-8)
        losses["eos"] = (eos_loss * eos_w).sum() / eos_w_sum * eos_weight
    else:
        eos_valid = loss_mask.float()
        n_eos_valid = eos_valid.sum().clamp(min=1)
        losses["eos"] = (eos_loss * eos_valid).sum() / n_eos_valid * eos_weight

    if activity_target is not None and "activity" in logits:
        n_act = logits["activity"].size(-1)
        act_mask = (activity_target < n_act).float() * valid
        t_act = activity_target.clamp(max=n_act - 1)
        ce_act = F.cross_entropy(
            logits["activity"].reshape(-1, n_act), t_act.reshape(-1),
            reduction="none",
        ).reshape(valid.shape)
        losses["activity"] = (ce_act * act_mask).sum() / act_mask.sum().clamp(min=1) * activity_weight

    if daily_stats_target is not None and "daily_stats" in logits:
        mse_daily = F.mse_loss(logits["daily_stats"], daily_stats_target, reduction="mean")
        losses["daily"] = mse_daily * daily_stats_weight

    if kl_z is not None:
        losses["kl_z"] = kl_z * kl_weight

    if lambda_ent > 0 and "_residual_logits" in logits:
        res_logits = logits["_residual_logits"]           # (B, T, n_cells)
        logits["_residual_scales"]            # (B, T)
        res_p = F.softmax(res_logits, dim=-1)
        res_ent = -(res_p * (res_p + 1e-10).log()).sum(dim=-1)  # (B, T)
        max_ent = math.log(res_logits.size(-1))
        neg_ent = (max_ent - res_ent) * valid                    # penalize low entropy
        losses["ent_reg"] = lambda_ent * neg_ent.sum() / n_valid

    total = sum(losses.values())
    breakdown = {k: v.item() for k, v in losses.items()}
    breakdown["total"] = total.item()
    return total, breakdown


def train(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    # Reproducibility: seed Python random, NumPy, and torch (CPU + CUDA).
    # Without this, model init / dropout / schedule sampling drift across runs.
    import random as _random

    import numpy as _np
    _random.seed(args.seed)
    _np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ROOT = Path(__file__).resolve().parents[2]
    DATASETS = {
        "geolife_beijing": ROOT / "data/geolife_beijing/preprocessed/train.jsonl",
        "tencent_beijing": ROOT / "data/tencent_beijing/preprocessed/train.jsonl",
    }
    train_path = DATASETS[args.dataset]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    snap_deg = 0.005
    logger.info("Building grid vocab (snap=%.4f°) ...", snap_deg)
    grid_vocab = build_grid_vocab(train_path, snap_deg=snap_deg, min_count=args.min_cell_count)
    grid_vocab.save(out_dir / "grid_vocab.json")
    logger.info("Grid vocab: %d cells", grid_vocab.n_cells)

    logger.info("Fitting scale GMM (K=%d, max_sigma_log=%s) ...",
                 args.n_scales, args.max_sigma_log)
    scale_gmm = fit_scale_gmm(
        train_path, n_components=args.n_scales,
        max_sigma_log=args.max_sigma_log,
    )
    scale_gmm.save(out_dir / "scale_gmm.json")

    logger.info("Tokenizing %s ...", args.dataset)
    sequences, user_to_idx, user_anchors = tokenize_dataset(train_path, grid_vocab, scale_gmm)
    logger.info("Tokenized %d day-sequences, %d users", len(sequences), len(user_to_idx))

    logger.info("Clustering users into personas ...")
    user_to_persona = cluster_users(train_path, scale_gmm)
    logger.info("Clustered %d users into personas", len(user_to_persona))

    from collections import Counter
    cell_dist = Counter()
    scale_dist = Counter()
    for seq in sequences:
        for tok in seq.tokens:
            cell_dist[tok.cell_id] += 1
            scale_dist[tok.scale] += 1
    total_tok = sum(cell_dist.values())
    top5_cells = cell_dist.most_common(5)
    logger.info("Top-5 cells: %s", [(cid, f"{cnt/total_tok*100:.1f}%") for cid, cnt in top5_cells])
    for i in range(args.n_scales):
        cnt = scale_dist.get(i, 0)
        logger.info("Scale %d (%s): %d (%.1f%%)",
                     i, scale_gmm.scale_names[i] if i < len(scale_gmm.scale_names) else f"s{i}",
                     cnt, cnt / total_tok * 100 if total_tok > 0 else 0)

    n_cells = grid_vocab.n_cells
    n_scales = args.n_scales

    anchor_distributions = build_anchor_distributions(user_to_persona, user_anchors)
    n_anchored = sum(1 for a in user_anchors.values() if a.get("home") is not None)
    logger.info("Anchors: %d/%d users have home cell", n_anchored, len(user_anchors))

    user_to_idx = {uid: i for i, uid in enumerate(sorted(user_to_persona.keys()))}
    n_train_users = len(user_to_idx)
    logger.info("User vocab: %d users", n_train_users)

    # Empirical scale frequency from tokenizer outputs
    empirical_scale_freq = [
        scale_dist.get(i, 0) / max(total_tok, 1) for i in range(n_scales)
    ]
    logger.info("Empirical scale freq (from tokenizer): %s",
                 [f"{f:.4f}" for f in empirical_scale_freq])

    dataset = TrajectoryDataset(
        sequences, user_to_persona,
        n_cells=n_cells,
        n_scales=n_scales,
        user_anchors=user_anchors,
        context_days=args.context_days,
        max_tokens_per_day=args.max_tokens_per_day,
        user_to_idx=user_to_idx,
    )

    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    collate = partial(collate_fn, n_cells=n_cells, n_scales=n_scales)
    nw = args.num_workers
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=nw, drop_last=True,
        pin_memory=(nw > 0), persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=nw,
        pin_memory=(nw > 0), persistent_workers=(nw > 0),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Device: %s", device)

    cell_coords = grid_vocab.id_to_cell
    gmm_means_log = scale_gmm.means_log[scale_gmm.order].tolist()
    gmm_variances_log = scale_gmm.variances_log[scale_gmm.order].tolist()
    alpha_res_per_scale = None

    model = ScaleTraj(
        n_cells=n_cells,
        n_scales=n_scales,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        cell_coords=cell_coords,
        n_train_users=n_train_users,
        gmm_means_log=gmm_means_log,
        gmm_variances_log=gmm_variances_log,
        empirical_scale_freq=empirical_scale_freq,
        alpha_marginal=getattr(args, "alpha_marginal", 1.0),
        alpha_h=getattr(args, "alpha_h", 1.0),
        alpha_dist=args.alpha_dist,
        alpha_res=args.alpha_res,
        alpha_res_per_scale=alpha_res_per_scale,
        disable_jacobian=getattr(args, "disable_jacobian", False),
        shared_residual=getattr(args, "shared_residual", False),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("ScaleTraj: %d parameters (%.2f MB), %d cells, %d scales, %d users",
                n_params, n_params * 4 / 1e6, n_cells, n_scales, n_train_users)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    use_amp = (device.type == "cuda" and args.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        logger.info("AMP enabled (float16)")

    best_val_loss = float("inf")
    patience_counter = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        ss_prob = min(0.5, epoch / args.epochs)
        model.head.ss_prob = ss_prob

        model.train()
        epoch_loss = 0.0
        epoch_breakdown = {}
        n_batches = 0
        step_in_epoch = 0

        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=use_amp):
                z = model.user_codes(batch["user_idx"])
                mu_p = model.mu_persona[batch["persona_id"]]
                logvar_p = model.logvar_persona[batch["persona_id"]].clamp(-4, 4)
                var_p = logvar_p.exp().clamp(min=1e-6)
                kl_z = 0.5 * (((z - mu_p) ** 2) / var_p + logvar_p).sum(dim=-1).mean()

                logits = model(
                    batch["cells"][:, :-1],
                    batch["scales"][:, :-1],
                    batch["durations"][:, :-1],
                    batch["arrivals"][:, :-1],
                    batch["day_type"],
                    batch["persona_id"],
                    batch["pad_mask"][:, :-1],
                    target_cells=batch["cells"][:, 1:],
                    target_scales=batch["scales"][:, 1:],
                    hist_cells=batch["hist_cells"],
                    hist_scales=batch["hist_scales"],
                    hist_durations=batch["hist_durations"],
                    hist_arrivals=batch["hist_arrivals"],
                    hist_day_counts=batch["hist_day_counts"],
                    hist_day_boundaries=batch["hist_day_boundaries"],
                    home_cells=batch["home_cell"],
                    work_cells=batch["work_cell"],
                    user_latent=z,
                    return_residual=(args.lambda_ent > 0),
                )

                day_weights = None
                if getattr(args, "day_norm_loss", False):
                    mask_shifted = (~batch["pad_mask"][:, 1:]).float()
                    ctx_len = batch["ctx_len"]
                    B, T = mask_shifted.shape
                    positions = torch.arange(T, device=mask_shifted.device).unsqueeze(0)
                    target_mask = (positions >= (ctx_len - 1).unsqueeze(1)).float() * mask_shifted
                    n_target = target_mask.sum(dim=1, keepdim=True).clamp(min=1)
                    day_weights = torch.where(
                        target_mask > 0,
                        1.0 / n_target,
                        torch.zeros_like(mask_shifted),
                    )

                loss, breakdown = compute_loss(
                    logits,
                    {
                        "cell": batch["cells"][:, 1:],
                        "scale": batch["scales"][:, 1:],
                        "duration": batch["durations"][:, 1:],
                        "arrival": batch["arrivals"][:, 1:],
                    },
                    (~batch["pad_mask"][:, 1:]).float(),
                    n_cells=n_cells,
                    n_scales=n_scales,
                    label_smoothing=args.label_smoothing,
                    activity_target=batch["activities"][:, 1:],
                    daily_stats_target=batch["daily_stats"],
                    daily_stats_weight=args.daily_stats_weight,
                    kl_z=kl_z,
                    kl_weight=args.kl_weight,
                    lambda_ent=args.lambda_ent,
                    day_weights=day_weights,
                    eos_weight=args.eos_weight,
                )

                if args.dist_consistency_weight > 0:
                    dc_loss = _dist_consistency_loss(
                        logits["cell"], batch["cells"][:, :-1],
                        batch["cells"][:, 1:], batch["pad_mask"][:, 1:],
                        model.log_dist, model.gmm_mu, model.gmm_var,
                        batch["scales"][:, 1:], n_cells,
                    )
                    loss = loss + args.dist_consistency_weight * dc_loss
                    breakdown["dist_con"] = dc_loss.item()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            for k, v in breakdown.items():
                epoch_breakdown[k] = epoch_breakdown.get(k, 0.0) + v
            n_batches += 1
            step_in_epoch += 1

        scheduler.step()
        avg_train = epoch_loss / max(n_batches, 1)
        for k in epoch_breakdown:
            epoch_breakdown[k] /= max(n_batches, 1)

        model.head.ss_prob = 0.0
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for batch in val_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                z = model.user_codes(batch["user_idx"])
                logits = model(
                    batch["cells"][:, :-1],
                    batch["scales"][:, :-1],
                    batch["durations"][:, :-1],
                    batch["arrivals"][:, :-1],
                    batch["day_type"],
                    batch["persona_id"],
                    batch["pad_mask"][:, :-1],
                    target_cells=batch["cells"][:, 1:],
                    target_scales=batch["scales"][:, 1:],
                    hist_cells=batch["hist_cells"],
                    hist_scales=batch["hist_scales"],
                    hist_durations=batch["hist_durations"],
                    hist_arrivals=batch["hist_arrivals"],
                    hist_day_counts=batch["hist_day_counts"],
                    hist_day_boundaries=batch["hist_day_boundaries"],
                    home_cells=batch["home_cell"],
                    work_cells=batch["work_cell"],
                    user_latent=z,
                )
                val_day_weights = None
                if getattr(args, "day_norm_loss", False):
                    mask_shifted = (~batch["pad_mask"][:, 1:]).float()
                    ctx_len = batch["ctx_len"]
                    B, T = mask_shifted.shape
                    positions = torch.arange(T, device=mask_shifted.device).unsqueeze(0)
                    target_mask = (positions >= (ctx_len - 1).unsqueeze(1)).float() * mask_shifted
                    n_target = target_mask.sum(dim=1, keepdim=True).clamp(min=1)
                    val_day_weights = torch.where(
                        target_mask > 0,
                        1.0 / n_target,
                        torch.zeros_like(mask_shifted),
                    )

                loss, _ = compute_loss(
                    logits,
                    {
                        "cell": batch["cells"][:, 1:],
                        "scale": batch["scales"][:, 1:],
                        "duration": batch["durations"][:, 1:],
                        "arrival": batch["arrivals"][:, 1:],
                    },
                    (~batch["pad_mask"][:, 1:]).float(),
                    n_cells=n_cells,
                    n_scales=n_scales,
                    label_smoothing=0.0,
                    activity_target=batch["activities"][:, 1:],
                    daily_stats_target=batch["daily_stats"],
                    daily_stats_weight=args.daily_stats_weight,
                    kl_weight=args.kl_weight,
                    day_weights=val_day_weights,
                    eos_weight=args.eos_weight,
                )
                if args.dist_consistency_weight > 0:
                    dc_loss = _dist_consistency_loss(
                        logits["cell"], batch["cells"][:, :-1],
                        batch["cells"][:, 1:], batch["pad_mask"][:, 1:],
                        model.log_dist, model.gmm_mu, model.gmm_var,
                        batch["scales"][:, 1:], n_cells,
                    )
                    loss = loss + args.dist_consistency_weight * dc_loss
                val_loss += loss.item()
                n_val += 1

        avg_val = val_loss / max(n_val, 1)
        epoch_record = {
            "epoch": epoch, "train_loss": avg_train, "val_loss": avg_val,
            "ss_prob": ss_prob,
            **{f"train_{k}": v for k, v in epoch_breakdown.items()},
        }
        history.append(epoch_record)

        if epoch <= 5 or epoch % 10 == 0 or epoch == args.epochs:
            base_msg = (
                "Epoch %3d | train=%.4f val=%.4f | ss=%.2f | "
                "cell=%.3f scale=%.3f dur=%.3f dg=%.3f arr=%.3f eos=%.3f | lr=%.2e"
            )
            base_args = (
                epoch, avg_train, avg_val, ss_prob,
                epoch_breakdown.get("cell", 0), epoch_breakdown.get("scale", 0),
                epoch_breakdown.get("dur_cls", 0), epoch_breakdown.get("dur_gauss", 0),
                epoch_breakdown.get("arrival", 0), epoch_breakdown.get("eos", 0),
                scheduler.get_last_lr()[0],
            )
            if args.dist_consistency_weight > 0:
                base_msg += " | dc=%.4f"
                base_args += (epoch_breakdown.get("dist_con", 0),)
            logger.info(base_msg, *base_args)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": avg_val,
                "user_to_persona": user_to_persona,
                "user_to_idx": user_to_idx,
                "cell_coords": cell_coords,
                "anchor_distributions": anchor_distributions,
                "config": {
                    "n_cells": n_cells, "n_scales": n_scales,
                    "d_model": args.d_model, "n_heads": args.n_heads,
                    "n_layers": args.n_layers, "dropout": args.dropout,
                    "n_train_users": n_train_users,
                    "alpha_res_per_scale": alpha_res_per_scale,
                    "shared_residual": getattr(args, "shared_residual", False),
                },
            }, out_dir / "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, args.patience)
                break

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Training complete. Best val=%.4f → %s", best_val_loss, out_dir / "best.pt")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train ScaleTraj")
    parser.add_argument("--dataset", type=str, default="geolife_beijing",
                        choices=["geolife_beijing", "tencent_beijing"])
    parser.add_argument("--output", type=str, default="models/default")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_scales", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--context_days", type=int, default=3)
    parser.add_argument("--max_tokens_per_day", type=int, default=20)
    parser.add_argument("--min_cell_count", type=int, default=1)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_jacobian", action="store_true")
    parser.add_argument("--alpha_res", type=float, default=0.5)
    parser.add_argument("--alpha_dist", type=float, default=2.0)
    parser.add_argument("--kl_weight", type=float, default=0.05)
    parser.add_argument("--max_sigma_log", type=float, default=0.7,
                        help="Cap each GMM component's log-sigma to this value during fit. "
                             "Without the cap the macro band absorbs urban+regional and loses "
                             "the heavy-tail structure that the SFP eval partition is built on. "
                             "Set to 'None' (negative) only for diagnostic comparison.")
    parser.add_argument("--lambda_ent", type=float, default=0.0)
    parser.add_argument("--shared_residual", action="store_true")
    parser.add_argument("--day_norm_loss", action="store_true")
    parser.add_argument("--eos_weight", type=float, default=1.0)
    parser.add_argument("--daily_stats_weight", type=float, default=0.1)
    parser.add_argument("--dist_consistency_weight", type=float, default=0.0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
