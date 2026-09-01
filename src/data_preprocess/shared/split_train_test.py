"""Split daily_stays.jsonl into train/test by chronological per-user split.

Supports optional max_N filtering and user downsampling via config.

Usage:
    python -m src.data_preprocess.shared.split_train_test --dataset geolife_beijing
    python -m src.data_preprocess.shared.split_train_test --dataset tencent_beijing
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path

from src.data_preprocess.shared.config import DATASET_CONFIG


def _rebuild_anchors(out_dir: Path, train_uids: set[str]) -> dict:
    """Build filtered anchors dict containing only training users."""
    for name in ["anchors.pkl", "anchors.json", "home_work.pkl", "home_work.json"]:
        p = out_dir / name
        if p.exists():
            if p.suffix == ".pkl":
                import pickle
                with open(p, "rb") as f:
                    hw_all = pickle.load(f)
            else:
                with open(p) as f:
                    hw_all = json.load(f)
            break
    else:
        raise FileNotFoundError(
            f"No anchors or home_work file in {out_dir}. "
            "Run Stage 1 preprocessing first."
        )

    anchor_filtered = {str(uid): v for uid, v in hw_all.items() if str(uid) in train_uids}
    missing = train_uids - set(str(k) for k in hw_all)
    if missing:
        print(f"  WARNING: {len(missing)} users in data but not in anchors, "
              f"skipping: {sorted(missing)[:10]}")
    return anchor_filtered


def _verify_alignment(train_path: Path, anchor_path: Path) -> None:
    """Assert users(train.jsonl) == users(anchors.json)."""
    train_uids: set[str] = set()
    with open(train_path) as f:
        for line in f:
            train_uids.add(json.loads(line)["user_id"])

    with open(anchor_path) as f:
        anchor_uids = set(json.load(f).keys())

    if train_uids != anchor_uids:
        only_train = train_uids - anchor_uids
        only_anchor = anchor_uids - train_uids
        raise AssertionError(
            f"train/anchors user set mismatch: "
            f"{len(only_train)} in train only, {len(only_anchor)} in anchors only. "
            f"Examples: train_only={sorted(only_train)[:3]}, anchor_only={sorted(only_anchor)[:3]}"
        )


def split(jsonl_path: Path, out_dir: Path, train_ratio: float = 0.8,
          min_days: int = 1,
          max_n_threshold: int | None = None,
          sample_size: int | None = None,
          sample_seed: int = 42) -> dict:
    user_records: dict[str, list[dict]] = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            user_records[rec["user_id"]].append(rec)

    n_before = len(user_records)
    if min_days > 1:
        user_records = {u: recs for u, recs in user_records.items()
                        if len(recs) >= min_days}
        print(f"min_days={min_days}: {n_before} → {len(user_records)} users")

    n_after_min_days = len(user_records)
    n_dropped_max_n = 0

    if max_n_threshold is not None:
        filtered = {}
        for uid, recs in user_records.items():
            user_max_n = max(len(r["stays"]) for r in recs)
            if user_max_n <= max_n_threshold:
                filtered[uid] = recs
            else:
                n_dropped_max_n += 1
        user_records = filtered
        print(f"max_n_threshold={max_n_threshold}: {n_after_min_days} → {len(user_records)} users "
              f"({n_dropped_max_n} dropped)")

    n_after_filter = len(user_records)

    if sample_size is not None and len(user_records) > sample_size:
        rng = random.Random(sample_seed)
        sampled_uids = rng.sample(sorted(user_records.keys()), sample_size)
        user_records = {uid: user_records[uid] for uid in sampled_uids}
        print(f"sample_size={sample_size} (seed={sample_seed}): "
              f"{n_after_filter} → {len(user_records)} users")

    for uid in user_records:
        user_records[uid].sort(key=lambda r: r["date"])

    out_dir.mkdir(parents=True, exist_ok=True)
    train_uids = set(user_records.keys())

    anchor_filtered = _rebuild_anchors(out_dir, train_uids)

    anchor_uids = set(anchor_filtered.keys())
    dropped_no_anchor = set(user_records.keys()) - anchor_uids
    if dropped_no_anchor:
        user_records = {uid: recs for uid, recs in user_records.items()
                        if uid in anchor_uids}
        print(f"  Dropped {len(dropped_no_anchor)} users without anchors: "
              f"{len(dropped_no_anchor) + len(user_records)} → {len(user_records)} users")

    train_path = out_dir / "train.jsonl"
    test_path = out_dir / "test.jsonl"
    anchor_out_path = out_dir / "anchors.json"
    n_train = n_test = 0

    tmp_train = tempfile.NamedTemporaryFile(
        mode="w", dir=out_dir, suffix=".tmp", delete=False)
    tmp_test = tempfile.NamedTemporaryFile(
        mode="w", dir=out_dir, suffix=".tmp", delete=False)
    tmp_hw = tempfile.NamedTemporaryFile(
        mode="w", dir=out_dir, suffix=".tmp", delete=False)

    try:
        for uid in sorted(user_records.keys()):
            recs = user_records[uid]
            n_days = len(recs)
            n_train_days = max(1, math.floor(n_days * train_ratio))
            for i, rec in enumerate(recs):
                if i < n_train_days:
                    tmp_train.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_train += 1
                else:
                    tmp_test.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_test += 1

        json.dump(anchor_filtered, tmp_hw)

        tmp_train.close()
        tmp_test.close()
        tmp_hw.close()

        os.replace(tmp_train.name, train_path)
        os.replace(tmp_test.name, test_path)
        os.replace(tmp_hw.name, anchor_out_path)
    except BaseException:
        tmp_train.close()
        tmp_test.close()
        tmp_hw.close()
        for p in (tmp_train.name, tmp_test.name, tmp_hw.name):
            if os.path.exists(p):
                os.unlink(p)
        raise

    _verify_alignment(train_path, anchor_out_path)
    print(f"  anchors.json rebuilt: {len(anchor_filtered)} users")

    stats = {
        "n_users_input": n_before,
        "n_users_after_min_days": n_after_min_days,
        "n_users_dropped_max_n": n_dropped_max_n,
        "max_n_threshold": max_n_threshold,
        "n_users_after_filter": n_after_filter,
        "sample_size": sample_size,
        "sample_seed": sample_seed,
        "n_users": len(user_records),
        "n_train_records": n_train,
        "n_test_records": n_test,
        "train_ratio_target": train_ratio,
        "train_ratio_actual": n_train / (n_train + n_test) if (n_train + n_test) > 0 else 0,
        "min_days": min_days,
    }

    with open(out_dir / "split_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Split: {n_train} train + {n_test} test records ({len(user_records)} users)")
    print(f"  train → {train_path}")
    print(f"  test  → {test_path}")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tencent_beijing")
    parser.add_argument("--train_ratio", type=float, default=None,
                        help="Override train ratio (default: from config or 0.8)")
    parser.add_argument("--min_days", type=int, default=1)
    parser.add_argument("--max_n_threshold", type=int, default=None)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=42)
    args = parser.parse_args()

    ds_cfg = DATASET_CONFIG.get(args.dataset, {})

    train_ratio = args.train_ratio or ds_cfg.get("train_ratio", 0.8)
    max_n = args.max_n_threshold if args.max_n_threshold is not None else ds_cfg.get("user_max_n_threshold")
    sample_sz = args.sample_size if args.sample_size is not None else ds_cfg.get("user_sample_size")
    seed = args.sample_seed if args.sample_seed != 42 else ds_cfg.get("user_sample_seed", 42)

    base = Path("data") / args.dataset / "preprocessed"
    jsonl_path = base / "daily_stays.jsonl"

    split(jsonl_path, base, train_ratio, min_days=args.min_days,
          max_n_threshold=max_n, sample_size=sample_sz, sample_seed=seed)


if __name__ == "__main__":
    main()
