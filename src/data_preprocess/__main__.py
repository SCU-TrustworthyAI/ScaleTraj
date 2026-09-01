"""Unified data preprocessing pipeline.

Runs Stage 1 (raw → daily_stays.jsonl) → split → (optional relabel) for a dataset.

Usage:
    python -m src.data_preprocess --dataset geolife_beijing
    python -m src.data_preprocess --dataset tencent_beijing \
        --stay_input /path/to/user_stay_points.txt \
        --hw_input /path/to/user_hw.txt
    python -m src.data_preprocess --dataset geolife_beijing --skip_stage1
    python -m src.data_preprocess --dataset geolife_beijing --relabel dbscan
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.data_preprocess.shared.config import DATASET_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified preprocessing pipeline")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG.keys()))
    parser.add_argument("--skip_stage1", action="store_true",
                        help="Skip Stage 1 (raw → daily_stays), only re-split")
    parser.add_argument("--relabel", choices=["grid", "dbscan"], default=None,
                        help="Run frequency-based relabeling after split")
    parser.add_argument("--relabel_eps", type=float, default=300,
                        help="DBSCAN eps in meters (default: 300)")

    parser.add_argument("--geolife_dir", default=None,
                        help="GeoLife raw data dir (required for geolife_beijing Stage 1)")
    parser.add_argument("--stay_input", default=None,
                        help="Tencent stay points file (required for tencent_* Stage 1)")
    parser.add_argument("--hw_input", default=None,
                        help="Tencent home/work file (required for tencent_* Stage 1)")
    args = parser.parse_args()

    ds = args.dataset
    cfg = DATASET_CONFIG[ds]
    out_dir = Path("data") / ds / "preprocessed"
    t0 = time.time()

    # ── Stage 1: raw → daily_stays.jsonl + anchors ──
    if not args.skip_stage1:
        print(f"{'='*55}")
        print(f"[Stage 1] {ds}: raw → daily_stays.jsonl")
        print(f"{'='*55}")

        if ds.startswith("geolife"):
            if args.geolife_dir is None:
                parser.error(f"--geolife_dir required for {ds} Stage 1")
            import sys

            from src.data_preprocess.geolife.preprocess_geolife import (
                main as geolife_main,
            )
            sys.argv = [
                "preprocess_geolife",
                "--geolife_dir", args.geolife_dir,
                "--output_dir", str(out_dir),
            ]
            geolife_main()

        elif ds.startswith("tencent"):
            if args.stay_input is None or args.hw_input is None:
                parser.error(f"--stay_input and --hw_input required for {ds} Stage 1")
            import sys

            from src.data_preprocess.tencent.preprocess_tencent import (
                main as tencent_main,
            )
            sys.argv = [
                "preprocess_tencent",
                "--dataset", ds,
                "--stay_input", args.stay_input,
                "--hw_input", args.hw_input,
            ]
            tencent_main()
        else:
            raise ValueError(f"No Stage 1 handler for dataset: {ds}")

    # ── Stage 2: split ──
    print(f"\n{'='*55}")
    print(f"[Stage 2] {ds}: daily_stays.jsonl → train/test split")
    print(f"{'='*55}")

    from src.data_preprocess.shared.split_train_test import split
    train_ratio = cfg.get("train_ratio", 0.8)
    jsonl_path = out_dir / "daily_stays.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"{jsonl_path} not found. Run Stage 1 first.")

    split(
        jsonl_path, out_dir,
        train_ratio=train_ratio,
        max_n_threshold=cfg.get("user_max_n_threshold"),
        sample_size=cfg.get("user_sample_size"),
        sample_seed=cfg.get("user_sample_seed", 42),
    )

    # ── Stage 3 (optional): frequency-based relabel ──
    if args.relabel:
        print(f"\n{'='*55}")
        print(f"[Stage 3] {ds}: frequency-based relabeling (mode={args.relabel})")
        print(f"{'='*55}")

        from src.data_preprocess.shared.relabel_by_frequency import (
            build_user_ranks,
            find_centroid,
            relabel_file,
            write_anchors,
        )

        train_path = out_dir / "train.jsonl"
        test_path = out_dir / "test.jsonl"

        user_ranks = build_user_ranks(
            train_path, mode=args.relabel, eps_m=args.relabel_eps,
        )
        find_centroid(train_path, user_ranks)
        print(f"  {len(user_ranks)} users ranked")

        train_stats = relabel_file(
            train_path, train_path, user_ranks,
            mode=args.relabel, eps_m=args.relabel_eps,
        )
        print(f"  train: {train_stats}")

        if test_path.exists():
            test_stats = relabel_file(
                test_path, test_path, user_ranks,
                mode=args.relabel, eps_m=args.relabel_eps,
            )
            print(f"  test:  {test_stats}")

        write_anchors(out_dir / "anchors.json", user_ranks)
        print("  anchors.json updated")

    elapsed = time.time() - t0
    print(f"\n[Done] {ds} pipeline finished in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
