"""Preprocess raw Tencent CDR data (user_stay_points.txt + user_hw.txt)
→ daily_stays.jsonl + anchors.pkl.

Four stages, run in sequence by main():

  1.1   load_tencent_stays  — parse raw text files, drop bad rows
  1.2   detect_anchors      — home/work cell from the hw hint file
  1.2.5 denoise_fragments   — merge CDR ping-pong on same cell
  1.3+1.4 label_and_filter  — intent labels, filter users/days, write outputs

The equivalent GeoLife pipeline lives in ../geolife/preprocess_geolife.py.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import json
import math
import pickle
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from src.data_preprocess.shared.config import DATASET_CONFIG, get_day_type

BEIJING_TZ = datetime.timezone(datetime.timedelta(hours=8))


# ── helpers ────────────────────────────────────────────────────────────────

def _cell_id(lng: float, lat: float, grid: float) -> tuple[int, int]:
    return int(math.floor(lng / grid)), int(math.floor(lat / grid))


def _haversine_km(c1: list[float], c2: list[float]) -> float:
    lng1, lat1 = math.radians(c1[0]), math.radians(c1[1])
    lng2, lat2 = math.radians(c2[0]), math.radians(c2[1])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


# ── Stage 1.1: raw load & validation ──────────────────────────────────────

def load_tencent_stays(
    cfg: dict,
) -> tuple[dict[str, list[dict]], dict[str, dict], dict[str, int]]:
    """Parse user_stay_points.txt and user_hw.txt.

    Returns
    -------
    user_stays : {user_id: [stay_record, ...]} — one flat list per user.
    hw : {user_id: {"home": [lng, lat] | None, "work": [lng, lat] | None}}.
    stats : counters (n_users_loaded, n_stays_loaded, n_dropped_coord,
        n_dropped_duration, n_hw_entries, n_cross_boundary_splits).
    """
    stays_path = cfg["stay_input_path"]
    hw_path = cfg["hw_input_path"]
    day_start = cfg.get("day_start_hour", 0)

    # --- stays ---
    user_stays: dict[str, list[dict]] = {}
    n_dropped_coord = 0
    n_dropped_duration = 0
    n_cross_boundary_splits = 0

    with open(stays_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            uid = parts[0]
            raw_stays = ast.literal_eval(parts[1])
            for s in raw_stays:
                lng, lat, t_in, t_out = s
                if lng == 0 and lat == 0:
                    n_dropped_coord += 1
                    continue
                if t_in >= t_out:
                    n_dropped_duration += 1
                    continue

                dt_in = datetime.datetime.fromtimestamp(t_in, tz=BEIJING_TZ)
                dt_out = datetime.datetime.fromtimestamp(t_out, tz=BEIJING_TZ)
                shifted_in = dt_in - datetime.timedelta(hours=day_start)
                date_str = shifted_in.strftime("%Y-%m-%d")
                day_type = get_day_type(date_str)
                start_min = max(((dt_in.hour - day_start) % 24) * 60 + dt_in.minute, 0.0)
                end_min_raw = (dt_out - dt_in).total_seconds() / 60.0 + start_min

                if end_min_raw > 1440:
                    n_cross_boundary_splits += 1

                rec = {
                    "coord": [lng, lat],
                    "t_in": t_in,
                    "t_out": t_out,
                    "date": date_str,
                    "day_type": day_type,
                    "start_min": round(start_min, 1),
                    "end_min": min(round(end_min_raw, 1), 1440.0),
                }
                user_stays.setdefault(uid, []).append(rec)

    # --- hw ---
    hw: dict[str, dict] = {}
    with open(hw_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = ast.literal_eval(line)
            uid = str(entry["user_id"])
            home = entry.get("home")
            work = entry.get("work")
            if isinstance(home, list) and len(home) == 2 and home[0] != 0:
                hw_home = home
            else:
                hw_home = None
            if isinstance(work, list) and len(work) == 2 and work[0] != 0:
                hw_work = work
            else:
                hw_work = None
            hw[uid] = {"home": hw_home, "work": hw_work}

    stats = {
        "n_users_loaded": len(user_stays),
        "n_stays_loaded": sum(len(v) for v in user_stays.values()),
        "n_dropped_coord": n_dropped_coord,
        "n_dropped_duration": n_dropped_duration,
        "n_hw_entries": len(hw),
        "n_cross_boundary_splits": n_cross_boundary_splits,
    }
    return user_stays, hw, stats


# ── Stage 1.2: anchor detection ───────────────────────────────────────────

def detect_anchors(
    user_stays: dict[str, list[dict]],
    hw: dict[str, dict],
    cfg: dict,
) -> tuple[dict[str, dict], dict]:
    grid = cfg["anchor_grid_size_deg"]

    anchors: dict[str, dict] = {}
    n_no_hw = 0

    for uid in user_stays:
        if uid not in hw:
            n_no_hw += 1
            continue

        t1_coord = hw[uid]["home"]
        t2_coord = hw[uid]["work"]

        if t1_coord is None:
            n_no_hw += 1
            continue

        t1_cell = _cell_id(t1_coord[0], t1_coord[1], grid)
        t2_cell = _cell_id(t2_coord[0], t2_coord[1], grid) if t2_coord else None

        anchors[uid] = {
            "t1_coord": t1_coord,
            "t1_cell": t1_cell,
            "t2_coord": t2_coord,
            "t2_cell": t2_cell,
        }

    stats = {
        "n_users_with_anchors": len(anchors),
        "n_no_hw_entry": n_no_hw,
    }
    return anchors, stats


# ── Stage 1.2.5: CDR fragmentation denoising ─────────────────────────────
#
# Merge consecutive same-coord stays on the same date. Short-gap same-cell
# pairs are CDR reporting artifacts (cell-tower ping-pong); longer gaps are
# genuine same-cell returns and should stay as separate stays.

def denoise_fragments(
    user_stays: dict[str, list[dict]],
    anchors: dict[str, dict],
    cfg: dict,
) -> tuple[dict[str, list[dict]], int]:
    """Merge consecutive same-coord, same-date stays.

    Returns the modified user_stays (in-place) and total merge count.
    """
    total_merges = 0

    for uid, stays in user_stays.items():
        stays.sort(key=lambda s: s["t_in"])

        merged: list[dict] = []
        for s in stays:
            if not merged:
                merged.append(s.copy())
                continue

            prev = merged[-1]
            if prev["coord"] == s["coord"] and prev["date"] == s["date"]:
                prev["t_out"] = s["t_out"]
                prev["end_min"] = s["end_min"]
                total_merges += 1
            else:
                merged.append(s.copy())

        user_stays[uid] = merged

    return user_stays, total_merges


# ── Stage 1.3+1.4: intent labelling, filtering, output ────────────────────

def label_and_filter(
    user_stays: dict[str, list[dict]],
    anchors: dict[str, dict],
    cfg: dict,
    out_dir: Path,
) -> dict:
    """Label intents (t1 / t2 / other), filter users and days, write outputs.

    Returns a preprocessing_report dict summarising drops.
    """
    grid = cfg["anchor_grid_size_deg"]
    min_days = cfg["min_observed_days"]
    max_median_step = cfg["max_median_step_km"]
    min_span_hours = cfg["min_user_day_span_hours"]

    # --- 1.3: Intent labelling ---
    for uid, stays in user_stays.items():
        if uid not in anchors:
            for s in stays:
                s["intent_class"] = "other"
            continue

        ua = anchors[uid]
        t1_cell = ua["t1_cell"]
        t2_cell = ua["t2_cell"]

        for s in stays:
            cid = _cell_id(s["coord"][0], s["coord"][1], grid)
            if cid == t1_cell:
                s["intent_class"] = "t1"
            elif cid == t2_cell:
                s["intent_class"] = "t2"
            else:
                s["intent_class"] = "other"

    # --- 1.4: Filtering ---
    n_users_in = len(user_stays)
    drop_few_days = 0
    drop_no_home = 0
    drop_median_step = 0
    n_user_days_in = 0
    n_user_days_dropped_span = 0
    n_user_days_dropped_empty = 0

    kept_users: dict[str, dict] = {}

    for uid, stays in user_stays.items():
        if uid not in anchors:
            drop_no_home += 1
            continue

        by_date: dict[str, list[dict]] = defaultdict(list)
        for s in stays:
            by_date[s["date"]].append(s)

        # Min observed days
        if len(by_date) < min_days:
            drop_few_days += 1
            continue

        # Median step distance
        step_dists = []
        sorted_stays = sorted(stays, key=lambda s: s["t_in"])
        for i in range(1, len(sorted_stays)):
            d = _haversine_km(sorted_stays[i - 1]["coord"], sorted_stays[i]["coord"])
            step_dists.append(d)
        if step_dists:
            median_step = statistics.median(step_dists)
            if median_step > max_median_step:
                drop_median_step += 1
                continue

        # Filter user-days
        kept_days: dict[str, list[dict]] = {}
        for date_str, day_stays in by_date.items():
            n_user_days_in += 1
            if len(day_stays) < 1:
                n_user_days_dropped_empty += 1
                continue
            day_stays_sorted = sorted(day_stays, key=lambda s: s["start_min"])
            span_min = day_stays_sorted[-1]["end_min"] - day_stays_sorted[0]["start_min"]
            if span_min < min_span_hours * 60:
                n_user_days_dropped_span += 1
                continue
            kept_days[date_str] = day_stays_sorted

        if len(kept_days) < min_days:
            drop_few_days += 1
            continue

        kept_users[uid] = kept_days

    # --- Write outputs ---
    out_dir.mkdir(parents=True, exist_ok=True)

    # daily_stays.jsonl
    jsonl_path = out_dir / "daily_stays.jsonl"
    n_lines = 0
    with open(jsonl_path, "w") as f:
        for uid, days in sorted(kept_users.items()):
            for date_str in sorted(days.keys()):
                day_stays = days[date_str]
                day_type = day_stays[0]["day_type"]
                record = {
                    "user_id": uid,
                    "date": date_str,
                    "day_type": day_type,
                    "stays": [
                        {
                            "start_min": s["start_min"],
                            "end_min": s["end_min"],
                            "coord": s["coord"],
                            "intent_class": s["intent_class"],
                        }
                        for s in day_stays
                    ],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_lines += 1

    # anchors.pkl
    anchor_out = {}
    for uid in kept_users:
        a = anchors[uid]
        anchor_out[uid] = {
            "t1_coord": a["t1_coord"],
            "t1_cell": a["t1_cell"],
            "t2_coord": a["t2_coord"],
            "t2_cell": a["t2_cell"],
        }
    with open(out_dir / "anchors.pkl", "wb") as f:
        pickle.dump(anchor_out, f)

    report = {
        "n_users_in": n_users_in,
        "n_users_kept": len(kept_users),
        "n_users_dropped_by_reason": {
            "fewer_than_10_days": drop_few_days,
            "no_home_anchor": drop_no_home,
            "median_step_too_large": drop_median_step,
        },
        "n_user_days_in": n_user_days_in,
        "n_user_days_kept": n_lines,
        "n_user_days_dropped_by_reason": {
            "span_too_short": n_user_days_dropped_span,
            "empty": n_user_days_dropped_empty,
        },
    }

    with open(out_dir / "preprocessing_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


# ── Main pipeline ─────────────────────────────────────────────────────────

def _stays_per_day(user_stays: dict[str, list[dict]]) -> list[int]:
    """Count stays per (user, date) — used for pre/post fragmentation diagnostics."""
    counts = []
    for _uid, stays in user_stays.items():
        by_date: dict[str, int] = defaultdict(int)
        for s in stays:
            t_in = s.get("t_in") or ""
            d = s.get("date") or t_in[:10]
            by_date[d] += 1
        counts.extend(by_date.values())
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tencent CDR preprocessing: raw stay points → daily_stays.jsonl",
    )
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG.keys()))
    parser.add_argument("--stay_input", required=True,
                        help="Path to raw stay points file (e.g. user_stay_points.txt)")
    parser.add_argument("--hw_input", required=True,
                        help="Path to raw home/work file (e.g. user_hw.txt)")
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    cfg["stay_input_path"] = args.stay_input
    cfg["hw_input_path"] = args.hw_input

    out_dir = Path("data") / args.dataset / "preprocessed"
    t0 = time.time()

    # 1.1 — Raw load & validation
    print(f"[Stage 1.1] Loading {args.dataset}...")
    user_stays, hw, load_stats = load_tencent_stays(cfg)
    print(f"  {load_stats['n_users_loaded']} users, {load_stats['n_stays_loaded']} stays loaded")
    print(f"  Dropped: {load_stats['n_dropped_coord']} coord, {load_stats['n_dropped_duration']} duration")

    # 1.2 — Anchor detection
    print("[Stage 1.2] Detecting anchors...")
    anchors, anchor_stats = detect_anchors(user_stays, hw, cfg)
    print(f"  {anchor_stats['n_users_with_anchors']} users with anchors")
    print(f"  Dropped: {anchor_stats['n_no_hw_entry']} no-hw-entry")

    # 1.2.5 — CDR fragmentation denoising
    print("[Stage 1.2.5] CDR fragmentation denoising...")
    pre_counts = _stays_per_day(user_stays)
    pre_q = statistics.quantiles(pre_counts, n=4) if pre_counts else [0, 0, 0]
    print(f"  [pre-merge] stays_per_day  p25={pre_q[0]:.0f}  p50={pre_q[1]:.0f}  "
          f"p75={pre_q[2]:.0f}  n_user_days={len(pre_counts)}")

    user_stays, n_frag_merges = denoise_fragments(user_stays, anchors, cfg)

    post_counts = _stays_per_day(user_stays)
    post_q = statistics.quantiles(post_counts, n=4) if post_counts else [0, 0, 0]
    print(f"  [post-merge] stays_per_day p25={post_q[0]:.0f}  p50={post_q[1]:.0f}  "
          f"p75={post_q[2]:.0f}  n_user_days={len(post_counts)}")
    print(f"  {n_frag_merges} fragment merges")

    # 1.3 + 1.4 — Label + filter + write
    print("[Stage 1.3+1.4] Labelling, filtering, writing...")
    report = label_and_filter(user_stays, anchors, cfg, out_dir)

    elapsed = time.time() - t0
    report["n_frag_merges"] = n_frag_merges
    report["load_stats"] = load_stats
    report["anchor_stats"] = anchor_stats
    report["elapsed_seconds"] = round(elapsed, 1)

    with open(out_dir / "preprocessing_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[Stage 1] Done in {elapsed:.1f}s")
    print(f"  Users: {report['n_users_in']} → {report['n_users_kept']}")
    print(f"  User-days: {report['n_user_days_in']} → {report['n_user_days_kept']}")
    print(f"  Outputs: {out_dir}/")


if __name__ == "__main__":
    main()
