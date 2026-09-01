"""Preprocess raw GeoLife GPS traces (.plt) → daily_stays.jsonl + anchors.json.

Stay-point detection uses Zheng et al. (2008) distance+time thresholds.
Output format is identical to the Tencent pipeline (daily_stays.jsonl + anchors.json).

Usage:
    python -m src.data_preprocess.geolife.preprocess_geolife \
        --geolife_dir "/path/to/Geolife Trajectories 1.3/Data" \
        --output_dir data/geolife_beijing/preprocessed/
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

BEIJING_TZ = timezone(timedelta(hours=8))

# ── Stay-point detection (Zheng et al. 2008) ─────────────────────
STAY_DIST_M = 200
STAY_TIME_MIN = 10
MERGE_DIST_M = 200

# ── Filtering ────────────────────────────────────────────────────
MIN_STAYS_PER_DAY = 2
MAX_STEP_KM = 2000

# ── Beijing filter ───────────────────────────────────────────────
BEIJING_BBOX = {"lng_min": 115.4, "lng_max": 117.5,
                "lat_min": 39.4, "lat_max": 41.1}

# ── Home / work detection ────────────────────────────────────────
HW_GRID_DEG = 0.005
HOME_HOURS = set(range(0, 7)) | {22, 23}
WORK_HOURS = set(range(9, 18))
HW_MATCH_KM = 1.0


# ── Geo helpers ──────────────────────────────────────────────────

def _haversine_m(c1, c2):
    lng1, lat1 = c1
    lng2, lat2 = c2
    R = 6_371_000
    la1, lo1, la2, lo2 = map(math.radians, [lat1, lng1, lat2, lng2])
    dlat, dlon = la2 - la1, lo2 - lo1
    a = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(min(1.0, math.sqrt(a)))


def _haversine_km(c1, c2):
    return _haversine_m(c1, c2) / 1000


def _is_beijing(lng, lat):
    b = BEIJING_BBOX
    return b["lng_min"] <= lng <= b["lng_max"] and b["lat_min"] <= lat <= b["lat_max"]


def _get_day_type(date_str: str) -> str:
    from src.data_preprocess.shared.config import get_day_type
    d = datetime.fromisoformat(date_str).date() if isinstance(date_str, str) else date_str
    return get_day_type(d)


# ── .plt parsing ─────────────────────────────────────────────────

def _parse_plt(path: Path) -> list[tuple]:
    """Parse one .plt → [(lng, lat, unix_ts), ...]."""
    points = []
    with open(path, errors="replace") as f:
        for i, line in enumerate(f):
            if i < 6:
                continue
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            try:
                lat = float(parts[0])
                lng = float(parts[1])
                dt = datetime.strptime(f"{parts[5]} {parts[6]}", "%Y-%m-%d %H:%M:%S")
                dt = dt.replace(tzinfo=BEIJING_TZ)
                if lng != 0 and lat != 0:
                    points.append((lng, lat, dt.timestamp()))
            except (ValueError, IndexError):
                continue
    return points


def _load_user_points(user_dir: Path) -> list[tuple]:
    traj_dir = user_dir / "Trajectory"
    if not traj_dir.exists():
        return []
    pts = []
    for f in sorted(traj_dir.glob("*.plt")):
        pts.extend(_parse_plt(f))
    pts.sort(key=lambda p: p[2])
    return pts


# ── Stay-point detection ─────────────────────────────────────────

def _detect_stays(points: list[tuple], dist_m: float, time_min: float) -> list[dict]:
    """Zheng et al. (2008): stay where user remains within dist_m for ≥ time_min."""
    n = len(points)
    if n == 0:
        return []
    stays = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and _haversine_m(points[i][:2], points[j][:2]) <= dist_m:
            j += 1
        dur = (points[j - 1][2] - points[i][2]) / 60
        if dur >= time_min:
            lngs = [p[0] for p in points[i:j]]
            lats = [p[1] for p in points[i:j]]
            stays.append({
                "coord": [sum(lngs) / len(lngs), sum(lats) / len(lats)],
                "t_in": points[i][2],
                "t_out": points[j - 1][2],
            })
            i = j
        else:
            i += 1
    return stays


def _merge_consecutive(stays: list[dict], dist_m: float) -> list[dict]:
    if len(stays) <= 1:
        return stays
    merged = [stays[0].copy()]
    for s in stays[1:]:
        prev = merged[-1]
        if _haversine_m(prev["coord"], s["coord"]) <= dist_m:
            d1 = max(prev["t_out"] - prev["t_in"], 1)
            d2 = max(s["t_out"] - s["t_in"], 1)
            w1, w2 = d1 / (d1 + d2), d2 / (d1 + d2)
            prev["coord"] = [prev["coord"][0] * w1 + s["coord"][0] * w2,
                             prev["coord"][1] * w1 + s["coord"][1] * w2]
            prev["t_out"] = s["t_out"]
        else:
            merged.append(s.copy())
    return merged


# ── Home / work detection ─────────────────────────────────────────

def _detect_home_work(stays: list[dict]) -> dict | None:
    grid: dict[tuple, dict] = defaultdict(
        lambda: {"total": 0.0, "night": 0.0, "work": 0.0, "lngs": [], "lats": []})

    for s in stays:
        lng, lat = s["coord"]
        cell = (round(lat / HW_GRID_DEG) * HW_GRID_DEG,
                round(lng / HW_GRID_DEG) * HW_GRID_DEG)
        dur = (s["t_out"] - s["t_in"]) / 60
        hour = datetime.fromtimestamp(s["t_in"], tz=BEIJING_TZ).hour
        g = grid[cell]
        g["total"] += dur
        g["lngs"].append(lng)
        g["lats"].append(lat)
        if hour in HOME_HOURS:
            g["night"] += dur
        if hour in WORK_HOURS:
            g["work"] += dur

    if not grid:
        return None

    home_cell = max(grid, key=lambda c: grid[c]["night"])
    if grid[home_cell]["night"] == 0:
        home_cell = max(grid, key=lambda c: grid[c]["total"])

    hd = grid[home_cell]
    home = [sum(hd["lngs"]) / len(hd["lngs"]), sum(hd["lats"]) / len(hd["lats"])]

    work_cell = None
    for cell in sorted(grid, key=lambda c: -grid[c]["work"]):
        if cell != home_cell and grid[cell]["work"] > 0:
            work_cell = cell
            break
    if work_cell is None:
        for cell in sorted(grid, key=lambda c: -grid[c]["total"]):
            if cell != home_cell:
                work_cell = cell
                break
    if work_cell is None:
        return None

    wd = grid[work_cell]
    work = [sum(wd["lngs"]) / len(wd["lngs"]), sum(wd["lats"]) / len(wd["lats"])]
    return {"t1_coord": home, "t2_coord": work}


# ── Day splitting + intent ────────────────────────────────────────

def _assign_intent(coord, t1, t2):
    if _haversine_km(coord, t1) <= HW_MATCH_KM:
        return "t1"
    if _haversine_km(coord, t2) <= HW_MATCH_KM:
        return "t2"
    return "other"


def _split_by_day(stays: list[dict]) -> dict[str, list[dict]]:
    days: dict[str, list[dict]] = defaultdict(list)
    for s in stays:
        dt_in = datetime.fromtimestamp(s["t_in"], tz=BEIJING_TZ)
        date_str = dt_in.strftime("%Y-%m-%d")
        start_min = dt_in.hour * 60 + dt_in.minute + dt_in.second / 60
        end_min = start_min + (s["t_out"] - s["t_in"]) / 60
        days[date_str].append({
            "start_min": round(start_min, 1),
            "end_min": min(round(end_min, 1), 1440.0),
            "coord": [round(s["coord"][0], 6), round(s["coord"][1], 6)],
        })
    return dict(days)


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GeoLife .plt → daily_stays.jsonl")
    parser.add_argument("--geolife_dir", required=True,
                        help="Path to Geolife Trajectories 1.3/Data/")
    parser.add_argument("--output_dir", default="data/geolife_beijing/preprocessed/")
    parser.add_argument("--stay_dist_m", type=float, default=STAY_DIST_M)
    parser.add_argument("--stay_time_min", type=float, default=STAY_TIME_MIN)
    parser.add_argument("--min_stays_day", type=int, default=MIN_STAYS_PER_DAY)
    from src.data_preprocess.shared.config import DATASET_CONFIG
    _geo_cfg = DATASET_CONFIG["geolife_beijing"]
    parser.add_argument("--min_days_user", type=int, default=_geo_cfg["min_observed_days"])
    args = parser.parse_args()

    geo_dir = Path(args.geolife_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    user_dirs = sorted(d for d in geo_dir.iterdir() if d.is_dir())
    print(f"[Step 0] {len(user_dirs)} user directories in {geo_dir}")

    all_records: list[dict] = []
    home_work: dict[str, dict] = {}

    drop_no_pts = drop_no_stays = drop_no_hw = 0
    drop_non_bj = drop_few_days = 0
    intent_ctr: Counter = Counter()
    daytype_ctr: Counter = Counter()
    step_dists: list[float] = []

    for ui, user_dir in enumerate(user_dirs):
        uid = user_dir.name
        if (ui + 1) % 20 == 0 or ui == len(user_dirs) - 1:
            print(f"  [{ui+1}/{len(user_dirs)}] processing user {uid}...")

        pts = _load_user_points(user_dir)
        if not pts:
            drop_no_pts += 1
            continue

        stays = _detect_stays(pts, args.stay_dist_m, args.stay_time_min)
        stays = _merge_consecutive(stays, MERGE_DIST_M)
        if not stays:
            drop_no_stays += 1
            continue

        hw = _detect_home_work(stays)
        if hw is None:
            drop_no_hw += 1
            continue

        if not _is_beijing(hw["t1_coord"][0], hw["t1_coord"][1]):
            drop_non_bj += 1
            continue

        day_stays = _split_by_day(stays)
        records: list[dict] = []
        for date_str in sorted(day_stays):
            slist = [s for s in day_stays[date_str]
                     if _haversine_km(s["coord"], hw["t1_coord"]) <= MAX_STEP_KM]
            if len(slist) < args.min_stays_day:
                continue
            for s in slist:
                s["intent_class"] = _assign_intent(s["coord"], hw["t1_coord"], hw["t2_coord"])
            records.append({
                "user_id": uid,
                "date": date_str,
                "day_type": _get_day_type(date_str),
                "stays": slist,
            })

        if len(records) < args.min_days_user:
            drop_few_days += 1
            continue

        home_work[uid] = {
            "t1_coord": [round(hw["t1_coord"][0], 6), round(hw["t1_coord"][1], 6)],
            "t2_coord": [round(hw["t2_coord"][0], 6), round(hw["t2_coord"][1], 6)],
        }
        all_records.extend(records)

        for rec in records:
            daytype_ctr[rec["day_type"]] += 1
            for s in rec["stays"]:
                intent_ctr[s["intent_class"]] += 1
            for i in range(len(rec["stays"]) - 1):
                step_dists.append(_haversine_km(rec["stays"][i]["coord"],
                                               rec["stays"][i + 1]["coord"]))

    # ── Stats ──
    n_users = len(home_work)
    n_days = len(all_records)
    n_stays = sum(len(r["stays"]) for r in all_records)
    step_arr = np.array(step_dists) if step_dists else np.array([0.0])

    report = {
        "source": "raw .plt (Geolife Trajectories 1.3)",
        "stay_params": {"dist_m": args.stay_dist_m, "time_min": args.stay_time_min,
                        "merge_dist_m": MERGE_DIST_M,
                        "min_stays_per_day": args.min_stays_day,
                        "min_days_per_user": args.min_days_user},
        "n_users_raw": len(user_dirs),
        "dropped": {"no_points": drop_no_pts, "no_stays": drop_no_stays,
                     "no_hw": drop_no_hw, "non_beijing": drop_non_bj,
                     "few_days": drop_few_days},
        "n_users_kept": n_users,
        "n_days": n_days,
        "n_stays": n_stays,
        "avg_days_per_user": round(n_days / max(n_users, 1), 1),
        "avg_stays_per_day": round(n_stays / max(n_days, 1), 1),
        "intent_distribution": dict(intent_ctr.most_common()),
        "day_type_distribution": dict(daytype_ctr.most_common()),
        "step_distance_km": {
            "p50": round(float(np.percentile(step_arr, 50)), 2),
            "p95": round(float(np.percentile(step_arr, 95)), 1),
            "max": round(float(step_arr.max()), 1),
        },
    }

    # ── Write ──
    print(f"\n[Output] {out_dir}/")

    with open(out_dir / "daily_stays.jsonl", "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  daily_stays.jsonl  {n_days} records, {n_users} users, {n_stays} stays")

    with open(out_dir / "anchors.json", "w") as f:
        json.dump(home_work, f)
    print(f"  anchors.json       {n_users} users")

    with open(out_dir / "preprocessing_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*55}")
    print("GeoLife Preprocessing Report (raw .plt)")
    print(f"{'='*55}")
    for k, v in report.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
