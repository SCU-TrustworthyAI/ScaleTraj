"""Relabel intent_class by visit frequency instead of time-rule-based inference.

PRIMARY   (→ "t1")    = user's most frequently visited location
SECONDARY (→ "t2")    = user's second most frequently visited location
OTHER     (→ "other") = everything else

This makes the T1/T2/O framework robust for check-in data where
time-based inference is unreliable.

Supports two clustering modes:
  - grid: fixed lat/lon grid cells (fast, but boundary artifacts)
  - dbscan: DBSCAN on haversine distance (robust, no boundary artifacts)
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN


def _cell(coord: list[float], grid: float = 0.001) -> tuple[int, int]:
    return int(math.floor(coord[0] / grid)), int(math.floor(coord[1] / grid))


def _haversine_m(c1: list[float], c2: list[float]) -> float:
    lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
    lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371000 * 2 * math.asin(math.sqrt(a))


def _dbscan_cluster_user(
    coords: list[list[float]], eps_m: float = 300, min_samples: int = 1
) -> list[int]:
    """Cluster a user's visit coordinates with DBSCAN on haversine distance."""
    if len(coords) == 0:
        return []
    arr = np.radians(np.array(coords))
    db = DBSCAN(eps=eps_m / 6371000, min_samples=min_samples, metric="haversine")
    return db.fit_predict(arr).tolist()


def build_user_ranks(
    train_path: Path,
    grid: float = 0.001,
    mode: str = "grid",
    eps_m: float = 300,
) -> dict[str, dict]:
    """Build per-user frequency-ranked locations from training data."""
    if mode == "dbscan":
        return _build_user_ranks_dbscan(train_path, eps_m)
    return _build_user_ranks_grid(train_path, grid)


def _build_user_ranks_grid(
    train_path: Path,
    grid: float = 0.001,
) -> dict[str, dict]:
    user_visits: dict[str, Counter] = defaultdict(Counter)

    with open(train_path) as f:
        for line in f:
            rec = json.loads(line)
            uid = rec["user_id"]
            for s in rec["stays"]:
                cell = _cell(s["coord"], grid)
                user_visits[uid][cell] += 1

    user_ranks: dict[str, dict] = {}
    for uid, ctr in user_visits.items():
        ranked = ctr.most_common()
        if len(ranked) < 1:
            continue
        primary_cell = ranked[0][0]
        secondary_cell = ranked[1][0] if len(ranked) > 1 else primary_cell
        user_ranks[uid] = {
            "primary_cell": primary_cell,
            "secondary_cell": secondary_cell,
            "primary_coord": None,
            "secondary_coord": None,
            "primary_visits": ranked[0][1],
            "secondary_visits": ranked[1][1] if len(ranked) > 1 else 0,
            "mode": "grid",
        }
    return user_ranks


def _build_user_ranks_dbscan(
    train_path: Path,
    eps_m: float = 300,
) -> dict[str, dict]:
    user_coords: dict[str, list[list[float]]] = defaultdict(list)

    with open(train_path) as f:
        for line in f:
            rec = json.loads(line)
            uid = rec["user_id"]
            for s in rec["stays"]:
                user_coords[uid].append(s["coord"])

    user_ranks: dict[str, dict] = {}
    for uid, coords in user_coords.items():
        labels = _dbscan_cluster_user(coords, eps_m)
        cluster_counts: Counter = Counter()
        cluster_coords: dict[int, list[list[float]]] = defaultdict(list)
        for coord, lab in zip(coords, labels, strict=False):
            cluster_counts[lab] += 1
            cluster_coords[lab].append(coord)

        ranked = cluster_counts.most_common()
        if len(ranked) < 1:
            continue

        pc_id = ranked[0][0]
        sc_id = ranked[1][0] if len(ranked) > 1 else pc_id

        def _centroid(clist):
            return [
                sum(c[0] for c in clist) / len(clist),
                sum(c[1] for c in clist) / len(clist),
            ]

        user_ranks[uid] = {
            "primary_cell": pc_id,
            "secondary_cell": sc_id,
            "primary_coord": _centroid(cluster_coords[pc_id]),
            "secondary_coord": _centroid(cluster_coords[sc_id]),
            "primary_visits": ranked[0][1],
            "secondary_visits": ranked[1][1] if len(ranked) > 1 else 0,
            "mode": "dbscan",
            "_cluster_coords": cluster_coords,
        }
    return user_ranks


def find_centroid(
    train_path: Path,
    user_ranks: dict[str, dict],
    grid: float = 0.001,
) -> None:
    """Find centroid coordinates for primary/secondary cells (grid mode only)."""
    if user_ranks and next(iter(user_ranks.values())).get("mode") == "dbscan":
        for _uid, ranks in user_ranks.items():
            ranks.pop("_cluster_coords", None)
        return

    cell_coords: dict[str, dict[tuple, list[list[float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    with open(train_path) as f:
        for line in f:
            rec = json.loads(line)
            uid = rec["user_id"]
            if uid not in user_ranks:
                continue
            for s in rec["stays"]:
                cell = _cell(s["coord"], grid)
                cell_coords[uid][cell].append(s["coord"])

    for uid, ranks in user_ranks.items():
        pc = ranks["primary_cell"]
        sc = ranks["secondary_cell"]

        if pc in cell_coords[uid]:
            coords = cell_coords[uid][pc]
            ranks["primary_coord"] = [
                sum(c[0] for c in coords) / len(coords),
                sum(c[1] for c in coords) / len(coords),
            ]

        if sc in cell_coords[uid]:
            coords = cell_coords[uid][sc]
            ranks["secondary_coord"] = [
                sum(c[0] for c in coords) / len(coords),
                sum(c[1] for c in coords) / len(coords),
            ]

        if ranks["primary_coord"] is None:
            ranks["primary_coord"] = [
                (pc[0] + 0.5) * grid,
                (pc[1] + 0.5) * grid,
            ]
        if ranks["secondary_coord"] is None:
            ranks["secondary_coord"] = ranks["primary_coord"]


def relabel_file(
    input_path: Path,
    output_path: Path,
    user_ranks: dict[str, dict],
    grid: float = 0.001,
    mode: str = "grid",
    eps_m: float = 300,
) -> dict[str, int]:
    """Relabel a jsonl file using frequency-based ranks."""
    stats: Counter[str] = Counter()

    with open(input_path) as fin:
        records = [json.loads(line) for line in fin]

    if mode == "dbscan":
        _relabel_records_dbscan(records, user_ranks, eps_m, stats)
    else:
        _relabel_records_grid(records, user_ranks, grid, stats)

    with open(output_path, "w") as fout:
        for rec in records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return dict(stats)


def _relabel_records_grid(records, user_ranks, grid, stats):
    for rec in records:
        uid = rec["user_id"]
        if uid not in user_ranks:
            continue
        ranks = user_ranks[uid]
        pc = ranks["primary_cell"]
        sc = ranks["secondary_cell"]
        for s in rec["stays"]:
            cell = _cell(s["coord"], grid)
            if cell == pc:
                s["intent_class"] = "t1"
                stats["primary"] += 1
            elif cell == sc:
                s["intent_class"] = "t2"
                stats["secondary"] += 1
            else:
                s["intent_class"] = "other"
                stats["other"] += 1


def _relabel_records_dbscan(records, user_ranks, eps_m, stats):
    for rec in records:
        uid = rec["user_id"]
        if uid not in user_ranks:
            continue
        ranks = user_ranks[uid]
        p_coord = ranks["primary_coord"]
        s_coord = ranks["secondary_coord"]
        for s in rec["stays"]:
            d_p = _haversine_m(s["coord"], p_coord)
            d_s = _haversine_m(s["coord"], s_coord)
            if d_p <= eps_m:
                s["intent_class"] = "t1"
                stats["primary"] += 1
            elif d_s <= eps_m:
                s["intent_class"] = "t2"
                stats["secondary"] += 1
            else:
                s["intent_class"] = "other"
                stats["other"] += 1


def write_anchors(
    output_path: Path,
    user_ranks: dict[str, dict],
) -> None:
    """Write anchors.json with frequency-based primary/secondary coords."""
    anchors = {}
    for uid, ranks in user_ranks.items():
        anchors[uid] = {
            "t1_coord": ranks["primary_coord"],
            "t2_coord": ranks["secondary_coord"],
        }
    with open(output_path, "w") as f:
        json.dump(anchors, f, indent=2)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Relabel intent_class by visit frequency"
    )
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--mode", choices=["grid", "dbscan"], default="dbscan")
    parser.add_argument("--grid", type=float, default=0.005)
    parser.add_argument("--eps", type=float, default=300, help="DBSCAN eps in meters")
    args = parser.parse_args()

    data_dir = args.dataset_dir
    train_path = data_dir / "train.jsonl"
    test_path = data_dir / "test.jsonl"
    hw_path = data_dir / "anchors.json"

    mode = args.mode
    grid = args.grid
    eps_m = args.eps

    label = f"mode={mode}, eps={eps_m}m" if mode == "dbscan" else f"mode={mode}, grid={grid}"
    print(f"Building frequency ranks from {train_path} ({label})...")
    user_ranks = build_user_ranks(train_path, grid, mode=mode, eps_m=eps_m)
    find_centroid(train_path, user_ranks, grid)
    print(f"  {len(user_ranks)} users")

    for uid in list(user_ranks.keys())[:3]:
        r = user_ranks[uid]
        print(f"  {uid}: primary={r['primary_visits']} visits, secondary={r['secondary_visits']} visits")

    top2_cov = []
    for _uid, r in user_ranks.items():
        total_v = r["primary_visits"] + r["secondary_visits"]
        top2_cov.append(total_v)
    print(f"  mean top-2 visits: {np.mean(top2_cov):.1f}")

    # Backup originals (only from timerule_bak if exists, else from current)
    for p in [train_path, test_path, hw_path]:
        bak = p.with_suffix(p.suffix + ".timerule_bak")
        if not bak.exists() and p.exists():
            import shutil
            shutil.copy2(p, bak)
            print(f"  Backed up {p.name} → {bak.name}")

    print(f"\nRelabeling {train_path}...")
    train_stats = relabel_file(train_path, train_path, user_ranks, grid, mode=mode, eps_m=eps_m)
    total = sum(train_stats.values())
    print(f"  primary={train_stats.get('primary',0)} ({train_stats.get('primary',0)/total*100:.1f}%)")
    print(f"  secondary={train_stats.get('secondary',0)} ({train_stats.get('secondary',0)/total*100:.1f}%)")
    print(f"  other={train_stats.get('other',0)} ({train_stats.get('other',0)/total*100:.1f}%)")

    if test_path.exists():
        print(f"\nRelabeling {test_path}...")
        test_stats = relabel_file(test_path, test_path, user_ranks, grid, mode=mode, eps_m=eps_m)
        total = sum(test_stats.values())
        print(f"  primary={test_stats.get('primary',0)} ({test_stats.get('primary',0)/total*100:.1f}%)")
        print(f"  secondary={test_stats.get('secondary',0)} ({test_stats.get('secondary',0)/total*100:.1f}%)")
        print(f"  other={test_stats.get('other',0)} ({test_stats.get('other',0)/total*100:.1f}%)")

    print(f"\nWriting {hw_path}...")
    write_anchors(hw_path, user_ranks)
    print("Done.")


if __name__ == "__main__":
    main()
