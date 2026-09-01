"""Hierarchical Mobility Evaluator.

11-metric evaluation framework matching the paper's Table 3:

  Spatial (2)
      SD   -- Step Distance JSD            P(Dd) distribution (log-binned)
      RG   -- Radius of Gyration JSD       JSD of per-trajectory R_g values

  Temporal (2)
      SI   -- Step Interval JSD            P(Dt) distribution
      STAY -- Stay Duration JSD            P(stay) distribution

  Behavioural (4)
      DTD  -- Daily Travel Distance        total km per trajectory (JSD + mean/std)
      DUL  -- Daily Unique Locations       distinct places per trajectory (JSD + mean/std)
      Visitation -- cell-visit frequency   per-user Zipf ζ (f_k ~ k^{-ζ}) JSD
      IRank -- individual-rank correlation top-3 visitation proportion JSD

  Scale (1)
      SFP        -- Scale Fidelity Profile (inter-scale proportion JSD)

  Power-law (2)
      |Db|  -- absolute beta difference    truncated power-law exponent
      |Dk|  -- absolute kappa difference   exponential cutoff distance

Each JSD metric compares the empirical distributions of the corresponding
descriptor across real and generated trajectory collections.

Input
-----
Real and simulated trajectory lists.  Each trajectory is a list of points in
any of the following formats (auto-detected):

  * [time_frac, poi_id, [lat, lng]]
  * [t_start, t_end, poi_name, poi_id, [lon, lat]]  -- Tencent raw
  * dict  with keys t_i / l_i or lat / lng
  * DayTrajectory objects     (scaleTraj output -- stays are unpacked)
  * GroundedStay objects       (scaleTraj stays with start_min/end_min/coord)

Time values > 1.5 are treated as 144-slot integers and converted to fractions.

Output
------
Structured dict:
  {
    "micro":      {"SD": {"jsd": ...}, "SI": ..., "STAY": ...},
    "meso":       {"RG": {"jsd": ...}, "DTD": ..., "DUL": ...},
    "paper":      {"Visitation": ..., "IRank": ...},
    "macro":      {"power_law": {...}, "gmm": {...}, "scale_buckets": {...}},
    "jsd_scores": {flat dict of the 9 JSD values for quick access},
    "figures":    {name -> path},
  }
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.spatial import cKDTree
from sklearn.mixture import GaussianMixture

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_SMOOTH     = 1e-9     # Laplace additive smoothing per bin (CoPB convention)
_TIME_SLOTS = 144      # 10-min slots per day
_R0_KM      = 0.1     # Δr₀ offset in P(Δr) ~ (Δr + Δr₀)^{-β} exp(−Δr/κ)

# Fallback bin edges used only when the caller does not pass
# scale_boundaries_km. Prefer the per-dataset GMM boundaries passed in.
SCALE_BINS: list[tuple[str, float, float]] = [
    ("micro", 0.0,   0.5),
    ("neighborhood",  0.5,   3.0),
    ("urban",          3.0,  27.0),
    ("regional", 27.0, 88.0),
    ("macro",    88.0, float("inf")),
]

_SCALE_LABELS = ("neighbourhood", "city", "urban-agglomeration", "regional", "region")


def _bins_from_boundaries(boundaries: list[float]) -> list[tuple[str, float, float]]:
    full = [0.0] + list(boundaries) + [float("inf")]
    n_bins = len(full) - 1
    labels = _SCALE_LABELS[:n_bins - 1] + (_SCALE_LABELS[-1],)
    return [(labels[i], full[i], full[i + 1]) for i in range(len(labels))]

# ─────────────────────────────────────────────────────────────────────────────
# Internal normalised point
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Pt:
    t_start:  float   # fraction of 24 h  [0, 1)
    t_end:    float   # fraction of 24 h  (equals t_start when unknown)
    lat:      float
    lng:      float
    category: str     # activity category label (empty → unknown)
    poi_id:   str     # POI identifier          (empty → unknown)


TrajList = list[list[_Pt]]


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sf(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _valid_coord(lat: float, lng: float) -> bool:
    return -90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0


def _frac(t: float) -> float:
    """Convert slot index (0–143) or fraction to [0, 1)."""
    return t / _TIME_SLOTS if t > 1.5 else t


def _normalize_point(raw: Any) -> _Pt | None:
    """Parse one raw point into _Pt; returns None on failure."""
    if isinstance(raw, _Pt):
        return raw

    # GroundedStay (scaleTraj pipeline output)
    if hasattr(raw, "start_min") and hasattr(raw, "coord"):
        lng, lat = raw.coord
        if not _valid_coord(lat, lng):
            return None
        return _Pt(
            t_start=raw.start_min / 1440,
            t_end=raw.end_min / 1440,
            lat=lat, lng=lng,
            category=raw.intent_semantic.value if raw.intent_semantic else "",
            poi_id=raw.poi_id or "",
        )

    # Object with t_i / l_i attributes (e.g. baseline trajectory formats)
    if hasattr(raw, "t_i") and hasattr(raw, "l_i"):
        lng, lat = raw.l_i
        if not _valid_coord(lat, lng):
            return None
        return _Pt(
            t_start=_frac(_sf(raw.t_i)),
            t_end=_frac(_sf(getattr(raw, "t_end", raw.t_i))),
            lat=lat, lng=lng,
            category=str(getattr(raw, "o_i", "") or ""),
            poi_id=str(getattr(raw, "poi_id", "") or ""),
        )

    # dict with start_min / coord (daily_stays.jsonl or DayTrajectory serialised stays)
    if isinstance(raw, dict) and "start_min" in raw:
        coord = raw.get("coord") or raw.get("coord_snapped")
        if coord and len(coord) >= 2:
            lng, lat = _sf(coord[0]), _sf(coord[1])
            if _valid_coord(lat, lng):
                return _Pt(
                    t_start=_sf(raw["start_min"]) / 1440,
                    t_end=_sf(raw.get("end_min", raw["start_min"])) / 1440,
                    lat=lat, lng=lng,
                    category=str(raw.get("intent_semantic") or raw.get("intent_class") or ""),
                    poi_id=str(raw.get("poi_id") or ""),
                )

    # dict (including JSON-serialised TrajectoryPoint with l_i / t_i keys)
    if isinstance(raw, dict):
        if "l_i" in raw and "t_i" in raw:
            coord = raw["l_i"]
            if not isinstance(coord, (list, tuple)) or len(coord) < 2:
                return None
            lng = _sf(coord[0]); lat = _sf(coord[1])
            if not _valid_coord(lat, lng):
                return None
            t = _frac(_sf(raw["t_i"]))
            return _Pt(
                t_start=t,
                t_end=_frac(_sf(raw.get("t_end", raw["t_i"]))),
                lat=lat, lng=lng,
                category=str(raw.get("o_i", "") or ""),
                poi_id=str(raw.get("poi_id", "") or ""),
            )
        lat = _sf(raw.get("lat", raw.get("latitude", 0.0)))
        lng = _sf(raw.get("lng", raw.get("lon", raw.get("longitude", 0.0))))
        if not _valid_coord(lat, lng):
            return None
        t = _sf(raw.get("time", raw.get("t_start", raw.get("t_i", 0.0))))
        return _Pt(
            t_start=_frac(t),
            t_end=_frac(_sf(raw.get("t_end", t))),
            lat=lat, lng=lng,
            category=str(raw.get("category", raw.get("activity", raw.get("intent", ""))) or ""),
            poi_id=str(raw.get("poi_id", "") or ""),
        )

    # list / tuple. Tencent format: [t_start, t_end, poi_name, poi_id, [lon, lat]]
    if isinstance(raw, (list, tuple)):
        n = len(raw)
        if n >= 5 and isinstance(raw[4], (list, tuple)) and len(raw[4]) >= 2:
            lng = _sf(raw[4][0]); lat = _sf(raw[4][1])
            if not _valid_coord(lat, lng):
                return None
            return _Pt(
                t_start=_frac(_sf(raw[0])), t_end=_frac(_sf(raw[1])),
                lat=lat, lng=lng,
                category=str(raw[2]) if n > 2 else "",
                poi_id=str(raw[3]) if n > 3 else "",
            )
        # Standard / LLMob: [time, poi_or_cat, [lat, lng]]
        if n >= 3 and isinstance(raw[2], (list, tuple)) and len(raw[2]) >= 2:
            lat = _sf(raw[2][0]); lng = _sf(raw[2][1])
            if not _valid_coord(lat, lng):
                return None
            t = _frac(_sf(raw[0]))
            cat = str(raw[1]) if not isinstance(raw[1], (int, float)) else ""
            pid = str(raw[1]) if isinstance(raw[1], str) else ""
            return _Pt(t_start=t, t_end=t, lat=lat, lng=lng, category=cat, poi_id=pid)

    return None


def build_location_grid(train_jsonl_path: str) -> np.ndarray:
    """Extract unique (lat, lng) from training JSONL for grid snapping."""
    import json as _json
    coords = set()
    with open(train_jsonl_path) as f:
        for line in f:
            rec = _json.loads(line)
            for s in rec.get("stays", []):
                coord = s.get("coord")
                if coord and len(coord) >= 2:
                    lng, lat = float(coord[0]), float(coord[1])
                    if _valid_coord(lat, lng):
                        coords.add((lat, lng))
    return np.array(sorted(coords), dtype=np.float64)


def snap_to_locations(trajs: TrajList, grid: np.ndarray) -> TrajList:
    """Snap each point in trajs to the nearest location in grid (lat, lng)."""
    tree = cKDTree(grid)
    out: TrajList = []
    for traj in trajs:
        snapped = []
        for pt in traj:
            _, idx = tree.query([pt.lat, pt.lng])
            s_lat, s_lng = grid[idx]
            snapped.append(_Pt(
                t_start=pt.t_start, t_end=pt.t_end,
                lat=float(s_lat), lng=float(s_lng),
                category=pt.category, poi_id=pt.poi_id,
            ))
        out.append(snapped)
    return out


def normalize_trajectories(raw: Any) -> TrajList:
    """Convert any supported raw trajectory collection to TrajList."""
    out: TrajList = []
    for raw_traj in raw:
        if hasattr(raw_traj, "stays"):
            raw_traj = raw_traj.stays
        elif isinstance(raw_traj, dict) and "stays" in raw_traj:
            raw_traj = raw_traj["stays"]
        pts = [_normalize_point(p) for p in raw_traj]
        pts = [p for p in pts if p is not None]
        if pts:
            out.append(pts)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────

def _hav_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0088
    la1, lo1, la2, lo2 = map(math.radians, [lat1, lng1, lat2, lng2])
    dlat = la2 - la1; dlon = lo2 - lo1
    a = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
    return R * 2.0 * math.asin(min(1.0, math.sqrt(a)))


# ─────────────────────────────────────────────────────────────────────────────
# Distribution utilities
# ─────────────────────────────────────────────────────────────────────────────

def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen–Shannon Divergence with Laplace smoothing (CoPB convention).

    Adds ε to every bin before normalisation, ensuring no bin has exactly
    zero probability mass. This stabilises JSD on sparse histograms
    (e.g. macro scale with <500 observations across 100 bins).
    """
    p = p.astype(float) + _SMOOTH
    q = q.astype(float) + _SMOOTH
    p = p / p.sum()
    q = q / q.sum()
    m = (p + q) / 2.0
    js = 0.5 * scipy.stats.entropy(p, m) + 0.5 * scipy.stats.entropy(q, m)
    return float(np.clip(js, 0.0, math.log(2)))


# ═════════════════════════════════════════════════════════════════════════════
# MIRAGE-style JSD binning — used uniformly by every distributional metric
# below (SD, SI, STAY, RG). ``_MIRAGE_BINS`` adaptive bins over
# ``[0, max(real, sim)]`` per the KDD 2025 MIRAGE reference implementation
# (Deng et al., "Revisiting Synthetic Human Trajectories", § 5.1.3 and
# open-source ``statistical_metrics.py``). Motivation for switching from
# paper-specific fixed ranges / overflow schemes:
#   - LLMob / ELLMob pick arbitrary ranges (0-10 km, 0-100 km) that make
#     JSD numbers unreproducible without exact binning config.
#   - CoPB Scott's-rule bins depend on sample size so the same two datasets
#     give different JSD at different n.
#   - MIRAGE's single "100 bins, 0-max" rule is reproducible and uniform
#     across all metrics, and their paper explicitly motivates this choice.
# ═════════════════════════════════════════════════════════════════════════════

_MIRAGE_BINS = 100


def _jsd_mirage(real_arr: np.ndarray, sim_arr: np.ndarray,
                 bins: int = _MIRAGE_BINS) -> float:
    """MIRAGE 100-bin JSD over ``[0, max(real, sim)]`` (``bins`` equal-width)."""
    real_arr = real_arr[np.isfinite(real_arr) & (real_arr >= 0.0)]
    sim_arr  = sim_arr[np.isfinite(sim_arr)  & (sim_arr  >= 0.0)]
    if real_arr.size == 0 or sim_arr.size == 0:
        return float("nan")
    max_val = max(float(real_arr.max()), float(sim_arr.max()))
    if max_val <= 0.0:
        return 0.0
    edges = np.linspace(0.0, max_val, bins + 1)
    r_hist, _ = np.histogram(real_arr, bins=edges)
    s_hist, _ = np.histogram(sim_arr,  bins=edges)
    return _jsd(r_hist.astype(float), s_hist.astype(float))


def _jsd_logbin(real_arr: np.ndarray, sim_arr: np.ndarray,
                bins: int = _MIRAGE_BINS) -> float:
    """Log-spaced 100-bin JSD for heavy-tailed distributions (distances).

    Canonical for power-law displacement analysis (Milojević 2010,
    Yan et al. 2013, Zhao et al. 2015). Gives equal resolution per
    decade, avoiding the systematic under-resolution of tails that
    linear binning imposes.
    """
    real_arr = real_arr[np.isfinite(real_arr) & (real_arr > 0.0)]
    sim_arr  = sim_arr[np.isfinite(sim_arr)  & (sim_arr  > 0.0)]
    if real_arr.size == 0 or sim_arr.size == 0:
        return float("nan")
    all_pos = np.concatenate([real_arr, sim_arr])
    lo = float(all_pos.min())
    hi = float(all_pos.max())
    if hi <= lo:
        return 0.0
    edges = np.logspace(np.log10(lo * 0.99), np.log10(hi * 1.01), bins + 1)
    r_hist, _ = np.histogram(real_arr, bins=edges)
    s_hist, _ = np.histogram(sim_arr,  bins=edges)
    return _jsd(r_hist.astype(float), s_hist.astype(float))



# ═════════════════════════════════════════════════════════════════════════════
# MICRO-SCALE: Decision Step Fidelity
# ═════════════════════════════════════════════════════════════════════════════

def _step_distances_km(trajs: TrajList) -> np.ndarray:
    """Per-step haversine distance between consecutive events (km)."""
    out: list[float] = []
    for traj in trajs:
        for i in range(len(traj) - 1):
            out.append(_hav_km(traj[i].lat, traj[i].lng,
                                traj[i + 1].lat, traj[i + 1].lng))
    return np.asarray(out, dtype=float)


def compute_sd_jsd(real: TrajList, sim: TrajList) -> float:
    """Step Distance (SD) JSD — per-step Δd in km, log-spaced 100-bin."""
    return _jsd_logbin(_step_distances_km(real), _step_distances_km(sim))


def _step_intervals_sec(trajs: TrajList) -> np.ndarray:
    """Per-step time gap between consecutive events, in seconds."""
    out: list[float] = []
    for traj in trajs:
        for i in range(len(traj) - 1):
            dt_frac = traj[i + 1].t_start - traj[i].t_start
            if dt_frac < 0.0:
                dt_frac += 1.0
            out.append(dt_frac * 86400.0)
    return np.asarray(out, dtype=float)


def compute_si_jsd(real: TrajList, sim: TrajList) -> float:
    """Step Interval (SI) JSD — per-step Δt in seconds, MIRAGE 100-bin adaptive."""
    return _jsd_mirage(_step_intervals_sec(real), _step_intervals_sec(sim))


def _stay_durations_min(trajs: TrajList) -> np.ndarray:
    """Per-event stay duration (minutes). Skips sub-minute artefacts."""
    out: list[float] = []
    for traj in trajs:
        for pt in traj:
            dur_min = (pt.t_end - pt.t_start) * 24.0 * 60.0
            if dur_min > 1.0:
                out.append(dur_min)
    return np.asarray(out, dtype=float)


def compute_stay_jsd(real: TrajList, sim: TrajList) -> float:
    """Stay Duration (STAY) JSD — per-event dwell in minutes, MIRAGE 100-bin."""
    return _jsd_mirage(_stay_durations_min(real), _stay_durations_min(sim))



# ═════════════════════════════════════════════════════════════════════════════
# MESO-SCALE: Individual Patterns
# ═════════════════════════════════════════════════════════════════════════════

def _radius_of_gyration_km(traj: list[_Pt]) -> float:
    """Per-trajectory radius of gyration R_g (km).

    Follows MIRAGE ``statistical_metrics.py:radius`` for direct numerical
    comparability with KDD'25 Table 2:

        R_g = √( mean_i  haversine(r_i , r_cm) )

    where the centroid r_cm is the arithmetic mean of (lat, lng). This is
    slightly different from the classical Σ|r-r_cm|² variant but matches the
    MIRAGE open-source protocol head-to-head.
    """
    if not traj:
        return 0.0
    lats = np.fromiter((p.lat for p in traj), dtype=float, count=len(traj))
    lngs = np.fromiter((p.lng for p in traj), dtype=float, count=len(traj))
    c_lat = float(lats.mean()); c_lng = float(lngs.mean())
    dists = np.array([_hav_km(la, lo, c_lat, c_lng)
                      for la, lo in zip(lats, lngs, strict=False)], dtype=float)
    return float(math.sqrt(float(dists.mean())))


def compute_rg_jsd(real: TrajList, sim: TrajList) -> float:
    """Radius of Gyration (RG) JSD — per-trajectory R_g, log-spaced 100-bin."""
    r = np.array([_radius_of_gyration_km(t) for t in real if len(t) >= 1],
                 dtype=float)
    s = np.array([_radius_of_gyration_km(t) for t in sim  if len(t) >= 1],
                 dtype=float)
    return _jsd_logbin(r, s)


def _daily_travel_distance_km(traj: list[_Pt]) -> float:
    """Total haversine distance traveled in one trajectory (km)."""
    total = 0.0
    for i in range(len(traj) - 1):
        total += _hav_km(traj[i].lat, traj[i].lng,
                         traj[i + 1].lat, traj[i + 1].lng)
    return total


def _daily_unique_locations(traj: list[_Pt], decimals: int = 3) -> int:
    """Number of distinct (lat, lng) locations visited in one trajectory."""
    return len({(round(pt.lat, decimals), round(pt.lng, decimals)) for pt in traj})


def compute_dtd(real: TrajList, sim: TrajList) -> dict[str, Any]:
    """Daily Travel Distance — JSD + mean/std of per-trajectory total km."""
    r = np.array([_daily_travel_distance_km(t) for t in real if len(t) >= 2],
                 dtype=float)
    s = np.array([_daily_travel_distance_km(t) for t in sim if len(t) >= 2],
                 dtype=float)
    return {
        "jsd": _jsd_logbin(r, s),
        "real_mean": float(r.mean()) if r.size else float("nan"),
        "real_std": float(r.std()) if r.size else float("nan"),
        "sim_mean": float(s.mean()) if s.size else float("nan"),
        "sim_std": float(s.std()) if s.size else float("nan"),
    }


def compute_dul(real: TrajList, sim: TrajList) -> dict[str, Any]:
    """Daily Unique Locations — JSD + mean/std of per-trajectory count."""
    r = np.array([_daily_unique_locations(t) for t in real if t],
                 dtype=float)
    s = np.array([_daily_unique_locations(t) for t in sim if t],
                 dtype=float)
    return {
        "jsd": _jsd_mirage(r, s),
        "real_mean": float(r.mean()) if r.size else float("nan"),
        "real_std": float(r.std()) if r.size else float("nan"),
        "sim_mean": float(s.mean()) if s.size else float("nan"),
        "sim_std": float(s.std()) if s.size else float("nan"),
    }


# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
# PAPER-ALIGNED: Yan et al. 2026 (M2LSimu) Table 1 metrics
#
# 500 m × 500 m grid discretisation is applied so that "location identity"
# is consistent across Tencent POI_id data and TrajSynVAE grid-cell data.
# Grid at the equator: 500 m ≈ 0.0045° lat and 0.0045° / cos(lat) lng.
# ═════════════════════════════════════════════════════════════════════════════

_GRID_METERS = 500.0
# 500 m in degrees latitude (constant). Lng conversion needs cos(lat).
_TIME_BINS_PER_DAY = 48


def _cell_id(lat: float, lng: float, grid_m: float = _GRID_METERS) -> tuple[int, int]:
    """Discretise (lat, lng) to a 500 m × 500 m grid-cell index.

    Grid alignment is global (no per-city offset). Adjacent cells have
    consecutive integer indices along each axis.
    """
    lat_deg = grid_m / 111_320.0
    # cos(lat) ≈ 1 near equator; use lat-dependent to preserve isotropy.
    lng_deg = grid_m / (111_320.0 * max(0.01, math.cos(math.radians(lat))))
    return int(math.floor(lat / lat_deg)), int(math.floor(lng / lng_deg))


def _cell_centroid(lat: float, lng: float, grid_m: float = _GRID_METERS) -> tuple[float, float]:
    """Return the (lat, lng) centre of the 500 m cell containing (lat, lng).

    Used by the coarse-graining step: every raw point is replaced with its
    cell centroid so all downstream distance computations run in grid space
    (Yan 2026 § 2.3).
    """
    lat_deg = grid_m / 111_320.0
    lng_deg = grid_m / (111_320.0 * max(0.01, math.cos(math.radians(lat))))
    row = math.floor(lat / lat_deg)
    col = math.floor(lng / lng_deg)
    return (row + 0.5) * lat_deg, (col + 0.5) * lng_deg


def _traj_cells(traj: list[_Pt], grid_m: float = _GRID_METERS) -> list[tuple[int, int]]:
    """Map a trajectory to its sequence of grid-cell IDs in time order."""
    return [_cell_id(pt.lat, pt.lng, grid_m) for pt in traj]


def _coarse_grain_point(pt: _Pt, grid_m: float = _GRID_METERS,
                         bins_per_day: int = _TIME_BINS_PER_DAY) -> _Pt:
    """Snap a point to (cell-centroid, time-bin) per Yan 2026 § 2.3.

    Spatial: (lat, lng) → centre of its 500 m grid cell.
    Temporal: t_start / t_end → lower boundary of its 30 min time bin.
    """
    c_lat, c_lng = _cell_centroid(pt.lat, pt.lng, grid_m)
    def _snap(t: float) -> float:
        if not math.isfinite(t):
            return t
        t = max(0.0, min(0.9999, float(t)))
        return math.floor(t * bins_per_day) / bins_per_day
    return _Pt(
        t_start=_snap(pt.t_start), t_end=_snap(pt.t_end),
        lat=c_lat, lng=c_lng,
        category=pt.category, poi_id=pt.poi_id,
    )


def coarse_grain(trajs: TrajList, grid_m: float = _GRID_METERS,
                  bins_per_day: int = _TIME_BINS_PER_DAY) -> TrajList:
    """Coarse-grain every point in every trajectory (Yan 2026 S 2.3).

    Applied once at the top of ``MobilityEvaluator.evaluate`` so every
    downstream metric sees the same 500 m x 30 min representation.
    """
    return [[_coarse_grain_point(pt, grid_m, bins_per_day) for pt in traj]
            for traj in trajs]


# ─────────────────────────────────────────────────────────────────────────────
# Visitation frequency — per-user ζ from f_k ∝ k^(-ζ)
# ─────────────────────────────────────────────────────────────────────────────

def _fit_user_zeta(cells: list[tuple[int, int]]) -> float:
    """Fit per-user Zipf exponent ζ on rank-frequency relation f_k ∝ k^{-ζ}.

    Returns NaN if the user has fewer than 3 distinct cells.
    """
    if len(cells) < 3:
        return float("nan")
    counts: dict[tuple, int] = {}
    for c in cells:
        counts[c] = counts.get(c, 0) + 1
    freqs = np.asarray(sorted(counts.values(), reverse=True), dtype=float)
    if freqs.size < 3:
        return float("nan")
    ranks = np.arange(1, freqs.size + 1, dtype=float)
    # log-log regression: log f = -ζ log k + c
    log_k = np.log(ranks)
    log_f = np.log(freqs)
    slope, _ = np.polyfit(log_k, log_f, 1)
    return float(-slope)


def _per_user_zetas(trajs: TrajList, grid_m: float = _GRID_METERS) -> np.ndarray:
    """Compute per-user (per-trajectory) Zipf ζ values."""
    out = []
    for traj in trajs:
        if len(traj) < 3:
            continue
        z = _fit_user_zeta(_traj_cells(traj, grid_m))
        if np.isfinite(z):
            out.append(z)
    return np.asarray(out, dtype=float)


def compute_visitation_zeta_jsd(
    real: TrajList, sim: TrajList, grid_m: float = _GRID_METERS
) -> dict[str, float]:
    """Visitation (cell-visit frequency) — per-user Zipf ζ distribution JSD.

    Prior work [28] reports *"ζ ≈ 1.2 ± 0.1"* for the per-user rank-frequency
    fit ``f_k ∝ k^{-ζ}``. We fit ζ per user (on 500 m grid cells), then
    compare sim vs real distributions across users.

    Returns
    -------
    dict with keys:
      - ``jsd``: JSD on the per-user ζ distribution (hard-binned 0..3).
      - ``mean_real`` / ``mean_sim``: mean ζ across users — the headline
        number comparable against paper's ζ ≈ 1.2 ± 0.1.
      - ``std_real`` / ``std_sim``: per-user ζ std (width of the distribution).
      - ``n_real`` / ``n_sim``: user count contributing.
    """
    nan_result = {"jsd": float("nan"), "mean_real": float("nan"),
                  "mean_sim": float("nan"), "std_real": float("nan"),
                  "std_sim": float("nan"), "n_real": 0, "n_sim": 0}
    r = _per_user_zetas(real, grid_m)
    s = _per_user_zetas(sim, grid_m)
    if r.size == 0 or s.size == 0:
        return nan_result
    return {
        "jsd": _jsd_mirage(r, s),
        "mean_real": float(np.mean(r)),
        "mean_sim": float(np.mean(s)),
        "std_real": float(np.std(r)),
        "std_sim": float(np.std(s)),
        "n_real": int(r.size),
        "n_sim": int(s.size),
    }


# ─────────────────────────────────────────────────────────────────────────────
# I-Rank — per-user top-k visitation proportion
# ─────────────────────────────────────────────────────────────────────────────

def _per_user_top_k_proportions(
    trajs: TrajList, k: int = 3, grid_m: float = _GRID_METERS
) -> np.ndarray:
    """Per-user fraction of visits to top-k most-visited locations.

    For each user trajectory, count visits per grid cell, sort descending,
    and return sum(top-k counts) / total_visits.  Users with < k distinct
    cells get proportion = 1.0 (all visits are in top-k by definition).
    """
    out: list[float] = []
    for traj in trajs:
        cells = _traj_cells(traj, grid_m)
        if len(cells) < 2:
            continue
        counts: dict[tuple, int] = {}
        for c in cells:
            counts[c] = counts.get(c, 0) + 1
        sorted_counts = sorted(counts.values(), reverse=True)
        total = sum(sorted_counts)
        top_k_sum = sum(sorted_counts[:k])
        out.append(top_k_sum / total)
    return np.asarray(out, dtype=float)


def compute_irank_jsd(
    real: TrajList, sim: TrajList, grid_m: float = _GRID_METERS
) -> float:
    """IRank (individual-rank correlation) — top-3 visitation proportion JSD.

    Measures how well the model reproduces individual location concentration
    by comparing the distribution of "share of visits going to a
    trajectory's top-3 cells" between real and synthetic populations.
    Metric family from MoveSim (Feng et al. 2020) and follow-up works.
    """
    r = _per_user_top_k_proportions(real, k=3, grid_m=grid_m)
    s = _per_user_top_k_proportions(sim, k=3, grid_m=grid_m)
    if r.size == 0 or s.size == 0:
        return float("nan")
    return _jsd_mirage(r, s)


# ═════════════════════════════════════════════════════════════════════════════
# MACRO-SCALE: Emergent Collective Scaling Laws
# ═════════════════════════════════════════════════════════════════════════════

def _extract_distances_km(trajs: TrajList, max_km: float = math.inf) -> np.ndarray:
    """Extract all intra-trajectory step distances (km). ``max_km`` caps
    outliers; default ``inf`` keeps the entire tail (required to fit the
    truncated power-law's κ cutoff, which reflects the natural geographic
    span of the dataset — Beijing ~50 km, GeoLife ~8000 km)."""
    out = []
    for traj in trajs:
        for i in range(len(traj) - 1):
            d = _hav_km(traj[i].lat, traj[i].lng, traj[i+1].lat, traj[i+1].lng)
            if 0.0 < d <= max_km:
                out.append(d)
    return np.asarray(out, dtype=float)


def empirical_ccdf(distances_km: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted_x, P(X ≥ x)) for positive finite distances."""
    arr = distances_km[np.isfinite(distances_km) & (distances_km > 0.0)]
    if arr.size == 0:
        return np.array([]), np.array([])
    x = np.sort(arr)
    y = (x.size - np.arange(x.size, dtype=float)) / float(x.size)
    return x, y


def fit_power_law_paper_aligned(
    distances_km: np.ndarray,
    x_min: float = 2.0,
    r0_km: float = 1.5,
    kappa_upper_km: float = 20000.0,
) -> dict[str, Any]:
    """Publication-aligned truncated power-law fit: Clauset β + MLE κ.

    Goal: reproduce the Gonzalez 2008 form ``P(Δd) ∝ (Δd + r0)^{-β} · exp(-Δd/κ)``
    with values that compare directly to published numbers:

      - Gonzalez 2008 D1 (EU continent):  β=1.75 ± 0.15, r0=1.5 km, κ≈400 km
      - Gonzalez 2008 D2 (subset):        β=1.75 ± 0.15, r0=1.5 km, κ≈80 km
      - Yan 2026 Tencent Beijing subset:  β≈1.624, κ≈36.1 km

    **Why Clauset β + fixed-β MLE κ** (not joint PDF-MLE)?
    The joint MLE on short-tail datasets (Beijing max ~60 km) is ill-
    conditioned: β and κ trade off against each other across the data's
    mode (0.5-3 km), and the optimiser frequently collapses to β ≈ 0 (pure
    exponential) even when the tail is clearly power-law. Clauset's tail-
    only MLE at x_min=2 km is a more robust β estimate (matches paper
    convention). Holding β fixed at the Clauset value and fitting κ
    alone is a well-posed 1D MLE that gives a meaningful κ.

    Parameters
    ----------
    x_min : Tail threshold for Clauset β. Default 2.0 km (Gonzalez/LLMob/CoPB).
    r0_km : Gonzalez's Δr₀ = 1.5 km (paper-exact).
    kappa_upper_km : Upper bound on κ during optimisation.

    Returns
    -------
    dict with keys ``beta``, ``kappa``, ``x_min``, ``r0_km``, ``n``,
    ``beta_source``, ``kappa_method``.
    """
    from scipy.integrate import quad
    from scipy.optimize import minimize_scalar

    nan_result = dict(
        beta=float("nan"), kappa=float("nan"), x_min=x_min, r0_km=r0_km, n=0,
        beta_source="clauset_tail_mle_xmin_2km",
        kappa_method="1d_mle_beta_fixed",
    )

    d = np.asarray(distances_km, dtype=float)
    d = d[np.isfinite(d) & (d > 0.0)]
    if d.size < 100:
        return nan_result

    # Step 1: β from Clauset tail MLE @ x_min=2 km (paper convention).
    clauset = fit_power_law_clauset(d)
    beta = clauset.get("alpha_xmin_2km", float("nan"))
    if not np.isfinite(beta) or beta <= 0.0:
        return nan_result

    # Step 2: fit κ with β fixed, truncated-PL MLE over [x_min, ∞).
    d_fit = d[d >= x_min]
    n = int(d_fit.size)
    if n < 100:
        return nan_result
    log_shift_sum = float(np.sum(np.log(d_fit + r0_km)))
    d_sum = float(np.sum(d_fit))

    def _neg_log_likelihood(log_kappa: float) -> float:
        kappa = float(np.exp(log_kappa))
        if kappa <= 0.0 or kappa > kappa_upper_km:
            return np.inf
        try:
            Z, _ = quad(
                lambda x: (x + r0_km) ** (-beta) * np.exp(-x / kappa),
                x_min, np.inf, limit=300,
            )
        except Exception:
            return np.inf
        if Z <= 0.0 or not np.isfinite(Z):
            return np.inf
        return n * np.log(Z) + beta * log_shift_sum + d_sum / kappa

    try:
        res = minimize_scalar(
            _neg_log_likelihood,
            bounds=(np.log(0.5), np.log(kappa_upper_km)),
            method="bounded",
            options={"xatol": 1e-4},
        )
    except Exception:
        return nan_result
    kappa = float(np.exp(res.x))

    return dict(
        beta=float(beta),
        kappa=kappa,
        x_min=float(x_min),
        r0_km=float(r0_km),
        n=n,
        beta_source="clauset_tail_mle_xmin_2km",
        kappa_method="1d_mle_beta_fixed",
    )


def fit_power_law_clauset(
    distances_km: np.ndarray,
    x_min_candidates: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0),
    min_tail_size: int = 50,
) -> dict[str, Any]:
    """Clauset-Shalizi-Newman 2009 MLE for a pure power-law tail.

    Fits ``P(Δd) ∝ Δd^(−α)`` for ``Δd ≥ x_min`` using the closed-form MLE::

        α̂ = 1 + n / Σ_i log(x_i / x_min)

    and selects ``x_min`` by minimising the Kolmogorov-Smirnov statistic
    between the empirical tail CDF and the fitted power-law CDF
    (Clauset et al. 2009, Eq. 3.3 + §3.3).

    Returns
    -------
    dict with keys ``alpha`` (= published β), ``x_min`` (auto-selected),
    ``ks_stat``, ``n_tail``. NaNs → insufficient data.
    """

    nan = dict(alpha=float("nan"), x_min=float("nan"),
               ks_stat=float("nan"), n_tail=0,
               alpha_xmin_2km=float("nan"), n_tail_xmin_2km=0)
    d = np.asarray(distances_km, dtype=float)
    d = d[np.isfinite(d) & (d > 0.0)]
    if d.size < min_tail_size * 2:
        return nan

    def _alpha_and_ks(xm: float) -> tuple[float, float, int] | None:
        tail = d[d >= xm]
        n = int(tail.size)
        if n < min_tail_size:
            return None
        log_ratio_sum = float(np.sum(np.log(tail / xm)))
        if log_ratio_sum <= 0:
            return None
        alpha = 1.0 + n / log_ratio_sum
        # Empirical tail CDF
        sorted_tail = np.sort(tail)
        emp_cdf = np.arange(1, n + 1) / n
        # Theoretical CDF for pure power law with lower bound xm:
        #   F(x) = 1 - (x / xm)^(1 - alpha)
        th_cdf = 1.0 - (sorted_tail / xm) ** (1.0 - alpha)
        ks = float(np.max(np.abs(emp_cdf - th_cdf)))
        return alpha, ks, n

    best: tuple[float, float, int, float] | None = None  # (alpha, ks, n, xm)
    for xm in x_min_candidates:
        result = _alpha_and_ks(xm)
        if result is None:
            continue
        alpha, ks, n = result
        if best is None or ks < best[1]:
            best = (alpha, ks, n, xm)
    if best is None:
        return nan
    alpha_hat, ks_hat, n_tail, xm_hat = best

    # Paper convention (Gonzalez 2008 / LLMob / CoPB): fix x_min at 2 km
    # and report α there. This is the value that matches "β ≈ 1.75 ± 0.15"
    # in the literature regardless of KS-optimal selection.
    paper_result = _alpha_and_ks(2.0)
    alpha_paper = float(paper_result[0]) if paper_result else float("nan")
    n_tail_paper = int(paper_result[2]) if paper_result else 0

    return dict(
        alpha=float(alpha_hat),
        x_min=float(xm_hat),
        ks_stat=float(ks_hat),
        n_tail=int(n_tail),
        alpha_xmin_2km=alpha_paper,
        n_tail_xmin_2km=n_tail_paper,
    )


def fit_gmm_log_distance(
    distances_km: np.ndarray,
    n_components_range: list[int] | None = None,
    criterion: str = "bic",
    random_state: int = 42,
    n_init: int = 5,
) -> dict[str, Any]:
    """Fit a GMM in log-distance space; each component is a lognormal scale."""
    if n_components_range is None:
        n_components_range = [2, 3, 4, 5]

    arr = distances_km[np.isfinite(distances_km) & (distances_km > 0.0)]
    if arr.size < 50:
        return {"error": "insufficient_data", "n_components": 0, "components": []}

    log_arr = np.log(arr).reshape(-1, 1)
    best_model = None
    best_score = float("inf")
    scores = []

    for k in sorted(set(n_components_range)):
        try:
            mdl = GaussianMixture(
                n_components=k, covariance_type="full",
                random_state=random_state, n_init=n_init,
            )
            mdl.fit(log_arr)
            bic = float(mdl.bic(log_arr))
            aic = float(mdl.aic(log_arr))
            score = bic if criterion == "bic" else aic
            scores.append({"n": k, "bic": bic, "aic": aic})
            if score < best_score:
                best_score = score; best_model = mdl
        except Exception:
            pass

    if best_model is None:
        return {"error": "fit_failed", "n_components": 0, "components": []}

    means   = best_model.means_.reshape(-1)
    vars_   = best_model.covariances_.reshape(-1)
    weights = best_model.weights_.reshape(-1)

    components = []
    for i in range(best_model.n_components):
        mu    = float(means[i])
        sigma = float(np.sqrt(max(float(vars_[i]), 1e-12)))
        components.append(dict(
            weight=float(weights[i]),
            mu_log=mu, sigma_log=sigma,
            mean_scale_km=float(np.exp(mu)),
        ))
    components.sort(key=lambda c: c["mean_scale_km"])
    for rank, c in enumerate(components, 1):
        c["rank"] = rank

    return dict(
        n_components=int(best_model.n_components),
        criterion=criterion,
        best_score=float(best_score),
        candidate_scores=scores,
        components=components,
    )


def scale_bucket_stats(
    distances_km: np.ndarray,
    scale_bins: list[tuple[str, float, float]] | None = None,
) -> dict[str, dict[str, float]]:
    """Fraction of steps in each scale bucket."""
    bins = scale_bins or SCALE_BINS
    arr = distances_km[np.isfinite(distances_km) & (distances_km > 0.0)]
    total = arr.size
    result: dict[str, dict[str, float]] = {}
    for name, lo, hi in bins:
        mask = (arr >= lo) & (arr < hi)
        cnt  = int(mask.sum())
        result[name] = dict(
            count=cnt,
            fraction=cnt / total if total else 0.0,
            mean_km=float(arr[mask].mean()) if cnt else float("nan"),
        )
    return result



def compute_scale_jsd(
    real_dists: np.ndarray,
    sim_dists: np.ndarray,
    scale_bins: list[tuple[str, float, float]] | None = None,
) -> dict[str, Any]:
    """Scale Fidelity Profile (SFP) and its per-scale decomposition.

    SFP = JSD(π_real || π_sim)

    where π is the K-bin proportion vector over Alessandretti scale bands.
    This is the metric reported in the paper (Table 3) and exported as
    `jsd_scores["SFP"]`.

    Also returns a chain-rule composite as a diagnostic side product:

        sfp_composite = JSD(π_real || π_sim)
                      + Σ_k w_k · JSD(P_k^real || P_k^sim)

    with intra-scale shape terms P_k weighted by w_k = (π_k^real+π_k^sim)/2.
    The composite is not reported in the paper — inspect it if you want to
    know whether an SFP miss is driven by proportion drift (dominant term)
    versus intra-scale shape drift (residual).

    Returns dict with:
      - scale_jsd     : JSD of the K-bin proportion vector — the paper's SFP
      - per_scale     : {bin_name: {jsd, n_real, n_sim, w_k}}
      - sfp_composite : proportion JSD + Σ_k w_k · intra-scale JSD (diagnostic)
    """
    bins = scale_bins or SCALE_BINS
    K = len(bins)
    r_arr = real_dists[np.isfinite(real_dists) & (real_dists > 0.0)]
    s_arr = sim_dists[np.isfinite(sim_dists) & (sim_dists > 0.0)]

    r_total = max(float(r_arr.size), 1.0)
    s_total = max(float(s_arr.size), 1.0)

    r_counts = np.zeros(K, dtype=float)
    s_counts = np.zeros(K, dtype=float)

    per_scale: dict[str, Any] = {}
    for i, (name, lo, hi) in enumerate(bins):
        r_in = r_arr[(r_arr >= lo) & (r_arr < hi)]
        s_in = s_arr[(s_arr >= lo) & (s_arr < hi)]
        r_counts[i] = float(r_in.size)
        s_counts[i] = float(s_in.size)
        r_frac = r_in.size / r_total
        s_frac = s_in.size / s_total
        w_k = (r_frac + s_frac) / 2.0
        if r_in.size >= 5 and s_in.size >= 5:
            per_scale[name] = dict(jsd=_jsd_logbin(r_in, s_in),
                                   n_real=int(r_in.size), n_sim=int(s_in.size),
                                   w_k=w_k)
        else:
            per_scale[name] = dict(jsd=float("nan"),
                                   n_real=int(r_in.size), n_sim=int(s_in.size),
                                   w_k=w_k)

    proportion_jsd = float(_jsd(r_counts, s_counts))

    intra_sum = 0.0
    for info in per_scale.values():
        if math.isfinite(info["jsd"]):
            intra_sum += info["w_k"] * info["jsd"]

    return dict(
        scale_jsd=proportion_jsd,               # = paper's SFP (exported as jsd_scores["SFP"])
        per_scale=per_scale,
        sfp_composite=proportion_jsd + intra_sum,   # diagnostic, not in paper
    )


def _analyse_distances(
    distances_km: np.ndarray,
    scale_bins: list[tuple[str, float, float]] | None = None,
) -> dict[str, Any]:
    pl_paper = fit_power_law_paper_aligned(distances_km)
    gmm      = fit_gmm_log_distance(distances_km)
    buckets  = scale_bucket_stats(distances_km, scale_bins)
    return dict(
        n_steps=int(distances_km.size),
        power_law=pl_paper,          # Clauset β @ x_min=2km + MLE κ (González-comparable)
        gmm=gmm,
        scale_buckets=buckets,
    )


def _compute_delta(real_info: dict, sim_info: dict) -> dict[str, Any]:
    """Compute Δ = sim − real for key scaling parameters."""
    rpl = real_info["power_law"]; spl = sim_info["power_law"]

    def _d(key: str) -> float:
        rv, sv = rpl.get(key, float("nan")), spl.get(key, float("nan"))
        if math.isnan(rv) or math.isnan(sv):
            return float("nan")
        return sv - rv

    bucket_delta = {}
    all_names = set(real_info["scale_buckets"].keys()) | set(sim_info["scale_buckets"].keys())
    for name in sorted(all_names):
        rf = real_info["scale_buckets"].get(name, {}).get("fraction", float("nan"))
        sf = sim_info["scale_buckets"].get(name, {}).get("fraction", float("nan"))
        bucket_delta[name] = sf - rf if not (math.isnan(rf) or math.isnan(sf)) else float("nan")

    real_comps = real_info["gmm"].get("components", [])
    sim_comps  = sim_info["gmm"].get("components", [])
    gmm_delta  = []
    for i in range(min(len(real_comps), len(sim_comps))):
        rc, sc = real_comps[i], sim_comps[i]
        gmm_delta.append(dict(
            rank=i + 1,
            delta_weight=sc["weight"] - rc["weight"],
            delta_mu_log=sc["mu_log"] - rc["mu_log"],
            delta_mean_scale_km=sc["mean_scale_km"] - rc["mean_scale_km"],
        ))

    return dict(
        delta_beta=_d("beta"),
        delta_kappa=_d("kappa"),
        bucket_fraction_delta=bucket_delta,
        gmm_component_delta=gmm_delta,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

_BLUE   = "#1f77b4"
_ORANGE = "#ff7f0e"
_BAR_W  = 0.35


def _save(fig: plt.Figure, path: Path) -> str:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _dual_bar(
    ax: plt.Axes, r_hist: np.ndarray, s_hist: np.ndarray,
    labels: list[str], title: str, ylabel: str = "Relative Frequency",
) -> None:
    r_norm = r_hist / (r_hist.sum() + _SMOOTH)
    s_norm = s_hist / (s_hist.sum() + _SMOOTH)
    x = np.arange(len(r_norm))
    ax.bar(x - _BAR_W / 2, r_norm, _BAR_W, label="Real", color=_BLUE,   alpha=0.85)
    ax.bar(x + _BAR_W / 2, s_norm, _BAR_W, label="Sim",  color=_ORANGE, alpha=0.85)
    step = max(1, len(labels) // 8)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(labels[::step], rotation=35, ha="right", fontsize=7)
    ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)


def _mirage_hist_labels(
    real_arr: np.ndarray, sim_arr: np.ndarray,
    bins: int = _MIRAGE_BINS, label_every: int = 10,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Bin ``real`` and ``sim`` with MIRAGE 100-bin adaptive + tick labels.

    Returns ``(r_hist, s_hist, labels)`` suitable for ``_dual_bar``. Labels
    show the left edge of every ``label_every``-th bin so plots stay legible.
    """
    r = real_arr[np.isfinite(real_arr) & (real_arr >= 0.0)]
    s = sim_arr[np.isfinite(sim_arr) & (sim_arr >= 0.0)]
    if r.size == 0 or s.size == 0:
        return np.zeros(bins), np.zeros(bins), [""] * bins
    max_val = max(float(r.max()), float(s.max()))
    if max_val <= 0.0:
        return np.zeros(bins), np.zeros(bins), [""] * bins
    edges = np.linspace(0.0, max_val, bins + 1)
    r_hist, _ = np.histogram(r, bins=edges)
    s_hist, _ = np.histogram(s, bins=edges)
    labels = [f"{edges[i]:.1f}" if i % label_every == 0 else ""
              for i in range(bins)]
    return r_hist.astype(float), s_hist.astype(float), labels


def plot_micro_distributions(
    real: TrajList, sim: TrajList, output_dir: Path
) -> dict[str, str]:
    """Dual-bar charts for SD, SI, Stay Duration — MIRAGE 100-bin adaptive."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    r_sd, s_sd = _step_distances_km(real), _step_distances_km(sim)
    if r_sd.size > 0 and s_sd.size > 0:
        r_h, s_h, labels = _mirage_hist_labels(r_sd, s_sd)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        _dual_bar(ax, r_h, s_h, labels, "Step Distance (SD) — P(Δd)")
        ax.set_xlabel("Distance (km)", fontsize=8)
        paths["sd"] = _save(fig, output_dir / "sd_comparison.png")

    r_si, s_si = _step_intervals_sec(real), _step_intervals_sec(sim)
    if r_si.size > 0 and s_si.size > 0:
        r_h, s_h, labels = _mirage_hist_labels(r_si, s_si)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        _dual_bar(ax, r_h, s_h, labels, "Step Interval (SI) — P(Δt)")
        ax.set_xlabel("Δt (seconds)", fontsize=8)
        paths["si"] = _save(fig, output_dir / "si_comparison.png")

    r_stay, s_stay = _stay_durations_min(real), _stay_durations_min(sim)
    if r_stay.size > 0 and s_stay.size > 0:
        r_h, s_h, labels = _mirage_hist_labels(r_stay, s_stay)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        _dual_bar(ax, r_h, s_h, labels, "Stay Duration (STAY) — P(stay)")
        ax.set_xlabel("Stay duration (min)", fontsize=8)
        paths["stay"] = _save(fig, output_dir / "stay_comparison.png")

    return paths


def plot_meso_distributions(
    real: TrajList, sim: TrajList, output_dir: Path
) -> dict[str, str]:
    """Dual-bar chart for RG -- MIRAGE 100-bin adaptive."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    r_rg = np.array([_radius_of_gyration_km(t) for t in real if len(t) >= 1],
                    dtype=float)
    s_rg = np.array([_radius_of_gyration_km(t) for t in sim  if len(t) >= 1],
                    dtype=float)
    if r_rg.size > 0 and s_rg.size > 0:
        r_h, s_h, labels = _mirage_hist_labels(r_rg, s_rg)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        _dual_bar(ax, r_h, s_h, labels, "Radius of Gyration (RG)")
        ax.set_xlabel("R_g (km)", fontsize=8)
        paths["rg"] = _save(fig, output_dir / "rg_comparison.png")

    return paths


def plot_ccdf_fit(
    real_distances: np.ndarray,
    sim_distances:  np.ndarray,
    real_pl: dict[str, float],
    sim_pl:  dict[str, float],
    real_gmm: dict[str, Any],
    sim_gmm:  dict[str, Any],
    output_dir: Path,
) -> str:
    """Log-log CCDF with fitted truncated power-law P(r>x) ~ (x+r₀)^{-β} exp(−x/κ)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))

    x_r, y_r = empirical_ccdf(real_distances)
    x_s, y_s = empirical_ccdf(sim_distances)

    if x_r.size:
        ax.plot(x_r, y_r, color=_BLUE,   lw=1.8, alpha=0.7, label="Real (empirical)")
    if x_s.size:
        ax.plot(x_s, y_s, color=_ORANGE, lw=1.8, alpha=0.7, label="Sim (empirical)")

    np.concatenate([x_r, x_s]) if x_r.size and x_s.size else (x_r if x_r.size else x_s)
    for pl, _color, tag in [(real_pl, _BLUE, "Real"), (sim_pl, _ORANGE, "Sim")]:
        b = pl.get("beta", float("nan"))
        k = pl.get("kappa", float("nan"))
        if not math.isnan(b):
            k_str = f", κ={k:.0f}km" if not math.isnan(k) else ""
            ax.plot([], [], " ", label=f"{tag}: β={b:.3f}{k_str} (Clauset@2km)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Displacement Δr (km)", fontsize=10)
    ax.set_ylabel("CCDF:  P(Δr ≥ x)", fontsize=10)
    ax.set_title("Travel Distance CCDF — Truncated Power-Law Fit  P(Δr) ~ (Δr+Δr₀)^{-β} e^{-Δr/κ}", fontsize=10)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return _save(fig, output_dir / "ccdf_fit.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluator
# ─────────────────────────────────────────────────────────────────────────────

class MobilityEvaluator:
    """Hierarchical mobility evaluator: Micro / Meso / Macro.

    Computes the 11 paper metrics: SD, RG, SI, STAY, DTD, DUL,
    Visitation, IRank, SFP, |Db|, |Dk|.

    Parameters
    ----------
    output_dir :      Directory for figures and artefacts.
    max_dist_km :     Outlier cap for macro-scale distance extraction.
    gmm_components :  Candidate GMM component counts (BIC selection).
    gmm_criterion :   "bic" or "aic" for GMM model selection.
    r0_km :           Dr0 offset in the power-law model (default 0.1 km).
    """

    def __init__(
        self,
        output_dir:     str = "eval_output",
        max_dist_km:    float = math.inf,
        gmm_components: list[int] | None = None,
        gmm_criterion:  str = "bic",
        r0_km:          float = _R0_KM,
        scale_boundaries_km: list[float] | None = None,
    ) -> None:
        self.output_dir     = Path(output_dir)
        self.max_dist_km    = max_dist_km
        self.gmm_components = gmm_components or [2, 3, 4, 5]
        self.gmm_criterion  = gmm_criterion
        self.r0_km          = r0_km
        self.scale_bins     = _bins_from_boundaries(scale_boundaries_km) if scale_boundaries_km else SCALE_BINS

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        real_trajs: Any,
        sim_trajs:  Any,
        *,
        run_scaling: bool = True,
        run_plots:   bool = True,
        snap_grid: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run full hierarchical evaluation.

        Parameters
        ----------
        snap_grid : (N, 2) array of (lat, lng) reference locations.
                    When provided, all sim coordinates are snapped to the
                    nearest reference location before evaluation. Use
                    ``build_location_grid(train_jsonl)`` to construct this
                    from training data.

        Returns a structured dict with keys:
          micro, meso, paper, macro, jsd_scores, figures.
        """
        real = normalize_trajectories(real_trajs)
        sim  = normalize_trajectories(sim_trajs)

        if not real or not sim:
            raise ValueError("Both real_trajs and sim_trajs must contain at least one valid trajectory.")

        if snap_grid is not None:
            sim = snap_to_locations(sim, snap_grid)

        real = coarse_grain(real)
        sim  = coarse_grain(sim)

        # ── Micro-scale ──────────────────────────────────────────────────────
        sd_jsd   = compute_sd_jsd(real, sim)
        si_jsd   = compute_si_jsd(real, sim)
        stay_jsd = compute_stay_jsd(real, sim)
        micro = dict(
            SD   = {"jsd": sd_jsd},
            SI   = {"jsd": si_jsd},
            STAY = {"jsd": stay_jsd},
        )

        # ── Meso-scale ───────────────────────────────────────────────────────
        rg_jsd = compute_rg_jsd(real, sim)
        dtd_stats = compute_dtd(real, sim)
        dul_stats = compute_dul(real, sim)

        meso = dict(
            RG  = {"jsd": rg_jsd},
            DTD = dtd_stats,
            DUL = dul_stats,
        )

        # ── Behavioural (paper-aligned) ──────────────────────────────────────
        visitation_stats = compute_visitation_zeta_jsd(real, sim)
        visitation_jsd   = visitation_stats["jsd"]
        irank_jsd        = compute_irank_jsd(real, sim)
        paper = dict(
            Visitation   = visitation_stats,
            IRank        = {"jsd": irank_jsd},
        )

        # ── Macro-scale ──────────────────────────────────────────────────────
        macro: dict[str, Any] = {}
        r_dist = np.array([]); s_dist = np.array([])

        if run_scaling:
            r_dist = _extract_distances_km(real, self.max_dist_km)
            s_dist = _extract_distances_km(sim,  self.max_dist_km)

            real_info = _analyse_distances(r_dist, self.scale_bins)
            sim_info  = _analyse_distances(s_dist, self.scale_bins)
            delta     = _compute_delta(real_info, sim_info)

            macro = dict(
                power_law=dict(
                    real=real_info["power_law"],
                    sim=sim_info["power_law"],
                    delta_beta=delta["delta_beta"],
                    delta_kappa=delta["delta_kappa"],
                ),
                gmm=dict(real=real_info["gmm"], sim=sim_info["gmm"]),
                scale_buckets=dict(
                    real=real_info["scale_buckets"],
                    sim=sim_info["scale_buckets"],
                    delta=delta["bucket_fraction_delta"],
                ),
            )

            scale_jsd_result = compute_scale_jsd(r_dist, s_dist, self.scale_bins)
            macro["scale_jsd_result"] = scale_jsd_result

        # ── Flat JSD summary ─────────────────────────────────────────────────
        jsd_scores = dict(
            SD          = sd_jsd,
            SI          = si_jsd,
            STAY        = stay_jsd,
            RG          = rg_jsd,
            DTD         = dtd_stats["jsd"],
            DUL         = dul_stats["jsd"],
            Visitation  = visitation_jsd,
            IRank       = irank_jsd,
        )
        if run_scaling and "scale_jsd_result" in macro:
            sjr = macro["scale_jsd_result"]
            jsd_scores["SFP"] = sjr["scale_jsd"]

        # ── Figures ──────────────────────────────────────────────────────────
        figures: dict[str, str] = {}
        if run_plots:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            try:
                figures.update(plot_micro_distributions(real, sim, self.output_dir))
            except Exception as exc:
                warnings.warn(f"Micro distribution plots failed: {exc}", stacklevel=2)

            try:
                figures.update(plot_meso_distributions(real, sim, self.output_dir))
            except Exception as exc:
                warnings.warn(f"Meso distribution plots failed: {exc}", stacklevel=2)

            if run_scaling and r_dist.size and s_dist.size:
                try:
                    figures["ccdf_fit"] = plot_ccdf_fit(
                        r_dist, s_dist,
                        real_info["power_law"], sim_info["power_law"],
                        real_info["gmm"], sim_info["gmm"],
                        self.output_dir,
                    )
                except Exception as exc:
                    warnings.warn(f"CCDF fit plot failed: {exc}", stacklevel=2)

        return dict(
            micro      = micro,
            meso       = meso,
            paper      = paper,
            macro      = macro,
            jsd_scores = jsd_scores,
            figures    = figures,
        )

    def jsd_summary(self, real_trajs: Any, sim_trajs: Any) -> dict[str, float]:
        """Return only the flat JSD scores (no plots, no scaling)."""
        r = coarse_grain(normalize_trajectories(real_trajs))
        s = coarse_grain(normalize_trajectories(sim_trajs))
        return dict(
            SD   = compute_sd_jsd(r, s),
            SI   = compute_si_jsd(r, s),
            STAY = compute_stay_jsd(r, s),
            RG   = compute_rg_jsd(r, s),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    real_trajs:     Any,
    sim_trajs:      Any,
    output_dir:     str = "eval_output",
    max_dist_km:    float = math.inf,
    gmm_components: list[int] | None = None,
    run_scaling:    bool = True,
    run_plots:      bool = True,
    scale_boundaries_km: list[float] | None = None,
) -> dict[str, Any]:
    """Evaluate simulated trajectories against real ground truth.

    Returns
    -------
    dict with keys: micro, meso, paper, macro, jsd_scores, figures.
    """
    ev = MobilityEvaluator(
        output_dir=output_dir,
        max_dist_km=max_dist_km,
        gmm_components=gmm_components,
        scale_boundaries_km=scale_boundaries_km,
    )
    return ev.evaluate(real_trajs, sim_trajs, run_scaling=run_scaling, run_plots=run_plots)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint: python -m src.evaluation.mobility_evaluator ...
# ─────────────────────────────────────────────────────────────────────────────

def _main() -> None:
    import argparse as _ap
    import json as _json
    import sys as _sys

    p = _ap.ArgumentParser(
        description="Evaluate a synthetic trajectory JSONL against a real one; "
                    "writes metrics.json to --output.",
    )
    p.add_argument("--real", required=True,
                   help="Path to real (test) trajectories JSONL.")
    p.add_argument("--sim", required=True,
                   help="Path to simulated trajectories JSONL.")
    p.add_argument("--output", required=True,
                   help="Output directory; metrics.json will be written here.")
    p.add_argument("--scale-boundaries-km", default=None,
                   help="Comma-separated K-1 GMM decision boundaries in km "
                        "(e.g. '0.6,2.7,23.8' for GeoLife). If omitted, uses "
                        "the evaluator's default fallback bins.")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip figure generation.")
    args = p.parse_args()

    boundaries = None
    if args.scale_boundaries_km:
        boundaries = [float(x) for x in args.scale_boundaries_km.split(",")]

    real_path = Path(args.real)
    sim_path = Path(args.sim)
    out_dir = Path(args.output)
    if not real_path.exists():
        print(f"ERROR: --real not found: {real_path}", file=_sys.stderr); _sys.exit(1)
    if not sim_path.exists():
        print(f"ERROR: --sim not found: {sim_path}", file=_sys.stderr); _sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(real_path) as f:
        real_trajs = [_json.loads(line) for line in f]
    with open(sim_path) as f:
        sim_trajs = [_json.loads(line) for line in f]
    print(f"  real: {len(real_trajs)} records; sim: {len(sim_trajs)} records")

    ev = MobilityEvaluator(output_dir=str(out_dir), scale_boundaries_km=boundaries)
    results = ev.evaluate(
        real_trajs, sim_trajs,
        run_scaling=True, run_plots=not args.no_plots,
    )
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        _json.dump(results, f, indent=2, default=str)

    js = results.get("jsd_scores", {})
    print(f"  SD={js.get('SD',0):.4f}  RG={js.get('RG',0):.4f}  "
          f"SI={js.get('SI',0):.4f}  STAY={js.get('STAY',0):.4f}")
    print(f"  DTD={js.get('DTD',0):.4f}  DUL={js.get('DUL',0):.4f}  "
          f"Visit={js.get('Visitation',0):.4f}  IRank={js.get('IRank',0):.4f}")
    print(f"  SFP={js.get('SFP',0):.4f}")
    pl = results.get("macro", {}).get("power_law", {})
    if pl:
        print(f"  |Δβ|={abs(pl.get('delta_beta',0)):.4f}  "
              f"|Δκ|={abs(pl.get('delta_kappa',0)):.2f} km")
    print(f"  metrics saved: {metrics_path}")


if __name__ == "__main__":
    _main()
