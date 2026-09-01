"""Tokenize real trajectories into discrete multi-field tokens for Transformer training.

Each activity stop becomes a token with fields:
    (cell_id, scale, duration_bin, arrival_bin)

cell_id: grid cell index built from occupied cells in training data
scale:   GMM posterior assignment on log(haversine distance)
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DAY_TYPE_VOCAB = ["workday", "weekend", "holiday"]
N_DAY_TYPES = len(DAY_TYPE_VOCAB)
N_DURATION_BINS = 24
N_ARRIVAL_BINS = 288  # 5-min slots over 24h

SPECIAL_TOKENS = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[SEP]": 3}

DURATION_BIN_EDGES_MIN = [
    0, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120,
    150, 180, 210, 240, 300, 360, 420, 480, 600, 720, 900, 1080, 1440,
]


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def duration_to_bin(dur_min: float) -> int:
    for i in range(len(DURATION_BIN_EDGES_MIN) - 1):
        if dur_min < DURATION_BIN_EDGES_MIN[i + 1]:
            return i
    return N_DURATION_BINS - 1


def arrival_to_bin(start_min: float) -> int:
    return min(int(start_min // 5), N_ARRIVAL_BINS - 1)


# ---------------------------------------------------------------------------
# Grid cell vocabulary
# ---------------------------------------------------------------------------

@dataclass
class GridVocab:
    """Grid vocabulary with compact cell IDs from training data."""
    snap_deg: float
    lng_min: float = 0.0
    lat_min: float = 0.0
    n_rows: int = 0
    n_cols: int = 0
    cell_to_id: dict[int, int] = field(default_factory=dict)  # city_cell_id -> compact_id
    id_to_cell: list[tuple[float, float]] = field(default_factory=list)  # compact_id -> (lng_centroid, lat_centroid)
    cell_counts: list[int] = field(default_factory=list)

    @property
    def n_cells(self) -> int:
        return len(self.id_to_cell)

    @property
    def lng_max(self) -> float:
        return self.lng_min + self.n_cols * self.snap_deg

    @property
    def lat_max(self) -> float:
        return self.lat_min + self.n_rows * self.snap_deg

    def coord_to_cell_id(self, lon: float, lat: float) -> int:
        """Returns compact cell id, or -1 if out of bbox / out of train vocab."""
        if not (self.lng_min <= lon <= self.lng_max
                and self.lat_min <= lat <= self.lat_max):
            return -1
        col = int((lon - self.lng_min) / self.snap_deg)
        row = int((lat - self.lat_min) / self.snap_deg)
        col = min(col, self.n_cols - 1)
        row = min(row, self.n_rows - 1)
        city_cell = row * self.n_cols + col
        return self.cell_to_id.get(city_cell, -1)

    def save(self, path: str | Path):
        path = Path(path)
        # Persist city_cell_id in compact order so we can reconstruct.
        city_ids = [0] * len(self.id_to_cell)
        for city_id, compact_id in self.cell_to_id.items():
            city_ids[compact_id] = city_id
        data = {
            "snap_deg": self.snap_deg,
            "lng_min": self.lng_min, "lat_min": self.lat_min,
            "n_rows": self.n_rows, "n_cols": self.n_cols,
            "city_cell_ids": city_ids,
            "cell_counts": self.cell_counts,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> GridVocab:
        with open(path) as f:
            data = json.load(f)
        if "city_cell_ids" not in data:
            raise ValueError(
                f"grid_vocab.json at {path} is missing 'city_cell_ids'; "
                "regenerate with the current vocab builder."
            )
        vocab = cls(
            snap_deg=data["snap_deg"],
            lng_min=data["lng_min"], lat_min=data["lat_min"],
            n_rows=data["n_rows"], n_cols=data["n_cols"],
        )
        for compact_id, city_cell in enumerate(data["city_cell_ids"]):
            vocab.cell_to_id[city_cell] = compact_id
            col = city_cell % vocab.n_cols
            row = city_cell // vocab.n_cols
            cx = vocab.lng_min + (col + 0.5) * vocab.snap_deg
            cy = vocab.lat_min + (row + 0.5) * vocab.snap_deg
            vocab.id_to_cell.append((cx, cy))
        vocab.cell_counts = list(data["cell_counts"])
        return vocab


def build_grid_vocab(
    train_path: str | Path,
    snap_deg: float = 0.005,
    min_count: int = 1,
) -> GridVocab:
    """Baseline-aligned vocab construction.

    1. Collect all training (lng, lat) stays.
    2. bbox = floor(min/snap)*snap .. ceil(max/snap)*snap.
    3. city_cell_id = row*n_cols + col, row/col from int((coord-min)/snap).
    4. Keep cells with count >= min_count (default 1 = compact vocab, == baseline).
    5. Compact IDs assigned by descending frequency.
    """
    lngs, lats = [], []
    with open(train_path) as f:
        for line in f:
            rec = json.loads(line)
            for s in rec.get("stays", []):
                lngs.append(s["coord"][0])
                lats.append(s["coord"][1])
    if not lngs:
        raise ValueError(f"No stays found in {train_path}")
    lngs = np.asarray(lngs); lats = np.asarray(lats)

    lng_min = float(np.floor(lngs.min() / snap_deg) * snap_deg)
    lat_min = float(np.floor(lats.min() / snap_deg) * snap_deg)
    lng_max = float(np.ceil(lngs.max() / snap_deg) * snap_deg)
    lat_max = float(np.ceil(lats.max() / snap_deg) * snap_deg)
    n_cols = int(round((lng_max - lng_min) / snap_deg))
    n_rows = int(round((lat_max - lat_min) / snap_deg))

    cols = np.clip(((lngs - lng_min) / snap_deg).astype(int), 0, n_cols - 1)
    rows = np.clip(((lats - lat_min) / snap_deg).astype(int), 0, n_rows - 1)
    city_cells = rows * n_cols + cols

    counter: Counter[int] = Counter(int(c) for c in city_cells)

    vocab = GridVocab(
        snap_deg=snap_deg, lng_min=lng_min, lat_min=lat_min,
        n_rows=n_rows, n_cols=n_cols,
    )
    for city_cell, cnt in counter.most_common():
        if cnt < min_count:
            break
        col = city_cell % n_cols
        row = city_cell // n_cols
        cx = lng_min + (col + 0.5) * snap_deg
        cy = lat_min + (row + 0.5) * snap_deg
        vocab.cell_to_id[city_cell] = len(vocab.id_to_cell)
        vocab.id_to_cell.append((cx, cy))
        vocab.cell_counts.append(cnt)

    logger.info(
        "Grid vocab (baseline-aligned): %d cells at %.4f° (bbox=[%.4f,%.4f] x [%.4f,%.4f], %d×%d grid, min_count=%d)",
        vocab.n_cells, snap_deg, lng_min, lng_max, lat_min, lat_max, n_cols, n_rows, min_count,
    )
    return vocab


# ---------------------------------------------------------------------------
# GMM scale decomposition
# ---------------------------------------------------------------------------

@dataclass
class ScaleGMM:
    """GMM fitted on log(distance) for data-driven scale decomposition."""
    n_components: int
    means_log: np.ndarray      # (K,) mean in log-km space
    variances_log: np.ndarray  # (K,) variance in log-km space
    weights: np.ndarray        # (K,) mixture weights
    order: np.ndarray          # (K,) indices sorted by mean ascending

    @property
    def means_km(self) -> np.ndarray:
        return np.exp(self.means_log[self.order])

    @property
    def scale_names(self) -> list[str]:
        k = self.n_components
        if k == 4:
            return ["neighbourhood", "city", "urban-agglomeration", "region"]
        if k == 5:
            return ["neighbourhood", "city", "urban-agglomeration", "regional", "region"]
        return [f"s{i+1}" for i in range(k)]

    def dist_to_scale_idx(self, dist_km: float) -> int:
        """Assign scale by GMM posterior (MAP assignment).

        Training-time labeling only (baked into existing checkpoints via the
        tokenized training labels, personas, and marginal prior). Omits the
        -0.5*log(var) normalizing term that _gmm_scale_boundaries_km() (used
        for the SFP evaluator and Figure 2) correctly includes, so this is an
        approximation of the true GMM decision boundary, not identical to it.
        Left as-is intentionally: changing it would not affect any existing
        checkpoint but would make freshly-retrained runs diverge from the
        models that actually produced the paper's numbers.
        """
        if dist_km < 0.01:
            return int(self.order[0])
        log_d = math.log(dist_km)
        best_idx = 0
        best_score = -float("inf")
        for k in range(self.n_components):
            mu = self.means_log[k]
            var = self.variances_log[k]
            w = self.weights[k]
            score = math.log(w + 1e-30) - 0.5 * ((log_d - mu) ** 2 / max(var, 1e-9))
            if score > best_score:
                best_score = score
                best_idx = k
        for rank, orig_idx in enumerate(self.order):
            if orig_idx == best_idx:
                return rank
        return 0

    def save(self, path: str | Path):
        data = {
            "n_components": self.n_components,
            "means_log": self.means_log.tolist(),
            "variances_log": self.variances_log.tolist(),
            "weights": self.weights.tolist(),
            "order": self.order.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> ScaleGMM:
        with open(path) as f:
            data = json.load(f)
        return cls(
            n_components=data["n_components"],
            means_log=np.array(data["means_log"]),
            variances_log=np.array(data["variances_log"]),
            weights=np.array(data["weights"]),
            order=np.array(data["order"], dtype=int),
        )


def fit_scale_gmm(
    train_path: str | Path,
    n_components: int = 4,
    min_dist_km: float = 0.01,
    max_sigma_log: float | None = None,
) -> ScaleGMM:
    """Fit GMM on log(haversine distance) from all consecutive-stop transitions.

    Each component corresponds to one mobility scale level.
    Theory: Alessandretti et al. (2020) — container model predicts
    log-normal displacement per hierarchical level.

    `max_sigma_log` (optional) caps each component's sigma in log-space and
    re-runs EM on the constrained covariances. This prevents a sparse macro
    component with a runaway sigma (long-tail blow-up).
    """
    from sklearn.mixture import GaussianMixture

    dists: list[float] = []
    with open(train_path) as f:
        for line in f:
            rec = json.loads(line)
            stays = rec.get("stays", [])
            for i in range(1, len(stays)):
                c0, c1 = stays[i - 1]["coord"], stays[i]["coord"]
                d = _haversine_km(c0[0], c0[1], c1[0], c1[1])
                if d > min_dist_km:
                    dists.append(d)

    log_d = np.log(dists).reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=10)
    gmm.fit(log_d)

    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    weights = gmm.weights_

    if max_sigma_log is not None:
        max_var = float(max_sigma_log) ** 2
        clamped_idx = np.where(variances > max_var)[0]
        if len(clamped_idx) > 0:
            logger.info("Clamping %d component(s) variance to <= %.4f (sigma <= %.2f)",
                        len(clamped_idx), max_var, max_sigma_log)
            variances = np.minimum(variances, max_var)
            # E-step with clamped covariances, then re-fit means + weights via M-step.
            means, variances, weights = _refit_gmm_constrained(
                log_d.flatten(), means, variances, weights, n_iter=20,
                max_sigma_log=max_sigma_log,
            )

    order = np.argsort(means)

    if n_components == 4:
        _names = ["neighbourhood", "city", "urban-agglomeration", "region"]
    elif n_components == 5:
        _names = ["neighbourhood", "city", "urban-agglomeration", "regional", "region"]
    else:
        _names = [f"s{i+1}" for i in range(n_components)]

    logger.info("Scale GMM (K=%d):", n_components)
    for rank, idx in enumerate(order):
        logger.info("  scale %d (%s): mean=%.2f km, sigma_log=%.2f, weight=%.3f",
                     rank, _names[rank],
                     math.exp(means[idx]), math.sqrt(variances[idx]), weights[idx])

    return ScaleGMM(
        n_components=n_components,
        means_log=means,
        variances_log=variances,
        weights=weights,
        order=order,
    )


def _refit_gmm_constrained(
    log_d: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray,
    n_iter: int = 20,
    max_sigma_log: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-run EM with sigma capped each M-step. Plain Gaussian mixture in 1D."""
    K = len(means)
    x = log_d.reshape(-1, 1)  # (N, 1)
    max_var = float(max_sigma_log) ** 2 if max_sigma_log is not None else None

    for _ in range(n_iter):
        # E-step
        diff = x - means.reshape(1, K)  # (N, K)
        log_pdf = -0.5 * (np.log(2 * np.pi * variances) + diff ** 2 / variances)
        log_resp = log_pdf + np.log(weights)
        log_norm = np.logaddexp.reduce(log_resp, axis=1, keepdims=True)
        resp = np.exp(log_resp - log_norm)  # (N, K)

        # M-step
        Nk = resp.sum(axis=0)
        Nk = np.maximum(Nk, 1e-12)
        weights = Nk / Nk.sum()
        means = (resp * x).sum(axis=0) / Nk
        new_var = (resp * (x - means.reshape(1, K)) ** 2).sum(axis=0) / Nk
        if max_var is not None:
            new_var = np.minimum(new_var, max_var)
        variances = np.maximum(new_var, 1e-6)

    return means, variances, weights


# ---------------------------------------------------------------------------
# Token / sequence data structures
# ---------------------------------------------------------------------------

LOC_OTHER, LOC_HOME, LOC_WORK = 0, 1, 2
N_LOC_TYPES = 3


@dataclass
class ActivityToken:
    cell_id: int     # grid cell index
    scale: int       # GMM-assigned scale index (sorted by mean)
    duration: int    # index into duration bins
    arrival: int     # index into arrival bins
    loc_type: int = LOC_OTHER  # 0=other, 1=home, 2=work


@dataclass
class DaySequence:
    user_id: str
    date: str
    day_type: int    # index into DAY_TYPE_VOCAB
    tokens: list[ActivityToken]


# ---------------------------------------------------------------------------
# Home / work anchor extraction
# ---------------------------------------------------------------------------

def extract_user_anchors(
    train_path: str | Path,
    grid_vocab: GridVocab,
) -> dict[str, dict[str, int | None]]:
    """Identify home and work cell_id per user from training data.

    Home: most frequent cell during night hours (20h-8h).
    Work: most frequent cell during day hours (8h-18h), excluding home cell.
    Returns {uid: {"home": cell_id_or_None, "work": cell_id_or_None}}.
    """
    from collections import Counter as _Counter

    user_night: dict[str, _Counter] = {}
    user_day: dict[str, _Counter] = {}

    with open(train_path) as f:
        for line in f:
            rec = json.loads(line)
            uid = rec["user_id"]
            if uid not in user_night:
                user_night[uid] = _Counter()
                user_day[uid] = _Counter()
            for stay in rec.get("stays", []):
                h = int(stay["start_min"] // 60) % 24
                cid = grid_vocab.coord_to_cell_id(stay["coord"][0], stay["coord"][1])
                if cid < 0:
                    continue  # baseline-aligned: drop out-of-vocab stays
                if h >= 20 or h < 8:
                    user_night[uid][cid] += 1
                elif 8 <= h < 18:
                    user_day[uid][cid] += 1

    anchors: dict[str, dict[str, int | None]] = {}
    for uid in user_night:
        home_cid = user_night[uid].most_common(1)[0][0] if user_night[uid] else None
        day_cells = user_day[uid]
        if home_cid is not None:
            day_cells.pop(home_cid, None)
        work_cid = day_cells.most_common(1)[0][0] if day_cells else None
        anchors[uid] = {"home": home_cid, "work": work_cid}
    return anchors


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_day(
    record: dict,
    grid_vocab: GridVocab,
    scale_gmm: ScaleGMM,
    user_anchors: dict[str, dict[str, int | None]] | None = None,
) -> DaySequence | None:
    stays = record.get("stays", [])
    if not stays:
        return None

    day_type = DAY_TYPE_VOCAB.index(record["day_type"]) if record["day_type"] in DAY_TYPE_VOCAB else 0

    uid = record["user_id"]
    home_cid = user_anchors[uid]["home"] if user_anchors and uid in user_anchors else None
    work_cid = user_anchors[uid]["work"] if user_anchors and uid in user_anchors else None

    tokens: list[ActivityToken] = []
    prev_kept_coord: tuple[float, float] | None = None
    for stay in stays:
        lon, lat = stay["coord"]
        dur_min = stay["end_min"] - stay["start_min"]
        if dur_min < 0:
            dur_min = 0.0

        cell_id = grid_vocab.coord_to_cell_id(lon, lat)
        if cell_id < 0:
            # Stay falls outside train-vocab grid → drop (baseline-aligned behavior).
            continue

        if prev_kept_coord is None:
            scale_idx = 0
        else:
            dist = _haversine_km(prev_kept_coord[0], prev_kept_coord[1], lon, lat)
            scale_idx = scale_gmm.dist_to_scale_idx(dist)

        if cell_id == home_cid:
            lt = LOC_HOME
        elif cell_id == work_cid:
            lt = LOC_WORK
        else:
            lt = LOC_OTHER

        tokens.append(ActivityToken(
            cell_id=cell_id,
            scale=scale_idx,
            duration=duration_to_bin(dur_min),
            arrival=arrival_to_bin(stay["start_min"]),
            loc_type=lt,
        ))
        prev_kept_coord = (lon, lat)

    if not tokens:
        return None

    return DaySequence(
        user_id=record["user_id"],
        date=record["date"],
        day_type=day_type,
        tokens=tokens,
    )


N_PERSONAS = 4

N_ACTIVITY = 4
ACTIVITY_HOME = 0
ACTIVITY_WORK = 1
ACTIVITY_TRANSIT = 2
ACTIVITY_OTHER = 3


def derive_activity(
    cell_id: int,
    home_cell: int | None,
    work_cell: int | None,
    arrival_min: float,
    duration_min: float,
) -> int:
    """Heuristic activity label from (cell, home/work anchors, time, duration).

    Used as auxiliary supervision signal so the model learns activity-aware
    representations without committing to discrete activity tokens at input.
    """
    if home_cell is not None and cell_id == home_cell:
        return ACTIVITY_HOME
    if work_cell is not None and cell_id == work_cell:
        return ACTIVITY_WORK
    if duration_min <= 30:
        return ACTIVITY_TRANSIT
    if 360 <= arrival_min <= 1320 and duration_min >= 180:
        return ACTIVITY_WORK
    if (arrival_min <= 360 or arrival_min >= 1260) and duration_min >= 240:
        return ACTIVITY_HOME
    return ACTIVITY_OTHER


def cluster_users(
    train_path: str | Path,
    scale_gmm: ScaleGMM,
    n_clusters: int = N_PERSONAS,
) -> dict[str, int]:
    """KMeans on per-user scale_weight vectors → persona cluster ID."""
    from sklearn.cluster import KMeans

    n_scales = scale_gmm.n_components
    user_steps: dict[str, list[float]] = {}
    with open(train_path) as f:
        for line in f:
            rec = json.loads(line)
            uid = rec["user_id"]
            stays = rec.get("stays", [])
            if uid not in user_steps:
                user_steps[uid] = [0.0] * n_scales
            for i in range(1, len(stays)):
                c0, c1 = stays[i - 1]["coord"], stays[i]["coord"]
                d = _haversine_km(c0[0], c0[1], c1[0], c1[1])
                si = scale_gmm.dist_to_scale_idx(d)
                user_steps[uid][si] += 1

    uids = []
    vecs = []
    for uid, counts in user_steps.items():
        total = sum(counts)
        if total > 0:
            uids.append(uid)
            vecs.append([c / total for c in counts])

    X = np.array(vecs)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return {uid: int(labels[i]) for i, uid in enumerate(uids)}


def tokenize_dataset(
    train_path: str | Path,
    grid_vocab: GridVocab,
    scale_gmm: ScaleGMM,
) -> tuple[list[DaySequence], dict[str, int], dict[str, dict[str, int | None]]]:
    """Tokenize an entire train.jsonl using pre-built grid vocab and scale GMM.

    Returns (sequences, user_id_to_idx, user_anchors).
    """
    user_anchors = extract_user_anchors(train_path, grid_vocab)
    user_ids: dict[str, int] = {}
    sequences: list[DaySequence] = []

    with open(train_path) as f:
        for line in f:
            rec = json.loads(line)
            uid = rec["user_id"]
            if uid not in user_ids:
                user_ids[uid] = len(user_ids)

            seq = tokenize_day(rec, grid_vocab, scale_gmm, user_anchors=user_anchors)
            if seq is not None:
                sequences.append(seq)

    return sequences, user_ids, user_anchors


def build_anchor_distributions(
    user_to_persona: dict[str, int],
    user_anchors: dict[str, dict[str, int | None]],
) -> dict[int, dict[str, dict[int, float]]]:
    """Build per-persona probability distributions over home/work cells.

    Returns {persona_id: {"home": {cell_id: prob}, "work": {cell_id: prob}}}.
    """
    from collections import Counter as _Counter
    from collections import defaultdict

    persona_home: dict[int, _Counter] = defaultdict(_Counter)
    persona_work: dict[int, _Counter] = defaultdict(_Counter)

    for uid, persona in user_to_persona.items():
        anchors = user_anchors.get(uid, {})
        if anchors.get("home") is not None:
            persona_home[persona][anchors["home"]] += 1
        if anchors.get("work") is not None:
            persona_work[persona][anchors["work"]] += 1

    distributions: dict[int, dict[str, dict[int, float]]] = {}
    for persona in set(user_to_persona.values()):
        hc = persona_home[persona]
        wc = persona_work[persona]
        h_total = sum(hc.values()) or 1
        w_total = sum(wc.values()) or 1
        distributions[persona] = {
            "home": {k: v / h_total for k, v in hc.items()},
            "work": {k: v / w_total for k, v in wc.items()},
        }

    return distributions


# ---------------------------------------------------------------------------
# Priors for generation
# ---------------------------------------------------------------------------

def fit_n_given_day_type(
    train_path: str | Path,
) -> dict[str, list[float]]:
    """Fit P(N|day_type) from training data — histogram of stops-per-day."""
    counts_by_dt: dict[str, Counter] = {dt: Counter() for dt in DAY_TYPE_VOCAB}

    with open(train_path) as f:
        for line in f:
            rec = json.loads(line)
            dt = rec.get("day_type", "workday")
            if dt not in counts_by_dt:
                dt = "workday"
            n_stays = len(rec.get("stays", []))
            if n_stays > 0:
                counts_by_dt[dt][n_stays] += 1

    result: dict[str, list[float]] = {}
    for dt in DAY_TYPE_VOCAB:
        ctr = counts_by_dt[dt]
        if not ctr:
            result[dt] = [0.0, 1.0]
            continue
        max_n = max(ctr.keys())
        total = sum(ctr.values())
        probs = [ctr.get(n, 0) / total for n in range(max_n + 1)]
        result[dt] = probs

    return result


def sample_n_from_prior(
    n_prior: dict[str, list[float]],
    day_type: str,
    rng_obj=None,
) -> int:
    """Sample N from P(N|day_type). Returns at least 2."""
    import random as _random

    probs = n_prior.get(day_type, n_prior.get("workday", [0.0, 1.0]))
    r = (rng_obj.random() if rng_obj else _random.random())
    cumsum = 0.0
    for n, p in enumerate(probs):
        cumsum += p
        if r <= cumsum:
            return max(n, 2)
    return max(len(probs) - 1, 2)


def fit_scale_prior(
    train_path: str | Path,
    scale_gmm: ScaleGMM,
) -> list[int]:
    """Compute per-scale training counts via GMM assignment.

    Returns raw counts (not proportions) so that Dirichlet posterior
    rebalancing can weight confidence by sample size.
    """
    n_scales = scale_gmm.n_components
    scale_counts = [0] * n_scales
    with open(train_path) as f:
        for line in f:
            rec = json.loads(line)
            stays = rec.get("stays", [])
            for i in range(1, len(stays)):
                c0 = stays[i - 1]["coord"]
                c1 = stays[i]["coord"]
                dist = _haversine_km(c0[0], c0[1], c1[0], c1[1])
                si = scale_gmm.dist_to_scale_idx(dist)
                scale_counts[si] += 1

    return scale_counts
