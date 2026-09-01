"""PyTorch Dataset for ScaleTraj training.

Each sample is a sliding window of consecutive days per user.
Tokens: (cell_id, scale, duration, arrival) per activity stop.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .tokenizer import (
    DURATION_BIN_EDGES_MIN,
    N_ACTIVITY,
    N_ARRIVAL_BINS,
    N_DURATION_BINS,
    SPECIAL_TOKENS,
    DaySequence,
    derive_activity,
)

PAD_DUR = N_DURATION_BINS
PAD_ARR = N_ARRIVAL_BINS
PAD_ACT = N_ACTIVITY


def make_special_ids(n_vocab: int) -> dict[str, int]:
    """Create special token IDs offset from vocab size."""
    return {name: n_vocab + idx for name, idx in SPECIAL_TOKENS.items()}


class TrajectoryDataset(Dataset):
    """Multi-day sliding window dataset for cross-day Transformer training."""

    MAX_HIST_DAYS = 30
    MAX_HIST_TOKENS = 200

    def __init__(
        self,
        sequences: list[DaySequence],
        user_to_persona: dict[str, int],
        n_cells: int,
        n_scales: int,
        user_anchors: dict[str, dict[str, int | None]] | None = None,
        context_days: int = 3,
        max_tokens_per_day: int = 20,
        user_to_idx: dict[str, int] | None = None,
    ):
        self.user_to_persona = user_to_persona
        self.context_days = context_days
        self.max_per_day = max_tokens_per_day
        self.n_cells = n_cells
        self.n_scales = n_scales
        self.user_anchors = user_anchors or {}

        self.cell_special = make_special_ids(n_cells)
        self.scale_special = make_special_ids(n_scales)

        pad_cell = n_cells + SPECIAL_TOKENS["[PAD]"]

        by_user: dict[str, list[DaySequence]] = defaultdict(list)
        for seq in sequences:
            by_user[seq.user_id].append(seq)
        for uid in by_user:
            by_user[uid].sort(key=lambda s: s.date)

        if user_to_idx is None:
            user_to_idx = {uid: i for i, uid in enumerate(sorted(by_user.keys()))}
        self.user_to_idx = user_to_idx
        self.n_users = len(user_to_idx)

        self.samples: list[dict] = []
        for uid, days in by_user.items():
            persona = user_to_persona.get(uid, 0)
            user_idx = user_to_idx.get(uid, 0)
            anchors = self.user_anchors.get(uid, {})
            home_cell_raw = anchors.get("home")
            work_cell_raw = anchors.get("work")
            home_cell = home_cell_raw if home_cell_raw is not None else pad_cell
            work_cell = work_cell_raw if work_cell_raw is not None else pad_cell
            for i in range(len(days)):
                ctx_start = max(0, i - context_days)
                ctx_days_list = days[ctx_start:i]
                hist_end = max(0, i - context_days)
                hist_start = max(0, hist_end - self.MAX_HIST_DAYS)
                hist_days_list = days[hist_start:hist_end]
                target = days[i]
                self.samples.append({
                    "user_id": uid,
                    "user_idx": user_idx,
                    "persona_id": persona,
                    "context": ctx_days_list,
                    "history": hist_days_list,
                    "target": target,
                    "home_cell": home_cell,
                    "work_cell": work_cell,
                    "home_cell_real": home_cell_raw,
                    "work_cell_real": work_cell_raw,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        target = sample["target"]
        home_real = sample["home_cell_real"]
        work_real = sample["work_cell_real"]

        BOS_CELL = self.cell_special["[BOS]"]
        SEP_CELL = self.cell_special["[SEP]"]
        EOS_CELL = self.cell_special["[EOS]"]
        BOS_SCALE = self.scale_special["[BOS]"]
        SEP_SCALE = self.scale_special["[SEP]"]
        EOS_SCALE = self.scale_special["[EOS]"]

        cells = [BOS_CELL]
        scales = [BOS_SCALE]
        durs = [0]
        arrs = [0]
        loc_types = [0]
        activities = [PAD_ACT]

        for ctx_day in sample["context"]:
            for tok in ctx_day.tokens[: self.max_per_day]:
                cells.append(tok.cell_id)
                scales.append(tok.scale)
                durs.append(tok.duration)
                arrs.append(tok.arrival)
                loc_types.append(tok.loc_type)
                arr_min = tok.arrival * (1440.0 / N_ARRIVAL_BINS)
                dur_min = DURATION_BIN_EDGES_MIN[min(tok.duration, N_DURATION_BINS - 1)]
                activities.append(derive_activity(tok.cell_id, home_real, work_real, arr_min, dur_min))
            cells.append(SEP_CELL)
            scales.append(SEP_SCALE)
            durs.append(0)
            arrs.append(0)
            loc_types.append(0)
            activities.append(PAD_ACT)

        ctx_len = len(scales)

        target_durs_min: list[float] = []
        target_scales_list: list[int] = []
        for tok in target.tokens[: self.max_per_day]:
            cells.append(tok.cell_id)
            scales.append(tok.scale)
            durs.append(tok.duration)
            arrs.append(tok.arrival)
            loc_types.append(tok.loc_type)
            arr_min = tok.arrival * (1440.0 / N_ARRIVAL_BINS)
            dur_min = DURATION_BIN_EDGES_MIN[min(tok.duration, N_DURATION_BINS - 1)]
            activities.append(derive_activity(tok.cell_id, home_real, work_real, arr_min, dur_min))
            target_durs_min.append(dur_min)
            target_scales_list.append(tok.scale)

        cells.append(EOS_CELL)
        scales.append(EOS_SCALE)
        durs.append(PAD_DUR)
        arrs.append(PAD_ARR)
        loc_types.append(0)
        activities.append(PAD_ACT)

        n_visits_target = float(len(target_durs_min))
        mean_dur_target = (sum(target_durs_min) / max(len(target_durs_min), 1)) if target_durs_min else 0.0
        max_scale_target = float(max(target_scales_list)) if target_scales_list else 0.0

        hist_cells, hist_scales, hist_durs, hist_arrs, hist_loc_types = [], [], [], [], []
        hist_day_boundaries = []
        for hist_day in sample["history"]:
            day_start = len(hist_cells)
            for tok in hist_day.tokens[: self.max_per_day]:
                hist_cells.append(tok.cell_id)
                hist_scales.append(tok.scale)
                hist_durs.append(tok.duration)
                hist_arrs.append(tok.arrival)
                hist_loc_types.append(tok.loc_type)
            day_end = len(hist_cells)
            if day_end > day_start:
                hist_day_boundaries.append((day_start, day_end))
            if len(hist_cells) >= self.MAX_HIST_TOKENS:
                break

        n_hist_days = len(hist_day_boundaries)
        n_hist_tokens = len(hist_cells)

        if n_hist_tokens == 0:
            hist_cells = [0]
            hist_scales = [0]
            hist_durs = [0]
            hist_arrs = [0]
            hist_loc_types = [0]
            n_hist_tokens = 1

        if n_hist_days == 0:
            hist_day_boundaries = [(0, 0)]

        return {
            "cells": torch.tensor(cells, dtype=torch.long),
            "scales": torch.tensor(scales, dtype=torch.long),
            "durations": torch.tensor(durs, dtype=torch.long),
            "arrivals": torch.tensor(arrs, dtype=torch.long),
            "loc_types": torch.tensor(loc_types, dtype=torch.long),
            "activities": torch.tensor(activities, dtype=torch.long),
            "day_type": torch.tensor(target.day_type, dtype=torch.long),
            "persona_id": torch.tensor(sample["persona_id"], dtype=torch.long),
            "user_idx": torch.tensor(sample["user_idx"], dtype=torch.long),
            "ctx_len": torch.tensor(ctx_len, dtype=torch.long),
            "home_cell": torch.tensor(sample["home_cell"], dtype=torch.long),
            "work_cell": torch.tensor(sample["work_cell"], dtype=torch.long),
            "hist_cells": torch.tensor(hist_cells, dtype=torch.long),
            "hist_scales": torch.tensor(hist_scales, dtype=torch.long),
            "hist_durations": torch.tensor(hist_durs, dtype=torch.long),
            "hist_arrivals": torch.tensor(hist_arrs, dtype=torch.long),
            "hist_loc_types": torch.tensor(hist_loc_types, dtype=torch.long),
            "hist_day_counts": torch.tensor(n_hist_days, dtype=torch.long),
            "hist_day_boundaries": torch.tensor(hist_day_boundaries, dtype=torch.long),
            "daily_stats": torch.tensor(
                [n_visits_target, mean_dur_target / 1440.0, max_scale_target],
                dtype=torch.float32,
            ),
        }


def collate_fn(batch: list[dict], n_cells: int, n_scales: int) -> dict[str, torch.Tensor]:
    PAD_CELL = n_cells + SPECIAL_TOKENS["[PAD]"]
    PAD_SCALE = n_scales + SPECIAL_TOKENS["[PAD]"]
    max_len = max(b["scales"].size(0) for b in batch)

    max_hist_tokens = max(b["hist_cells"].size(0) for b in batch)
    max_hist_days = max(b["hist_day_boundaries"].size(0) for b in batch)

    padded = {
        "cells": [],
        "scales": [],
        "durations": [],
        "arrivals": [],
        "loc_types": [],
        "activities": [],
        "day_type": [],
        "persona_id": [],
        "user_idx": [],
        "ctx_len": [],
        "home_cell": [],
        "work_cell": [],
        "pad_mask": [],
        "hist_cells": [],
        "hist_scales": [],
        "hist_durations": [],
        "hist_arrivals": [],
        "hist_loc_types": [],
        "hist_day_counts": [],
        "hist_day_boundaries": [],
        "daily_stats": [],
    }

    for b in batch:
        T = b["scales"].size(0)
        pad_len = max_len - T

        padded["cells"].append(F.pad(b["cells"], (0, pad_len), value=PAD_CELL))
        padded["scales"].append(F.pad(b["scales"], (0, pad_len), value=PAD_SCALE))
        padded["durations"].append(F.pad(b["durations"], (0, pad_len), value=PAD_DUR))
        padded["arrivals"].append(F.pad(b["arrivals"], (0, pad_len), value=PAD_ARR))
        padded["loc_types"].append(F.pad(b["loc_types"], (0, pad_len), value=0))
        padded["activities"].append(F.pad(b["activities"], (0, pad_len), value=PAD_ACT))
        padded["day_type"].append(b["day_type"])
        padded["persona_id"].append(b["persona_id"])
        padded["user_idx"].append(b["user_idx"])
        padded["ctx_len"].append(b["ctx_len"])
        padded["home_cell"].append(b["home_cell"])
        padded["work_cell"].append(b["work_cell"])
        padded["daily_stats"].append(b["daily_stats"])
        mask = torch.cat([torch.zeros(T, dtype=torch.bool), torch.ones(pad_len, dtype=torch.bool)])
        padded["pad_mask"].append(mask)

        ht = b["hist_cells"].size(0)
        hpad = max_hist_tokens - ht
        padded["hist_cells"].append(F.pad(b["hist_cells"], (0, hpad), value=0))
        padded["hist_scales"].append(F.pad(b["hist_scales"], (0, hpad), value=0))
        padded["hist_durations"].append(F.pad(b["hist_durations"], (0, hpad), value=0))
        padded["hist_arrivals"].append(F.pad(b["hist_arrivals"], (0, hpad), value=0))
        padded["hist_loc_types"].append(F.pad(b["hist_loc_types"], (0, hpad), value=0))
        padded["hist_day_counts"].append(b["hist_day_counts"])

        hd = b["hist_day_boundaries"].size(0)
        dpad = max_hist_days - hd
        padded["hist_day_boundaries"].append(
            F.pad(b["hist_day_boundaries"], (0, 0, 0, dpad), value=0)
        )

    return {k: torch.stack(v) for k, v in padded.items()}
