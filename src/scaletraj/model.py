"""ScaleTraj: scale-first cascade Transformer for trajectory generation.

Architecture:
  - Scale head: logits = α_marginal · log(empirical_scale_freq)
                       + α_h · MLP(h).
  - Cell head:  logits = α_dist · GMM_distance_prior(prev_cell, scale)
                       + α_res · per_scale_residual_MLP(h)
                       − Jacobian_correction(scale).
  - Duration head: classification + Gaussian regression.
  - Arrival head: classification with causal time constraint.

All α are fixed hyperparameters. Per-scale residual MLPs r^(k) (one per
scale band, contained inside the Scale-Aware Cell Selection Head) learn
scale-specific corrections.

Predicts (scale, cell_id, duration, arrival) per activity stop.
Cascaded head: scale → cell → duration → arrival.
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import (
    DURATION_BIN_EDGES_MIN,
    N_ACTIVITY,
    N_ARRIVAL_BINS,
    N_DURATION_BINS,
    N_PERSONAS,
    SPECIAL_TOKENS,
)

N_DAY_TYPES = 3
N_SPECIAL = len(SPECIAL_TOKENS)
N_LATENT = 16
N_DAILY_STATS = 3


class FourierTimeEncoding(nn.Module):
    def __init__(self, d_model: int, n_freqs: int = 16):
        super().__init__()
        self.register_buffer(
            "freqs",
            torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), n_freqs)),
        )
        self.proj = nn.Sequential(
            nn.Linear(n_freqs * 2 + 1, d_model),
            nn.GELU(),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = t.unsqueeze(-1) * self.freqs * (2 * math.pi)
        return self.proj(torch.cat([t.unsqueeze(-1), angles.sin(), angles.cos()], dim=-1))


def _apply_rope(x: torch.Tensor) -> torch.Tensor:
    B, T, d = x.shape
    half = d // 2
    pos = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(1)
    freqs = 1.0 / (10000.0 ** (torch.arange(0, half, device=x.device, dtype=x.dtype) / half))
    angles = pos * freqs
    cos_a = angles.cos().unsqueeze(0)
    sin_a = angles.sin().unsqueeze(0)
    x1, x2 = x[..., :half], x[..., half: 2 * half]
    out = torch.cat([x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a], dim=-1)
    if d % 2 == 1:
        out = torch.cat([out, x[..., -1:]], dim=-1)
    return out


def compute_log_dist_matrix(cell_coords: torch.Tensor, eps_km: float = 0.05) -> torch.Tensor:
    """Pairwise log haversine distance, shape (n_cells, n_cells)."""
    cell_coords.shape[0]
    lon = torch.deg2rad(cell_coords[:, 0])
    lat = torch.deg2rad(cell_coords[:, 1])
    dlon = lon.unsqueeze(0) - lon.unsqueeze(1)
    dlat = lat.unsqueeze(0) - lat.unsqueeze(1)
    a = (dlat / 2).sin() ** 2 + lat.unsqueeze(1).cos() * lat.unsqueeze(0).cos() * (dlon / 2).sin() ** 2
    dist_km = 2 * 6371.0 * a.clamp(max=1.0).sqrt().asin()
    return dist_km.clamp(min=eps_km).log()


def compute_per_scale_log_cell_density(
    log_dist: torch.Tensor,
    gmm_sigmas: torch.Tensor,
    bandwidth_factor: float = 0.5,
    bandwidth_floor: float = 0.05,
) -> torch.Tensor:
    """For each (scale s, prev_cell p, cell c), compute log of cell density
    at log_dist(p, c) under a Gaussian KDE with bandwidth = σ_s × factor.

    This is the Jacobian correction: cells at distance d from p have density
    ∝ d × (local cell density at d), and we want to subtract this from the
    geometric Gaussian log-prob so cell-level softmax doesn't systematically
    overshoot toward more populous rings.

    Returns: (n_scales, n_cells, n_cells)
    Memory: n_scales × n_cells² × 4 bytes.
    """
    n_cells = log_dist.shape[0]
    n_scales = gmm_sigmas.shape[0]
    device = log_dist.device
    bandwidths = (gmm_sigmas * bandwidth_factor).clamp(min=bandwidth_floor)
    result = torch.zeros(n_scales, n_cells, n_cells, device=device, dtype=log_dist.dtype)
    # Adaptive batch to keep peak memory ~2 GB
    batch = max(1, min(64, int(2e9 / (n_scales * n_cells * n_cells * 4 + 1))))
    for p0 in range(0, n_cells, batch):
        p1 = min(p0 + batch, n_cells)
        ld = log_dist[p0:p1]                                        # (B, C)
        diff = ld.unsqueeze(2) - ld.unsqueeze(1)                    # (B, C, C)
        diff_s = diff.unsqueeze(0) / bandwidths[:, None, None, None]  # (S, B, C, C)
        K = (-0.5 * diff_s ** 2).exp()
        density = K.mean(dim=3)                                     # (S, B, C)
        result[:, p0:p1, :] = density.clamp(min=1e-9).log()
    return result


class ScaleSelectionHead(nn.Module):
    """Scale prediction = α_marginal · log(empirical_scale_freq) + α_h · MLP(h).

    The empirical marginal anchors base rate (esp. for rare scales like macro).
    Residual MLP is intentionally small so it can only add context-dependent shifts.
    Both α are fixed hyperparameters (not learnable) to prevent collapse to MLE.
    """

    def __init__(self, d_model: int, n_scales: int,
                 alpha_marginal: float = 1.0, alpha_h: float = 1.0,
                 residual_hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.alpha_marginal = alpha_marginal
        self.alpha_h = alpha_h
        self.residual = nn.Sequential(
            nn.Linear(d_model, residual_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(residual_hidden, n_scales),
        )

    def forward(self, h: torch.Tensor, marginal_log: torch.Tensor) -> torch.Tensor:
        """h: (B, T, d_model); marginal_log: (n_scales,)."""
        residual_logits = self.residual(h)
        return self.alpha_marginal * marginal_log + self.alpha_h * residual_logits


class ScaleAwareCellSelectionHead(nn.Module):
    """Cell prediction = α_dist · (geometric_log_prob - Jacobian)
                       + α_res · MLP(h, scale_feat, prev_cell_emb).

    Geometric: Gaussian on log-distance, scale-conditional, with KDE Jacobian
    correction (subtract log cell-density to avoid systematic far-cell bias).
    Residual: small MLP — only learns context-dependent fine-tuning on top of
    the explicit priors.
    """

    def __init__(self, d_model: int, n_cells: int, n_scales: int,
                 alpha_dist: float = 2.0,
                 alpha_res: float = 0.5,
                 alpha_res_per_scale: list[float] | None = None,
                 d_scale_feat: int = 32, residual_hidden: int = 32, dropout: float = 0.1,
                 shared_residual: bool = False):
        super().__init__()
        self.n_cells = n_cells
        self.n_scales = n_scales
        self.alpha_dist = alpha_dist
        self.alpha_res = alpha_res
        self.shared_residual = shared_residual
        if alpha_res_per_scale is not None:
            assert len(alpha_res_per_scale) == n_scales, (
                f"alpha_res_per_scale must have length n_scales={n_scales}, "
                f"got {len(alpha_res_per_scale)}")
            self.register_buffer(
                "alpha_res_vec",
                torch.tensor(alpha_res_per_scale, dtype=torch.float32),
            )
        else:
            self.alpha_res_vec = None
        d_in = d_model + d_scale_feat + d_model
        if shared_residual:
            self.residual_shared = nn.Sequential(
                nn.Linear(d_in, residual_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(residual_hidden, n_cells),
            )
        else:
            self.scale_residuals = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_in, residual_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(residual_hidden, n_cells),
                )
                for _ in range(n_scales)
            ])

    def forward(
        self,
        h: torch.Tensor,                  # (B, T, d_model)
        scale_feat: torch.Tensor,         # (B, T, d_scale_feat)
        prev_cells: torch.Tensor,         # (B, T) cell indices, may include special tokens
        pred_scales: torch.Tensor,        # (B, T) scale indices
        prev_cell_emb: torch.Tensor,      # (B, T, d_model)
        log_dist: torch.Tensor,           # (n_cells, n_cells) buffer
        log_cell_density: torch.Tensor,   # (n_scales, n_cells, n_cells) buffer
        gmm_mu: torch.Tensor,             # (n_scales,) buffer
        gmm_var: torch.Tensor,            # (n_scales,) buffer
        return_residual: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns logits, or (logits, residual_logits, scale_ids) when return_residual=True."""
        B, T = prev_cells.shape
        # Map special-token prev to cell 0 for indexing (will mask geo below)
        prev_safe = prev_cells.clamp(max=self.n_cells - 1).clamp(min=0)
        scale_safe = pred_scales.clamp(max=self.n_scales - 1).clamp(min=0)

        prev_log_dist = log_dist[prev_safe]  # (B, T, n_cells)
        mu_s = gmm_mu[scale_safe].unsqueeze(-1)              # (B, T, 1)
        var_s = gmm_var[scale_safe].clamp(min=0.01).unsqueeze(-1)
        geo_gaussian = -0.5 * (prev_log_dist - mu_s) ** 2 / var_s

        # Jacobian: subtract log cell density at that (scale, prev_cell, ·)
        jac = log_cell_density[scale_safe, prev_safe]        # (B, T, n_cells)
        geo_logits = geo_gaussian - jac

        # Zero out geometric term for special-token prev (no valid geometry)
        is_special = (prev_cells >= self.n_cells).unsqueeze(-1)
        geo_logits = geo_logits.masked_fill(is_special, 0.0)

        logits = self.alpha_dist * geo_logits

        # Residual contribution: deterministic per-scale path.
        h_in = torch.cat([h, scale_feat, prev_cell_emb], dim=-1)
        residual_out = None

        if self.alpha_res_vec is not None or self.alpha_res > 0:
            if self.shared_residual:
                residual_out = self.residual_shared(h_in)
            else:
                residual_out = torch.zeros(B, T, self.n_cells, device=h.device, dtype=h_in.dtype)
                for s in range(self.n_scales):
                    mask_s = (scale_safe == s)
                    if not mask_s.any():
                        continue
                    residual_out[mask_s] = self.scale_residuals[s](h_in[mask_s]).to(residual_out.dtype)
            if self.alpha_res_vec is not None:
                per_step_alpha = self.alpha_res_vec[scale_safe].unsqueeze(-1)
                logits = logits + per_step_alpha * residual_out
            else:
                logits = logits + self.alpha_res * residual_out

        if return_residual and residual_out is not None:
            return logits, residual_out, scale_safe
        return logits


class ScaleFirstCascadeHead(nn.Module):
    """Cascaded 4-head: scale → cell → duration → arrival."""

    def __init__(self, d_model: int, n_cells: int, n_scales: int, dropout: float = 0.1,
                 alpha_marginal: float = 1.0, alpha_h: float = 1.0,
                 alpha_dist: float = 2.0, alpha_res: float = 0.5,
                 alpha_res_per_scale: list[float] | None = None,
                 shared_residual: bool = False):
        super().__init__()
        self.n_cells = n_cells
        self.n_scales = n_scales
        self.ss_prob = 0.0
        self.d_model = d_model

        d_scale_feat = 32
        self.scale_head = ScaleSelectionHead(
            d_model, n_scales,
            alpha_marginal=alpha_marginal, alpha_h=alpha_h,
            residual_hidden=d_scale_feat, dropout=dropout,
        )
        self.scale_compress = nn.Linear(n_scales, d_scale_feat)

        self.cell_head = ScaleAwareCellSelectionHead(
            d_model, n_cells, n_scales,
            alpha_dist=alpha_dist, alpha_res=alpha_res,
            alpha_res_per_scale=alpha_res_per_scale,
            d_scale_feat=d_scale_feat, residual_hidden=32, dropout=dropout,
            shared_residual=shared_residual,
        )

        d_cell_feat = 64
        self.cell_compress = nn.Linear(n_cells, d_cell_feat)

        d_dur_in = d_model + d_cell_feat + d_scale_feat
        self.head_duration_cls = nn.Sequential(
            nn.Linear(d_dur_in, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, N_DURATION_BINS),
        )
        self.head_duration_mu = nn.Linear(d_dur_in, 1)
        self.head_duration_logsig = nn.Linear(d_dur_in, 1)

        d_arr_in = d_model + d_cell_feat + d_scale_feat + 2
        self.head_arrival = nn.Sequential(
            nn.Linear(d_arr_in, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, N_ARRIVAL_BINS),
        )

    def forward(
        self,
        h: torch.Tensor,
        prev_cells: torch.Tensor,
        marginal_log: torch.Tensor,
        log_dist: torch.Tensor,
        log_cell_density: torch.Tensor,
        gmm_mu: torch.Tensor,
        gmm_var: torch.Tensor,
        scale_gt: torch.Tensor | None = None,
        cell_gt: torch.Tensor | None = None,
        prev_cell_emb: torch.Tensor | None = None,
        return_residual: bool = False,
    ) -> dict[str, torch.Tensor]:
        use_pred = self.training and random.random() < self.ss_prob

        logits_scale = self.scale_head(h, marginal_log)
        if scale_gt is not None and not use_pred:
            pred_scales = scale_gt.clamp(max=self.n_scales - 1)
        else:
            pred_scales = logits_scale.argmax(-1).clamp(max=self.n_scales - 1)
        scale_oh = F.one_hot(pred_scales, self.n_scales).float()
        scale_feat = self.scale_compress(scale_oh)

        if prev_cell_emb is None:
            prev_cell_emb = torch.zeros_like(h)
        cell_head_out = self.cell_head(
            h, scale_feat, prev_cells, pred_scales, prev_cell_emb,
            log_dist, log_cell_density, gmm_mu, gmm_var,
            return_residual=return_residual,
        )
        if return_residual and isinstance(cell_head_out, tuple):
            logits_cell, residual_logits, residual_scales = cell_head_out
        else:
            logits_cell = cell_head_out
            residual_logits = residual_scales = None

        if cell_gt is not None and not use_pred:
            cell_oh = F.one_hot(cell_gt.clamp(max=self.n_cells - 1), self.n_cells).float()
        else:
            cell_oh = F.one_hot(logits_cell.argmax(-1).clamp(max=self.n_cells - 1), self.n_cells).float()
        cell_feat = self.cell_compress(cell_oh)

        h_dur = torch.cat([h, cell_feat, scale_feat], dim=-1)
        logits_dur = self.head_duration_cls(h_dur)
        dur_mu = self.head_duration_mu(h_dur)
        dur_logsig = self.head_duration_logsig(h_dur)

        dur_feat = torch.cat([dur_mu, dur_logsig], dim=-1)
        h_arr = torch.cat([h, cell_feat, scale_feat, dur_feat], dim=-1)
        logits_arrival = self.head_arrival(h_arr)

        out = {
            "cell": logits_cell,
            "scale": logits_scale,
            "duration": logits_dur,
            "dur_mu": dur_mu.squeeze(-1),
            "dur_logsig": dur_logsig.squeeze(-1),
            "arrival": logits_arrival,
        }
        if residual_logits is not None:
            out["_residual_logits"] = residual_logits
            out["_residual_scales"] = residual_scales
        return out


class SpatialFourierEncoding(nn.Module):
    """Fourier encoding for continuous (lon, lat) coordinates.

    Provides geometric inductive bias: nearby cells get similar embeddings.
    """
    def __init__(self, d_model: int, n_freqs: int = 32):
        super().__init__()
        self.register_buffer(
            "freqs",
            torch.exp(torch.linspace(math.log(1.0), math.log(500.0), n_freqs)),
        )
        self.proj = nn.Sequential(
            nn.Linear(n_freqs * 4 + 2, d_model),
            nn.GELU(),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (..., 2) with [lon, lat] in degrees."""
        lon_rad = coords[..., 0:1] * (math.pi / 180.0)
        lat_rad = coords[..., 1:2] * (math.pi / 180.0)
        angles_lon = lon_rad * self.freqs
        angles_lat = lat_rad * self.freqs
        fourier = torch.cat([
            lon_rad, lat_rad,
            angles_lon.sin(), angles_lon.cos(),
            angles_lat.sin(), angles_lat.cos(),
        ], dim=-1)
        return self.proj(fourier)


class ScaleTraj(nn.Module):
    """Scale-first cascade Transformer: predicts (scale, cell, duration, arrival)."""

    def __init__(
        self,
        n_cells: int,
        n_scales: int = 5,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.2,
        max_seq_len: int = 128,
        cell_coords: list[tuple[float, float]] | None = None,
        n_train_users: int = 0,
        n_latent: int = N_LATENT,
        gmm_means_log: list[float] | None = None,
        gmm_variances_log: list[float] | None = None,
        empirical_scale_freq: list[float] | None = None,
        alpha_marginal: float = 1.0, alpha_h: float = 1.0,
        alpha_dist: float = 2.0, alpha_res: float = 0.5,
        alpha_res_per_scale: list[float] | None = None,
        disable_jacobian: bool = False,
        shared_residual: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_cells = n_cells
        self.n_scales = n_scales
        self.n_latent = n_latent
        self.n_train_users = n_train_users
        self.alpha_dist = alpha_dist
        self.alpha_res = alpha_res

        self.emb_cell = nn.Embedding(n_cells + N_SPECIAL, d_model)
        self.spatial_fourier = SpatialFourierEncoding(d_model)
        self.emb_scale = nn.Embedding(n_scales + N_SPECIAL, d_model)
        self.fourier_duration = FourierTimeEncoding(d_model)
        self.fourier_arrival = FourierTimeEncoding(d_model)

        if cell_coords is not None:
            special_coords = [(0.0, 0.0)] * N_SPECIAL
            all_coords = list(cell_coords) + special_coords
            self.register_buffer(
                "_cell_coords",
                torch.tensor(all_coords, dtype=torch.float32),
            )
        else:
            self.register_buffer(
                "_cell_coords",
                torch.zeros(n_cells + N_SPECIAL, 2),
            )

        edges = DURATION_BIN_EDGES_MIN
        centers = []
        for i in range(N_DURATION_BINS):
            lo, hi = edges[i], edges[min(i + 1, len(edges) - 1)]
            centers.append((lo + hi) / 2.0)
        self.register_buffer("_dur_bin_centers", torch.tensor(centers, dtype=torch.float32))

        # Scale-aware mixture priors as buffers (not trainable)
        if gmm_means_log is not None and gmm_variances_log is not None:
            gmm_mu_t = torch.tensor(gmm_means_log, dtype=torch.float32)
            gmm_var_t = torch.tensor(gmm_variances_log, dtype=torch.float32).clamp(min=0.01)
        else:
            gmm_mu_t = torch.zeros(n_scales, dtype=torch.float32)
            gmm_var_t = torch.ones(n_scales, dtype=torch.float32)
        self.register_buffer("gmm_mu", gmm_mu_t)
        self.register_buffer("gmm_var", gmm_var_t)

        if empirical_scale_freq is not None:
            freq_t = torch.tensor(empirical_scale_freq, dtype=torch.float32).clamp(min=1e-6)
            marginal_log_t = freq_t.log()
        else:
            marginal_log_t = torch.zeros(n_scales, dtype=torch.float32)
        self.register_buffer("marginal_log", marginal_log_t)

        if torch.cuda.is_available():
            _kde_device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            _kde_device = torch.device("mps")
        else:
            _kde_device = torch.device("cpu")
        if cell_coords is not None:
            cell_coords_real = torch.tensor(cell_coords, dtype=torch.float32, device=_kde_device)
            log_dist_t = compute_log_dist_matrix(cell_coords_real)
        else:
            log_dist_t = torch.zeros(n_cells, n_cells, dtype=torch.float32, device=_kde_device)
        # persistent=False: these buffers are derivable from cell_coords + gmm_var
        # (both already in checkpoint config), so we recompute at init instead of
        # bloating the .pt file with ~200 MB of derived data.
        self.register_buffer("log_dist", log_dist_t, persistent=False)

        if disable_jacobian:
            log_cell_density_t = torch.zeros(n_scales, n_cells, n_cells, dtype=torch.float32, device=_kde_device)
        else:
            sigmas_dev = gmm_var_t.to(_kde_device).sqrt()
            log_cell_density_t = compute_per_scale_log_cell_density(log_dist_t, sigmas_dev)
        self.register_buffer("log_cell_density", log_cell_density_t, persistent=False)

        self.emb_day_type = nn.Embedding(N_DAY_TYPES, d_model)
        self.emb_persona = nn.Embedding(N_PERSONAS, d_model)
        self.emb_budget = nn.Linear(1, d_model)
        self.fourier_gap = FourierTimeEncoding(d_model)
        self.fourier_cumtime = FourierTimeEncoding(d_model)
        self.anchor_proj = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
        )
        self.hist_type_emb = nn.Parameter(torch.randn(d_model) * 0.02)

        if n_train_users > 0:
            self.user_codes = nn.Embedding(n_train_users, n_latent)
            nn.init.normal_(self.user_codes.weight, std=0.1)
        else:
            self.user_codes = None
        self.mu_persona = nn.Parameter(torch.zeros(N_PERSONAS, n_latent))
        self.logvar_persona = nn.Parameter(torch.zeros(N_PERSONAS, n_latent))
        self.z_proj = nn.Sequential(
            nn.Linear(n_latent, d_model),
            nn.GELU(),
        )

        self.input_proj = nn.Sequential(
            nn.Linear(d_model * 10, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.input_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = ScaleFirstCascadeHead(
            d_model, n_cells, n_scales, dropout=dropout,
            alpha_marginal=alpha_marginal, alpha_h=alpha_h,
            alpha_dist=alpha_dist, alpha_res=alpha_res,
            alpha_res_per_scale=alpha_res_per_scale,
            shared_residual=shared_residual,
        )
        self.head_eos = nn.Linear(d_model, 1)

        self.head_activity = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, N_ACTIVITY),
        )
        self.head_daily = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, N_DAILY_STATS),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=0.5)

    @torch.no_grad()
    def sample_persona_latent(self, persona_id: int, device: torch.device | None = None) -> torch.Tensor:
        """Sample z ~ N(mu_persona, sigma_persona) for a chosen persona."""
        if device is None:
            device = next(self.parameters()).device
        mu = self.mu_persona[persona_id].to(device)
        logvar = self.logvar_persona[persona_id].to(device)
        sigma = (0.5 * logvar).exp()
        return mu + sigma * torch.randn_like(mu)

    def _estimate_budget(self, arrivals: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        arr_min = arrivals.float() * (1440.0 / N_ARRIVAL_BINS)
        dur_idx = durations.clamp(max=N_DURATION_BINS - 1)
        edges = torch.tensor(
            DURATION_BIN_EDGES_MIN[:N_DURATION_BINS],
            device=durations.device, dtype=torch.float32,
        )
        dur_min = edges[dur_idx]
        used = arr_min + dur_min
        return (1440.0 - used).clamp(min=0.0) / 1440.0

    def _embed_tokens(
        self,
        cells: torch.Tensor,
        scales: torch.Tensor,
        durations: torch.Tensor,
        arrivals: torch.Tensor,
        day_types: torch.Tensor,
        persona_ids: torch.Tensor,
        home_cells: torch.Tensor | None = None,
        work_cells: torch.Tensor | None = None,
        user_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T = scales.shape
        budget = self._estimate_budget(arrivals, durations)

        arr_cont = arrivals.clamp(0, N_ARRIVAL_BINS - 1).float() / N_ARRIVAL_BINS
        dur_idx = durations.clamp(max=N_DURATION_BINS - 1)
        dur_cont = torch.log1p(self._dur_bin_centers[dur_idx]) / math.log(1441.0)

        arr_min = arrivals.clamp(0, N_ARRIVAL_BINS - 1).float() * (1440.0 / N_ARRIVAL_BINS)
        dur_min_vals = self._dur_bin_centers[dur_idx]
        dep_min = arr_min + dur_min_vals
        gap_min = torch.zeros(B, T, device=scales.device, dtype=arr_min.dtype)
        if T > 1:
            gap_min[:, 1:] = (arr_min[:, 1:] - dep_min[:, :-1]).clamp(min=0)
        gap_cont = gap_min / 1440.0
        cumtime_cont = dep_min / 1440.0

        cell_emb = self.emb_cell(cells) + self.spatial_fourier(self._cell_coords[cells])
        time_emb = self.fourier_cumtime(cumtime_cont)

        if home_cells is not None and work_cells is not None:
            home_emb = self.emb_cell(home_cells).unsqueeze(1).expand(B, T, -1)
            work_emb = self.emb_cell(work_cells).unsqueeze(1).expand(B, T, -1)
            anchor_cond = self.anchor_proj(torch.cat([home_emb, work_emb, time_emb], dim=-1))
        else:
            anchor_cond = torch.zeros(B, T, self.d_model, device=cells.device)

        if user_latent is not None:
            z_emb = self.z_proj(user_latent).unsqueeze(1).expand(B, T, -1)
            anchor_cond = anchor_cond + z_emb

        e = torch.cat([
            cell_emb,
            self.emb_scale(scales),
            self.fourier_duration(dur_cont),
            self.fourier_arrival(arr_cont),
            time_emb,
            anchor_cond,
            self.emb_day_type(day_types).unsqueeze(1).expand(B, T, -1),
            self.emb_persona(persona_ids).unsqueeze(1).expand(B, T, -1),
            self.emb_budget(budget.unsqueeze(-1)),
            self.fourier_gap(gap_cont),
        ], dim=-1)
        return self.input_proj(e)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        cells: torch.Tensor,
        scales: torch.Tensor,
        durations: torch.Tensor,
        arrivals: torch.Tensor,
        day_types: torch.Tensor,
        persona_ids: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        target_cells: torch.Tensor | None = None,
        target_scales: torch.Tensor | None = None,
        hist_cells: torch.Tensor | None = None,
        hist_scales: torch.Tensor | None = None,
        hist_durations: torch.Tensor | None = None,
        hist_arrivals: torch.Tensor | None = None,
        hist_day_counts: torch.Tensor | None = None,
        hist_day_boundaries: torch.Tensor | None = None,
        home_cells: torch.Tensor | None = None,
        work_cells: torch.Tensor | None = None,
        user_latent: torch.Tensor | None = None,
        return_residual: bool = False,
    ) -> dict[str, torch.Tensor]:
        B, T = scales.shape
        x = self._embed_tokens(cells, scales, durations, arrivals, day_types, persona_ids,
                               home_cells=home_cells, work_cells=work_cells,
                               user_latent=user_latent)

        hist_x, hist_pad = self._embed_history(
            B, x.device,
            hist_cells, hist_scales, hist_durations, hist_arrivals,
            hist_day_counts, hist_day_boundaries,
            day_types, persona_ids,
            home_cells, work_cells,
            user_latent=user_latent,
        )
        K = hist_x.size(1)

        full_x = torch.cat([hist_x, x], dim=1)
        full_x = self.input_norm(full_x)
        full_x = _apply_rope(full_x)

        T_full = K + T
        causal = self._causal_mask(T_full, scales.device)
        if pad_mask is not None:
            full_pad_mask = torch.cat([hist_pad, pad_mask], dim=1)
        else:
            full_pad_mask = torch.cat([hist_pad, torch.zeros(B, T, dtype=torch.bool, device=hist_pad.device)], dim=1)
        h = self.transformer(full_x, mask=causal, src_key_padding_mask=full_pad_mask)

        h = h[:, K:, :]
        prev_cell_emb = self.emb_cell(cells) + self.spatial_fourier(self._cell_coords[cells])

        logits = self.head(
            h, cells,
            self.marginal_log, self.log_dist, self.log_cell_density,
            self.gmm_mu, self.gmm_var,
            scale_gt=target_scales, cell_gt=target_cells,
            prev_cell_emb=prev_cell_emb,
            return_residual=return_residual,
        )
        logits["eos"] = self.head_eos(h).squeeze(-1)
        logits["activity"] = self.head_activity(h)

        if pad_mask is not None:
            valid = (~pad_mask).float().unsqueeze(-1)
            day_pool = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            day_pool = h.mean(dim=1)
        logits["daily_stats"] = self.head_daily(day_pool)

        return logits

    def _embed_history(
        self,
        B: int,
        device: torch.device,
        hist_cells: torch.Tensor | None,
        hist_scales: torch.Tensor | None,
        hist_durations: torch.Tensor | None,
        hist_arrivals: torch.Tensor | None,
        hist_day_counts: torch.Tensor | None,
        hist_day_boundaries: torch.Tensor | None,
        day_types: torch.Tensor | None,
        persona_ids: torch.Tensor | None,
        home_cells: torch.Tensor | None,
        work_cells: torch.Tensor | None,
        user_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed history tokens visit-level for direct attention prepend.

        Returns (hist_x, hist_pad_mask). When no history is available, returns
        zero-length tensors so downstream concat is a no-op.
        """
        if hist_cells is None or hist_cells.numel() == 0:
            return (
                torch.zeros(B, 0, self.d_model, device=device),
                torch.zeros(B, 0, dtype=torch.bool, device=device),
            )

        K = hist_cells.size(1)
        dt = day_types if day_types is not None else torch.zeros(B, dtype=torch.long, device=device)
        pi = persona_ids if persona_ids is not None else torch.zeros(B, dtype=torch.long, device=device)

        hist_x = self._embed_tokens(
            hist_cells, hist_scales, hist_durations, hist_arrivals, dt, pi,
            home_cells=home_cells, work_cells=work_cells,
            user_latent=user_latent,
        )
        hist_x = hist_x + self.hist_type_emb.view(1, 1, -1)

        if hist_day_counts is not None and hist_day_boundaries is not None:
            day_idx = (hist_day_counts - 1).clamp(min=0).unsqueeze(1)
            last_end = torch.gather(hist_day_boundaries[..., 1], 1, day_idx).squeeze(1)
            valid_lens = torch.where(hist_day_counts > 0, last_end, torch.zeros_like(last_end))
            valid_lens = valid_lens.clamp(min=1)
            positions = torch.arange(K, device=device).unsqueeze(0)
            hist_pad = positions >= valid_lens.unsqueeze(1)
        else:
            hist_pad = torch.zeros(B, K, dtype=torch.bool, device=device)

        return hist_x, hist_pad

    @torch.no_grad()
    def generate(
        self,
        persona_id: int,
        day_type: int,
        n_stops: int | None = None,
        max_len: int = 30,
        eos_threshold: float = 0.5,
        min_stops: int = 0,
        temperature: float = 0.9,
        top_k: int = 0,
        device: torch.device | None = None,
        context_tokens: list[tuple[int, int, int, int]] | None = None,
        cell_coords: list[tuple[float, float]] | None = None,
        scale_gmm_params: dict | None = None,
        cell_freq_log: torch.Tensor | None = None,
        freq_alpha: float = 0.0,
        return_alpha: float = 0.3,
        arrival_mask: str = "hard",
        hist_tokens: list[tuple[int, int, int, int]] | None = None,
        hist_day_boundaries: list[tuple[int, int]] | None = None,
        cell_temperature: float | None = None,
        home_cell: int | None = None,
        work_cell: int | None = None,
        user_latent: torch.Tensor | None = None,
    ) -> list[tuple[int, int, int, int]]:
        """Autoregressive generation with temporal consistency constraints.

        arrival_mask: "hard" (mask bins < current_time), "soft" (penalize), "none"
        hist_tokens: list of (cell, scale, dur, arr) from previous days (visit-level)
        hist_day_boundaries: list of (start, end) token index pairs per past day
        """
        if device is None:
            device = next(self.parameters()).device
        self.eval()

        t_temp = temperature
        target_len = max_len

        bos_cell = self.n_cells + SPECIAL_TOKENS["[BOS]"]
        bos_scale = self.n_scales + SPECIAL_TOKENS["[BOS]"]

        cell_seq = [bos_cell]
        scale_seq = [bos_scale]
        dur_seq = [0]
        arr_seq = [0]

        if context_tokens:
            for c, s, d, a in context_tokens:
                cell_seq.append(c)
                scale_seq.append(s)
                dur_seq.append(d)
                arr_seq.append(a)

        persona_t = torch.tensor([persona_id], device=device)
        dt_t = torch.tensor([day_type], device=device)

        pad_cell = self.n_cells + SPECIAL_TOKENS["[PAD]"]
        home_t = torch.tensor([home_cell if home_cell is not None else pad_cell], device=device)
        work_t = torch.tensor([work_cell if work_cell is not None else pad_cell], device=device)

        if user_latent is None:
            ul = torch.zeros(1, self.n_latent, device=device)
        elif user_latent.dim() == 1:
            ul = user_latent.unsqueeze(0).to(device)
        else:
            ul = user_latent.to(device)

        if hist_tokens:
            hc = torch.tensor([[t[0] for t in hist_tokens]], device=device, dtype=torch.long)
            hs = torch.tensor([[t[1] for t in hist_tokens]], device=device, dtype=torch.long)
            hd = torch.tensor([[t[2] for t in hist_tokens]], device=device, dtype=torch.long)
            ha = torch.tensor([[t[3] for t in hist_tokens]], device=device, dtype=torch.long)
            if hist_day_boundaries:
                hb = torch.tensor([hist_day_boundaries], device=device, dtype=torch.long)
                hdc = torch.tensor([len(hist_day_boundaries)], device=device, dtype=torch.long)
            else:
                hb = torch.tensor([[[0, len(hist_tokens)]]], device=device, dtype=torch.long)
                hdc = torch.tensor([1], device=device, dtype=torch.long)
            hist_x, hist_pad = self._embed_history(
                1, device, hc, hs, hd, ha, hdc, hb,
                dt_t, persona_t, home_t, work_t,
                user_latent=ul,
            )
        else:
            hist_x = torch.zeros(1, 0, self.d_model, device=device)
            hist_pad = torch.zeros(1, 0, dtype=torch.bool, device=device)
        K = hist_x.size(1)

        if cell_coords is not None:
            torch.tensor(
                [[math.radians(lat), math.radians(lon)] for lon, lat in cell_coords],
                device=device, dtype=torch.float32,
            )

        generated: list[tuple[int, int, int, int]] = []
        current_time_min = 0.0
        visit_counts: dict[int, int] = {}

        for step in range(target_len):
            if current_time_min >= 1420.0:
                break

            T = len(scale_seq)
            cell_t = torch.tensor([cell_seq], device=device)
            sc_t = torch.tensor([scale_seq], device=device)
            dur_t = torch.tensor([dur_seq], device=device)
            arr_t = torch.tensor([arr_seq], device=device)

            x = self._embed_tokens(cell_t, sc_t, dur_t, arr_t, dt_t, persona_t,
                                   home_cells=home_t, work_cells=work_t,
                                   user_latent=ul)
            full_x = torch.cat([hist_x, x], dim=1)
            full_x = self.input_norm(full_x)
            full_x = _apply_rope(full_x)
            T_full = K + T
            causal = self._causal_mask(T_full, device)
            full_pad = torch.cat([hist_pad, torch.zeros(1, T, dtype=torch.bool, device=device)], dim=1)
            h = self.transformer(full_x, mask=causal, src_key_padding_mask=full_pad)

            last_h = h[:, -1:, :]
            prev_cell_t = cell_t[:, -1:]
            prev_cell_emb = self.emb_cell(prev_cell_t) + self.spatial_fourier(self._cell_coords[prev_cell_t])

            eos_logit = self.head_eos(last_h[:, 0, :]).squeeze(-1)
            # Step counts the number of generated stays so far (excluding the
            # context). Enforce min_stops as a soft floor: don't terminate until
            # we have at least min_stops stays.
            if step > 0 and step >= min_stops and \
               torch.sigmoid(eos_logit).item() > eos_threshold:
                break

            # Scale-aware decoding: sample scale FIRST, then condition cell on it
            scale_logits_raw = self.head.scale_head(last_h, self.marginal_log)[:, 0, :self.n_scales].clone()
            scale_id = _sample(scale_logits_raw, temperature, top_k)
            # Build full forward through cascade with scale_gt = sampled scale_id
            scale_gt_t = torch.tensor([[scale_id]], device=device, dtype=torch.long)

            logits = self.head(
                last_h, prev_cell_t,
                self.marginal_log, self.log_dist, self.log_cell_density,
                self.gmm_mu, self.gmm_var,
                scale_gt=scale_gt_t,
                prev_cell_emb=prev_cell_emb,
            )

            cell_logits = logits["cell"][:, 0, :self.n_cells].clone()

            if cell_freq_log is not None and freq_alpha > 0:
                cell_logits += freq_alpha * cell_freq_log.to(device)

            if return_alpha > 0 and visit_counts:
                for vc, cnt in visit_counts.items():
                    if 0 <= vc < self.n_cells:
                        cell_logits[0, vc] += return_alpha * math.log(1 + cnt)

            c_temp = cell_temperature if cell_temperature is not None else temperature
            cell_id = _sample(cell_logits.squeeze(0), c_temp, top_k)
            visit_counts[cell_id] = visit_counts.get(cell_id, 0) + 1

            dur_logits = logits["duration"][:, 0, :N_DURATION_BINS].clone()
            remaining_min = 1440.0 - current_time_min
            max_dur_bin = N_DURATION_BINS - 1
            for b in range(N_DURATION_BINS):
                if DURATION_BIN_EDGES_MIN[b] > remaining_min:
                    max_dur_bin = max(b - 1, 0)
                    break
            if max_dur_bin < N_DURATION_BINS - 1:
                dur_logits[:, max_dur_bin + 1:] = -float("inf")
            dur_id = _sample(dur_logits, t_temp, top_k)

            arr_logits = logits["arrival"][:, 0, :N_ARRIVAL_BINS].clone()
            current_bin = max(0, int(current_time_min / 5.0))
            if arrival_mask == "hard" and current_bin > 0:
                if current_bin >= N_ARRIVAL_BINS:
                    break
                arr_logits[:, :current_bin] = -float("inf")
            elif arrival_mask == "soft" and current_bin > 0:
                bins = torch.arange(N_ARRIVAL_BINS, device=device, dtype=torch.float32)
                arr_logits[0] -= 0.5 * (current_bin - bins).clamp(min=0)
            arr_id = _sample(arr_logits, t_temp, top_k)

            arr_min_val = arr_id * 5.0 + 2.5
            dur_lo = DURATION_BIN_EDGES_MIN[min(dur_id, N_DURATION_BINS - 1)]
            dur_hi = DURATION_BIN_EDGES_MIN[min(dur_id + 1, len(DURATION_BIN_EDGES_MIN) - 1)]
            current_time_min = arr_min_val + max((dur_lo + dur_hi) / 2.0, 5.0)

            generated.append((cell_id, scale_id, dur_id, arr_id))
            cell_seq.append(cell_id)
            scale_seq.append(scale_id)
            dur_seq.append(dur_id)
            arr_seq.append(arr_id)

        return generated


def _sample(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    logits = logits / max(temperature, 1e-8)
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[..., -1:]] = -float("inf")
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()
