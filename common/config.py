"""System (Table I) and experiment configuration, presets and stable seeding.

All defaults equal the source paper. Do not change them here; new regimes
must go through CLI options whose default is the paper value (claude.md 1.2).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

SPEED_OF_LIGHT = 299_792_458.0


def dbm_to_w(p_dbm: float) -> float:
    """Convert dBm to watts."""
    return 10.0 ** ((p_dbm - 30.0) / 10.0)


def w_to_dbm(p_w: float) -> float:
    return 10.0 * math.log10(p_w) + 30.0


# ---------------------------------------------------------------------------
# Stable seeding (claude.md 5.8): never use Python's hash().
# ---------------------------------------------------------------------------
def stable_seed(*parts: Any) -> int:
    """Deterministic 63-bit seed from a tuple of str/int/float parts (sha256)."""
    text = "|".join(repr(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def make_rng(*parts: Any) -> np.random.Generator:
    return np.random.default_rng(stable_seed(*parts))


# ---------------------------------------------------------------------------
# Table I
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SysConfig:
    # array / carrier
    N: int = 32
    fc_hz: float = 3.5e9
    de_over_lambda: float = 0.5
    # fading / hardware
    K_R: float = 1.0
    kappa: float = 0.1
    # path loss
    alpha_ar: float = 2.2
    alpha_ru: float = 2.3
    d0: float = 1.0
    beta0: float = 1.0
    d_min: float = 0.1
    # noise (dBm). sigma_w2 is not in Table I; it cancels in DEP and is only a feature.
    sigma_r2_dbm: float = -70.0
    sigma_u2_dbm: float = -70.0
    sigma_w2_dbm: float = -70.0
    # RIS circuit power per element (mW)
    P_S_mw: float = 0.1
    P_DC_mw: float = 0.3162
    # operating ranges (LHS)
    P_tot_dbm_range: tuple[float, float] = (30.0, 40.0)
    xi_req_range: tuple[float, float] = (0.75, 0.95)
    # requirements
    R_qos: float = 0.2
    R_dec_pub: float = 0.1
    R_dec_bob: float = 0.1
    p_max: float = 0.1
    # allocation domain
    mb_min: float = 0.05
    mb_max: float = 0.49
    s_min: float = 0.05
    s_max: float = 0.95
    # search margins
    xi_margin: float = 0.03
    rate_margin: float = 0.01
    # geometry (m)
    pos_alice: tuple[float, float] = (0.0, 0.0)
    pos_ris: tuple[float, float] = (95.0, 5.0)
    u_r: tuple[float, float] = (0.0, 1.0)
    region_bob: tuple[tuple[float, float], tuple[float, float]] = ((105.0, 115.0), (-5.0, 5.0))
    region_willie: tuple[tuple[float, float], tuple[float, float]] = ((96.0, 104.0), (-4.0, 4.0))
    region_pub: tuple[tuple[float, float], tuple[float, float]] = ((102.0, 122.0), (-8.0, 12.0))
    # assessment error allocation (Appendix A-C)
    J_family: int = 288
    G_max: int = 280
    alpha_tot: float = 0.05
    # Willie CV grouping threshold
    cv_threshold: float = 0.9

    # ---- derived quantities -------------------------------------------------
    @property
    def lam(self) -> float:
        return SPEED_OF_LIGHT / self.fc_hz

    @property
    def de(self) -> float:
        return self.de_over_lambda * self.lam

    @property
    def sigma_r2(self) -> float:
        return dbm_to_w(self.sigma_r2_dbm)

    @property
    def sigma_u2(self) -> float:
        return dbm_to_w(self.sigma_u2_dbm)

    @property
    def sigma_w2(self) -> float:
        return dbm_to_w(self.sigma_w2_dbm)

    @property
    def P_S(self) -> float:
        return self.P_S_mw * 1e-3

    @property
    def P_DC(self) -> float:
        return self.P_DC_mw * 1e-3

    @property
    def Gamma_pub(self) -> float:
        return 2.0 ** self.R_dec_pub - 1.0

    @property
    def Gamma_bob(self) -> float:
        return 2.0 ** self.R_dec_bob - 1.0

    @property
    def alpha_g(self) -> float:
        """Per-group error allowance alpha_tot / (2 G_max) (eq. 36)."""
        return self.alpha_tot / (2.0 * self.G_max)

    def P0(self, P_tot_dbm: float) -> float:
        """Available power after RIS circuit costs (eq. 5), in W."""
        return dbm_to_w(P_tot_dbm) - self.N * (self.P_S + self.P_DC)

    def mb_ceiling(self, K: int) -> float:
        """Necessary bound (19): m_b <= (1+k^2) 2^{-K R_qos} - k^2."""
        k2 = self.kappa ** 2
        return (1.0 + k2) * 2.0 ** (-K * self.R_qos) - k2

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Experiment configuration and presets (claude.md 6.2)
# ---------------------------------------------------------------------------
@dataclass
class ExpConfig:
    preset: str = "quick"
    n_train_layouts: int = 6
    n_val_layouts: int = 3
    n_test_layouts: int = 4
    cases_per_layout: int = 4
    replicates: int = 2
    search_samples: int = 4000
    assess_samples: int = 20000
    label_samples: int = 20000
    k_train: tuple[int, ...] = (2, 3, 4, 5)
    k_test: tuple[int, ...] = (2, 3, 4, 5, 6)
    swarm: int = 30
    budget_full: int = 1800
    budget_short: int | None = None  # None -> read from selected_budget.json
    budgets_val: tuple[int, ...] = (150, 300, 450)
    archive_size: int = 5
    label_runs: int = 2
    epochs: int = 200
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    train_seeds: tuple[int, ...] = (7, 17, 27)
    pso_seeds: tuple[int, ...] = (101, 202, 303)
    delta_adjust: float = 0.005
    bootstrap_n: int = 2000
    bootstrap_seed: int = 7317
    seed: int = 2026
    workers: int = 1
    device: str = "auto"      # training device: auto|cpu|cuda (inference for timing is CPU)
    # QPSO (only used by Quantummodel_code)
    beta_start: float = 1.0
    beta_end: float = 0.5
    boundary: str = "reflect"
    anchor_w0: float = 0.5
    anchor_frac: float = 0.3
    anchor_gate: bool = True
    extra: dict = field(default_factory=dict)

    @property
    def K_max(self) -> int:
        return max(max(self.k_train), max(self.k_test))

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: dict[str, dict] = {
    "quick": dict(n_train_layouts=6, n_val_layouts=3, n_test_layouts=4, cases_per_layout=4,
                  replicates=2, search_samples=4000, assess_samples=20000, label_samples=20000),
    "medium": dict(n_train_layouts=16, n_val_layouts=8, n_test_layouts=20, cases_per_layout=8,
                   replicates=3, search_samples=10000, assess_samples=100000, label_samples=100000),
    "paper": dict(n_train_layouts=32, n_val_layouts=16, n_test_layouts=40, cases_per_layout=8,
                  replicates=3, search_samples=10000, assess_samples=100000, label_samples=100000),
}


def exp_from_preset(preset: str, **overrides: Any) -> ExpConfig:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    cfg = ExpConfig(preset=preset, **PRESETS[preset])
    clean = {k: v for k, v in overrides.items() if v is not None}
    cfg = replace(cfg, **clean)
    assert cfg.replicates <= len(cfg.train_seeds) and cfg.replicates <= len(cfg.pso_seeds), \
        "replicates must be <= number of paired (train, PSO) seeds"
    return cfg


SPLIT_ID_BASE = {"train": 1000, "val": 2000, "test": 4000}
