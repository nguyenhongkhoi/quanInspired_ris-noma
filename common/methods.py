"""Method registry (claude.md 7). Keys are used verbatim in CSV files and tables."""
from __future__ import annotations

import importlib
from dataclasses import dataclass

DISPLAY = {
    "full_pso": "Full PSO",
    "rand_pso": "Random short PSO",
    "direct_mlp": "Direct pooled MLP",
    "direct_gnn": "Direct GNN",
    "mlp_pso": "MLP + short PSO",
    "gnn_pso": "GNN + short PSO",
    "full_qpso": "Full QPSO",
    "rand_qpso": "Random short QPSO",
    "mlp_qpso": "MLP + short QPSO",
    "gnn_qpso": "GNN + short QPSO",
    "gnn_aqpso": "GNN-anchored QPSO",
    "gnn_qqpso": "GNN + QQ-QPSO",
    "gnn_qqpso_noF": "GNN + QQ-QPSO w/o F",
    "gnn_qqpso_noB": "GNN + QQ-QPSO w/o B",
    "gnn_qqpso_noP": "GNN + QQ-QPSO w/o P",
    "gnn_pso_p": "GNN + short PSO + P",
    # diagnostics (validation only, Table VIII): raw predictions, delta = 0
    "direct_mlp_raw": "Direct pooled MLP (raw)",
    "direct_gnn_raw": "Direct GNN (raw)",
}

BASELINE_METHODS = ("full_pso", "rand_pso", "direct_mlp", "direct_gnn", "mlp_pso", "gnn_pso")
QUANTUM_METHODS = ("full_qpso", "rand_qpso", "mlp_qpso", "gnn_qpso", "gnn_aqpso")
# gnn_qqpso: GNN init + F (feasible rank-weighted mbest) + B (feasibility-adaptive beta)
# + P (m_b line-search polish); configuration chosen on validation (run_quantum.py).
QQ_METHODS = ("gnn_qqpso", "gnn_qqpso_noF", "gnn_qqpso_noB", "gnn_qqpso_noP")
QQ_OFF = dict(mbest_mode="mean", beta_mode="linear", polish_pops=0)
# Fairness control: PSO (Algorithm 1) + the same component P as gnn_qqpso (polish_pops taken
# from the selected gnn_qqpso config, NOT tuned separately). Answers "is the gain from P or QPSO?".
POLISH_METHODS = ("gnn_pso_p",)

# key -> (optimizer, init, budget kind, anchored)
_TABLE = {
    "full_pso": ("pso", None, "full", False),
    "rand_pso": ("pso", None, "short", False),
    "direct_mlp": (None, "mlp", "zero", False),
    "direct_gnn": (None, "gnn", "zero", False),
    "direct_mlp_raw": (None, "mlp", "zero", False),
    "direct_gnn_raw": (None, "gnn", "zero", False),
    "mlp_pso": ("pso", "mlp", "short", False),
    "gnn_pso": ("pso", "gnn", "short", False),
    "full_qpso": ("qpso", None, "full", False),
    "rand_qpso": ("qpso", None, "short", False),
    "mlp_qpso": ("qpso", "mlp", "short", False),
    "gnn_qpso": ("qpso", "gnn", "short", False),
    "gnn_aqpso": ("qpso", "gnn", "short", True),
    "gnn_qqpso": ("qpso", "gnn", "short", False),
    "gnn_qqpso_noF": ("qpso", "gnn", "short", False),
    "gnn_qqpso_noB": ("qpso", "gnn", "short", False),
    "gnn_qqpso_noP": ("qpso", "gnn", "short", False),
    "gnn_pso_p": ("pso", "gnn", "short", False),
}

_OPTIMIZERS = {"pso": ("baseline_code.pso", "run_pso"),
               "qpso": ("Quantummodel_code.qpso", "run_qpso")}


@dataclass(frozen=True)
class MethodSpec:
    key: str
    budget: int
    optimizer: str | None
    init: str | None
    delta: float
    anchored: bool = False
    tag: str = ""            # free label, e.g. anchor grid point "w0.5_f0.3"
    anchor_w0: float = 0.0
    anchor_frac: float = 0.0
    mbest_mode: str = "mean"     # QQ component F ("feas_rank") — QPSO only
    beta_mode: str = "linear"    # QQ component B ("feas")      — QPSO only
    polish_pops: int = 0         # QQ component P (>0)          — QPSO only

    @property
    def seed_branch(self) -> str:
        """Full PSO/QPSO use their own seed branch; short/hybrid share one (pairing)."""
        return "full" if self.key.startswith("full_") else "short"


def qq_ablation(key: str, qq: dict) -> dict:
    """Configuration of an ablation key derived from the selected gnn_qqpso config."""
    out = dict(QQ_OFF, **qq)
    if key == "gnn_qqpso_noF":
        out["mbest_mode"] = QQ_OFF["mbest_mode"]
    elif key == "gnn_qqpso_noB":
        out["beta_mode"] = QQ_OFF["beta_mode"]
    elif key == "gnn_qqpso_noP":
        out["polish_pops"] = QQ_OFF["polish_pops"]
    return out


def make_spec(key: str, budget_short: int, budget_full: int, delta: float,
              anchor_w0: float = 0.0, anchor_frac: float = 0.0, tag: str = "",
              qq: dict | None = None) -> MethodSpec:
    if key not in _TABLE:
        raise KeyError(f"unknown method {key!r}; known: {sorted(_TABLE)}")
    opt, init, kind, anch = _TABLE[key]
    budget = {"full": budget_full, "short": budget_short, "zero": 0}[kind]
    d = 0.0 if key.endswith("_raw") else delta
    q = dict(QQ_OFF)
    if key in QQ_METHODS:
        assert qq is not None, f"{key} needs a qq configuration (mbest_mode, beta_mode, polish_pops)"
        q = qq_ablation(key, qq)
    elif key in POLISH_METHODS:
        assert qq is not None, f"{key} needs the selected polish_pops (qq configuration)"
        q["polish_pops"] = int(qq["polish_pops"])      # PSO has no mbest/beta: only P is shared
    elif qq:
        raise ValueError(f"qq configuration given for non-QQ method {key!r}")
    return MethodSpec(key, int(budget), opt, init, d, anch, tag,
                      anchor_w0 if anch else 0.0, anchor_frac if anch else 0.0,
                      str(q["mbest_mode"]), str(q["beta_mode"]), int(q["polish_pops"]))


def get_optimizer(name: str):
    mod, fn = _OPTIMIZERS[name]
    return getattr(importlib.import_module(mod), fn)
