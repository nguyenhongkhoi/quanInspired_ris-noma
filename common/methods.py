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
    # diagnostics (validation only, Table VIII): raw predictions, delta = 0
    "direct_mlp_raw": "Direct pooled MLP (raw)",
    "direct_gnn_raw": "Direct GNN (raw)",
}

BASELINE_METHODS = ("full_pso", "rand_pso", "direct_mlp", "direct_gnn", "mlp_pso", "gnn_pso")
QUANTUM_METHODS = ("full_qpso", "rand_qpso", "mlp_qpso", "gnn_qpso", "gnn_aqpso")

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

    @property
    def seed_branch(self) -> str:
        """Full PSO/QPSO use their own seed branch; short/hybrid share one (pairing)."""
        return "full" if self.key.startswith("full_") else "short"


def make_spec(key: str, budget_short: int, budget_full: int, delta: float,
              anchor_w0: float = 0.0, anchor_frac: float = 0.0, tag: str = "") -> MethodSpec:
    if key not in _TABLE:
        raise KeyError(f"unknown method {key!r}; known: {sorted(_TABLE)}")
    opt, init, kind, anch = _TABLE[key]
    budget = {"full": budget_full, "short": budget_short, "zero": 0}[kind]
    d = 0.0 if key.endswith("_raw") else delta
    return MethodSpec(key, int(budget), opt, init, d, anch, tag,
                      anchor_w0 if anch else 0.0, anchor_frac if anch else 0.0)


def get_optimizer(name: str):
    mod, fn = _OPTIMIZERS[name]
    return getattr(importlib.import_module(mod), fn)
