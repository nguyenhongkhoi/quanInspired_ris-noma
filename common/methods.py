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
    "gnn_qwqpso": "GNN + QW-QPSO",
    "gnn_qwqpso_noW": "GNN + QW-QPSO w/o W",
    "gnn_qwqpso_noB": "GNN + QW-QPSO w/o B",
    "gnn_pso_wp": "GNN + short PSO + W + P",
    "gnn_pso_wpt": "GNN + short PSO + W + P (tuned)",
    "full_ga": "Full GA",
    "gnn_ga": "GNN + short GA",
    "full_de": "Full DE",
    "gnn_de": "GNN + short DE",
    "prior_qwqpso": "Prior + QW-QPSO",
    "prior_direct": "Physics prior (direct)",
    "prior_polish": "Prior + line polish (no swarm)",
    "prior_dqpso": "DEP-prior + boxed QW-QPSO",
    "prior_dqpso_noW": "DEP-prior + boxed QW-QPSO w/o W",
    "prior_dqpso_noBox": "DEP-prior + QW-QPSO w/o box",
    "prior_ddirect": "DEP-prior (direct)",
    "prior_dpolish": "DEP-prior + line polish (no swarm)",
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
# gnn_qwqpso: QPSO-specific components on top of P: W (quantum-well initialisation around the
# GNN prediction) + B (short-budget beta schedule). Config chosen on validation (run_quantum.py).
QW_METHODS = ("gnn_qwqpso", "gnn_qwqpso_noW", "gnn_qwqpso_noB")
QW_OFF = dict(well_scale=0.0, well_frac=0.0, beta_start=None, beta_end=None, polish_pops=0)
# Controls for "does W need QPSO?": PSO (Algorithm 1 update) + the same W + P.
#   gnn_pso_wp : W and P copied from the selected gnn_qwqpso config, PSO inertia as in the paper
#                (only the update rule differs from gnn_qwqpso; identical initial population).
#   gnn_pso_wpt: W, inertia schedule and P tuned on validation with a grid of the same size as
#                gnn_qwqpso's (7 W x 3 inertia x 2 P = 42 cells) -> balanced tuning effort.
PW_METHODS = ("gnn_pso_wp", "gnn_pso_wpt")
# Learning-free methods: init = closed-form physics prior with RIS share s = prior_s (validation).
PRIOR_METHODS = ("prior_qwqpso", "prior_direct", "prior_polish")
PRIOR_BASE_METHODS = ("prior_direct", "prior_polish")   # swarm-free references, own s on validation
# DEP-aware prior (common/prior.py dep_prior): m_b on min(DEP bound, QoS bound) computed from the
# case's Willie moments (0 objective evaluations); the swarm searches a box z_b <= zb_cap, so its
# job is the public split / RIS share / m_b below the cap, not finding the DEP boundary.
DPRIOR_METHODS = ("prior_dqpso", "prior_dqpso_noW", "prior_dqpso_noBox")
DPRIOR_BASE_METHODS = ("prior_ddirect", "prior_dpolish")
BOX_METHODS = ("prior_dqpso", "prior_dqpso_noW", "prior_dpolish")
PRIOR_INITS = ("prior", "dep_prior")

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
    "gnn_qwqpso": ("qpso", "gnn", "short", False),
    "gnn_qwqpso_noW": ("qpso", "gnn", "short", False),
    "gnn_qwqpso_noB": ("qpso", "gnn", "short", False),
    "gnn_pso_wp": ("pso", "gnn", "short", False),
    "gnn_pso_wpt": ("pso", "gnn", "short", False),
    # other metaheuristics (comparison baselines, standard settings, no tuning)
    "full_ga": ("ga", None, "full", False),
    "gnn_ga": ("ga", "gnn", "short", False),
    "full_de": ("de", None, "full", False),
    "gnn_de": ("de", "gnn", "short", False),
    # learning-free variant: closed-form physics prior (common/prior.py) instead of the GNN
    "prior_qwqpso": ("qpso", "prior", "short", False),
    "prior_direct": (None, "prior", "zero", False),
    "prior_polish": ("linepolish", "prior", "short", False),
    "prior_dqpso": ("qpso", "dep_prior", "short", False),
    "prior_dqpso_noW": ("qpso", "dep_prior", "short", False),
    "prior_dqpso_noBox": ("qpso", "dep_prior", "short", False),
    "prior_ddirect": (None, "dep_prior", "zero", False),
    "prior_dpolish": ("linepolish", "dep_prior", "short", False),
}

_OPTIMIZERS = {"pso": ("baseline_code.pso", "run_pso"),
               "qpso": ("Quantummodel_code.qpso", "run_qpso"),
               "ga": ("baseline_code.ga", "run_ga"),
               "de": ("baseline_code.de", "run_de"),
               "linepolish": ("baseline_code.linepolish", "run_linepolish")}


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
    polish_pops: int = 0         # component P (>0)             — QPSO, and PSO for gnn_pso_p
    well_scale: float = 0.0      # QW component W: well length L (z units)
    well_frac: float = 0.0       # QW component W: fraction of the S-1 random particles drawn in the well
    beta_start: float | None = None   # QW component B: per-method beta schedule (None -> exp default)
    beta_end: float | None = None
    inertia_start: float | None = None   # PSO inertia schedule (None -> Algorithm 1: 0.9 -> 0.4)
    inertia_end: float | None = None
    prior_s: float | None = None         # RIS share of the physics prior (init == "prior"/"dep_prior")
    zb_box: bool = False                 # restrict z_b <= zb_cap (DEP/QoS bound), prior_d* only

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


def qw_ablation(key: str, qw: dict) -> dict:
    out = dict(QW_OFF, **qw)
    if key == "gnn_qwqpso_noW":
        out["well_scale"], out["well_frac"] = 0.0, 0.0
    elif key == "gnn_qwqpso_noB":
        out["beta_start"], out["beta_end"] = None, None
    return out


def make_spec(key: str, budget_short: int, budget_full: int, delta: float,
              anchor_w0: float = 0.0, anchor_frac: float = 0.0, tag: str = "",
              qq: dict | None = None, qw: dict | None = None, pw: dict | None = None,
              prior_s: float | None = None) -> MethodSpec:
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
    w = dict(QW_OFF)
    i_s = i_e = None
    if key in QW_METHODS:
        assert qw is not None, f"{key} needs a qw configuration (well_scale, well_frac, beta_*, polish_pops)"
        w = qw_ablation(key, qw)
        q["polish_pops"] = int(w["polish_pops"])
    elif key in DPRIOR_METHODS:
        assert qw is not None, f"{key} needs a qw configuration (well_scale, well_frac, beta_*, polish_pops)"
        w = dict(QW_OFF, **{k: v for k, v in qw.items() if k in QW_OFF})
        if key == "prior_dqpso_noW":
            w["well_scale"], w["well_frac"] = 0.0, 0.0
        q["polish_pops"] = int(w["polish_pops"])
    elif key == "prior_qwqpso":
        assert qw is not None, f"{key} needs a qw configuration (well_scale, well_frac, beta_*, polish_pops)"
        w = dict(QW_OFF, **{k: v for k, v in qw.items() if k in QW_OFF})
        q["polish_pops"] = int(w["polish_pops"])
    elif key == "gnn_pso_wp":
        assert qw is not None, f"{key} needs the selected gnn_qwqpso configuration (W and P are copied)"
        w.update(well_scale=float(qw["well_scale"]), well_frac=float(qw["well_frac"]))
        q["polish_pops"] = int(qw["polish_pops"])      # beta has no PSO meaning: not copied
    elif key == "gnn_pso_wpt":
        assert pw is not None, f"{key} needs a pw configuration (well_scale, well_frac, inertia_*, polish_pops)"
        w.update(well_scale=float(pw["well_scale"]), well_frac=float(pw["well_frac"]))
        q["polish_pops"] = int(pw["polish_pops"])
        i_s, i_e = float(pw["inertia_start"]), float(pw["inertia_end"])
    elif qw:
        raise ValueError(f"qw configuration given for non-QW method {key!r}")
    if pw and key != "gnn_pso_wpt":
        raise ValueError(f"pw configuration given for {key!r}")
    if init in PRIOR_INITS:
        assert prior_s is not None, f"{key} needs prior_s (RIS share of the physics prior)"
    elif prior_s is not None:
        raise ValueError(f"prior_s given for non-prior method {key!r}")
    bs = None if w["beta_start"] is None else float(w["beta_start"])
    be = None if w["beta_end"] is None else float(w["beta_end"])
    assert (bs is None) == (be is None), "beta_start and beta_end must be set together"
    return MethodSpec(key, int(budget), opt, init, d, anch, tag,
                      anchor_w0 if anch else 0.0, anchor_frac if anch else 0.0,
                      str(q["mbest_mode"]), str(q["beta_mode"]), int(q["polish_pops"]),
                      float(w["well_scale"]), float(w["well_frac"]), bs, be, i_s, i_e,
                      None if prior_s is None else float(prior_s), key in BOX_METHODS)


def get_optimizer(name: str):
    mod, fn = _OPTIMIZERS[name]
    return getattr(importlib.import_module(mod), fn)
