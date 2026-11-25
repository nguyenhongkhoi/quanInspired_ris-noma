"""RIS-phase extension (harder problem): allocation + a common RIS phase offset Delta.

The paper fixes the RIS phases to the Bob-aligned rule theta_n = -arg(h_ar,n) - arg(h_rb,n).
Here the optimiser also chooses an offset Delta in [-pi, pi]^N added to that rule in every block:
    theta_n(m) = -arg(h_ar,n(m)) - arg(h_rb,n(m)) + Delta_n .
Delta = 0 is exactly the paper's model, so every allocation of the paper is still available and
the closed-form (fair) prior stays a valid start point. Delta trades Bob's coherent gain against
Willie's detectability (DEP) and the public users' gains: no closed form.

Particle z in [0,1]^(K+1+N): the first K+1 coordinates are the paper's allocation coordinates
(claude.md 5.1), the last N map to Delta_n = pi (2 z - 1)  (z = 0.5 <-> Delta = 0).

All physics (rates, SIC, outage, Gamma DEP, ranking) is the single implementation in
common/evaluator.py; this module only supplies per-particle gains X, Y at rho = 1.
"""
from __future__ import annotations

import json
import time

import numpy as np

from .assessment import BOUND_FLAGS, POINT_FLAGS, assess_allocation
from .channel import bank_seed, build_bank, generate_layout
from .evaluator import Problem, alloc_to_z, to_original, z_to_alloc

PHASE_SPAN = np.pi
PHASE_METHODS = ("phase_fdirect", "phase_fqpso", "phase_fullpso")


def z_to_delta(zp: np.ndarray) -> np.ndarray:
    return PHASE_SPAN * (2.0 * np.asarray(zp, dtype=float) - 1.0)


class PhaseProblem(Problem):
    """Problem whose particles carry N extra phase coordinates. evaluate_alloc (no XY) = Delta 0."""

    def __init__(self, sys, bank, P_tot_dbm, xi_req, mode="search", sic_order=None, eligible=None):
        assert bank.A0 is not None and bank.G0 is not None, "build the bank with keep_raw=True"
        super().__init__(sys, bank, P_tot_dbm, xi_req, mode, sic_order=sic_order, eligible=eligible)
        M, R, N = bank.A0.shape
        assert (M, R) == (self.M, self.K + 2) and N == sys.N, (bank.A0.shape, self.M, self.K, sys.N)
        self.N = N
        self.D = self.K + 1 + N
        self._A = bank.A0.reshape(M * R, N)          # row index = m * R + r
        self._G = bank.G0.reshape(M * R, N)

    def gains(self, zp: np.ndarray):
        """zp (P, N) phase coordinates -> X, Y each (P, M, K+2) at rho = 1."""
        zp = np.atleast_2d(np.asarray(zp, dtype=float))
        P = zp.shape[0]
        assert zp.shape == (P, self.N), zp.shape
        E = np.exp(1j * z_to_delta(zp))              # (P, N); exactly 1 at z = 0.5
        R = self.K + 2
        H = (self._A @ E.T).T.reshape(P, self.M, R)
        G = (self._G @ E.T).T.reshape(P, self.M, R)
        return np.abs(H) ** 2, np.abs(G) ** 2

    def evaluate_z(self, z: np.ndarray, keep_U: bool = False):
        z = np.atleast_2d(np.asarray(z, dtype=float))
        assert z.shape[1] == self.D, f"z has {z.shape[1]} coords, expected D={self.D}"
        K = self.K
        mb, s, m_sic = z_to_alloc(self.sys, z[:, :K + 1], K)
        return self.evaluate_alloc(mb, s, m_sic, keep_U=keep_U, XY=self.gains(z[:, K + 1:]))


def phase_group(task: dict) -> list[dict]:
    """Worker (top level, spawn-safe): all cases x replicates x methods of one (layout, K)."""
    from baseline_code.pso import run_pso
    from Quantummodel_code.qpso import run_qpso
    from .prior import fair_prior
    from .runner import opt_seed

    sys_, exp = task["sys"], task["exp"]
    lid, K, split = int(task["layout_id"]), int(task["K"]), task["split"]
    cfg = task["cfg"]                    # selected dprior config (validation of nopol_B_01)
    methods = task["methods"]
    B_short = int(task["budget_short"])
    S = exp.swarm
    N = sys_.N

    t0 = time.perf_counter()
    layout = generate_layout(sys_, exp.seed, lid, split, exp.K_max)
    bank_s = build_bank(sys_, layout, K, exp.search_samples, bank_seed(exp.seed, "search", lid, K), keep_raw=True)
    share_ms = (time.perf_counter() - t0) * 1e3 / max(1, len(task["cases"]))
    a_seed = bank_seed(exp.seed, "assess", lid, K)
    bank_a0 = build_bank(sys_, layout, K, exp.assess_samples, a_seed)      # Delta = 0 assessment bank

    rows = []
    for case in task["cases"]:
        P_tot, xi = float(case["P_tot_dbm"]), float(case["xi_req"])
        for r in range(int(task["replicates"])):
            for key in methods:
                t1 = time.perf_counter()
                prob = PhaseProblem(sys_, bank_s, P_tot, xi, "search")
                n_prior = 0
                init_feas = None
                hist = None
                if key in ("phase_fdirect", "phase_fqpso"):
                    mb0, s0, m0, n_prior = fair_prior(prob, float(cfg["prior_s"]))      # Delta = 0
                    z0 = np.concatenate([alloc_to_z(sys_, mb0, s0, m0[None, :])[0], np.full(N, 0.5)])
                if key == "phase_fdirect":
                    z_best = z0
                    n_evals = n_prior
                    budget = 0
                elif key == "phase_fqpso":
                    budget = B_short
                    seed = opt_seed(exp.seed, case["case_id"], exp.pso_seeds[r], "short", budget)
                    assert n_prior <= S, n_prior
                    res = run_qpso(prob, budget - S, S, seed, init_z=z0,
                                   beta_start=float(cfg["beta_start"]), beta_end=float(cfg["beta_end"]),
                                   boundary=exp.boundary, polish_pops=0,
                                   well_scale=float(cfg["well_scale"]), well_frac=float(cfg["well_frac"]))
                    z_best, n_evals, init_feas = res.best.z, res.n_evals + n_prior, res.init_feasible
                    hist = res.history_best_rb
                elif key == "phase_fullpso":
                    budget = exp.budget_full
                    seed = opt_seed(exp.seed, case["case_id"], exp.pso_seeds[r], "full", budget)
                    res = run_pso(prob, budget, S, seed)                   # Algorithm 1, random init
                    z_best, n_evals = res.best.z, res.n_evals
                    hist = res.history_best_rb
                else:
                    raise KeyError(key)
                assert n_evals <= max(budget, n_prior), (key, n_evals, budget)
                time_ms = share_ms + (time.perf_counter() - t1) * 1e3

                # untimed: search check of the returned point + assessment at its phases
                z_best = np.asarray(z_best, dtype=float).reshape(prob.D)
                ev_s = prob.evaluate_z(z_best[None, :])
                mb, s, m_sic = float(ev_s.mb[0]), float(ev_s.s[0]), ev_s.m_sic[0]
                delta = z_to_delta(z_best[K + 1:])
                Xd, _ = prob.gains(z_best[None, K + 1:])
                X0 = bank_s.X
                g_bob = float(Xd[0, :, 0].mean() / X0[:, 0].mean())
                g_wil = float(Xd[0, :, K + 1].mean() / X0[:, K + 1].mean())
                if np.all(delta == 0.0):
                    bank_a = bank_a0
                else:
                    bank_a = build_bank(sys_, layout, K, exp.assess_samples, a_seed, phase_offset=delta)
                prob_a = Problem(sys_, bank_a, P_tot, xi, "assess", sic_order=prob.sic_order, eligible=None)
                a = assess_allocation(sys_, prob_a, mb, s, m_sic)
                rows.append(dict(
                    group_key=task["group_key"], case_id=case["case_id"], layout_id=lid, split=split, K=K,
                    case_idx=case["case_idx"], P_tot_dbm=P_tot, xi_req=xi, replicate=r,
                    method=key, budget=budget, n_evals=int(n_evals), n_prior=int(n_prior),
                    mb=mb, s=s,
                    m_orig=json.dumps([round(float(x), 9) for x in to_original(np.asarray(m_sic), prob.sic_order)]),
                    search_feasible=bool(ev_s.feasible[0]), init_feasible=init_feas,
                    R_b_search=float(ev_s.R_b[0]), xi_gamma_search=float(ev_s.xi_gamma[0]),
                    R_b=a["R_b"], R_pub_min=a["R_pub_min"], R_L_min=a["R_L_min"], p_b=a["p_b"], F_b=a["F_b"],
                    p_bU=a["p_bU"], xi_emp=a["xi_emp"], xi_gamma=a["xi_gamma"], xi_L=a["xi_L"],
                    passed=a["passed"], category=a["category"],
                    **{k: a[k] for k in POINT_FLAGS}, **{k: a[k] for k in BOUND_FLAGS},
                    rb_ceiling=a["rb_ceiling"], cv_w=prob.cv_w, eligible=prob_a.eligible,
                    phase_rms=float(np.sqrt(np.mean(delta ** 2))), phase_maxabs=float(np.max(np.abs(delta))),
                    gain_bob_rel=g_bob, gain_willie_rel=g_wil,
                    delta=json.dumps([round(float(x), 5) for x in delta]),
                    time_ms=time_ms,
                    hist_rb=json.dumps([round(float(v), 6) for v in hist]) if hist is not None else "",
                    cfg=json.dumps(cfg, sort_keys=True),
                ))
    return rows
