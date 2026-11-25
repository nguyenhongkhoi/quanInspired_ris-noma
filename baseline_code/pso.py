"""Feasibility-first PSO (Algorithm 1, Appendix B-B).

Pairing rule (claude.md 5.6): all random draws are made in the same order whether
or not an initial particle is supplied; the learned particle only overwrites z[0].

polish_pops (default 0 = Algorithm 1 of the paper, unchanged): the last n populations
of the budget run component P (common/swarm.py run_polish, the same code as QPSO's P).
Only used by the fairness control gnn_pso_p; inertia schedule spans the T-1-n updates.

zb_max (default 1 = unit box, original code path): restricted box z_b <= zb_max, same rule as
QPSO (initial draws scaled into the box, positions clipped to [0, hi]); used by prior_fpso only.

well_scale/well_frac (default 0 = off): component W (common/swarm.py well_init, the SAME
code and rng stream as gnn_qwqpso), used only by the controls gnn_pso_wp / gnn_pso_wpt
("does the quantum-well initialisation need QPSO?"). With W on, gnn_pso_wp and gnn_qwqpso
start from the identical 30-particle population; only the update rule differs.
"""
from __future__ import annotations

import numpy as np

from common.evaluator import Problem
from common.swarm import SearchResult, SwarmMemory, box_hi, check_budget, run_polish, well_init


def run_pso(problem: Problem, budget: int, swarm: int, seed: int,
            init_z: np.ndarray | None = None, archive_size: int = 0,
            c1: float = 1.5, c2: float = 1.5, w_start: float = 0.9, w_end: float = 0.4,
            v_init: float = 0.1, v_max: float = 0.25, polish_pops: int = 0,
            well_scale: float = 0.0, well_frac: float = 0.0, boundary: str = "reflect",
            zb_max: float = 1.0, **_unused) -> SearchResult:
    S = int(swarm)
    T = check_budget(budget, S)
    polish_pops = int(polish_pops)
    assert 0 <= polish_pops <= T - 1, f"polish_pops={polish_pops} needs budget > {polish_pops}*swarm"
    D = problem.K + 1
    rng = np.random.default_rng(seed)
    z = rng.random((S, D))
    v = rng.uniform(-v_init, v_init, (S, D))
    hi = box_hi(D, zb_max)                     # None -> unit box (original path)
    if hi is not None:
        z = z * hi[None, :]
    if init_z is not None:
        init_z = np.asarray(init_z, float).reshape(D)
        assert np.all((init_z >= 0) & (init_z <= 1)), "init particle outside [0,1]"
        if hi is not None:
            assert init_z[0] <= hi[0] + 1e-9, (init_z[0], hi[0])
        z[0] = init_z
        z = well_init(z, init_z, seed, well_scale, well_frac, boundary, hi=hi)   # component W (off by default)
    ev = problem.evaluate_z(z)
    init_feas = bool(ev.feasible[0]) if init_z is not None else None
    mem = SwarmMemory(problem, archive_size)
    mem.init(z, ev)
    n_upd = T - 1 - polish_pops
    ws = np.linspace(w_start, w_end, n_upd) if n_upd > 0 else np.array([])
    for t in range(n_upd):
        r1 = rng.random((S, D))
        r2 = rng.random((S, D))
        g = mem.gbest_z()
        v = ws[t] * v + c1 * r1 * (mem.pz - z) + c2 * r2 * (g[None, :] - z)
        v = np.clip(v, -v_max, v_max)
        z = np.clip(z + v, 0.0, 1.0) if hi is None else np.clip(z + v, 0.0, hi[None, :])
        ev = problem.evaluate_z(z)
        mem.update(z, ev)
    run_polish(problem, mem, S, polish_pops, zb_max)
    res = mem.result(n_evals=S * T, init_feasible=init_feas)
    assert len(res.history_best_rb) == T
    return res
