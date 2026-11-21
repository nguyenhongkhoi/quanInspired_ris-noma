"""Feasibility-first PSO (Algorithm 1, Appendix B-B).

Pairing rule (claude.md 5.6): all random draws are made in the same order whether
or not an initial particle is supplied; the learned particle only overwrites z[0].
"""
from __future__ import annotations

import numpy as np

from common.evaluator import Problem
from common.swarm import SearchResult, SwarmMemory, check_budget


def run_pso(problem: Problem, budget: int, swarm: int, seed: int,
            init_z: np.ndarray | None = None, archive_size: int = 0,
            c1: float = 1.5, c2: float = 1.5, w_start: float = 0.9, w_end: float = 0.4,
            v_init: float = 0.1, v_max: float = 0.25, **_unused) -> SearchResult:
    S = int(swarm)
    T = check_budget(budget, S)
    D = problem.K + 1
    rng = np.random.default_rng(seed)
    z = rng.random((S, D))
    v = rng.uniform(-v_init, v_init, (S, D))
    if init_z is not None:
        init_z = np.asarray(init_z, float).reshape(D)
        assert np.all((init_z >= 0) & (init_z <= 1)), "init particle outside [0,1]"
        z[0] = init_z
    ev = problem.evaluate_z(z)
    init_feas = bool(ev.feasible[0]) if init_z is not None else None
    mem = SwarmMemory(problem, archive_size)
    mem.init(z, ev)
    ws = np.linspace(w_start, w_end, T - 1) if T > 1 else np.array([])
    for t in range(T - 1):
        r1 = rng.random((S, D))
        r2 = rng.random((S, D))
        g = mem.gbest_z()
        v = ws[t] * v + c1 * r1 * (mem.pz - z) + c2 * r2 * (g[None, :] - z)
        v = np.clip(v, -v_max, v_max)
        z = np.clip(z + v, 0.0, 1.0)
        ev = problem.evaluate_z(z)
        mem.update(z, ev)
    return mem.result(n_evals=S * T, init_feasible=init_feas)
