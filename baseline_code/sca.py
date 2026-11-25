"""Sine Cosine Algorithm (Mirjalili, Knowledge-Based Systems 2016) with feasibility-first
destination (comparison baseline). Standard settings, NOT tuned:
  * population S (= swarm size), budget B = S*T: T evaluated populations;
  * destination P = best candidate so far under the same feasibility-first ranking as PSO/QPSO
    (common.swarm.SwarmMemory gbest);
  * update per agent i and dimension d:
        r1 = a - t * a / (T - 1)   (a = 2, linear to 0 over the T-1 updates)
        r2 ~ U(0, 2 pi), r3 ~ U(0, 2), r4 ~ U(0, 1)
        z <- z + r1 sin(r2) |r3 P - z|   if r4 < 0.5
        z <- z + r1 cos(r2) |r3 P - z|   otherwise
  * positions clipped to [0, 1].
Pairing: the initial population is drawn exactly like PSO (z, then an unused velocity draw).
Returns the best-ranked candidate ever evaluated.
"""
from __future__ import annotations

import numpy as np

from common.evaluator import Problem
from common.swarm import SearchResult, SwarmMemory, check_budget


def run_sca(problem: Problem, budget: int, swarm: int, seed: int,
            init_z: np.ndarray | None = None, archive_size: int = 0,
            a: float = 2.0, v_init: float = 0.1, **_unused) -> SearchResult:
    S = int(swarm)
    T = check_budget(budget, S)
    D = problem.K + 1
    rng = np.random.default_rng(seed)
    z = rng.random((S, D))
    _ = rng.uniform(-v_init, v_init, (S, D))     # same draw as PSO velocity init (pairing)
    if init_z is not None:
        init_z = np.asarray(init_z, float).reshape(D)
        assert np.all((init_z >= 0) & (init_z <= 1)), "init particle outside [0,1]"
        z[0] = init_z
    ev = problem.evaluate_z(z)
    init_feas = bool(ev.feasible[0]) if init_z is not None else None
    mem = SwarmMemory(problem, archive_size)
    mem.init(z, ev)

    n_upd = T - 1
    for t in range(n_upd):
        r1 = a - t * a / max(n_upd, 1)
        P = mem.gbest_z()
        assert P.shape == (D,)
        r2 = rng.uniform(0.0, 2.0 * np.pi, (S, D))
        r3 = rng.uniform(0.0, 2.0, (S, D))
        r4 = rng.random((S, D))
        step = np.abs(r3 * P[None, :] - z)
        z = z + r1 * np.where(r4 < 0.5, np.sin(r2), np.cos(r2)) * step
        z = np.clip(z, 0.0, 1.0)
        evz = problem.evaluate_z(z)
        mem.update(z, evz)

    res = mem.result(n_evals=S * T, init_feasible=init_feas)
    assert len(res.history_best_rb) == T
    return res
