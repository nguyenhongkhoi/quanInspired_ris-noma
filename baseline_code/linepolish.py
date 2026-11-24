"""Swarm-free baseline: structural line search on the covert fraction (no PSO/QPSO update).

Answers "does the prior + boundary polishing already match Full PSO without a swarm?".
Budget B = S*T, deterministic (no random numbers):
  * population 1: S points that copy the initial allocation (the physics prior) and vary only
    z_b on the uniform grid linspace(0, 1, S); point 0 is the prior itself;
  * populations 2..T: component P (common/swarm.py run_polish), i.e. the same boundary-polishing
    line search used by QW-QPSO and GNN+PSO+P, around the best candidate so far.
The public split and the RIS share stay at the prior values throughout (Corollary 3 direction).
"""
from __future__ import annotations

import numpy as np

from common.evaluator import Problem
from common.swarm import SearchResult, SwarmMemory, check_budget, run_polish


def run_linepolish(problem: Problem, budget: int, swarm: int, seed: int,
                   init_z: np.ndarray | None = None, archive_size: int = 0, zb_max: float = 1.0,
                   **_unused) -> SearchResult:
    S = int(swarm)
    T = check_budget(budget, S)
    D = problem.K + 1
    assert init_z is not None, "linepolish needs an initial allocation (prior)"
    init_z = np.asarray(init_z, float).reshape(D)
    assert np.all((init_z >= 0) & (init_z <= 1)), "init point outside [0,1]"
    Z = np.repeat(init_z[None, :], S, axis=0)
    assert 0.0 < zb_max <= 1.0 and init_z[0] <= zb_max + 1e-9, (zb_max, init_z[0])
    Z[1:, 0] = np.linspace(0.0, zb_max, S - 1)     # z_b scan; row 0 keeps the prior exactly
    ev = problem.evaluate_z(Z)
    mem = SwarmMemory(problem, archive_size)
    mem.init(Z, ev)
    run_polish(problem, mem, S, T - 1, zb_max)
    res = mem.result(n_evals=S * T, init_feasible=bool(ev.feasible[0]))
    assert len(res.history_best_rb) == T
    return res
