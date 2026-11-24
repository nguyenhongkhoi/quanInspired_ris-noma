"""Differential evolution DE/rand/1/bin with feasibility-first selection (comparison baseline).

Standard settings (Storn & Price 1997; Deb's feasibility rules for constraints):
  * population S (= swarm size), budget B = S*T: T evaluated populations;
  * mutant v_i = z_r1 + F (z_r2 - z_r3), r1, r2, r3 distinct and != i, F = 0.5;
  * binomial crossover with CR = 0.9 (one coordinate always from the mutant);
  * trial clipped to [0, 1];
  * one-to-one survival: the trial replaces its target if it ranks better under the same
    feasibility-first rule as PSO/QPSO (common.evaluator.is_better, via SwarmMemory).
Pairing: the initial population is drawn exactly like PSO (z, then an unused velocity
draw); the learned particle only overwrites z[0].
"""
from __future__ import annotations

import numpy as np

from common.evaluator import Problem
from common.swarm import SearchResult, SwarmMemory, check_budget


def run_de(problem: Problem, budget: int, swarm: int, seed: int,
           init_z: np.ndarray | None = None, archive_size: int = 0,
           F: float = 0.5, CR: float = 0.9, v_init: float = 0.1, **_unused) -> SearchResult:
    S = int(swarm)
    assert S >= 4, "DE/rand/1 needs at least 4 individuals"
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
    mem = SwarmMemory(problem, archive_size)     # mem.pz IS the DE population (one-to-one survival)
    mem.init(z, ev)

    idx = np.arange(S)
    for _t in range(T - 1):
        X = mem.pz
        r = np.empty((S, 3), dtype=int)
        for i in range(S):
            r[i] = rng.choice(idx[idx != i], size=3, replace=False)
        V = X[r[:, 0]] + F * (X[r[:, 1]] - X[r[:, 2]])
        cross = rng.random((S, D)) < CR
        jrand = rng.integers(0, D, size=S)
        cross[idx, jrand] = True
        U = np.clip(np.where(cross, V, X), 0.0, 1.0)
        assert U.shape == (S, D)
        mem.update(U, problem.evaluate_z(U))     # trial replaces target iff it ranks better

    res = mem.result(n_evals=S * T, init_feasible=init_feas)
    assert len(res.history_best_rb) == T
    return res
