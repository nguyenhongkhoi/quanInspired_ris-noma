"""Parallel simulated annealing with feasibility-first energy (comparison baseline).

Standard settings (Kirkpatrick et al. 1983, Metropolis acceptance; geometric cooling), NOT tuned:
  * S independent chains (S = swarm size) advanced in lockstep, so one "population" = S proposals
    and the budget B = S*T has the same meaning as for PSO/QPSO/GA (T evaluated populations);
  * energy consistent with the feasibility-first ranking used everywhere else:
        E = -R_b          if feasible      (always <= 0)
        E = nu (> 0)      if infeasible    (violation; any feasible point beats any infeasible one);
  * proposal: z' = clip(z + N(0, sigma_t^2 I), 0, 1), sigma_t linear sigma0 -> sigma1;
  * Metropolis: accept if dE <= 0, else with prob exp(-dE / T_t);
  * temperature geometric T0 -> T1 over the T-1 moves.
Pairing: the initial population is drawn exactly like PSO (z, then an unused velocity draw), so
for a given seed SA starts from the same S points as PSO/QPSO/GA. Returns the best-ranked
candidate ever evaluated (common.swarm.SwarmMemory), like every other optimizer.
"""
from __future__ import annotations

import numpy as np

from common.evaluator import Problem
from common.swarm import SearchResult, SwarmMemory, check_budget


def _energy(feasible: np.ndarray, nu: np.ndarray, rb: np.ndarray) -> np.ndarray:
    feasible = np.asarray(feasible, bool)
    nu = np.asarray(nu, float)
    rb = np.asarray(rb, float)
    # infeasible points have nu > 0; keep a tiny floor so E_infeasible > 0 >= E_feasible always
    return np.where(feasible, -rb, np.maximum(nu, 1e-12))


def run_sa(problem: Problem, budget: int, swarm: int, seed: int,
           init_z: np.ndarray | None = None, archive_size: int = 0,
           t0: float = 0.5, t1: float = 1e-3, sigma0: float = 0.1, sigma1: float = 0.01,
           v_init: float = 0.1, **_unused) -> SearchResult:
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
    cur = z.copy()
    e_cur = _energy(ev.feasible, ev.nu, ev.R_b)
    assert e_cur.shape == (S,)

    n_moves = T - 1
    for t in range(n_moves):
        frac = t / max(n_moves - 1, 1)
        temp = t0 * (t1 / t0) ** frac
        sigma = sigma0 + (sigma1 - sigma0) * frac
        prop = np.clip(cur + sigma * rng.standard_normal((S, D)), 0.0, 1.0)
        evp = problem.evaluate_z(prop)
        mem.update(prop, evp)
        e_new = _energy(evp.feasible, evp.nu, evp.R_b)
        dE = e_new - e_cur
        u = rng.random(S)
        acc = (dE <= 0.0) | (u < np.exp(-np.maximum(dE, 0.0) / temp))
        cur[acc] = prop[acc]
        e_cur[acc] = e_new[acc]

    res = mem.result(n_evals=S * T, init_feasible=init_feas)
    assert len(res.history_best_rb) == T
    return res
