"""Real-coded genetic algorithm with feasibility-first selection (comparison baseline).

Standard settings (Deb & Agrawal 1995; Deb 2000 constraint handling):
  * population S (= swarm size), budget B = S*T: T evaluated populations
    (initial population + T-1 generations of S offspring each);
  * binary tournament on the feasibility-first rank (common.evaluator.rank_order,
    i.e. the same ranking as PSO/QPSO: feasible > infeasible; rate; violation);
  * simulated binary crossover (SBX, eta_c = 15, p_c = 0.9, per-variable prob 0.5);
  * polynomial mutation (eta_m = 20, p_m = 1/D);
  * (mu + lambda) survival: parents + offspring ranked, best S kept (elitist);
  * positions clipped to [0, 1].
Pairing: the initial population is drawn exactly like PSO (z, then an unused velocity
draw), so for a given seed GA, PSO and QPSO start from the same S points and the learned
particle only overwrites z[0]. Returns the best-ranked candidate ever evaluated.
"""
from __future__ import annotations

import numpy as np

from common.evaluator import Problem, rank_order
from common.swarm import SearchResult, SwarmMemory, check_budget


def _sbx(p1: np.ndarray, p2: np.ndarray, rng, eta: float, pc: float) -> tuple[np.ndarray, np.ndarray]:
    n, D = p1.shape
    do_pair = rng.random((n, 1)) < pc
    do_var = rng.random((n, D)) < 0.5
    u = rng.random((n, D))
    beta = np.where(u <= 0.5, (2.0 * u) ** (1.0 / (eta + 1.0)),
                    (1.0 / (2.0 * (1.0 - u) + 1e-300)) ** (1.0 / (eta + 1.0)))
    c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
    c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
    mask = do_pair & do_var
    return np.where(mask, c1, p1), np.where(mask, c2, p2)


def _poly_mutation(z: np.ndarray, rng, eta: float, pm: float) -> np.ndarray:
    u = rng.random(z.shape)
    delta = np.where(u < 0.5, (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0,
                     1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0)))
    hit = rng.random(z.shape) < pm
    return np.where(hit, z + delta, z)          # variable range is [0, 1]


def run_ga(problem: Problem, budget: int, swarm: int, seed: int,
           init_z: np.ndarray | None = None, archive_size: int = 0,
           eta_c: float = 15.0, p_c: float = 0.9, eta_m: float = 20.0,
           v_init: float = 0.1, **_unused) -> SearchResult:
    S = int(swarm)
    T = check_budget(budget, S)
    D = problem.K + 1
    pm = 1.0 / D
    rng = np.random.default_rng(seed)
    z = rng.random((S, D))
    _ = rng.uniform(-v_init, v_init, (S, D))     # same draw as PSO velocity init (pairing)
    if init_z is not None:
        init_z = np.asarray(init_z, float).reshape(D)
        assert np.all((init_z >= 0) & (init_z <= 1)), "init particle outside [0,1]"
        z[0] = init_z
    ev = problem.evaluate_z(z)
    init_feas = bool(ev.feasible[0]) if init_z is not None else None
    mem = SwarmMemory(problem, archive_size)          # tracks the best candidate ever evaluated
    mem.init(z, ev)
    pop, pf, pnu, prb = z.copy(), ev.feasible.copy(), ev.nu.copy(), ev.R_b.copy()

    for _t in range(T - 1):
        order = rank_order(pf, pnu, prb)
        rank = np.empty(S, dtype=int)
        rank[order] = np.arange(S)                    # 0 = best
        a = rng.integers(0, S, size=S)
        b = rng.integers(0, S, size=S)
        parents = np.where(rank[a] <= rank[b], a, b)  # binary tournament
        half = S // 2
        p1, p2 = pop[parents[:half]], pop[parents[half:2 * half]]
        c1, c2 = _sbx(p1, p2, rng, eta_c, p_c)
        kids = np.vstack([c1, c2])
        if kids.shape[0] < S:                         # odd S: copy one extra parent
            kids = np.vstack([kids, pop[parents[-1]][None, :]])
        kids = np.clip(_poly_mutation(kids, rng, eta_m, pm), 0.0, 1.0)
        assert kids.shape == (S, D)
        evk = problem.evaluate_z(kids)
        mem.update(kids, evk)
        # (mu + lambda) elitist survival under the same feasibility-first ranking
        allz = np.vstack([pop, kids])
        allf = np.concatenate([pf, evk.feasible])
        allnu = np.concatenate([pnu, evk.nu])
        allrb = np.concatenate([prb, evk.R_b])
        keep = rank_order(allf, allnu, allrb)[:S]
        pop, pf, pnu, prb = allz[keep], allf[keep], allnu[keep], allrb[keep]

    res = mem.result(n_evals=S * T, init_feasible=init_feas)
    assert len(res.history_best_rb) == T
    return res
