"""Swarm bookkeeping shared by PSO and QPSO (personal/global best, archive).

Both optimizers use the same feasibility-first ranking from common.evaluator,
so the only difference between PSO and QPSO is the position update rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .evaluator import EvalBatch, Problem, is_better, rank_order, z_to_alloc


@dataclass
class Candidate:
    z: np.ndarray
    mb: float
    s: float
    m_sic: np.ndarray
    R_b: float
    feasible: bool
    nu: float


@dataclass
class SearchResult:
    best: Candidate
    n_evals: int
    init_feasible: bool | None = None          # search feasibility of particle 0 at first eval
    archive: list[Candidate] = field(default_factory=list)
    history_best_rb: list[float] = field(default_factory=list)   # gbest R_b after each population
    history_best_feas: list[bool] = field(default_factory=list)


class SwarmMemory:
    def __init__(self, problem: Problem, archive_size: int = 0):
        self.problem = problem
        self.archive_size = int(archive_size)
        self._arch: list[Candidate] = []
        self._keys: set = set()
        self.hist_rb: list[float] = []
        self.hist_feas: list[bool] = []

    # ------------------------------------------------------------------
    def init(self, z: np.ndarray, ev: EvalBatch) -> None:
        self.pz = z.copy()
        self.pf = ev.feasible.copy()
        self.pnu = ev.nu.copy()
        self.prb = ev.R_b.copy()
        self._archive_add(z, ev)
        self._record()

    def update(self, z: np.ndarray, ev: EvalBatch) -> None:
        better = is_better(ev.feasible, ev.nu, ev.R_b, self.pf, self.pnu, self.prb)
        self.pz[better] = z[better]
        self.pf[better] = ev.feasible[better]
        self.pnu[better] = ev.nu[better]
        self.prb[better] = ev.R_b[better]
        self._archive_add(z, ev)
        self._record()

    def gbest_index(self) -> int:
        return int(rank_order(self.pf, self.pnu, self.prb)[0])

    def gbest_z(self) -> np.ndarray:
        return self.pz[self.gbest_index()].copy()

    def _record(self) -> None:
        g = self.gbest_index()
        self.hist_rb.append(float(self.prb[g]))
        self.hist_feas.append(bool(self.pf[g]))

    # ------------------------------------------------------------------
    def _cand(self, z_row: np.ndarray, f, nu, rb) -> Candidate:
        mb, s, m = z_to_alloc(self.problem.sys, z_row[None, :], self.problem.K)
        return Candidate(z_row.copy(), float(mb[0]), float(s[0]), m[0].copy(), float(rb), bool(f), float(nu))

    def _archive_add(self, z: np.ndarray, ev: EvalBatch) -> None:
        if self.archive_size <= 0:
            return
        pool = self._arch + [self._cand(z[i], ev.feasible[i], ev.nu[i], ev.R_b[i]) for i in range(z.shape[0])]
        order = rank_order(np.array([c.feasible for c in pool]), np.array([c.nu for c in pool]),
                           np.array([c.R_b for c in pool]))
        new, keys = [], set()
        for i in order:
            c = pool[i]
            key = tuple(np.round(np.concatenate([[c.mb, c.s], c.m_sic]), 6))
            if key in keys:
                continue
            keys.add(key)
            new.append(c)
            if len(new) >= self.archive_size:
                break
        self._arch = new

    def result(self, n_evals: int, init_feasible: bool | None = None) -> SearchResult:
        g = self.gbest_index()
        best = self._cand(self.pz[g], self.pf[g], self.pnu[g], self.prb[g])
        return SearchResult(best=best, n_evals=n_evals, init_feasible=init_feasible,
                            archive=list(self._arch), history_best_rb=self.hist_rb,
                            history_best_feas=self.hist_feas)


def polish_population(mem: SwarmMemory, S: int, step: float | None) -> tuple[np.ndarray, float]:
    """Component P: one batched line-search population on z_b (covert fraction) around gbest.

    Shared by PSO and QPSO (same code -> fair comparison). Other coordinates of gbest are kept.
    First population (step=None): scan [z_b, 1] if gbest is feasible (push m_b up), else
    [0, z_b) (pull m_b down to regain feasibility). Later populations: +/- one previous grid step.
    Returns (Z of shape (S, D), next step). Uses no random numbers.
    """
    gi = mem.gbest_index()
    g = mem.pz[gi].copy()
    zb = float(g[0])
    if step is None:
        if bool(mem.pf[gi]) and zb < 1.0 - 1e-9:
            grid = zb + (1.0 - zb) * np.arange(1, S + 1) / S
            nxt = (1.0 - zb) / S
        elif (not bool(mem.pf[gi])) and zb > 1e-9:
            grid = zb * np.arange(0, S) / S
            nxt = zb / S
        else:                                             # at the box edge: symmetric local scan
            nxt = 1.0 / S
            grid = zb + np.linspace(-nxt * S / 2, nxt * S / 2, S)
    else:
        grid = zb + np.linspace(-step, step, S)
        nxt = 2.0 * step / max(S - 1, 1)
    # clip: zb + (1 - zb) * S / S can exceed 1 by one ulp in floating point
    grid = np.clip(np.asarray(grid, dtype=float), 0.0, 1.0)
    assert grid.shape == (S,), grid.shape
    Z = np.repeat(g[None, :], S, axis=0)
    Z[:, 0] = grid
    assert Z.shape == mem.pz.shape and np.all((Z >= 0) & (Z <= 1))
    return Z, float(nxt)


def run_polish(problem: Problem, mem: SwarmMemory, S: int, n_pops: int) -> None:
    """Spend the last n_pops populations of the budget on polish_population (updates mem)."""
    step = None
    for _ in range(int(n_pops)):
        Z, step = polish_population(mem, S, step)
        mem.update(Z, problem.evaluate_z(Z))


def check_budget(budget: int, swarm: int) -> int:
    assert budget >= swarm and budget % swarm == 0, f"budget {budget} must be a multiple of swarm {swarm}"
    return budget // swarm


WELL_STREAM = 7919          # sub-stream id for component W (keeps the main rng sequence aligned)


def reflect01(z: np.ndarray) -> np.ndarray:
    """Reflection into [0, 1] (same rule as QPSO's default boundary)."""
    y = np.mod(z, 2.0)
    return np.where(y > 1.0, 2.0 - y, y)


def well_init(z: np.ndarray, init_z: np.ndarray, seed: int, well_scale: float, well_frac: float,
              boundary: str = "reflect") -> np.ndarray:
    """Component W, shared by PSO (control gnn_pso_wp/_wpt) and QPSO (gnn_qwqpso).

    Particles 1..round(f*(S-1)) are redrawn as z_hat +/- L*ln(1/u) (u ~ U(0,1], sign 1/2),
    then reflected (or clipped) into [0,1]. Draws come from the separate stream
    (seed, WELL_STREAM), so the main rng sequence of the optimizer is unchanged.
    Bit-identical to the original implementation inside Quantummodel_code/qpso.py.
    """
    S, D = z.shape
    n_well = int(round(float(well_frac) * (S - 1))) if well_scale > 0.0 else 0
    assert 0 <= n_well <= S - 1, n_well
    if n_well == 0:
        return z
    rng_w = np.random.default_rng([int(seed), WELL_STREAM])
    u_w = 1.0 - rng_w.random((n_well, D))
    s_w = np.where(rng_w.random((n_well, D)) < 0.5, -1.0, 1.0)
    y = init_z[None, :] + s_w * float(well_scale) * np.log(1.0 / u_w)
    z = z.copy()
    z[1:1 + n_well] = np.clip(y, 0.0, 1.0) if boundary == "clip" else reflect01(y)
    assert z.shape == (S, D) and np.all((z >= 0) & (z <= 1))
    return z
