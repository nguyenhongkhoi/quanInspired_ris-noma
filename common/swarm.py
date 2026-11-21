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


def check_budget(budget: int, swarm: int) -> int:
    assert budget >= swarm and budget % swarm == 0, f"budget {budget} must be a multiple of swarm {swarm}"
    return budget // swarm
