"""Quantum-behaved PSO (Sun, Feng, Xu, CEC 2004) — quantum-INSPIRED, classical CPU code.

Same coordinates, evaluator, ranking, swarm size and budget definition as PSO
(claude.md 5.7). Only the position update differs:

    mbest = mean_i p_i
    P     = phi * p_i + (1 - phi) * g                  (phi ~ U(0,1) per dim)
    P     = (1 - w_a) P + w_a z_hat                    (gnn_aqpso only)
    z     = P +/- beta |mbest - z| ln(1/u)              (u ~ U(0,1], sign 1/2)

Pairing: the initial swarm is drawn exactly like PSO (z then an unused velocity
draw), so for a given seed QPSO and PSO start from the same 30 particles; all
later draws are made in the same order with or without a learned particle.
"""
from __future__ import annotations

import numpy as np

from common.evaluator import Problem
from common.swarm import SearchResult, SwarmMemory, check_budget


def _boundary(z: np.ndarray, mode: str) -> np.ndarray:
    if mode == "clip":
        return np.clip(z, 0.0, 1.0)
    y = np.mod(z, 2.0)                       # reflection into [0, 1]
    return np.where(y > 1.0, 2.0 - y, y)


def run_qpso(problem: Problem, budget: int, swarm: int, seed: int,
             init_z: np.ndarray | None = None, archive_size: int = 0,
             beta_start: float = 1.0, beta_end: float = 0.5, boundary: str = "reflect",
             anchor_z: np.ndarray | None = None, anchor_w0: float = 0.0,
             anchor_frac: float = 0.0, anchor_gate: bool = True,
             v_init: float = 0.1, **_unused) -> SearchResult:
    assert boundary in ("reflect", "clip")
    S = int(swarm)
    T = check_budget(budget, S)
    D = problem.K + 1
    rng = np.random.default_rng(seed)
    z = rng.random((S, D))
    _ = rng.uniform(-v_init, v_init, (S, D))   # same draw as PSO velocity init (pairing)
    if init_z is not None:
        init_z = np.asarray(init_z, float).reshape(D)
        z[0] = init_z
    ev = problem.evaluate_z(z)
    init_feas = bool(ev.feasible[0]) if init_z is not None else None
    mem = SwarmMemory(problem, archive_size)
    mem.init(z, ev)

    n_upd = T - 1
    betas = np.linspace(beta_start, beta_end, n_upd) if n_upd > 0 else np.array([])
    use_anchor = anchor_z is not None and anchor_w0 > 0.0 and anchor_frac > 0.0
    if use_anchor and anchor_gate:
        use_anchor = bool(init_feas)
    n_anchor = int(round(anchor_frac * n_upd)) if use_anchor else 0
    if use_anchor:
        anchor_z = np.asarray(anchor_z, float).reshape(D)

    for t in range(n_upd):
        phi = rng.random((S, D))
        u = 1.0 - rng.random((S, D))          # (0, 1]
        sign = np.where(rng.random((S, D)) < 0.5, -1.0, 1.0)
        g = mem.gbest_z()
        mbest = mem.pz.mean(axis=0)
        P = phi * mem.pz + (1.0 - phi) * g[None, :]
        if t < n_anchor:
            w_a = anchor_w0 * (1.0 - t / n_anchor)
            P = (1.0 - w_a) * P + w_a * anchor_z[None, :]
        z = P + sign * betas[t] * np.abs(mbest[None, :] - z) * np.log(1.0 / u)
        z = _boundary(z, boundary)
        ev = problem.evaluate_z(z)
        mem.update(z, ev)
    res = mem.result(n_evals=S * T, init_feasible=init_feas)
    return res
