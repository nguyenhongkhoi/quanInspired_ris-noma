"""Quantum-behaved PSO (Sun, Feng, Xu, CEC 2004) — quantum-INSPIRED, classical CPU code.

Same coordinates, evaluator, ranking, swarm size and budget definition as PSO
(claude.md 5.7). Only the position update differs:

    mbest = mean_i p_i                                  (mbest_mode="mean")
    P     = phi * p_i + (1 - phi) * g                  (phi ~ U(0,1) per dim)
    P     = (1 - w_a) P + w_a z_hat                    (gnn_aqpso only)
    z     = P +/- beta |mbest - z| ln(1/u)              (u ~ U(0,1], sign 1/2)

gnn_qqpso adds three problem-specific components (each switchable, all default
OFF so that run_qpso(...) with defaults is exactly the plain QPSO above):

  F  mbest_mode="feas_rank": mbest = rank-weighted mean (weights 1.5 -> 0.5,
     WQPSO-style) over the FEASIBLE personal bests only (all pbests if none is
     feasible). Infeasible particles no longer drag the swarm centre.
  B  beta_mode="feas": beta_t = beta_end + (beta_lin_t - beta_end) * (1 - f_t),
     f_t = fraction of feasible pbests. Wide search while the swarm is mostly
     infeasible, contraction once feasibility is reached.
  P  polish_pops=n: the LAST n populations of the budget are a batched line
     search on z_b (covert fraction m_b) around gbest, other coordinates fixed.
     R_b increases with m_b while DEP, public rates and SIC margins all tighten,
     so the best allocation sits on the feasibility boundary in m_b. Population 1
     scans [z_b, 1] if gbest is feasible (push m_b up) or [0, z_b) if not (pull
     m_b down to regain feasibility); later populations refine +/- one grid step.
     Uses evaluations from the same budget (B = S*T unchanged), no extra draws.
     Implemented once in common/swarm.py (run_polish) and shared with PSO (gnn_pso_p).

gnn_qwqpso adds a QPSO-specific initialisation:
  W  well_scale=L, well_frac=f: particles 1..round(f*(S-1)) are drawn from the QPSO
     delta-potential-well distribution centred on the learned prediction z_hat:
         z = z_hat +/- L * ln(1/u),  u ~ U(0,1], sign 1/2   (reflected into [0,1])
     i.e. the same heavy-tailed law QPSO uses in its update, with the GNN output as the
     attractor. Draws come from a SEPARATE stream (seed, WELL_STREAM) so the main rng
     sequence is identical to gnn_qpso. Particle 0 stays exactly z_hat.

prior_dqpso restricts the search box (zb_max < 1): z_b in [0, zb_max], where zb_max is the
DEP/QoS necessary bound on m_b (common/prior.py zb_cap, 0 objective evaluations). Uniform
particles are scaled into the box (same draws), the well and every update are reflected into
[0, hi], and the polish scans inside the box. zb_max = 1 is exactly the code path above.

Pairing: the initial swarm is drawn exactly like PSO (z then an unused velocity
draw), so for a given seed QPSO and PSO start from the same 30 particles; all
later draws are made in the same order with or without a learned particle.
"""
from __future__ import annotations

import numpy as np

from common.evaluator import Problem, rank_order
from common.swarm import SearchResult, SwarmMemory, box_hi, check_budget, run_polish, to_box

MBEST_MODES = ("mean", "rank", "feas_rank")
WELL_STREAM = 7919                      # sub-stream id for component W (keeps main rng aligned)
BETA_MODES = ("linear", "feas")


def _boundary(z: np.ndarray, mode: str) -> np.ndarray:
    if mode == "clip":
        return np.clip(z, 0.0, 1.0)
    y = np.mod(z, 2.0)                       # reflection into [0, 1]
    return np.where(y > 1.0, 2.0 - y, y)


def _rank_weights(n: int) -> np.ndarray:
    w = np.linspace(1.5, 0.5, n) if n > 1 else np.ones(1)
    return w / w.sum()


def _mbest(mem: SwarmMemory, mode: str) -> np.ndarray:
    if mode == "mean":
        return mem.pz.mean(axis=0)
    order = rank_order(mem.pf, mem.pnu, mem.prb)          # best first
    if mode == "feas_rank":
        feas_order = order[mem.pf[order]]
        if feas_order.size > 0:
            order = feas_order
    w = _rank_weights(order.size)
    out = (w[:, None] * mem.pz[order]).sum(axis=0)
    assert out.shape == (mem.pz.shape[1],)
    return out


def run_qpso(problem: Problem, budget: int, swarm: int, seed: int,
             init_z: np.ndarray | None = None, archive_size: int = 0,
             beta_start: float = 1.0, beta_end: float = 0.5, boundary: str = "reflect",
             anchor_z: np.ndarray | None = None, anchor_w0: float = 0.0,
             anchor_frac: float = 0.0, anchor_gate: bool = True,
             mbest_mode: str = "mean", beta_mode: str = "linear", polish_pops: int = 0,
             well_scale: float = 0.0, well_frac: float = 0.0, zb_max: float = 1.0,
             v_init: float = 0.1, **_unused) -> SearchResult:
    assert boundary in ("reflect", "clip")
    assert mbest_mode in MBEST_MODES, mbest_mode
    assert beta_mode in BETA_MODES, beta_mode
    S = int(swarm)
    T = check_budget(budget, S)
    polish_pops = int(polish_pops)
    assert 0 <= polish_pops <= T - 1, f"polish_pops={polish_pops} needs budget > {polish_pops}*swarm"
    D = problem.K + 1
    rng = np.random.default_rng(seed)
    z = rng.random((S, D))
    _ = rng.uniform(-v_init, v_init, (S, D))   # same draw as PSO velocity init (pairing)
    hi = box_hi(D, zb_max)                     # None -> unit box (original path)
    if hi is not None:
        z = z * hi[None, :]
    if init_z is not None:
        init_z = np.asarray(init_z, float).reshape(D)
        if hi is not None:
            assert init_z[0] <= hi[0] + 1e-9, (init_z[0], hi[0])
        z[0] = init_z
        n_well = int(round(float(well_frac) * (S - 1))) if well_scale > 0.0 else 0
        assert 0 <= n_well <= S - 1, n_well
        if n_well > 0:                                    # component W: quantum-well initialisation
            rng_w = np.random.default_rng([int(seed), WELL_STREAM])
            u_w = 1.0 - rng_w.random((n_well, D))          # (0, 1]
            s_w = np.where(rng_w.random((n_well, D)) < 0.5, -1.0, 1.0)
            y_w = init_z[None, :] + s_w * float(well_scale) * np.log(1.0 / u_w)
            z[1:1 + n_well] = _boundary(y_w, boundary) if hi is None else to_box(y_w, hi, boundary)
            assert z.shape == (S, D) and np.all((z >= 0) & (z <= 1))
    ev = problem.evaluate_z(z)
    init_feas = bool(ev.feasible[0]) if init_z is not None else None
    mem = SwarmMemory(problem, archive_size)
    mem.init(z, ev)

    n_upd = T - 1 - polish_pops                  # QPSO position updates
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
        mbest = _mbest(mem, mbest_mode)
        P = phi * mem.pz + (1.0 - phi) * g[None, :]
        if t < n_anchor:
            w_a = anchor_w0 * (1.0 - t / n_anchor)
            P = (1.0 - w_a) * P + w_a * anchor_z[None, :]
        beta = float(betas[t])
        if beta_mode == "feas":
            f_t = float(np.mean(mem.pf))
            beta = beta_end + (beta - beta_end) * (1.0 - f_t)
        z = P + sign * beta * np.abs(mbest[None, :] - z) * np.log(1.0 / u)
        z = _boundary(z, boundary) if hi is None else to_box(z, hi, boundary)
        ev = problem.evaluate_z(z)
        mem.update(z, ev)

    run_polish(problem, mem, S, polish_pops, zb_max)    # component P (common/swarm.py, shared with PSO)

    res = mem.result(n_evals=S * T, init_feasible=init_feas)
    assert len(res.history_best_rb) == T
    return res
