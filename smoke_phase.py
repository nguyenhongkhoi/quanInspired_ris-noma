"""Smoke test for the RIS-phase extension (common/phase.py). Runs on the RUNNER machine (~1-2 min).

Checks:
  1. build_bank(phase_offset=0) == build_bank() (same draws, paper model untouched);
  2. PhaseProblem at Delta = 0 (z_phase = 0.5) == Problem.evaluate_alloc (R_b, xi_Gamma, feasibility);
  3. fast gains at a random Delta == build_bank(phase_offset=Delta) (search path == assessment path);
  4. one 30-particle population timing; QW-QPSO 120 from the fair prior and PSO 150 run, keep the
     prior when nothing better is found, return D = K+1+N coordinates.
    python smoke_phase.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from baseline_code.pso import run_pso  # noqa: E402
from common.channel import bank_seed, build_bank, generate_layout, layout_ids, lhs_pairs  # noqa: E402
from common.config import SysConfig  # noqa: E402
from common.evaluator import Problem, alloc_to_z  # noqa: E402
from common.phase import PhaseProblem, z_to_delta  # noqa: E402
from common.prior import fair_prior  # noqa: E402
from Quantummodel_code.qpso import run_qpso  # noqa: E402

fails = 0


def check(name: str, ok: bool, val) -> None:
    global fails
    print(f"{'PASS' if ok else 'FAIL'}  {name}  value={val}", flush=True)
    fails += 0 if ok else 1


sys_ = SysConfig()
SEED, KMAX, S, N = 2026, 10, 30, SysConfig().N
for lid in layout_ids("val", 2):
    lay = generate_layout(sys_, SEED, lid, "val", KMAX)
    P_tot, xi = (float(v) for v in lhs_pairs(sys_, SEED, lid, 1)[0])
    for K in (2, 6, 10):
        sd = bank_seed(SEED, "search", lid, K)
        b0 = build_bank(sys_, lay, K, 10_000, sd)
        bz = build_bank(sys_, lay, K, 10_000, sd, phase_offset=np.zeros(N))
        braw = build_bank(sys_, lay, K, 10_000, sd, keep_raw=True)
        check(f"L{lid} K{K} offset 0 == paper bank", np.array_equal(b0.X, bz.X) and np.array_equal(b0.Y, bz.Y)
              and np.array_equal(b0.X, braw.X), f"max|dX|={np.abs(b0.X - bz.X).max():.1e}")
        prob0 = Problem(sys_, b0, P_tot, xi, "search")
        prob = PhaseProblem(sys_, braw, P_tot, xi, "search")
        mb, s, m, n = fair_prior(prob, 0.2)
        z0 = np.concatenate([alloc_to_z(sys_, mb, s, m[None, :])[0], np.full(N, 0.5)])
        e0 = prob0.evaluate_alloc([mb], [s], m[None, :])
        e1 = prob.evaluate_z(z0[None, :])
        check(f"L{lid} K{K} Delta=0 == paper evaluator",
              abs(e0.R_b[0] - e1.R_b[0]) < 1e-9 and abs(e0.xi_gamma[0] - e1.xi_gamma[0]) < 1e-9
              and bool(e0.feasible[0]) == bool(e1.feasible[0]),
              f"R_b {e0.R_b[0]:.6f}/{e1.R_b[0]:.6f} xi {e0.xi_gamma[0]:.6f}/{e1.xi_gamma[0]:.6f} "
              f"feas {bool(e0.feasible[0])}/{bool(e1.feasible[0])} prior evals={n}")
        rng = np.random.default_rng(lid * 100 + K)
        zp = rng.random(N)
        Xf, Yf = prob.gains(zp[None, :])
        bd = build_bank(sys_, lay, K, 10_000, sd, phase_offset=z_to_delta(zp))
        err = max(np.abs(Xf[0] - bd.X).max() / bd.X.max(), np.abs(Yf[0] - bd.Y).max() / bd.Y.max())
        check(f"L{lid} K{K} fast gains == assessment-path bank", err < 1e-9,
              f"rel err={err:.1e} gBob={Xf[0, :, 0].mean() / b0.X[:, 0].mean():.3f} "
              f"gWillie={Xf[0, :, K + 1].mean() / b0.X[:, K + 1].mean():.3f}")
        Z = np.concatenate([np.tile(z0[:K + 1], (S, 1)), rng.random((S, N))], axis=1)
        t0 = time.perf_counter()
        prob.evaluate_z(Z)
        tp = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        rq = run_qpso(prob, 120, S, seed=123, init_z=z0, beta_start=0.5, beta_end=0.25, polish_pops=0,
                      well_scale=0.05, well_frac=1.0)
        tq = time.perf_counter() - t0
        t0 = time.perf_counter()
        rp = run_pso(prob, 150, S, seed=123)
        tps = time.perf_counter() - t0
        ok = (rq.best.z.shape == (prob.D,) and rp.best.z.shape == (prob.D,)
              and (not e1.feasible[0] or (rq.best.feasible and rq.best.R_b >= e1.R_b[0] - 1e-12)))
        check(f"L{lid} K{K} swarms over D={prob.D}", ok,
              f"pop={tp:.0f} ms | qpso120 {tq:.1f}s feas={rq.best.feasible} R_b={rq.best.R_b:.4f} "
              f"(prior {e1.R_b[0]:.4f}) |Delta|rms={np.sqrt(np.mean(z_to_delta(rq.best.z[K + 1:]) ** 2)):.3f} "
              f"| pso150 {tps:.1f}s feas={rp.best.feasible} R_b={rp.best.R_b:.4f}")
print(f"=== smoke_phase: {'ALL PASS' if fails == 0 else f'{fails} FAIL'}")
sys.exit(1 if fails else 0)
