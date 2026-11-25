"""Smoke test for the fair prior (branch feat/prior-fair). Runs on the RUNNER machine (~1 min).

Checks on a few Set-B-like validation cases (paper seed, K_max = 10):
  * fair_prior is search-feasible whenever mb_min is, uses <= 14 evaluations, m_b <= dep_prior's;
  * boxed QPSO and boxed PSO run from it with B - S = 120 evaluations and stay inside the box.
Prints PASS/FAIL lines; exit code != 0 on failure.
    python smoke_fair.py
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
from common.prior import FAIR_BISECT_ITERS, dep_prior, equal_ceiling_split, fair_prior, zb_cap  # noqa: E402
from Quantummodel_code.qpso import run_qpso  # noqa: E402

fails = 0


def check(name: str, ok: bool, val) -> None:
    global fails
    print(f"{'PASS' if ok else 'FAIL'}  {name}  value={val}")
    fails += 0 if ok else 1


sys_ = SysConfig()
SEED, KMAX, S, B = 2026, 10, 30, 150
for lid in layout_ids("val", 2):
    lay = generate_layout(sys_, SEED, lid, "val", KMAX)
    P_tot, xi = (float(v) for v in lhs_pairs(sys_, SEED, lid, 1)[0])
    for K in (2, 6, 8, 10):
        bank = build_bank(sys_, lay, K, 10_000, bank_seed(SEED, "search", lid, K))
        prob = Problem(sys_, bank, P_tot, xi, "search")
        mb_d, _, m_d = dep_prior(prob, 0.2)
        ev_d = prob.evaluate_alloc([mb_d], [0.2], m_d[None, :])
        t0 = time.perf_counter()
        mb, s, m, n = fair_prior(prob, 0.2)
        ms = (time.perf_counter() - t0) * 1e3
        ev = prob.evaluate_alloc([mb], [s], m[None, :])
        m_lo = equal_ceiling_split(sys_, K, sys_.mb_min)
        lo_feas = bool(prob.evaluate_alloc([sys_.mb_min], [s], m_lo[None, :]).feasible[0])
        check(f"L{lid} K{K} fair_prior", (bool(ev.feasible[0]) or not lo_feas)
              and 1 <= n <= 2 + FAIR_BISECT_ITERS and mb <= mb_d + 1e-12,
              f"dep: mb={mb_d:.4f} feas={bool(ev_d.feasible[0])} R_b={ev_d.R_b[0]:.4f} R_pub_min={ev_d.R_pub.min():.4f}"
              f" | fair: mb={mb:.4f} feas={bool(ev.feasible[0])} R_b={ev.R_b[0]:.4f} R_pub_min={ev.R_pub.min():.4f}"
              f" n_evals={n} ({ms:.0f} ms)")
        z0 = alloc_to_z(sys_, mb, s, m[None, :])[0]
        zc = zb_cap(prob, 0.2)
        rq = run_qpso(prob, B - S, S, seed=123, init_z=z0, beta_start=0.5, beta_end=0.25,
                      polish_pops=0, well_scale=0.05, well_frac=1.0, zb_max=zc)
        rp = run_pso(prob, B - S, S, seed=123, init_z=z0, polish_pops=0, well_scale=0.05,
                     well_frac=1.0, zb_max=zc)
        ok = (rq.n_evals == B - S and rp.n_evals == B - S and rq.best.z[0] <= zc + 1e-9
              and rp.best.z[0] <= zc + 1e-9
              and (not ev.feasible[0] or (rq.best.feasible and rp.best.feasible
                                          and rq.best.R_b >= ev.R_b[0] - 1e-12 and rp.best.R_b >= ev.R_b[0] - 1e-12)))
        check(f"L{lid} K{K} boxed swarms from fair prior", ok,
              f"qpso feas={rq.best.feasible} R_b={rq.best.R_b:.4f} | pso feas={rp.best.feasible} R_b={rp.best.R_b:.4f}"
              f" | prior R_b={ev.R_b[0]:.4f}")
print(f"=== smoke_fair: {'ALL PASS' if fails == 0 else f'{fails} FAIL'}")
sys.exit(1 if fails else 0)
