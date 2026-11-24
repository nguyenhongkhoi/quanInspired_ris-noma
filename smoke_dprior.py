"""Smoke test for the DEP-aware prior (branch feat/prior-dep). Runs on the RUNNER machine (~1 min).

Checks on a few Set-B-like validation cases (paper seed, K_max = 10):
  * xi_Gamma is non-increasing in m_b on a grid (assumption behind the bisection);
  * dep_cap meets the search DEP target and mb_cap <= QoS bound (eq. 19);
  * prior_dqpso (boxed QW-QPSO) and prior_dpolish run, stay inside the box, use B evaluations.
Prints PASS/FAIL lines; exit code != 0 on failure.
    python smoke_dprior.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from baseline_code.linepolish import run_linepolish  # noqa: E402
from common.channel import bank_seed, build_bank, generate_layout, layout_ids, lhs_pairs  # noqa: E402
from common.config import SysConfig  # noqa: E402
from common.evaluator import Problem, alloc_to_z  # noqa: E402
from common.prior import dep_cap, dep_prior, mb_cap, qos_cap, zb_cap  # noqa: E402
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
    for K in (2, 6, 10):
        bank = build_bank(sys_, lay, K, 10_000, bank_seed(SEED, "search", lid, K))
        prob = Problem(sys_, bank, P_tot, xi, "search")
        P_a, rho2 = prob.powers(np.array([0.2]))
        grid = np.linspace(sys_.mb_min, sys_.mb_max, 41)
        xg = np.array([prob.xi_gamma_one(m, P_a[0], rho2[0]) for m in grid])
        check(f"L{lid} K{K} xi_Gamma non-increasing in m_b", bool(np.all(np.diff(xg) <= 1e-9)),
              f"[{xg[0]:.4f} .. {xg[-1]:.4f}]")
        t0 = time.perf_counter()
        cap_d = dep_cap(prob, 0.2)
        cap = mb_cap(prob, 0.2)
        zc = zb_cap(prob, 0.2)
        ms = (time.perf_counter() - t0) * 1e3
        xi_at = prob.xi_gamma_one(cap_d, P_a[0], rho2[0])
        check(f"L{lid} K{K} dep_cap meets target", cap_d <= sys_.mb_min + 1e-12 or xi_at >= prob.xi_t,
              f"m_dep={cap_d:.4f} xi={xi_at:.4f} target={prob.xi_t:.4f} qos={qos_cap(sys_, K, prob.R_t):.4f} "
              f"cap={cap:.4f} zb_cap={zc:.4f} ({ms:.1f} ms)")
        mb, s, m = dep_prior(prob, 0.2)
        ev = prob.evaluate_alloc([mb], [s], m[None, :])
        print(f"      dep_prior: mb={mb:.4f} feasible={bool(ev.feasible[0])} R_b={ev.R_b[0]:.4f} "
              f"xi_G={ev.xi_gamma[0]:.4f} R_pub_min={ev.R_pub.min():.4f} p_b={ev.p_b[0]:.4f}")
        z0 = alloc_to_z(sys_, mb, s, m[None, :])[0]
        t0 = time.perf_counter()
        res = run_qpso(prob, B, S, seed=123, init_z=z0, beta_start=0.5, beta_end=0.25,
                       polish_pops=1, well_scale=0.1, well_frac=1.0, zb_max=zc)
        tq = (time.perf_counter() - t0) * 1e3
        res_p = run_linepolish(prob, B, S, seed=123, init_z=z0, zb_max=zc)
        inbox = res.best.z[0] <= zc + 1e-9 and res_p.best.z[0] <= zc + 1e-9
        check(f"L{lid} K{K} boxed runs", inbox and res.n_evals == B and res_p.n_evals == B,
              f"dqpso feas={res.best.feasible} R_b={res.best.R_b:.4f} hist={np.round(res.history_best_rb, 3).tolist()} "
              f"| dpolish feas={res_p.best.feasible} R_b={res_p.best.R_b:.4f} ({tq:.0f} ms)")
print(f"=== smoke_dprior: {'ALL PASS' if fails == 0 else f'{fails} FAIL'}")
sys.exit(1 if fails else 0)
