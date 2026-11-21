"""Sanity checks for the shared physical model (claude.md 11).

Run on the RUN MACHINE (not on the coding machine):
    python sanity_check.py --out-dir sanity
Writes ./result/<out-dir>/sanity.csv and exits with code 1 if any check FAILs.

Status values: PASS / FAIL (hard checks), WARN (soft, machine-dependent), INFO.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from common.assessment import (assess_allocation, clopper_pearson_upper,  # noqa: E402
                               empirical_dep, eps_dkw)
from common.channel import build_bank, generate_layout  # noqa: E402
from common.cli import resolve_run_dir  # noqa: E402
from common.config import SysConfig, make_rng, stable_seed  # noqa: E402
from common.evaluator import (Problem, alloc_to_z, gamma_params, is_better,  # noqa: E402
                              rank_order, to_original, to_sic, z_to_alloc)
from scipy.special import gammainc  # noqa: E402

SANITY_SEED = 31415
K_MAX = 6
ROWS: list[dict] = []


def report(name: str, status: str, value, expected: str, note: str = "") -> None:
    if isinstance(value, float):
        vtxt = f"{value:.6g}"
    else:
        vtxt = str(value)
    line = f"{status:<4}  {name}  gia_tri={vtxt}  ky_vong={expected}"
    if note:
        line += f"  ({note})"
    print(line, flush=True)
    ROWS.append(dict(name=name, status=status, value=vtxt, expected=expected, note=note))


def check(name: str, ok: bool, value, expected: str, note: str = "", soft: bool = False) -> None:
    report(name, "PASS" if ok else ("WARN" if soft else "FAIL"), value, expected, note)


def layout(sys_: SysConfig, lid: int):
    return generate_layout(sys_, SANITY_SEED, lid, "sanity", K_MAX)


def bank_for(sys_: SysConfig, lid: int, K: int, M: int, kind: str = "search"):
    return build_bank(sys_, layout(sys_, lid), K, M, stable_seed(SANITY_SEED, kind, lid, K))


# ---------------------------------------------------------------------------
# Independent brute-force references (written directly from eqs. 12-15)
# ---------------------------------------------------------------------------
def brute_U(sys_: SysConfig, bank, P_tot_dbm: float, s: float, col: int) -> np.ndarray:
    P0 = sys_.P0(P_tot_dbm)
    Pa, Pr = (1 - s) * P0, s * P0
    rho2 = Pr / (sys_.N * (Pa / bank.L_ar + sys_.sigma_r2))
    g = rho2 * bank.X[:, col] / (bank.L_ar * bank.L_ru[col])
    q = rho2 * bank.Y[:, col] / bank.L_ru[col]
    return Pa * g / (sys_.sigma_r2 * q + sys_.sigma_u2)


def brute_rates_outage(sys_: SysConfig, prob: Problem, P_tot_dbm, mb, s, m_sic):
    bank = prob.bank
    K = bank.K
    k2 = sys_.kappa ** 2
    U_b = brute_U(sys_, bank, P_tot_dbm, s, 0)
    U_pub = [brute_U(sys_, bank, P_tot_dbm, s, 1 + prob.sic_order[j]) for j in range(K)]
    R = np.zeros(K)
    ok = np.ones(bank.M, dtype=bool)
    for j in range(K):
        A_j = mb + m_sic[j + 1:].sum() + k2
        decoders = [U_b] + U_pub[j:]
        r = np.min(np.stack([np.log2(1 + m_sic[j] * U / (A_j * U + 1)) for U in decoders]), axis=0)
        R[j] = r.mean()
        ok &= (m_sic[j] * U_b / (A_j * U_b + 1)) >= sys_.Gamma_pub
    gb = mb * U_b / (k2 * U_b + 1)
    ok &= gb >= sys_.Gamma_bob
    R_b = np.log2(1 + gb).mean()
    return R, R_b, 1.0 - ok.mean(), U_b


def brute_empirical_dep(Z0, Z1) -> float:
    ts = np.concatenate([[0.0], np.unique(np.concatenate([Z0, Z1]))])
    best = 0.0
    for t in ts:
        best = max(best, np.mean(Z0 <= t) - np.mean(Z1 <= t))
    return 1.0 - best


def grid_gamma_dep(k0, th0, k1, th1, n=400_001) -> float:
    lo = min(k0 * th0, k1 * th1) * 1e-6
    hi = max(k0 * th0, k1 * th1) * 60.0
    t = np.geomspace(lo, hi, n)
    return float(np.min(1.0 - gammainc(k0, t / th0) + gammainc(k1, t / th1)))


def dense(i: int, o: int) -> int:
    return i * o + o


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Sanity checks (claude.md 11)")
    ap.add_argument("--out-dir", default="sanity", help="run name (-> ./result/<name>) or path")
    ap.add_argument("--fast", action="store_true", help="smaller banks for the Delta/CV checks")
    args = ap.parse_args()
    out_dir = resolve_run_dir(args.out_dir, "sanity")
    out_dir.mkdir(parents=True, exist_ok=True)
    sys_ = SysConfig()
    t_start = time.perf_counter()
    print(f"[sanity] out_dir={out_dir}")

    # ---------------- A. closed-form constants --------------------------------
    check("alpha_g", abs(sys_.alpha_g - 8.9286e-5) < 1e-8, sys_.alpha_g, "8.9286e-5")
    two_eps = 2 * eps_dkw(sys_, 100_000)
    check("2*eps_M (M=1e5)", abs(two_eps - 0.01869) < 5e-5, two_eps, "~0.01869")
    pbu0 = clopper_pearson_upper(sys_, 0, 100_000)
    check("p_bU (F_b=0, M=1e5)", abs(pbu0 - 1.61e-4) < 0.01e-4, pbu0, "~1.61e-4")
    c5, c6 = sys_.mb_ceiling(5), sys_.mb_ceiling(6)
    check("m_b ceiling K=5", abs(c5 - 0.49500) < 5e-5, c5, "0.49500")
    check("m_b ceiling K=6", abs(c6 - 0.42963) < 5e-5, c6, "0.42963")
    check("P0(30 dBm) > 0", sys_.P0(30.0) > 0, sys_.P0(30.0), "> 0 W")
    gnn = (dense(24, 32) + dense(47, 32)
           + 2 * (dense(66, 32) + dense(32, 32) + dense(96, 32) + dense(32, 32))
           + dense(96, 32) + dense(32, 2) + dense(128, 32) + dense(32, 1))
    mlp = (dense(24, 32) + dense(47, 32) + 2 * (dense(32, 112) + dense(112, 32))
           + dense(96, 32) + dense(32, 2) + dense(128, 32) + dense(32, 1))
    check("GNN params (Table VI, analytic)", gnn == 24387, gnn, "24387", "model thuc se kiem khi co models.py")
    check("MLP params (Table VI, analytic)", mlp == 24291, mlp, "24291", "model thuc se kiem khi co models.py")
    try:
        from common.models import HAS_TORCH
        if HAS_TORCH:
            from common.models import AllocNet, count_params
            ng, nm = count_params(AllocNet("gnn", sys_)), count_params(AllocNet("mlp", sys_))
            check("GNN params (torch model)", ng == 24387, ng, "24387")
            check("MLP params (torch model)", nm == 24291, nm, "24291")
        else:
            report("torch model params", "WARN", "no torch", "24387 / 24291", "PyTorch chua cai")
    except Exception as exc:  # noqa: BLE001
        report("torch model params", "FAIL", repr(exc), "24387 / 24291")

    # ---------------- B. determinism ------------------------------------------
    s1 = stable_seed(2026, "search", 1000, 4)
    s2 = stable_seed(2026, "search", 1000, 4)
    check("stable_seed deterministic", s1 == s2 and s1 != stable_seed(2026, "assess", 1000, 4), s1, "same tuple -> same seed")
    b1 = bank_for(sys_, 9000, 3, 3000)
    b2 = bank_for(sys_, 9000, 3, 3000)
    check("bank reproducible", np.array_equal(b1.X, b2.X) and np.array_equal(b1.Y, b2.Y), True, "identical arrays")

    # ---------------- C. coordinate maps --------------------------------------
    rng = make_rng(SANITY_SEED, "coords")
    K = 4
    z = rng.random((50, K + 1))
    mb, s, m = z_to_alloc(sys_, z, K)
    sums = mb + m.sum(axis=1)
    mb2, s2_, m2 = z_to_alloc(sys_, alloc_to_z(sys_, mb, s, m), K)
    err = max(np.abs(mb - mb2).max(), np.abs(s - s2_).max(), np.abs(m - m2).max())
    check("z -> alloc -> z -> alloc", err < 1e-12 and np.abs(sums - 1).max() < 1e-12, err, "< 1e-12")
    order = rng.permutation(K)
    err2 = np.abs(to_sic(to_original(m, order), order) - m).max()
    check("SIC <-> original identities", err2 == 0.0, err2, "0")

    # ---------------- D. rates / outage vs brute force ------------------------
    P_tot = 35.0
    bank4 = bank_for(sys_, 9001, 4, 4000)
    prob4 = Problem(sys_, bank4, P_tot, 0.85, mode="search")
    z = rng.random((6, 5))
    mb, s, m = z_to_alloc(sys_, z, 4)
    ev = prob4.evaluate_alloc(mb, s, m)
    e_r = e_b = e_p = 0.0
    for i in range(6):
        R, Rb, pb, _ = brute_rates_outage(sys_, prob4, P_tot, mb[i], s[i], m[i])
        e_r = max(e_r, np.abs(R - ev.R_pub[i]).max())
        e_b = max(e_b, abs(Rb - ev.R_b[i]))
        e_p = max(e_p, abs(pb - ev.p_b[i]))
    check("public rate suffix-min vs brute", e_r < 1e-10, e_r, "< 1e-10")
    check("Bob rate vs brute", e_b < 1e-10, e_b, "< 1e-10")
    check("outage (random alloc) vs brute", e_p <= 3 / bank4.M, e_p, "<= 3/M")

    # outage-sensitive allocation: d_2 tiny so T_b ~ median(U_b)
    bank2 = bank_for(sys_, 9002, 2, 4000)
    prob2 = Problem(sys_, bank2, P_tot, 0.85, mode="search")
    mb0, s0 = 0.05, 0.5
    U_b = brute_U(sys_, bank2, P_tot, s0, 0)
    A2 = mb0 + sys_.kappa ** 2
    eps = 1.0 / (A2 * float(np.median(U_b)))
    m2v = sys_.Gamma_pub * A2 * (1 + eps)
    m_s = np.array([1 - mb0 - m2v, m2v])
    ev2 = prob2.evaluate_alloc([mb0], [s0], m_s[None, :])
    _, _, pb_brute, _ = brute_rates_outage(sys_, prob2, P_tot, mb0, s0, m_s)
    check("outage Prop.1 vs brute (p~0.5)", abs(ev2.p_b[0] - pb_brute) <= 3 / bank2.M and 0.05 < pb_brute < 0.95,
          float(ev2.p_b[0]), f"brute={pb_brute:.4f}")
    ev3 = prob2.evaluate_alloc([0.3], [0.5], np.array([[0.699, 0.001]]))
    check("non-positive SIC margin -> P_out=1", (not ev3.margins_ok[0]) and ev3.p_b[0] == 1.0,
          float(ev3.p_b[0]), "1.0")

    # ---------------- E. empirical DEP vs brute -------------------------------
    small = bank_for(sys_, 9003, 2, 400, kind="assess")
    pa_small = Problem(sys_, small, P_tot, 0.85, mode="assess")
    e_emp = 0.0
    for mbv in (0.05, 0.15, 0.3):
        Z0, Z1 = pa_small.willie_samples(mbv, 0.5)
        e_emp = max(e_emp, abs(empirical_dep(Z0, Z1) - brute_empirical_dep(Z0, Z1)))
    check("empirical DEP vs brute ECDF", e_emp < 1e-12, e_emp, "< 1e-12")

    # ---------------- F. Gamma DEP vs dense grid ------------------------------
    e_g = 0.0
    for mbv, sv in ((0.05, 0.3), (0.1, 0.5), (0.2, 0.7), (0.45, 0.9)):
        P_a, rho2 = prob4.powers(np.array([sv]))
        a0, a1, b = prob4.willie_coeffs(mbv, P_a[0], rho2[0])
        k0, th0 = gamma_params(a0 * prob4.EXw + b * prob4.EYw, a0 ** 2 * prob4.VXw + b ** 2 * prob4.VYw)
        k1, th1 = gamma_params(a1 * prob4.EXw + b * prob4.EYw, a1 ** 2 * prob4.VXw + b ** 2 * prob4.VYw)
        xg = prob4.xi_gamma_one(mbv, P_a[0], rho2[0])
        e_g = max(e_g, abs(xg - grid_gamma_dep(k0, th0, k1, th1)))
    check("Gamma DEP (Brent) vs dense grid", e_g < 1e-6, e_g, "< 1e-6")

    # ---------------- G. ranking ------------------------------------------------
    r_ok = (bool(is_better(True, 0.0, 1.0, False, 0.0, 9.0))
            and bool(is_better(True, 0.0, 2.0, True, 0.0, 1.0))
            and bool(is_better(False, 0.1, 0.0, False, 0.2, 9.0))
            and bool(is_better(False, 0.1, 2.0, False, 0.1, 1.0))
            and not bool(is_better(True, 0.0, 1.0, True, 0.0, 1.0)))
    ro = rank_order(np.array([False, True, True, False]), np.array([0.5, 0, 0, 0.2]), np.array([9, 1, 2, 3.0]))
    check("feasibility-first ranking", r_ok and ro.tolist() == [2, 1, 3, 0], ro.tolist(), "[2, 1, 3, 0]")

    # ---------------- H. paper reference point: R_b, p_bU ------------------------
    lid_ref = 9010
    bank_s = bank_for(sys_, lid_ref, 4, 10_000)
    ps = Problem(sys_, bank_s, 35.0, 0.85, mode="search")
    mb_r, s_r = 0.2, 0.5
    m_eq = np.full(4, (1 - mb_r) / 4)
    Rb = float(ps.evaluate_alloc([mb_r], [s_r], m_eq[None, :]).R_b[0])
    ceil = math.log2(1 + mb_r / sys_.kappa ** 2)
    check("R_b @35dBm,s=.5,mb=.2,K=4", 4.38 <= Rb <= ceil + 1e-9, Rb, f"~4.3923 (<= ceiling {ceil:.4f})")
    bank_a = bank_for(sys_, lid_ref, 4, 100_000, kind="assess")
    pa = Problem(sys_, bank_a, 35.0, 0.85, mode="assess", sic_order=ps.sic_order, eligible=ps.eligible)
    res = assess_allocation(sys_, pa, mb_r, s_r, m_eq)
    if res["F_b"] == 0:
        check("p_bU on assessment (M=1e5, no outage)", abs(res["p_bU"] - 1.61e-4) < 0.01e-4, res["p_bU"], "~1.61e-4")
    else:
        report("p_bU on assessment (M=1e5)", "WARN", res["p_bU"], "~1.61e-4", f"F_b={res['F_b']} outage events")
    report("assessment example", "INFO", res["category"], "-",
           f"xi_emp={res['xi_emp']:.4f} xi_g={res['xi_gamma']:.4f} Rmin={res['R_pub_min']:.3f} eligible={res['eligible']}")

    # ---------------- I. Delta = xi_Gamma - xi_emp --------------------------------
    M_d = 20_000 if args.fast else 100_000
    deltas = []
    for lid in range(9100, 9112):
        bs = bank_for(sys_, lid, 2, 10_000)
        p_s = Problem(sys_, bs, 35.0, 0.85, mode="search")
        ba = bank_for(sys_, lid, 2, M_d, kind="assess")
        p_a = Problem(sys_, ba, 35.0, 0.85, mode="assess", sic_order=p_s.sic_order, eligible=p_s.eligible)
        for mbv in (0.05, 0.1, 0.15):
            m2_ = np.full(2, (1 - mbv) / 2)
            r = assess_allocation(sys_, p_a, mbv, 0.5, m2_)
            deltas.append(r["xi_gamma"] - r["xi_emp"])
    deltas = np.array(deltas)
    check("Delta mean (xi_G - xi_emp)", 0.0 < deltas.mean() < 0.01, float(deltas.mean()),
          "duong, co 1e-3 (bai: +0.0012..+0.0021)", f"n={deltas.size}, M={M_d}")
    check("Delta max |.|", np.abs(deltas).max() < 0.02, float(np.abs(deltas).max()), "< 0.02 (bai: <= 0.007)")

    # ---------------- J. Willie CV spread, eligibility ------------------------------
    cvs, elig = [], []
    for lid in range(9200, 9240):
        bs = bank_for(sys_, lid, 2, 5_000 if args.fast else 10_000)
        p_s = Problem(sys_, bs, 35.0, 0.85)
        cvs.append(p_s.cv_w)
        elig.append(p_s.eligible)
    cvs = np.array(cvs)
    check("Willie CV spans threshold 0.9", cvs.min() < 0.9 < cvs.max() and 0.2 < cvs.min() and cvs.max() < 1.6,
          f"[{cvs.min():.3f}, {cvs.max():.3f}]", "~[0.4, 1.1], chua 0.9", f"low-CV {int((cvs < 0.9).sum())}/40")
    report("eligible layouts (K=2)", "INFO", f"{int(np.sum(elig))}/40", "phan lon eligible")

    # ---------------- K. timing ------------------------------------------------------
    zpop = make_rng(SANITY_SEED, "timing").random((30, 5))
    ps.evaluate_z(zpop)  # warm-up
    ts = []
    for _ in range(20):
        t0 = time.perf_counter()
        ps.evaluate_z(zpop)
        ts.append(time.perf_counter() - t0)
    t_pop = 1e3 * float(np.median(ts))
    check("population 30 x 1e4 samples (ms)", t_pop <= 150.0, t_pop, "~30 ms (bai)", soft=True)
    report("Full PSO 1800 estimate (ms)", "INFO", 60 * t_pop, "~2000 ms (bai: 1991)", "60 x population; PSO chua viet")

    # ---------------- summary ----------------------------------------------------------
    with open(out_dir / "sanity.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "status", "value", "expected", "note"])
        w.writeheader()
        w.writerows(ROWS)
    n_fail = sum(r["status"] == "FAIL" for r in ROWS)
    n_warn = sum(r["status"] == "WARN" for r in ROWS)
    n_pass = sum(r["status"] == "PASS" for r in ROWS)
    print(f"[sanity] PASS={n_pass} FAIL={n_fail} WARN={n_warn}  "
          f"time={time.perf_counter() - t_start:.1f}s  csv={out_dir / 'sanity.csv'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
