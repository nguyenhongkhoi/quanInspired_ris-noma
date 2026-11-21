"""Independent feasibility assessment (claude.md 3.7, Appendix A).

The returned allocation is fixed, then evaluated on an assessment bank that was
never seen by the search. The SIC order and eligibility come from the input
(search) bank of the case.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import beta as beta_dist

from .config import SysConfig
from .evaluator import Problem

POINT_FLAGS = ("pt_xi_emp", "pt_xi_gamma", "pt_rate", "pt_outage", "pt_domain", "pt_sic", "pt_eligible")
BOUND_FLAGS = ("bd_xi", "bd_rate", "bd_outage")


def empirical_dep(Z0: np.ndarray, Z1: np.ndarray) -> float:
    """xi_emp = 1 - sup_{t>=0} (F0(t) - F1(t)); thresholds = ECDF jump points and t = 0.

    Detector declares H1 when Z > t (strict), so F_i(t) = Pr{Z_i <= t}.
    """
    Z0 = np.sort(np.asarray(Z0, dtype=float))
    Z1 = np.sort(np.asarray(Z1, dtype=float))
    t = np.concatenate([Z0, Z1, [0.0]])
    t = t[t >= 0.0]
    F0 = np.searchsorted(Z0, t, side="right") / Z0.size
    F1 = np.searchsorted(Z1, t, side="right") / Z1.size
    sup = max(0.0, float(np.max(F0 - F1)))
    return 1.0 - sup


def eps_dkw(sys: SysConfig, M: int) -> float:
    return math.sqrt(math.log(12.0 * sys.J_family / sys.alpha_g) / (2.0 * M))


def hoeffding_width(sys: SysConfig, M: int, K: int) -> float:
    return math.sqrt(math.log(3.0 * sys.J_family * K / sys.alpha_g) / (2.0 * M))


def clopper_pearson_upper(sys: SysConfig, F: int, M: int) -> float:
    if F >= M:
        return 1.0
    q = 1.0 - sys.alpha_g / (3.0 * sys.J_family)
    return float(beta_dist.ppf(q, F + 1, M - F))


def assess_allocation(sys: SysConfig, prob_assess: Problem, mb: float, s: float,
                      m_sic: np.ndarray) -> dict:
    """Assess one fixed allocation on `prob_assess` (mode='assess').

    Returns a flat dict with point/bound flags and statistics (trials.csv fields).
    """
    assert prob_assess.mode == "assess"
    K = prob_assess.K
    M = prob_assess.M
    m_sic = np.asarray(m_sic, dtype=float).reshape(K)
    ev = prob_assess.evaluate_alloc(np.array([mb]), np.array([s]), m_sic[None, :])
    R_pub = ev.R_pub[0]
    R_b = float(ev.R_b[0])
    F_b = int(ev.n_out[0])
    p_b = float(ev.p_b[0])
    xi_gamma = float(ev.xi_gamma[0])
    A = ev.A[0]

    Z0, Z1 = prob_assess.willie_samples(mb, s)
    xi_emp = empirical_dep(Z0, Z1)

    eps = eps_dkw(sys, M)
    xi_L = max(0.0, xi_emp - 2.0 * eps)
    c_j = np.log2(1.0 + m_sic / A)
    R_L = np.maximum(0.0, R_pub - c_j * hoeffding_width(sys, M, K))
    p_bU = clopper_pearson_upper(sys, F_b, M)

    tol = 1e-9
    domain_ok = bool(sys.mb_min - tol <= mb <= sys.mb_max + tol
                     and sys.s_min - tol <= s <= sys.s_max + tol
                     and np.all(m_sic >= -tol)
                     and abs(mb + m_sic.sum() - 1.0) <= 1e-7)
    pt = dict(
        pt_xi_emp=xi_emp >= prob_assess.xi_req,
        pt_xi_gamma=xi_gamma >= prob_assess.xi_req,
        pt_rate=bool(np.all(R_pub >= sys.R_qos)),
        pt_outage=p_b <= sys.p_max,
        pt_domain=domain_ok,
        pt_sic=bool(ev.margins_ok[0]),
        pt_eligible=prob_assess.eligible,
    )
    bd = dict(
        bd_xi=xi_L >= prob_assess.xi_req,
        bd_rate=bool(np.all(R_L >= sys.R_qos)),
        bd_outage=p_bU <= sys.p_max,
    )
    pt_all = all(pt.values())
    bd_all = all(bd.values())
    passed = pt_all and bd_all
    category = "pass" if passed else ("point_only" if pt_all else "point_fail")
    out = dict(
        R_b=R_b, R_pub_min=float(R_pub.min()), R_L_min=float(R_L.min()),
        p_b=p_b, F_b=F_b, p_bU=p_bU, xi_emp=xi_emp, xi_gamma=xi_gamma, xi_L=xi_L,
        passed=bool(passed), category=category,
        mb_gt_ceiling=bool(mb > sys.mb_ceiling(K)),
        rb_ceiling=float(math.log2(1.0 + mb / sys.kappa ** 2)),
        eligible=prob_assess.eligible,
    )
    out.update({k: bool(v) for k, v in pt.items()})
    out.update({k: bool(v) for k, v in bd.items()})
    return out
