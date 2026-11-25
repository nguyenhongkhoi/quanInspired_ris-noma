"""Physics-informed prior allocations (no learning, no objective evaluation).

1) physics_prior (K-only, case independent) -- closed form from eq. (18)-(19):

  * public fractions: equal public-rate ceilings. With A_0 = 1 + k^2 and A_K = m_b + k^2,
        A_j = (1 + k^2) rho^j,  rho = ((m_b + k^2) / (1 + k^2))^(1/K),  m_j = A_{j-1} - A_j,
    so every c_j = log2(A_{j-1}/A_j) = log2(1/rho) is the same (telescoping), in SIC order;
  * m_b: the necessary public-QoS bound evaluated at the SEARCH rate target,
        m_b = (1 + k^2) 2^{-K (R_qos + rate_margin)} - k^2, clipped to [mb_min, mb_max];
  * s: a constant chosen on validation (s_const).

2) dep_prior (case dependent, still 0 objective evaluations):
  The Gamma-approximated DEP xi_Gamma depends on the allocation only through (m_b, s)
  (H0 uses a_0 = (1 - m_b) a_1; the public split does not enter), and it decreases as m_b
  grows (the two hypotheses separate). Hence, for a given s, the largest m_b meeting the
  SEARCH covertness target is found by bisection on xi_Gamma using only the Willie
  moments cached in the Problem (no rate / outage computation over the sample bank):
        m_dep(s) = max { m_b : xi_Gamma(m_b, s) >= xi_req + xi_margin }.
  The prior is then m_b = min(m_dep(s), m_qos(K), mb_max) with the equal-ceiling public split,
  i.e. it sits on the intersection of the two necessary bounds instead of the QoS bound only.
  zb_cap() turns the same bound into an upper limit of the z_b search coordinate.

Everything is returned in SIC order.
"""
from __future__ import annotations

import numpy as np

from .config import SysConfig

DEP_BISECT_ITERS = 50


def qos_cap(sys_: SysConfig, K: int, rate_target: float) -> float:
    """Necessary public-QoS bound on m_b, eq. (19), at a given rate target."""
    k2 = sys_.kappa ** 2
    return float((1.0 + k2) * 2.0 ** (-int(K) * float(rate_target)) - k2)


def equal_ceiling_split(sys_: SysConfig, K: int, mb: float) -> np.ndarray:
    """Public fractions (SIC order) with equal rate ceilings c_j for a given m_b."""
    K = int(K)
    k2 = sys_.kappa ** 2
    rho = ((mb + k2) / (1.0 + k2)) ** (1.0 / K)
    A = (1.0 + k2) * rho ** np.arange(K + 1)
    m = A[:-1] - A[1:]
    assert m.shape == (K,) and np.all(m > 0), m
    assert abs(mb + m.sum() - 1.0) < 1e-9, (mb, m.sum())
    return m


def physics_prior(sys_: SysConfig, K: int, s_const: float) -> tuple[float, float, np.ndarray]:
    K = int(K)
    assert K >= 1, K
    mb = qos_cap(sys_, K, sys_.R_qos + sys_.rate_margin)
    mb = float(np.clip(mb, sys_.mb_min, sys_.mb_max))
    m = equal_ceiling_split(sys_, K, mb)
    s = float(np.clip(s_const, sys_.s_min, sys_.s_max))
    return mb, s, m


# ---------------------------------------------------------------------------
# DEP-aware prior (needs the case's search Problem: Willie moments + search target)
# ---------------------------------------------------------------------------
def dep_cap(prob, s: float) -> float:
    """Largest m_b in [mb_min, mb_max] with xi_Gamma(m_b, s) >= prob.xi_t (bisection).

    Returns mb_min if even mb_min violates the target (DEP-infeasible case; the swarm
    then works at the lower edge), mb_max if mb_max already meets it.
    """
    sys_ = prob.sys
    P_a, rho2 = prob.powers(np.array([float(s)]))
    pa, r2 = float(P_a[0]), float(rho2[0])

    def ok(mb: float) -> bool:
        return prob.xi_gamma_one(mb, pa, r2) >= prob.xi_t

    lo, hi = float(sys_.mb_min), float(sys_.mb_max)
    if ok(hi):
        return hi
    if not ok(lo):
        return lo
    for _ in range(DEP_BISECT_ITERS):            # invariant: ok(lo), not ok(hi)
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    assert ok(lo), lo
    return lo


def mb_cap(prob, s: float) -> float:
    """min(DEP bound at s, QoS bound eq. (19) at the search rate target, mb_max), >= mb_min."""
    sys_ = prob.sys
    c = min(dep_cap(prob, s), qos_cap(sys_, prob.K, prob.R_t), float(sys_.mb_max))
    return float(max(c, sys_.mb_min))


def dep_prior(prob, s_const: float) -> tuple[float, float, np.ndarray]:
    """DEP-aware prior: m_b on the tighter of the two necessary bounds, equal-ceiling split."""
    sys_ = prob.sys
    s = float(np.clip(s_const, sys_.s_min, sys_.s_max))
    mb = mb_cap(prob, s)
    m = equal_ceiling_split(sys_, prob.K, mb)
    return mb, s, m


def uniform_prior(prob, s_const: float) -> tuple[float, float, np.ndarray]:
    """Weak prior for the "can the swarm find the public split?" test: m_b on the necessary
    DEP/QoS bound (mb_cap, 0 objective evaluations) but the public power split EQUALLY,
    m_j = (1 - m_b)/K, instead of the equal-ceiling split. No feasibility repair."""
    sys_ = prob.sys
    K = prob.K
    s = float(np.clip(s_const, sys_.s_min, sys_.s_max))
    mb = mb_cap(prob, s)
    m = np.full(K, (1.0 - mb) / K)
    assert abs(mb + m.sum() - 1.0) < 1e-12
    return mb, s, m


FAIR_BISECT_ITERS = 12      # (mb_max - mb_min) / 2^12 ~ 1e-4 in m_b; total <= 2 + 12 = 14 evaluations


def fair_prior(prob, s_const: float, n_iter: int = FAIR_BISECT_ITERS) -> tuple[float, float, np.ndarray, int]:
    """Fair prior: the DEP prior made feasible under the SAME search targets as the swarms.

    dep_prior puts m_b on the QoS bound of eq. (19), which uses the rate CEILINGS c_j. The actual
    public rates R_j = E[min log2(1 + gamma)] are strictly below c_j, so at large K the dep_prior
    violates the search target R_qos + rate_margin (it only passes assessment thanks to the margin).
    Here m_b is bisected in [mb_min, mb_cap(s)] with the full search-feasibility test
    (prob.evaluate_alloc: xi_Gamma, all public rates, Bob outage, eligibility) and the same
    equal-ceiling public split. Every test costs ONE objective evaluation and is counted.

    Returns (mb, s, m_sic, n_evals). If even mb_min is search-infeasible, the better of the two
    end points under the feasibility-first ranking is returned (no feasible prior exists).
    """
    from .evaluator import rank_order
    sys_ = prob.sys
    K = prob.K
    s = float(np.clip(s_const, sys_.s_min, sys_.s_max))

    def ev(mb: float):
        m = equal_ceiling_split(sys_, K, mb)
        e = prob.evaluate_alloc(np.array([mb]), np.array([s]), m[None, :])
        return bool(e.feasible[0]), float(e.nu[0]), float(e.R_b[0]), m

    hi = mb_cap(prob, s)
    f_hi, nu_hi, rb_hi, m_hi = ev(hi)
    n = 1
    if f_hi:
        return hi, s, m_hi, n
    lo = float(sys_.mb_min)
    f_lo, nu_lo, rb_lo, m_lo = ev(lo)
    n += 1
    if not f_lo:                                   # no feasible point on the prior's 1-D path
        i = int(rank_order([f_hi, f_lo], [nu_hi, nu_lo], [rb_hi, rb_lo])[0])
        return ((hi, s, m_hi, n) if i == 0 else (lo, s, m_lo, n))
    m_best = m_lo
    for _ in range(int(n_iter)):                  # invariant: feasible(lo), not feasible(hi)
        mid = 0.5 * (lo + hi)
        f_mid, _, _, m_mid = ev(mid)
        n += 1
        if f_mid:
            lo, m_best = mid, m_mid
        else:
            hi = mid
    assert n <= 2 + int(n_iter), n
    return lo, s, m_best, n


def zb_cap(prob, s_prior: float) -> float:
    """Upper limit of the z_b coordinate for the restricted search box.

    The DEP bound varies (weakly) with s, so the cap is the maximum of mb_cap over
    s in {s_min, s_prior, s_max}; in practice the noise term b_r is negligible next to
    a_1 at these powers, so m_dep is almost independent of s. Kept >= 1e-6 so the box
    never degenerates.
    """
    sys_ = prob.sys
    caps = [mb_cap(prob, s) for s in (sys_.s_min, float(s_prior), sys_.s_max)]
    zb = (max(caps) - sys_.mb_min) / (sys_.mb_max - sys_.mb_min)
    return float(min(1.0, max(zb, 1e-6)))
