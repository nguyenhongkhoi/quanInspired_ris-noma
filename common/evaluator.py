"""Physical evaluator (claude.md 3.2-3.6, 5.1).

A `Problem` binds one operating case (bank, P_tot, xi_req) and evaluates a batch
of allocations at once (vectorised over particles). Search and assessment use the
same class; only targets differ (search: xi_req+0.03, R_qos+0.01).

Allocations inside the evaluator are in SIC order: m_sic[:, j] is the fraction of
the j-th public in increasing mean cascaded gain. Use `to_original` / `to_sic`
to convert to/from original public identities.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammainc, gammaincc, gammaln

from .channel import ChannelBank
from .config import SysConfig

LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# Particle coordinates (claude.md 5.1)
# ---------------------------------------------------------------------------
def z_to_alloc(sys: SysConfig, z: np.ndarray, K: int):
    """z (P, K+1) in [0,1] -> (mb (P,), s (P,), m_sic (P, K))."""
    z = np.atleast_2d(np.asarray(z, dtype=float))
    assert z.shape[1] == K + 1, f"z has {z.shape[1]} coords, expected K+1={K + 1}"
    mb = sys.mb_min + (sys.mb_max - sys.mb_min) * z[:, 0]
    s = sys.s_min + (sys.s_max - sys.s_min) * z[:, -1]
    P = z.shape[0]
    cuts = np.sort(z[:, 1:K], axis=1)                                   # (P, K-1)
    tau = np.concatenate([np.zeros((P, 1)), cuts, np.ones((P, 1))], axis=1)
    m_sic = (1.0 - mb)[:, None] * np.diff(tau, axis=1)
    assert m_sic.shape == (P, K)
    return mb, s, m_sic


def alloc_to_z(sys: SysConfig, mb, s, m_sic) -> np.ndarray:
    """Canonical inverse (eq. 38). m_sic must already be in SIC order."""
    mb = np.atleast_1d(np.asarray(mb, dtype=float))
    s = np.atleast_1d(np.asarray(s, dtype=float))
    m_sic = np.atleast_2d(np.asarray(m_sic, dtype=float))
    K = m_sic.shape[1]
    zb = (mb - sys.mb_min) / (sys.mb_max - sys.mb_min)
    zs = (s - sys.s_min) / (sys.s_max - sys.s_min)
    zc = np.cumsum(m_sic, axis=1)[:, :K - 1] / (1.0 - mb)[:, None]
    z = np.concatenate([zb[:, None], zc, zs[:, None]], axis=1)
    return np.clip(z, 0.0, 1.0)


def to_original(m_sic: np.ndarray, sic_order: np.ndarray) -> np.ndarray:
    """sic_order[pos] = original public index. Returns m in original identities."""
    m_sic = np.asarray(m_sic)
    out = np.empty_like(m_sic)
    out[..., sic_order] = m_sic
    return out


def to_sic(m_orig: np.ndarray, sic_order: np.ndarray) -> np.ndarray:
    return np.asarray(m_orig)[..., sic_order]


# ---------------------------------------------------------------------------
# Gamma-approximation DEP (claude.md 3.5, Appendix A-A)
# ---------------------------------------------------------------------------
_Y_LO, _Y_HI = -40.0, 40.0


def gamma_dep(k0: float, th0: float, k1: float, th1: float) -> float:
    """min DEP between Gamma(k0, th0) [H0] and Gamma(k1, th1) [H1].

    DEP(t) = Q(k0, t/th0) + P(k1, t/th1); evaluated at every positive pdf
    intersection and compared with the endpoint value 1.
    """
    if not (k0 > 0 and k1 > 0 and th0 > 0 and th1 > 0):
        return 1.0
    p_ref = math.sqrt(k0 * th0 * k1 * th1)
    t0, t1 = th0 / p_ref, th1 / p_ref
    dk = k1 - k0
    c = 1.0 / t0 - 1.0 / t1
    C = gammaln(k1) - gammaln(k0) + k1 * math.log(t1) - k0 * math.log(t0)
    if abs(dk) < 1e-14 and abs(c) < 1e-14:
        return 1.0  # identical distributions

    def f(y: float) -> float:
        return dk * y + c * math.exp(y) - C

    pts = [_Y_LO, _Y_HI]
    if c != 0.0 and -dk / c > 0.0:
        y_star = math.log(-dk / c)
        if _Y_LO < y_star < _Y_HI:
            pts = [_Y_LO, y_star, _Y_HI]
    roots: list[float] = []
    for a, b in zip(pts[:-1], pts[1:]):
        fa, fb = f(a), f(b)
        if fa == 0.0:
            roots.append(a)
        elif fa * fb < 0.0:
            roots.append(brentq(f, a, b, xtol=1e-13, rtol=1e-13, maxiter=200))
    if len(pts) == 3 and abs(f(pts[1])) < 1e-12:   # tangency
        roots.append(pts[1])
    if pts[-1] not in roots and f(pts[-1]) == 0.0:
        roots.append(pts[-1])
    best = 1.0
    for y in roots:
        t = p_ref * math.exp(y)
        dep = float(gammaincc(k0, t / th0) + gammainc(k1, t / th1))
        best = min(best, dep)
    return best


def gamma_params(mu: float, v: float) -> tuple[float, float]:
    """Moment-matched (shape, scale)."""
    if not (mu > 0 and v > 0):
        return 0.0, 0.0
    return mu * mu / v, v / mu


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------
@dataclass
class EvalBatch:
    mb: np.ndarray          # (P,)
    s: np.ndarray           # (P,)
    m_sic: np.ndarray       # (P, K)
    R_b: np.ndarray         # (P,)
    R_pub: np.ndarray       # (P, K) SIC order
    p_b: np.ndarray         # (P,)  outage estimate (1 if margins fail)
    n_out: np.ndarray       # (P,)  number of outage samples (M if margins fail)
    margins_ok: np.ndarray  # (P,) bool
    xi_gamma: np.ndarray    # (P,)
    feasible: np.ndarray    # (P,) bool (w.r.t. the Problem's targets)
    nu: np.ndarray          # (P,) normalised violation (0 if feasible)
    A: np.ndarray           # (P, K) A_j
    U_b: np.ndarray | None = None  # (P, M) only when keep_U=True

    def take(self, idx) -> "EvalBatch":
        sel = {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        return EvalBatch(**sel)


def is_better(fa, nua, rba, fb, nub, rbb):
    """Feasibility-first ranking (claude.md 3.6): True where candidate a strictly beats b."""
    fa = np.asarray(fa, bool)
    fb = np.asarray(fb, bool)
    both_inf = ~fa & ~fb
    return ((fa & ~fb)
            | (fa & fb & (rba > rbb))
            | (both_inf & ((nua < nub) | ((nua == nub) & (rba > rbb)))))


def rank_order(feasible, nu, R_b) -> np.ndarray:
    """Indices sorted best-first under feasibility-first ranking (stable)."""
    feasible = np.asarray(feasible, bool)
    key_nu = np.where(feasible, 0.0, nu)
    return np.lexsort((-np.asarray(R_b), key_nu, ~feasible))


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------
class Problem:
    """One operating case on one sample bank.

    Parameters
    ----------
    mode : 'search' uses xi_req+margin / R_qos+margin; 'assess' uses original targets.
    sic_order, eligible : if given (assessment), reuse the case's statistics from the
        input bank; otherwise compute them from this bank.
    """

    def __init__(self, sys: SysConfig, bank: ChannelBank, P_tot_dbm: float, xi_req: float,
                 mode: str = "search", sic_order: np.ndarray | None = None,
                 eligible: bool | None = None):
        assert mode in ("search", "assess")
        self.sys = sys
        self.bank = bank
        self.K = K = bank.K
        self.M = bank.M
        self.P_tot_dbm = float(P_tot_dbm)
        self.xi_req = float(xi_req)
        self.mode = mode
        self.P0 = sys.P0(P_tot_dbm)
        assert self.P0 > 0, f"P0 <= 0 at P_tot={P_tot_dbm} dBm"
        if mode == "search":
            self.xi_t = xi_req + sys.xi_margin
            self.R_t = sys.R_qos + sys.rate_margin
        else:
            self.xi_t = xi_req
            self.R_t = sys.R_qos

        L_ar = bank.L_ar
        cw = bank.col_willie
        # mean cascaded signal gain (rho independent) for Bob + publics (original ids)
        self.mean_gain = bank.X[:, :K + 1].mean(axis=0) / (L_ar * bank.L_ru[:K + 1])
        if sic_order is None:
            sic_order = np.argsort(self.mean_gain[1:], kind="stable")
        self.sic_order = np.asarray(sic_order, dtype=int)
        assert sorted(self.sic_order.tolist()) == list(range(K))
        if eligible is None:
            eligible = bool(self.mean_gain[0] > self.mean_gain[1:].max())
        self.eligible = bool(eligible)

        cols = np.concatenate([[0], 1 + self.sic_order])          # Bob, publics in SIC order
        # U = P_a rho2 X / (L_ar (sigma_r2 rho2 Y + sigma_u2 L_ru))
        self._Xs = bank.X[:, cols] / L_ar                          # (M, K+1)
        self._Ys = sys.sigma_r2 * bank.Y[:, cols]                  # (M, K+1)
        self._Ds = sys.sigma_u2 * bank.L_ru[cols]                  # (K+1,)
        # Willie
        Xw = bank.X[:, cw]
        Yw = bank.Y[:, cw]
        self.L_rw = float(bank.L_ru[cw])
        self.Xw, self.Yw = Xw, Yw
        self.EXw, self.VXw = float(Xw.mean()), float(Xw.var())
        self.EYw, self.VYw = float(Yw.mean()), float(Yw.var())
        self.cv_w = float(np.sqrt(self.VXw) / self.EXw)

    # ------------------------------------------------------------------
    def powers(self, s: np.ndarray):
        sys = self.sys
        P_a = (1.0 - s) * self.P0
        P_r = s * self.P0
        rho2 = P_r / (sys.N * (P_a / self.bank.L_ar + sys.sigma_r2))
        return P_a, rho2

    def willie_coeffs(self, mb, P_a, rho2):
        """(a0, a1, b_r) already multiplied by rho^2 so that Z_i = a_i X1 + b_r Y1 at rho=1."""
        k2 = self.sys.kappa ** 2
        a1 = (1.0 + k2) * P_a / (self.bank.L_ar * self.L_rw) * rho2
        a0 = (1.0 - mb) * a1
        b = self.sys.sigma_r2 / self.L_rw * rho2
        return a0, a1, b

    def xi_gamma_one(self, mb: float, P_a: float, rho2: float) -> float:
        a0, a1, b = self.willie_coeffs(mb, P_a, rho2)
        mu0 = a0 * self.EXw + b * self.EYw
        mu1 = a1 * self.EXw + b * self.EYw
        v0 = a0 * a0 * self.VXw + b * b * self.VYw     # covariance omitted (eq. 23)
        v1 = a1 * a1 * self.VXw + b * b * self.VYw
        k0, th0 = gamma_params(mu0, v0)
        k1, th1 = gamma_params(mu1, v1)
        return gamma_dep(k0, th0, k1, th1)

    # ------------------------------------------------------------------
    def evaluate_z(self, z: np.ndarray, keep_U: bool = False) -> EvalBatch:
        mb, s, m_sic = z_to_alloc(self.sys, z, self.K)
        return self.evaluate_alloc(mb, s, m_sic, keep_U=keep_U)

    def evaluate_alloc(self, mb, s, m_sic, keep_U: bool = False) -> EvalBatch:
        sys = self.sys
        K = self.K
        mb = np.atleast_1d(np.asarray(mb, dtype=float))
        s = np.atleast_1d(np.asarray(s, dtype=float))
        m_sic = np.atleast_2d(np.asarray(m_sic, dtype=float))
        P = mb.shape[0]
        assert s.shape == (P,) and m_sic.shape == (P, K), (s.shape, m_sic.shape, P, K)
        k2 = sys.kappa ** 2

        P_a, rho2 = self.powers(s)
        # U: (P, M, K+1)
        U = (P_a * rho2)[:, None, None] * self._Xs[None, :, :] / (
            rho2[:, None, None] * self._Ys[None, :, :] + self._Ds[None, None, :])
        U_b = U[:, :, 0]
        U_pub = U[:, :, 1:]

        # A_j = mb + sum_{l>j} m_l + k^2
        tail = np.cumsum(m_sic[:, ::-1], axis=1)[:, ::-1] - m_sic
        A = mb[:, None] + tail + k2
        # decoders of stream j: Bob and C_j..C_K -> gamma increasing in U, so use suffix-min
        U_suf = np.minimum.accumulate(U_pub[:, :, ::-1], axis=2)[:, :, ::-1]
        U_dec = np.minimum(U_suf, U_b[:, :, None])
        gam = m_sic[:, None, :] * U_dec / (A[:, None, :] * U_dec + 1.0)
        R_pub = np.log1p(gam).mean(axis=1) / LN2                    # (P, K)
        gam_b = mb[:, None] * U_b / (k2 * U_b + 1.0)
        R_b = np.log1p(gam_b).mean(axis=1) / LN2                    # (P,)

        # Bob's complete-chain outage (Proposition 1)
        d_pub = m_sic - sys.Gamma_pub * A
        d_b = mb - sys.Gamma_bob * k2
        margins_ok = np.all(d_pub > 0, axis=1) & (d_b > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            T = np.maximum(np.max(sys.Gamma_pub / d_pub, axis=1), sys.Gamma_bob / d_b)
        T = np.where(margins_ok, T, np.inf)
        n_out = np.where(margins_ok, (U_b < T[:, None]).sum(axis=1), self.M).astype(np.int64)
        p_b = n_out / self.M

        xi_g = np.array([self.xi_gamma_one(mb[i], P_a[i], rho2[i]) for i in range(P)])

        feas = ((xi_g >= self.xi_t) & np.all(R_pub >= self.R_t, axis=1)
                & (p_b <= sys.p_max) & self.eligible)
        nu = (np.maximum(self.xi_t - xi_g, 0.0) / self.xi_t
              + np.maximum(self.R_t - R_pub, 0.0).sum(axis=1) / self.R_t
              + np.maximum(p_b - sys.p_max, 0.0) / sys.p_max
              + (0.0 if self.eligible else 1.0))
        nu = np.where(feas, 0.0, nu)
        return EvalBatch(mb=mb, s=s, m_sic=m_sic, R_b=R_b, R_pub=R_pub, p_b=p_b, n_out=n_out,
                         margins_ok=margins_ok, xi_gamma=xi_g, feasible=feas, nu=nu, A=A,
                         U_b=U_b if keep_U else None)

    # ------------------------------------------------------------------
    def willie_samples(self, mb: float, s: float):
        """Z0, Z1 samples on this bank (for empirical DEP)."""
        P_a, rho2 = self.powers(np.array([s], dtype=float))
        a0, a1, b = self.willie_coeffs(mb, P_a[0], rho2[0])
        Z0 = a0 * self.Xw + b * self.Yw
        Z1 = a1 * self.Xw + b * self.Yw
        return Z0, Z1
