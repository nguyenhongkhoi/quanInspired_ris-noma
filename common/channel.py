"""Geometry, layouts, LHS operating cases and fading sample banks (claude.md 3.1, 6.1).

Receiver column convention inside a ChannelBank (original identities):
    col 0          -> Bob
    col 1..K       -> public users C_1..C_K (first K of the layout, original order)
    col K+1        -> Willie
X[:, c] = |sum_n h_ar,n h_rc,n e^{j theta_n}|^2 and Y[:, c] = |sum_n h_rc,n e^{j theta_n}|^2
are cached at rho = 1; theta follows the Bob-aligned rule (4) per block.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SPLIT_ID_BASE, ExpConfig, SysConfig, make_rng

_CHUNK = 4096  # fixed chunk size -> bank contents depend only on (seed, M)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def element_positions(sys: SysConfig) -> np.ndarray:
    """p_{r,n} = p_r + (n - (N+1)/2) d_e u_r, n = 1..N. Shape (N, 2)."""
    n = np.arange(1, sys.N + 1, dtype=float)
    offs = (n - (sys.N + 1) / 2.0) * sys.de
    return np.asarray(sys.pos_ris)[None, :] + offs[:, None] * np.asarray(sys.u_r)[None, :]


def los_phase(sys: SysConfig, p_v: np.ndarray) -> np.ndarray:
    """phi_{rv,n} = -(2 pi / lambda)(||p_v - p_{r,n}|| - d_rv). Shape (N,)."""
    p_v = np.asarray(p_v, dtype=float)
    p_rn = element_positions(sys)
    d_rv = np.linalg.norm(p_v - np.asarray(sys.pos_ris))
    d_n = np.linalg.norm(p_v[None, :] - p_rn, axis=1)
    return -(2.0 * np.pi / sys.lam) * (d_n - d_rv)


def path_loss(sys: SysConfig, d: float | np.ndarray, alpha: float) -> np.ndarray:
    """L = (max(d, d_min)/d0)^alpha / beta0 (distances from the RIS centre)."""
    d = np.maximum(np.asarray(d, dtype=float), sys.d_min)
    return (d / sys.d0) ** alpha / sys.beta0


def dist_to_ris(sys: SysConfig, p: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.atleast_2d(p) - np.asarray(sys.pos_ris)[None, :], axis=1)


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Layout:
    layout_id: int
    split: str
    pos_bob: np.ndarray      # (2,)
    pos_willie: np.ndarray   # (2,)
    pos_pub: np.ndarray      # (K_max, 2), original public identities


def _uniform_in(rng: np.random.Generator, region, size=None) -> np.ndarray:
    (x0, x1), (y0, y1) = region
    if size is None:
        return np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
    return np.column_stack([rng.uniform(x0, x1, size), rng.uniform(y0, y1, size)])


def generate_layout(sys: SysConfig, master_seed: int, layout_id: int, split: str, K_max: int) -> Layout:
    rng = make_rng(master_seed, "layout", int(layout_id))
    pos_bob = _uniform_in(rng, sys.region_bob)
    pos_willie = _uniform_in(rng, sys.region_willie)
    pos_pub = _uniform_in(rng, sys.region_pub, size=K_max)
    assert pos_pub.shape == (K_max, 2)
    return Layout(int(layout_id), split, pos_bob, pos_willie, pos_pub)


def layout_ids(split: str, n: int) -> list[int]:
    base = SPLIT_ID_BASE[split]
    assert n < 1000, "layout id would overflow its split range"
    return [base + i for i in range(n)]


def lhs_pairs(sys: SysConfig, master_seed: int, layout_id: int, n: int) -> np.ndarray:
    """Randomized Latin-hypercube (P_tot_dbm, xi_req) pairs, shared by all K. Shape (n, 2)."""
    rng = make_rng(master_seed, "lhs", int(layout_id))
    out = np.empty((n, 2))
    for d, (lo, hi) in enumerate((sys.P_tot_dbm_range, sys.xi_req_range)):
        perm = rng.permutation(n)
        u = (perm + rng.random(n)) / n
        out[:, d] = lo + (hi - lo) * u
    return out


def build_cases(sys: SysConfig, exp: ExpConfig, split: str) -> list[dict]:
    """All operating cases of one split: layout x K x LHS pair."""
    n_lay = {"train": exp.n_train_layouts, "val": exp.n_val_layouts, "test": exp.n_test_layouts}[split]
    ks = exp.k_test if split == "test" else exp.k_train
    rows = []
    for lid in layout_ids(split, n_lay):
        pairs = lhs_pairs(sys, exp.seed, lid, exp.cases_per_layout)
        for K in ks:
            for ci in range(exp.cases_per_layout):
                rows.append(dict(case_id=f"{lid}_K{K}_c{ci}", layout_id=lid, split=split, K=int(K),
                                 case_idx=ci, P_tot_dbm=float(pairs[ci, 0]), xi_req=float(pairs[ci, 1])))
    return rows


# ---------------------------------------------------------------------------
# Channel bank
# ---------------------------------------------------------------------------
@dataclass
class ChannelBank:
    layout_id: int
    K: int
    M: int
    X: np.ndarray        # (M, K+2) signal gain at rho=1 (Bob, publics, Willie)
    Y: np.ndarray        # (M, K+2) scalar RIS-noise gain at rho=1
    L_ar: float
    L_ru: np.ndarray     # (K+2,) path loss RIS->receiver (Bob, publics, Willie)
    pos: np.ndarray      # (K+2, 2) receiver positions
    mean_har2: float     # mean |h_ar,n|^2 over the bank

    @property
    def col_willie(self) -> int:
        return self.K + 1


def build_bank(sys: SysConfig, layout: Layout, K: int, M: int, seed: int) -> ChannelBank:
    """Draw M independent fading blocks for (layout, K) and cache X, Y at rho = 1."""
    assert 1 <= K <= layout.pos_pub.shape[0], f"K={K} exceeds layout K_max"
    assert M >= 1
    rng = np.random.default_rng(seed)
    N = sys.N
    pos = np.vstack([layout.pos_bob[None, :], layout.pos_pub[:K], layout.pos_willie[None, :]])
    R = K + 2
    assert pos.shape == (R, 2)

    a_los = np.sqrt(sys.K_R / (sys.K_R + 1.0))
    a_nl = np.sqrt(1.0 / (sys.K_R + 1.0))
    los_ar = a_los * np.exp(1j * los_phase(sys, np.asarray(sys.pos_alice)))           # (N,)
    los_ru = a_los * np.exp(1j * np.stack([los_phase(sys, p) for p in pos]))            # (R, N)

    X = np.empty((M, R))
    Y = np.empty((M, R))
    sum_har2 = 0.0
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for start in range(0, M, _CHUNK):
        m = min(_CHUNK, M - start)
        z_ar = (rng.standard_normal((m, N)) + 1j * rng.standard_normal((m, N))) * inv_sqrt2
        h_ar = los_ar[None, :] + a_nl * z_ar                                            # (m, N)
        z_ru = (rng.standard_normal((m, R, N)) + 1j * rng.standard_normal((m, R, N))) * inv_sqrt2
        h_ru = los_ru[None, :, :] + a_nl * z_ru                                         # (m, R, N)
        # theta_n = -arg(h_ar,n) - arg(h_rb,n)  (Bob = column 0)
        e_th = np.exp(-1j * (np.angle(h_ar) + np.angle(h_ru[:, 0, :])))                 # (m, N)
        g = h_ru * e_th[:, None, :]
        H = np.einsum("mn,mrn->mr", h_ar, g)
        G = g.sum(axis=2)
        X[start:start + m] = np.abs(H) ** 2
        Y[start:start + m] = np.abs(G) ** 2
        sum_har2 += float(np.sum(np.abs(h_ar) ** 2))

    L_ar = float(path_loss(sys, dist_to_ris(sys, np.asarray(sys.pos_alice))[0], sys.alpha_ar))
    L_ru = path_loss(sys, dist_to_ris(sys, pos), sys.alpha_ru)
    assert L_ru.shape == (R,)
    assert np.all(np.isfinite(X)) and np.all(np.isfinite(Y))
    return ChannelBank(layout.layout_id, K, M, X, Y, L_ar, L_ru, pos, sum_har2 / (M * N))


def bank_seed(master_seed: int, kind: str, layout_id: int, K: int) -> int:
    """kind in {'search','assess','select','labelassess', ...}."""
    from .config import stable_seed
    return stable_seed(master_seed, kind, int(layout_id), int(K))
