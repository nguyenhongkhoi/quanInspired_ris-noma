"""Graph features (claude.md 5.4, Appendix B-A).

Node order: index 0 = Bob, indices 1..K = publics in SIC order.
All statistics come from the INPUT (search) bank only; no label or assessment
result is used as an input feature.
"""
from __future__ import annotations

import math

import numpy as np

from .config import SysConfig, dbm_to_w
from .evaluator import Problem

NODE_DIM = 15
GLOBAL_DIM = 24
EDGE_DIM = 2


def _log10(x: float) -> float:
    return math.log10(max(float(x), 1e-300))


def channel_stats(sig: np.ndarray, noise: np.ndarray) -> list[float]:
    """log10 mean signal, CV signal, log10 mean noise, CV noise, corr(signal, noise)."""
    ms, mn = float(sig.mean()), float(noise.mean())
    ss, sn = float(sig.std()), float(noise.std())
    corr = float(np.corrcoef(sig, noise)[0, 1]) if (ss > 0 and sn > 0) else 0.0
    return [_log10(ms), ss / ms if ms > 0 else 0.0, _log10(mn), sn / mn if mn > 0 else 0.0, corr]


def node_features(prob: Problem) -> np.ndarray:
    sys = prob.sys
    bank = prob.bank
    K = prob.K
    pr = np.asarray(sys.pos_ris)
    out = np.zeros((K + 1, NODE_DIM))
    cols = [0] + [1 + int(j) for j in prob.sic_order]
    for i, c in enumerate(cols):
        is_bob = i == 0
        rel = (bank.pos[c] - pr) / 100.0
        L = float(bank.L_ru[c])
        sig = bank.X[:, c] / (bank.L_ar * L)
        noise = bank.Y[:, c] / L
        qos = 0.0 if is_bob else sys.R_qos
        rdec = sys.R_dec_bob if is_bob else sys.R_dec_pub
        gam = sys.Gamma_bob if is_bob else sys.Gamma_pub
        rank = (K + 1) if is_bob else i
        out[i] = [1.0 if is_bob else 0.0, 0.0 if is_bob else 1.0,
                  rel[0], rel[1], _log10(L), _log10(sys.sigma_u2),
                  qos, rdec, gam, rank / (K + 1)] + channel_stats(sig, noise)
    assert out.shape == (K + 1, NODE_DIM) and np.all(np.isfinite(out))
    return out


def global_features(prob: Problem) -> np.ndarray:
    sys = prob.sys
    bank = prob.bank
    cw = bank.col_willie
    L_rw = float(bank.L_ru[cw])
    w_stats = channel_stats(bank.X[:, cw] / (bank.L_ar * L_rw), bank.Y[:, cw] / L_rw)
    g = [prob.xi_req,
         _log10(dbm_to_w(prob.P_tot_dbm)), _log10(prob.P0),
         sys.kappa, float(sys.N), sys.K_R,
         _log10(bank.L_ar), _log10(L_rw),
         _log10(sys.sigma_r2), _log10(sys.sigma_w2),
         sys.p_max,
         sys.mb_min, sys.mb_max, sys.s_min, sys.s_max,
         float(prob.K),
         _log10(bank.mean_har2)] + w_stats + [_log10(sys.P_S), _log10(sys.P_DC)]
    out = np.asarray(g, dtype=float)
    assert out.shape == (GLOBAL_DIM,) and np.all(np.isfinite(out))
    return out


def edge_flags(K: int) -> np.ndarray:
    """e[v, u] = [decode, interference]; ranks: publics 1..K (SIC order), Bob K+1."""
    n = K + 1
    rank = np.array([K + 1] + list(range(1, K + 1)))
    e = np.zeros((n, n, EDGE_DIM))
    for v in range(n):
        for u in range(n):
            if v == u:
                continue
            v_pub = v != 0
            e[v, u, 0] = 1.0 if (v_pub and rank[v] < rank[u]) else 0.0
            e[v, u, 1] = 1.0 if ((not v_pub) or rank[v] > rank[u]) else 0.0
    return e


def case_features(prob: Problem) -> tuple[np.ndarray, np.ndarray]:
    return node_features(prob), global_features(prob)


class Normalizer:
    """Mean/population-std standardisation fitted on labelled TRAIN graphs only."""

    def __init__(self, n_mean, n_scale, g_mean, g_scale):
        self.n_mean = np.asarray(n_mean, float)
        self.n_scale = np.asarray(n_scale, float)
        self.g_mean = np.asarray(g_mean, float)
        self.g_scale = np.asarray(g_scale, float)

    @staticmethod
    def fit(nodes: list[np.ndarray], globs: list[np.ndarray]) -> "Normalizer":
        N = np.concatenate(nodes, axis=0)   # every active node equally weighted
        G = np.stack(globs, axis=0)
        n_std = N.std(axis=0)
        g_std = G.std(axis=0)
        n_scale = np.where(n_std < 1e-12, 1.0, n_std)
        g_scale = np.where(g_std < 1e-12, 1.0, g_std)
        return Normalizer(N.mean(axis=0), n_scale, G.mean(axis=0), g_scale)

    def nodes(self, x: np.ndarray) -> np.ndarray:
        return (x - self.n_mean) / self.n_scale

    def glob(self, g: np.ndarray) -> np.ndarray:
        return (g - self.g_mean) / self.g_scale

    def to_dict(self) -> dict:
        return dict(n_mean=self.n_mean.tolist(), n_scale=self.n_scale.tolist(),
                    g_mean=self.g_mean.tolist(), g_scale=self.g_scale.tolist())

    @staticmethod
    def from_dict(d: dict) -> "Normalizer":
        return Normalizer(d["n_mean"], d["n_scale"], d["g_mean"], d["g_scale"])


def collate(node_list, glob_list, norm: Normalizer | None = None):
    """Pad a list of graphs. Returns numpy arrays (node, glob, edge, mask, K)."""
    B = len(node_list)
    Ks = np.array([x.shape[0] - 1 for x in node_list], dtype=int)
    n = int(Ks.max()) + 1
    node = np.zeros((B, n, NODE_DIM))
    edge = np.zeros((B, n, n, EDGE_DIM))
    mask = np.zeros((B, n), dtype=bool)
    glob = np.zeros((B, GLOBAL_DIM))
    for b, (x, g) in enumerate(zip(node_list, glob_list)):
        k = Ks[b]
        node[b, :k + 1] = norm.nodes(x) if norm else x
        glob[b] = norm.glob(g) if norm else g
        edge[b, :k + 1, :k + 1] = edge_flags(int(k))
        mask[b, :k + 1] = True
    return node, glob, edge, mask, Ks
