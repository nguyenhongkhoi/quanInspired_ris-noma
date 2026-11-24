"""Physics-informed prior allocation (no learning, no objective evaluation).

Closed form derived from the problem structure (draft Sec. 3.1, eq. (17)):

  * public fractions: equal public-rate ceilings. With A_0 = 1 + k^2 and A_K = m_b + k^2,
        A_j = (1 + k^2) rho^j,  rho = ((m_b + k^2) / (1 + k^2))^(1/K),  m_j = A_{j-1} - A_j,
    so every c_j = log2(A_{j-1}/A_j) = log2(1/rho) is the same (telescoping), in SIC order;
  * m_b: the necessary public-QoS bound evaluated at the SEARCH rate target,
        m_b = (1 + k^2) 2^{-K (R_qos + rate_margin)} - k^2, clipped to [mb_min, mb_max];
  * s: a constant chosen on validation (s_const).

The prior uses only K and the fixed system constants; it does not look at the channel
bank, so it is identical for every case with the same K. Returned m is in SIC order.
"""
from __future__ import annotations

import numpy as np

from .config import SysConfig


def physics_prior(sys_: SysConfig, K: int, s_const: float) -> tuple[float, float, np.ndarray]:
    K = int(K)
    assert K >= 1, K
    k2 = sys_.kappa ** 2
    mb = (1.0 + k2) * 2.0 ** (-K * (sys_.R_qos + sys_.rate_margin)) - k2
    mb = float(np.clip(mb, sys_.mb_min, sys_.mb_max))
    rho = ((mb + k2) / (1.0 + k2)) ** (1.0 / K)
    A = (1.0 + k2) * rho ** np.arange(K + 1)
    m = A[:-1] - A[1:]
    assert m.shape == (K,) and np.all(m > 0), m
    assert abs(mb + m.sum() - 1.0) < 1e-9, (mb, m.sum())
    s = float(np.clip(s_const, sys_.s_min, sys_.s_max))
    return mb, s, m
