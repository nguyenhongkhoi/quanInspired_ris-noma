"""GNN and pooled-MLP allocators (claude.md 5.5, Table VI). Requires PyTorch.

Parameter counts must be GNN = 24,387 and pooled MLP = 24,291 (asserted).
Outputs are in SIC order: m[:, j] for public j (padding entries are 0).
"""
from __future__ import annotations

import numpy as np

from .config import SysConfig

N_PARAMS = {"gnn": 24387, "mlp": 24291}

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


def adjust_prediction(sys: SysConfig, mb, s, m_sic, delta: float):
    """Output adjustment (eq. 29): lower m_b by delta, rescale publics; s unchanged."""
    mb = np.asarray(mb, float)
    m_sic = np.asarray(m_sic, float)
    mb2 = np.maximum(sys.mb_min, mb - delta)
    m2 = m_sic * ((1.0 - mb2) / (1.0 - mb))[..., None]
    return mb2, np.asarray(s, float), m2


if HAS_TORCH:

    class GNNBlock(nn.Module):
        def __init__(self, h: int = 32, e: int = 2):
            super().__init__()
            self.msg1 = nn.Linear(2 * h + e, h)
            self.msg2 = nn.Linear(h, h)
            self.upd1 = nn.Linear(3 * h, h)
            self.upd2 = nn.Linear(h, h)

        def forward(self, h, c, edge, mask):
            B, n, d = h.shape
            hv = h[:, :, None, :].expand(B, n, n, d)       # source v along dim 1
            hu = h[:, None, :, :].expand(B, n, n, d)       # destination u along dim 2
            m = F.relu(self.msg2(F.relu(self.msg1(torch.cat([hv, hu, edge], dim=-1)))))
            eye = torch.eye(n, dtype=torch.bool, device=h.device)[None]
            pm = (mask[:, :, None] & mask[:, None, :] & ~eye).to(h.dtype)   # (B, v, u)
            n_other = (mask.sum(dim=1, keepdim=True).to(h.dtype) - 1.0).clamp(min=1.0)  # (B,1)
            m_u = (m * pm[..., None]).sum(dim=1) / n_other[:, :, None]       # (B, n, d)
            upd = self.upd2(F.relu(self.upd1(torch.cat([h, m_u, c[:, None, :].expand(B, n, d)], -1))))
            return F.relu(h + upd) * mask[..., None].to(h.dtype)

    class MLPBlock(nn.Module):
        def __init__(self, h: int = 32, inner: int = 112):
            super().__init__()
            self.l1 = nn.Linear(h, inner)
            self.l2 = nn.Linear(inner, h)

        def forward(self, h, c, edge, mask):
            return F.relu(h + self.l2(F.relu(self.l1(h)))) * mask[..., None].to(h.dtype)

    class AllocNet(nn.Module):
        def __init__(self, arch: str, sys: SysConfig, hidden: int = 32, blocks: int = 2,
                     node_dim: int = 15, glob_dim: int = 24):
            super().__init__()
            assert arch in ("gnn", "mlp")
            self.arch = arch
            self.mb_min, self.mb_max = sys.mb_min, sys.mb_max
            self.s_min, self.s_max = sys.s_min, sys.s_max
            self.g_enc = nn.Linear(glob_dim, hidden)
            self.n_enc = nn.Linear(node_dim + hidden, hidden)
            blk = GNNBlock if arch == "gnn" else MLPBlock
            self.blocks = nn.ModuleList([blk(hidden) for _ in range(blocks)])
            self.bob1 = nn.Linear(3 * hidden, hidden)
            self.bob2 = nn.Linear(hidden, 2)
            self.pub1 = nn.Linear(4 * hidden, hidden)
            self.pub2 = nn.Linear(hidden, 1)

        def forward(self, node, glob, edge, mask):
            """node (B,n,15) glob (B,24) edge (B,n,n,2) mask (B,n) bool -> mb (B,), s (B,), m (B,n-1)."""
            B, n, _ = node.shape
            maskf = mask.to(node.dtype)
            c = F.relu(self.g_enc(glob))
            h = F.relu(self.n_enc(torch.cat([node, c[:, None, :].expand(B, n, c.shape[-1])], -1)))
            h = h * maskf[..., None]
            for blk in self.blocks:
                h = blk(h, c, edge, mask)
            mean_h = (h * maskf[..., None]).sum(1) / maskf.sum(1, keepdim=True).clamp(min=1.0)
            h_bob = h[:, 0, :]
            o = self.bob2(F.relu(self.bob1(torch.cat([h_bob, mean_h, c], -1))))
            mb = self.mb_min + (self.mb_max - self.mb_min) * torch.sigmoid(o[:, 0])
            s = self.s_min + (self.s_max - self.s_min) * torch.sigmoid(o[:, 1])
            hp = h[:, 1:, :]
            d = hp.shape[-1]
            ctx = torch.cat([h_bob, mean_h, c], -1)[:, None, :].expand(B, n - 1, 3 * d)
            oj = self.pub2(F.relu(self.pub1(torch.cat([hp, ctx], -1))))[..., 0]   # (B, n-1)
            pmask = mask[:, 1:]
            oj = oj.masked_fill(~pmask, float("-inf"))
            m = (1.0 - mb)[:, None] * torch.softmax(oj, dim=1)
            m = torch.nan_to_num(m, nan=0.0) * pmask.to(m.dtype)
            return mb, s, m

    def count_params(model: "nn.Module") -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def build_model(arch: str, sys: SysConfig) -> "AllocNet":
        model = AllocNet(arch, sys)
        n = count_params(model)
        assert n == N_PARAMS[arch], f"{arch}: {n} parameters, expected {N_PARAMS[arch]}"
        return model

    def alloc_loss(mb, s, m, mb_t, s_t, m_t, K):
        """L = 1/(35 B) sum [32 (mb-mb*)^2 + (s-s*)^2 + (2/K) sum_j (m_j-m_j*)^2]."""
        B = mb.shape[0]
        pub = ((m - m_t) ** 2).sum(dim=1) * (2.0 / K.to(m.dtype))
        return (32.0 * (mb - mb_t) ** 2 + (s - s_t) ** 2 + pub).sum() / (35.0 * B)

    def to_tensors(node, glob, edge, mask, device="cpu"):
        dt = torch.float32
        return (torch.as_tensor(node, dtype=dt, device=device),
                torch.as_tensor(glob, dtype=dt, device=device),
                torch.as_tensor(edge, dtype=dt, device=device),
                torch.as_tensor(mask, dtype=torch.bool, device=device))

    @torch.no_grad()
    def predict_numpy(model, node, glob, edge, mask, device="cpu"):
        model.eval()
        mb, s, m = model(*to_tensors(node, glob, edge, mask, device))
        return (mb.double().cpu().numpy(), s.double().cpu().numpy(), m.double().cpu().numpy())

    def save_checkpoint(path, model, norm_dict: dict, meta: dict) -> None:
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "normalizer": norm_dict, "meta": meta}, str(path))

    def load_checkpoint(path, sys: SysConfig):
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        model = build_model(ck["meta"]["arch"], sys)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        return model, ck["normalizer"], ck["meta"]
