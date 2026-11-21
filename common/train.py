"""Supervised training of GNN / pooled MLP (claude.md 5.5, Appendix C-B).

Adam lr 1e-3, weight decay 1e-4, grad-norm clip 1, batch 32, 200 epochs.
Checkpoint = lowest validation loss (validation labels only; no test data).
Runs on GPU when available (--device auto); allocation-time inference is CPU.
"""
from __future__ import annotations

import copy
import csv
import os
import time
from pathlib import Path

import numpy as np

from .config import ExpConfig, SysConfig
from .features import Normalizer, collate
from .progress import ProgressLine

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS


def resolve_device(name: str) -> str:
    import torch
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
    return name


def _pack(data: dict, norm: Normalizer, device: str):
    import torch
    node, glob, edge, mask, Ks = collate(data["node"], data["glob"], norm)
    B, n = mask.shape
    m = np.zeros((B, n - 1))
    for b, mm in enumerate(data["m"]):
        m[b, :mm.size] = mm
    f = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
    return dict(node=f(node), glob=f(glob), edge=f(edge), mask=f(mask, torch.bool),
                K=f(Ks), mb=f(np.array(data["mb"])), s=f(np.array(data["s"])), m=f(m))


def _batch(T: dict, idx):
    return {k: v[idx] for k, v in T.items()}


def fit_normalizer(train: dict) -> Normalizer:
    assert train["node"], "no labelled training graphs"
    return Normalizer.fit(train["node"], train["glob"])


def train_one(arch: str, seed: int, train: dict, val: dict, norm: Normalizer, sys_: SysConfig,
              exp: ExpConfig, device: str, models_dir: Path, quiet: bool = False, log=None) -> dict:
    import torch
    from .models import alloc_loss, build_model, count_params, save_checkpoint

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    rng = np.random.default_rng(seed)
    model = build_model(arch, sys_).to(device)
    n_params = count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=exp.lr, weight_decay=exp.weight_decay)
    Tt = _pack(train, norm, device)
    Tv = _pack(val, norm, device) if val["node"] else None
    N = Tt["mb"].shape[0]

    best = dict(val=float("inf"), epoch=0, state=None)
    log_path = models_dir / f"train_log_{arch}_seed{seed}.csv"
    rows = []
    pl = ProgressLine("train", exp.epochs, unit=f"ep {arch} s{seed}", quiet=quiet)
    for ep in range(1, exp.epochs + 1):
        t_ep = time.perf_counter()
        model.train()
        perm = rng.permutation(N)
        tot, cnt = 0.0, 0
        for st in range(0, N, exp.batch_size):
            idx = torch.as_tensor(perm[st:st + exp.batch_size], device=device)
            b = _batch(Tt, idx)
            mb, s, m = model(b["node"], b["glob"], b["edge"], b["mask"])
            loss = alloc_loss(mb, s, m, b["mb"], b["s"], b["m"], b["K"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), exp.grad_clip)
            opt.step()
            tot += float(loss.detach()) * idx.numel()
            cnt += idx.numel()
        tr_loss = tot / max(1, cnt)
        model.eval()
        with torch.no_grad():
            if Tv is not None:
                mb, s, m = model(Tv["node"], Tv["glob"], Tv["edge"], Tv["mask"])
                va_loss = float(alloc_loss(mb, s, m, Tv["mb"], Tv["s"], Tv["m"], Tv["K"]))
            else:
                va_loss = tr_loss
        if va_loss < best["val"]:
            best = dict(val=va_loss, epoch=ep, state=copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}))
        dt = time.perf_counter() - t_ep
        rows.append(dict(epoch=ep, train_loss=tr_loss, val_loss=va_loss, best_epoch=best["epoch"], sec=dt))
        pl.update(ep, f"loss={tr_loss:.2e} val={va_loss:.2e} best@{best['epoch']} {dt:.2f}s/ep")
    pl.close(f"[train   ] {arch} seed {seed}: best epoch {best['epoch']} val={best['val']:.4e} params={n_params}")

    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    model.load_state_dict(best["state"])
    model = model.cpu()
    meta = dict(arch=arch, seed=seed, epoch=best["epoch"], val_loss=best["val"], n_params=n_params,
                device=device, delta=exp.delta_adjust, n_train=int(N),
                n_val=int(Tv["mb"].shape[0]) if Tv is not None else 0)
    save_checkpoint(models_dir / f"{arch}_seed{seed}.pt", model, norm.to_dict(), meta)
    if log:
        log.info(f"train {arch} seed={seed} best_epoch={best['epoch']} val_loss={best['val']:.6e} "
                 f"n_params={n_params} device={device} n_train={N}")
    return meta
