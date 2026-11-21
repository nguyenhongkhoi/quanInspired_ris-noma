"""Shared argparse options (claude.md 9) and run-directory resolution."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import PRESETS, ExpConfig, exp_from_preset

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "result"
STAGES = ("cases", "labels", "train", "validate", "test", "tables")


def resolve_run_dir(value: str | None, default_name: str) -> Path:
    """Bare name -> <ROOT>/result/<name>; relative path -> ROOT/<path>; absolute kept."""
    if value is None or value == "":
        return RESULT_ROOT / default_name
    if "/" not in value and "\\" not in value and value not in (".", ".."):
        return RESULT_ROOT / value
    p = Path(value)
    return p if p.is_absolute() else (ROOT / p).resolve()


def _int_list(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x.strip())


def _str_list(text: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in text.split(",") if x.strip())


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    p.add_argument("--stage", default="all", help="all or comma list of " + ",".join(STAGES))
    p.add_argument("--out-dir", default=None, help="run name (-> ./result/<name>) or path")
    p.add_argument("--methods", type=_str_list, default=None)
    p.add_argument("--k-train", type=_int_list, default=None)
    p.add_argument("--k-test", type=_int_list, default=None)
    p.add_argument("--n-train-layouts", type=int, default=None)
    p.add_argument("--n-val-layouts", type=int, default=None)
    p.add_argument("--n-test-layouts", type=int, default=None)
    p.add_argument("--cases-per-layout", type=int, default=None)
    p.add_argument("--replicates", type=int, default=None)
    p.add_argument("--search-samples", type=int, default=None)
    p.add_argument("--assess-samples", type=int, default=None)
    p.add_argument("--label-samples", type=int, default=None)
    p.add_argument("--swarm", type=int, default=None)
    p.add_argument("--budget-full", type=int, default=None)
    p.add_argument("--budget-short", type=int, default=None)
    p.add_argument("--budgets-val", type=_int_list, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log-every", type=int, default=1)


def add_quantum_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--baseline-dir", default=None)
    p.add_argument("--beta-start", type=float, default=None)
    p.add_argument("--beta-end", type=float, default=None)
    p.add_argument("--boundary", choices=("reflect", "clip"), default=None)
    p.add_argument("--anchor-w0", type=float, default=None)
    p.add_argument("--anchor-frac", type=float, default=None)
    p.add_argument("--anchor-gate", dest="anchor_gate", action="store_true", default=None)
    p.add_argument("--no-anchor-gate", dest="anchor_gate", action="store_false")


def parse_stages(text: str) -> list[str]:
    if text.strip() == "all":
        return list(STAGES)
    out = [s.strip() for s in text.split(",") if s.strip()]
    bad = [s for s in out if s not in STAGES]
    if bad:
        raise SystemExit(f"unknown stage(s): {bad}; valid: all,{','.join(STAGES)}")
    return [s for s in STAGES if s in out]   # canonical order


def exp_config_from_args(args: argparse.Namespace) -> ExpConfig:
    ov = dict(
        k_train=args.k_train, k_test=args.k_test,
        n_train_layouts=args.n_train_layouts, n_val_layouts=args.n_val_layouts,
        n_test_layouts=args.n_test_layouts, cases_per_layout=args.cases_per_layout,
        replicates=args.replicates, search_samples=args.search_samples,
        assess_samples=args.assess_samples, label_samples=args.label_samples,
        swarm=args.swarm, budget_full=args.budget_full, budget_short=args.budget_short,
        budgets_val=args.budgets_val, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, seed=args.seed,
        workers=args.workers if args.workers is not None else default_workers(),
    )
    for name in ("beta_start", "beta_end", "boundary", "anchor_w0", "anchor_frac", "anchor_gate"):
        if hasattr(args, name):
            ov[name] = getattr(args, name)
    return exp_from_preset(args.preset, **ov)
