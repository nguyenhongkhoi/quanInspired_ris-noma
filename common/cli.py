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
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None,
                   help="training device for GNN/MLP (default auto = cuda if available)")


def add_quantum_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--baseline-dir", default=None)
    p.add_argument("--beta-start", type=float, default=None)
    p.add_argument("--beta-end", type=float, default=None)
    p.add_argument("--boundary", choices=("reflect", "clip"), default=None)
    p.add_argument("--anchor-w0", type=float, default=None)
    p.add_argument("--anchor-frac", type=float, default=None)
    p.add_argument("--anchor-gate", dest="anchor_gate", action="store_true", default=None)
    p.add_argument("--no-anchor-gate", dest="anchor_gate", action="store_false")
    p.add_argument("--k-val", type=_int_list, default=None,
                   help="K values of validation cases (default = k_train); e.g. 2,3,4,5,6 to tune for unseen K")
    p.add_argument("--qq-mbest", choices=("mean", "rank", "feas_rank"), default=None,
                   help="force gnn_qqpso component F instead of the validation choice")
    p.add_argument("--qq-beta", choices=("linear", "feas"), default=None,
                   help="force gnn_qqpso component B instead of the validation choice")
    p.add_argument("--qq-polish", type=int, default=None,
                   help="force gnn_qqpso component P (polish populations) instead of the validation choice")
    p.add_argument("--qq-polish-grid", type=_int_list, default=None,
                   help="polish_pops values tried on validation (default 0,1,2)")
    p.add_argument("--qw-scale", type=float, default=None, help="force gnn_qwqpso well length L (z units)")
    p.add_argument("--qw-frac", type=float, default=None, help="force gnn_qwqpso fraction of particles in the well")
    p.add_argument("--qw-beta-start", type=float, default=None, help="force gnn_qwqpso beta_start")
    p.add_argument("--qw-beta-end", type=float, default=None, help="force gnn_qwqpso beta_end")
    p.add_argument("--qw-polish", type=int, default=None, help="force gnn_qwqpso polish populations")
    p.add_argument("--qw-polish-grid", type=_int_list, default=None,
                   help="polish_pops values tried for gnn_qwqpso on validation (default 1,2)")
    p.add_argument("--k-subset", type=_int_list, default=None,
                   help="run validate/test only on these K (cases filtered; k_test of the baseline run unchanged)")


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
        lr=args.lr, seed=args.seed, device=args.device,
        workers=args.workers if args.workers is not None else default_workers(),
    )
    for name in ("beta_start", "beta_end", "boundary", "anchor_w0", "anchor_frac", "anchor_gate", "k_val"):
        if hasattr(args, name):
            ov[name] = getattr(args, name)
    return exp_from_preset(args.preset, **ov)


# ---------------------------------------------------------------------------
# Run initialisation: config.json, run.log, resume safety
# ---------------------------------------------------------------------------
CRITICAL_KEYS = ("seed", "search_samples", "assess_samples", "label_samples", "n_train_layouts",
                 "n_val_layouts", "n_test_layouts", "cases_per_layout", "replicates", "k_train",
                 "k_test", "swarm", "budget_full")


def _versions() -> dict:
    import platform
    import sys as _sys
    v = {"python": _sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "scipy", "pandas", "matplotlib", "torch"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = None
    try:
        import subprocess
        v["git"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                  text=True, timeout=5).stdout.strip() or None
    except Exception:
        v["git"] = None
    return v


def _norm(v):
    return list(v) if isinstance(v, (list, tuple)) else v


def load_exp_config(run_dir: Path) -> ExpConfig:
    import json
    p = Path(run_dir) / "config.json"
    if not p.exists():
        raise SystemExit(f"missing {p}: run that stage first")
    d = json.loads(p.read_text(encoding="utf-8"))["exp"]
    for k in ("k_train", "k_test", "k_val", "budgets_val", "train_seeds", "pso_seeds"):
        if k in d and d[k] is not None:
            d[k] = tuple(d[k])
    known = set(ExpConfig.__dataclass_fields__)
    return ExpConfig(**{k: v for k, v in d.items() if k in known})


def init_run(out_dir: Path, sys_, exp: ExpConfig, args, kind: str, overwrite: bool = False):
    """Create out_dir, run.log and config.json; refuse to resume with different critical settings."""
    import json
    from datetime import datetime

    from .progress import setup_run_logger
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config.json"
    if cfg_path.exists() and not overwrite:
        old = json.loads(cfg_path.read_text(encoding="utf-8")).get("exp", {})
        diff = {k: (old.get(k), _norm(getattr(exp, k))) for k in CRITICAL_KEYS
                if k in old and _norm(old.get(k)) != _norm(getattr(exp, k))}
        if diff:
            raise SystemExit(f"[{kind}] {out_dir} was created with different settings {diff}.\n"
                             f"Use another --out-dir name, or add --overwrite to start this run again.")
    log = setup_run_logger(out_dir)
    cfg = dict(kind=kind, created=datetime.now().isoformat(timespec="seconds"), sys=sys_.to_dict(),
               exp=exp.to_dict(), args={k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()},
               versions=_versions())
    cfg_path.write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
    log.info(f"=== {kind} run start: out_dir={out_dir}")
    log.info(f"exp={json.dumps(exp.to_dict(), default=str)}")
    log.info(f"versions={json.dumps(cfg['versions'])}")
    return log
