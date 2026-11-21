"""Quantum-inspired (classical QPSO) pipeline (claude.md 8).

Reuses cases, labels, models and the selected short budget of a baseline run
(--baseline-dir). Nothing is retrained. Stages used: validate, test, tables.
    python Quantummodel_code/run_quantum.py --preset quick --stage all --baseline-dir bl_quick_01 --out-dir q_quick_01
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from common.cli import (CRITICAL_KEYS, add_common_args, add_quantum_args,  # noqa: E402
                        exp_config_from_args, init_run, load_exp_config, parse_stages,
                        resolve_run_dir)
from common.config import SysConfig  # noqa: E402
from common.methods import QUANTUM_METHODS, make_spec  # noqa: E402
from common.runner import run_trial_stage  # noqa: E402

ANCHOR_W0_GRID = (0.25, 0.5, 0.75)
ANCHOR_FRAC_GRID = (0.2, 0.3, 0.5)


def check_against_baseline(exp, bdir: Path) -> None:
    bexp = load_exp_config(bdir)
    norm = lambda v: list(v) if isinstance(v, (list, tuple)) else v  # noqa: E731
    diff = {k: (norm(getattr(bexp, k)), norm(getattr(exp, k))) for k in CRITICAL_KEYS
            if norm(getattr(bexp, k)) != norm(getattr(exp, k))}
    if diff:
        raise SystemExit(f"[quantum] settings differ from baseline run {bdir}: {diff}\n"
                         f"Use the same --preset and overrides as the baseline run (paired comparison).")


def read_cases(bdir: Path, split: str) -> list[dict]:
    p = bdir / "cases" / f"cases_{split}.csv"
    if not p.exists():
        raise SystemExit(f"missing {p}: run the baseline 'cases' stage first")
    return pd.read_csv(p).to_dict("records")


def baseline_budget(bdir: Path, exp) -> int:
    if exp.budget_short:
        return int(exp.budget_short)
    p = bdir / "validation" / "selected_budget.json"
    if not p.exists():
        raise SystemExit(f"missing {p}: run the baseline 'validate' stage first (or pass --budget-short)")
    return int(json.loads(p.read_text(encoding="utf-8"))["budget_short"])


def stage_validate(out_dir, bdir, sys_, exp, log, quiet, overwrite):
    from common.tables import load_csv, val_budget_table
    vdir = out_dir / "validation"
    vdir.mkdir(parents=True, exist_ok=True)
    B_sel = baseline_budget(bdir, exp)
    specs = []
    for B in exp.budgets_val:
        for k in ("rand_qpso", "mlp_qpso", "gnn_qpso"):
            specs.append(make_spec(k, B, exp.budget_full, exp.delta_adjust))
    specs.append(make_spec("full_qpso", B_sel, exp.budget_full, exp.delta_adjust))
    for w0, fr in itertools.product(ANCHOR_W0_GRID, ANCHOR_FRAC_GRID):
        specs.append(make_spec("gnn_aqpso", B_sel, exp.budget_full, exp.delta_adjust,
                               anchor_w0=w0, anchor_frac=fr, tag=f"w{w0:g}_f{fr:g}"))
    csv_path = vdir / "val_trials.csv"
    run_trial_stage(sys_, exp, read_cases(bdir, "val"), specs, bdir / "models", csv_path, "validate",
                    ["rand_qpso", "gnn_qpso", "full_qpso"], quiet, log, overwrite)
    val = load_csv(csv_path)
    val_budget_table(val[val["method"] != "gnn_aqpso"]).to_csv(vdir / "val_budget_summary.csv", index=False)
    g = val[val["method"] == "gnn_aqpso"]
    grid = g.groupby(["anchor_w0", "anchor_frac"]).agg(passed=("passed", "sum"), trials=("passed", "size"),
                                                       gated=("gated", "mean")).reset_index()
    grid.to_csv(vdir / "anchor_grid.csv", index=False)
    # validation-only selection: max passed, then max gated, then smaller w0 (deterministic)
    best = grid.sort_values(["passed", "gated", "anchor_w0", "anchor_frac"],
                            ascending=[False, False, True, True]).iloc[0]
    sel = dict(anchor_w0=float(best["anchor_w0"]), anchor_frac=float(best["anchor_frac"]),
               budget_short=B_sel, passed=int(best["passed"]), trials=int(best["trials"]), gated=float(best["gated"]))
    (vdir / "selected_anchor.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
    msg = f"[validate] selected anchor_w0={sel['anchor_w0']} anchor_frac={sel['anchor_frac']} (B={B_sel})"
    log.info(msg)
    if not quiet:
        print(msg)


def selected_anchor(out_dir: Path, args, exp) -> tuple[float, float]:
    if args.anchor_w0 is not None and args.anchor_frac is not None:
        return float(args.anchor_w0), float(args.anchor_frac)
    p = out_dir / "validation" / "selected_anchor.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        w0 = float(args.anchor_w0) if args.anchor_w0 is not None else d["anchor_w0"]
        fr = float(args.anchor_frac) if args.anchor_frac is not None else d["anchor_frac"]
        return w0, fr
    return exp.anchor_w0, exp.anchor_frac


def stage_test(out_dir, bdir, sys_, exp, methods, args, log, quiet, overwrite):
    B = baseline_budget(bdir, exp)
    w0, fr = selected_anchor(out_dir, args, exp)
    log.info(f"[test] budget_short={B} anchor_w0={w0} anchor_frac={fr} gate={exp.anchor_gate} "
             f"beta={exp.beta_start}->{exp.beta_end} boundary={exp.boundary}")
    specs = [make_spec(k, B, exp.budget_full, exp.delta_adjust, anchor_w0=w0, anchor_frac=fr) for k in methods]
    run_trial_stage(sys_, exp, read_cases(bdir, "test"), specs, bdir / "models", out_dir / "test" / "trials.csv",
                    "test", [m for m in ("gnn_qpso", "gnn_aqpso", "full_qpso") if m in methods], quiet, log, overwrite)


def stage_tables(out_dir, bdir, sys_, exp, log, quiet, plots):
    from common.tables import load_csv, make_all
    test_p = out_dir / "test" / "trials.csv"
    val_p = out_dir / "validation" / "val_trials.csv"
    test = load_csv(test_p) if test_p.exists() else None
    val = load_csv(val_p) if val_p.exists() else None
    if val is not None:
        val = val[(val["method"] != "gnn_aqpso")]   # grid shown separately (fig_qpso_anchor_grid)
    cases_p = bdir / "cases" / "cases_test.csv"
    cases = pd.read_csv(cases_p) if cases_p.exists() else None
    pairs = [("gnn_qpso", "rand_qpso"), ("gnn_qpso", "mlp_qpso"), ("gnn_qpso", "full_qpso"),
             ("gnn_aqpso", "gnn_qpso"), ("gnn_aqpso", "rand_qpso")]
    make_all(out_dir, sys_, exp, test, val, labels_dir=None, models_dir=bdir / "models", cases=cases,
             ref="gnn_qpso", pairs=pairs, plots=plots, optimizer_name="QPSO", quiet=quiet, log=log)
    if plots and val_p.exists():
        from common.tables import fig_anchor_grid
        fig_anchor_grid(load_csv(val_p), out_dir / "figures")
    if not quiet:
        print(f"[tables  ] -> {out_dir / 'tables'}  figures -> {out_dir / 'figures'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Quantum-inspired QPSO pipeline (classical CPU)")
    add_common_args(ap)
    add_quantum_args(ap)
    args = ap.parse_args()
    sys_ = SysConfig()
    exp = exp_config_from_args(args)
    bdir = resolve_run_dir(args.baseline_dir, f"baseline_{args.preset}")
    out_dir = resolve_run_dir(args.out_dir, f"quantum_{args.preset}")
    stages = [s for s in parse_stages(args.stage) if s in ("validate", "test", "tables")]
    methods = list(args.methods) if args.methods else list(QUANTUM_METHODS)
    check_against_baseline(exp, bdir)
    if args.dry_run:
        from baseline_code.run_baseline import dry_run
        dry_run(sys_, exp, stages, methods)
        print(f"  + validate anchor grid {len(ANCHOR_W0_GRID)}x{len(ANCHOR_FRAC_GRID)} (gnn_aqpso) on validation")
        return 0
    log = init_run(out_dir, sys_, exp, args, "quantum", overwrite=args.overwrite)
    log.info(f"baseline_dir={bdir}")
    print(f"[quantum] out_dir={out_dir} baseline_dir={bdir} stages={stages} workers={exp.workers}")
    for st in stages:
        t0 = time.perf_counter()
        log.info(f"--- stage {st} start")
        if st == "validate":
            stage_validate(out_dir, bdir, sys_, exp, log, args.quiet, args.overwrite)
        elif st == "test":
            stage_test(out_dir, bdir, sys_, exp, methods, args, log, args.quiet, args.overwrite)
        elif st == "tables":
            stage_tables(out_dir, bdir, sys_, exp, log, args.quiet, not args.no_plots)
        log.info(f"--- stage {st} done in {time.perf_counter() - t0:.1f}s")
    log.info("=== quantum run end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
