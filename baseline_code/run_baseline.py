"""Baseline pipeline of the source paper (claude.md 8).

Stages: cases -> labels -> train -> validate -> test -> tables
    python baseline_code/run_baseline.py --preset quick --stage all --out-dir bl_quick_01
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.channel import build_cases  # noqa: E402
from common.cli import (add_common_args, exp_config_from_args, init_run,  # noqa: E402
                        parse_stages, resolve_run_dir)
from common.config import SysConfig  # noqa: E402
from common.methods import BASELINE_METHODS, make_spec  # noqa: E402
from common.progress import ProgressLine  # noqa: E402
from common.runner import append_rows, done_groups, run_parallel, run_trial_stage  # noqa: E402

EVAL_MS_PER_1E4 = 2.0   # measured on the run machine: ~58 ms / 30 particles at M=1e4 (sanity, 2026-11-21)


# ---------------------------------------------------------------------------
def read_cases(out_dir: Path, split: str) -> list[dict]:
    p = out_dir / "cases" / f"cases_{split}.csv"
    if not p.exists():
        raise SystemExit(f"missing {p}: run --stage cases first")
    return pd.read_csv(p).to_dict("records")


def stage_cases(out_dir, sys_, exp, log, quiet):
    d = out_dir / "cases"
    d.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        rows = build_cases(sys_, exp, split)
        pd.DataFrame(rows).to_csv(d / f"cases_{split}.csv", index=False)
        log.info(f"[cases] {split}: {len(rows)} cases")
        if not quiet:
            print(f"[cases   ] {split}: {len(rows)} cases -> {d / f'cases_{split}.csv'}")


def stage_labels(out_dir, sys_, exp, log, quiet, overwrite):
    from common.labels import assemble_labels_npz, label_group, save_group_feats
    ldir = out_dir / "labels"
    ldir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for split in ("train", "val"):
        csv_path = ldir / f"labels_{split}.csv"
        if overwrite and csv_path.exists():
            csv_path.unlink()
        skip = done_groups(csv_path)
        groups: dict = {}
        for c in read_cases(out_dir, split):
            groups.setdefault((int(c["layout_id"]), int(c["K"])), []).append(c)
        for (lid, K), cs in sorted(groups.items()):
            key = f"{lid}_K{K}"
            if key in skip:
                continue
            tasks.append(dict(group_key=key, sys=sys_, exp=exp, split=split, layout_id=lid, K=K,
                              cases=sorted(cs, key=lambda c: int(c["case_idx"])), csv=str(csv_path)))
    log.info(f"[labels] todo groups={len(tasks)} workers={exp.workers}")
    if not tasks:
        if not quiet:
            print("[labels  ] nothing to do (all groups present)")
        assemble_labels_npz(ldir)
        return
    pl = ProgressLine("labels", len(tasks), unit="grp", quiet=quiet)
    st = dict(done=0, cases=0, lab=0, feas=0)
    t0 = time.perf_counter()

    def on_result(task, res):
        append_rows(Path(task["csv"]), res["rows"])
        save_group_feats(ldir / "feat", task["group_key"], res["feats"])
        st["done"] += 1
        st["cases"] += len(res["rows"])
        st["lab"] += sum(int(r["has_label"]) for r in res["rows"])
        st["feas"] += sum(int(r["pso_best_feasible"]) for r in res["rows"])
        pl.update(st["done"], f"feas={100 * st['feas'] / st['cases']:.1f}% cov={st['lab']}/{st['cases']}")
        log.info(f"[labels] {task['split']} {task['group_key']}: labelled {sum(int(r['has_label']) for r in res['rows'])}"
                 f"/{len(res['rows'])}")

    run_parallel(tasks, label_group, exp.workers, on_result, log)
    msg = (f"[labels  ] done {len(tasks)} groups in {time.perf_counter() - t0:.1f}s; "
           f"coverage {st['lab']}/{st['cases']}")
    pl.close(msg)
    log.info(msg)
    assemble_labels_npz(ldir)


def stage_train(out_dir, sys_, exp, log, quiet, overwrite):
    from common.labels import load_label_set
    from common.train import fit_normalizer, resolve_device, train_one
    ldir = out_dir / "labels"
    mdir = out_dir / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    tr = load_label_set(ldir, "train")
    va = load_label_set(ldir, "val")
    if not tr["node"]:
        raise SystemExit("no labelled training graphs: check labels stage")
    norm = fit_normalizer(tr)
    device = resolve_device(exp.device)
    log.info(f"[train] n_train={len(tr['node'])} n_val={len(va['node'])} device={device}")
    if not quiet:
        print(f"[train   ] labelled graphs: train={len(tr['node'])} val={len(va['node'])} device={device}")
    metas = []
    for arch in ("gnn", "mlp"):
        for r in range(exp.replicates):
            seed = exp.train_seeds[r]
            ck = mdir / f"{arch}_seed{seed}.pt"
            if ck.exists() and not overwrite:
                from common.models import load_checkpoint
                _, _, meta = load_checkpoint(ck, sys_)
                metas.append(meta)
                if not quiet:
                    print(f"[train   ] {ck.name} exists (epoch {meta['epoch']}), skip")
                continue
            metas.append(train_one(arch, seed, tr, va, norm, sys_, exp, device, mdir, quiet, log))
    (mdir / "checkpoints.json").write_text(json.dumps(metas, indent=2), encoding="utf-8")


def baseline_val_specs(exp):
    specs = []
    for B in exp.budgets_val:
        for k in ("rand_pso", "mlp_pso", "gnn_pso"):
            specs.append(make_spec(k, B, exp.budget_full, exp.delta_adjust))
    for k in ("full_pso", "direct_mlp", "direct_gnn", "direct_mlp_raw", "direct_gnn_raw"):
        specs.append(make_spec(k, exp.budgets_val[0], exp.budget_full, exp.delta_adjust))
    return specs


def select_budget(val: pd.DataFrame, exp, full_key="full_pso", learned=("mlp_pso", "gnn_pso")) -> dict:
    """Smallest budget where BOTH learned initializers lose <= 1 pp pass and keep >= 95% gated of Full."""
    f = val[val["method"] == full_key]
    p_full = 100 * f["passed"].mean()
    g_full = f["gated"].mean()
    rows, chosen = [], None
    for B in sorted(exp.budgets_val):
        ok_all = True
        for m in learned:
            g = val[(val["method"] == m) & (val["budget"] == B)]
            p, gr = 100 * g["passed"].mean(), g["gated"].mean()
            ok = (p_full - p) <= 1.0 and gr >= 0.95 * g_full
            ok_all &= ok
            rows.append(dict(budget=int(B), method=m, pass_pct=p, gated=gr, ok=bool(ok)))
        if ok_all and chosen is None:
            chosen = int(B)
    met = chosen is not None
    return dict(budget_short=chosen if met else int(max(exp.budgets_val)), target_met=met,
                full_pass_pct=p_full, full_gated=g_full, checks=rows)


def stage_validate(out_dir, sys_, exp, log, quiet, overwrite):
    from common.tables import load_csv, val_budget_table
    vdir = out_dir / "validation"
    vdir.mkdir(parents=True, exist_ok=True)
    csv_path = vdir / "val_trials.csv"
    run_trial_stage(sys_, exp, read_cases(out_dir, "val"), baseline_val_specs(exp), out_dir / "models",
                    csv_path, "validate", ["rand_pso", "mlp_pso", "gnn_pso", "full_pso"], quiet, log, overwrite)
    val = load_csv(csv_path)
    val_budget_table(val).to_csv(vdir / "val_budget_summary.csv", index=False)
    sel = select_budget(val, exp)
    (vdir / "selected_budget.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
    msg = (f"[validate] selected budget_short={sel['budget_short']} target_met={sel['target_met']} "
           f"(Full PSO pass {sel['full_pass_pct']:.2f}% gated {sel['full_gated']:.4f})")
    log.info(msg)
    if not quiet:
        print(msg)


def short_budget(out_dir: Path, exp) -> int:
    if exp.budget_short:
        return int(exp.budget_short)
    p = out_dir / "validation" / "selected_budget.json"
    if not p.exists():
        raise SystemExit(f"missing {p}: run --stage validate first (or pass --budget-short)")
    return int(json.loads(p.read_text(encoding="utf-8"))["budget_short"])


def stage_test(out_dir, sys_, exp, methods, log, quiet, overwrite):
    B = short_budget(out_dir, exp)
    specs = [make_spec(k, B, exp.budget_full, exp.delta_adjust) for k in methods]
    run_trial_stage(sys_, exp, read_cases(out_dir, "test"), specs, out_dir / "models",
                    out_dir / "test" / "trials.csv", "test", [m for m in ("gnn_pso", "full_pso") if m in methods],
                    quiet, log, overwrite)


def stage_tables(out_dir, sys_, exp, log, quiet, plots):
    from common.tables import load_csv, make_all
    test_p = out_dir / "test" / "trials.csv"
    val_p = out_dir / "validation" / "val_trials.csv"
    test = load_csv(test_p) if test_p.exists() else None
    val = load_csv(val_p) if val_p.exists() else None
    cases_p = out_dir / "cases" / "cases_test.csv"
    cases = pd.read_csv(cases_p) if cases_p.exists() else None
    make_all(out_dir, sys_, exp, test, val, labels_dir=out_dir / "labels", models_dir=out_dir / "models",
             cases=cases, ref="gnn_pso", pairs=[("gnn_pso", "rand_pso"), ("gnn_pso", "mlp_pso"), ("gnn_pso", "full_pso")],
             plots=plots, optimizer_name="PSO", quiet=quiet, log=log)
    if not quiet:
        print(f"[tables  ] -> {out_dir / 'tables'}  figures -> {out_dir / 'figures'}")


# ---------------------------------------------------------------------------
def dry_run(sys_, exp, stages, methods):
    nK_tr, nK_te = len(exp.k_train), len(exp.k_test)
    n_tr = exp.n_train_layouts * nK_tr * exp.cases_per_layout
    n_va = exp.n_val_layouts * nK_tr * exp.cases_per_layout
    n_te = exp.n_test_layouts * nK_te * exp.cases_per_layout
    ev = EVAL_MS_PER_1E4 * exp.search_samples / 1e4
    ass = 30.0 * exp.assess_samples / 1e5
    B = exp.budget_short or exp.budgets_val[0]
    lab_ms = (n_tr + n_va) * exp.label_runs * exp.budget_full * ev
    val_evals = 3 * sum(exp.budgets_val) + exp.budget_full
    val_ms = n_va * exp.replicates * (val_evals * ev + 7 * ass)
    te_evals = sum(exp.budget_full if m.startswith("full_") else (0 if m.startswith("direct") else B)
                   for m in methods)
    te_ms = n_te * exp.replicates * (te_evals * ev + len(methods) * ass)
    W = max(1, exp.workers)
    print(f"[dry-run] preset={exp.preset} workers={W} device={exp.device}")
    print(f"  cases: train={n_tr} val={n_va} test={n_te}  (K train {exp.k_train}, K test {exp.k_test})")
    print(f"  samples: search={exp.search_samples} assess={exp.assess_samples} label={exp.label_samples}")
    print(f"  test trials per method = {n_te * exp.replicates}; methods = {list(methods)}; short budget ~ {B}")
    est = {"labels": lab_ms, "validate": val_ms, "test": te_ms}
    for st in stages:
        if st in est:
            print(f"  ~{st:<8}: {est[st] / 1e3 / 60:.1f} CPU-min  -> ~{est[st] / 1e3 / 60 / W:.1f} min with {W} workers")
    print("  (uoc tinh tho: 1 objective evaluation ~ "
          f"{ev:.2f} ms; train 6 models tren GPU ~ vai phut)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Baseline pipeline (PSO / MLP / GNN)")
    add_common_args(ap)
    args = ap.parse_args()
    sys_ = SysConfig()
    exp = exp_config_from_args(args)
    out_dir = resolve_run_dir(args.out_dir, f"baseline_{args.preset}")
    stages = parse_stages(args.stage)
    methods = list(args.methods) if args.methods else list(BASELINE_METHODS)
    if args.dry_run:
        dry_run(sys_, exp, stages, methods)
        return 0
    log = init_run(out_dir, sys_, exp, args, "baseline", overwrite=args.overwrite)
    print(f"[baseline] out_dir={out_dir} stages={stages} workers={exp.workers}")
    for st in stages:
        t0 = time.perf_counter()
        log.info(f"--- stage {st} start")
        if st == "cases":
            stage_cases(out_dir, sys_, exp, log, args.quiet)
        elif st == "labels":
            stage_labels(out_dir, sys_, exp, log, args.quiet, args.overwrite)
        elif st == "train":
            stage_train(out_dir, sys_, exp, log, args.quiet, args.overwrite)
        elif st == "validate":
            stage_validate(out_dir, sys_, exp, log, args.quiet, args.overwrite)
        elif st == "test":
            stage_test(out_dir, sys_, exp, methods, log, args.quiet, args.overwrite)
        elif st == "tables":
            stage_tables(out_dir, sys_, exp, log, args.quiet, not args.no_plots)
        log.info(f"--- stage {st} done in {time.perf_counter() - t0:.1f}s")
    log.info("=== baseline run end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
