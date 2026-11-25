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

import dataclasses  # noqa: E402

import pandas as pd  # noqa: E402

from common.cli import (CRITICAL_KEYS, add_common_args, add_quantum_args,  # noqa: E402
                        exp_config_from_args, init_run, load_exp_config, parse_stages,
                        resolve_run_dir)
from common.config import SysConfig  # noqa: E402
from common.methods import (DPRIOR_BASE_METHODS, DPRIOR_METHODS, FPRIOR_BASE_METHODS, FPRIOR_METHODS,  # noqa: E402
                            POLISH_METHODS, PRIOR_PSO_METHODS, UPRIOR_BASE_METHODS, UPRIOR_METHODS,
                            UPRIOR_PLAIN_METHODS,
                            PRIOR_BASE_METHODS, PRIOR_METHODS, PW_METHODS, QQ_METHODS, QQ_OFF,
                            QUANTUM_METHODS, QW_METHODS, make_spec)
from common.runner import run_trial_stage  # noqa: E402

# Quick run (2026-11-21) selected (0.75, 0.5) = upper corner of the 3x3 grid -> grid extended.
ANCHOR_W0_GRID = (0.25, 0.5, 0.75, 1.0)
ANCHOR_FRAC_GRID = (0.2, 0.3, 0.5, 0.7)
# gnn_qqpso validation grid (2 x 2 x len(polish grid)); (mean, linear, 0) == plain gnn_qpso.
QQ_MBEST_GRID = ("mean", "feas_rank")
QQ_BETA_GRID = ("linear", "feas")
QQ_POLISH_GRID = (0, 1, 2)
GRID_METHODS = ("gnn_aqpso", "gnn_qqpso", "gnn_qwqpso", "gnn_pso_wpt", "prior_qwqpso", "prior_direct",
                "prior_polish", "prior_dqpso", "prior_ddirect",
                "prior_dpolish", "prior_pso")   # validation rows of these keys are grid points
# gnn_qwqpso validation grid: W (well_scale, well_frac) x B (beta_start, beta_end) x P (polish_pops).
# (0, 0) = no well; (1.0, 0.5) = default beta; (0, 0, 1.0, 0.5, P=2) == selected gnn_qqpso.
QW_WELL_GRID = ((0.0, 0.0), (0.05, 0.5), (0.05, 1.0), (0.1, 0.5), (0.1, 1.0), (0.2, 0.5), (0.2, 1.0))
QW_BETA_GRID = ((1.0, 0.5), (0.75, 0.35), (0.5, 0.25))
QW_POLISH_GRID = (1, 2)
# gnn_pso_wpt validation grid, same size as gnn_qwqpso's: W (same 7 cells) x PSO inertia schedule
# (3 cells; (0.9, 0.4) = Algorithm 1) x P (1, 2) = 42 cells.
PW_INERTIA_GRID = ((0.9, 0.4), (0.7, 0.3), (0.5, 0.2))
PW_POLISH_GRID = (1, 2)
PW_DEFAULT_INERTIA = (0.9, 0.4)
# prior_qwqpso validation grid: RIS share of the prior x well length (full swarm) x beta schedule,
# polishing depth 1 (selected for gnn_qwqpso) -> 5 x 3 x 2 = 30 cells.
PRIOR_S_GRID = (0.2, 0.35, 0.5, 0.65, 0.8)
PRIOR_WELL_GRID = (0.05, 0.1, 0.2)
PRIOR_BETA_GRID = ((0.5, 0.25), (1.0, 0.5))
PRIOR_POLISH = 1
# prior_dqpso validation grid (DEP-aware prior + boxed QW-QPSO): s x L x beta = 2 x 3 x 2 = 12 cells,
# P = 1. Swarm-free references prior_ddirect / prior_dpolish choose only s on the same s grid.
# Pre-registered rule: max passed -> max gated -> narrower well -> larger beta_start -> smaller s.
DPRIOR_S_GRID = (0.2, 0.5)
DPRIOR_WELL_GRID = (0.05, 0.1, 0.2)
DPRIOR_BETA_GRID = ((0.5, 0.25), (1.0, 0.5))
DPRIOR_POLISH = 1
# --prior-polish 0 (pure swarm, no boundary polish): the swarm must travel from the prior to the
# boundary by itself, so the grid adds a wider well and a stronger contraction:
# s {0.2, 0.5} x L {0.05, 0.1, 0.2, 0.4} x beta {0.5->0.25, 1.0->0.5, 1.0->0.25} = 24 cells per prior.
NOPOL_S_GRID = (0.2, 0.5)
NOPOL_WELL_GRID = (0.05, 0.1, 0.2, 0.4)
NOPOL_BETA_GRID = ((0.5, 0.25), (1.0, 0.5), (1.0, 0.25))
# prior_pso grid = prior_qwqpso grid with the QPSO beta schedule replaced by a PSO inertia schedule
# (same number of cells): (0.9, 0.4) = Algorithm 1, (0.5, 0.2) = contracted (like beta 0.5 -> 0.25),
# and for --prior-polish 0 a third (0.7, 0.3). Same selection rule as prior_qwqpso.
PPSO_INERTIA_GRID = ((0.9, 0.4), (0.5, 0.2))
NOPOL_INERTIA_GRID = ((0.9, 0.4), (0.7, 0.3), (0.5, 0.2))


def prior_tag(s_: float, w: dict) -> str:
    return f"S{s_:g}_" + qw_tag(w)


def pw_tag(w: dict) -> str:
    return (f"W{w['well_scale']:g}x{w['well_frac']:g}_I{w['inertia_start']:g}-{w['inertia_end']:g}"
            f"_P{int(w['polish_pops'])}")


def filter_k(cases: list[dict], args) -> list[dict]:
    ks = getattr(args, "k_subset", None)
    if not ks:
        return cases
    out = [c for c in cases if int(c["K"]) in set(ks)]
    assert out, f"--k-subset {ks} leaves no cases"
    return out


def qw_tag(w: dict) -> str:
    return (f"W{w['well_scale']:g}x{w['well_frac']:g}_B{w['beta_start']:g}-{w['beta_end']:g}"
            f"_P{int(w['polish_pops'])}")


def qq_tag(q: dict) -> str:
    return f"F{q['mbest_mode']}_B{q['beta_mode']}_P{int(q['polish_pops'])}"


def qq_ncomp(q: dict) -> int:
    return int(q["mbest_mode"] != QQ_OFF["mbest_mode"]) + int(q["beta_mode"] != QQ_OFF["beta_mode"]) \
        + int(int(q["polish_pops"]) != QQ_OFF["polish_pops"])


def check_against_baseline(exp, bdir: Path) -> None:
    bexp = load_exp_config(bdir)
    norm = lambda v: list(v) if isinstance(v, (list, tuple)) else v  # noqa: E731
    diff = {k: (norm(getattr(bexp, k)), norm(getattr(exp, k))) for k in CRITICAL_KEYS
            if norm(getattr(bexp, k)) != norm(getattr(exp, k))}
    if diff:
        raise SystemExit(f"[quantum] settings differ from baseline run {bdir}: {diff}\n"
                         f"Use the same --preset and overrides as the baseline run (paired comparison).")


def read_cases(bdir: Path, split: str, sys_=None, exp=None) -> list[dict]:
    p = bdir / "cases" / f"cases_{split}.csv"
    if not p.exists():
        raise SystemExit(f"missing {p}: run the baseline 'cases' stage first")
    rows = pd.read_csv(p).to_dict("records")
    if split != "val" or exp is None or not exp.k_val or set(exp.k_val) == set(exp.k_train):
        return rows
    # --k-val: rebuild validation cases for the requested K from the same layouts / LHS pairs.
    from common.channel import build_cases
    assert max(exp.k_val) <= exp.K_max, f"k_val {exp.k_val} exceeds layout K_max={exp.K_max}"
    new = build_cases(sys_, dataclasses.replace(exp, k_train=tuple(exp.k_val)), "val")
    old = {r["case_id"]: r for r in rows}
    n_chk = 0
    for r in new:                      # cases that exist in the baseline file must match exactly
        o = old.get(r["case_id"])
        if o is not None:
            assert abs(float(o["P_tot_dbm"]) - r["P_tot_dbm"]) < 1e-9 and abs(float(o["xi_req"]) - r["xi_req"]) < 1e-9, \
                f"rebuilt val case {r['case_id']} differs from baseline cases_val.csv"
            n_chk += 1
    assert n_chk > 0 or not old, "no overlap between rebuilt and baseline validation cases"
    return new


def baseline_budget(bdir: Path, exp) -> int:
    if exp.budget_short:
        return int(exp.budget_short)
    p = bdir / "validation" / "selected_budget.json"
    if not p.exists():
        raise SystemExit(f"missing {p}: run the baseline 'validate' stage first (or pass --budget-short)")
    return int(json.loads(p.read_text(encoding="utf-8"))["budget_short"])


def _grid_free(df: pd.DataFrame) -> pd.DataFrame:
    """Drop validation grid rows (anchor grid / QQ grid) before generic tables."""
    if df is None or df.empty or "tag" not in df.columns:
        return df
    return df[~(df["method"].isin(GRID_METHODS) & (df["tag"].astype(str) != ""))]


def stage_validate(out_dir, bdir, sys_, exp, methods, args, log, quiet, overwrite):
    from common.tables import add_kgroup, load_csv, val_budget_table
    vdir = out_dir / "validation"
    vdir.mkdir(parents=True, exist_ok=True)
    B_sel = baseline_budget(bdir, exp)
    T = B_sel // exp.swarm
    pp_old = PRIOR_POLISH if args.prior_polish is None else int(args.prior_polish)
    pp_d = DPRIOR_POLISH if args.prior_polish is None else int(args.prior_polish)
    assert 0 <= pp_old <= T - 1 and 0 <= pp_d <= T - 1, (pp_old, pp_d, T)
    specs = []
    for B in exp.budgets_val:
        for k in ("rand_qpso", "mlp_qpso", "gnn_qpso"):
            if k in methods:
                specs.append(make_spec(k, B, exp.budget_full, exp.delta_adjust))
    if "full_qpso" in methods:
        specs.append(make_spec("full_qpso", B_sel, exp.budget_full, exp.delta_adjust))
    if "gnn_aqpso" in methods:
        for w0, fr in itertools.product(ANCHOR_W0_GRID, ANCHOR_FRAC_GRID):
            specs.append(make_spec("gnn_aqpso", B_sel, exp.budget_full, exp.delta_adjust,
                                   anchor_w0=w0, anchor_frac=fr, tag=f"w{w0:g}_f{fr:g}"))
    use_qq = any(m in QQ_METHODS for m in methods)
    if use_qq:
        pgrid = tuple(args.qq_polish_grid) if args.qq_polish_grid else QQ_POLISH_GRID
        for mb_m, be_m, pp in itertools.product(QQ_MBEST_GRID, QQ_BETA_GRID, pgrid):
            if pp > T - 1:
                log.warning(f"[validate] skip qq polish_pops={pp}: budget {B_sel} has only {T - 1} updates")
                continue
            q = dict(mbest_mode=mb_m, beta_mode=be_m, polish_pops=int(pp))
            specs.append(make_spec("gnn_qqpso", B_sel, exp.budget_full, exp.delta_adjust, tag=qq_tag(q), qq=q))
    use_qw = any(m in QW_METHODS for m in methods)
    if use_qw:
        pg = tuple(args.qw_polish_grid) if args.qw_polish_grid else QW_POLISH_GRID
        for (ws, wf), (bs, be), pp in itertools.product(QW_WELL_GRID, QW_BETA_GRID, pg):
            if pp > T - 1:
                log.warning(f"[validate] skip qw polish_pops={pp}: budget {B_sel} has only {T - 1} updates")
                continue
            w = dict(well_scale=ws, well_frac=wf, beta_start=bs, beta_end=be, polish_pops=int(pp))
            specs.append(make_spec("gnn_qwqpso", B_sel, exp.budget_full, exp.delta_adjust, tag=qw_tag(w), qw=w))
    use_prior = "prior_qwqpso" in methods
    if use_prior:
        grids = ((NOPOL_S_GRID, NOPOL_WELL_GRID, NOPOL_BETA_GRID) if pp_old == 0
                 else (PRIOR_S_GRID, PRIOR_WELL_GRID, PRIOR_BETA_GRID))
        for s_, L, (bs, be) in itertools.product(*grids):
            w = dict(well_scale=L, well_frac=1.0, beta_start=bs, beta_end=be, polish_pops=pp_old)
            specs.append(make_spec("prior_qwqpso", B_sel, exp.budget_full, exp.delta_adjust,
                                   tag=prior_tag(s_, w), qw=w, prior_s=s_))
    use_ppso = "prior_pso" in methods
    if use_ppso:
        grids = ((NOPOL_S_GRID, NOPOL_WELL_GRID, NOPOL_INERTIA_GRID) if pp_old == 0
                 else (PRIOR_S_GRID, PRIOR_WELL_GRID, PPSO_INERTIA_GRID))
        for s_, L, (i_s, i_e) in itertools.product(*grids):
            w = dict(well_scale=L, well_frac=1.0, inertia_start=i_s, inertia_end=i_e, polish_pops=pp_old)
            specs.append(make_spec("prior_pso", B_sel, exp.budget_full, exp.delta_adjust,
                                   tag=f"S{s_:g}_" + pw_tag(w), pw=w, prior_s=s_))
    use_dprior = "prior_dqpso" in methods
    if use_dprior:
        grids = ((NOPOL_S_GRID, NOPOL_WELL_GRID, NOPOL_BETA_GRID) if pp_d == 0
                 else (DPRIOR_S_GRID, DPRIOR_WELL_GRID, DPRIOR_BETA_GRID))
        for s_, L, (bs, be) in itertools.product(*grids):
            w = dict(well_scale=L, well_frac=1.0, beta_start=bs, beta_end=be, polish_pops=pp_d)
            specs.append(make_spec("prior_dqpso", B_sel, exp.budget_full, exp.delta_adjust,
                                   tag=prior_tag(s_, w), qw=w, prior_s=s_))
    base_m = [m for m in PRIOR_BASE_METHODS + DPRIOR_BASE_METHODS if m in methods]
    for m in base_m:                                  # swarm-free references: only s is chosen
        for s_ in (DPRIOR_S_GRID if m in DPRIOR_BASE_METHODS else PRIOR_S_GRID):
            specs.append(make_spec(m, B_sel, exp.budget_full, exp.delta_adjust, tag=f"S{s_:g}", prior_s=s_))
    use_pw = "gnn_pso_wpt" in methods
    if use_pw:
        for (ws, wf), (i_s, i_e), pp in itertools.product(QW_WELL_GRID, PW_INERTIA_GRID, PW_POLISH_GRID):
            if pp > T - 1:
                log.warning(f"[validate] skip pw polish_pops={pp}: budget {B_sel} has only {T - 1} updates")
                continue
            w = dict(well_scale=ws, well_frac=wf, inertia_start=i_s, inertia_end=i_e, polish_pops=int(pp))
            specs.append(make_spec("gnn_pso_wpt", B_sel, exp.budget_full, exp.delta_adjust, tag=pw_tag(w), pw=w))
    if not specs:
        raise SystemExit("[validate] nothing to validate for --methods " + ",".join(methods))
    csv_path = vdir / "val_trials.csv"
    watch = [m for m in ("rand_qpso", "gnn_qpso", "full_qpso", "gnn_qqpso", "gnn_qwqpso", "gnn_pso_wpt",
                         "prior_qwqpso", "prior_dqpso", "prior_dpolish", "prior_pso")
             if any(sp.key == m for sp in specs)]
    run_trial_stage(sys_, exp, filter_k(read_cases(bdir, "val", sys_, exp), args), specs, bdir / "models", csv_path, "validate",
                    watch, quiet, log, overwrite)
    val = load_csv(csv_path)
    vb = val_budget_table(_grid_free(val))
    if not vb.empty:
        vb.to_csv(vdir / "val_budget_summary.csv", index=False)

    if "gnn_aqpso" in methods:
        g = val[val["method"] == "gnn_aqpso"]
        grid = g.groupby(["anchor_w0", "anchor_frac"]).agg(passed=("passed", "sum"), trials=("passed", "size"),
                                                           gated=("gated", "mean")).reset_index()
        grid.to_csv(vdir / "anchor_grid.csv", index=False)
        # validation-only selection: max passed, then max gated, then smaller w0 (deterministic)
        best = grid.sort_values(["passed", "gated", "anchor_w0", "anchor_frac"],
                                ascending=[False, False, True, True]).iloc[0]
        sel = dict(anchor_w0=float(best["anchor_w0"]), anchor_frac=float(best["anchor_frac"]),
                   budget_short=B_sel, passed=int(best["passed"]), trials=int(best["trials"]),
                   gated=float(best["gated"]))
        (vdir / "selected_anchor.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
        msg = f"[validate] selected anchor_w0={sel['anchor_w0']} anchor_frac={sel['anchor_frac']} (B={B_sel})"
        log.info(msg)
        if not quiet:
            print(msg)

    if use_qq:
        g = add_kgroup(val[val["method"] == "gnn_qqpso"].copy(), exp)
        g["qq_polish"] = g["qq_polish"].astype(int)
        keys = ["qq_mbest", "qq_beta", "qq_polish"]
        grid = g.groupby(keys).agg(passed=("passed", "sum"), trials=("passed", "size"),
                                   gated=("gated", "mean")).reset_index()
        for kg, gk in g.groupby("Kgroup"):            # per-K-group breakdown (reporting only)
            sub = gk.groupby(keys).agg(**{f"passed_K{kg}": ("passed", "sum"),
                                          f"gated_K{kg}": ("gated", "mean")}).reset_index()
            grid = grid.merge(sub, on=keys, how="left")
        grid["n_comp"] = [qq_ncomp(dict(mbest_mode=a, beta_mode=b, polish_pops=c))
                          for a, b, c in zip(grid["qq_mbest"], grid["qq_beta"], grid["qq_polish"])]
        grid.to_csv(vdir / "qq_grid.csv", index=False)
        # PRE-REGISTERED rule (validation only): max passed, then max gated, then fewer components,
        # then fewer polish populations.
        best = grid.sort_values(["passed", "gated", "n_comp", "qq_polish"],
                                ascending=[False, False, True, True]).iloc[0]
        sel = dict(mbest_mode=str(best["qq_mbest"]), beta_mode=str(best["qq_beta"]),
                   polish_pops=int(best["qq_polish"]), budget_short=B_sel, k_val=list(exp.k_val or exp.k_train),
                   passed=int(best["passed"]), trials=int(best["trials"]), gated=float(best["gated"]))
        (vdir / "selected_qq.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
        msg = (f"[validate] selected gnn_qqpso: F={sel['mbest_mode']} B={sel['beta_mode']} "
               f"P={sel['polish_pops']} (B={B_sel}, K_val={sel['k_val']}) "
               f"pass {sel['passed']}/{sel['trials']} gated {sel['gated']:.4f}")
        log.info(msg)
        log.info("qq grid:\n" + grid.to_string(index=False))
        if not quiet:
            print(msg)
            print(grid.sort_values(["passed", "gated"], ascending=False).to_string(index=False))


    if use_qw:
        g = add_kgroup(val[val["method"] == "gnn_qwqpso"].copy(), exp)
        g["qq_polish"] = g["qq_polish"].astype(int)
        keys = ["qw_scale", "qw_frac", "qw_bs", "qw_be", "qq_polish"]
        grid = g.groupby(keys).agg(passed=("passed", "sum"), trials=("passed", "size"),
                                   gated=("gated", "mean")).reset_index()
        for kg, gk in g.groupby("Kgroup"):            # per-K-group breakdown (reporting only)
            sub = gk.groupby(keys).agg(**{f"passed_K{kg}": ("passed", "sum"),
                                          f"gated_K{kg}": ("gated", "mean")}).reset_index()
            grid = grid.merge(sub, on=keys, how="left")
        grid["n_comp"] = [int(a > 0) + int((abs(b - exp.beta_start) > 1e-12) or (abs(c - exp.beta_end) > 1e-12))
                          for a, b, c in zip(grid["qw_scale"], grid["qw_bs"], grid["qw_be"])]
        grid.to_csv(vdir / "qw_grid.csv", index=False)
        # PRE-REGISTERED rule (validation only): max passed, then max gated, then fewer components,
        # then fewer polish populations, then smaller well.
        best = grid.sort_values(["passed", "gated", "n_comp", "qq_polish", "qw_scale", "qw_frac"],
                                ascending=[False, False, True, True, True, True]).iloc[0]
        sel = dict(well_scale=float(best["qw_scale"]), well_frac=float(best["qw_frac"]),
                   beta_start=float(best["qw_bs"]), beta_end=float(best["qw_be"]),
                   polish_pops=int(best["qq_polish"]), budget_short=B_sel, k_val=list(exp.k_val or exp.k_train),
                   passed=int(best["passed"]), trials=int(best["trials"]), gated=float(best["gated"]))
        (vdir / "selected_qw.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
        msg = (f"[validate] selected gnn_qwqpso: W=({sel['well_scale']:g},{sel['well_frac']:g}) "
               f"B=({sel['beta_start']:g}->{sel['beta_end']:g}) P={sel['polish_pops']} (B={B_sel}, "
               f"K_val={sel['k_val']}) pass {sel['passed']}/{sel['trials']} gated {sel['gated']:.4f}")
        log.info(msg)
        log.info("qw grid:\n" + grid.to_string(index=False))
        if not quiet:
            print(msg)
            print(grid.sort_values(["passed", "gated"], ascending=False).head(15).to_string(index=False))


    if use_pw:
        g = add_kgroup(val[val["method"] == "gnn_pso_wpt"].copy(), exp)
        g["qq_polish"] = g["qq_polish"].astype(int)
        keys = ["qw_scale", "qw_frac", "pw_ws", "pw_we", "qq_polish"]
        grid = g.groupby(keys).agg(passed=("passed", "sum"), trials=("passed", "size"),
                                   gated=("gated", "mean")).reset_index()
        for kg, gk in g.groupby("Kgroup"):
            sub = gk.groupby(keys).agg(**{f"passed_K{kg}": ("passed", "sum"),
                                          f"gated_K{kg}": ("gated", "mean")}).reset_index()
            grid = grid.merge(sub, on=keys, how="left")
        grid["n_comp"] = [int(a > 0) + int((abs(b - PW_DEFAULT_INERTIA[0]) > 1e-12) or
                                           (abs(c - PW_DEFAULT_INERTIA[1]) > 1e-12))
                          for a, b, c in zip(grid["qw_scale"], grid["pw_ws"], grid["pw_we"])]
        grid.to_csv(vdir / "pw_grid.csv", index=False)
        # SAME pre-registered rule as gnn_qwqpso: max passed, max gated, fewer components,
        # fewer polish populations, smaller well.
        best = grid.sort_values(["passed", "gated", "n_comp", "qq_polish", "qw_scale", "qw_frac"],
                                ascending=[False, False, True, True, True, True]).iloc[0]
        sel = dict(well_scale=float(best["qw_scale"]), well_frac=float(best["qw_frac"]),
                   inertia_start=float(best["pw_ws"]), inertia_end=float(best["pw_we"]),
                   polish_pops=int(best["qq_polish"]), budget_short=B_sel, k_val=list(exp.k_val or exp.k_train),
                   passed=int(best["passed"]), trials=int(best["trials"]), gated=float(best["gated"]))
        (vdir / "selected_pw.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
        msg = (f"[validate] selected gnn_pso_wpt: W=({sel['well_scale']:g},{sel['well_frac']:g}) "
               f"I=({sel['inertia_start']:g}->{sel['inertia_end']:g}) P={sel['polish_pops']} "
               f"pass {sel['passed']}/{sel['trials']} gated {sel['gated']:.4f}")
        log.info(msg)
        log.info("pw grid:\n" + grid.to_string(index=False))
        if not quiet:
            print(msg)


    if use_prior:
        g = add_kgroup(val[val["method"] == "prior_qwqpso"].copy(), exp)
        keys = ["prior_s", "qw_scale", "qw_bs", "qw_be"]
        grid = g.groupby(keys).agg(passed=("passed", "sum"), trials=("passed", "size"),
                                   gated=("gated", "mean")).reset_index()
        for kg, gk in g.groupby("Kgroup"):
            sub = gk.groupby(keys).agg(**{f"passed_K{kg}": ("passed", "sum"),
                                          f"gated_K{kg}": ("gated", "mean")}).reset_index()
            grid = grid.merge(sub, on=keys, how="left")
        grid.to_csv(vdir / "prior_grid.csv", index=False)
        # pre-registered rule: max passed, max gated, then narrower well, then larger beta_start
        best = grid.sort_values(["passed", "gated", "qw_scale", "qw_bs"],
                                ascending=[False, False, True, False]).iloc[0]
        sel = dict(prior_s=float(best["prior_s"]), well_scale=float(best["qw_scale"]), well_frac=1.0,
                   beta_start=float(best["qw_bs"]), beta_end=float(best["qw_be"]), polish_pops=pp_old,
                   budget_short=B_sel, k_val=list(exp.k_val or exp.k_train),
                   passed=int(best["passed"]), trials=int(best["trials"]), gated=float(best["gated"]))
        (vdir / "selected_prior.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
        msg = (f"[validate] selected prior_qwqpso: s={sel['prior_s']:g} L={sel['well_scale']:g} "
               f"beta=({sel['beta_start']:g}->{sel['beta_end']:g}) P={pp_old} "
               f"pass {sel['passed']}/{sel['trials']} gated {sel['gated']:.4f}")
        log.info(msg)
        log.info("prior grid:\n" + grid.to_string(index=False))
        if not quiet:
            print(msg)

    if use_ppso:
        g = add_kgroup(val[val["method"] == "prior_pso"].copy(), exp)
        keys = ["prior_s", "qw_scale", "pw_ws", "pw_we"]
        grid = g.groupby(keys).agg(passed=("passed", "sum"), trials=("passed", "size"),
                                   gated=("gated", "mean")).reset_index()
        for kk, gk in g.groupby("K"):
            sub = gk.groupby(keys).agg(**{f"passed_K{kk}": ("passed", "sum"),
                                          f"gated_K{kk}": ("gated", "mean")}).reset_index()
            grid = grid.merge(sub, on=keys, how="left")
        grid.to_csv(vdir / "ppso_grid.csv", index=False)
        # same rule as prior_qwqpso: max passed, max gated, narrower well, larger start (inertia)
        best = grid.sort_values(["passed", "gated", "qw_scale", "pw_ws", "prior_s"],
                                ascending=[False, False, True, False, True]).iloc[0]
        sel = dict(prior_s=float(best["prior_s"]), well_scale=float(best["qw_scale"]), well_frac=1.0,
                   inertia_start=float(best["pw_ws"]), inertia_end=float(best["pw_we"]), polish_pops=pp_old,
                   budget_short=B_sel, k_val=list(exp.k_val or exp.k_train),
                   passed=int(best["passed"]), trials=int(best["trials"]), gated=float(best["gated"]))
        (vdir / "selected_ppso.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
        msg = (f"[validate] selected prior_pso: s={sel['prior_s']:g} L={sel['well_scale']:g} "
               f"inertia=({sel['inertia_start']:g}->{sel['inertia_end']:g}) P={pp_old} "
               f"pass {sel['passed']}/{sel['trials']} gated {sel['gated']:.4f}")
        log.info(msg)
        log.info("ppso grid:\n" + grid.to_string(index=False))
        if not quiet:
            print(msg)

    if use_dprior:
        g = add_kgroup(val[val["method"] == "prior_dqpso"].copy(), exp)
        keys = ["prior_s", "qw_scale", "qw_bs", "qw_be"]
        grid = g.groupby(keys).agg(passed=("passed", "sum"), trials=("passed", "size"),
                                   gated=("gated", "mean")).reset_index()
        for kk, gk in g.groupby("K"):                  # per K (K_val may include 8, 10)
            sub = gk.groupby(keys).agg(**{f"passed_K{kk}": ("passed", "sum"),
                                          f"gated_K{kk}": ("gated", "mean")}).reset_index()
            grid = grid.merge(sub, on=keys, how="left")
        grid.to_csv(vdir / "dprior_grid.csv", index=False)
        best = grid.sort_values(["passed", "gated", "qw_scale", "qw_bs", "prior_s"],
                                ascending=[False, False, True, False, True]).iloc[0]
        sel = dict(prior_s=float(best["prior_s"]), well_scale=float(best["qw_scale"]), well_frac=1.0,
                   beta_start=float(best["qw_bs"]), beta_end=float(best["qw_be"]), polish_pops=pp_d,
                   budget_short=B_sel, k_val=list(exp.k_val or exp.k_train),
                   passed=int(best["passed"]), trials=int(best["trials"]), gated=float(best["gated"]))
        (vdir / "selected_dprior.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
        msg = (f"[validate] selected prior_dqpso: s={sel['prior_s']:g} L={sel['well_scale']:g} "
               f"beta=({sel['beta_start']:g}->{sel['beta_end']:g}) P={pp_d} "
               f"pass {sel['passed']}/{sel['trials']} gated {sel['gated']:.4f}")
        log.info(msg)
        log.info("dprior grid:\n" + grid.to_string(index=False))
        if not quiet:
            print(msg)


    if base_m:
        sel_all = {}
        for m in base_m:
            g = add_kgroup(val[val["method"] == m].copy(), exp)
            grid = g.groupby("prior_s").agg(passed=("passed", "sum"), trials=("passed", "size"),
                                            gated=("gated", "mean")).reset_index()
            for kg, gk in g.groupby("Kgroup"):
                sub = gk.groupby("prior_s").agg(**{f"passed_K{kg}": ("passed", "sum"),
                                                   f"gated_K{kg}": ("gated", "mean")}).reset_index()
                grid = grid.merge(sub, on="prior_s", how="left")
            grid.insert(0, "method", m)
            grid.to_csv(vdir / f"{m}_grid.csv", index=False)
            best = grid.sort_values(["passed", "gated", "prior_s"], ascending=[False, False, True]).iloc[0]
            sel_all[m] = dict(prior_s=float(best["prior_s"]), passed=int(best["passed"]),
                              trials=int(best["trials"]), gated=float(best["gated"]))
            msg = (f"[validate] selected {m}: s={sel_all[m]['prior_s']:g} pass {sel_all[m]['passed']}/"
                   f"{sel_all[m]['trials']} gated {sel_all[m]['gated']:.4f}")
            log.info(msg)
            log.info(f"{m} grid:\n" + grid.to_string(index=False))
            if not quiet:
                print(msg)
        (vdir / "selected_prior_base.json").write_text(json.dumps(sel_all, indent=2), encoding="utf-8")


def selected_prior_base(out_dir: Path) -> dict:
    p = out_dir / "validation" / "selected_prior_base.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def selected_prior(out_dir: Path) -> tuple[float, dict]:
    p = out_dir / "validation" / "selected_prior.json"
    if not p.exists():
        raise SystemExit(f"missing {p}: run stage 'validate' with prior_qwqpso first")
    d = json.loads(p.read_text(encoding="utf-8"))
    w = dict(well_scale=float(d["well_scale"]), well_frac=float(d["well_frac"]),
             beta_start=float(d["beta_start"]), beta_end=float(d["beta_end"]), polish_pops=int(d["polish_pops"]))
    return float(d["prior_s"]), w


def selected_dprior(out_dir: Path) -> tuple[float, dict]:
    p = out_dir / "validation" / "selected_dprior.json"
    if not p.exists():
        raise SystemExit(f"missing {p}: run stage 'validate' with prior_dqpso first")
    d = json.loads(p.read_text(encoding="utf-8"))
    w = dict(well_scale=float(d["well_scale"]), well_frac=float(d["well_frac"]),
             beta_start=float(d["beta_start"]), beta_end=float(d["beta_end"]), polish_pops=int(d["polish_pops"]))
    return float(d["prior_s"]), w


def selected_ppso(out_dir: Path) -> tuple[float, dict]:
    p = out_dir / "validation" / "selected_ppso.json"
    if not p.exists():
        raise SystemExit(f"missing {p}: run stage 'validate' with prior_pso first")
    d = json.loads(p.read_text(encoding="utf-8"))
    w = dict(well_scale=float(d["well_scale"]), well_frac=float(d["well_frac"]),
             inertia_start=float(d["inertia_start"]), inertia_end=float(d["inertia_end"]),
             polish_pops=int(d["polish_pops"]))
    return float(d["prior_s"]), w


def selected_pw(out_dir: Path) -> dict:
    p = out_dir / "validation" / "selected_pw.json"
    if not p.exists():
        raise SystemExit(f"missing {p}: run stage 'validate' with gnn_pso_wpt first")
    d = json.loads(p.read_text(encoding="utf-8"))
    return dict(well_scale=float(d["well_scale"]), well_frac=float(d["well_frac"]),
                inertia_start=float(d["inertia_start"]), inertia_end=float(d["inertia_end"]),
                polish_pops=int(d["polish_pops"]))


def selected_qw(out_dir: Path, args) -> dict:
    p = out_dir / "validation" / "selected_qw.json"
    forced = (args.qw_scale, args.qw_frac, args.qw_beta_start, args.qw_beta_end, args.qw_polish)
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        w = dict(well_scale=float(d["well_scale"]), well_frac=float(d["well_frac"]),
                 beta_start=float(d["beta_start"]), beta_end=float(d["beta_end"]), polish_pops=int(d["polish_pops"]))
    elif any(v is None for v in forced):
        raise SystemExit(f"missing {p}: run stage 'validate' with gnn_qwqpso first "
                         f"(or force all of --qw-scale --qw-frac --qw-beta-start --qw-beta-end --qw-polish)")
    else:
        w = {}
    for k, v in zip(("well_scale", "well_frac", "beta_start", "beta_end", "polish_pops"), forced):
        if v is not None:
            w[k] = int(v) if k == "polish_pops" else float(v)
    return w


def selected_qq(out_dir: Path, args) -> dict:
    p = out_dir / "validation" / "selected_qq.json"
    q = dict(QQ_OFF)
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        q.update(mbest_mode=d["mbest_mode"], beta_mode=d["beta_mode"], polish_pops=int(d["polish_pops"]))
    elif args.qq_mbest is None or args.qq_beta is None or args.qq_polish is None:
        raise SystemExit(f"missing {p}: run stage 'validate' with gnn_qqpso first "
                         f"(or force all of --qq-mbest --qq-beta --qq-polish)")
    if args.qq_mbest is not None:
        q["mbest_mode"] = args.qq_mbest
    if args.qq_beta is not None:
        q["beta_mode"] = args.qq_beta
    if args.qq_polish is not None:
        q["polish_pops"] = int(args.qq_polish)
    return q


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
    specs = []
    qq = selected_qq(out_dir, args) if any(m in QQ_METHODS + POLISH_METHODS for m in methods) else None
    if qq is not None:
        log.info(f"[test] gnn_qqpso config {qq}")
        if not quiet:
            print(f"[test    ] gnn_qqpso config {qq}")
    qw = selected_qw(out_dir, args) if any(m in QW_METHODS + ("gnn_pso_wp",) for m in methods) else None
    pw = selected_pw(out_dir) if "gnn_pso_wpt" in methods else None
    pr = selected_prior(out_dir) if "prior_qwqpso" in methods else None
    dpr = (selected_dprior(out_dir)
           if any(m in DPRIOR_METHODS + FPRIOR_METHODS + FPRIOR_BASE_METHODS + UPRIOR_METHODS
                   + UPRIOR_BASE_METHODS + UPRIOR_PLAIN_METHODS for m in methods)
           else None)
    ppr = selected_ppso(out_dir) if any(m in PRIOR_PSO_METHODS for m in methods) else None
    if ppr is not None:
        log.info(f"[test] prior_pso config s={ppr[0]} {ppr[1]}")
        if not quiet:
            print(f"[test    ] prior_pso config s={ppr[0]} {ppr[1]}")
    if dpr is not None:
        log.info(f"[test] dprior config s={dpr[0]} {dpr[1]}")
        if not quiet:
            print(f"[test    ] dprior config s={dpr[0]} {dpr[1]}")
    pb = selected_prior_base(out_dir)
    if pr is not None:
        log.info(f"[test] prior config s={pr[0]} {pr[1]}")
        if not quiet:
            print(f"[test    ] prior config s={pr[0]} {pr[1]}")
    if pw is not None:
        log.info(f"[test] gnn_pso_wpt config {pw}")
        if not quiet:
            print(f"[test    ] gnn_pso_wpt config {pw}")
    if qw is not None:
        log.info(f"[test] gnn_qwqpso config {qw}")
        if not quiet:
            print(f"[test    ] gnn_qwqpso config {qw}")
    for k in methods:
        if k in QW_METHODS:
            sp = make_spec(k, B, exp.budget_full, exp.delta_adjust, qw=qw)
            ref = make_spec("gnn_qwqpso", B, exp.budget_full, exp.delta_adjust, qw=qw)
            def _eff(x):    # effective (well, beta) of a spec: None beta -> run default
                return (x.well_scale, x.well_frac,
                        exp.beta_start if x.beta_start is None else x.beta_start,
                        exp.beta_end if x.beta_end is None else x.beta_end)
            if k != "gnn_qwqpso" and _eff(sp) == _eff(ref):
                msg = f"[test    ] skip {k}: selected gnn_qwqpso has that component off (identical run)"
                log.info(msg)
                if not quiet:
                    print(msg)
                continue
            specs.append(sp)
        elif k in QQ_METHODS:
            sp = make_spec(k, B, exp.budget_full, exp.delta_adjust, qq=qq)
            if k != "gnn_qqpso" and (sp.mbest_mode, sp.beta_mode, sp.polish_pops) == \
                    (qq["mbest_mode"], qq["beta_mode"], int(qq["polish_pops"])):
                msg = f"[test    ] skip {k}: selected gnn_qqpso has that component off (identical run)"
                log.info(msg)
                if not quiet:
                    print(msg)
                continue
            specs.append(sp)
        elif k in POLISH_METHODS:
            specs.append(make_spec(k, B, exp.budget_full, exp.delta_adjust, qq=qq))
        elif k == "gnn_pso_wp":
            specs.append(make_spec(k, B, exp.budget_full, exp.delta_adjust, qw=qw))
        elif k == "gnn_pso_wpt":
            specs.append(make_spec(k, B, exp.budget_full, exp.delta_adjust, pw=pw))
        elif k == "prior_qwqpso":
            specs.append(make_spec(k, B, exp.budget_full, exp.delta_adjust, qw=pr[1], prior_s=pr[0]))
        elif k == "prior_pso":
            specs.append(make_spec(k, B, exp.budget_full, exp.delta_adjust, pw=ppr[1], prior_s=ppr[0]))
        elif k == "prior_fullpso":                 # Algorithm 1 settings; s of the selected prior_pso
            specs.append(make_spec(k, B, exp.budget_full, exp.delta_adjust, prior_s=ppr[0]))
        elif k in DPRIOR_METHODS or k in FPRIOR_METHODS or k in UPRIOR_METHODS:
            specs.append(make_spec(k, B, exp.budget_full, exp.delta_adjust, qw=dpr[1], prior_s=dpr[0]))
        elif k in FPRIOR_BASE_METHODS or k in UPRIOR_BASE_METHODS or k in UPRIOR_PLAIN_METHODS:   # s of dprior cfg
            specs.append(make_spec(k, B, exp.budget_full, exp.delta_adjust, prior_s=dpr[0]))
        elif k in DPRIOR_BASE_METHODS:
            if k in pb:
                s_k = float(pb[k]["prior_s"])
            else:
                p_d = out_dir / "validation" / "selected_dprior.json"
                if not p_d.exists():
                    raise SystemExit(f"[test] {k}: no selected s (run validate with {k} or prior_dqpso first)")
                s_k = float(json.loads(p_d.read_text(encoding="utf-8"))["prior_s"])
            log.info(f"[test] {k} prior_s={s_k}")
            specs.append(make_spec(k, B, exp.budget_full, exp.delta_adjust, prior_s=s_k))
        elif k in PRIOR_BASE_METHODS:
            if k in pb:
                s_k = float(pb[k]["prior_s"])
            elif pr is not None:
                s_k = pr[0]
            else:
                p_qw = out_dir / "validation" / "selected_prior.json"
                if not p_qw.exists():
                    raise SystemExit(f"[test] {k}: no selected s (run validate with {k} or prior_qwqpso first)")
                s_k = float(json.loads(p_qw.read_text(encoding="utf-8"))["prior_s"])
            log.info(f"[test] {k} prior_s={s_k}")
            specs.append(make_spec(k, B, exp.budget_full, exp.delta_adjust, prior_s=s_k))
        else:
            specs.append(make_spec(k, B, exp.budget_full, exp.delta_adjust, anchor_w0=w0, anchor_frac=fr))
    run_trial_stage(sys_, exp, filter_k(read_cases(bdir, "test"), args), specs, bdir / "models",
                    out_dir / "test" / "trials.csv",
                    "test", [m for m in ("gnn_qpso", "gnn_qqpso", "gnn_qwqpso", "gnn_pso_p", "gnn_pso_wp", "gnn_pso_wpt",
                                         "full_qpso", "gnn_ga", "gnn_de", "full_ga", "full_de", "prior_qwqpso",
                                         "prior_dqpso", "prior_dpolish", "prior_pso", "prior_fullpso",
                                         "prior_fqpso", "prior_fpso", "prior_uqpso", "prior_uqpso_p",
                                         "prior_upso_p", "prior_udirect", "prior_upso", "prior_fpso_plain")
                                if m in methods],
                    quiet, log, overwrite)


def stage_tables(out_dir, bdir, sys_, exp, log, quiet, plots):
    from common.tables import load_csv, make_all
    test_p = out_dir / "test" / "trials.csv"
    val_p = out_dir / "validation" / "val_trials.csv"
    test = load_csv(test_p) if test_p.exists() else None
    val = load_csv(val_p) if val_p.exists() else None
    val_all = val
    if val is not None:
        val = _grid_free(val)                         # grids shown separately (anchor_grid.csv / qq_grid.csv)
    cases_p = bdir / "cases" / "cases_test.csv"
    cases = pd.read_csv(cases_p) if cases_p.exists() else None
    present = set(test["method"]) if test is not None else set()
    pairs = [("gnn_qpso", "rand_qpso"), ("gnn_qpso", "mlp_qpso"), ("gnn_qpso", "full_qpso"),
             ("gnn_aqpso", "gnn_qpso"), ("gnn_aqpso", "rand_qpso"),
             ("gnn_qqpso", "gnn_qpso"), ("gnn_qqpso", "gnn_qqpso_noF"), ("gnn_qqpso", "gnn_qqpso_noB"),
             ("gnn_qqpso", "gnn_qqpso_noP"), ("gnn_qqpso", "gnn_pso_p"),
             ("gnn_qwqpso", "gnn_qqpso"), ("gnn_qwqpso", "gnn_pso_p"), ("gnn_qwqpso", "gnn_qpso"),
             ("gnn_qwqpso", "gnn_qwqpso_noW"), ("gnn_qwqpso", "gnn_qwqpso_noB"),
             ("gnn_qwqpso", "gnn_pso_wp"), ("gnn_qwqpso", "gnn_pso_wpt"), ("gnn_pso_wp", "gnn_pso_p"),
             ("gnn_pso_wpt", "gnn_pso_p"),
             ("gnn_qwqpso", "gnn_ga"), ("gnn_qwqpso", "gnn_de"), ("gnn_qwqpso", "full_ga"),
             ("gnn_qwqpso", "full_de"), ("gnn_ga", "gnn_pso"), ("gnn_de", "gnn_pso"),
             ("prior_qwqpso", "prior_direct"), ("prior_qwqpso", "prior_polish"), ("prior_polish", "prior_direct"),
             ("prior_dqpso", "prior_dpolish"), ("prior_dqpso", "prior_dqpso_noW"),
             ("prior_dqpso", "prior_dqpso_noBox"), ("prior_dqpso", "prior_qwqpso"), ("prior_dqpso", "prior_polish"),
             ("prior_dpolish", "prior_ddirect"), ("prior_dpolish", "prior_polish"), ("prior_ddirect", "prior_direct"),
             ("prior_qwqpso", "prior_pso"), ("prior_qwqpso", "prior_fullpso"), ("prior_pso", "prior_direct"),
             ("prior_fqpso", "prior_fdirect"), ("prior_fpso", "prior_fdirect"), ("prior_fqpso", "prior_fpso"),
             ("prior_fdirect", "prior_ddirect"), ("prior_fqpso", "prior_dqpso"), ("prior_fqpso", "full_pso"),
             ("prior_fdirect", "full_pso"), ("prior_uqpso_p", "prior_uqpso"), ("prior_uqpso_p", "prior_upso_p"),
             ("prior_uqpso_p", "prior_udirect"), ("prior_upso_p", "prior_udirect"),
             ("prior_uqpso_p", "prior_upso"), ("prior_upso", "prior_udirect"),
             ("prior_fpso", "prior_fpso_plain"), ("prior_fqpso", "prior_fpso_plain"),
             ("prior_fpso_plain", "prior_fdirect"), ("prior_fpso_plain", "full_pso")]
    pairs = [(a, b) for a, b in pairs if a in present and b in present]
    make_all(out_dir, sys_, exp, test, val, labels_dir=None, models_dir=bdir / "models", cases=cases,
             ref="gnn_qpso", pairs=pairs, plots=plots, optimizer_name="QPSO", quiet=quiet, log=log)
    if plots and val_all is not None:
        from common.tables import fig_anchor_grid
        fig_anchor_grid(val_all, out_dir / "figures")
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
    methods = list(args.methods) if args.methods else list(QUANTUM_METHODS) + list(QQ_METHODS)
    check_against_baseline(exp, bdir)
    if args.dry_run:
        from baseline_code.run_baseline import dry_run
        dry_run(sys_, exp, stages, methods)
        if "gnn_aqpso" in methods:
            print(f"  + validate anchor grid {len(ANCHOR_W0_GRID)}x{len(ANCHOR_FRAC_GRID)} (gnn_aqpso) on validation")
        if any(m in QW_METHODS for m in methods):
            pg = args.qw_polish_grid or QW_POLISH_GRID
            print(f"  + validate gnn_qwqpso grid {len(QW_WELL_GRID)}x{len(QW_BETA_GRID)}x{len(pg)} on validation "
                  f"(K_val={exp.k_val or exp.k_train})")
        if any(m in QQ_METHODS for m in methods):
            pg = args.qq_polish_grid or QQ_POLISH_GRID
            print(f"  + validate gnn_qqpso grid {len(QQ_MBEST_GRID)}x{len(QQ_BETA_GRID)}x{len(pg)} on validation "
                  f"(K_val={exp.k_val or exp.k_train})")
        return 0
    log = init_run(out_dir, sys_, exp, args, "quantum", overwrite=args.overwrite)
    log.info(f"baseline_dir={bdir}")
    print(f"[quantum] out_dir={out_dir} baseline_dir={bdir} stages={stages} workers={exp.workers}")
    for st in stages:
        t0 = time.perf_counter()
        log.info(f"--- stage {st} start")
        if st == "validate":
            stage_validate(out_dir, bdir, sys_, exp, methods, args, log, args.quiet, args.overwrite)
        elif st == "test":
            stage_test(out_dir, bdir, sys_, exp, methods, args, log, args.quiet, args.overwrite)
        elif st == "tables":
            stage_tables(out_dir, bdir, sys_, exp, log, args.quiet, not args.no_plots)
        log.info(f"--- stage {st} done in {time.perf_counter() - t0:.1f}s")
    log.info("=== quantum run end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
