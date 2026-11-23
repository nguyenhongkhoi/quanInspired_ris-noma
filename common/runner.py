"""Parallel execution, incremental CSV with resume, and the allocation-trial worker.

Unit of work = one (layout, K) group (claude.md 6.3). Workers are top-level
functions (Windows spawn). Each worker process uses ONE BLAS/torch thread.
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np

from .assessment import BOUND_FLAGS, POINT_FLAGS, assess_allocation
from .channel import bank_seed, build_bank, generate_layout
from .config import stable_seed
from .evaluator import Problem, alloc_to_z, to_original
from .methods import MethodSpec, get_optimizer

THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "True" if v else "False"
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with open(path, "r", newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh))
    else:
        header = list(rows[0].keys())
        for r in rows[1:]:
            for k in r:
                if k not in header:
                    header.append(k)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in header})


def done_groups(path: Path, col: str = "group_key") -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    out = set()
    with open(path, "r", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.add(r.get(col, ""))
    return out


# ---------------------------------------------------------------------------
# Parallel engine
# ---------------------------------------------------------------------------
def _worker_init() -> None:
    for k in THREAD_ENV:
        os.environ[k] = "1"
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _safe_call(fn, task):
    try:
        return ("ok", fn(task))
    except Exception:
        return ("err", traceback.format_exc())


def run_parallel(tasks: list[dict], fn: Callable, workers: int,
                 on_result: Callable[[dict, object], None], log=None) -> None:
    """Run fn(task) for every task; call on_result(task, result) in the main process."""
    if not tasks:
        return
    if workers <= 1:
        _worker_init()
        for t in tasks:
            status, res = _safe_call(fn, t)
            if status == "err":
                if log:
                    log.error(f"group {t.get('group_key')} failed:\n{res}")
                raise RuntimeError(f"group {t.get('group_key')} failed:\n{res}")
            on_result(t, res)
        return
    saved = {k: os.environ.get(k) for k in THREAD_ENV}
    for k in THREAD_ENV:
        os.environ[k] = "1"   # inherited by spawned children before numpy import
    try:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_worker_init) as ex:
            futs = {ex.submit(_safe_call, fn, t): t for t in tasks}
            for fut in as_completed(futs):
                t = futs[fut]
                status, res = fut.result()
                if status == "err":
                    if log:
                        log.error(f"group {t.get('group_key')} failed:\n{res}")
                    for f in futs:
                        f.cancel()
                    raise RuntimeError(f"group {t.get('group_key')} failed:\n{res}")
                on_result(t, res)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Model cache (per worker process)
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict = {}


def get_model(models_dir: str, arch: str, seed: int, sys_):
    key = (models_dir, arch, seed)
    if key not in _MODEL_CACHE:
        from .features import Normalizer
        from .models import load_checkpoint
        path = Path(models_dir) / f"{arch}_seed{seed}.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing checkpoint {path} (run stage 'train' first)")
        model, norm, meta = load_checkpoint(path, sys_)
        _MODEL_CACHE[key] = (model, Normalizer.from_dict(norm), meta)
    return _MODEL_CACHE[key]


def predict_alloc(model, norm, prob: Problem):
    """Features + batch-one inference. Returns (mb, s, m_sic, infer_ms)."""
    from .features import case_features, collate
    from .models import predict_numpy
    xn, xg = case_features(prob)
    node, glob, edge, mask, _ = collate([xn], [xg], norm)
    t = time.perf_counter()
    mb, s, m = predict_numpy(model, node, glob, edge, mask)
    infer_ms = (time.perf_counter() - t) * 1e3
    mb0 = float(mb[0])
    m0 = np.asarray(m[0, :prob.K], dtype=float)
    assert m0.shape == (prob.K,) and np.all(m0 >= 0) and m0.sum() > 0, "bad public prediction"
    m0 = m0 / m0.sum() * (1.0 - mb0)          # exact simplex in float64 (model runs in float32)
    return mb0, float(s[0]), m0, infer_ms


# ---------------------------------------------------------------------------
# Allocation-trial worker
# ---------------------------------------------------------------------------
def opt_seed(master: int, case_id: str, pso_seed: int, branch: str, budget: int) -> int:
    """Same tuple for PSO and QPSO and for rand/mlp/gnn at one budget (pairing)."""
    return stable_seed(master, "opt", case_id, int(pso_seed), branch, int(budget))


def trial_group(task: dict) -> list[dict]:
    from .models import adjust_prediction
    sys_, exp = task["sys"], task["exp"]
    lid, K, split = int(task["layout_id"]), int(task["K"]), task["split"]
    cases: list[dict] = task["cases"]
    specs: list[MethodSpec] = task["specs"]
    R = int(task["replicates"])
    models_dir = task.get("models_dir")

    t0 = time.perf_counter()
    layout = generate_layout(sys_, exp.seed, lid, split, exp.K_max)
    bank_s = build_bank(sys_, layout, K, exp.search_samples, bank_seed(exp.seed, "search", lid, K))
    share_ms = (time.perf_counter() - t0) * 1e3 / max(1, len(cases))
    bank_a = build_bank(sys_, layout, K, exp.assess_samples, bank_seed(exp.seed, "assess", lid, K))

    need = sorted({s.init for s in specs if s.init})
    models = {}
    for r in range(R):
        for arch in need:
            models[(arch, r)] = get_model(models_dir, arch, exp.train_seeds[r], sys_)
    # warm-up (untimed)
    if models and cases:
        pw = Problem(sys_, bank_s, cases[0]["P_tot_dbm"], cases[0]["xi_req"], "search")
        for (model, norm, _) in models.values():
            predict_alloc(model, norm, pw)
        pw.evaluate_z(np.full((exp.swarm, K + 1), 0.5))

    rows = []
    for case in cases:
        P_tot, xi = float(case["P_tot_dbm"]), float(case["xi_req"])
        prob_a = None
        for r in range(R):
            for spec in specs:
                t1 = time.perf_counter()
                prob = Problem(sys_, bank_s, P_tot, xi, "search")
                infer_ms = 0.0
                pred = None
                if spec.init:
                    model, norm, _ = models[(spec.init, r)]
                    mb_p, s_p, m_p, infer_ms = predict_alloc(model, norm, prob)
                    mb_a, s_a, m_a = adjust_prediction(sys_, mb_p, s_p, m_p, spec.delta)
                    pred = (float(mb_a), float(s_a), np.asarray(m_a, float))
                init_feas = None
                hist = None
                if spec.optimizer is None:
                    assert pred is not None, f"{spec.key} needs a prediction"
                    mb, s, m_sic = pred
                    n_evals = 0
                else:
                    opt = get_optimizer(spec.optimizer)
                    seed = opt_seed(exp.seed, case["case_id"], exp.pso_seeds[r], spec.seed_branch, spec.budget)
                    init_z = alloc_to_z(sys_, pred[0], pred[1], pred[2][None, :])[0] if pred else None
                    kw = {}
                    if spec.optimizer == "qpso":
                        kw.update(beta_start=exp.beta_start, beta_end=exp.beta_end, boundary=exp.boundary,
                                  mbest_mode=spec.mbest_mode, beta_mode=spec.beta_mode,
                                  polish_pops=spec.polish_pops)
                    if spec.anchored:
                        kw.update(anchor_z=init_z, anchor_w0=spec.anchor_w0,
                                  anchor_frac=spec.anchor_frac, anchor_gate=exp.anchor_gate)
                    res = opt(prob, spec.budget, exp.swarm, seed, init_z=init_z, **kw)
                    mb, s, m_sic = res.best.mb, res.best.s, res.best.m_sic
                    n_evals = res.n_evals
                    init_feas = res.init_feasible
                    hist = [round(v, 6) if f else None for v, f in zip(res.history_best_rb, res.history_best_feas)]
                time_ms = share_ms + (time.perf_counter() - t1) * 1e3

                # untimed: search feasibility of the returned allocation
                ev_s = prob.evaluate_alloc([mb], [s], np.asarray(m_sic)[None, :])
                if prob_a is None:
                    prob_a = Problem(sys_, bank_a, P_tot, xi, "assess",
                                     sic_order=prob.sic_order, eligible=prob.eligible)
                a = assess_allocation(sys_, prob_a, mb, s, m_sic)
                row = dict(
                    group_key=task["group_key"], case_id=case["case_id"], layout_id=lid, split=split, K=K,
                    case_idx=case["case_idx"], P_tot_dbm=P_tot, xi_req=xi, replicate=r,
                    method=spec.key, tag=spec.tag, budget=spec.budget, n_evals=n_evals,
                    mb=float(mb), s=float(s),
                    m_orig=json.dumps([round(float(x), 9) for x in to_original(np.asarray(m_sic), prob.sic_order)]),
                    search_feasible=bool(ev_s.feasible[0]), init_feasible=init_feas,
                    R_b=a["R_b"], R_pub_min=a["R_pub_min"], R_L_min=a["R_L_min"], p_b=a["p_b"], F_b=a["F_b"],
                    p_bU=a["p_bU"], xi_emp=a["xi_emp"], xi_gamma=a["xi_gamma"], xi_L=a["xi_L"],
                    passed=a["passed"], category=a["category"],
                    **{k: a[k] for k in POINT_FLAGS}, **{k: a[k] for k in BOUND_FLAGS},
                    mb_gt_ceiling=a["mb_gt_ceiling"], rb_ceiling=a["rb_ceiling"],
                    cv_w=prob.cv_w, eligible=prob.eligible,
                    time_ms=time_ms, infer_ms=infer_ms,
                    anchor_w0=spec.anchor_w0, anchor_frac=spec.anchor_frac,
                    qq_mbest=spec.mbest_mode, qq_beta=spec.beta_mode, qq_polish=spec.polish_pops,
                    hist_rb=json.dumps(hist) if hist is not None else "",
                )
                rows.append(row)
    return rows


def run_trial_stage(sys_, exp, cases: list[dict], specs: list[MethodSpec], models_dir, out_csv: Path,
                    stage_tag: str, watch: list[str], quiet: bool, log, overwrite: bool = False) -> None:
    """Run all (layout, K) groups of `cases` x `specs`, append to out_csv, resume-safe.

    watch: method keys whose running pass rate / gated rate is shown on the progress line.
    """
    from .progress import ProgressLine
    if overwrite and out_csv.exists():
        out_csv.unlink()
    skip = done_groups(out_csv)
    tasks = make_trial_tasks(sys_, exp, cases, specs, models_dir, skip)
    n_all = len({(c["layout_id"], c["K"]) for c in cases})
    spec_txt = ",".join(s.key + "@" + str(s.budget) + ((":" + s.tag) if s.tag else "") for s in specs)
    log.info(f"[{stage_tag}] groups total={n_all} done={len(skip)} todo={len(tasks)} "
             f"specs={spec_txt} workers={exp.workers}")
    if not tasks:
        if not quiet:
            print(f"[{stage_tag:<8}] nothing to do ({len(skip)}/{n_all} groups already in {out_csv.name})")
        return
    pl = ProgressLine(stage_tag, len(tasks), unit="grp", quiet=quiet)
    stats = {k: [0, 0, 0.0] for k in watch}   # trials, passed, gated sum
    done = [0]
    t0 = time.perf_counter()

    def on_result(task, rows):
        append_rows(out_csv, rows)
        for r in rows:
            if r["method"] in stats:
                st = stats[r["method"]]
                st[0] += 1
                st[1] += int(r["passed"])
                st[2] += r["R_b"] if r["passed"] else 0.0
        done[0] += 1
        extra = " ".join(f"{k}:{100 * v[1] / v[0]:.1f}%/{v[2] / v[0]:.3f}" for k, v in stats.items() if v[0])
        if done[0] % max(1, getattr(exp, "log_every", 1)) == 0 or done[0] == len(tasks):
            pl.update(done[0], extra)
        log.info(f"[{stage_tag}] group {task['group_key']} done ({len(rows)} trials) {extra}")

    run_parallel(tasks, trial_group, exp.workers, on_result, log)
    summ = " ".join(f"{k}: pass {100 * v[1] / v[0]:.2f}% Rg {v[2] / v[0]:.4f}" for k, v in stats.items() if v[0])
    msg = f"[{stage_tag}] finished {len(tasks)} groups in {time.perf_counter() - t0:.1f}s  {summ}"
    pl.close(msg)
    log.info(msg)


def make_trial_tasks(sys_, exp, cases: list[dict], specs: list[MethodSpec], models_dir,
                     skip: set[str] | None = None) -> list[dict]:
    groups: dict[tuple, list] = {}
    for c in cases:
        groups.setdefault((int(c["layout_id"]), int(c["K"]), c["split"]), []).append(c)
    tasks = []
    for (lid, K, split), cs in sorted(groups.items()):
        key = f"{lid}_K{K}"
        if skip and key in skip:
            continue
        cs = sorted(cs, key=lambda c: int(c["case_idx"]))
        tasks.append(dict(group_key=key, sys=sys_, exp=exp, split=split, layout_id=lid, K=K,
                          cases=cs, specs=list(specs), replicates=exp.replicates,
                          models_dir=str(models_dir) if models_dir else None))
    return tasks
