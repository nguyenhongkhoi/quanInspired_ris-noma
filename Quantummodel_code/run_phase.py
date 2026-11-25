"""RIS-phase extension (harder problem, common/phase.py): fair prior + QW-QPSO vs Full PSO 1800.

Uses the test cases of a baseline run (default prior_blB, Set B) and the swarm configuration
selected on VALIDATION by the no-polish run (nopol_B_01/validation/selected_dprior.json).
No validation is run here and the test set is never used for any choice.

Methods (common/phase.py):
  phase_fdirect : fair prior, Delta = 0 (the paper's phases; reference, <= 14 evaluations)
  phase_fqpso   : fair prior (Delta = 0) in particle 0 + QW-QPSO over allocation AND phases,
                  B = 150 in total (prior evaluations + 120 swarm), no polish
  phase_fullpso : Full PSO (Algorithm 1), 1800 evaluations, random init, allocation AND phases

Two machines: --shard i/n keeps the (layout, K) groups whose sorted index % n == i; give each
machine its own --out-dir, merge the trials.csv files afterwards (summarize_phase.py).

  python Quantummodel_code/run_phase.py --baseline-dir prior_blB --config-from nopol_B_01 \
      --out-dir phase_B_s0 --shard 0/2 --workers 19
Resume: groups already in <out-dir>/test/trials.csv are skipped.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.cli import default_workers, load_exp_config, resolve_run_dir  # noqa: E402
from common.config import SysConfig  # noqa: E402
from common.phase import PHASE_METHODS, phase_group  # noqa: E402
from common.runner import append_rows, done_groups, run_parallel  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline-dir", default="prior_blB")
    p.add_argument("--config-from", default="nopol_B_01",
                   help="run whose validation/selected_dprior.json gives prior_s, W and beta")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--methods", default=",".join(PHASE_METHODS))
    p.add_argument("--k-test", default="2,3,4,5,6,8,10")
    p.add_argument("--replicates", type=int, default=1)
    p.add_argument("--budget-short", type=int, default=150)
    p.add_argument("--shard", default="0/1", help="i/n: keep groups with index %% n == i")
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--max-groups", type=int, default=0, help="smoke: stop after this many groups (0 = all)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bdir = resolve_run_dir(args.baseline_dir, "prior_blB")
    cdir = resolve_run_dir(args.config_from, "nopol_B_01")
    out = resolve_run_dir(args.out_dir, "phase_B")
    (out / "test").mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("phase")
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(out / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)

    methods = [m for m in args.methods.split(",") if m]
    for m in methods:
        assert m in PHASE_METHODS, f"unknown method {m}; known {PHASE_METHODS}"
    ks = [int(k) for k in args.k_test.split(",") if k]
    i_sh, n_sh = (int(v) for v in args.shard.split("/"))
    assert 0 <= i_sh < n_sh, args.shard

    sys_ = SysConfig()
    exp = load_exp_config(bdir)
    exp = dataclasses.replace(exp, workers=int(args.workers), replicates=int(args.replicates))
    assert max(ks) <= exp.K_max, (ks, exp.K_max)
    assert args.budget_short % exp.swarm == 0 and args.budget_short > exp.swarm, args.budget_short
    cfg_p = cdir / "validation" / "selected_dprior.json"
    if not cfg_p.exists():
        raise SystemExit(f"missing {cfg_p}")
    cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    # polish is always off here (pure swarm); only s, W and beta are taken from the validation choice
    cfg = {k: cfg[k] for k in ("prior_s", "well_scale", "well_frac", "beta_start", "beta_end")}
    (out / "config_phase.json").write_text(json.dumps(dict(
        args=vars(args), swarm_cfg=cfg, exp=exp.to_dict(), baseline_dir=str(bdir), config_from=str(cfg_p)),
        indent=2, default=str), encoding="utf-8")

    with open(bdir / "cases" / "cases_test.csv", newline="", encoding="utf-8") as f:
        cases = [r for r in csv.DictReader(f) if int(r["K"]) in ks]
    for c in cases:
        c["layout_id"], c["K"], c["case_idx"] = int(c["layout_id"]), int(c["K"]), int(c["case_idx"])
    groups: dict[tuple, list] = {}
    for c in cases:
        groups.setdefault((c["layout_id"], c["K"], c["split"]), []).append(c)
    keys = sorted(groups)
    mine = [k for i, k in enumerate(keys) if i % n_sh == i_sh]
    out_csv = out / "test" / "trials.csv"
    skip = done_groups(out_csv)
    tasks = []
    for (lid, K, split) in mine:
        gk = f"{lid}_K{K}"
        if gk in skip:
            continue
        tasks.append(dict(group_key=gk, sys=sys_, exp=exp, split=split, layout_id=lid, K=K,
                          cases=sorted(groups[(lid, K, split)], key=lambda c: c["case_idx"]),
                          methods=methods, replicates=exp.replicates, budget_short=args.budget_short, cfg=cfg))
    if args.max_groups:
        tasks = tasks[:args.max_groups]
    msg = (f"[phase] groups all={len(keys)} shard {i_sh}/{n_sh}={len(mine)} done={len(skip)} todo={len(tasks)} "
           f"methods={methods} K={ks} replicates={exp.replicates} workers={exp.workers} cfg={cfg}")
    log.info(msg)
    print(msg, flush=True)

    stats = {m: [0, 0, 0.0] for m in methods}
    done = [0]
    t0 = time.perf_counter()

    def on_result(task, rows):
        append_rows(out_csv, rows)
        for r in rows:
            st = stats[r["method"]]
            st[0] += 1
            st[1] += int(r["passed"])
            st[2] += r["R_b"] if r["passed"] else 0.0
        done[0] += 1
        extra = " ".join(f"{k}:{100 * v[1] / v[0]:.1f}%/{v[2] / v[0]:.4f}" for k, v in stats.items() if v[0])
        line = f"[phase] {done[0]}/{len(tasks)} grp {task['group_key']} {time.perf_counter() - t0:.0f}s {extra}"
        log.info(line)
        if not args.quiet:
            print(line, flush=True)

    run_parallel(tasks, phase_group, exp.workers, on_result, log)
    summ = " ".join(f"{k}: pass {100 * v[1] / v[0]:.2f}% Rg {v[2] / v[0]:.4f}" for k, v in stats.items() if v[0])
    msg = f"[phase] finished {len(tasks)} groups in {time.perf_counter() - t0:.1f}s  {summ}"
    log.info(msg)
    print(msg, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
