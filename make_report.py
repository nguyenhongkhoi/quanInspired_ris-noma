"""Merge baseline (PSO) and quantum-inspired (QPSO) runs into comparison tables/figures.

    python make_report.py --baseline-dir bl_quick_01 --quantum-dir q_quick_01 --out-dir cmp_quick_01

Paired differences are computed on identical (case, replicate) pairs: both runs
use the same cases, search banks, assessment banks and optimizer seeds.
QPSO is quantum-INSPIRED classical code; no quantum advantage is claimed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from common.cli import CRITICAL_KEYS, load_exp_config, resolve_run_dir  # noqa: E402
from common.config import SysConfig  # noqa: E402
from common.progress import setup_run_logger  # noqa: E402
from common.tables import load_csv, make_all  # noqa: E402

DEFAULT_PAIRS = "gnn_qpso:gnn_pso,gnn_aqpso:gnn_pso,rand_qpso:rand_pso,full_qpso:full_pso,gnn_aqpso:gnn_qpso"


def main() -> int:
    ap = argparse.ArgumentParser(description="PSO vs QPSO comparison report")
    ap.add_argument("--baseline-dir", default=None)
    ap.add_argument("--quantum-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--preset", default="quick", help="only used for default directory names")
    ap.add_argument("--ref", default="gnn_pso")
    ap.add_argument("--pairs", default=DEFAULT_PAIRS)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    bdir = resolve_run_dir(args.baseline_dir, f"baseline_{args.preset}")
    qdir = resolve_run_dir(args.quantum_dir, f"quantum_{args.preset}")
    out_dir = resolve_run_dir(args.out_dir, f"compare_{args.preset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_run_logger(out_dir)
    log.info(f"=== compare: baseline={bdir} quantum={qdir}")

    bexp, qexp = load_exp_config(bdir), load_exp_config(qdir)
    norm = lambda v: list(v) if isinstance(v, (list, tuple)) else v  # noqa: E731
    diff = {k: (getattr(bexp, k), getattr(qexp, k)) for k in CRITICAL_KEYS
            if norm(getattr(bexp, k)) != norm(getattr(qexp, k))}
    if diff:
        raise SystemExit(f"baseline and quantum runs used different settings: {diff}")

    tb, tq = load_csv(bdir / "test" / "trials.csv"), load_csv(qdir / "test" / "trials.csv")
    kb = set(zip(tb["case_id"], tb["replicate"]))
    kq = set(zip(tq["case_id"], tq["replicate"]))
    if kb != kq:
        log.warning(f"case/replicate sets differ: baseline={len(kb)} quantum={len(kq)} common={len(kb & kq)}")
        print(f"[compare] WARNING: case sets differ (common {len(kb & kq)}); paired stats use common pairs only")
    test = pd.concat([tb, tq], ignore_index=True)

    vals = []
    for d in (bdir, qdir):
        p = d / "validation" / "val_trials.csv"
        if p.exists():
            v = load_csv(p)
            vals.append(v[v["method"] != "gnn_aqpso"])
    val = pd.concat(vals, ignore_index=True) if vals else None

    pairs = [tuple(p.split(":")) for p in args.pairs.split(",") if p.strip()]
    cases_p = bdir / "cases" / "cases_test.csv"
    make_all(out_dir, SysConfig(), bexp, test, val, labels_dir=bdir / "labels", models_dir=bdir / "models",
             cases=pd.read_csv(cases_p) if cases_p.exists() else None, ref=args.ref, pairs=pairs,
             plots=not args.no_plots, optimizer_name="PSO / QPSO", quiet=args.quiet, log=log)
    (out_dir / "compare_sources.json").write_text(json.dumps(
        dict(baseline_dir=str(bdir), quantum_dir=str(qdir), ref=args.ref, pairs=pairs), indent=2), encoding="utf-8")
    print(f"[compare] tables -> {out_dir / 'tables'}  figures -> {out_dir / 'figures'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
