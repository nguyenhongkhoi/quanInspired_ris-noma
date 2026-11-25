"""Merge trials.csv of the RIS-phase run (one dir per machine/shard) and print/save per-K tables.

  python summarize_phase.py --dirs phase_B_s0,phase_B_s1 --out-dir phase_B_cmp
Writes <out-dir>/trials_merged.csv, summary_perK.csv, paired_perK.csv (bootstrap 95% CI over
layouts, 2000 resamples, seed 7317, as table4 of the paper pipeline).
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.cli import resolve_run_dir  # noqa: E402

PAIRS = (("phase_fqpso", "phase_fdirect"), ("phase_fqpso", "phase_fullpso"), ("phase_fullpso", "phase_fdirect"))


def tb(x) -> bool:
    return str(x) in ("True", "1", "true", "1.0")


def boot_ci(by_layout: dict, n: int = 2000, seed: int = 7317):
    lays = sorted(by_layout)
    vals = [np.asarray(by_layout[k]) for k in lays]
    allv = np.concatenate(vals)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n):
        idx = rng.integers(0, len(lays), len(lays))
        means.append(np.concatenate([vals[i] for i in idx]).mean())
    return float(allv.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    out = resolve_run_dir(a.out_dir, "phase_cmp")
    out.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    for d in a.dirs.split(","):
        f = resolve_run_dir(d, d) / "test" / "trials.csv"
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                key = (r["case_id"], r["replicate"], r["method"])
                assert key not in seen, f"duplicate trial {key} (overlapping shards?)"
                seen.add(key)
                rows.append(r)
    with open(out / "trials_merged.csv", "w", newline="", encoding="utf-8") as fh:
        cols = list(rows[0].keys())
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    st = defaultdict(lambda: defaultdict(list))
    pair = defaultdict(dict)
    for r in rows:
        K, m = int(r["K"]), r["method"]
        g = float(r["R_b"]) if tb(r["passed"]) else 0.0
        s = st[(m, K)]
        s["pass"].append(tb(r["passed"]))
        s["gated"].append(g)
        s["sf"].append(tb(r["search_feasible"]))
        s["evals"].append(float(r["n_evals"]))
        s["rms"].append(float(r["phase_rms"]))
        s["gb"].append(float(r["gain_bob_rel"]))
        s["gw"].append(float(r["gain_willie_rel"]))
        s["mb"].append(float(r["mb"]))
        pair[(K, r["case_id"], r["replicate"])][m] = (g, int(r["layout_id"]))
    methods = sorted({m for m, _ in st})
    Ks = sorted({k for _, k in st})
    summ = []
    print("K  method          n     pass%   R_gated  sfeas%  evals  m_b     |Delta|rms  gBob  gWillie")
    for K in Ks:
        for m in methods:
            s = st[(m, K)]
            if not s["pass"]:
                continue
            row = dict(K=K, method=m, n=len(s["pass"]), pass_pct=100 * np.mean(s["pass"]),
                       R_gated=np.mean(s["gated"]), search_feas_pct=100 * np.mean(s["sf"]),
                       evals=np.mean(s["evals"]), mb=np.mean(s["mb"]), phase_rms=np.mean(s["rms"]),
                       gain_bob_rel=np.mean(s["gb"]), gain_willie_rel=np.mean(s["gw"]))
            summ.append(row)
            print(f"{K:<2} {m:15} {row['n']:5d} {row['pass_pct']:7.2f} {row['R_gated']:8.4f} "
                  f"{row['search_feas_pct']:6.1f} {row['evals']:6.0f} {row['mb']:.4f}  {row['phase_rms']:8.3f}  "
                  f"{row['gain_bob_rel']:.3f}  {row['gain_willie_rel']:.3f}")
    with open(out / "summary_perK.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)

    prs = []
    print("\npaired R_gated diff (mean [95% CI over layouts]), %trials >0 / <0")
    for K in Ks:
        for x, y in PAIRS:
            by_l = defaultdict(list)
            for (k, _, _), v in pair.items():
                if k == K and x in v and y in v:
                    by_l[v[x][1]].append(v[x][0] - v[y][0])
            if not by_l:
                continue
            mu, lo, hi = boot_ci(by_l)
            d = np.concatenate([np.asarray(v) for v in by_l.values()])
            prs.append(dict(K=K, pair=f"{x}-{y}", n=len(d), mean=mu, ci_lo=lo, ci_hi=hi,
                            pct_pos=100 * np.mean(d > 1e-4), pct_neg=100 * np.mean(d < -1e-4)))
            print(f"K={K:<2} {x}-{y:15} {mu:+.4f} [{lo:+.4f}, {hi:+.4f}]  {prs[-1]['pct_pos']:.1f}% / "
                  f"{prs[-1]['pct_neg']:.1f}%  n={len(d)}")
    if prs:
        with open(out / "paired_perK.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(prs[0].keys()))
            w.writeheader()
            w.writerows(prs)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
