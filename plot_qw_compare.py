"""Comparison figure: GNN + QW-QPSO vs GNN + short PSO (paper) vs GNN + short PSO + P.

Reads only existing CSVs (no experiment is run):
    <baseline-dir>/test/trials.csv   (full_pso, gnn_pso)
    <quantum-dir>/test/trials.csv    (gnn_pso_p, gnn_qwqpso)
Writes into <out-dir>:
    figures/fig_qw_compare.png (300 dpi) + .pdf
    tables/qw_compare_perK.csv   pass / gated / % of Full PSO / public-rate failures per K
    tables/qw_paired_perK.csv    paired gated diff per K, 95% bootstrap CI over layouts

    python plot_qw_compare.py --baseline-dir bl_paper_01 --quantum-dir qw_paper_01 --out-dir cmp_qw_paper_01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from common.tables import load_csv

REF = "full_pso"
METHODS = ["gnn_pso", "gnn_pso_p", "gnn_qwqpso"]
LABEL = {"full_pso": "Full PSO (B=1800)", "gnn_pso": "GNN + short PSO [paper]",
         "gnn_pso_p": "GNN + short PSO + P", "gnn_qwqpso": "GNN + QW-QPSO (proposed)"}
COLOR = {"full_pso": "#555555", "gnn_pso": "#d62728", "gnn_pso_p": "#ff7f0e", "gnn_qwqpso": "#1f4e9c"}
MARK = {"full_pso": "*", "gnn_pso": "^", "gnn_pso_p": "o", "gnn_qwqpso": "D"}
PAIRS = [("gnn_qwqpso", "gnn_pso"), ("gnn_qwqpso", "gnn_pso_p")]
BOOT_N, BOOT_SEED = 2000, 7317


def run_dir(arg: str) -> Path:
    p = Path(arg)
    if len(p.parts) == 1:
        return ROOT / "result" / arg
    return p if p.is_absolute() else ROOT / p


def boot_ci(x: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(x), size=(BOOT_N, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", default="bl_paper_01")
    ap.add_argument("--quantum-dir", default="qw_paper_01")
    ap.add_argument("--out-dir", default="cmp_qw_paper_01")
    ap.add_argument("--k-train", default="2,3,4,5", help="K values the networks were trained on")
    args = ap.parse_args()
    k_train = [int(v) for v in args.k_train.split(",") if v.strip()]
    trained_lab = f"{min(k_train)}-{max(k_train)}"

    tb = load_csv(run_dir(args.baseline_dir) / "test" / "trials.csv")
    tq = load_csv(run_dir(args.quantum_dir) / "test" / "trials.csv")
    df = pd.concat([tb[tb["method"].isin([REF, "gnn_pso"])],
                    tq[tq["method"].isin(["gnn_pso_p", "gnn_qwqpso"])]], ignore_index=True)
    for m in [REF] + METHODS:
        assert (df["method"] == m).any(), f"method {m} missing"
    Ks = sorted(int(k) for k in df["K"].unique())
    out = run_dir(args.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    # ---- per-K summary ------------------------------------------------------
    unseen = [k for k in Ks if k not in k_train]
    groups = [(str(k), [k]) for k in Ks] + [(trained_lab, [k for k in Ks if k in k_train])]
    rows = []
    for gname, kk in groups:
        d = df[df["K"].isin(kk)]
        ref_g = d[d["method"] == REF]["gated"].mean()
        for m in [REF] + METHODS:
            x = d[d["method"] == m]
            rate_fail = int((~x["pt_rate"]).sum()) if "pt_rate" in x.columns else -1
            rows.append(dict(K=gname, method=m, trials=len(x), pass_pct=100.0 * x["passed"].mean(),
                             R_gated=x["gated"].mean(), pct_of_full=100.0 * x["gated"].mean() / ref_g,
                             fail_pt_rate=rate_fail))
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "tables" / "qw_compare_perK.csv", index=False)

    # ---- paired per K -------------------------------------------------------
    prow = []
    for gname, kk in groups:
        d = df[df["K"].isin(kk)]
        for a, b in PAIRS:
            A = d[d["method"] == a][["case_id", "replicate", "layout_id", "gated", "passed"]]
            B = d[d["method"] == b][["case_id", "replicate", "gated", "passed"]]
            mm = A.merge(B, on=["case_id", "replicate"], suffixes=("_a", "_b"))
            assert len(mm) == len(A) == len(B), (gname, a, b, len(mm), len(A), len(B))
            mm["dg"] = mm["gated_a"] - mm["gated_b"]
            mm["dp"] = 100.0 * (mm["passed_a"].astype(float) - mm["passed_b"].astype(float))
            per = mm.groupby("layout_id")[["dg", "dp"]].mean()
            lo, hi = boot_ci(per["dg"].values)
            prow.append(dict(K=gname, a=a, b=b, pairs=len(mm), layouts=len(per), diff_gated=per["dg"].mean(),
                             ci_low=lo, ci_high=hi, diff_pass_pp=per["dp"].mean(),
                             significant=bool(lo > 0 or hi < 0)))
    paired = pd.DataFrame(prow)
    paired.to_csv(out / "tables" / "qw_paired_perK.csv", index=False)

    # ---- figure -------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(15 + 0.6 * max(0, len(Ks) - 5), 4.3))
    x = np.arange(len(Ks))
    s = summ[summ["K"].isin([str(k) for k in Ks])]
    for ax, col, ylab in [(axs[0], "R_gated", r"$\hat R_{\rm gated}$ (bit/s/Hz)"),
                          (axs[1], "pass_pct", "Pass (%)")]:
        for m in [REF] + METHODS:
            y = [float(s[(s["K"] == str(k)) & (s["method"] == m)][col].iloc[0]) for k in Ks]
            ax.plot(x, y, marker=MARK[m], color=COLOR[m], label=LABEL[m],
                    ls="--" if m == REF else "-", lw=2.2 if m == "gnn_qwqpso" else 1.5, ms=6)
        for k in unseen:
            i = Ks.index(k)
            ax.axvspan(i - 0.4, i + 0.4, color="0.9", zorder=0)
            ax.text(i, ax.get_ylim()[0], "unseen K", ha="center", va="bottom", fontsize=8, color="0.3")
        ax.set_xticks(x, [str(k) for k in Ks])
        ax.set_xlabel("Number of public users K")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    axs[0].set_title("(a) Gated covert rate on test")
    axs[1].set_title("(b) Pass rate on test")
    axs[0].legend(fontsize=8, loc="lower left")

    ax = axs[2]
    glabels = [str(k) for k in Ks] + [trained_lab]
    w = 0.36
    for j, (a, b) in enumerate(PAIRS):
        p = paired[(paired["a"] == a) & (paired["b"] == b)].set_index("K").loc[glabels]
        xx = np.arange(len(glabels)) + (j - 0.5) * w
        err = np.vstack([p["diff_gated"] - p["ci_low"], p["ci_high"] - p["diff_gated"]])
        ax.bar(xx, p["diff_gated"], w, yerr=err, capsize=3, color=COLOR[b], alpha=0.85,
               label=f"QW-QPSO - {LABEL[b].replace(' [paper]', '')}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(glabels)), glabels)
    ax.set_xlabel("K group")
    ax.set_ylabel(r"Paired $\Delta\hat R_{\rm gated}$ (95% CI, by layout)")
    ax.set_title("(c) Paired gain of QW-QPSO")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, axis="y")

    b_short = int(df[df["method"] == "gnn_pso"]["budget"].mode().iloc[0])
    fig.suptitle(f"Same budget B = {b_short} evaluations, paired trials", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / "figures" / f"fig_qw_compare.{ext}", dpi=300)
    plt.close(fig)
    print(summ.to_string(index=False))
    print(paired.to_string(index=False))
    print(f"[qw_compare] -> {out / 'figures' / 'fig_qw_compare.png'}")


if __name__ == "__main__":
    main()
