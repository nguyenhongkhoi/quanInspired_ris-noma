"""Story figures for the K-scaling run: quality vs cost of GNN + QW-QPSO.

Reads only existing CSVs (no experiment is run):
    <baseline-dir>/test/trials.csv   (full_pso, gnn_pso)
    <quantum-dir>/test/trials.csv    (gnn_pso_p, gnn_qwqpso)
Writes into <out-dir>/figures (png 300 dpi + pdf) and <out-dir>/tables:
    fig_ks_quality_cost   (a) % of Full PSO gated rate per K, (b) pass rate per K
    fig_ks_anytime        best feasible R_b on the search bank vs evaluations (log x), one panel per K group
    fig_ks_cost           % of Full PSO vs mean evaluations and vs mean wall time per trial
    fig_ks_outcomes       pass / point_only / point_fail per method, K = 6, 8, 10
    fig_ks_layouts        per-layout gated rate: QW-QPSO vs GNN+PSO+P (K = 5, 6, 8)
    tables/ks_cost.csv, tables/ks_layout_wins.csv

    python plot_kscale_story.py --baseline-dir bl_kscale_01 --quantum-dir qw_kscale_01 --out-dir cmp_kscale_01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from common.tables import load_csv

REF = "full_pso"
SHORT = ["gnn_pso", "gnn_pso_p", "gnn_qwqpso"]
ALL = [REF] + SHORT
LABEL = {"full_pso": "Full PSO (B=1800)", "gnn_pso": "GNN + short PSO [paper]",
         "gnn_pso_p": "GNN + short PSO + P", "gnn_qwqpso": "GNN + QW-QPSO (proposed)"}
SHORT_LABEL = {"full_pso": "Full PSO", "gnn_pso": "GNN+PSO", "gnn_pso_p": "GNN+PSO+P", "gnn_qwqpso": "QW-QPSO"}
COLOR = {"full_pso": "#6b6b66", "gnn_pso": "#2a78d6", "gnn_pso_p": "#eb6834", "gnn_qwqpso": "#1baf7a"}
MARK = {"full_pso": "s", "gnn_pso": "o", "gnn_pso_p": "^", "gnn_qwqpso": "D"}
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e4e3de"


def run_dir(arg: str) -> Path:
    p = Path(arg)
    if len(p.parts) == 1:
        return ROOT / "result" / arg
    return p if p.is_absolute() else ROOT / p


def style(ax) -> None:
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def save(fig, out: Path, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out / "figures" / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    print(f"[ks] -> {out / 'figures' / (stem + '.png')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", default="bl_kscale_01")
    ap.add_argument("--quantum-dir", default="qw_kscale_01")
    ap.add_argument("--out-dir", default="cmp_kscale_01")
    ap.add_argument("--k-train", default="2,3,4,5")
    args = ap.parse_args()
    k_train = [int(v) for v in args.k_train.split(",") if v.strip()]
    tlab = f"{min(k_train)}-{max(k_train)}"

    tb = load_csv(run_dir(args.baseline_dir) / "test" / "trials.csv")
    tq = load_csv(run_dir(args.quantum_dir) / "test" / "trials.csv")
    df = pd.concat([tb[tb["method"].isin([REF, "gnn_pso"])],
                    tq[tq["method"].isin(["gnn_pso_p", "gnn_qwqpso"])]], ignore_index=True)
    for m in ALL:
        assert (df["method"] == m).any(), f"method {m} missing"
    df["K"] = df["K"].astype(int)
    Ks = sorted(df["K"].unique())
    unseen = [k for k in Ks if k not in k_train]
    df["Kg"] = np.where(df["K"].isin(k_train), tlab, df["K"].astype(str))
    groups = [tlab] + [str(k) for k in unseen]
    out = run_dir(args.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})

    # ---- per-K summary ------------------------------------------------------
    def summ(d: pd.DataFrame) -> dict:
        ref = d[d["method"] == REF]["gated"].mean()
        r = {}
        for m in ALL:
            x = d[d["method"] == m]
            r[m] = dict(pass_pct=100 * x["passed"].mean(), gated=x["gated"].mean(),
                        pct_full=100 * x["gated"].mean() / ref if ref > 0 else np.nan,
                        evals=x["n_evals"].mean() if "n_evals" in x.columns else np.nan,
                        time_ms=x["time_ms"].mean() if "time_ms" in x.columns else np.nan)
        return r
    perK = {k: summ(df[df["K"] == k]) for k in Ks}
    perG = {g: summ(df[df["Kg"] == g]) for g in groups}

    # ---- Fig 1: quality per K ----------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
    x = np.arange(len(Ks))
    w = 0.26
    ax = axs[0]
    for j, m in enumerate(SHORT):
        y = [perK[k][m]["pct_full"] for k in Ks]
        ax.bar(x + (j - 1) * w, y, w - 0.03, color=COLOR[m], label=LABEL[m], edgecolor="white", lw=0.6)
    ax.axhline(100, color=COLOR[REF], ls="--", lw=1.2, label=LABEL[REF] + " = 100%")
    ax.axhline(99, color=INK2, ls=":", lw=1.0)
    ax.text(len(Ks) - 0.5, 99, "99%", va="bottom", ha="right", fontsize=7, color=INK2)
    ax.set_ylabel("Gated rate, % of Full PSO")
    ax.set_title("(a) Quality with B = 150 (12x fewer evaluations)")
    ax.legend(fontsize=7.5, loc="lower left", frameon=False)
    ax2 = axs[1]
    for m in ALL:
        y = [perK[k][m]["pass_pct"] for k in Ks]
        ax2.plot(x, y, marker=MARK[m], color=COLOR[m], label=SHORT_LABEL[m], lw=2.0, ms=6,
                 ls="--" if m == REF else "-", mec="white", mew=0.8)
    ax2.set_ylabel("Pass rate (%)")
    ax2.set_title("(b) Pass rate (independent assessment)")
    ax2.legend(fontsize=7.5, loc="lower left", frameon=False)
    for a in axs:
        style(a)
        a.set_xticks(x, [str(k) + ("*" if k in unseen else "") for k in Ks])
        a.set_xlabel("Number of public users K   (* = not seen in training)")
        for k in unseen:
            i = Ks.index(k)
            a.axvspan(i - 0.45, i + 0.45, color="#f3f2ee", zorder=0)
    axs[0].set_ylim(0, 105)
    axs[1].set_ylim(-3, 103)
    fig.tight_layout()
    save(fig, out, "fig_ks_quality_cost")
    plt.close(fig)

    # ---- Fig 2: anytime -----------------------------------------------------
    S = 30
    fig, axs = plt.subplots(1, len(groups), figsize=(3.3 * len(groups), 3.4), sharey=False)
    axs = np.atleast_1d(axs)
    for ax, g in zip(axs, groups):
        dg = df[df["Kg"] == g]
        ref_final = None
        curves = {}
        for m in ALL:
            h = dg[(dg["method"] == m) & (dg["hist_rb"].fillna("") != "")]["hist_rb"]
            if h.empty:
                continue
            H = [json.loads(v) for v in h]
            T = max(len(v) for v in H)
            A = np.zeros((len(H), T))
            for i, v in enumerate(H):
                a = np.array([0.0 if t is None else t for t in v])
                A[i, :len(a)] = a
                A[i, len(a):] = a[-1]
            curves[m] = ((np.arange(T) + 1) * S, A.mean(axis=0))
        if REF in curves:
            ref_final = curves[REF][1][-1]
        for m, (ev, y) in curves.items():
            yy = 100 * y / ref_final if ref_final else y
            ax.plot(ev, yy, color=COLOR[m], lw=2.0, ls="--" if m == REF else "-",
                    marker=MARK[m] if m != REF else None, ms=4, label=SHORT_LABEL[m])
        ax.axvline(150, color=INK2, lw=0.8, ls=":")
        ax.text(150, 2, " B=150", fontsize=7, color=INK2)
        ax.set_xscale("log")
        ax.set_ylim(0, 105)
        ax.set_xlabel("Objective evaluations (log)")
        ax.set_title(f"K = {g}" + (" (trained)" if g == tlab else " (unseen)"))
        style(ax)
    axs[0].set_ylabel("Best feasible R_b, % of Full PSO final\n(search bank, mean over trials)")
    axs[0].legend(fontsize=7.5, loc="lower right", frameon=False)
    fig.tight_layout()
    save(fig, out, "fig_ks_anytime")
    plt.close(fig)

    # ---- Fig 3: cost (evals and wall time) vs quality ----------------------
    rows = []
    for g in groups:
        for m in ALL:
            r = perG[g][m]
            rows.append(dict(Kgroup=g, method=m, evals=r["evals"], time_ms=r["time_ms"],
                             pass_pct=r["pass_pct"], R_gated=r["gated"], pct_of_full=r["pct_full"]))
    cost = pd.DataFrame(rows)
    cost.to_csv(out / "tables" / "ks_cost.csv", index=False)
    show = [g for g in groups if g in (tlab, "6", "8")]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, col, xl in [(axs[0], "evals", "Mean objective evaluations per trial (log)"),
                        (axs[1], "time_ms", "Mean allocation time per trial, ms (log)")]:
        if cost[col].isna().all():
            ax.text(0.5, 0.5, f"no '{col}' column", transform=ax.transAxes, ha="center")
            continue
        for m in ALL:
            c = cost[(cost["method"] == m) & cost["Kgroup"].isin(show)]
            ax.scatter(c[col], c["pct_of_full"], s=60, marker=MARK[m], color=COLOR[m],
                       edgecolor="white", lw=0.8, label=SHORT_LABEL[m], zorder=3)
            for _, r in c.iterrows():
                ax.annotate(f"K={r['Kgroup']}", (r[col], r["pct_of_full"]), xytext=(5, -3),
                            textcoords="offset points", fontsize=6.5, color=INK2)
        ax.set_xscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel("Gated rate, % of Full PSO")
        style(ax)
    axs[0].set_title("(a) Quality vs search budget")
    axs[1].set_title("(b) Quality vs wall time (same machine, 1 thread/worker)")
    axs[0].legend(fontsize=7.5, loc="lower right", frameon=False)
    fig.tight_layout()
    save(fig, out, "fig_ks_cost")
    plt.close(fig)

    # ---- Fig 4: check outcomes ---------------------------------------------
    ks_o = [k for k in unseen]
    fig, axs = plt.subplots(1, len(ks_o), figsize=(3.6 * len(ks_o), 3.6), sharey=True)
    axs = np.atleast_1d(axs)
    cat_col = {"pass": "#1baf7a", "point_only": "#eda100", "point_fail": "#c9c8c1"}
    for ax, k in zip(axs, ks_o):
        d = df[df["K"] == k]
        bottom = np.zeros(len(ALL))
        for cat in ("pass", "point_only", "point_fail"):
            v = np.array([100 * (d[d["method"] == m]["category"] == cat).mean() for m in ALL])
            ax.bar(np.arange(len(ALL)), v, 0.7, bottom=bottom, color=cat_col[cat], label=cat,
                   edgecolor="white", lw=1.0)
            bottom += v
        for i, m in enumerate(ALL):
            pr = 100 * (~d[d["method"] == m]["pt_rate"]).mean() if "pt_rate" in d.columns else np.nan
            ax.text(i, 101, f"QoS fail\n{pr:.0f}%", ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax.set_xticks(np.arange(len(ALL)), [SHORT_LABEL[m] for m in ALL], rotation=25, ha="right")
        ax.set_title(f"K = {k} (unseen)", pad=22)
        ax.set_ylim(0, 100)
        style(ax)
    axs[0].set_ylabel("Share of trials (%)")
    axs[-1].legend(fontsize=7.5, loc="lower right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    save(fig, out, "fig_ks_outcomes")
    plt.close(fig)

    # ---- Fig 5: per-layout scatter -----------------------------------------
    ks_l = [k for k in (max(k_train), 6, 8) if k in Ks]
    fig, axs = plt.subplots(1, len(ks_l), figsize=(3.6 * len(ks_l), 3.6))
    axs = np.atleast_1d(axs)
    wins = []
    for ax, k in zip(axs, ks_l):
        d = df[df["K"] == k]
        per = d.groupby(["layout_id", "method"])["gated"].mean().unstack()
        a, b = per["gnn_qwqpso"], per["gnn_pso_p"]
        nw, nt, nl = int((a > b + 1e-9).sum()), int((abs(a - b) <= 1e-9).sum()), int((a < b - 1e-9).sum())
        wins.append(dict(K=k, layouts=len(per), qw_better=nw, tie=nt, qw_worse=nl))
        lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
        pad = 0.03 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK2, lw=0.8, ls="--")
        ax.scatter(b, a, s=26, color=COLOR["gnn_qwqpso"], edgecolor="white", lw=0.6, zorder=3)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("GNN + PSO + P (gated, per layout)")
        ax.set_title(f"K = {k}: QW better {nw} / tie {nt} / worse {nl}")
        style(ax)
    axs[0].set_ylabel("GNN + QW-QPSO (gated, per layout)")
    fig.tight_layout()
    save(fig, out, "fig_ks_layouts")
    plt.close(fig)
    pd.DataFrame(wins).to_csv(out / "tables" / "ks_layout_wins.csv", index=False)

    print(cost.to_string(index=False))
    print(pd.DataFrame(wins).to_string(index=False))


if __name__ == "__main__":
    main()
