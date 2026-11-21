"""Tables (CSV + console + tables.md) and figures (PNG 300 dpi + PDF) — claude.md 10.4.

Everything is computed from CSV files inside the run directory, so
`--stage tables` regenerates all outputs without re-running experiments.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExpConfig, SysConfig
from .methods import DISPLAY

BOOL_COLS = ("passed", "search_feasible", "init_feasible", "eligible", "mb_gt_ceiling",
             "pt_xi_emp", "pt_xi_gamma", "pt_rate", "pt_outage", "pt_domain", "pt_sic", "pt_eligible",
             "bd_xi", "bd_rate", "bd_outage", "has_label", "la_passed", "pso_best_feasible")

ORDER = list(DISPLAY.keys())
COLORS = {
    "full_pso": "#1f1f1f", "rand_pso": "#8c8c8c", "direct_mlp": "#9ecae1", "direct_gnn": "#fdae6b",
    "mlp_pso": "#3182bd", "gnn_pso": "#e6550d",
    "full_qpso": "#54278f", "rand_qpso": "#bcbddc", "mlp_qpso": "#31a354", "gnn_qpso": "#d62728",
    "gnn_aqpso": "#e7298a", "direct_mlp_raw": "#c6dbef", "direct_gnn_raw": "#fdd0a2",
}
MARKERS = {"full_pso": "s", "rand_pso": "o", "direct_mlp": "v", "direct_gnn": "^", "mlp_pso": "D",
           "gnn_pso": "*", "full_qpso": "P", "rand_qpso": "X", "mlp_qpso": "d", "gnn_qpso": "h",
           "gnn_aqpso": "p"}


# ---------------------------------------------------------------------------
# Loading / helpers
# ---------------------------------------------------------------------------
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str) == "True"
    if "passed" in df.columns and "R_b" in df.columns:
        df["gated"] = np.where(df["passed"], df["R_b"], 0.0)
    if "tag" in df.columns:
        df["tag"] = df["tag"].fillna("").astype(str)
    return df


def methods_in(df: pd.DataFrame) -> list[str]:
    present = set(df["method"].unique())
    return [m for m in ORDER if m in present] + sorted(present - set(ORDER))


def name(m: str) -> str:
    return DISPLAY.get(m, m)


def kgroup_label(exp: ExpConfig) -> str:
    kt = sorted(exp.k_train)
    return f"{kt[0]}-{kt[-1]}" if len(kt) > 1 else str(kt[0])


def add_kgroup(df: pd.DataFrame, exp: ExpConfig) -> pd.DataFrame:
    lab = kgroup_label(exp)
    df = df.copy()
    df["Kgroup"] = [lab if k in exp.k_train else str(k) for k in df["K"]]
    return df


def boot_ci(per_layout: np.ndarray, n: int, seed: int) -> tuple[float, float]:
    per_layout = np.asarray(per_layout, float)
    if per_layout.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, per_layout.size, (n, per_layout.size))
    means = per_layout[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def layout_cv(df: pd.DataFrame, exp: ExpConfig) -> pd.Series:
    """Per-layout Willie CV (mean over its trained-K trials), threshold groups."""
    d = df[df["K"].isin(exp.k_train)] if df["K"].isin(exp.k_train).any() else df
    return d.groupby("layout_id")["cv_w"].mean()


def to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    cols = list(df.columns)

    def f(v):
        if isinstance(v, (float, np.floating)):
            return "nan" if np.isnan(v) else format(v, floatfmt)
        return str(v)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f(r[c]) for c in cols) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in methods_in(df):
        d = df[df["method"] == m]
        rows.append(dict(Method=name(m), key=m, Trials=len(d), Passed=int(d["passed"].sum()),
                         **{"Pass(%)": 100.0 * d["passed"].mean(), "R_gated": d["gated"].mean(),
                            "Time(ms)": d["time_ms"].mean(), "Infer(ms)": d["infer_ms"].mean(),
                            "B": int(d["budget"].mode().iloc[0])}))
    return pd.DataFrame(rows)


def paired_table(df: pd.DataFrame, exp: ExpConfig, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """(a - b) per layout mean of gated rate and pass rate, bootstrap over layouts."""
    rows = []
    for kg in sorted(df["Kgroup"].unique(), key=lambda s: (len(s), s)):
        d = df[df["Kgroup"] == kg]
        for a, b in pairs:
            A = d[d["method"] == a][["case_id", "replicate", "layout_id", "gated", "passed"]]
            B = d[d["method"] == b][["case_id", "replicate", "gated", "passed"]]
            if A.empty or B.empty:
                continue
            mm = A.merge(B, on=["case_id", "replicate"], suffixes=("_a", "_b"))
            if mm.empty:
                continue
            mm["dg"] = mm["gated_a"] - mm["gated_b"]
            mm["dp"] = 100.0 * (mm["passed_a"].astype(float) - mm["passed_b"].astype(float))
            per = mm.groupby("layout_id")[["dg", "dp"]].mean()
            lo, hi = boot_ci(per["dg"].values, exp.bootstrap_n, exp.bootstrap_seed)
            plo, phi = boot_ci(per["dp"].values, exp.bootstrap_n, exp.bootstrap_seed)
            sig = "khong khac biet co y nghia" if lo <= 0.0 <= hi else ("A > B" if lo > 0 else "A < B")
            rows.append(dict(K=kg, A=name(a), B=name(b), a=a, b=b, pairs=len(mm), layouts=len(per),
                             diff_gated=per["dg"].mean(), ci_low=lo, ci_high=hi,
                             diff_pass_pp=per["dp"].mean(), pass_ci_low=plo, pass_ci_high=phi, verdict=sig))
    return pd.DataFrame(rows)


def gamma_dep_table(df: pd.DataFrame, exp: ExpConfig, sys_: SysConfig, ref: str) -> pd.DataFrame:
    d = df[df["method"] == ref].copy()
    if d.empty:
        return pd.DataFrame()
    cv = layout_cv(df, exp)
    d["CVgroup"] = np.where(d["layout_id"].map(cv) < sys_.cv_threshold, "Low", "Other")
    d["delta"] = d["xi_gamma"] - d["xi_emp"]
    rows = []
    for (kg, cg), g in d.groupby(["Kgroup", "CVgroup"]):
        a = g["delta"].abs()
        rows.append(dict(K=kg, CVgroup=cg, Trials=len(g), mean_delta=g["delta"].mean(), mean_abs=a.mean(),
                         p95_abs=float(np.percentile(a, 95)), max_abs=a.max()))
    return pd.DataFrame(rows)


def cv_group_table(df: pd.DataFrame, exp: ExpConfig, sys_: SysConfig) -> pd.DataFrame:
    d = df[df["K"].isin(exp.k_train)].copy()
    if d.empty:
        return pd.DataFrame()
    cv = layout_cv(df, exp)
    d["CVgroup"] = np.where(d["layout_id"].map(cv) < sys_.cv_threshold, "Low", "Other")
    rows = []
    for m in methods_in(d):
        for cg in ("Low", "Other"):
            g = d[(d["method"] == m) & (d["CVgroup"] == cg)]
            if g.empty:
                continue
            per = g.groupby("layout_id")["gated"].mean()
            lo, hi = boot_ci(per.values, exp.bootstrap_n, exp.bootstrap_seed)
            per_p = 100 * g.groupby("layout_id")["passed"].mean()
            plo, phi = boot_ci(per_p.values, exp.bootstrap_n, exp.bootstrap_seed)
            rows.append(dict(Method=name(m), key=m, CVgroup=cg, layouts=g["layout_id"].nunique(), Trials=len(g),
                             **{"Pass(%)": 100 * g["passed"].mean(), "pass_lo": plo, "pass_hi": phi,
                                "R_gated": g["gated"].mean(), "gated_lo": lo, "gated_hi": hi}))
    return pd.DataFrame(rows)


def check_outcome_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kg in sorted(df["Kgroup"].unique(), key=lambda s: (len(s), s)):
        for m in methods_in(df):
            g = df[(df["Kgroup"] == kg) & (df["method"] == m)]
            if g.empty:
                continue
            vc = g["category"].value_counts()
            n = len(g)
            rows.append(dict(K=kg, Method=name(m), key=m, Trials=n,
                             **{"pass(%)": 100 * vc.get("pass", 0) / n,
                                "point_only(%)": 100 * vc.get("point_only", 0) / n,
                                "point_fail(%)": 100 * vc.get("point_fail", 0) / n,
                                "fail_bd_xi": int((~g["bd_xi"]).sum()), "fail_bd_rate": int((~g["bd_rate"]).sum()),
                                "fail_pt_rate": int((~g["pt_rate"]).sum()), "fail_pt_xi_emp": int((~g["pt_xi_emp"]).sum())}))
    return pd.DataFrame(rows)


def diagnostics_table(df: pd.DataFrame, labels_dir: Path | None) -> pd.DataFrame:
    rows = []
    for m in methods_in(df):
        g = df[df["method"] == m]
        rows.append(dict(item="method", Method=name(m), key=m, Trials=len(g),
                         rb_over_ceiling=float((g["R_b"] / g["rb_ceiling"]).mean()),
                         nonpass_mb_gt_ceiling=int((~g["passed"] & g["mb_gt_ceiling"]).sum()),
                         outage_event_trials=int((g["F_b"] > 0).sum()),
                         search_infeasible_but_pass=int((~g["search_feasible"] & g["passed"]).sum()),
                         ineligible=int((~g["eligible"]).sum())))
    if labels_dir is not None:
        for split in ("train", "val"):
            p = labels_dir / f"labels_{split}.csv"
            if p.exists():
                L = load_csv(p)
                lab = L[L["has_label"]]
                rows.append(dict(item=f"label_coverage_{split}", Method="-", key="-", Trials=len(L),
                                 labelled=int(L["has_label"].sum()),
                                 coverage_pct=100 * L["has_label"].mean(),
                                 label_assess_pass_pct=100 * lab["la_passed"].mean() if len(lab) else float("nan")))
    return pd.DataFrame(rows)


def val_budget_table(val: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, B), g in val.groupby(["method", "budget"]):
        if m.endswith("_raw"):
            continue
        rows.append(dict(Method=name(m), key=m, B=int(B), Trials=len(g), **{
            "Pass(%)": 100 * g["passed"].mean(), "R_gated": g["gated"].mean(), "Time(ms)": g["time_ms"].mean()}))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["_o"] = out["key"].map({k: i for i, k in enumerate(ORDER)})
        out = out.sort_values(["_o", "B"]).drop(columns="_o")
    return out


def table1_params(sys_: SysConfig) -> pd.DataFrame:
    r = [("RIS elements; carrier; spacing", f"{sys_.N}; {sys_.fc_hz / 1e9:g} GHz; lambda_c/2"),
         ("Rician factor; distortion coefficient", f"K_R={sys_.K_R:g}; kappa={sys_.kappa:g}"),
         ("Path-loss exponents", f"alpha_ar={sys_.alpha_ar:g}, alpha_ru={sys_.alpha_ru:g}"),
         ("Reference distance; reference gains", f"d0={sys_.d0:g} m; beta0={sys_.beta0:g}"),
         ("Minimum link distance", f"{sys_.d_min:g} m"),
         ("RIS and receiver noise powers", f"{sys_.sigma_r2_dbm:g} dBm each"),
         ("Per-element switching / circuit power", f"{sys_.P_S_mw:g} / {sys_.P_DC_mw:g} mW"),
         ("Total power; required DEP", f"{sys_.P_tot_dbm_range[0]:g}-{sys_.P_tot_dbm_range[1]:g} dBm; "
                                       f"{sys_.xi_req_range[0]:g}-{sys_.xi_req_range[1]:g}"),
         ("Public mean-rate requirement", f"R_QoS={sys_.R_qos:g} bit/s/Hz"),
         ("Public / Bob fixed decoding targets", f"{sys_.R_dec_pub:g} / {sys_.R_dec_bob:g} bit/s/Hz"),
         ("Bob-outage limit", f"p_max={sys_.p_max:g}"),
         ("Covert-fraction bounds", f"{sys_.mb_min:g} <= m_b <= {sys_.mb_max:g}"),
         ("RIS power-share bounds", f"{sys_.s_min:g} <= s <= {sys_.s_max:g}"),
         ("Search DEP margin; rate margin", f"{sys_.xi_margin:g}; {sys_.rate_margin:g} bit/s/Hz")]
    return pd.DataFrame(r, columns=["Parameter", "Setting"])


def table6_network() -> pd.DataFrame:
    from .models import N_PARAMS
    r = [("Global encoder (both)", "24 -> 32", "R"), ("Node encoder (both)", "47 -> 32", "R"),
         ("Bob/split head (both)", "96 -> 32 -> 2", "R, I"), ("Public head (both)", "128 -> 32 -> 1", "R, I"),
         ("GNN message, per block", "66 -> 32 -> 32", "R, R"), ("GNN update, per block", "96 -> 32 -> 32", "R, I"),
         ("MLP pointwise, per block", "32 -> 112 -> 32", "R, I"),
         ("Total parameters", f"GNN {N_PARAMS['gnn']:,}; MLP {N_PARAMS['mlp']:,}", "-")]
    return pd.DataFrame(r, columns=["Component", "Dimensions", "Activations"])


def table7_checkpoints(models_dir: Path | None) -> pd.DataFrame:
    if models_dir is None or not (models_dir / "checkpoints.json").exists():
        return pd.DataFrame()
    meta = json.loads((models_dir / "checkpoints.json").read_text(encoding="utf-8"))
    rows = [dict(Model="GNN" if m["arch"] == "gnn" else "Pooled MLP", Seed=m["seed"], Epoch=m["epoch"],
                 val_loss=m["val_loss"], delta=m["delta"], params=m["n_params"], device=m["device"]) for m in meta]
    return pd.DataFrame(rows)


def table8_adjust(val: pd.DataFrame | None) -> pd.DataFrame:
    if val is None:
        return pd.DataFrame()
    rows = []
    for arch, lab in (("gnn", "GNN"), ("mlp", "Pooled MLP")):
        for key, dlt in ((f"direct_{arch}_raw", 0.0), (f"direct_{arch}", None)):
            g = val[val["method"] == key]
            if g.empty:
                continue
            rows.append(dict(Architecture=lab, delta="0" if dlt == 0.0 else "adjusted",
                             Passed=int(g["passed"].sum()), Trials=len(g), **{"Pass(%)": 100 * g["passed"].mean()}))
    return pd.DataFrame(rows)


def anytime_curves(df: pd.DataFrame, swarm: int) -> pd.DataFrame:
    """Mean search-bank best feasible R_b (infeasible -> 0) versus evaluations, per K group."""
    rows = []
    kgs = sorted(df["Kgroup"].unique(), key=lambda s: (len(s), s)) if "Kgroup" in df.columns else ["all"]
    for kg in kgs:
        dk = df[df["Kgroup"] == kg] if kg != "all" else df
        for m in methods_in(dk):
            g = dk[(dk["method"] == m) & (dk["hist_rb"].fillna("") != "")]
            if g.empty:
                continue
            H = [json.loads(h) for h in g["hist_rb"]]
            T = max(len(h) for h in H)
            A = np.zeros((len(H), T))
            for i, h in enumerate(H):
                v = np.array([0.0 if x is None else x for x in h])
                A[i, :len(v)] = v
                A[i, len(v):] = v[-1]
            mean = A.mean(axis=0)
            for t in range(T):
                rows.append(dict(K=kg, key=m, Method=name(m), evals=(t + 1) * swarm, mean_best_rb_search=mean[t]))
    return pd.DataFrame(rows)


def anytime_target(at: pd.DataFrame, full_key: str = "full_pso", frac: float = 0.99) -> pd.DataFrame:
    """Evaluations each method needs to reach frac x (Full reference final value), per K group."""
    if at.empty or full_key not in set(at["key"]):
        return pd.DataFrame()
    rows = []
    for kg, d in at.groupby("K", sort=False):
        f = d[d["key"] == full_key]
        if f.empty:
            continue
        target = frac * f["mean_best_rb_search"].iloc[-1]
        for m in list(dict.fromkeys(d["key"])):
            g = d[d["key"] == m]
            hit = g[g["mean_best_rb_search"] >= target]
            rows.append(dict(K=kg, Method=name(m), key=m, target=target,
                             evals_to_target=int(hit["evals"].iloc[0]) if not hit.empty else None,
                             final_evals=int(g["evals"].iloc[-1]), final_value=g["mean_best_rb_search"].iloc[-1]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight"})
    return plt


def _save(fig, fig_dir: Path, stem: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{stem}.png", dpi=300)
    fig.savefig(fig_dir / f"{stem}.pdf")
    import matplotlib.pyplot as plt
    plt.close(fig)


def fig1_layout(sys_: SysConfig, exp: ExpConfig, cases: pd.DataFrame | None, fig_dir: Path) -> None:
    from matplotlib.patches import Rectangle
    from .channel import element_positions, generate_layout
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw=dict(width_ratios=[1.0, 1.25]))
    for a in ax:
        for reg, col, lab in ((sys_.region_bob, "#e6550d", "Bob region"),
                              (sys_.region_willie, "#54278f", "Willie region"),
                              (sys_.region_pub, "#3182bd", "Public region")):
            (x0, x1), (y0, y1) = reg
            a.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=col, lw=1.2, ls="--", label=lab))
        pe = element_positions(sys_)
        a.plot(pe[:, 0], pe[:, 1], color="k", lw=3, label="RIS (ULA)")
    ax[0].plot(*sys_.pos_alice, "ks", ms=8, label="Alice")
    ax[0].annotate("Alice", sys_.pos_alice, xytext=(4, 6), textcoords="offset points")
    ax[0].annotate("RIS", sys_.pos_ris, xytext=(-25, 8), textcoords="offset points")
    ax[0].plot([sys_.pos_alice[0], sys_.pos_ris[0]], [sys_.pos_alice[1], sys_.pos_ris[1]], "k:", lw=1)
    ax[0].set_xlim(-5, 130)
    ax[0].set_ylim(-20, 25)
    ax[0].set_title("(a) Geometry (direct Alice-receiver links blocked)")
    lids = sorted(cases["layout_id"].unique()) if cases is not None and not cases.empty else []
    for i, lid in enumerate(lids):
        split = cases[cases["layout_id"] == lid]["split"].iloc[0]
        L = generate_layout(sys_, exp.seed, int(lid), split, exp.K_max)
        kw = dict(alpha=0.6, ms=4)
        ax[1].plot(*L.pos_bob, "o", color="#e6550d", label="Bob" if i == 0 else None, **kw)
        ax[1].plot(*L.pos_willie, "^", color="#54278f", label="Willie" if i == 0 else None, **kw)
        ax[1].plot(L.pos_pub[:, 0], L.pos_pub[:, 1], ".", color="#3182bd", label="Public users" if i == 0 else None, **kw)
    ax[1].set_xlim(92, 125)
    ax[1].set_ylim(-10, 14)
    ax[1].set_title(f"(b) Sampled layouts ({len(lids)} layouts, K_max={exp.K_max})")
    for a in ax:
        a.set_xlabel("x (m)")
        a.set_ylabel("y (m)")
        a.set_aspect("equal", adjustable="box")
    ax[1].legend(loc="upper right", fontsize=7)
    ax[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Fig. 1. Multiuser covert downlink assisted by an active RIS")
    _save(fig, fig_dir, "fig1_layout")


def fig2_pipeline(fig_dir: Path, optimizer: str = "PSO") -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_axis_off()
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.6)

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=fc, ec="k", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, lw=0.9))

    ax.text(0.1, 4.35, "(a) Offline supervision", fontsize=9, weight="bold")
    yA = 3.2
    xs = [0.1, 1.95, 3.8, 5.65, 7.5, 9.35]
    txt = ["Training\nscenarios", f"Full {optimizer}\ncandidate archives", "Independent\nlabel selection",
           "Train GNN /\npooled MLP", "Validation\nselection", "Fixed models +\noutput adjustment"]
    for i, (x, t) in enumerate(zip(xs, txt)):
        box(x, yA, 1.5, 0.8, t, "#deebf7")
        if i:
            arrow(xs[i - 1] + 1.5, yA + 0.4, x, yA + 0.4)
    box(3.8, 2.35, 1.5, 0.55, "Independent\nlabel assessment", "#f0f0f0")
    arrow(4.55, yA, 4.55, 2.9)
    ax.text(5.65, 2.55, "training-only normalisation; labels assessed, never reselected", fontsize=7, style="italic")

    ax.text(0.1, 1.85, "(b) Online allocation", fontsize=9, weight="bold")
    yB = 0.7
    xs2 = [0.1, 1.95, 3.8, 5.65, 7.5, 9.35]
    txt2 = ["New case:\nstatistical features", "Trained network\n+ fixed adjustment", "Direct allocation\nor 1 learned particle",
            f"Short {optimizer}\n(+29 random particles)", "Fixed returned\nallocation", "Independent\nfeasibility assessment"]
    for i, (x, t) in enumerate(zip(xs2, txt2)):
        box(x, yB, 1.5, 0.8, t, "#fee6ce")
        if i:
            arrow(xs2[i - 1] + 1.5, yB + 0.4, x, yB + 0.4)
    arrow(10.1, yA, 10.1, yB + 0.8)
    ax.text(5.5, 0.25, "Report pass rate, gated rate and allocation time. No assessment feedback to prediction or search.",
            ha="center", fontsize=7.5, style="italic")
    fig.suptitle("Fig. 2. Offline supervision and online allocation with independent assessment")
    _save(fig, fig_dir, "fig2_pipeline")


def fig3_val_budget(vb: pd.DataFrame, fig_dir: Path, title: str = "") -> None:
    if vb.empty:
        return
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    full_keys = [k for k in vb["key"].unique() if k.startswith("full_")]
    for k in [k for k in methods_in(vb.rename(columns={"key": "method"})) if not k.startswith("full_") and not k.startswith("direct")]:
        g = vb[vb["key"] == k].sort_values("B")
        for a, col in zip(ax, ("Pass(%)", "R_gated", "Time(ms)")):
            a.plot(g["B"], g[col], marker=MARKERS.get(k, "o"), color=COLORS.get(k), label=name(k))
    for k in full_keys:
        g = vb[vb["key"] == k]
        for a, col in zip(ax, ("Pass(%)", "R_gated", "Time(ms)")):
            a.axhline(float(g[col].iloc[0]), color=COLORS.get(k, "k"), ls="--", lw=1, label=f"{name(k)} ({int(g['B'].iloc[0])})")
    for a, yl in zip(ax, ("Assessment pass rate (%)", "Gated covert rate (bit/s/Hz)", "Mean allocation time (ms)")):
        a.set_xlabel("Objective evaluations")
        a.set_ylabel(yl)
        a.set_xticks(sorted(vb[~vb["key"].str.startswith(("full_", "direct"))]["B"].unique()))
    ax[0].legend(fontsize=7)
    fig.suptitle(f"Fig. 3. Validation performance versus objective evaluations {title}")
    _save(fig, fig_dir, "fig3_val_budget")


def fig4_vs_K(df: pd.DataFrame, exp: ExpConfig, fig_dir: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    unseen = [k for k in sorted(df["K"].unique()) if k not in exp.k_train]
    for m in methods_in(df):
        g = df[df["method"] == m].groupby("K").agg(p=("passed", "mean"), r=("gated", "mean")).reset_index()
        ax[0].plot(g["K"], 100 * g["p"], marker=MARKERS.get(m, "o"), color=COLORS.get(m), label=name(m))
        ax[1].plot(g["K"], g["r"], marker=MARKERS.get(m, "o"), color=COLORS.get(m), label=name(m))
    for a in ax:
        for k in unseen:
            a.axvspan(k - 0.4, k + 0.4, color="#ffeda0", alpha=0.6)
            a.text(k, a.get_ylim()[0], "Unseen K", ha="center", va="bottom", fontsize=7)
        a.set_xlabel("Number of public users K")
        a.set_xticks(sorted(df["K"].unique()))
    ax[0].set_ylabel("Assessment pass rate (%)")
    ax[1].set_ylabel("Gated covert rate (bit/s/Hz)")
    ax[1].legend(fontsize=7)
    fig.suptitle("Fig. 4. Held-out performance versus public-user count")
    _save(fig, fig_dir, "fig4_vs_K")


def fig5_rate_vs_time(df: pd.DataFrame, exp: ExpConfig, fig_dir: Path) -> None:
    plt = _plt()
    groups = sorted(df["Kgroup"].unique(), key=lambda s: (len(s), s))
    fig, ax = plt.subplots(1, len(groups), figsize=(5 * len(groups), 3.8), squeeze=False)
    for a, kg in zip(ax[0], groups):
        d = df[df["Kgroup"] == kg]
        for m in methods_in(d):
            g = d[d["method"] == m]
            per = g.groupby("layout_id")["gated"].mean()
            lo, hi = boot_ci(per.values, exp.bootstrap_n, exp.bootstrap_seed)
            y = float(per.mean())                       # same statistic as the bootstrap
            a.errorbar(g["time_ms"].mean(), y, yerr=[[max(0.0, y - lo)], [max(0.0, hi - y)]], fmt=MARKERS.get(m, "o"),
                       color=COLORS.get(m), capsize=3, label=name(m))
        a.set_xscale("log")
        a.set_xlabel("Mean allocation time (ms; log scale)")
        a.set_ylabel("Gated covert rate (bit/s/Hz)")
        a.set_title(f"K = {kg}")
    ax[0][-1].legend(fontsize=7)
    fig.suptitle("Fig. 5. Gated covert rate versus mean allocation time (95% layout-bootstrap CI)")
    _save(fig, fig_dir, "fig5_rate_vs_time")


def fig6_cv(cvt: pd.DataFrame, fig_dir: Path) -> None:
    if cvt.empty:
        return
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    groups = ["Low", "Other"]
    meths = list(dict.fromkeys(cvt["key"]))
    w = 0.8 / max(1, len(meths))
    for i, m in enumerate(meths):
        g = cvt[cvt["key"] == m].set_index("CVgroup").reindex(groups)
        x = np.arange(len(groups)) + (i - (len(meths) - 1) / 2) * w
        for a, col, lo, hi in ((ax[0], "Pass(%)", "pass_lo", "pass_hi"), (ax[1], "R_gated", "gated_lo", "gated_hi")):
            y = g[col].values
            err = np.nan_to_num(np.vstack([y - g[lo].values, g[hi].values - y]), nan=0.0)
            a.bar(x, y, w, color=COLORS.get(m), label=name(m),
                  yerr=np.maximum(err, 0.0), capsize=2, ecolor="k", error_kw=dict(lw=0.6))
    lay = cvt.groupby("CVgroup")["layouts"].max().reindex(groups).fillna(0).astype(int)
    for a in ax:
        a.set_xticks(range(len(groups)))
        a.set_xticklabels([f"Low CV (<0.9)\n{lay['Low']} layouts", f"Other (>=0.9)\n{lay['Other']} layouts"])
    ax[0].set_ylabel("Assessment pass rate (%)")
    ax[1].set_ylabel("Gated covert rate (bit/s/Hz)")
    ax[1].legend(fontsize=7)
    fig.suptitle("Fig. 6. Held-out performance across Willie-channel variability groups (trained K)")
    _save(fig, fig_dir, "fig6_cv_groups")


def fig7_outcomes(cot: pd.DataFrame, fig_dir: Path) -> None:
    if cot.empty:
        return
    plt = _plt()
    groups = sorted(cot["K"].unique(), key=lambda s: (len(s), s))
    fig, ax = plt.subplots(1, len(groups), figsize=(6 * len(groups), 0.45 * cot["key"].nunique() + 1.5), squeeze=False)
    cols = (("pass(%)", "#31a354", "All assessment checks passed"),
            ("point_only(%)", "#fdae6b", "Point checks passed; confidence bounds failed"),
            ("point_fail(%)", "#de2d26", "At least one point check failed"))
    for a, kg in zip(ax[0], groups):
        g = cot[cot["K"] == kg].iloc[::-1]
        left = np.zeros(len(g))
        for c, colr, lab in cols:
            a.barh(g["Method"], g[c], left=left, color=colr, label=lab)
            left += g[c].values
        for y, v in enumerate(g["pass(%)"].values):
            a.text(min(v, 99) - 1, y, f"{v:.1f}%", va="center", ha="right", fontsize=7, color="w")
        a.set_xlim(0, 100)
        a.set_xlabel("Share of allocation trials (%)")
        a.set_title(f"K = {kg}: {int(cot[cot['K'] == kg]['Trials'].max())} trials / method")
    ax[0][0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Fig. 7. Point-check and confidence-bound outcomes on held-out layouts")
    _save(fig, fig_dir, "fig7_check_outcomes")


def fig_paired(pt: pd.DataFrame, fig_dir: Path, stem: str = "fig_paired_diff") -> None:
    if pt.empty:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(pt) + 1.2))
    lab = [f"K={r.K}: {r.A} - {r.B}" for r in pt.itertuples()]
    y = np.arange(len(pt))[::-1]
    xerr = np.maximum(np.vstack([pt["diff_gated"] - pt["ci_low"], pt["ci_high"] - pt["diff_gated"]]), 0.0)
    ax.errorbar(pt["diff_gated"], y, xerr=xerr, fmt="o", color="k", capsize=3)
    ax.axvline(0, color="r", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=7)
    ax.set_xlabel("Paired gated-rate difference (bit/s/Hz), 95% layout-bootstrap CI")
    fig.suptitle("Paired differences")
    _save(fig, fig_dir, stem)


def fig_anytime(at: pd.DataFrame, fig_dir: Path, full_key: str = "full_pso") -> None:
    if at.empty:
        return
    plt = _plt()
    groups = list(dict.fromkeys(at["K"]))
    fig, axs = plt.subplots(1, len(groups), figsize=(6.5 * len(groups), 4), squeeze=False)
    for ax, kg in zip(axs[0], groups):
        d = at[at["K"] == kg]
        for m in list(dict.fromkeys(d["key"])):
            g = d[d["key"] == m]
            ls = "--" if m.endswith("qpso") else "-"
            ax.plot(g["evals"], g["mean_best_rb_search"], color=COLORS.get(m), ls=ls, label=name(m))
        if full_key in set(d["key"]):
            tgt = 0.99 * d[d["key"] == full_key]["mean_best_rb_search"].iloc[-1]
            ax.axhline(tgt, color="k", ls=":", lw=1, label=f"99% of {name(full_key)} final")
        ax.set_xscale("log")
        ax.set_xlabel("Objective evaluations (log)")
        ax.set_ylabel("Mean best feasible R_b on search bank")
        ax.set_title(f"K = {kg}")
    axs[0][-1].legend(fontsize=7)
    fig.suptitle("Anytime curves (search-bank objective; infeasible = 0)")
    _save(fig, fig_dir, "fig_anytime")


def fig_anchor_grid(val: pd.DataFrame, fig_dir: Path) -> None:
    g = val[(val["method"] == "gnn_aqpso") & (val["tag"] != "")]
    if g.empty:
        return
    plt = _plt()
    s = g.groupby(["anchor_w0", "anchor_frac"]).agg(p=("passed", "mean"), r=("gated", "mean")).reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for f, gg in s.groupby("anchor_frac"):
        ax[0].plot(gg["anchor_w0"], 100 * gg["p"], marker="o", label=f"frac={f:g}")
        ax[1].plot(gg["anchor_w0"], gg["r"], marker="o", label=f"frac={f:g}")
    for a, yl in zip(ax, ("Pass (%)", "Gated rate")):
        a.set_xlabel("anchor_w0")
        a.set_ylabel(yl)
    ax[1].legend(fontsize=7)
    fig.suptitle("GNN-anchored QPSO: validation grid (anchor_w0 x anchor_frac)")
    _save(fig, fig_dir, "fig_qpso_anchor_grid")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
CSV_ONLY = {"table_anytime.csv"}   # long tables: CSV + tables.md only, not printed to console


def write_tables(out_dir: Path, tables: dict[str, pd.DataFrame], titles: dict[str, str], quiet: bool = False) -> None:
    tdir = out_dir / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    md = []
    for fname, df in tables.items():
        if df is None or df.empty:
            continue
        df.to_csv(tdir / fname, index=False)
        txt = f"## {titles.get(fname, fname)}\n\n`{fname}`\n\n{to_markdown(df)}\n"
        md.append(txt)
        if not quiet and fname not in CSV_ONLY:
            print(f"\n=== {titles.get(fname, fname)} ({fname}) ===")
            with pd.option_context("display.max_columns", 30, "display.width", 200, "display.precision", 4):
                print(df.to_string(index=False))
    (tdir / "tables.md").write_text("\n".join(md), encoding="utf-8")


def make_all(out_dir: Path, sys_: SysConfig, exp: ExpConfig, test: pd.DataFrame | None,
             val: pd.DataFrame | None = None, labels_dir: Path | None = None, models_dir: Path | None = None,
             cases: pd.DataFrame | None = None, ref: str = "gnn_pso", pairs: list[tuple[str, str]] | None = None,
             plots: bool = True, optimizer_name: str = "PSO", quiet: bool = False, log=None) -> None:
    T, titles = {}, {}

    def put(fname, df, title):
        T[fname] = df
        titles[fname] = title

    put("table1_params.csv", table1_params(sys_), "Table I: physical and operating parameters")
    put("table6_network.csv", table6_network(), "Table VI: main-network dense layers")
    put("table7_checkpoints.csv", table7_checkpoints(models_dir), "Table VII: frozen checkpoints")
    vb = pd.DataFrame()
    if val is not None and not val.empty:
        vb = val_budget_table(val)
        put("table_val_budget.csv", vb, "Validation budget sweep (Fig. 3 data)")
        put("table8_adjustment.csv", table8_adjust(val), "Table VIII: raw vs adjusted predictions (validation)")
    if test is not None and not test.empty:
        test = add_kgroup(test, exp)
        lab = kgroup_label(exp)
        put("table2_trainedK.csv", summary_table(test[test["Kgroup"] == lab]), f"Table II: held-out results, K={lab}")
        unseen = test[test["Kgroup"] != lab]
        if not unseen.empty:
            put("table3_unseenK.csv", summary_table(unseen), "Table III: held-out results, unseen K")
        if pairs is None:
            pairs = [(ref, m) for m in methods_in(test) if m != ref and not m.startswith("direct")]
        put("table4_paired.csv", paired_table(test, exp, pairs), f"Table IV: paired differences (A - B), ref={ref}")
        put("table5_gamma_dep.csv", gamma_dep_table(test, exp, sys_, ref), f"Table V: Gamma - empirical DEP ({name(ref)})")
        cvt = cv_group_table(test, exp, sys_)
        put("table_cv_groups.csv", cvt, "Willie-CV groups (Fig. 6 data)")
        cot = check_outcome_table(test)
        put("table_check_outcomes.csv", cot, "Check outcomes (Fig. 7 data)")
        put("table_diagnostics.csv", diagnostics_table(test, labels_dir), "Diagnostics")
        at = anytime_curves(test, exp.swarm) if "hist_rb" in test.columns else pd.DataFrame()
        if not at.empty:
            put("table_anytime.csv", at, "Anytime curves (search bank)")
            fk = "full_pso" if "full_pso" in set(at["key"]) else "full_qpso"
            put("table_anytime_target.csv", anytime_target(at, fk),
                f"Evaluations to reach 99% of {name(fk)} final search objective")
    write_tables(out_dir, T, titles, quiet=quiet)
    if log:
        log.info(f"tables written: {sorted(k for k, v in T.items() if v is not None and not v.empty)}")
    if not plots:
        return
    fig_dir = out_dir / "figures"
    fig1_layout(sys_, exp, cases, fig_dir)
    fig2_pipeline(fig_dir, optimizer_name)
    if not vb.empty:
        fig3_val_budget(vb, fig_dir)
    if val is not None and "anchor_w0" in val.columns:
        fig_anchor_grid(val, fig_dir)
    if test is not None and not test.empty:
        fig4_vs_K(test, exp, fig_dir)
        fig5_rate_vs_time(test, exp, fig_dir)
        fig6_cv(T.get("table_cv_groups.csv", pd.DataFrame()), fig_dir)
        fig7_outcomes(T.get("table_check_outcomes.csv", pd.DataFrame()), fig_dir)
        fig_paired(T.get("table4_paired.csv", pd.DataFrame()), fig_dir)
        if "table_anytime.csv" in T:
            full = "full_qpso" if "full_qpso" in set(test["method"]) and "full_pso" not in set(test["method"]) else "full_pso"
            fig_anytime(T["table_anytime.csv"], fig_dir, full)
    if log:
        log.info(f"figures written to {fig_dir}")
