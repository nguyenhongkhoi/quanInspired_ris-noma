"""PSO-generated reference labels (claude.md 5.3, Section IV-C).

Per train/val case: 2 Full PSO runs -> archives (<= 5 each); candidates that are
search-feasible AND pass assessment on an independent selection bank compete on
selection-bank R_b; the winner is then assessed (record only) on another bank.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .assessment import assess_allocation
from .channel import bank_seed, build_bank, generate_layout
from .config import stable_seed
from .evaluator import Problem, to_original
from .features import case_features


def label_group(task: dict) -> dict:
    from baseline_code.pso import run_pso
    sys_, exp = task["sys"], task["exp"]
    lid, K, split = int(task["layout_id"]), int(task["K"]), task["split"]
    layout = generate_layout(sys_, exp.seed, lid, split, exp.K_max)
    bank_s = build_bank(sys_, layout, K, exp.search_samples, bank_seed(exp.seed, "search", lid, K))
    bank_sel = build_bank(sys_, layout, K, exp.label_samples, bank_seed(exp.seed, "select", lid, K))
    bank_la = build_bank(sys_, layout, K, exp.label_samples, bank_seed(exp.seed, "labelassess", lid, K))

    rows, nodes, globs, ids = [], [], [], []
    for case in task["cases"]:
        t0 = time.perf_counter()
        P_tot, xi = float(case["P_tot_dbm"]), float(case["xi_req"])
        prob = Problem(sys_, bank_s, P_tot, xi, "search")
        xn, xg = case_features(prob)
        nodes.append(xn)
        globs.append(xg)
        ids.append(case["case_id"])
        cands, best_feas = [], []
        for run in range(exp.label_runs):
            seed = stable_seed(exp.seed, "labelpso", case["case_id"], run)
            res = run_pso(prob, exp.budget_full, exp.swarm, seed, archive_size=exp.archive_size)
            cands.extend(res.archive)
            best_feas.append(res.best.feasible)
        prob_sel = Problem(sys_, bank_sel, P_tot, xi, "assess", sic_order=prob.sic_order, eligible=prob.eligible)
        best, best_a, n_feas, n_pass = None, None, 0, 0
        for c in cands:
            if not c.feasible:
                continue
            n_feas += 1
            a = assess_allocation(sys_, prob_sel, c.mb, c.s, c.m_sic)
            if a["passed"]:
                n_pass += 1
                if best is None or a["R_b"] > best_a["R_b"]:
                    best, best_a = c, a
        row = dict(group_key=task["group_key"], case_id=case["case_id"], layout_id=lid, split=split, K=K,
                   case_idx=case["case_idx"], P_tot_dbm=P_tot, xi_req=xi,
                   n_cands=len(cands), n_cands_feasible=n_feas, n_cands_pass=n_pass,
                   pso_best_feasible=any(best_feas), eligible=prob.eligible, cv_w=prob.cv_w,
                   has_label=best is not None,
                   # fixed column set (CSV header is taken from the first group written)
                   mb=None, s=None, m_sic=None, m_orig=None, R_b_sel=None, la_passed=None,
                   la_category=None, la_R_b=None, la_xi_emp=None, la_xi_L=None, la_R_L_min=None, la_p_bU=None)
        if best is not None:
            prob_la = Problem(sys_, bank_la, P_tot, xi, "assess", sic_order=prob.sic_order, eligible=prob.eligible)
            la = assess_allocation(sys_, prob_la, best.mb, best.s, best.m_sic)
            row.update(mb=best.mb, s=best.s,
                       m_sic=json.dumps([float(x) for x in best.m_sic]),
                       m_orig=json.dumps([float(x) for x in to_original(best.m_sic, prob.sic_order)]),
                       R_b_sel=best_a["R_b"], la_passed=la["passed"], la_category=la["category"],
                       la_R_b=la["R_b"], la_xi_emp=la["xi_emp"], la_xi_L=la["xi_L"],
                       la_R_L_min=la["R_L_min"], la_p_bU=la["p_bU"])
        row["time_s"] = time.perf_counter() - t0
        rows.append(row)
    feats = dict(case_ids=np.array(ids), node=np.stack(nodes), glob=np.stack(globs))
    return dict(rows=rows, feats=feats)


def save_group_feats(feat_dir: Path, group_key: str, feats: dict) -> None:
    feat_dir.mkdir(parents=True, exist_ok=True)
    np.savez(feat_dir / f"{group_key}.npz", **feats)


def load_label_set(labels_dir: Path, split: str) -> dict:
    """Labelled graphs of one split: lists of node/glob arrays and SIC-order targets."""
    import pandas as pd
    csv_path = labels_dir / f"labels_{split}.csv"
    df = pd.read_csv(csv_path)
    df = df[df["has_label"].astype(str) == "True"]
    feat_cache: dict = {}
    out = dict(case_id=[], node=[], glob=[], K=[], mb=[], s=[], m=[], layout_id=[])
    for _, r in df.iterrows():
        gk = r["group_key"]
        if gk not in feat_cache:
            d = np.load(labels_dir / "feat" / f"{gk}.npz", allow_pickle=False)
            feat_cache[gk] = {str(c): i for i, c in enumerate(d["case_ids"])}, d["node"], d["glob"]
        idx, node, glob = feat_cache[gk]
        i = idx[str(r["case_id"])]
        out["case_id"].append(r["case_id"])
        out["node"].append(node[i])
        out["glob"].append(glob[i])
        out["K"].append(int(r["K"]))
        out["mb"].append(float(r["mb"]))
        out["s"].append(float(r["s"]))
        out["m"].append(np.asarray(json.loads(r["m_sic"]), float))
        out["layout_id"].append(int(r["layout_id"]))
    return out


def assemble_labels_npz(labels_dir: Path) -> None:
    """Write labels/labels.npz (padded) for archival, from per-group feature files."""
    blocks = {}
    for split in ("train", "val"):
        p = labels_dir / f"labels_{split}.csv"
        if not p.exists():
            continue
        d = load_label_set(labels_dir, split)
        if not d["node"]:
            continue
        n = max(x.shape[0] for x in d["node"])
        B = len(d["node"])
        node = np.zeros((B, n, d["node"][0].shape[1]))
        m = np.zeros((B, n - 1))
        for b, (x, mm) in enumerate(zip(d["node"], d["m"])):
            node[b, :x.shape[0]] = x
            m[b, :mm.size] = mm
        blocks.update({f"{split}_case_id": np.array(d["case_id"]), f"{split}_node": node,
                       f"{split}_glob": np.stack(d["glob"]), f"{split}_K": np.array(d["K"]),
                       f"{split}_mb": np.array(d["mb"]), f"{split}_s": np.array(d["s"]), f"{split}_m": m})
    if blocks:
        np.savez(labels_dir / "labels.npz", **blocks)
