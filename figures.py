"""Figures 2-7 from the result files."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from common import MODELS, DATASETS, DS_NAMES

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"; FIG.mkdir(exist_ok=True)
TAB = RES / "tables"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
COL = {"LR": "#1f77b4", "RF": "#2ca02c", "XGB": "#d62728", "SVM": "#9467bd", "MLP": "#ff7f0e", "NB": "#8c564b"}


def fig_auc():
    perf = pd.read_csv(TAB / "performance.csv")
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.9), sharey=False)
    for ax, ds in zip(axes, DATASETS):
        d = perf[perf.dataset == ds].set_index("model").loc[MODELS]
        ax.bar(range(6), d.AUC * 100, yerr=d.AUC_sd * 100, color=[COL[m] for m in MODELS],
               capsize=3, edgecolor="k", lw=0.5)
        ax.set_xticks(range(6)); ax.set_xticklabels(MODELS, rotation=45)
        ax.set_ylim(40, 100); ax.axhline(50, color="grey", ls=":", lw=0.8)
        ax.set_title(DS_NAMES[ds], fontsize=10)
        for i, v in enumerate(d.AUC * 100):
            ax.text(i, 41.5, f"{v:.1f}", ha="center", fontsize=7, color="white", fontweight="bold")
    axes[0].set_ylabel("AUC-ROC (%)")
    fig.tight_layout(); fig.savefig(FIG / "fig2_auc.png", dpi=300)


def fig_calibration():
    """Reliability diagram: 10 quantile bins of predicted risk (first CV repeat, pooled OOF)."""
    z = np.load(RES / "oof_rep0.npz")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, ds in zip(axes, DATASETS):
        y = z[f"{ds}_y"]
        hi = 1.0 if ds != "D130" else 0.4
        for m in MODELS:
            p = z[f"{ds}_{m}"]
            edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
            b = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
            mp = [p[b == k].mean() for k in range(len(edges) - 1) if (b == k).any()]
            oy = [y[b == k].mean() for k in range(len(edges) - 1) if (b == k).any()]
            ax.plot(mp, oy, "-o", color=COL[m], lw=1.2, ms=2.5, label=m)
        ax.plot([0, hi], [0, hi], "k--", lw=0.8)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_title(DS_NAMES[ds], fontsize=10)
        ax.set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Observed event rate")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "fig3_calibration.png", dpi=300)


def fig_hier():
    hb = pd.read_csv(TAB / "hier_bayes.csv")
    hb = hb[hb.metric == "AUC"].copy()
    hb["label"] = hb.a + " vs " + hb.b
    hb = hb.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    y = np.arange(len(hb))
    ax.barh(y, hb.p_right, color="#2e7d32", label="P(first model better)")
    ax.barh(y, hb.p_rope, left=hb.p_right, color="#bdbdbd", label="P(practically equivalent, |ΔAUC| ≤ 0.01)")
    ax.barh(y, hb.p_left, left=hb.p_right + hb.p_rope, color="#c62828", label="P(second model better)")
    ax.set_yticks(y); ax.set_yticklabels(hb.label); ax.set_xlim(0, 1)
    ax.axvline(0.95, color="k", ls=":", lw=0.8); ax.axvline(0.05, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Posterior probability (hierarchical Bayesian correlated t-test, next dataset)")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "fig4_bayes.png", dpi=300)


def fig_cd():
    import json
    pooled = json.load(open(RES / "summary_bench.json"))["pooled"]["AUC"]
    R = pooled["ranks"]; cd = pooled["cd"]
    order = sorted(R, key=R.get)
    fig, ax = plt.subplots(figsize=(6.5, 2.1)); ax.set_xlim(0.7, 6.3); ax.set_ylim(0.15, 1.05); ax.axis("off")
    ax.hlines(0.8, 1, 6, color="k", lw=1)
    for k in range(1, 7):
        ax.vlines(k, 0.78, 0.84, color="k", lw=1); ax.text(k, 0.9, str(k), ha="center", fontsize=8)
    ax.hlines(0.97, 1, 1 + cd, color="k", lw=1.5); ax.text(1 + cd / 2, 0.99, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8)
    for i, m in enumerate(order):
        r = R[m]; y = 0.45 - 0.12 * (i % 3) if i < 3 else 0.45 - 0.12 * ((5 - i) % 3)
        left = i < 3
        xt = 0.8 if left else 6.2
        ax.plot([r, r, xt], [0.8, y, y], color=COL[m], lw=1)
        ax.text(xt + (-0.03 if left else 0.03), y, f"{m} ({r:.2f})", ha="right" if left else "left", va="center", fontsize=8)
    # cliques: groups whose rank range < CD
    cl = []
    rs = [R[m] for m in order]
    for i in range(6):
        j = max(k for k in range(6) if rs[k] - rs[i] < cd)
        if j > i and not any(a <= i and j <= b for a, b in cl):
            cl.append((i, j))
    for n, (i, j) in enumerate(cl):
        yy = 0.72 - 0.05 * n
        ax.hlines(yy, rs[i] - 0.03, rs[j] + 0.03, color="#1565c0", lw=3)
    fig.savefig(FIG / "fig5_cd.png", dpi=300, bbox_inches="tight")


def fig_shap():
    s = pd.read_csv(TAB / "shap_stability.csv")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3), sharey=True)
    series = [("seed", "feature", "tau", "τ, seed", "#90caf9"),
              ("boot", "feature", "tau", "τ, bootstrap", "#1565c0"),
              ("boot", "feature", "wtau", "weighted τ, bootstrap", "#ef6c00"),
              ("boot", "feature", "rbo", "RBO, bootstrap", "#6a1b9a"),
              ("boot", "group", "tau", "grouped τ, bootstrap", "#2e7d32")]
    w = 0.16
    for ax, ds in zip(axes, DATASETS):
        for k, (axis, level, met, lab, c) in enumerate(series):
            vals, sds, mins = [], [], []
            for m in ["LR", "RF", "XGB"]:
                r = s[(s.dataset == ds) & (s.model == m) & (s.axis == axis) & (s.level == level)].iloc[0]
                vals.append(r[f"{met}_mean"]); sds.append(r[f"{met}_sd"]); mins.append(r[f"{met}_min"])
            x = np.arange(3) + (k - 2) * w
            ax.bar(x, vals, w, yerr=sds, color=c, label=lab, capsize=1.5, error_kw={"lw": 0.6})
            ax.scatter(x, mins, marker="v", s=9, color="k", zorder=3)
        ax.set_xticks(range(3)); ax.set_xticklabels(["LR", "RF", "XGB"])
        ax.axhline(0.7, color="grey", ls="--", lw=0.7); ax.set_ylim(0, 1.05)
        ax.set_title(DS_NAMES[ds], fontsize=10)
    axes[0].set_ylabel("Pairwise agreement (30 runs)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(FIG / "fig6_shap_stability.png", dpi=300)


def fig_perm():
    p = pd.read_csv(TAB / "perm_stability.csv")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.1), sharey=True)
    des = [("model", "model seed only", "#1565c0"), ("perm", "permutation seed only", "#ef6c00"),
           ("both", "both vary", "#757575")]
    w = 0.26
    for ax, ds in zip(axes, DATASETS):
        for k, (d, lab, c) in enumerate(des):
            r = p[(p.dataset == ds) & (p.design == d)].set_index("model").loc[MODELS]
            x = np.arange(6) + (k - 1) * w
            ax.bar(x, r.tau_mean, w, yerr=r.tau_sd, color=c, label=lab, capsize=1.5, error_kw={"lw": 0.6})
            ax.scatter(x, r.tau_min, marker="v", s=8, color="k", zorder=3)
        ax.set_xticks(range(6)); ax.set_xticklabels(MODELS, rotation=45)
        ax.set_ylim(-0.15, 1.05); ax.axhline(0, color="k", lw=0.5)
        ax.set_title(DS_NAMES[ds], fontsize=10)
    axes[0].set_ylabel("Kendall τ of permutation importance")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(FIG / "fig7_permutation.png", dpi=300)


if __name__ == "__main__":
    which = sys.argv[1:] or ["auc", "cal", "hier", "cd", "shap", "perm"]
    fns = {"auc": fig_auc, "cal": fig_calibration, "hier": fig_hier, "cd": fig_cd, "shap": fig_shap, "perm": fig_perm}
    for w_ in which:
        fns[w_]()
