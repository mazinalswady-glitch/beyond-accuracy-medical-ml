"""Statistical analysis and table generation from the raw experiment outputs.

Usage: python src/analysis.py [bench|nested|shap|perm|pima|all]
"""
import sys, json, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).parent))
from common import MODELS, DATASETS
from stats_tests import (holm, corrected_resampled_t, bayes_correlated_t, friedman_nemenyi,
                         hierarchical_bayes)

RES = Path(__file__).resolve().parent
TAB = RES  # flat layout: summary tables are saved as tbl_*.csv
PAIRS = list(itertools.combinations(MODELS, 2))
# metric: (higher_is_better, region of practical equivalence)
METRICS = {"AUC": (True, 0.01), "MCC": (True, 0.02), "BAL": (True, 0.01), "BRIER": (False, 0.005)}
RHO = 0.1  # n_test / n for 10-fold CV


def fmt(m, s, d=1, pct=True):
    k = 100 if pct else 1
    return f"{m*k:.{d}f} ± {s*k:.{d}f}"


def pair_diffs(df, a, b, metric, keys):
    A = df[df.model == a].sort_values(keys)[metric].values
    B = df[df.model == b].sort_values(keys)[metric].values
    hib, _ = METRICS[metric]
    return (A - B) if hib else (B - A)  # positive = a better


def bench():
    df = pd.read_csv(RES / "benchmark_folds.csv")
    cal = pd.read_csv(RES / "calibration.csv")
    out = {}
    # ---- Table 2: performance
    rows = []
    for ds in DATASETS:
        for m in MODELS:
            s = df[(df.dataset == ds) & (df.model == m)]
            c = cal[(cal.dataset == ds) & (cal.model == m)]
            rows.append({"dataset": ds, "model": m,
                         **{k: s[k].mean() for k in ["ACC", "BAL", "F1", "AUC", "MCC", "BRIER"]},
                         **{k + "_sd": s[k].std() for k in ["ACC", "BAL", "F1", "AUC", "MCC", "BRIER"]},
                         "slope": c.slope.mean(), "slope_sd": c.slope.std(),
                         "intercept": c.intercept.mean(), "intercept_sd": c.intercept.std()})
    perf = pd.DataFrame(rows); perf.to_csv(TAB / "tbl_performance.csv", index=False)
    # ---- per-dataset dependence-aware tests
    rows = []
    for metric in METRICS:
        rope = METRICS[metric][1]
        for ds in DATASETS:
            d = df[df.dataset == ds]
            ratio = d.n_test.iloc[0] / d.n_train.iloc[0]
            block = []
            for a, b in PAIRS:
                x = pair_diffs(d, a, b, metric, ["repeat", "fold"])
                t, p = corrected_resampled_t(x, ratio)
                pl, pr, pg = bayes_correlated_t(x, RHO, rope)
                block.append({"metric": metric, "dataset": ds, "a": a, "b": b, "mean_diff": x.mean(),
                              "t": t, "p": p, "P_b_better": pl, "P_rope": pr, "P_a_better": pg})
            ph = holm([r["p"] for r in block])
            for r, h in zip(block, ph):
                r["p_holm"] = h
            rows += block
    sig = pd.DataFrame(rows); sig.to_csv(TAB / "tbl_sig_perdataset.csv", index=False)
    # ---- pooled descriptive analysis (40 blocks = repeat 0 x 10 folds x 4 datasets)
    pooled = {}
    for metric in METRICS:
        hib = METRICS[metric][0]
        d0 = df[df.repeat == 0]
        M = np.column_stack([d0[d0.model == m].sort_values(["dataset", "fold"])[metric].values for m in MODELS])
        if not hib:
            M = -M
        chi2, p, R, cd = friedman_nemenyi(M)
        wil = []
        for i, j in itertools.combinations(range(6), 2):
            pw = stats.wilcoxon(M[:, i], M[:, j], zero_method="zsplit").pvalue
            wil.append((MODELS[i], MODELS[j], pw))
        ph = holm([w[2] for w in wil])
        # dataset-level Friedman (N = 4 independent blocks)
        Md = np.column_stack([df[df.model == m].groupby("dataset")[metric].mean().loc[DATASETS].values for m in MODELS])
        if not hib:
            Md = -Md
        chi2d, pd_, Rd, cdd = friedman_nemenyi(Md)
        pooled[metric] = {"chi2": chi2, "p": p, "ranks": dict(zip(MODELS, R.round(3))), "cd": cd,
                          "wilcoxon": [(a, b, pw, h) for (a, b, pw), h in zip(wil, ph)],
                          "ds_level": {"chi2": chi2d, "p": pd_, "ranks": dict(zip(MODELS, Rd.round(3))), "cd": cdd}}
    out["pooled"] = pooled
    # ---- Bayesian hierarchical (Benavoli et al., 2017)
    hb = []
    for metric in ["AUC", "MCC", "BRIER"]:
        rope = METRICS[metric][1]
        for a, b in PAIRS:
            diffs = [pair_diffs(df[df.dataset == ds], a, b, metric, ["repeat", "fold"]) for ds in DATASETS]
            r = hierarchical_bayes(diffs, RHO, rope)
            hb.append({"metric": metric, "a": a, "b": b, **r})
            print("HB", metric, a, b, {k: round(v, 3) for k, v in r.items()}, flush=True)
    pd.DataFrame(hb).to_csv(TAB / "tbl_hier_bayes.csv", index=False)
    json.dump(out, open(RES / "summary_bench.json", "w"), indent=1, default=float)


def nested():
    df = pd.read_csv(RES / "nested_folds.csv")
    rows = []
    for ds in DATASETS:
        for m in MODELS:
            s = df[(df.dataset == ds) & (df.model == m)]
            dd = s[s.config == "default"].sort_values("fold"); tt = s[s.config == "tuned"].sort_values("fold")
            rows.append({"dataset": ds, "model": m, "AUC_default": dd.AUC.mean(), "AUC_default_sd": dd.AUC.std(),
                         "AUC_tuned": tt.AUC.mean(), "AUC_tuned_sd": tt.AUC.std(),
                         "MCC_tuned": tt.MCC.mean(), "BRIER_tuned": tt.BRIER.mean(),
                         "BRIER_default": dd.BRIER.mean(),
                         "best_params_mode": tt.best_params.mode().iloc[0]})
    pd.DataFrame(rows).to_csv(TAB / "tbl_nested_summary.csv", index=False)
    sig = []
    for cfg in ("default", "tuned"):
        for metric in ("AUC", "MCC", "BRIER"):
            rope = METRICS[metric][1]
            for ds in DATASETS:
                d = df[(df.dataset == ds) & (df.config == cfg)]
                ratio = d.n_test.iloc[0] / d.n_train.iloc[0]
                block = []
                for a, b in PAIRS:
                    x = pair_diffs(d, a, b, metric, ["fold"])
                    t, p = corrected_resampled_t(x, ratio)
                    pl, pr, pg = bayes_correlated_t(x, RHO, rope)
                    block.append({"config": cfg, "metric": metric, "dataset": ds, "a": a, "b": b,
                                  "mean_diff": x.mean(), "t": t, "p": p,
                                  "P_b_better": pl, "P_rope": pr, "P_a_better": pg})
                for r, h in zip(block, holm([r["p"] for r in block])):
                    r["p_holm"] = h
                sig += block
    pd.DataFrame(sig).to_csv(TAB / "tbl_nested_sig.csv", index=False)
    hb = []
    for cfg in ("default", "tuned"):
        for a, b in PAIRS:
            diffs = [pair_diffs(df[(df.dataset == ds) & (df.config == cfg)], a, b, "AUC", ["fold"]) for ds in DATASETS]
            r = hierarchical_bayes(diffs, RHO, 0.01)
            hb.append({"config": cfg, "a": a, "b": b, **r})
            print("HBn", cfg, a, b, {k: round(v, 3) for k, v in r.items()}, flush=True)
    pd.DataFrame(hb).to_csv(TAB / "tbl_nested_hier_bayes.csv", index=False)
    # Friedman on tuned, pooled 40 blocks (descriptive)
    res = {}
    for cfg in ("default", "tuned"):
        d = df[df.config == cfg]
        M = np.column_stack([d[d.model == m].sort_values(["dataset", "fold"]).AUC.values for m in MODELS])
        chi2, p, R, cd = friedman_nemenyi(M)
        res[cfg] = {"chi2": chi2, "p": p, "ranks": dict(zip(MODELS, R.round(3))), "cd": cd}
    json.dump(res, open(RES / "summary_nested.json", "w"), indent=1, default=float)


def shap_tables():
    s = pd.read_csv(RES / "shap_stability.csv")
    s.to_csv(TAB / "tbl_shap_stability.csv", index=False)
    z = np.load(RES / "shap_importances.npz", allow_pickle=True)
    freq = []
    for ds in DATASETS:
        for level, key, names in (("feature", "", z[f"{ds}_features"]), ("group", "_grp", z[f"{ds}_groups"])):
            for m in ["LR", "RF", "XGB"]:
                imp = z[f"{ds}_{m}_boot{key}"]
                top = np.argsort(-imp, 1)[:, :5]
                cnt = pd.Series(names[top.ravel()]).value_counts()
                for f, c in cnt.items():
                    freq.append({"dataset": ds, "level": level, "model": m, "unit": f, "top5_count": int(c)})
    pd.DataFrame(freq).to_csv(TAB / "tbl_shap_top5_freq.csv", index=False)


def perm_tables():
    p = pd.read_csv(RES / "perm_stability.csv"); p.to_csv(TAB / "tbl_perm_stability.csv", index=False)


def pima():
    df = pd.read_csv(RES / "pima_imputation.csv")
    t = df.groupby(["model", "strategy"])[["AUC", "MCC", "BRIER"]].mean().unstack(1)
    t.to_csv(TAB / "tbl_pima_imputation.csv")
    # does the model ranking change?
    r = {}
    for s in df.strategy.unique():
        r[s] = df[df.strategy == s].groupby("model").AUC.mean().rank(ascending=False).to_dict()
    json.dump(r, open(RES / "summary_pima.json", "w"), indent=1)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in [("bench", bench), ("nested", nested), ("shap", shap_tables),
                     ("perm", perm_tables), ("pima", pima)]:
        if what in ("all", name):
            fn()
