"""E4 two-axis SHAP explanation-stability profile.

Axis 1 (seed):      30 retrainings on the full data, model seed = 0..29, data fixed.
Axis 2 (bootstrap): 30 retrainings on bootstrap resamples (seed 1000+r), model seed fixed at 0.
Global importance = mean |SHAP| over a FIXED evaluation set (the full dataset; for D130 a
fixed stratified subsample of 300 patients, because
path-dependent TreeSHAP on fully grown Random Forest trees (depth ~33, ~1,400 leaves) costs ~0.2 s per patient), so that differences between runs come only
from the model. Grouped importance = mean |sum of member SHAP values| per feature group.

SHAP: TreeSHAP (path-dependent, no background) for RF (shap.TreeExplainer) and XGBoost
(native pred_contribs); exact linear SHAP for LR with the (standardised, imputed) training
sample of each run as background (interventional/independent masker), i.e.
phi_ij = w_j * (z_ij - mean_train(z_j)).

Output: results/shap_importances.npz, results/shap_stability.csv
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import shap, xgboost as xgb
from sklearn.model_selection import train_test_split
sys.path.insert(0, str(Path(__file__).parent))
from data import load
from common import DATASETS, make_model, pairwise_stability, correlation_groups

RES = Path(__file__).resolve().parents[1] / "results"
R = 30
SHAP_MODELS = ["LR", "RF", "XGB"]


def shap_matrix(m, pipe, Xtr, Xev):
    Ztr = pipe[:-1].transform(Xtr) if len(pipe) > 1 else Xtr.values
    Zev = pipe[:-1].transform(Xev) if len(pipe) > 1 else Xev.values
    clf = pipe[-1]
    if m == "LR":
        return clf.coef_[0] * (Zev - Ztr.mean(0))
    if m == "RF":
        sv = shap.TreeExplainer(clf).shap_values(Zev, check_additivity=False)
        sv = np.asarray(sv)
        return sv[..., 1] if sv.ndim == 3 else sv
    if m == "XGB":
        c = clf.get_booster().predict(xgb.DMatrix(Zev), pred_contribs=True)
        return c[:, :-1]
    raise ValueError(m)


def _one(m, X, y, Xev, G, axis, r):
    if axis == "seed":
        idx, seed = np.arange(len(y)), r
    else:
        idx, seed = np.random.RandomState(1000 + r).randint(0, len(y), len(y)), 0
    Xtr, ytr = X.iloc[idx], y[idx]
    pipe = make_model(m, seed=seed).fit(Xtr, ytr)
    S = shap_matrix(m, pipe, Xtr, Xev)
    return np.abs(S).mean(0), np.abs(S @ G).mean(0)


def main(datasets=DATASETS):
    from joblib import Parallel, delayed
    tag = "" if list(datasets) == DATASETS else "_" + "_".join(datasets)
    rows, store = [], {}
    for ds in datasets:
        X, y, meta = load(ds)
        if ds == "D130":
            ev_idx, _ = train_test_split(np.arange(len(y)), train_size=300, stratify=y, random_state=5)
        else:
            ev_idx = np.arange(len(y))
        Xev = X.iloc[ev_idx]
        groups = correlation_groups(X, thr=0.8, base_groups=meta["groups"])
        gnames = list(dict.fromkeys(groups))
        G = np.array([[1.0 if g == gn else 0.0 for gn in gnames] for g in groups])  # F x G
        store[f"{ds}_features"] = np.array(X.columns); store[f"{ds}_groups"] = np.array(gnames)
        for m in SHAP_MODELS:
            for axis in ("seed", "boot"):
                t0 = time.time()
                out = Parallel(n_jobs=2)(delayed(_one)(m, X, y, Xev, G, axis, r) for r in range(R))
                imp, gimp = np.array([o[0] for o in out]), np.array([o[1] for o in out])
                store[f"{ds}_{m}_{axis}"] = imp; store[f"{ds}_{m}_{axis}_grp"] = gimp
                for level, M in (("feature", imp), ("group", gimp)):
                    r_ = {"dataset": ds, "model": m, "axis": axis, "level": level, "n_units": M.shape[1]}
                    r_.update(pairwise_stability(M)); rows.append(r_)
                print(ds, m, axis, f"{time.time()-t0:.0f}s",
                      round(rows[-2]["tau_mean"], 3), round(rows[-2]["wtau_mean"], 3), round(rows[-1]["tau_mean"], 3), flush=True)
            pd.DataFrame(rows).to_csv(RES / f"shap_stability{tag}.csv", index=False)
            np.savez(RES / f"shap_importances{tag}.npz", **store)


def collinearity():
    """Variance inflation factors of the standardised design (LR multicollinearity)."""
    from numpy.linalg import cond, inv
    rows = []
    for ds in DATASETS:
        X, y, _ = load(ds)
        Z = make_model("LR").fit(X, y)[:-1].transform(X)
        C = np.corrcoef(Z, rowvar=False)
        vif = np.diag(inv(C))
        rows.append({"dataset": ds, "n_features": Z.shape[1], "cond_number": float(cond(Z)),
                     "max_vif": float(vif.max()), "n_vif_gt10": int((vif > 10).sum()),
                     "median_vif": float(np.median(vif))})
    pd.DataFrame(rows).to_csv(RES / "collinearity.csv", index=False)
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    ds = sys.argv[1:] or DATASETS
    if list(ds) == DATASETS:
        collinearity()
    main(ds)
