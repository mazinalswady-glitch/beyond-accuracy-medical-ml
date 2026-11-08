"""Revision analyses (reviewer comments 1 and 4): SHAP stability for TUNED configurations and for a
HELD-OUT evaluation set.

usage: python exp_rev_shap.py <config> <eval> [datasets...]
  config = default | tuned   (tuned = configuration selected most often by the nested CV, tbl_nested_summary.csv)
  eval   = insample | heldout
insample: identical protocol to exp_shap.py (evaluation set = full data; D130 = 300-patient sample, seed 5);
          axes = bootstrap for all models, plus seed axis for Random Forest.
heldout : stratified 30% held-out split (seed 3) is removed first; bootstrap resamples (seeds 1000..1029) are drawn from the
          remaining 70% only, and global mean |SHAP| is computed on the held-out patients (D130: stratified 300 of them, seed 5),
          so that no evaluation patient was seen by any retrained model.
Output: shap_stability_<config>_<eval>.csv, shap_importances_<config>_<eval>.npz
"""
import ast, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0, str(Path(__file__).parent))
from data import load
from common import DATASETS, make_model, pairwise_stability, correlation_groups
from exp_shap import shap_matrix

RES = Path(__file__).resolve().parent
R = 30
config, evalmode = sys.argv[1], sys.argv[2]
dsets = sys.argv[3:] or DATASETS
SHAP_MODELS = ["LR", "XGB", "RF"]
summ = pd.read_csv(RES / "tbl_nested_summary.csv")


def kw_for(ds, m):
    if config == "default":
        return {}
    return ast.literal_eval(summ[(summ.dataset == ds) & (summ.model == m)].best_params_mode.iloc[0])


def one(m, X, y, pool, Xev, G, axis, r, kw):
    if axis == "seed":
        idx, seed = pool, r
    else:
        idx, seed = pool[np.random.RandomState(1000 + r).randint(0, len(pool), len(pool))], 0
    Xtr, ytr = X.iloc[idx], y[idx]
    pipe = make_model(m, seed=seed).set_params(**kw).fit(Xtr, ytr)
    S = shap_matrix(m, pipe, Xtr, Xev)
    return np.abs(S).mean(0), np.abs(S @ G).mean(0)


tag = f"{config}_{evalmode}" + ("" if list(dsets) == DATASETS else "_" + "_".join(dsets))
rows, store = [], {}
_csv, _npz = RES / f"shap_stability_{tag}.csv", RES / f"shap_importances_{tag}.npz"
if _csv.exists() and _npz.exists():   # resume after interruption
    rows = pd.read_csv(_csv).to_dict("records"); store = dict(np.load(_npz, allow_pickle=True))
done = {(r["dataset"], r["model"], r["axis"]) for r in rows}
for ds in dsets:
    X, y, meta = load(ds)
    allidx = np.arange(len(y))
    if evalmode == "heldout":
        pool, ho = train_test_split(allidx, test_size=0.3, stratify=y, random_state=3)
        ev = train_test_split(ho, train_size=300, stratify=y[ho], random_state=5)[0] if ds == "D130" else ho
    else:
        pool = allidx
        ev = train_test_split(allidx, train_size=300, stratify=y, random_state=5)[0] if ds == "D130" else allidx
    Xev = X.iloc[ev]
    groups = correlation_groups(X, thr=0.8, base_groups=meta["groups"])
    gnames = list(dict.fromkeys(groups))
    G = np.array([[1.0 if g == gn else 0.0 for gn in gnames] for g in groups])
    store[f"{ds}_features"] = np.array(X.columns); store[f"{ds}_groups"] = np.array(gnames)
    for m in SHAP_MODELS:
        kw = kw_for(ds, m)
        axes = ["boot"] + (["seed"] if (evalmode == "insample" and m == "RF") else [])
        for axis in axes:
            if (ds, m, axis) in done:
                continue
            t0 = time.time()
            out = [one(m, X, y, pool, Xev, G, axis, r, kw) for r in range(R)]
            imp, gimp = np.array([o[0] for o in out]), np.array([o[1] for o in out])
            store[f"{ds}_{m}_{axis}"] = imp; store[f"{ds}_{m}_{axis}_grp"] = gimp
            for level, M in (("feature", imp), ("group", gimp)):
                r_ = {"config": config, "eval": evalmode, "dataset": ds, "model": m, "axis": axis, "level": level,
                      "n_units": M.shape[1], "params": str(kw)}
                r_.update(pairwise_stability(M)); rows.append(r_)
            print(ds, m, axis, config, evalmode, f"{time.time()-t0:.0f}s", round(rows[-2]["tau_mean"], 3), round(rows[-1]["tau_mean"], 3), flush=True)
            pd.DataFrame(rows).to_csv(RES / f"shap_stability_{tag}.csv", index=False)
            np.savez(RES / f"shap_importances_{tag}.npz", **store)
