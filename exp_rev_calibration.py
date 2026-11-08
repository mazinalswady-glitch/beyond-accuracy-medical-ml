"""Revision analysis (reviewer comment 2): calibration slope/intercept of the TUNED models.

The out-of-fold predictions of the nested cross-validation (outer 10-fold, seed 7) are
regenerated: for every outer fold the configuration selected by the inner grid search
(recorded in nested_folds.csv) is refitted on the outer training set with model seed 0.
The AUC/Brier of the regenerated predictions are checked against nested_folds.csv.
Calibration slope and intercept are estimated on the pooled out-of-fold predictions
(identical definition to Table 4), with a patient-level bootstrap 95% interval.

Output: oof_nested.npz, tbl_nested_calibration.csv
"""
import ast, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
sys.path.insert(0, str(Path(__file__).parent))
from data import load
from common import MODELS, DATASETS, make_model, calibration_slope_intercept

RES = Path(__file__).resolve().parent
nf = pd.read_csv(RES / "nested_folds.csv")
rows, store = [], {}
if (RES / "tbl_nested_calibration.csv").exists():   # resume after interruption
    rows = pd.read_csv(RES / "tbl_nested_calibration.csv").to_dict("records")
    store = dict(np.load(RES / "oof_nested.npz"))
done = {(r["dataset"], r["model"]) for r in rows}
rng = np.random.RandomState(2026)
for ds in DATASETS:
    X, y, _ = load(ds)
    outer = list(StratifiedKFold(10, shuffle=True, random_state=7).split(X, y))
    for m in MODELS:
        if (ds, m) in done:
            continue
        t0 = time.time()
        P = {"default": np.zeros(len(y)), "tuned": np.zeros(len(y))}
        chk = {"default": [], "tuned": []}
        for f, (tr, te) in enumerate(outer):
            bp = ast.literal_eval(nf[(nf.dataset == ds) & (nf.model == m) & (nf.config == "tuned") & (nf.fold == f)].best_params.iloc[0])
            for cfg, kw in (("default", {}), ("tuned", bp)):
                mdl = make_model(m, seed=0).set_params(**kw).fit(X.iloc[tr], y[tr])
                p = mdl.predict_proba(X.iloc[te])[:, 1]
                P[cfg][te] = p
                ref = nf[(nf.dataset == ds) & (nf.model == m) & (nf.config == cfg) & (nf.fold == f)].iloc[0]
                chk[cfg].append((abs(roc_auc_score(y[te], p) - ref.AUC), abs(brier_score_loss(y[te], p) - ref.BRIER)))
        for cfg in ("default", "tuned"):
            p = P[cfg]; store[f"{ds}_{m}_{cfg}"] = p
            s, i = calibration_slope_intercept(y, p)
            bs = []
            for _ in range(300):
                ix = rng.randint(0, len(y), len(y))
                bs.append(calibration_slope_intercept(y[ix], p[ix]))
            bs = np.array(bs)
            rows.append({"dataset": ds, "model": m, "config": cfg, "slope": s, "intercept": i,
                         "slope_lo": np.nanpercentile(bs[:, 0], 2.5), "slope_hi": np.nanpercentile(bs[:, 0], 97.5),
                         "int_lo": np.nanpercentile(bs[:, 1], 2.5), "int_hi": np.nanpercentile(bs[:, 1], 97.5),
                         "brier": brier_score_loss(y, p), "auc_pooled": roc_auc_score(y, p),
                         "max_abs_dev_AUC_vs_recorded": max(c[0] for c in chk[cfg]),
                         "max_abs_dev_Brier_vs_recorded": max(c[1] for c in chk[cfg])})
        store[f"{ds}_y"] = y
        print(ds, m, f"{time.time()-t0:.0f}s", [round(r["slope"], 2) for r in rows[-2:]], flush=True)
        pd.DataFrame(rows).to_csv(RES / "tbl_nested_calibration.csv", index=False)
        np.savez_compressed(RES / "oof_nested.npz", **store)
