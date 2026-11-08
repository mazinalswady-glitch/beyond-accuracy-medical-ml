"""Revision analysis (reviewer comment 3): sensitivity of the threshold-dependent metrics
to the decision threshold.

Same outer 10-fold partition as the nested analysis (seed 7), default configurations.
Within every outer training set the threshold is chosen WITHOUT touching the test fold:
inner stratified CV (5-fold; 3-fold for Diabetes-130; seed 11) gives out-of-fold risks, and
the threshold maximising Youden's J = sensitivity + specificity - 1 is selected. The model is
refitted on the whole outer training set and the test fold is classified with that threshold.
For the SVM the decision function is used (Platt scaling is monotone, so the classification
is identical to thresholding the probability).
The prevalence-threshold results used in the main text are in nested_folds.csv (config = default).

Output: threshold_sensitivity_folds.csv
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_curve, accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef)
sys.path.insert(0, str(Path(__file__).parent))
from data import load
from common import MODELS, DATASETS, make_model

RES = Path(__file__).resolve().parent
rows = []
if (RES / "threshold_sensitivity_folds.csv").exists():   # resume after interruption
    rows = pd.read_csv(RES / "threshold_sensitivity_folds.csv").to_dict("records")
done = {(r["dataset"], r["model"]) for r in rows}
for ds in DATASETS:
    X, y, _ = load(ds)
    inner_k = 3 if ds == "D130" else 5
    outer = list(StratifiedKFold(10, shuffle=True, random_state=7).split(X, y))
    for m in MODELS:
        if (ds, m) in done:
            continue
        t0 = time.time()
        method = "decision_function" if m == "SVM" else "predict_proba"
        for f, (tr, te) in enumerate(outer):
            Xtr, ytr, Xte, yte = X.iloc[tr], y[tr], X.iloc[te], y[te]
            est = make_model(m, seed=0)
            if m == "SVM":
                est.set_params(clf__probability=False)
            oof = cross_val_predict(est, Xtr, ytr, method=method, n_jobs=1,
                                   cv=StratifiedKFold(inner_k, shuffle=True, random_state=11))
            oof = oof[:, 1] if oof.ndim == 2 else oof
            fpr, tpr, thr = roc_curve(ytr, oof)
            t = thr[np.argmax(tpr - fpr)]
            fit = est.fit(Xtr, ytr)
            s = fit.decision_function(Xte) if m == "SVM" else fit.predict_proba(Xte)[:, 1]
            yh = (s >= t).astype(int)
            rows.append({"dataset": ds, "model": m, "fold": f, "thr_youden": float(t),
                         "ACC": accuracy_score(yte, yh), "BAL": balanced_accuracy_score(yte, yh),
                         "F1": f1_score(yte, yh, zero_division=0), "MCC": matthews_corrcoef(yte, yh)})
        print(ds, m, f"{time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(RES / "threshold_sensitivity_folds.csv", index=False)
