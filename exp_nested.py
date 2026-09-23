"""E3 nested cross-validation: outer 10-fold stratified CV (identical partition for all
models), inner stratified grid search optimising AUC. The default configuration is
evaluated on the same outer folds so that default and tuned results are paired and
the significance tests can be repeated on the tuned models.

Output: results/nested_folds.csv
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.base import clone
sys.path.insert(0, str(Path(__file__).parent))
from data import load
from common import MODELS, DATASETS, make_model, all_metrics

RES = Path(__file__).resolve().parents[1] / "results"
OUTER_SEED, INNER_SEED = 7, 11

GRIDS = {
    "LR": {"clf__C": [0.01, 0.1, 1, 10, 100]},
    "RF": {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 5, 10], "clf__min_samples_leaf": [1, 5]},
    "XGB": {"clf__n_estimators": [100, 300], "clf__max_depth": [2, 3, 6], "clf__learning_rate": [0.03, 0.1, 0.3]},
    "SVM": {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.1]},
    "MLP": {"clf__hidden_layer_sizes": [(64, 32), (32,), (16,)], "clf__alpha": [1e-4, 1e-2, 1.0]},
    "NB": {"clf__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1]},
}


def tuned_fit(m, X, y, inner_k):
    est = make_model(m, seed=0)
    if m == "SVM":  # tune on the decision function (same AUC, avoids inner Platt CV); refit with probabilities
        est.set_params(clf__probability=False)
    gs = GridSearchCV(est, GRIDS[m], scoring="roc_auc", n_jobs=2, refit=False,
                      cv=StratifiedKFold(inner_k, shuffle=True, random_state=INNER_SEED))
    gs.fit(X, y)
    final = make_model(m, seed=0).set_params(**gs.best_params_)
    return final.fit(X, y), gs.best_params_


def main():
    rows = []
    for ds in DATASETS:
        X, y, _ = load(ds)
        inner_k = 3 if ds == "D130" else 5
        outer = list(StratifiedKFold(10, shuffle=True, random_state=OUTER_SEED).split(X, y))
        for m in MODELS:
            t0 = time.time()
            for f, (tr, te) in enumerate(outer):
                Xtr, ytr, Xte, yte = X.iloc[tr], y[tr], X.iloc[te], y[te]
                pd_ = make_model(m, seed=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
                mdl, bp = tuned_fit(m, Xtr, ytr, inner_k)
                pt = mdl.predict_proba(Xte)[:, 1]
                for cfg, p in (("default", pd_), ("tuned", pt)):
                    r = {"dataset": ds, "model": m, "fold": f, "config": cfg,
                         "n_train": len(tr), "n_test": len(te),
                         "best_params": str(bp) if cfg == "tuned" else ""}
                    r.update(all_metrics(yte, p, thr=ytr.mean())); rows.append(r)
            print(ds, m, f"{time.time()-t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(RES / "nested_folds.csv", index=False)


if __name__ == "__main__":
    main()
