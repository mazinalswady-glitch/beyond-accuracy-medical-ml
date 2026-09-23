"""E6 Pima imputation sensitivity: fold-internal median imputation (main analysis) vs.
multivariate iterative imputation (MICE-style, BayesianRidge) vs. dropping the two
variables with the highest missingness (insulin 48.7%, skin thickness 29.6%) with median
imputation of the rest. Same 5 x 10 repeated stratified CV partition as the main benchmark.

Output: results/pima_imputation.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
sys.path.insert(0, str(Path(__file__).parent))
from data import load
from common import MODELS, make_model, all_metrics

RES = Path(__file__).resolve().parents[1] / "results"


def _run(strategy, m, X, y, tr, te, rep, fold):
    if strategy == "median":
        mdl = make_model(m)
    elif strategy == "iterative":
        mdl = make_model(m, imputer=IterativeImputer(max_iter=25, random_state=0, sample_posterior=False))
    else:  # drop insulin & skin thickness
        X = X.drop(columns=["insulin", "skin_thickness"]); mdl = make_model(m)
    mdl.fit(X.iloc[tr], y[tr]); p = mdl.predict_proba(X.iloc[te])[:, 1]
    r = {"strategy": strategy, "model": m, "repeat": rep, "fold": fold}
    r.update(all_metrics(y[te], p, thr=y[tr].mean())); return r


def main():
    X, y, _ = load("PIMA")
    splits = list(RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42).split(X, y))
    rows = Parallel(n_jobs=2)(delayed(_run)(s, m, X, y, tr, te, i // 10, i % 10)
                              for s in ("median", "iterative", "drop")
                              for m in MODELS for i, (tr, te) in enumerate(splits))
    df = pd.DataFrame(rows); df.to_csv(RES / "pima_imputation.csv", index=False)
    print(df.groupby(["strategy", "model"])[["AUC", "MCC", "BRIER"]].mean().round(3).unstack(0))


if __name__ == "__main__":
    main()
