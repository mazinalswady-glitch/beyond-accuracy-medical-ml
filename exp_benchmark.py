"""E0 determinism check + E1 benchmark (5 x 10-fold repeated stratified CV).

Outputs
  results/determinism.csv      max |delta p| across 5 re-seeded trainings on the full data
  results/benchmark_folds.csv  per-fold metrics (identical partitions for all models)
  results/calibration.csv      calibration slope / intercept per repeat (pooled OOF)
  results/oof_rep0.npz         out-of-fold probabilities of the first repeat (calibration plots)
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold
sys.path.insert(0, str(Path(__file__).parent))
from data import load
from common import MODELS, DATASETS, make_model, all_metrics, calibration_slope_intercept

RES = Path(__file__).resolve().parent; RES.mkdir(exist_ok=True)
CV_SEED, N_SPLITS, N_REPEATS = 42, 10, 5


def determinism():
    rows = []
    for ds in DATASETS:
        X, y, _ = load(ds)
        for m in MODELS:
            P = np.array([make_model(m, seed=s).fit(X, y).predict_proba(X)[:, 1] for s in range(5)])
            rows.append({"dataset": ds, "model": m, "max_abs_dp": float(np.abs(P - P[0]).max())})
            print(ds, m, rows[-1]["max_abs_dp"], flush=True)
    pd.DataFrame(rows).to_csv(RES / "determinism.csv", index=False)


def _fold(ds, X, y, m, rep, fold, tr, te):
    mdl = make_model(m, seed=0).fit(X.iloc[tr], y[tr])
    p = mdl.predict_proba(X.iloc[te])[:, 1]
    r = {"dataset": ds, "model": m, "repeat": rep, "fold": fold, "n_train": len(tr), "n_test": len(te)}
    r.update(all_metrics(y[te], p, thr=y[tr].mean()))
    return r, te, p


def benchmark():
    rows, calib, oof = [], [], {}
    for ds in DATASETS:
        X, y, _ = load(ds)
        cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=CV_SEED)
        splits = list(cv.split(X, y))
        for m in MODELS:
            t0 = time.time()
            out = Parallel(n_jobs=2)(delayed(_fold)(ds, X, y, m, i // N_SPLITS, i % N_SPLITS, tr, te)
                                     for i, (tr, te) in enumerate(splits))
            P = np.zeros((N_REPEATS, len(y)))
            for r, te, p in out:
                rows.append(r); P[r["repeat"], te] = p
            for rep in range(N_REPEATS):
                s, c = calibration_slope_intercept(y, P[rep])
                calib.append({"dataset": ds, "model": m, "repeat": rep, "slope": s, "intercept": c})
            oof[f"{ds}_{m}"] = P[0]; oof[f"{ds}_y"] = y
            print(ds, m, f"{time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(RES / "benchmark_folds.csv", index=False)
        pd.DataFrame(calib).to_csv(RES / "calibration.csv", index=False)
        np.savez(RES / "oof_rep0.npz", **oof)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "det"):
        determinism()
    if what in ("all", "bench"):
        benchmark()
