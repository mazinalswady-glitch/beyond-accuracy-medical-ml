"""E5 model-agnostic permutation-importance stability for all six models on all datasets,
with the two sources of randomness separated:

  design 'model': model seed varies (r = 0..R-1), permutation seed fixed (0)
  design 'perm' : model seed fixed (0), permutation seed varies (r = 0..R-1)
  design 'both' : both vary together (the v5 design)

Importance = mean decrease in AUC over n_repeats permutations on a fixed stratified
held-out test set (30%). Small datasets: R = 20, n_repeats = 50. D130: R = 10,
n_repeats = 30, permutation evaluated on a fixed stratified 1,000-patient subsample of
the test set (kernel-SVM prediction cost).

Output: results/perm_importances.npz, results/perm_stability.csv
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
sys.path.insert(0, str(Path(__file__).parent))
from data import load
from common import MODELS, DATASETS, make_model, pairwise_stability

RES = Path(__file__).resolve().parent


def main(datasets=DATASETS):
    rows, store = [], {}
    fn = "perm_stability.csv" if len(datasets) > 1 else f"perm_stability_{datasets[0]}.csv"
    for ds in datasets:
        X, y, _ = load(ds)
        R, NREP = (10, 30) if ds == "D130" else (20, 50)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=3)
        if ds == "D130":
            Xte, _, yte, _ = train_test_split(Xte, yte, train_size=1000, stratify=yte, random_state=4)
        for m in MODELS:
            fitted0 = make_model(m, seed=0).fit(Xtr, ytr)
            for design in ("model", "perm", "both"):
                t0 = time.time(); imps = []
                for r in range(R):
                    ms = 0 if design == "perm" else r
                    ps = 0 if design == "model" else r
                    mdl = fitted0 if ms == 0 else make_model(m, seed=ms).fit(Xtr, ytr)
                    pi = permutation_importance(mdl, Xte, yte, scoring="roc_auc", n_repeats=NREP,
                                                random_state=ps, n_jobs=2)
                    imps.append(pi.importances_mean)
                imps = np.array(imps); store[f"{ds}_{m}_{design}"] = imps
                r_ = {"dataset": ds, "model": m, "design": design, "R": R, "n_repeats": NREP}
                r_.update(pairwise_stability(imps)); rows.append(r_)
                print(ds, m, design, f"{time.time()-t0:.0f}s", round(r_["tau_mean"], 3), flush=True)
                pd.DataFrame(rows).to_csv(RES / fn, index=False)
        np.savez(RES / fn.replace("stability", "importances").replace(".csv", ".npz"), **store)


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else DATASETS)
