"""Models, metrics, and stability measures shared by all experiments."""
import numpy as np
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, matthews_corrcoef, brier_score_loss)
from xgboost import XGBClassifier

MODELS = ["LR", "RF", "XGB", "SVM", "MLP", "NB"]
MODEL_NAMES = {"LR": "Logistic Regression", "RF": "Random Forest", "XGB": "XGBoost",
               "SVM": "SVM (RBF)", "MLP": "MLP", "NB": "Naïve Bayes"}
DATASETS = ["BC", "PIMA", "HEART", "D130"]
DS_NAMES = {"BC": "Breast Cancer", "PIMA": "Pima Diabetes", "HEART": "Cleveland Heart",
            "D130": "Diabetes-130 Readmission"}

# Exact XGBoost configuration (library defaults made explicit). With subsample = 1 and
# colsample_* = 1 no random numbers are drawn, so the model is seed-invariant.
XGB_PARAMS = dict(n_estimators=100, max_depth=6, learning_rate=0.3, tree_method="hist",
                  max_bin=256, subsample=1.0, colsample_bytree=1.0, colsample_bylevel=1.0,
                  colsample_bynode=1.0, min_child_weight=1, reg_lambda=1.0, gamma=0.0,
                  objective="binary:logistic", eval_metric="logloss", n_jobs=1)


def base_estimator(name, seed=0, **kw):
    if name == "LR":
        return LogisticRegression(max_iter=5000, **kw)
    if name == "RF":
        return RandomForestClassifier(n_estimators=kw.pop("n_estimators", 100), random_state=seed, n_jobs=1, **kw)
    if name == "XGB":
        p = dict(XGB_PARAMS); p.update(kw)
        return XGBClassifier(random_state=seed, **p)
    if name == "SVM":
        # probability=True fits Platt scaling with an internal 5-fold CV whose
        # split depends on random_state -> NOT seed-invariant.
        return SVC(kernel="rbf", probability=True, random_state=seed, **kw)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=kw.pop("hidden_layer_sizes", (64, 32)),
                             max_iter=1000, random_state=seed, **kw)
    if name == "NB":
        return GaussianNB(**kw)
    raise ValueError(name)


SCALED = {"LR", "SVM", "MLP"}


def make_model(name, seed=0, imputer="median", **kw):
    steps = []
    if imputer == "median":
        steps.append(("imp", SimpleImputer(strategy="median")))
    elif imputer is not None:
        steps.append(("imp", imputer))
    if name in SCALED:
        steps.append(("sc", StandardScaler()))
    steps.append(("clf", base_estimator(name, seed, **kw)))
    return Pipeline(steps)


# ----------------------------------------------------------------- metrics
EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_slope_intercept(y, p):
    """Calibration slope: coefficient of logit(p) in logistic recalibration.
    Calibration intercept (calibration-in-the-large): intercept with logit(p) as offset."""
    import statsmodels.api as sm
    lp = _logit(p)
    try:
        slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    except Exception:
        slope = np.nan
    try:
        inter = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
    except Exception:
        inter = np.nan
    return float(slope), float(inter)


def all_metrics(y, p, thr=0.5):
    """Threshold-dependent metrics use `thr`; experiments pass the training-fold event
    rate (prevalence threshold), so that a patient is classified positive when the
    predicted risk exceeds the baseline risk of the training population."""
    yhat = (p >= thr).astype(int)
    return {"ACC": accuracy_score(y, yhat), "BAL": balanced_accuracy_score(y, yhat),
            "F1": f1_score(y, yhat, zero_division=0), "AUC": roc_auc_score(y, p),
            "MCC": matthews_corrcoef(y, yhat), "BRIER": brier_score_loss(y, p)}


# --------------------------------------------------------- stability measures
def ranking(imp):
    """Feature indices ordered by decreasing importance."""
    return np.argsort(-np.asarray(imp), kind="stable")


def kendall(a, b):
    return stats.kendalltau(a, b).statistic


def weighted_kendall(a, b):
    """Weighted Kendall tau (Vigna, 2015) with hyperbolic weights on decreasing
    importance: disagreements among top-ranked features weigh most."""
    return stats.weightedtau(a, b).statistic


def rbo(a, b, p=0.9):
    """Extrapolated rank-biased overlap (Webber et al., 2010) of two full rankings
    derived from importance vectors a, b (higher = more important)."""
    ra, rb = ranking(a), ranking(b)
    k = len(ra)
    sa, sb = set(), set()
    overlap, s = 0, 0.0
    for d in range(1, k + 1):
        x, y = ra[d - 1], rb[d - 1]
        if x == y:
            overlap += 1
        else:
            overlap += (x in sb) + (y in sa)
        sa.add(x); sb.add(y)
        s += (overlap / d) * p ** d
    return (overlap / k) * p ** k + (1 - p) / p * s


def jaccard_topk(a, b, k=5):
    A, B = set(ranking(a)[:k]), set(ranking(b)[:k])
    return len(A & B) / len(A | B)


def pairwise_stability(imps, k=5):
    """imps: (R, F) importance matrix; returns dict of summary statistics over all pairs."""
    R = imps.shape[0]
    out = {"tau": [], "wtau": [], "rbo": [], "j5": []}
    for i in range(R):
        for j in range(i + 1, R):
            a, b = imps[i], imps[j]
            if np.allclose(a, b, rtol=0, atol=1e-12):
                t = wt = r = jj = 1.0
            else:
                t, wt, r, jj = kendall(a, b), weighted_kendall(a, b), rbo(a, b), jaccard_topk(a, b, k)
            out["tau"].append(t); out["wtau"].append(wt); out["rbo"].append(r); out["j5"].append(jj)
    res = {}
    for key, v in out.items():
        v = np.asarray(v, float)
        res[key + "_mean"] = float(np.nanmean(v)); res[key + "_sd"] = float(np.nanstd(v, ddof=1))
        res[key + "_min"] = float(np.nanmin(v))
    return res


def correlation_groups(X, thr=0.8, base_groups=None):
    """Group columns by source variable (base_groups) and then merge groups whose
    members have |Spearman rho| >= thr (average-linkage hierarchical clustering)."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    cols = list(X.columns)
    src = [base_groups.get(c, c) if base_groups else c for c in cols]
    srcs = list(dict.fromkeys(src))
    # source-level correlation: max |rho| between any members
    rho = X.rank().corr().abs().fillna(0).values
    S = len(srcs)
    D = np.zeros((S, S))
    for i in range(S):
        for j in range(i + 1, S):
            ii = [k for k, s in enumerate(src) if s == srcs[i]]
            jj = [k for k, s in enumerate(src) if s == srcs[j]]
            D[i, j] = D[j, i] = 1 - rho[np.ix_(ii, jj)].max()
    if S > 1:
        lab = fcluster(linkage(squareform(D, checks=False), "average"), t=1 - thr, criterion="distance")
    else:
        lab = np.array([1])
    names = {}
    for s, l in zip(srcs, lab):
        names.setdefault(l, []).append(s)
    src_to_group = {}
    for l, members in names.items():
        gname = members[0] if len(members) == 1 else "{" + "+".join(members) + "}"
        for m in members:
            src_to_group[m] = gname
    return [src_to_group[s] for s in src]
