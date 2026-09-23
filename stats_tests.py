"""Dependence-aware and classical comparison procedures."""
import numpy as np
from scipy import stats

Q_NEMENYI_05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949}


def holm(p):
    p = np.asarray(p, float); m = len(p)
    order = np.argsort(p); adj = np.empty(m); run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx]); adj[idx] = min(1.0, run)
    return adj


def corrected_resampled_t(d, test_train_ratio):
    """Nadeau & Bengio (2003) corrected resampled t-test, repeated-CV form
    (Bouckaert & Frank, 2004). d: per-fold differences (J values)."""
    d = np.asarray(d, float); J = len(d)
    v = d.var(ddof=1)
    if v == 0:
        return (0.0, 1.0) if d.mean() == 0 else (np.inf * np.sign(d.mean()), 0.0)
    t = d.mean() / np.sqrt((1 / J + test_train_ratio) * v)
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))


def bayes_correlated_t(d, rho, rope):
    """Bayesian correlated t-test (Corani & Benavoli, 2015; Benavoli et al., 2017).
    Returns P(d < -rope), P(|d| <= rope), P(d > rope) for the mean difference."""
    d = np.asarray(d, float); J = len(d)
    v = d.var(ddof=1)
    scale = np.sqrt((1 / J + rho / (1 - rho)) * v)
    if scale == 0:
        m = d.mean(); return float(m < -rope), float(abs(m) <= rope), float(m > rope)
    post = stats.t(df=J - 1, loc=d.mean(), scale=scale)
    left = post.cdf(-rope); right = post.sf(rope)
    return float(left), float(1 - left - right), float(right)


def friedman_nemenyi(M):
    """M: (N blocks, k algorithms) scores, higher = better. Returns chi2, p, mean ranks, CD."""
    N, k = M.shape
    ranks = np.array([stats.rankdata(-row) for row in M])
    R = ranks.mean(0)
    chi2 = 12 * N / (k * (k + 1)) * (np.sum(R ** 2) - k * (k + 1) ** 2 / 4)
    p = stats.chi2.sf(chi2, k - 1)
    cd = Q_NEMENYI_05[k] * np.sqrt(k * (k + 1) / (6 * N))
    return float(chi2), float(p), R, float(cd)


def hierarchical_bayes(diffs_per_ds, rho, rope, draws=4000, seed=0):
    """Bayesian hierarchical correlated t-test (Benavoli et al., 2017, JMLR 18:77).

    diffs_per_ds: list of arrays of per-fold differences, one array per dataset.
    Compound-symmetric fold correlation rho = n_test / n. The likelihood is written
    through its sufficient statistics (fold mean and within-dataset sum of squares).
    Returns posterior probabilities for the difference on a *new* dataset
    (posterior predictive of delta), plus the posterior mean of delta0.
    """
    import pymc as pm
    xbar = np.array([np.mean(d) for d in diffs_per_ds])
    ss = np.array([np.sum((np.asarray(d) - np.mean(d)) ** 2) for d in diffs_per_ds])
    J = np.array([len(d) for d in diffs_per_ds])
    scale = max(np.abs(xbar).max(), np.sqrt(ss.max() / J.min()), 1e-3)
    with pm.Model():
        delta0 = pm.Uniform("delta0", -10 * scale, 10 * scale)
        sigma0 = pm.Uniform("sigma0", 0, 10 * scale)
        alpha = pm.Uniform("alpha", 0.5, 5); beta = pm.Uniform("beta", 0.05, 0.15)
        nu = pm.Gamma("nu", alpha=alpha, beta=beta)
        delta = pm.StudentT("delta", nu=nu, mu=delta0, sigma=sigma0, shape=len(xbar))
        sigma = pm.Uniform("sigma", 0, 10 * scale, shape=len(xbar))
        sd_mean = sigma * np.sqrt((1 - rho) / J + rho)
        pm.Normal("xbar", mu=delta, sigma=sd_mean, observed=xbar)
        # SS / (sigma^2 (1-rho)) ~ chi2(J-1)  <=>  SS ~ Gamma((J-1)/2, rate = 1/(2 sigma^2 (1-rho)))
        pm.Gamma("ss", alpha=(J - 1) / 2, beta=1 / (2 * sigma ** 2 * (1 - rho)), observed=ss)
        tr = pm.sample(draws=draws // 4, tune=1500, chains=4, cores=1, random_seed=seed,
                       target_accept=0.95, progressbar=False)
    post = tr.posterior
    d0 = post["delta0"].values.ravel(); s0 = post["sigma0"].values.ravel(); nu_ = post["nu"].values.ravel()
    rng = np.random.default_rng(seed)
    new = d0 + s0 * rng.standard_t(np.maximum(nu_, 0.5))
    return {"p_left": float(np.mean(new < -rope)), "p_rope": float(np.mean(np.abs(new) <= rope)),
            "p_right": float(np.mean(new > rope)), "delta0_mean": float(d0.mean())}
