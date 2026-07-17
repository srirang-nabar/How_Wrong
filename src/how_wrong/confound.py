"""Synthetic confounding: RCT -> observational datasets, and the repair
estimators graded against RCT ground truth (Stage 3).

Everything here follows HYPOTHESES.md amendment A2 exactly — mechanisms,
severity grids, estimator definitions, nuisance configs, and the
cross-fitting scheme were registered before any confounded estimation ran.

M1 (confounded assignment): keep treated w.p. sigmoid(γ·z), controls w.p.
sigmoid(−γ·z), where z is an observed covariate score — selection on
observables, so correction is possible in principle.
M2 (selective attrition): drop treated units with y = 0 w.p. p_drop —
selection on outcome, which no covariate-based estimator can repair.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from .folds import assign_folds

CONFOUND_FEATURES = ["f9", "f8", "f4"]
CONFOUND_SIGNS = np.array([1.0, -1.0, 1.0])
GAMMA_GRID = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
GAMMA_STAR = 1.0
PDROP_GRID = [0.0, 0.1, 0.25, 0.5]
N_REPLICATES = 20
CONFOUND_SEED = 20260717
PROPENSITY_CLIP = (0.01, 0.99)
ESTIMATOR_NAMES = ["naive", "outcome_regression", "ipw", "aipw"]


def confounder_score(df: pd.DataFrame) -> np.ndarray:
    """A2 confounder z: signed sum of standardized f9, f8, f4, standardized."""
    z = np.zeros(len(df))
    for feat, sign in zip(CONFOUND_FEATURES, CONFOUND_SIGNS):
        col = df[feat].to_numpy(dtype="float64")
        z += sign * (col - col.mean()) / col.std()
    return (z - z.mean()) / z.std()


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def confound_assignment(
    t: np.ndarray, z: np.ndarray, gamma: float, rng: np.random.Generator
) -> np.ndarray:
    """M1 keep-mask: treated kept w.p. sigmoid(γz), controls w.p. sigmoid(−γz)."""
    keep_p = np.where(t == 1, _sigmoid(gamma * z), _sigmoid(-gamma * z))
    return rng.random(len(t)) < keep_p


def selective_attrition(
    t: np.ndarray, y: np.ndarray, p_drop: float, rng: np.random.Generator
) -> np.ndarray:
    """M2 keep-mask: treated units with y == 0 dropped w.p. p_drop."""
    drop = (t == 1) & (y == 0) & (rng.random(len(t)) < p_drop)
    return ~drop


def _nuisance_reg(seed: int, n_jobs: int) -> LGBMRegressor:
    return LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31,
                         min_child_samples=200, random_state=seed,
                         n_jobs=n_jobs, verbose=-1)


def _nuisance_clf(seed: int, n_jobs: int) -> LGBMClassifier:
    return LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31,
                          min_child_samples=200, random_state=seed,
                          n_jobs=n_jobs, verbose=-1)


def _cross_fit_nuisances(X, t, y, row_ids, seed: int, n_jobs: int):
    """2-fold cross-fitted mu1(x), mu0(x), e(x) (A2: folds hashed on row_id)."""
    folds = assign_folds(row_ids, n_folds=2, seed=seed)
    mu1 = np.empty(len(y))
    mu0 = np.empty(len(y))
    e = np.empty(len(y))
    for k in (0, 1):
        tr, te = folds != k, folds == k
        m1 = _nuisance_reg(seed, n_jobs).fit(X[tr & (t == 1)], y[tr & (t == 1)])
        m0 = _nuisance_reg(seed, n_jobs).fit(X[tr & (t == 0)], y[tr & (t == 0)])
        pm = _nuisance_clf(seed, n_jobs).fit(X[tr], t[tr])
        mu1[te] = m1.predict(X[te])
        mu0[te] = m0.predict(X[te])
        e[te] = pm.predict_proba(X[te])[:, 1]
    return mu1, mu0, np.clip(e, *PROPENSITY_CLIP)


def estimate_all(
    df: pd.DataFrame, outcome: str, features: list[str],
    seed: int = CONFOUND_SEED, n_jobs: int = 4,
) -> dict[str, float]:
    """All four A2 estimators on one corrupted dataset. The estimators see
    only the corrupted rows and the given features — never the mechanism."""
    X = df[features].to_numpy(dtype="float64")
    t = df["treatment"].to_numpy()
    y = df[outcome].to_numpy(dtype="float64")
    naive = y[t == 1].mean() - y[t == 0].mean()
    mu1, mu0, e = _cross_fit_nuisances(X, t, y, df["row_id"].to_numpy(),
                                       seed, n_jobs)
    outcome_regression = float((mu1 - mu0).mean())
    w1, w0 = t / e, (1 - t) / (1 - e)
    ipw = float((w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum())
    aipw = float((mu1 - mu0 + t * (y - mu1) / e
                  - (1 - t) * (y - mu0) / (1 - e)).mean())
    return {"naive": float(naive), "outcome_regression": outcome_regression,
            "ipw": ipw, "aipw": aipw}


def run_cell(
    dev: pd.DataFrame, mechanism: str, severity: float, outcome: str,
    features: list[str], n_replicates: int = N_REPLICATES,
    seed: int = CONFOUND_SEED, n_jobs: int = 4,
) -> pd.DataFrame:
    """R corruption replicates of one (mechanism, severity) cell; returns
    replicate-level estimates for every estimator."""
    t = dev["treatment"].to_numpy()
    y = dev[outcome].to_numpy(dtype="float64")
    z = confounder_score(dev)
    child_seeds = np.random.SeedSequence(seed).spawn(n_replicates)
    rows = []
    for r, ss in enumerate(child_seeds):
        rng = np.random.default_rng(ss)
        if mechanism == "confounded_assignment":
            mask = confound_assignment(t, z, severity, rng)
        elif mechanism == "selective_attrition":
            mask = selective_attrition(t, y, severity, rng)
        else:
            raise ValueError(mechanism)
        est = estimate_all(dev[mask], outcome, features, seed=seed,
                           n_jobs=n_jobs)
        rows.append({"mechanism": mechanism, "severity": severity,
                     "outcome": outcome, "replicate": r,
                     "n_kept": int(mask.sum()),
                     "treated_share_kept": float(t[mask].mean()), **est})
    return pd.DataFrame(rows)


def summarize_bias(cell: pd.DataFrame, truth: float) -> dict:
    """Mean bias per estimator with 95% t-CIs across replicates (A2)."""
    from scipy import stats
    out = {}
    n = len(cell)
    tcrit = stats.t.ppf(0.975, n - 1)
    for name in ESTIMATOR_NAMES:
        b = cell[name].to_numpy() - truth
        se = b.std(ddof=1) / np.sqrt(n)
        t_stat = b.mean() / se if se > 0 else np.inf
        out[name] = {
            "mean_bias": float(b.mean()), "se": float(se),
            "ci_lo": float(b.mean() - tcrit * se),
            "ci_hi": float(b.mean() + tcrit * se),
            "p_two_sided": float(2 * stats.t.sf(abs(t_stat), n - 1)),
        }
    return out


def recovery_stats(cell: pd.DataFrame, truth: float) -> dict:
    """H3(b): per-replicate recovery rho = 1 − |bias_aipw|/|bias_naive|,
    one-sided t-test rho > 0, and the 95% t-CI (A2)."""
    from scipy import stats
    rho = (1 - (cell["aipw"] - truth).abs()
           / (cell["naive"] - truth).abs()).to_numpy()
    n = len(rho)
    se = rho.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    return {
        "mean_recovery": float(rho.mean()), "se": float(se),
        "ci_lo": float(rho.mean() - tcrit * se),
        "ci_hi": float(rho.mean() + tcrit * se),
        "p_one_sided_gt0": float(stats.t.sf(rho.mean() / se, n - 1)),
    }
