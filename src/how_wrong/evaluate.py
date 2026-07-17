"""Honest uplift evaluation (Stage 2).

Metric definitions (they vary across the literature — these are ours):

- **Qini curve** Q(k): after sorting by score desc, at each prefix
  Q = (cum treated successes) − (cum control successes) × Nt/Nc — the
  cumulative number of *incremental* successes attributable to treating the
  top-k, on the treated scale.
- **Qini coefficient**: (area under Q − area under the random-targeting
  diagonal) / n — average incremental successes per person over random.
  Scale depends on the outcome's base rate; compare within an outcome only.
- External AUUC references (e.g. the published 0.64) use
  `causalml.metrics.auuc_score` (normalized cumulative-gain area), computed
  separately in the fitting script for context.

Ties in scores are broken **randomly with a fixed seed**, never by input
order: the raw Criteo file is treatment-block-ordered (verified 2026-07-17
— whole 50k-row spans are single-arm), so positional tie-breaking would
correlate rank with treatment inside tied blocks and corrupt every
rank-based metric. Bootstrap CIs are seeded and row-level.

Also here: GATES decile calibration and the Chernozhukov et al. BLP test
used for H1 (spec pre-registered in HYPOTHESES.md amendment A1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .ate import diff_in_means

BOOTSTRAP_B = 500
BOOTSTRAP_SEED = 20260717
TIE_SEED = 20260717


def _prep(y, t, scores):
    y = np.asarray(y, dtype="float64")
    t = np.asarray(t, dtype="float64")
    s = np.asarray(scores, dtype="float64")
    jitter = np.random.default_rng(TIE_SEED).random(len(s))
    order = np.lexsort((jitter, -s))  # desc by score, seeded-random in ties
    return y[order], t[order]


def qini_curve(y, t, scores, n_points: int = 200) -> pd.DataFrame:
    """Qini curve at ~n_points grid positions (always includes the endpoint)."""
    y, t = _prep(y, t, scores)
    n = len(y)
    cum_t = np.cumsum(t)
    cum_c = np.arange(1, n + 1) - cum_t
    cum_yt = np.cumsum(y * t)
    cum_yc = np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = cum_yt - np.where(cum_c > 0, cum_yc * cum_t / cum_c, 0.0)
    idx = np.unique(np.linspace(0, n - 1, n_points).astype(int))
    return pd.DataFrame({
        "frac_targeted": (idx + 1) / n,
        "qini": q[idx],
        "random": q[-1] * (idx + 1) / n,
    })


def qini_coefficient(y, t, scores) -> float:
    """(area under Qini − area under random) / n. Per-person incremental
    successes over random targeting; higher is better."""
    y, t = _prep(y, t, scores)
    n = len(y)
    cum_t = np.cumsum(t)
    cum_c = np.arange(1, n + 1) - cum_t
    cum_yt = np.cumsum(y * t)
    cum_yc = np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = cum_yt - np.where(cum_c > 0, cum_yc * cum_t / cum_c, 0.0)
    area_q = q.mean()                       # trapezoid ≈ mean for dense curve
    area_random = q[-1] * (n + 1) / (2 * n)
    return float((area_q - area_random) / n)


def bootstrap_ci(
    y, t, scores, metric_fn=qini_coefficient,
    B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED, alpha: float = 0.05,
) -> dict:
    """Seeded row-level bootstrap CI for any (y, t, scores) metric."""
    y = np.asarray(y); t = np.asarray(t); s = np.asarray(scores)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        vals[b] = metric_fn(y[idx], t[idx], s[idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"point": float(metric_fn(y, t, s)), "ci_lo": float(lo),
            "ci_hi": float(hi), "se": float(vals.std(ddof=1)),
            "B": B, "seed": seed}


def gates_deciles(y, t, scores, n_groups: int = 10) -> pd.DataFrame:
    """Predicted vs realized group effects: rank by score into n_groups,
    realized effect = within-group diff-in-means (valid because treatment
    is randomized, hence random within any score group)."""
    df = pd.DataFrame({"y": np.asarray(y), "t": np.asarray(t),
                       "s": np.asarray(scores)})
    jitter = np.random.default_rng(TIE_SEED).random(len(df))
    rank = np.empty(len(df), dtype="int64")
    rank[np.lexsort((jitter, df["s"].to_numpy()))] = np.arange(len(df))
    df["group"] = pd.qcut(rank, n_groups, labels=range(1, n_groups + 1))
    rows = []
    for g, sub in df.groupby("group", observed=True):
        r = diff_in_means(sub["y"], sub["t"])
        rows.append({"group": int(g), "n": len(sub),
                     "predicted_cate": sub["s"].mean(),
                     "realized_ate": r.ate, "se": r.se,
                     "ci_lo": r.ci_lo, "ci_hi": r.ci_hi})
    return pd.DataFrame(rows).set_index("group")


def blp_test(y, t, scores, p: float) -> dict:
    """Chernozhukov et al. BLP for heterogeneity (H1 spec, amendment A1).

    OLS: Y ~ 1 + (S−S̄) + (T−p) + (T−p)(S−S̄), HC3 robust SEs.
    beta1 (on T−p) estimates the ATE; beta2 (on the interaction) is the
    heterogeneity loading — beta2 = 1 means S is perfectly calibrated,
    beta2 = 0 means S carries no signal about effect heterogeneity.
    """
    y = np.asarray(y, dtype="float64")
    t = np.asarray(t, dtype="float64")
    s = np.asarray(scores, dtype="float64")
    s_c = s - s.mean()
    tp = t - p
    X = sm.add_constant(np.column_stack([s_c, tp, tp * s_c]))
    fit = sm.OLS(y, X).fit(cov_type="HC3")
    return {
        "beta1_ate": float(fit.params[2]), "beta1_se": float(fit.bse[2]),
        "beta2_het": float(fit.params[3]), "beta2_se": float(fit.bse[3]),
        "beta2_p_value": float(fit.pvalues[3]),
        "n": int(len(y)), "p_assign": p,
    }


def clan_table(X: pd.DataFrame, scores, frac: float = 0.2) -> pd.DataFrame:
    """CLAN: covariate means in the top vs bottom `frac` by predicted CATE,
    with the difference in SD units (descriptive, exploratory)."""
    s = np.asarray(scores)
    lo_cut, hi_cut = np.quantile(s, [frac, 1 - frac])
    top, bot = X[s >= hi_cut], X[s <= lo_cut]
    sd = X.std(ddof=1).replace(0.0, np.nan)
    out = pd.DataFrame({
        "mean_top": top.mean(), "mean_bottom": bot.mean(),
        "diff_in_sd": (top.mean() - bot.mean()) / sd,
    })
    return out.sort_values("diff_in_sd", key=abs, ascending=False)
