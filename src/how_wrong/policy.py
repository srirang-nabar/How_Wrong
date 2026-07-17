"""Budget-constrained uplift targeting and its evaluation (Stage 4).

Follows HYPOTHESES.md amendment A3 exactly. The estimated incremental
conversions of a policy at budget k = (within-targeted-set diff-in-means)
× |targeted set|, with the targeted set = top k% by score. Ties are broken
by fixed-seed jitter (the Stage 2 convention — the raw file is
treatment-block-ordered, so positional tie-breaking is forbidden). The
IPW/Hájek variant (robustness, per CLAIMS C16) replaces the plain
within-set means with ê-weighted means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import TIE_SEED

POLICY_BOOTSTRAP_B = 500
POLICY_SEED = 20260717
K_PRIMARY = (0.10, 0.30)


def _order(scores: np.ndarray) -> np.ndarray:
    """Descending score order with seeded-random tie-breaking."""
    jitter = np.random.default_rng(TIE_SEED).random(len(scores))
    return np.lexsort((jitter, -scores))


def _subset_uplift(y, t, e=None):
    """Diff-in-means within a subset; Hájek-weighted when ê is given."""
    if t.sum() == 0 or (1 - t).sum() == 0:
        return np.nan
    if e is None:
        return y[t == 1].mean() - y[t == 0].mean()
    w1, w0 = t / e, (1 - t) / (1 - e)
    return (w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum()


def incremental_conversions(y, t, scores, k, e=None) -> float:
    """Estimated incremental conversions from treating the top k%."""
    y = np.asarray(y, dtype="float64")
    t = np.asarray(t, dtype="float64")
    s = np.asarray(scores, dtype="float64")
    top = _order(s)[: max(1, int(round(k * len(s))))]
    e_top = None if e is None else np.asarray(e, dtype="float64")[top]
    return float(_subset_uplift(y[top], t[top], e_top) * len(top))


def three_curves(y, t, scores_by_policy: dict, k_grid, e=None) -> pd.DataFrame:
    """Incremental conversions across the k-grid for each policy, plus the
    analytic random baseline (k × total incremental)."""
    y = np.asarray(y, dtype="float64")
    t = np.asarray(t, dtype="float64")
    total = _subset_uplift(y, t, None if e is None else np.asarray(e)) * len(y)
    rows = []
    for k in k_grid:
        row = {"k": k, "random": total * k}
        for name, s in scores_by_policy.items():
            row[name] = incremental_conversions(y, t, s, k, e)
        rows.append(row)
    return pd.DataFrame(rows)


def policy_difference_test(
    y, t, scores_a, scores_b, k, e=None,
    B: int = POLICY_BOOTSTRAP_B, seed: int = POLICY_SEED,
) -> dict:
    """A3 H2 test: bootstrap Δ(k) = incremental(a) − incremental(b).
    Joint row-level resample; both policies re-ranked and re-thresholded
    per resample. Two-sided percentile CI and bootstrap p."""
    y = np.asarray(y, dtype="float64")
    t = np.asarray(t, dtype="float64")
    sa = np.asarray(scores_a, dtype="float64")
    sb = np.asarray(scores_b, dtype="float64")
    ea = None if e is None else np.asarray(e, dtype="float64")
    n = len(y)
    point = (incremental_conversions(y, t, sa, k, ea)
             - incremental_conversions(y, t, sb, k, ea))
    rng = np.random.default_rng(seed)
    deltas = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        eb = None if ea is None else ea[idx]
        deltas[b] = (incremental_conversions(y[idx], t[idx], sa[idx], k, eb)
                     - incremental_conversions(y[idx], t[idx], sb[idx], k, eb))
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    p_boot = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {"k": k, "delta": point, "ci_lo": float(lo), "ci_hi": float(hi),
            "p_boot": float(max(p_boot, 1 / B)), "B": B, "seed": seed}


def roi_table(
    curve: pd.DataFrame, policy: str, n_total: int,
    value_per_conversion: float, cost_per_target: float,
) -> pd.DataFrame:
    """Profit(k) = value × incremental(k) − cost × k × N (A3 exploratory)."""
    out = curve[["k", policy]].copy()
    out["n_targeted"] = (out["k"] * n_total).round().astype("int64")
    out["revenue"] = out[policy] * value_per_conversion
    out["cost"] = out["n_targeted"] * cost_per_target
    out["profit"] = out["revenue"] - out["cost"]
    return out
