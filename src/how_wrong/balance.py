"""Covariate balance diagnostics for RCT arms (Stage 1).

SMD convention: (mean_t - mean_c) / sqrt((var_t + var_c) / 2), with the
|SMD| < 0.1 rule of thumb for "balanced". Categorical covariates are
one-hot encoded first via `encode_for_balance`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SMD_THRESHOLD = 0.1


def encode_for_balance(df: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """One-hot encode non-numeric covariates so smd_table sees numbers only."""
    cat = [c for c in covariates if not pd.api.types.is_numeric_dtype(df[c])]
    num = [c for c in covariates if c not in cat]
    out = df[num].astype("float64")
    if cat:
        dummies = pd.get_dummies(df[cat], columns=cat, dtype="float64")
        out = pd.concat([out, dummies], axis=1)
    return out


def smd_table(
    df: pd.DataFrame, treatment_col: str, covariates: list[str]
) -> pd.DataFrame:
    """Standardized mean differences per covariate, sorted by |SMD| desc."""
    X = encode_for_balance(df, covariates)
    t = df[treatment_col].to_numpy()
    mask = t == 1
    Xt, Xc = X[mask], X[~mask]
    mt, mc = Xt.mean(), Xc.mean()
    vt, vc = Xt.var(ddof=1), Xc.var(ddof=1)
    pooled_sd = np.sqrt((vt + vc) / 2)
    smd = (mt - mc) / pooled_sd.replace(0.0, np.nan)
    out = pd.DataFrame({
        "mean_treated": mt,
        "mean_control": mc,
        "smd": smd,
        "abs_smd": smd.abs(),
        "balanced": smd.abs() < SMD_THRESHOLD,
    })
    return out.sort_values("abs_smd", ascending=False)


def arm_summary(
    df: pd.DataFrame, treatment_col: str, outcome_cols: list[str]
) -> pd.DataFrame:
    """Per-arm n, share, and outcome rates."""
    g = df.groupby(treatment_col)
    out = g[outcome_cols].mean()
    out.insert(0, "n", g.size())
    out.insert(1, "share", out["n"] / len(df))
    return out
