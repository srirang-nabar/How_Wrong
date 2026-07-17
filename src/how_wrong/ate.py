"""ATE estimation with correct RCT inference, and power analysis (Stage 1).

`diff_in_means` uses the Neyman (unpooled) variance estimator — the
design-based standard error for a completely randomized experiment — with
normal-approximation CIs (arm sizes here are 10^5–10^7, so exact small-sample
corrections are irrelevant).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ATEResult:
    ate: float
    se: float
    ci_lo: float
    ci_hi: float
    p_value: float
    n_treated: int
    n_control: int
    mean_treated: float
    mean_control: float
    alpha: float

    def to_dict(self) -> dict:
        return asdict(self)


def diff_in_means(y, t, alpha: float = 0.05) -> ATEResult:
    """Difference-in-means ATE with Neyman SE, z-based CI and two-sided p."""
    y = np.asarray(y, dtype="float64")
    t = np.asarray(t)
    y1, y0 = y[t == 1], y[t == 0]
    n1, n0 = len(y1), len(y0)
    ate = y1.mean() - y0.mean()
    se = float(np.sqrt(y1.var(ddof=1) / n1 + y0.var(ddof=1) / n0))
    z = stats.norm.ppf(1 - alpha / 2)
    p = 2 * stats.norm.sf(abs(ate / se)) if se > 0 else float("nan")
    return ATEResult(
        ate=float(ate), se=se, ci_lo=float(ate - z * se), ci_hi=float(ate + z * se),
        p_value=float(p), n_treated=n1, n_control=n0,
        mean_treated=float(y1.mean()), mean_control=float(y0.mean()), alpha=alpha,
    )


def mde_two_proportions(
    p0: float, n1: int, n0: int, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Minimal detectable absolute lift for a two-proportion z-test.

    Normal approximation with variance evaluated at the base rate p0 on both
    arms — appropriate when the detectable lift is small relative to p0, as
    here (rare outcomes). Returns the absolute difference in proportions.
    """
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    se = np.sqrt(p0 * (1 - p0) * (1 / n1 + 1 / n0))
    return float((z_a + z_b) * se)


def power_two_proportions(
    delta: float, p0: float, n1: int, n0: int, alpha: float = 0.05
) -> float:
    """Power to detect an absolute lift `delta` over base rate p0."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(p0 * (1 - p0) * (1 / n1 + 1 / n0))
    return float(stats.norm.sf(z_a - abs(delta) / se)
                 + stats.norm.cdf(-z_a - abs(delta) / se))
