"""Stage 4 gate: policy machinery on synthetic data (runs anytime) and the
committed H2 artifacts (skip until scripts/04_policy.py runs).

The core synthetic is the quadrant story: "sure things" convert regardless
of treatment (high baseline, zero uplift) while "persuadables" convert only
if treated. Propensity targeting chases sure things; uplift targeting must
provably beat it on incremental conversions.
"""

import json

import numpy as np
import pytest

from how_wrong import data
from how_wrong.policy import (
    incremental_conversions, policy_difference_test, roi_table, three_curves,
)

pytestmark = pytest.mark.gate_stage4

STAGE4 = data.PROJECT_ROOT / "results" / "stage4"


@pytest.fixture(scope="module")
def quadrant_world():
    """50% sure things (p=0.6 either way), 50% persuadables (0.05 -> 0.25)."""
    rng = np.random.default_rng(21)
    n = 100_000
    sure = rng.random(n) < 0.5
    t = rng.binomial(1, 0.85, n)
    p = np.where(sure, 0.6, np.where(t == 1, 0.25, 0.05))
    y = rng.binomial(1, p).astype("float64")
    tau_true = np.where(sure, 0.0, 0.20)          # ideal uplift score
    propensity_score = np.where(sure, 0.6, 0.15)  # "likely buyer" score
    return y, t.astype("float64"), tau_true, propensity_score, sure


def test_uplift_targeting_beats_propensity_targeting(quadrant_world):
    y, t, tau, prop, sure = quadrant_world
    r = policy_difference_test(y, t, tau, prop, k=0.3, B=100, seed=5)
    assert r["delta"] > 0
    assert r["ci_lo"] > 0
    # propensity targeting at k=30% selects sure things -> ~zero incremental
    inc_prop = incremental_conversions(y, t, prop, 0.3)
    inc_tau = incremental_conversions(y, t, tau, 0.3)
    assert inc_prop < 0.25 * inc_tau


def test_full_budget_equals_total_uplift(quadrant_world):
    y, t, tau, _, _ = quadrant_world
    total = (y[t == 1].mean() - y[t == 0].mean()) * len(y)
    assert incremental_conversions(y, t, tau, 1.0) == pytest.approx(total)
    curve = three_curves(y, t, {"tau": tau}, [1.0])
    assert curve["tau"].iloc[0] == pytest.approx(curve["random"].iloc[0])


def test_bootstrap_deterministic_and_seed_sensitive(quadrant_world):
    y, t, tau, prop, _ = quadrant_world
    a = policy_difference_test(y, t, tau, prop, k=0.1, B=50, seed=3)
    b = policy_difference_test(y, t, tau, prop, k=0.1, B=50, seed=3)
    c = policy_difference_test(y, t, tau, prop, k=0.1, B=50, seed=4)
    assert a == b
    assert a != c


def test_ipw_variant_matches_plain_under_constant_propensity(quadrant_world):
    """With a truly constant ê the Hájek variant must agree with the plain
    diff-in-means (they differ only when ê varies)."""
    y, t, tau, _, _ = quadrant_world
    e = np.full(len(y), 0.85)
    plain = incremental_conversions(y, t, tau, 0.3)
    ipw = incremental_conversions(y, t, tau, 0.3, e=e)
    assert ipw == pytest.approx(plain, rel=1e-9)


def test_roi_table_arithmetic(quadrant_world):
    y, t, tau, _, _ = quadrant_world
    curve = three_curves(y, t, {"tau": tau}, [0.1, 0.3])
    roi = roi_table(curve, "tau", len(y), value_per_conversion=40,
                    cost_per_target=0.15)
    row = roi[roi.k == 0.3].iloc[0]
    assert row["profit"] == pytest.approx(
        row["tau"] * 40 - 0.3 * len(y) * 0.15, rel=1e-6, abs=1e-6)


# ---- committed-artifact checks (require scripts/04_policy.py) ----

@pytest.fixture(scope="module")
def results():
    path = STAGE4 / "policy_results.json"
    if not path.exists():
        pytest.skip("stage 4 policy script has not run yet")
    return json.loads(path.read_text())


def test_h2_recorded_consistently(results):
    h2 = results["h2"]
    p10, p30 = h2["primary"]["k10"], h2["primary"]["k30"]
    assert h2["p_h2"] == max(p10["p_boot"], p30["p_boot"])
    expected = (p10["delta"] > 0 and p10["ci_lo"] > 0
                and p30["delta"] > 0 and p30["ci_lo"] > 0)
    assert h2["verdict_h2_raw"] == expected
    # the C16 robustness variant must be present and complete
    assert set(h2["ipw_robust"]) == {"k10", "k30"}
    assert h2["robust_agrees"] == (
        (h2["ipw_robust"]["k10"]["ci_lo"] > 0)
        == (p10["ci_lo"] > 0)
        and (h2["ipw_robust"]["k30"]["ci_lo"] > 0) == (p30["ci_lo"] > 0)
    )


def test_curves_recomputable(results):
    """Spot-check: the stored CATE-policy point at k=0.30 must be
    recomputable from committed artifacts."""
    import pandas as pd
    curves = pd.read_csv(STAGE4 / "curves_conversion.csv")
    oof = pd.read_parquet(
        data.PROJECT_ROOT / "results" / "stage2" / "oof_cate.parquet")
    dev = data.load_criteo_dev()
    val = incremental_conversions(dev["conversion"], dev["treatment"],
                                  oof["conversion_dr_learner"], 0.30)
    stored = curves.loc[(curves["k"] - 0.30).abs() < 1e-9, "cate"].iloc[0]
    assert val == pytest.approx(stored, rel=1e-9)
