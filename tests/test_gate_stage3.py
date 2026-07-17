"""Stage 3 gate: mechanism correctness on synthetic data (runs anytime) and
the committed headline artifacts (skip until scripts/03_how_wrong.py runs).
"""

import json

import numpy as np
import pandas as pd
import pytest

from how_wrong import data
from how_wrong.confound import (
    GAMMA_STAR, confound_assignment, confounder_score, estimate_all,
    recovery_stats, selective_attrition, summarize_bias,
)

pytestmark = pytest.mark.gate_stage3

STAGE3 = data.PROJECT_ROOT / "results" / "stage3"


def synth_rct(n=40_000, seed=3):
    """Synthetic RCT: x0 drives both outcome and (post-corruption) selection;
    constant true effect tau = 0.05."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f9", "f8", "f4"])
    df["row_id"] = np.arange(n)
    df["treatment"] = rng.binomial(1, 0.85, n)
    signal = 0.3 * df["f9"] - 0.25 * df["f8"] + 0.15 * df["f4"]
    p_y = np.clip(0.1 + 0.08 * signal + 0.05 * df["treatment"], 0.01, 0.99)
    df["y"] = rng.binomial(1, p_y)
    return df, 0.05


def test_confounder_score_standardized():
    df, _ = synth_rct()
    z = confounder_score(df)
    assert abs(z.mean()) < 1e-9
    assert z.std() == pytest.approx(1.0, abs=1e-9)


def test_gamma_zero_is_unconfounded_thinning():
    df, _ = synth_rct()
    rng = np.random.default_rng(0)
    mask = confound_assignment(df["treatment"].to_numpy(),
                               confounder_score(df), 0.0, rng)
    assert mask.mean() == pytest.approx(0.5, abs=0.02)
    kept = df[mask]
    naive = (kept.y[kept.treatment == 1].mean()
             - kept.y[kept.treatment == 0].mean())
    assert naive == pytest.approx(0.05, abs=0.02)


def test_confounding_direction():
    """gamma > 0 must select high-z treated and low-z controls."""
    df, _ = synth_rct()
    z = confounder_score(df)
    t = df["treatment"].to_numpy()
    mask = confound_assignment(t, z, 1.5, np.random.default_rng(0))
    assert z[mask & (t == 1)].mean() > 0.3
    assert z[mask & (t == 0)].mean() < -0.3


def test_attrition_only_drops_treated_failures():
    df, _ = synth_rct()
    t, y = df["treatment"].to_numpy(), df["y"].to_numpy()
    mask = selective_attrition(t, y, 0.5, np.random.default_rng(0))
    dropped = ~mask
    assert (t[dropped] == 1).all()
    assert (y[dropped] == 0).all()
    n_at_risk = ((t == 1) & (y == 0)).sum()
    assert dropped.sum() / n_at_risk == pytest.approx(0.5, abs=0.03)


def test_corruption_deterministic_under_seed():
    df, _ = synth_rct()
    z = confounder_score(df)
    t = df["treatment"].to_numpy()
    a = confound_assignment(t, z, 1.0, np.random.default_rng(11))
    b = confound_assignment(t, z, 1.0, np.random.default_rng(11))
    c = confound_assignment(t, z, 1.0, np.random.default_rng(12))
    assert (a == b).all()
    assert (a != c).any()


def test_estimators_repair_observable_confounding():
    """The core mechanism check before the real run: under M1 the naive
    estimate must be materially biased while AIPW (which sees the
    confounders) recovers the truth."""
    df, tau = synth_rct()
    z = confounder_score(df)
    t = df["treatment"].to_numpy()
    mask = confound_assignment(t, z, 1.5, np.random.default_rng(5))
    est = estimate_all(df[mask], "y", ["f9", "f8", "f4"], n_jobs=1)
    naive_bias = abs(est["naive"] - tau)
    aipw_bias = abs(est["aipw"] - tau)
    assert naive_bias > 0.01, "confounding failed to bias the naive estimate"
    assert aipw_bias < 0.5 * naive_bias, (
        f"AIPW failed to repair: naive {naive_bias:.4f}, aipw {aipw_bias:.4f}"
    )
    assert abs(est["ipw"] - tau) < 0.5 * naive_bias


def test_nothing_repairs_selection_on_outcome():
    """M2 sanity: attrition on the outcome biases everything (the estimators
    are honest about their limits)."""
    df, tau = synth_rct()
    t, y = df["treatment"].to_numpy(), df["y"].to_numpy()
    mask = selective_attrition(t, y, 0.5, np.random.default_rng(5))
    est = estimate_all(df[mask], "y", ["f9", "f8", "f4"], n_jobs=1)
    for name in ("naive", "aipw"):
        assert est[name] - tau > 0.01, f"{name} unexpectedly unbiased under M2"


def test_summary_and_recovery_stats():
    cell = pd.DataFrame({
        "naive": [0.10, 0.11, 0.09, 0.105],
        "outcome_regression": [0.05] * 4,
        "ipw": [0.05] * 4,
        "aipw": [0.051, 0.049, 0.052, 0.048],
    })
    s = summarize_bias(cell, truth=0.05)
    assert s["naive"]["mean_bias"] == pytest.approx(0.05125)
    assert s["naive"]["ci_lo"] > 0
    r = recovery_stats(cell, truth=0.05)
    assert r["mean_recovery"] > 0.9
    assert r["p_one_sided_gt0"] < 0.01


# ---- committed-artifact checks (require scripts/03_how_wrong.py) ----

@pytest.fixture(scope="module")
def summary():
    path = STAGE3 / "summary.json"
    if not path.exists():
        pytest.skip("stage 3 experiment script has not run yet")
    return json.loads(path.read_text())


def test_h3_recorded_consistently(summary):
    h3 = summary["h3"]
    a, b = h3["a_naive_bias"], h3["b_recovery"]
    assert h3["gamma_star"] == GAMMA_STAR
    assert h3["p_h3"] == max(a["p_two_sided"], b["p_one_sided_gt0"])
    expected = (
        a["p_two_sided"] < 0.05
        and abs(a["mean_bias"]) >= 0.25 * abs(summary["truth"]["visit"])
        and b["mean_recovery"] >= 0.5
        and b["p_one_sided_gt0"] < 0.05
    )
    assert h3["verdict_h3_raw"] == expected


def test_severity_zero_placebo(summary):
    """At severity 0 the *naive* estimator must be unbiased — the placebo
    check that the corruption pipeline itself injects no confounding.

    The adjusted estimators are deliberately NOT held to zero there: at
    severity 0 they sit ~-0.3 pp below raw diff-in-means, and diagnosis
    (2026-07-17) showed this is largely the dataset's own residual
    non-randomness — v2.1 merges sub-experiments with different treatment
    ratios and covariate mixes (max SMD 0.049; e-hat spans [0.64, 0.98]
    with corr(e-hat, y) = +0.21), so covariate adjustment genuinely dissents
    from the pooled diff-in-means. We bound the offset instead: small
    relative to the gamma* naive bias it exists to repair."""
    naive_star = abs(summary["h3"]["a_naive_bias"]["mean_bias"])
    for cell in summary["grid"]:
        if cell["severity"] == 0 and cell["outcome"] == "visit":
            b = cell["bias"]
            assert b["naive"]["ci_lo"] <= 0 <= b["naive"]["ci_hi"], (
                f"{cell['mechanism']}: corruption pipeline biased at severity 0"
            )
            for name in ("outcome_regression", "ipw", "aipw"):
                assert abs(b[name]["mean_bias"]) < 0.1 * naive_star, (
                    f"{cell['mechanism']}/{name}: severity-0 offset "
                    f"{b[name]['mean_bias']:+.5f} exceeds 10% of the "
                    f"gamma* naive bias"
                )


def test_hidden_confounder_breaks_repair(summary):
    """The sensitivity story: denied f9/f8/f4, AIPW must lose most of its
    repair power at gamma*."""
    hidden = summary["exploratory"]["hidden_confounder"]
    full = next(c for c in summary["grid"]
                if c["mechanism"] == "confounded_assignment"
                and c["severity"] == GAMMA_STAR and c["outcome"] == "visit")
    assert abs(hidden["bias"]["aipw"]["mean_bias"]) > \
        3 * abs(full["bias"]["aipw"]["mean_bias"])
