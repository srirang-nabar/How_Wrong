"""Stage 2 gate (hard): every learner must recover a known CATE on
synthetic data — a learner that can't is disqualified. Plus interface
contracts, metric sanity, CI determinism, and (once the fitting script has
run) the committed H1/metrics artifacts.
"""

import json

import numpy as np
import pytest

from how_wrong import data
from how_wrong.evaluate import (
    blp_test, bootstrap_ci, gates_deciles, qini_coefficient, qini_curve,
)
from how_wrong.folds import assign_folds
from how_wrong.learners import cross_fit_cate, default_learners

pytestmark = pytest.mark.gate_stage2

STAGE2 = data.PROJECT_ROOT / "results" / "stage2"
P_SYNTH = 0.85


def synth(n=20_000, seed=42):
    """Simulated RCT with known CATE tau(x) in [0.05, 0.15]."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    tau = 0.05 + 0.10 / (1 + np.exp(-2 * X[:, 0]))
    t = rng.binomial(1, P_SYNTH, n)
    y = 0.1 + 0.05 * X[:, 1] + tau * t + rng.normal(0, 0.1, n)
    return X, t, y, tau


@pytest.fixture(scope="module")
def synth_data():
    return synth()


@pytest.mark.parametrize("idx,name", [(0, "t_learner"), (1, "x_learner"),
                                      (2, "dr_learner"), (3, "causal_forest")])
def test_synthetic_recovery(synth_data, idx, name):
    """THE hard gate: recover a known CATE within tolerance."""
    X, t, y, tau = synth_data
    learner = default_learners(P_SYNTH, fast=True)[idx]
    assert learner.name == name
    est = learner.fit(X, t, y).predict_cate(X)
    corr = np.corrcoef(est, tau)[0, 1]
    bias = abs(est.mean() - tau.mean())
    assert corr >= 0.7, f"{name}: corr(est, true)={corr:.3f} < 0.7"
    assert bias <= 0.015, f"{name}: mean-CATE bias {bias:.4f} > 0.015"


def test_learner_interface_contract(synth_data):
    X, t, y, _ = synth_data
    for lr in default_learners(P_SYNTH, fast=True):
        out = lr.fit(X[:2000], t[:2000], y[:2000]).predict_cate(X[:100])
        assert out.shape == (100,)
        assert np.isfinite(out).all()


def test_cross_fit_covers_all_rows():
    X, t, y, tau = synth(n=5_000)
    folds = assign_folds(np.arange(len(y)))
    oof = cross_fit_cate(lambda: default_learners(P_SYNTH, fast=True)[0],
                         X, t, y, folds)
    assert oof.shape == (len(y),)
    assert np.isfinite(oof).all()


def test_qini_sanity(synth_data):
    """True tau as score must beat a random score; random ≈ 0."""
    X, t, y, tau = synth_data
    rng = np.random.default_rng(0)
    q_true = qini_coefficient(y, t, tau)
    q_rand = qini_coefficient(y, t, rng.normal(size=len(y)))
    assert q_true > 3 * abs(q_rand)
    assert q_true > 0
    curve = qini_curve(y, t, tau)
    assert curve["frac_targeted"].iloc[-1] == 1.0
    # at 100% targeted, qini and random meet (both = total incremental)
    assert curve["qini"].iloc[-1] == pytest.approx(curve["random"].iloc[-1])


def test_rank_metrics_immune_to_treatment_ordered_ties():
    """Regression test for the Criteo file-ordering trap: the raw file is
    treatment-block-ordered, so positional tie-breaking would correlate rank
    with treatment inside tied score blocks. With seeded-random tie-breaking,
    a constant score must yield ~zero qini and near-uniform arm shares in
    every GATES decile — even when the input is sorted by treatment."""
    rng = np.random.default_rng(9)
    n = 100_000
    t = np.r_[np.ones(int(n * 0.85)), np.zeros(n - int(n * 0.85))]  # blocked!
    y = rng.binomial(1, 0.05 + 0.01 * t)
    s = np.zeros(n)  # everything tied
    ate = y[t == 1].mean() - y[t == 0].mean()
    assert abs(qini_coefficient(y, t, s)) < 0.5 * abs(ate) * 0.1
    g = gates_deciles(y, t, s)
    assert not g["realized_ate"].isna().any()
    assert (g["n"] == n // 10).all()
    assert g["realized_ate"].std() < 5 * g["se"].mean()


def test_bootstrap_ci_deterministic_under_seed(synth_data):
    X, t, y, tau = synth_data
    a = bootstrap_ci(y[:5000], t[:5000], tau[:5000], B=50, seed=7)
    b = bootstrap_ci(y[:5000], t[:5000], tau[:5000], B=50, seed=7)
    c = bootstrap_ci(y[:5000], t[:5000], tau[:5000], B=50, seed=8)
    assert a == b
    assert a != c
    assert a["ci_lo"] <= a["point"] <= a["ci_hi"]


def test_gates_deciles_recover_gradient(synth_data):
    """With the true tau as score, realized decile effects must rise."""
    X, t, y, tau = synth_data
    g = gates_deciles(y, t, tau)
    assert len(g) == 10
    assert g["n"].sum() == len(y)
    assert g.loc[10, "realized_ate"] > g.loc[1, "realized_ate"]
    # top-decile CI excludes the bottom-decile point estimate
    assert g.loc[10, "ci_lo"] > g.loc[1, "realized_ate"]


def test_blp_detects_heterogeneity_and_its_absence():
    X, t, y, tau = synth(n=30_000)
    r = blp_test(y, t, tau, P_SYNTH)  # perfectly calibrated proxy
    assert r["beta2_p_value"] < 1e-6
    assert r["beta2_het"] == pytest.approx(1.0, abs=0.15)
    # constant-effect world: no heterogeneity to find
    rng = np.random.default_rng(1)
    y0 = 0.1 + 0.08 * t + rng.normal(0, 0.1, len(t))
    r0 = blp_test(y0, t, rng.normal(size=len(t)), P_SYNTH)
    assert r0["beta2_p_value"] > 0.01


# ---- committed-artifact checks (require scripts/02_fit_cate.py) ----

@pytest.fixture(scope="module")
def h1():
    path = STAGE2 / "h1_blp.json"
    if not path.exists():
        pytest.skip("stage 2 fitting script has not run yet")
    return json.loads(path.read_text())


def test_h1_verdict_recorded(h1):
    primary = h1["primary"]
    assert primary["outcome"] == "visit"
    assert primary["proxy"] == "dr_learner"
    assert "beta2_p_value" in primary["blp"]
    assert h1["verdict_h1_raw"] == (
        h1["primary"]["blp"]["beta2_p_value"] < 0.05
        and h1["primary"]["blp"]["beta2_het"] > 0
    )


def test_metrics_recomputable_from_oof(h1):
    """Spot-check: stored qini for the primary learner/outcome must be
    recomputable from the committed OOF predictions."""
    import pandas as pd
    metrics = json.loads((STAGE2 / "metrics.json").read_text())
    oof = pd.read_parquet(STAGE2 / "oof_cate.parquet")
    dev = data.load_criteo_dev()
    assert (oof["row_id"].to_numpy() == dev["row_id"].to_numpy()).all()
    q = qini_coefficient(dev["visit"], dev["treatment"], oof["visit_dr_learner"])
    assert q == pytest.approx(metrics["visit"]["dr_learner"]["qini"]["point"],
                              rel=1e-9)
