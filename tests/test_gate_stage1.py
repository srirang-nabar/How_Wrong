"""Stage 1 gate: balance verified, subsample certified representative,
fold discipline in place.

Fast path only — everything asserts against the committed dev parquet and
`results/stage1/subsample_cert.json` (produced by
`scripts/01_make_dev_subsample.py`, the Tier 3 full-data pass).
"""

import json

import numpy as np
import pandas as pd
import pytest

from how_wrong import data
from how_wrong.ate import diff_in_means
from how_wrong.balance import smd_table
from how_wrong.folds import (
    N_FOLDS, assert_disjoint, assign_folds, fold_fingerprint, train_test_ids,
)

pytestmark = pytest.mark.gate_stage1

CERT_PATH = data.PROJECT_ROOT / "results" / "stage1" / "subsample_cert.json"
OUTCOMES = ["conversion", "visit"]


@pytest.fixture(scope="module")
def cert():
    return json.loads(CERT_PATH.read_text())


@pytest.fixture(scope="module")
def dev():
    return data.load_criteo_dev()


def test_dev_parquet_contract(dev):
    assert data.verify_sidecar(data.DEV_PARQUET)
    assert dev["row_id"].is_unique
    assert dev["row_id"].between(0, data.CRITEO_N_ROWS - 1).all()
    assert dev["row_id"].is_monotonic_increasing


def test_balance_full_data(cert):
    assert cert["verdicts"]["balance_full"] is True
    assert cert["full"]["smd_max_abs"] < 0.1


def test_balance_dev_subsample(dev):
    smd = smd_table(dev, "treatment", data.CRITEO_FEATURES)
    assert smd["abs_smd"].max() < 0.1


def test_subsample_certified_representative(cert):
    v = cert["verdicts"]
    assert v["cov_moments_representative"] is True
    assert cert["dev"]["cov_mean_max_abs_z"] < cert["criteria"]["cov_mean_abs_z_lt"]
    for y in OUTCOMES:
        assert v["ate_representative"][y] is True


def test_dev_ate_matches_cert(dev, cert):
    """The committed cert numbers must be recomputable from the parquet."""
    for y in OUTCOMES:
        r = diff_in_means(dev[y], dev["treatment"])
        assert r.ate == pytest.approx(cert["dev"]["ate"][y]["ate"], rel=1e-9)
        assert r.se == pytest.approx(cert["dev"]["ate"][y]["se"], rel=1e-9)


def test_sampling_determinism():
    rng = np.random.default_rng(7)
    n = 50_000
    df = pd.DataFrame({
        "row_id": np.arange(n),
        "f0": rng.normal(size=n),
        "treatment": rng.binomial(1, 0.85, n),
    })
    df["conversion"] = rng.binomial(1, 0.01 + 0.005 * df["treatment"])
    df["visit"] = rng.binomial(1, 0.05 + 0.01 * df["treatment"])
    a = data.make_dev_subsample(df, n=5_000, seed=1)
    b = data.make_dev_subsample(df, n=5_000, seed=1)
    c = data.make_dev_subsample(df, n=5_000, seed=2)
    assert a.equals(b)
    assert not a.equals(c)
    assert len(a) == 5_000
    # stratification preserves arm share to within rounding
    assert a["treatment"].mean() == pytest.approx(df["treatment"].mean(), abs=1e-3)


def test_fold_fingerprint_matches_cert(dev, cert):
    folds = assign_folds(dev["row_id"].to_numpy())
    assert fold_fingerprint(folds) == cert["folds"]["dev_fingerprint"]
    assert np.bincount(folds, minlength=N_FOLDS).tolist() == cert["folds"]["dev_counts"]


def test_fold_discipline(dev):
    ids = dev["row_id"].to_numpy()
    seen = 0
    for k in range(N_FOLDS):
        train, test = train_test_ids(ids, k)
        assert_disjoint(train, test)
        assert len(train) + len(test) == len(ids)
        seen += len(test)
    assert seen == len(ids)  # folds partition the data
    with pytest.raises(AssertionError, match="leakage"):
        assert_disjoint(np.array([1, 2, 3]), np.array([3, 4]))


def test_headline_numbers_match_claims(cert):
    """Pins CLAIMS.md C3–C7 to the committed cert — drift fails loudly."""
    conv, vis = cert["full"]["ate"]["conversion"], cert["full"]["ate"]["visit"]
    assert conv["ate"] == pytest.approx(0.0011519, abs=5e-8)      # C3
    assert conv["ci_lo"] == pytest.approx(0.0010845, abs=5e-8)
    assert conv["ci_hi"] == pytest.approx(0.0012192, abs=5e-8)
    assert vis["ate"] == pytest.approx(0.0103424, abs=5e-8)       # C4
    assert cert["full"]["smd_max_abs"] == pytest.approx(0.0488, abs=5e-5)   # C5
    assert cert["dev"]["cov_mean_max_abs_z"] == pytest.approx(1.83, abs=5e-3)  # C6
    assert cert["mde"]["mde_abs_at_1m"] == pytest.approx(0.000345, abs=5e-7)   # C7
    assert cert["mde"]["mde_abs_at_full"] == pytest.approx(0.0000923, abs=5e-7)


def test_fold_stability_across_contexts():
    """A row's fold depends only on its id — not on which dataset slice
    it appears in. This is what makes dev-vs-full fold membership stable."""
    ids = np.arange(10_000)
    subset = ids[::7]
    assert (assign_folds(ids)[subset] == assign_folds(subset)).all()
