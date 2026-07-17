"""Stage 0 gate: environment, raw-data presence, schema/row-count contracts.

Row counts and schemas are asserted against the values recorded in plan.md
and CLAIMS.md (C1, C2). Run the cheap subset with
`pytest -m "gate_stage0 and not slow"`; the full-file contract (gzip
decompression of 311 MB) is marked slow.
"""

import pytest

from how_wrong import data

pytestmark = pytest.mark.gate_stage0

# raw files are gitignored (311 MB); on a fresh clone (Tier 1: no downloads)
# the raw-data contracts skip rather than fail — Tier 3 restores them
needs_raw = pytest.mark.skipif(
    not data.CRITEO_PATH.exists() or not data.HILLSTROM_PATH.exists(),
    reason="raw data not present (Tier 1 clone) — see REPRODUCING.md Tier 3",
)


def test_causal_stack_imports():
    """The Stage 0 decision point: causalml AND econml both work on 3.13."""
    from causalml.inference.meta import (  # noqa: F401
        BaseDRRegressor, BaseTRegressor, BaseXRegressor,
    )
    from causalml.metrics import auuc_score, qini_score  # noqa: F401
    from econml.dr import DRLearner  # noqa: F401
    from econml.grf import CausalForest  # noqa: F401
    import lightgbm  # noqa: F401


@needs_raw
def test_raw_files_present_and_checksummed():
    for path in (data.CRITEO_PATH, data.HILLSTROM_PATH):
        assert path.exists(), f"missing raw file: {path}"
        assert data.verify_sidecar(path), f"checksum mismatch: {path}"


@needs_raw
def test_hillstrom_schema_and_rowcount():
    df = data.load_hillstrom()
    assert len(df) == data.HILLSTROM_N_ROWS  # CLAIMS.md C2
    assert df["treatment"].isin((0, 1)).all()
    # Three-arm RCT, ~1/3 each; binary treatment collapses the two email arms.
    assert df["treatment"].mean() == pytest.approx(2 / 3, abs=0.01)
    assert set(df["segment"].unique()) == {
        "Mens E-Mail", "Womens E-Mail", "No E-Mail",
    }


@needs_raw
def test_criteo_schema():
    df = data.load_criteo(nrows=1000)
    assert list(df.columns) == data.CRITEO_COLUMNS
    for col in ("treatment", "conversion", "visit", "exposure"):
        assert df[col].isin((0, 1)).all()


@pytest.mark.slow
@needs_raw
def test_criteo_full_rowcount():
    assert data.count_rows_gz() == data.CRITEO_N_ROWS  # CLAIMS.md C1
