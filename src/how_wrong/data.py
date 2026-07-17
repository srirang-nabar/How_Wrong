"""Raw-data loading with schema contracts.

Every loader validates its schema against the constants recorded here, which
in turn mirror the values recorded in plan.md (verified 2026-07-16/17).
`data/raw/` is read-only; derived artifacts live under `data/derived/`.
"""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_DERIVED = PROJECT_ROOT / "data" / "derived"

CRITEO_PATH = DATA_RAW / "criteo-research-uplift-v2.1.csv.gz"
HILLSTROM_PATH = DATA_RAW / "hillstrom.csv"

CRITEO_FEATURES = [f"f{i}" for i in range(12)]
CRITEO_COLUMNS = CRITEO_FEATURES + ["treatment", "conversion", "visit", "exposure"]
CRITEO_N_ROWS = 13_979_592

HILLSTROM_COLUMNS = [
    "recency", "history_segment", "history", "mens", "womens", "zip_code",
    "newbie", "channel", "segment", "visit", "conversion", "spend",
]
HILLSTROM_N_ROWS = 64_000

_CRITEO_DTYPES = {c: "float32" for c in CRITEO_FEATURES} | {
    c: "int8" for c in ("treatment", "conversion", "visit", "exposure")
}


class SchemaError(ValueError):
    """Raised when a raw file does not match its recorded contract."""


def _check_columns(df: pd.DataFrame, expected: list[str], name: str) -> None:
    if list(df.columns) != expected:
        raise SchemaError(
            f"{name}: columns {list(df.columns)} != expected {expected}"
        )


def load_criteo(path: Path = CRITEO_PATH, nrows: int | None = None) -> pd.DataFrame:
    """Load the Criteo uplift v2.1 CSV (gzipped). float32/int8 to fit in RAM."""
    df = pd.read_csv(path, nrows=nrows, dtype=_CRITEO_DTYPES)
    _check_columns(df, CRITEO_COLUMNS, "criteo")
    return df


def load_hillstrom(path: Path = HILLSTROM_PATH) -> pd.DataFrame:
    """Load the Hillstrom email RCT. Adds a binary `treatment` column
    (1 = either email arm, 0 = no e-mail) alongside the 3-arm `segment`."""
    df = pd.read_csv(path)
    _check_columns(df, HILLSTROM_COLUMNS, "hillstrom")
    if len(df) != HILLSTROM_N_ROWS:
        raise SchemaError(f"hillstrom: {len(df)} rows != {HILLSTROM_N_ROWS}")
    df["treatment"] = (df["segment"] != "No E-Mail").astype("int8")
    return df


DEV_PARQUET = DATA_DERIVED / "criteo_dev_1m.parquet"
DEV_N_ROWS = 1_000_000
DEV_SEED = 20260717
DEV_STRATA = ["treatment", "conversion", "visit"]


def make_dev_subsample(
    df: pd.DataFrame,
    n: int = DEV_N_ROWS,
    seed: int = DEV_SEED,
    strata: list[str] = DEV_STRATA,
) -> pd.DataFrame:
    """Fixed-seed stratified subsample with exactly `n` rows.

    Stratifies on treatment × conversion × visit so arm shares and outcome
    rates are preserved by construction. Per-stratum allocations use the
    largest-remainder method to hit `n` exactly. Requires a `row_id` column
    (original raw-file row index) so fold assignment stays stable between
    subsample and full data. Deterministic: same df + seed -> same rows.
    """
    if "row_id" not in df.columns:
        raise ValueError("df needs a row_id column (original row index)")
    sizes = df.groupby(strata, sort=True).size()
    raw = sizes * (n / len(df))
    alloc = np.floor(raw).astype("int64")
    remainder = (raw - alloc).sort_values(ascending=False)
    for key in remainder.index[: n - int(alloc.sum())]:
        alloc[key] += 1
    rng = np.random.default_rng(seed)
    parts = [
        group.sample(n=int(alloc[key]), random_state=rng)
        for key, group in df.groupby(strata, sort=True)
    ]
    return pd.concat(parts).sort_values("row_id").reset_index(drop=True)


def load_criteo_dev(path: Path = DEV_PARQUET) -> pd.DataFrame:
    """Load the committed 1M-row development subsample, contract-checked."""
    df = pd.read_parquet(path)
    _check_columns(df, ["row_id"] + CRITEO_COLUMNS, "criteo_dev")
    if len(df) != DEV_N_ROWS:
        raise SchemaError(f"criteo_dev: {len(df)} rows != {DEV_N_ROWS}")
    return df


def count_rows_gz(path: Path = CRITEO_PATH) -> int:
    """Count data rows (excluding header) of a gzipped text file without
    parsing it — streams the decompressed bytes and counts newlines."""
    n = 0
    with gzip.open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            n += chunk.count(b"\n")
    return n - 1


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def verify_sidecar(path: Path) -> bool:
    """Check `path` against its `<name>.sha256` sidecar (filename-relative)."""
    sidecar = path.with_name(path.name + ".sha256")
    recorded = sidecar.read_text().split()[0]
    return sha256_of(path) == recorded
