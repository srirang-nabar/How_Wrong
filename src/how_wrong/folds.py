"""Deterministic, hash-fingerprinted fold assignment (Stage 1).

Folds are a pure function of (row_id, seed) via a splitmix64 hash — never of
row order, dataset size, or library RNG state. The same row_id lands in the
same fold whether it appears in the dev subsample or the full data, so no
row can silently migrate between train and test across stages. Fold
fingerprints (sha256 of the assignment array) are asserted in gate tests so
any drift fails loudly.
"""

from __future__ import annotations

import hashlib

import numpy as np

N_FOLDS = 5
FOLD_SEED = 20260717


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorized splitmix64 finalizer (public-domain constants)."""
    z = x.astype(np.uint64, copy=True)
    with np.errstate(over="ignore"):
        z += np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return z


def assign_folds(
    row_ids, n_folds: int = N_FOLDS, seed: int = FOLD_SEED
) -> np.ndarray:
    """Map row_ids -> fold in [0, n_folds), deterministically."""
    ids = np.asarray(row_ids, dtype=np.uint64)
    return (_splitmix64(ids ^ _splitmix64(np.uint64(seed)))
            % np.uint64(n_folds)).astype(np.int8)


def fold_fingerprint(folds: np.ndarray) -> str:
    """sha256 of the fold-assignment array — the drift alarm."""
    return hashlib.sha256(np.ascontiguousarray(folds, dtype=np.int8)).hexdigest()


def train_test_ids(row_ids, test_fold: int, n_folds: int = N_FOLDS,
                   seed: int = FOLD_SEED) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(row_ids)
    folds = assign_folds(ids, n_folds=n_folds, seed=seed)
    return ids[folds != test_fold], ids[folds == test_fold]


def assert_disjoint(train_ids, test_ids) -> None:
    """Guard: raise if any id appears in both train and test."""
    overlap = np.intersect1d(np.asarray(train_ids), np.asarray(test_ids))
    if overlap.size:
        raise AssertionError(
            f"train/test leakage: {overlap.size} shared ids, e.g. {overlap[:5]}"
        )
