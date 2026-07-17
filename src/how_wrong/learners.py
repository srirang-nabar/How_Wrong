"""Uniform CATE-learner interface (Stage 2).

Every learner exposes `fit(X, t, y) -> self` and `predict_cate(X) -> (n,)`.
Backends: causalml meta-learners (T/X/DR) with LightGBM base learners and
the econml honest causal forest. The assignment propensity is *known* in an
RCT, so it is passed in — never estimated.

Configs are fixed by HYPOTHESES.md amendment A1 (no tuning against
evaluation metrics). `fast=True` variants exist only for the synthetic
recovery tests in the gate.
"""

from __future__ import annotations

import numpy as np
from causalml.inference.meta import BaseDRRegressor, BaseTRegressor, BaseXRegressor
from econml.grf import CausalForest
from lightgbm import LGBMRegressor

LEARNER_SEED = 20260717


def _lgbm(fast: bool, seed: int) -> LGBMRegressor:
    # n_jobs is pinned explicitly: n_jobs=-1 triggers pathological OpenMP
    # thread-spinning on small data on this hardware (42s vs 0.08s for an
    # identical fit, measured 2026-07-17).
    if fast:
        return LGBMRegressor(n_estimators=60, learning_rate=0.1, num_leaves=15,
                             min_child_samples=20, random_state=seed,
                             n_jobs=1, verbose=-1)
    return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63,
                         min_child_samples=200, random_state=seed,
                         n_jobs=4, verbose=-1)


class _CausalmlMeta:
    """Wraps a causalml meta-learner; passes the known propensity if used."""

    needs_propensity = False       # pass known p at fit time
    predicts_with_propensity = False  # ... and at predict time (X-learner)

    def __init__(self, name: str, model, p: float | None):
        self.name = name
        self._model = model
        self._p = p

    def fit(self, X, t, y):
        X = np.asarray(X, dtype="float64")
        t = np.asarray(t)
        y = np.asarray(y, dtype="float64")
        if self.needs_propensity:
            self._model.fit(X, t, y, p=np.full(len(t), self._p))
        else:
            self._model.fit(X, t, y)
        return self

    def predict_cate(self, X) -> np.ndarray:
        X = np.asarray(X, dtype="float64")
        if self.predicts_with_propensity:
            out = self._model.predict(X, p=np.full(len(X), self._p))
        else:
            out = self._model.predict(X)
        return np.asarray(out).ravel()


class TLearner(_CausalmlMeta):
    def __init__(self, fast: bool = False, seed: int = LEARNER_SEED):
        super().__init__("t_learner", BaseTRegressor(learner=_lgbm(fast, seed)),
                         p=None)


class XLearner(_CausalmlMeta):
    needs_propensity = True
    predicts_with_propensity = True

    def __init__(self, p: float, fast: bool = False, seed: int = LEARNER_SEED):
        super().__init__("x_learner", BaseXRegressor(learner=_lgbm(fast, seed)),
                         p=p)


class DRLearner(_CausalmlMeta):
    needs_propensity = True

    def __init__(self, p: float, fast: bool = False, seed: int = LEARNER_SEED):
        super().__init__("dr_learner", BaseDRRegressor(learner=_lgbm(fast, seed)),
                         p=p)


class CausalForestLearner:
    name = "causal_forest"

    def __init__(self, fast: bool = False, seed: int = LEARNER_SEED):
        self._model = CausalForest(
            n_estimators=20 if fast else 100,
            min_samples_leaf=20 if fast else 100,
            criterion="het", honest=True, random_state=seed, n_jobs=-1,
        )

    def fit(self, X, t, y):
        self._model.fit(np.asarray(X, dtype="float64"), np.asarray(t),
                        np.asarray(y, dtype="float64"))
        return self

    def predict_cate(self, X) -> np.ndarray:
        return np.asarray(
            self._model.predict(np.asarray(X, dtype="float64"))
        ).ravel()


def default_learners(p: float, fast: bool = False,
                     seed: int = LEARNER_SEED) -> list:
    """The four Stage 2 learners in fixed order (A1 configs)."""
    return [
        TLearner(fast, seed),
        XLearner(p, fast, seed),
        DRLearner(p, fast, seed),
        CausalForestLearner(fast, seed),
    ]


def cross_fit_cate(learner_factory, X, t, y, folds) -> np.ndarray:
    """Out-of-fold CATE predictions: for each fold k, fit a fresh learner on
    the other folds and predict on fold k. `learner_factory()` must return
    an unfitted learner; folds come from `how_wrong.folds.assign_folds`."""
    X = np.asarray(X, dtype="float64")
    t = np.asarray(t)
    y = np.asarray(y, dtype="float64")
    folds = np.asarray(folds)
    oof = np.full(len(y), np.nan)
    for k in np.unique(folds):
        train, test = folds != k, folds == k
        model = learner_factory().fit(X[train], t[train], y[train])
        oof[test] = model.predict_cate(X[test])
    assert not np.isnan(oof).any()
    return oof
