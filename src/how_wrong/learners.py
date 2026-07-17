"""Uniform CATE-learner interface (Stage 2).

Planned API — every learner exposes:
    fit(X, treatment, y) -> self
    predict_cate(X) -> np.ndarray of per-unit treatment-effect estimates

Backends: causalml meta-learners (T/X/DR with LightGBM base learners) and
econml CausalForest, wrapped so notebooks and tests never touch library
internals directly. The Stage 2 synthetic-recovery test disqualifies any
learner that cannot recover a known CATE within tolerance.
"""

from __future__ import annotations
