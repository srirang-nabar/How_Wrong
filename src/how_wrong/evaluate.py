"""Honest uplift evaluation: Qini/uplift curves, AUUC, bootstrap CIs,
GATES-style decile calibration (Stage 2).

Planned API:
    qini_curve(y, treatment, scores) / uplift_curve(...)
    auuc(y, treatment, scores) with seeded bootstrap CIs
    gates_deciles(y, treatment, scores, n_groups=10) — predicted vs realized
        group effects on held-out folds.
All resampling is seeded; CI determinism under a fixed seed is a Stage 2
coding test.
"""

from __future__ import annotations
