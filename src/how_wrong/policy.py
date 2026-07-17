"""Budget-constrained targeting policy (Stage 4).

Planned API:
    incremental_conversions(y, treatment, scores, k_grid) — incremental
        conversions when treating the top-k% by score, vs random and vs
        propensity-model targeting (the three-curve money chart).
    roi_table(..., cost_per_treatment, value_per_conversion) with
        sensitivity of the optimal k to the cost/value assumptions.
"""

from __future__ import annotations
