"""Synthetic confounding: RCT -> observational datasets (Stage 3).

The exact mechanisms and severity grid are pre-registered in HYPOTHESES.md
*before* any estimation runs. Planned API:
    confound(df, mechanism, severity, seed) -> biased DataFrame
    mechanisms: selective attrition (drop treated non-converters w.p. p);
        covariate-driven assignment (confound on f0–f2).
    repair estimators (naive diff-in-means, outcome regression, IPW, AIPW)
        evaluated against the known RCT ATE across the severity grid.
"""

from __future__ import annotations
