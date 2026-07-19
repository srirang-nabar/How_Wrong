# Resume bullets (each number = a CLAIMS.md row)

**Ad Uplift & Observational Bias Analysis (Causal ML, Uplift Modelling)** — Python, LightGBM, econml/causalml; 13.98M-user Criteo RCT

## CV-calibrated points (final; lengths matched to resume line format)

1. Used a 13.98M-user Criteo ad RCT as causal ground truth to price the bias of non-experimental methods
2. Injected pre-registered synthetic confounding: naive ATEs overshot the true ad effect 5.9×; cross-fitted doubly-robust AIPW recovered 93.7% of the bias, failing only under hidden confounders
3. Established effect heterogeneity via Chernozhukov BLP on held-out DR-learner scores (β₂=0.38, Holm p<1e-21); causal forest led 4 CATE learners with bootstrap-validated Qini/AUUC (0.84 visit)
4. Found uplift targeting underperformed propensity targeting at 0.3% base rates (−253/−329 conversions at k=10/30%, p=0.002); reported the pre-registered negative with full policy-curve evidence

Reviewer path: `report/SUMMARY_FOR_REVIEW.md` + pre-executed `notebooks/00_review_walkthrough.ipynb`.

## Long-form bullets (backup detail / interview prep)

- Used a 13.98M-user randomized experiment as causal ground truth to price
  the cost of not experimenting: under pre-registered synthetic
  confounding, naive observational estimates overshot the true ad effect
  by 5.9×, while cross-fitted doubly-robust (AIPW) correction recovered
  93.7% [92.1, 95.3] of the bias — and provably failed under
  selection-on-outcome and hidden-confounder regimes (C12–C15)
- Established treatment-effect heterogeneity with formal inference
  (Chernozhukov BLP β₂ = 0.38, p = 1e-22, Holm-corrected; GATES decile
  effects +1.5 pp → +5.0 pp) after every CATE learner (T/X/DR-learner,
  causal forest) passed a synthetic-recovery calibration gate (C8, C10)
- Adjudicated a pre-registered negative: uplift targeting underperformed
  propensity targeting by 253–329 incremental conversions at 10%/30%
  budgets (IPW-robust variant concurring) and diagnosed why — at sub-1%
  base rates there are no saturated "sure things", so the simple ranking
  is near-optimal (C17–C18)
- Audited the benchmark itself: detected that the public "RCT" pools
  sub-experiments with covariate-predictable assignment (ê ∈ [0.64, 0.98])
  and is treatment-block-ordered on disk — the latter silently corrupts
  any positionally tie-broken uplift metric; shipped seeded tie-breaking
  plus regression tests (C11, C16)
- Reproducibility: pre-registered hypotheses (dated amendments), 3-tier
  reproduction (10-min artifact verification → full 311 MB rebuild),
  hash-fingerprinted folds, every headline number asserted by an
  executable gate; ~7 CPU-hours total
