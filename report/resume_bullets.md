# Resume bullets (each number = a CLAIMS.md row)

**Ad Uplift & Observational Bias Analysis (Causal ML, Uplift Modelling)** — Python, LightGBM, econml/causalml; 13.98M-user Criteo RCT

## CV-calibrated points (final; ≤4 resume lines total — four one-liners)

1. Used a **13.98M-user Criteo ad RCT** as causal ground truth to price the bias of non-experimental methods
2. Under pre-registered confounding, naive ATEs overshot truth **5.9×**; **AIPW** recovered **93.7%** of the bias
3. Proved effect heterogeneity (**BLP** β₂=0.38, Holm p<1e-21); **causal forest** best of 4 CATE learners, **AUUC 0.84**
4. Adjudicated a **pre-registered negative**: uplift targeting lost to propensity ranking at 0.3% base rates

Reviewer path: `report/SUMMARY_FOR_REVIEW.md` + pre-executed `notebooks/00_review_walkthrough.ipynb`.
Alternate 2+1 packing (if a two-liner is preferred): keep points 1 and 3 as singles; merge 2+4 into one
two-liner: "Under pre-registered confounding naive ATEs overshot truth **5.9×** (AIPW recovered **93.7%**);
uplift targeting lost to propensity ranking at 0.3% base rates — both adjudicated as registered."

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
