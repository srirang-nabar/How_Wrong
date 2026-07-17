# Pre-registered hypotheses

**Registered: 2026-07-17, before any estimation.** Primary hypotheses are
tested at α = 0.05 with Holm correction across H1–H3. Everything not listed
as primary is exploratory and reported descriptively, without p-values
dressed up as confirmatory. Amendments to this file are dated and appended,
never edited in place.

**Datasets:** Criteo uplift v2.1 (13,979,592 rows; primary) and Hillstrom
email RCT (64,000 rows; development/secondary). All development runs on
Hillstrom plus a fixed-seed 1M-row Criteo subsample (certified representative
in Stage 1); the full 13.98M-row data is touched only for the pre-registered
final numbers in Stage 5.

**Outcomes:** `conversion` is the primary outcome; `visit` is the
higher-power secondary. In Criteo, `treatment` is the *randomized assignment*
(exposure is endogenous), so all effects are intention-to-treat.

## H1 (primary) — Treatment-effect heterogeneity exists

Chernozhukov et al. BLP/GATES on held-out folds rejects the null of a
constant treatment effect (BLP heterogeneity coefficient p < 0.05 after
Holm), on the Criteo development subsample, `visit` outcome (chosen for
power; `conversion` reported alongside as exploratory).

## H2 (primary) — CATE targeting beats propensity targeting

At budgets k = 10% and k = 30%, ranking by DR-learner CATE yields more
incremental conversions than ranking by a propensity-to-convert model
(outcome model ignoring treatment), evaluated on held-out folds. Verdict:
seeded bootstrap 95% CI of the difference excludes zero at both k points
(each k tested at the Holm-adjusted level for H2).

## H3 (primary, headline) — Naive observational estimation is wrong; DR repairs ≥ half the gap

Under a synthetic-confounding protocol applied to the RCT (mechanisms and
severity grid to be pre-registered by dated amendment to this file at the
start of Stage 3, **before** any confounded estimation runs):

- (a) the naive difference-in-means ATE is biased away from the known RCT
  ATE by a margin detectable at the pre-registered severities (bootstrap CI
  of the bias excludes zero), and
- (b) AIPW/doubly-robust correction recovers at least half of that gap
  (recovery fraction ≥ 0.5, bootstrap CI excluding 0.5 counts as strong
  support; point estimate ≥ 0.5 with CI excluding 0 counts as support).

The exact severity values, mechanisms, and estimator implementations are
fixed in the Stage 3 amendment; nothing in this section may be revised after
the first confounded estimate is computed.

## Exploratory (descriptive only)

- Learner ranking: T- vs X- vs DR-learner vs causal forest by AUUC/Qini
  (external reference point: a published campus project reports AUUC 0.64
  with an S-Learner on this dataset — context, not a target).
- CLAN feature descriptions of the most/least responsive groups.
- Sensitivity analysis: how much unobserved confounding breaks DR.
- Hillstrom results (all of them — it is the development dataset).

## Amendments

### A1 — 2026-07-17: H1 test specification (registered before any CATE estimation)

Fixed before `scripts/02_fit_cate.py` first runs:

- **Proxy:** the CATE proxy S for the H1 BLP test is the **DR-learner's**
  out-of-fold prediction (LightGBM base learners), computed via 5-fold
  cross-fitting on the hashed folds of Stage 1 (seed 20260717, fingerprint
  in `subsample_cert.json`). Other learners' BLP results are exploratory.
- **Outcome:** `visit` (as registered); `conversion` reported as exploratory.
- **Test:** Chernozhukov et al. BLP — OLS of Y on
  {1, S−S̄, (T−p), (T−p)(S−S̄)} with p = 0.85 (the known assignment rate)
  and HC3 robust standard errors, pooled across held-out folds. H1 is the
  two-sided test of the (T−p)(S−S̄) coefficient β₂ = 0; support requires
  β₂ > 0 with p < 0.05 after Holm across H1–H3.
- **Uncertainty:** all bootstrap CIs use B = 500, seed 20260717,
  row-level resampling.
- **Learner configs:** LightGBM (300 trees, lr 0.05, 63 leaves,
  min_child_samples 200); causal forest 100 honest trees. Fixed here — no
  tuning against evaluation metrics.

### A2 — 2026-07-17: Stage 3 confounding protocol (registered before any confounded estimation)

**Ground truth.** The diff-in-means RCT estimate on the *uncorrupted* 1M-row
dev subsample, per outcome. Bias of an estimator = its estimate on the
corrupted data − this truth. Primary outcome `visit`; `conversion`
exploratory.

**Confounder score.** z = standardize(std(f9) − std(f8) + std(f4)),
standardized on the uncorrupted dev subsample. Selection rule (fixed now):
the three features with the largest |corr| with `visit` in the *control arm*
of the dev subsample (measured 2026-07-17: f9 +0.48, f8 −0.43, f4 +0.27),
signed so z positively predicts visit. This deviates from the plan sketch
("confound on f0–f2") deliberately: those features are nearly
outcome-irrelevant (|corr| ≤ 0.11), so confounding on them would induce no
bias and make H3(a) vacuous. Design choice made before any estimator ran.

**Mechanism M1 — confounded assignment (selection on observables; the H3
mechanism).** From the RCT, keep treated unit i w.p. sigmoid(γ·z_i) and
control unit i w.p. sigmoid(−γ·z_i). γ = 0 is an unconfounded 50% thinning.
Severity grid γ ∈ {0, 0.25, 0.5, 1.0, 1.5, 2.0}; **primary severity
γ\* = 1.0**. The confounder is observed (f9, f8, f4 are in X), so
correction is possible in principle — this mechanism tests whether the
estimators deliver it.

**Mechanism M2 — selective attrition (selection on outcome; robustness
only).** Drop treated units with y = 0 w.p. p_drop ∈ {0, 0.1, 0.25, 0.5}
(y = the analyzed outcome). No covariate-based estimator can repair this by
construction; it is reported as the "what nothing fixes" arm and is NOT
part of the H3 adjudication.

**Estimators** (given only the corrupted dataset and all 12 features):
naive diff-in-means; outcome regression (G-computation, separate LightGBM
μ₁/μ₀); IPW (Hájek-normalized, LightGBM-classifier propensity); AIPW.
Nuisances shared across OR/IPW/AIPW and 2-fold cross-fitted (fold =
splitmix64(row_id, seed 20260717) mod 2); propensities clipped to
[0.01, 0.99]. Nuisance config: LightGBM 100 trees, lr 0.1, 31 leaves,
min_child_samples 200 — fixed here.

**Replication & inference.** R = 20 corruption replicates per
(mechanism, severity) cell, seeds spawned from SeedSequence(20260717).
Report mean bias with 95% t-CIs across replicates.

**H3 adjudication** at (M1, γ\* = 1.0, visit), following H3(a)/(b):

- (a) naive bias ≠ 0: two-sided one-sample t-test across replicates,
  plus materiality |mean bias| ≥ 0.25·|true ATE|;
- (b) recovery per replicate ρ = 1 − |bias_AIPW| / |bias_naive|; support =
  mean ρ ≥ 0.5 with one-sided t-test ρ > 0 at p < 0.05; strong support =
  95% CI for ρ entirely above 0.5.
- H3 p-value for the Holm family = max(p_a, p_b) (intersection–union test).

**Exploratory arms** (descriptive): hidden-confounder variant at γ\* —
estimators denied f9, f8, f4 (the unobserved-confounding sensitivity
story); conversion outcome at γ\*; OR and IPW individually across the grid.

### A3 — 2026-07-17: H2 policy-evaluation specification (registered before any policy evaluation)

Fixed before `scripts/04_policy.py` first runs:

- **Uplift score:** the DR-learner's out-of-fold **conversion** CATE from
  Stage 2 (committed in `results/stage2/oof_cate.parquet`) — conversion is
  the outcome H2 was registered on. No refitting, no learner substitution.
- **Propensity-targeting baseline:** LightGBM predicting conversion from
  the 12 features *ignoring treatment*, out-of-fold on the same 5 hashed
  folds, A1 full config (300 trees, lr 0.05, 63 leaves, mcs 200) — the
  industry-standard "target likely buyers" model.
- **Metric at budget k:** targeted set = top k% by score, ties broken by
  fixed-seed jitter (Stage 2 convention). Estimated incremental conversions
  = within-targeted-set diff-in-means × |targeted set|. **Primary** uses
  the plain diff-in-means (consistent with the A2 truth convention);
  **robustness (per C16)** repeats everything with Hájek/IPW-weighted
  within-set means using 2-fold cross-fitted ê (A2 nuisance config, clip
  [0.01, 0.99]). If primary and robust verdicts disagree, both are
  reported and H2 is adjudicated on the primary, with the disagreement
  flagged.
- **Test:** Δ(k) = incremental(CATE-ranked) − incremental(propensity-
  ranked), evaluated at the registered k ∈ {10%, 30%} on the dev
  subsample. Row-level bootstrap (B = 500, seed 20260717, joint resample —
  both policies recomputed per resample including re-thresholding);
  two-sided percentile CI and bootstrap p per k. Support for H2 requires
  Δ > 0 with CI excluding 0 at **both** k points; family p_H2 =
  max(p_10, p_30) (intersection–union, as for H3).
- **Exploratory (descriptive):** the full k-grid three-curve chart
  (CATE vs propensity vs analytic random baseline = k × total incremental);
  visit-outcome analogue; ROI translation at stated assumptions
  (value per conversion $40, cost per targeted user $0.15) with a
  sensitivity grid over both — assumptions are illustrative, clearly
  labelled, and not part of H2.
