# How Wrong Without the Experiment?

*Causal ML on the Criteo uplift RCT — pricing the bias of observational
methods against randomized ground truth, then stress-testing the uplift
playbook. All numbers below trace to a CLAIMS.md row and are asserted by
executable gates (`uv run pytest -m gate_stageN`).*

## 1. Question and design

Companies rarely have experiments; they have logs. This project uses a
13,979,592-user randomized ad experiment (Criteo uplift v2.1) as an answer
key: corrupt the RCT into the observational datasets a company would
actually have, re-estimate the treatment effect with the standard toolkit,
and measure — exactly — how wrong each method is. Two more acts follow:
formal heterogeneity inference, and a budget-constrained targeting policy.

Design discipline: three primary hypotheses pre-registered before any
estimation (dated amendments A1–A3 fixed test details before each stage
ran), Holm-corrected as a family; all development on a certified 1M-row
subsample; fold membership a pure hash of row id; every headline number
pinned by a test that fails on drift.

## 2. Ground truth (Stage 1)

Randomization checks pass (max |SMD| = 0.049, under the 0.1 convention but
see §6). Full-data intention-to-treat effects, Neyman inference:

| Outcome | ATE | 95% CI | p |
| ------- | --- | ------ | - |
| conversion | +0.1152 pp | [+0.1085, +0.1219] | 3.2e-246 |
| visit | +1.0342 pp | [+1.0056, +1.0629] | ≈ 0 |

Power analysis at the 0.19% control conversion rate: MDE 0.0345 pp at 1M
rows, 0.0092 pp at full scale — the pooled effect is detectable at 1M, but
decile-level contrasts are not, which is why heterogeneity development used
`visit`. The 1M dev subsample is certified representative (ATEs inside
full-data CIs; covariate moments max |z| = 1.83).

## 3. Heterogeneity — H1 supported (Stage 2)

Four learners (T-, X-, DR-learner, causal forest; LightGBM bases) behind
one interface, each required to recover a known CATE on synthetic data
before touching real data (all passed, corr 0.79–0.86). Out-of-fold
evaluation on the dev subsample, visit outcome:

- Qini (bootstrap 95% CIs): causal forest 0.00298 [0.00261, 0.00337] >
  DR 0.00206 ≈ X 0.00204 > T 0.00183; normalized AUUC 0.71–0.84
  (a published campus S-Learner reference on this dataset: 0.64).
- GATES calibration: realized decile effects rise monotonically from
  +1.53 pp [0.93, 2.12] to +4.99 pp [4.26, 5.72]. The proxy orders users
  well but over-disperses: the BLP heterogeneity loading β₂ = 0.384
  (se 0.039) — realized spread is ~0.38× the predicted spread.
- **H1 (pre-registered): β₂ > 0 with p = 9.9e-23 — supported.** Robustness
  (Stage 5): replacing the constant assignment rate with estimated ê
  attenuates the loading to β₂ = 0.27 but leaves it decisively positive
  (p = 1.1e-11, under the strictest Holm threshold) — part of the apparent
  heterogeneity loading reflects the merge artifact, most survives it.

## 4. The headline — H3 supported (Stage 3)

Protocol (A2, registered before estimation): corrupt the RCT by (M1)
covariate-dependent selection on the three most outcome-predictive
features, severity γ; (M2) dropping treated non-responders with
probability p — selection on outcome. Estimators see only the corrupted
data: naive diff-in-means, G-computation, IPW, AIPW (shared cross-fitted
LightGBM nuisances). 20 corruption replicates per cell; truth = the
uncorrupted dev RCT estimate.

At the pre-registered severity γ\* = 1.0, visit outcome:

- **Naive bias +6.08 pp [+6.06, +6.09] — 5.9× the true effect** (which is
  +1.03 pp). A campaign evaluation done this way would report a
  blockbuster where a modest lift exists. p = 4.1e-45.
- **AIPW recovers 93.7% [92.1%, 95.3%] of the gap**; p_H3 = 4.5e-29
  (intersection–union). Strong support: the recovery CI sits entirely
  above the registered 0.5 bar.
- The boundaries, quantified: under M2 (selection on outcome) *every*
  estimator fails — AIPW is still 2.5× wrong at 50% attrition, exactly as
  theory demands. Hide the three confounding features and AIPW's residual
  bias grows 6.5×. Doubly-robust ≠ magic: it repairs what its covariates
  can see, and only that.

## 5. The policy layer — H2 rejected (Stage 4)

The pre-registered test (A3): DR-learner conversion-CATE ranking vs a
"likely buyers" propensity ranking, incremental conversions in the top
k ∈ {10%, 30%}, joint bootstrap, plus an IPW-corrected robustness variant.

**Uplift targeting lost at both budgets**: Δ = −253 [−374, −121] at k=10%
and −329 [−444, −200] at k=30% incremental conversions; the robust variant
agrees. Both policies beat random targeting by 4–6× (k=10%: uplift 489,
propensity 743, random 115).

Why the textbook story fails here: (a) conversion CATEs are
noise-dominated at 1M rows — predicted by the Stage 1 power analysis; and
(b) more fundamentally, the sure-things argument requires *saturated*
customers, and at 0.3–5% base rates there are none — treatment effects
scale with baseline propensity, so "most likely to buy" ≈ "most
persuadable" and the low-variance propensity ranking is near-optimal (it
edges the CATE ranking even on visit, where the CATE signal is strong).
Uplift modelling needs both signal and saturation to pay; this regime has
neither.

## 6. What the answer key taught us about the answer key

Three dataset findings with methodological bite:

1. **The raw file is treatment-block-ordered** (whole 50k-row spans are
   single-arm). Combined with heavily tied tree predictions on rare
   outcomes (~50% of forest conversion CATEs are exactly 0.0), any
   rank-based metric that breaks ties positionally silently correlates
   rank with treatment. Symptoms we caught: NaN GATES deciles, a bootstrap
   CI excluding its own point estimate. Fix: seeded-random tie-breaking,
   enforced by a regression test.
2. **The "RCT" is a pool of sub-experiments** with different treatment
   ratios and covariate mixes: estimated propensities span [0.64, 0.98]
   and correlate +0.21 with the outcome. Covariate-adjusted estimators
   therefore *dissent* from raw diff-in-means by ≈ −0.3 pp at zero
   synthetic confounding — and are partly right to. The full-data AIPW
   companion (Stage 5) quantifies the truth range; the pre-registered
   benchmark (raw diff-in-means) was held fixed throughout.
3. **Holm family verdict:** H3 (p = 4.5e-29) and H1 (p = 9.9e-23) survive
   at α = 0.05; H2 is rejected with the effect opposite the registered
   direction.

## 7. Limitations

- Single RCT, single vertical (display advertising), anonymized features —
  external validity claims stop at "this regime".
- `treatment` is randomized *assignment*; exposure is endogenous — all
  effects are intention-to-treat.
- The ground truth carries ~0.3 pp definitional ambiguity (§6.2); headline
  ratios are robust to it (6.08 pp of injected bias dwarfs it) but
  fine-grained recovery fractions inherit some of it.
- Synthetic confounding is a model of real observational bias, not the
  thing itself; mechanisms and severities were chosen pre-registered, but
  reality mixes mechanisms.
- H2's verdict is regime-specific: rare outcomes, 1M-row training. It is
  a statement about when uplift modelling pays, not that it never does.

## 8. Reproduction

Three tiers (REPRODUCING.md): Tier 1 verifies every claim from committed
artifacts in minutes, no downloads (raw-data tests skip); Tier 2
recomputes from the committed 1M-row parquet; Tier 3 rebuilds from raw
(311 MB, HuggingFace `criteo/criteo-uplift`; historical URLs are dead).
Compute budget: ~7 CPU-hours total on one 8-core box, logged in
`results/stage5/finals.json`.
