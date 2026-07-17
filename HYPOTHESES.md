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

*(none yet)*
