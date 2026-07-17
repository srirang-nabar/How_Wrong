# Interview Q&A — How Wrong Without the Experiment?

Prepared answers anchored to the project's own numbers (every figure is a
CLAIMS.md row).

## The fundamental problem of causal inference

**Q: Why can't you just compare users who saw the ad to users who didn't?**
Because you never observe the same user in both states, so you're forced to
compare *different* users — and the users who see ads differ from those who
don't in ways that also drive the outcome. In our synthetic-observational
data this wasn't subtle: the naive comparison read +6.08 pp against a true
+1.03 pp effect — 5.9× too large. Randomization solves this by making the
two groups exchangeable; everything else is an attempt to reconstruct
exchangeability from covariates, which only works if the confounders are
measured (we showed AIPW recovering 93.7% with them, and failing 6.5×
worse without them).

**Q: What does an RCT buy you that adjustment can't?**
An *unconditional* guarantee. Every adjustment method is conditional on
"no unmeasured confounding" — untestable from the data itself. Our M2 arm
makes this concrete: when the missingness depends on the outcome (treated
non-responders vanish from logs), naive, regression, IPW and AIPW are all
biased and there is no covariate fix even in principle.

## Why high AUC ≠ high uplift

**Q: Your conversion model has AUC 0.9. Why not target its top scores?**
Because predicting *who converts* is not predicting *who is changed by
treatment*. The classic quadrant: sure things and lost causes convert (or
don't) regardless — treating them is waste; persuadables are the target;
do-not-disturbs are actively harmed. Uplift metrics (Qini/AUUC) measure
ranking by *incremental* outcome. Nuance from our data: the quadrant trap
requires saturated customers. At a 0.3% base rate nobody is a sure thing,
uplift scales with baseline propensity, and the propensity ranking
actually *beat* our CATE ranking (H2 rejected, Δ = −253 at k=10%) — a
reminder that the textbook argument has preconditions.

## Neyman orthogonality and the DR-learner

**Q: Why prefer a DR-learner / AIPW over a T-learner?**
The AIPW moment is Neyman-orthogonal: its first-order sensitivity to
nuisance errors is zero, so plug-in ML mistakes in μ(x) or e(x) enter only
as a *product* of both errors — you get valid rates if each nuisance
converges at n^(-1/4), and unbiasedness if either one is correct
("doubly robust"). A T-learner difference μ̂₁ − μ̂₀ has no such protection:
each arm's regularization bias lands directly in the estimate, and under
imbalanced arms (85/15 here) the control model is fit on 6× less data. We
saw orthogonality pay in Stage 3: AIPW tracked truth across the severity
grid while naive exploded. We also saw its limits: with the confounder
hidden, orthogonality protects against *estimation* error, not *omission*.

**Q: Why cross-fitting?**
To break the correlation between nuisance overfitting and the estimation
sample — own-observation overfitting otherwise biases the moment. All our
nuisances are cross-fitted on hash-assigned folds (fold = f(row_id), so
membership can't drift between datasets).

## Designing the experiment yourself

**Q: How would you size and run this experiment?**
Start from the decision: what lift changes the go/no-go? Our power
analysis says at a 0.19% base rate and 85/15 split you need ~1M users to
detect the *pooled* effect (MDE 0.035 pp at 80% power) and ~14M before
decile-level heterogeneity is measurable — subgroup claims cost an order
of magnitude more than headline claims. Then: randomize *assignment* and
log it (exposure is endogenous — analyze ITT); balance-check covariates
(SMDs), but also *audit the randomization itself* — our dataset is a merge
of sub-experiments and propensity was covertly predictable (ê ∈ [0.64,
0.98]); watch for interference/spillover (shared budgets, auctions),
novelty effects (run long enough to wash out), and attrition that differs
by arm (our M2 shows why that's fatal).

## Findings I'd volunteer

1. **The benchmark needed auditing** — the "RCT" carried mild real
   confounding from pooling sub-experiments; adjusted estimators dissented
   from raw diff-in-means by −0.3 pp at zero injected confounding and were
   partly right. Ground truth has error bars too.
2. **The file ordering trap** — treatment-block-ordered data plus tied
   scores plus positional tie-breaking corrupted rank metrics; caught via
   a bootstrap CI that excluded its own point estimate. Now a regression
   test.
3. **An honest pre-registered negative** — H2 failed, the robustness
   variant agreed, and the mechanism (no sure things at rare base rates)
   is more informative than a win would have been.
