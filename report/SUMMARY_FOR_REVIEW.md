# Reviewer summary — Ad Uplift & Observational Bias Analysis (Causal ML, Uplift Modelling)

**One paragraph.** The Criteo uplift dataset is a real randomized ad experiment (13.98M users,
randomized exposure, visit/conversion outcomes). Because randomization gives the *true* causal
effect, the project uses it as ground truth three ways: (1) synthetically corrupt the RCT into
"observational" data and measure exactly how wrong non-experimental estimates are, and how much
doubly-robust correction repairs; (2) estimate *which users* respond (heterogeneous treatment
effects) with formal inference; (3) test whether uplift-based targeting beats standard
"target likely buyers" under a budget. All hypotheses pre-registered (HYPOTHESES.md, dated),
Holm-corrected; every public number is a row in CLAIMS.md tied to a test and a notebook cell.

**Findings (Holm family verdicts, Stage 5):**

| # | Question | Result |
| - | -------- | ------ |
| H3 ✓ | How wrong is naive observational estimation? | Overshoots the true effect **5.9×** at the pre-registered severity; cross-fitted AIPW recovers **93.7% [92.1, 95.3]** of the gap — and provably fails under selection-on-outcome / hidden confounding |
| H1 ✓ | Is treatment-effect heterogeneity real? | Yes — Chernozhukov BLP β₂ = 0.384, p = 9.9e-23; causal forest best of 4 learners (AUUC 0.84 visit / 0.81 conversion) |
| H2 ✗ | Does uplift targeting beat propensity targeting? | **No — pre-registered negative.** Δ = −253 / −329 incremental conversions at k = 10% / 30% (p = 0.002, opposite direction): at 0.3% base rates there are no "sure things" to skip |

**How to review quickly (~5 min):**

1. Open `notebooks/00_review_walkthrough.ipynb` — commented, pre-executed; loads only committed
   results and reproduces every number the resume points cite, with a resume-point → evidence map
   in the last cell.
2. Optional deeper check: `uv sync --frozen && uv run pytest -q` (52 tests, ~40 s), or the tiered
   verification in `REPRODUCING.md` (fresh-machine Tier 1 log: `results/fresh_machine_run.log`, 2m19s).

**Scope honesty:** one RCT, ad-exposure treatment; synthetic (not natural) confounding, mechanisms
pre-registered before estimation; AUUC reference value from a published campus project may use a
different normalization (directional context only).
