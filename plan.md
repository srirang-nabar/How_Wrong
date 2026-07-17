# Ad Uplift & Observational Bias Analysis (Causal ML, Uplift Modelling) — Staged Implementation Plan

*a.k.a. How Wrong Without the Experiment?*

Causal ML on the Criteo uplift RCT (13.98M users, verified locally): use the randomized experiment as ground truth to price the bias of observational methods, then convert treatment-effect heterogeneity into a budget-constrained uplift-targeting policy. Full rationale: [../placement_projects/06_causal_uplift.md](../placement_projects/06_causal_uplift.md).

**How to use this file:** staged work orders with executable gates (`uv run pytest -m gate_stageN` + stage notebook runs top-to-bottom). Record headline numbers in the Results Log.

**Ground rules (every stage):**

- `uv` only; no git commits/pushes — the user handles version control.
- `data/raw/` is read-only and gitignored; derived artifacts regenerable by numbered scripts; seeds everywhere; configs in `configs/`.
- **Development discipline:** all iteration on Hillstrom (64k rows, committed-size) and a fixed 1M-row Criteo subsample; full-data runs only for pre-registered final numbers.
- Pre-registration in HYPOTHESES.md (dated, before estimation); primary hypotheses Holm-corrected; exploratory reported descriptively.
- External benchmark on record: a comparable published campus project reached **AUUC 0.64 with an S-Learner** on this dataset — a reference point for the model-comparison table (report as context, not as the goal).

## Reproducibility & Verification Charter

- **Tier 1 (≤10 min, no downloads):** notebooks run on committed derived artifacts (fold summaries, curves, fitted-model metadata) and assert every headline number vs `CLAIMS.md`.
- **Tier 2 (≤1 hr):** recompute all analysis from the committed 1M-row development subsample (parquet, committed if <100 MB compressed — else hash-manifested with a one-command fetch).
- **Tier 3 (hours, 311 MB download):** full-data reproduction from raw. Raw file + SHA256 already in `data/raw/` (verified 2026-07-16: gzip integrity OK; 13,979,592 rows; schema `f0..f11, treatment, conversion, visit, exposure`).
- `CLAIMS.md` number→evidence map; `results/MANIFEST.sha256`; fresh-machine gate at the end.

**Data status (probed & downloaded 2026-07-16):**

- ✅ Criteo uplift v2.1: 311 MB via HuggingFace (`criteo/criteo-uplift`), downloaded to `data/raw/` with SHA256 sidecar; old `go.criteo.net` and Azure URLs are dead (404) — provenance documented here so volunteers use the HF path.
- ✅ Hillstrom email RCT: downloaded to `data/raw/hillstrom.csv` (64k rows, SHA256 sidecar, schema verified).

**Notebook plan:** `01_rct_foundations.ipynb` (balance, ATE, power), `02_cate_estimators.ipynb` (learner comparison, Qini/AUUC with CIs, calibration), `03_policy.ipynb` (budget-constrained targeting, the three-curve money chart), `04_how_wrong.ipynb` (the headline: synthetic confounding → bias → partial repair).

## Stage 0 — Environment & protocol

- [x] `uv add pandas numpy scipy scikit-learn lightgbm econml causalml matplotlib pyarrow` and `uv add --dev pytest hypothesis jupyter nbconvert` (note: causalml build can be finicky — if it fights, econml + hand-rolled meta-learners suffice; decide here, document)
  - **Decision (2026-07-17): both causalml 0.17.0 and econml 0.16.0 install and import cleanly on Python 3.13** — no hand-rolling needed. Required pins: `scikit-learn<1.7` (econml ceiling) and `numpy<2.4` (else the resolver backtracks shap→numba to an unbuildable numba 0.53). Pins recorded in pyproject.toml; `uv sync --frozen` verified in a scratch copy.
- [x] Skeleton `src/how_wrong/`: `data.py`, `balance.py`, `learners.py`, `evaluate.py` (Qini/AUUC + CIs), `policy.py`, `confound.py`, `reproduce.py`
- [x] Repro skeleton: REPRODUCING.md, CLAIMS.md, manifest helper, pytest markers
- [x] HYPOTHESES.md (dated) — pre-register:
  - **H1 (primary):** treatment-effect heterogeneity exists (Chernozhukov et al. BLP/GATES p < 0.05 after Holm)
  - **H2 (primary):** CATE-ranked targeting beats propensity-model targeting on incremental conversions at k = 10%/30% budget (bootstrap CI excluding zero)
  - **H3 (primary, the headline):** under the pre-specified synthetic-confounding protocol, naive estimation biases the ATE by a detectable margin and doubly-robust correction recovers ≥ half the gap (exact numbers pre-registered in Stage 3 before running)
  - Exploratory: learner ranking (T/X/DR/causal forest), feature-level CLAN descriptions
- [x] README stub + empty headline table

**Gate passed 2026-07-17:** `uv run pytest -m gate_stage0` — 5 passed (imports, checksums, both schemas, full 13,979,592-row count); `uv sync --frozen` verified in scratch copy. Also fixed en route: both `*.sha256` sidecars had non-portable paths (absolute / wrong-relative) and would not have verified on a fresh machine — rewritten filename-relative.

**Gate (`gate_stage0`):** `uv sync --frozen` in scratch copy; data-loading smoke test reads both raw files and validates schemas + row counts against this plan's recorded values.

## Stage 1 — RCT foundations (week 1)

- [x] Covariate balance: SMDs across arms (both datasets) — the professional's first move; treatment share, outcome rates, the works
- [x] ATE with correct inference (visit + conversion); power analysis for conversion (~0.3% base rate — document what effect sizes are detectable)
- [x] Development subsample: fixed-seed 1M-row stratified Criteo sample → committed/manifested parquet; **subsample-representativeness test** (ATE and covariate moments within CI of full data)
- [x] Coding tests: loader schema contracts; sampling determinism; no train/test leakage in the split utilities (hash-fingerprinted fold assignments, code-guarded like Deep_Hedging's TEST pool)
- [x] `notebooks/01_rct_foundations.ipynb`

**Gate (`gate_stage1`):** balance verified (randomization holds); subsample certified representative; fold discipline in place.

**Gate passed 2026-07-17:** `uv run pytest -m gate_stage1` — 10 passed; notebook 01 executes top-to-bottom with all asserts. Artifacts: `data/derived/criteo_dev_1m.parquet` (9.9 MB zstd — small enough to commit outright) + sidecar, `results/stage1/{subsample_cert.json, balance_criteo_full.csv, arm_summary_full.csv}`, CLAIMS C3–C7. Notes: (1) max full-data |SMD| = 0.0488 — under the 0.1 threshold but not textbook-tiny; consistent with v2.1 being a merge of several incrementality tests — carry as a limitation in the report; (2) pooled conversion ATE is detectable even at 1M rows (power ≈ 100%), the visit-for-development rationale rests on decile-level power, where the 1M MDE rivals the whole effect.

## Stage 2 — CATE estimation & honest evaluation (weeks 2–3)

- [x] Meta-learners: T-, X-, DR-learner (gradient-boosting base learners); causal forest (econml); uniform interface in `learners.py`
- [x] Evaluation: Qini and uplift curves on held-out folds with seeded bootstrap CIs; AUUC comparison table (the external 0.64 reference lands here); decile-level GATES calibration (predicted vs realized group effects)
- [x] Heterogeneity inference (H1): Chernozhukov et al. BLP/GATES/CLAN on held-out folds
- [x] Coding tests: learner interface contracts; CI determinism under seed; a **synthetic-data recovery test** — on simulated data with known CATE, every learner must recover it within tolerance (the calibration certificate of this project)
- [x] `notebooks/02_cate_estimators.ipynb` asserting the comparison table

**Gate (`gate_stage2`, hard):** synthetic-recovery test green for every learner (a learner that can't recover a known CATE disqualifies itself); H1 verdict recorded with corrected p-values.

**Gate passed 2026-07-17:** 13 passed — all four learners pass synthetic recovery (corr 0.79–0.86); **H1 supported**: β₂ = 0.384 (se 0.039), p = 9.9e-23 (C8). Learner ranking C9 (forest first, AUUC 0.71–0.84 vs external 0.64). Findings en route: (1) **the raw file is treatment-block-ordered** — positional tie-breaking corrupts rank metrics (NaN GATES deciles, bootstrap-inconsistent Qini); fixed with seeded-random tie-breaking + regression test (C11); (2) LightGBM `n_jobs=-1` pathological on this box (42s vs 0.08s) — threads pinned; (3) H1 test spec pre-registered as amendment A1 before estimation; (4) DR proxy over-disperses at the extremes (GATES slope ≈ β₂ < 1) — orders users well, absolute CATEs shrink toward the mean; relevant for Stage 4 policy framing.

## Stage 3 — The headline: how wrong without the experiment? (weeks 3–4)

- [x] **Pre-register the confounding protocol before running** (append to HYPOTHESES.md): the biasing mechanisms (e.g., drop treated non-converters with probability p; confound assignment on f0–f2), severity grid, and the exact repair estimators (naive diff-in-means, outcome regression, IPW, AIPW/DR)
  - Registered as amendment A2. Deviation from the sketch, made *before* estimation: confound on f9/f8/f4 (the outcome-predictive features), not f0–f2 (|corr| ≤ 0.11 — would have made H3 vacuous).
- [x] Implement `confound.py`: RCT → synthetic observational datasets, mechanism + severity parameterized, seeded
- [x] The experiment: true ATE (known) vs naive vs corrected estimates across the severity grid; bias curves with CIs; "recovery fraction" per estimator
- [x] Robustness: results across both mechanisms; sensitivity-analysis framing (how much unobserved confounding would fully break DR?)
- [x] `notebooks/04_how_wrong.ipynb` — the headline chart: bias vs severity, three estimator lines, ground truth at zero
- [x] H3 verdict recorded

**Gate (`gate_stage3`):** headline row filled with Holm-corrected verdicts; every number asserted.

**Gate passed 2026-07-17:** 11 passed. **H3 supported (raw, strong):** at γ\* = 1.0 the naive ATE is wrong by **+6.08 pp = 5.9× the true effect** (C12); AIPW recovers **93.7% [92.1, 95.3]** of the gap (C13); p_H3 = 4.5e-29 — with H1 (9.9e-23), both already survive any Holm ordering at α = 0.05; formal adjudication after H2. M2 (selection on outcome) is unrepairable as theory demands (C14); hiding the confounders sextuples AIPW's bias (C15). **Deepest finding (C16):** at severity 0 the adjusted estimators dissent from raw diff-in-means by ≈ −0.3 pp and are partly *right* to — v2.1's merged sub-experiments make assignment mildly covariate-predictable (ê ∈ [0.64, 0.98], corr(ê, y) = +0.21; cf. Stage 1's max SMD 0.049), so the "RCT ground truth" itself carries ~0.3 pp definitional ambiguity. Report limitation, and a core interview talking point.

## Stage 4 — The policy layer (week 5)

**Adjustment after C16 (2026-07-17):** the within-top-k incremental-conversion metric assumes treatment is random within any selected subgroup — only approximately true given the merge artifact (ê ∈ [0.64, 0.98]). The Stage 4 pre-registration (amendment A3, before any policy evaluation runs) must specify: plain within-subset diff-in-means as the primary H2 metric (consistent with the A2 truth convention) **plus an IPW-corrected variant (estimated ê) as pre-registered robustness**. Disagreement between them is reported, not suppressed.

- [x] Budget-constrained targeting: rank by CATE, treat top-k%; incremental conversions vs (a) random, (b) propensity-model targeting, across the full k-grid — the three-curve money chart
- [x] ROI translation with stated cost/value assumptions + sensitivity of optimal k
- [x] H2 verdict (pre-registered k points, bootstrap CIs)
- [x] `notebooks/03_policy.ipynb`

**Gate (`gate_stage4`):** H2 adjudicated; money chart committed.

**Gate passed 2026-07-17:** 7 passed. **H2 REJECTED — a clean pre-registered negative (C17):** uplift targeting underperforms propensity targeting on conversion at both registered budgets (Δ = −253 and −329 incremental conversions, CIs excluding zero; IPW-robust variant concurs). Two mechanisms (C18): conversion CATEs are noise-dominated at 1M rows (as Stage 1's power analysis predicted), and — deeper — at 0.3–5% base rates there are no saturated "sure things", so uplift scales with baseline propensity and the likely-buyer ranking is a near-optimal, lower-variance uplift ranking (propensity edges the CATE ranking even on visit, where the CATE signal is strong). Both policies beat random by 4–6×. Family status: H1 and H3 survive Holm at α = 0.05 under any ordering; H2 rejected. The honest headline for the report: *uplift modelling needs both signal and saturation to beat the simple baseline — neither exists here.*

## Stage 5 — Full-data finals, write-up & verification pack

- [ ] Full 13.98M-row runs for the pre-registered final numbers only (no new analysis at full scale — that's fishing)
- [ ] **Post-C16 exploratory additions (labelled as such):** (a) covariate-adjusted companion ATE (AIPW, full data) so the report states the truth range — raw +1.03 pp vs adjusted ≈ +0.7 pp — with the merge-artifact explanation; (b) BLP robustness re-run with estimated ê in place of the constant 0.85 (A1 assumed flat propensity)
- [ ] README headline table; `report/report.md` (house style; limitations: single RCT, ad-exposure treatment nuances, external validity); `report/interview_qa.md` (fundamental problem of causal inference; why AUC ≠ uplift; Neyman orthogonality in the DR-learner; how you'd design the experiment; the sure-things/persuadables quadrant)
- [ ] Fresh-machine dry run (key-less, download-less Tier 1) → `results/fresh_machine_run.log`
- [ ] Resume bullets with real numbers (each a CLAIMS.md row)

**Gate (`gate_stage5`):** fresh-machine Tier 1 passes; budget of full-data runs documented (they're CPU-hours, not money — but log them).

## Known risks & pre-checked facts

| Risk | Status / mitigation |
| --- | --- |
| Dataset link rot | Solved: raw file + SHA256 committed locally; HF provenance documented (old URLs confirmed dead) |
| conversion ~0.3% → noisy CATEs | Power analysis in Stage 1; visit outcome as higher-power secondary; CIs on everything |
| causalml install friction | Decision point in Stage 0 with econml + hand-rolled fallback |
| Fishing across learners × outcomes × k | Primary/exploratory split, Holm on H1–H3, full-scale runs restricted to pre-registered finals |
| Iteration too slow at 14M rows | 1M-row certified subsample for all development |
| v2.1 is a pool of sub-experiments: treatment mildly covariate-predictable (ê ∈ [0.64, 0.98], corr(ê, y) +0.21), file treatment-block-ordered | **Known, quantified (C11, C16).** Seeded tie-breaking in all rank metrics; benchmark ambiguity ~0.3 pp bounded by gate; Stage 4 A3 adds IPW-robust evaluation; report limitation |

## Results Log

| Date | Stage | Headline number | Notes |
| ---- | ----- | --------------- | ----- |
| 2026-07-17 | 0 | gate_stage0: 5 passed | causalml 0.17 + econml 0.16 both fine on py3.13 (pins: sklearn<1.7, numpy<2.4); sha256 sidecars fixed to portable paths |
| 2026-07-17 | 1 | Full ITT ATE: conversion +0.1152 pp [+0.1085, +0.1219]; visit +1.0342 pp [+1.0056, +1.0629] | max \|SMD\| 0.0488; 1M dev subsample certified (max cov \|z\| 1.83, ATEs inside full CIs; 9.9 MB parquet); conversion MDE 0.0345 pp @1M / 0.0092 pp @full; gate_stage1: 10 passed (CLAIMS C3–C7) |
| 2026-07-17 | 2 | **H1 supported (raw):** BLP β₂ = 0.384 (se 0.039), p = 9.9e-23; GATES decile 10 vs 1: +4.99 pp vs +1.53 pp | Forest tops Qini (0.00298 vs DR 0.00206); conversion metrics noisy as power analysis predicted; treatment-block-ordered raw file discovered → seeded tie-breaking (C11); gate_stage2: 13 passed (C8–C11) |
| 2026-07-17 | 3 | **H3 supported (raw, strong): naive off by 5.9× the true ATE at γ\*; AIPW recovers 93.7% [92.1, 95.3]**; p_H3 = 4.5e-29 | M2 unrepairable (AIPW +2.5× at p_drop 0.5); hidden confounder → AIPW bias ×6.5; benchmark ambiguity ~0.3 pp from v2.1 merge (C16); gate_stage3: 11 passed (C12–C16) |
| 2026-07-17 | 4 | **H2 rejected (pre-registered negative): uplift targeting loses to propensity targeting** — Δ = −253 [−374, −121] @ k=10%, −329 [−444, −200] @ k=30%; IPW-robust agrees | Both policies beat random 4–6×; no "sure things" at rare base rates + noisy conversion CATE (C17–C18); gate_stage4: 7 passed |
