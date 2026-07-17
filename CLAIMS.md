# Claims register

Every headline number reported in the README, report, or resume bullets gets
a row here. "Asserted in" means an executable check (notebook assert or
pytest) that fails if the number drifts.

| # | Claim | Value | Computed in | Asserted in | Status |
| --- | ----- | ----- | ----------- | ----------- | ------ |
| C1 | Criteo v2.1 row count | 13,979,592 | data download (2026-07-16) | `tests/test_gate_stage0.py` | verified |
| C2 | Hillstrom row count | 64,000 | data download (2026-07-16) | `tests/test_gate_stage0.py` | verified |
| C3 | Full-data conversion ATE (ITT) | +0.1152 pp, 95% CI [+0.1085, +0.1219], p=3.2e-246 | `scripts/01_make_dev_subsample.py` | `tests/test_gate_stage1.py::test_headline_numbers_match_claims` | verified |
| C4 | Full-data visit ATE (ITT) | +1.0342 pp, 95% CI [+1.0056, +1.0629] | `scripts/01_make_dev_subsample.py` | `tests/test_gate_stage1.py::test_headline_numbers_match_claims` | verified |
| C5 | Covariate balance (full data) | max abs SMD = 0.0488 < 0.1 | `scripts/01_make_dev_subsample.py` | `tests/test_gate_stage1.py`, notebook 01 | verified |
| C6 | 1M dev subsample representative | ATEs inside full CIs; max cov-mean abs z = 1.83 < 4 | `scripts/01_make_dev_subsample.py` | `tests/test_gate_stage1.py`, notebook 01 | verified |
| C7 | Conversion MDE (α=.05, power .8, 85/15) | 0.0345 pp at 1M rows; 0.0092 pp at 13.98M | `how_wrong.ate.mde_two_proportions` | notebook 01, `test_headline_numbers_match_claims` | verified |
| C8 | H1: treatment-effect heterogeneity exists (visit, DR proxy, BLP per A1) | β₂ = 0.384 (se 0.039), p = 9.9e-23; survives Holm α/3 | `scripts/02_fit_cate.py` | `tests/test_gate_stage2.py`, notebook 02 | verified (raw; final Holm in Stage 5) |
| C9 | Learner ranking, visit Qini (OOF, dev 1M, B=500 CIs) | causal forest 0.00298 [0.00261, 0.00337] > DR 0.00206 > X 0.00204 > T 0.00183 | `scripts/02_fit_cate.py` | notebook 02, `test_metrics_recomputable_from_oof` | verified |
| C10 | GATES decile gradient (DR, visit) | decile 10: +4.99 pp [4.26, 5.72] vs decile 1: +1.53 pp [0.93, 2.12] | `scripts/02_fit_cate.py` | notebook 02 | verified |
| C11 | Criteo raw file is treatment-block-ordered (50k single-arm spans) — rank metrics require non-positional tie-breaking | methodological finding | probed 2026-07-17 | `test_rank_metrics_immune_to_treatment_ordered_ties` | verified |
| C12 | H3(a): naive observational ATE at γ\* = 1.0 (visit) | bias +6.08 pp [+6.06, +6.09] = **5.9× the true effect**; p = 4.1e-45 | `scripts/03_how_wrong.py` | `tests/test_gate_stage3.py`, notebook 04 | verified (raw) |
| C13 | H3(b): AIPW recovery of the confounding gap at γ\* | 93.7% [92.1%, 95.3%]; p_H3 = 4.5e-29 (IUT) | `scripts/03_how_wrong.py` | `tests/test_gate_stage3.py`, notebook 04 | verified (raw; Holm in Stage 5) |
| C14 | Selection on outcome (M2) is unrepairable: at p_drop = 0.5 every estimator biased | AIPW bias +2.58 pp (2.5× true ATE), CI excludes 0 | `scripts/03_how_wrong.py` | notebook 04, gate | verified |
| C15 | Hidden confounder kills the repair: AIPW denied f9/f8/f4 at γ\* | bias +2.48 pp vs +0.38 pp with them (6.5×) | `scripts/03_how_wrong.py` | `test_hidden_confounder_breaks_repair` | verified |
| C16 | Benchmark ambiguity: adjusted estimators sit ≈ −0.3 pp below raw diff-in-means at severity 0 — v2.1's merged sub-experiments make assignment mildly covariate-predictable (ê ∈ [0.64, 0.98], corr(ê, y) = +0.21) | limitation of the truth definition, bounded < 10% of γ\* naive bias | diagnosis 2026-07-17 | `test_severity_zero_placebo`, notebook 04 | verified |
