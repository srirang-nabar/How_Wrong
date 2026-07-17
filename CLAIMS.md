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
