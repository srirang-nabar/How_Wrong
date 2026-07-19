# How Wrong Without the Experiment?

Causal ML on the Criteo uplift RCT (13.98M users). The randomized experiment
provides ground truth; we use it to (1) price the bias of observational
methods under controlled synthetic confounding, (2) estimate heterogeneous
treatment effects with honest evaluation, and (3) convert them into a
budget-constrained uplift-targeting policy.

**Status: COMPLETE.** All five stages gated green (51 tests); Holm family
verdict: **H3 supported, H1 supported, H2 rejected**. Headline: naive
observational estimation misses the true effect by **5.9×**; doubly-robust
correction recovers **93.7%** when confounders are observed, none of it
under selection on outcome. The honest surprise: **uplift targeting lost
to "target likely buyers" on the pre-registered test** — at rare base
rates there are no sure things. Full write-up in
[report/report.md](report/report.md); every number is a CLAIMS.md row;
fresh-machine verification in 2m19s (`results/fresh_machine_run.log`).

## Headline results

| Question | Number | Evidence |
| -------- | ------ | -------- |
| How biased is the naive observational ATE? (H3) | **5.9× the true effect** (+6.08 pp on a +1.03 pp ATE) at the pre-registered severity | CLAIMS C12, notebook 04 |
| How much does doubly-robust correction recover? (H3) | **93.7%** [92.1%, 95.3%] of the gap — but 0% of selection-on-outcome, and it breaks if the confounder is unobserved | CLAIMS C13–C15, notebook 04 |
| Does treatment-effect heterogeneity exist? (H1) | **Yes** — BLP β₂ = 0.384, p = 9.9e-23, survives Holm (Stage 5 family) | CLAIMS C8, notebook 02 |
| CATE targeting vs propensity targeting at k=10%/30% (H2) | **Propensity wins** — uplift targeting rejected on the pre-registered test (Δ = −253 and −329 incremental conversions; robust variant agrees) | CLAIMS C17–C18, notebook 03 |
| Learner comparison (normalized AUUC) | **Causal forest best**: 0.84 visit / 0.81 conversion; DR 0.73 / 0.58; X 0.73 / 0.78; T 0.71 / 0.71 (bootstrap CIs in results/stage2/metrics.json; published 0.64 S-Learner reference — normalization may differ, so directional context only) | CLAIMS C9–C11, notebook 02 |

## Layout

- `plan.md` — staged work orders with executable gates
- `HYPOTHESES.md` — dated pre-registration (H1–H3, Holm-corrected)
- `REPRODUCING.md` / `CLAIMS.md` — verification pack
- `src/how_wrong/` — library code; `notebooks/` — the four analysis notebooks
- `data/raw/` — read-only, gitignored; checksums committed as `*.sha256`

## Quick start

```bash
uv sync
uv run pytest -m "gate_stage0 and not slow"
```
