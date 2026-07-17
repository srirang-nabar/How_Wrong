# How Wrong Without the Experiment?

Causal ML on the Criteo uplift RCT (13.98M users). The randomized experiment
provides ground truth; we use it to (1) price the bias of observational
methods under controlled synthetic confounding, (2) estimate heterogeneous
treatment effects with honest evaluation, and (3) convert them into a
budget-constrained uplift-targeting policy.

**Status: Stage 1 complete** — randomization verified (max |SMD| 0.0488),
full-data ITT ATEs estimated (conversion +0.115 pp, visit +1.034 pp, CLAIMS
C3–C7), 1M-row dev subsample certified representative, fold discipline in
place. Next: Stage 2 (CATE estimation & honest evaluation).

## Headline results

| Question | Number | Evidence |
| -------- | ------ | -------- |
| How biased is the naive observational ATE? (H3) | *pending Stage 3* | |
| How much does doubly-robust correction recover? (H3) | *pending Stage 3* | |
| Does treatment-effect heterogeneity exist? (H1) | *pending Stage 2* | |
| CATE targeting vs propensity targeting at k=10%/30% (H2) | *pending Stage 4* | |
| Learner comparison (AUUC, ref: 0.64 S-Learner) | *pending Stage 2* | |

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
