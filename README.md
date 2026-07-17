# How Wrong Without the Experiment?

Causal ML on the Criteo uplift RCT (13.98M users). The randomized experiment
provides ground truth; we use it to (1) price the bias of observational
methods under controlled synthetic confounding, (2) estimate heterogeneous
treatment effects with honest evaluation, and (3) convert them into a
budget-constrained uplift-targeting policy.

**Status: Stage 3 complete — the headline is in.** Naive observational
estimation misses the true effect by **5.9×** at the pre-registered
confounding severity; doubly-robust correction recovers **93.7%** of the
gap when the confounders are observed, **none** of it under selection on
outcome, and breaks when the confounder is hidden (CLAIMS C12–C16).
H1 heterogeneity confirmed (C8). Next: Stage 4 (the uplift-targeting
policy layer, H2), then full-data finals.

## Headline results

| Question | Number | Evidence |
| -------- | ------ | -------- |
| How biased is the naive observational ATE? (H3) | **5.9× the true effect** (+6.08 pp on a +1.03 pp ATE) at the pre-registered severity | CLAIMS C12, notebook 04 |
| How much does doubly-robust correction recover? (H3) | **93.7%** [92.1%, 95.3%] of the gap — but 0% of selection-on-outcome, and it breaks if the confounder is unobserved | CLAIMS C13–C15, notebook 04 |
| Does treatment-effect heterogeneity exist? (H1) | **Yes** — BLP β₂ = 0.384, p = 9.9e-23 (raw; Holm pending) | CLAIMS C8, notebook 02 |
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
