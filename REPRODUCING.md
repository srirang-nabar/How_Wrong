# Reproducing this project

Three tiers, increasing cost. All commands run from the repo root with
[uv](https://docs.astral.sh/uv/) installed; `uv sync` creates the exact
locked environment (Python 3.13, see `uv.lock`).

## Tier 1 — verify the claims (≤10 min, no downloads)

Runs the notebooks on committed derived artifacts (fold summaries, curves,
fitted-model metadata) and asserts every headline number against
`CLAIMS.md`:

```bash
uv sync --frozen
uv run pytest -m "gate_stage0 and not slow"
uv run jupyter nbconvert --execute --to notebook --inplace notebooks/*.ipynb   # from Stage 1
```

## Tier 2 — recompute the analysis (≤1 hr)

Recomputes everything from the 1M-row development subsample
(`data/derived/criteo_dev_1m.parquet`, committed or hash-manifested with a
one-command fetch — decided in Stage 1).

## Tier 3 — full reproduction from raw (hours, 311 MB download)

Raw data (checksums in `data/raw/*.sha256`):

- **Criteo uplift v2.1** — 311 MB from HuggingFace
  ([`criteo/criteo-uplift`](https://huggingface.co/datasets/criteo/criteo-uplift)).
  The historical `go.criteo.net` and Azure URLs are dead (404, confirmed
  2026-07-16); use the HF path. 13,979,592 rows, schema
  `f0..f11, treatment, conversion, visit, exposure`.
- **Hillstrom email RCT** — `hillstrom.csv`, 64,000 rows
  (MineThatData e-mail analytics challenge).

```bash
sha256sum -c data/raw/*.sha256          # run inside data/raw/
uv run pytest -m gate_stage0            # includes the slow full-file contract test
```

Then run the numbered scripts / stage gates in order
(`uv run pytest -m gate_stageN`).

## Integrity

- `results/MANIFEST.sha256` — hashes of every committed result artifact
  (`how_wrong.reproduce.write_manifest` / `check_manifest`).
- `CLAIMS.md` — the number→evidence map; every headline number states where
  it is computed and where it is asserted.
