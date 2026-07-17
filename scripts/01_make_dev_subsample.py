"""Stage 1 full-data pass (Tier 3): balance, ATEs, and the certified
1M-row development subsample.

Reads the full 13.98M-row raw file once; writes:
  data/derived/criteo_dev_1m.parquet (+ .sha256)   — the dev subsample
  results/stage1/balance_criteo_full.csv           — full-data SMD table
  results/stage1/arm_summary_full.csv              — arm shares/outcome rates
  results/stage1/subsample_cert.json               — representativeness cert
  results/MANIFEST.sha256                          — refreshed

Run: uv run python scripts/01_make_dev_subsample.py   (~5 min, ~3 GB RAM)
"""

import json
import time
from datetime import date

import numpy as np

from how_wrong import data
from how_wrong.ate import diff_in_means, mde_two_proportions, power_two_proportions
from how_wrong.balance import arm_summary, smd_table
from how_wrong.folds import FOLD_SEED, N_FOLDS, assign_folds, fold_fingerprint
from how_wrong.reproduce import RESULTS_DIR, write_manifest

STAGE1_DIR = RESULTS_DIR / "stage1"
OUTCOMES = ["conversion", "visit"]

t0 = time.time()
print("loading full criteo ...", flush=True)
full = data.load_criteo()
full.insert(0, "row_id", np.arange(len(full), dtype="int64"))
assert len(full) == data.CRITEO_N_ROWS
print(f"  {len(full):,} rows in {time.time() - t0:.0f}s", flush=True)

STAGE1_DIR.mkdir(parents=True, exist_ok=True)
data.DATA_DERIVED.mkdir(parents=True, exist_ok=True)

print("full-data balance + ATEs ...", flush=True)
smd_full = smd_table(full, "treatment", data.CRITEO_FEATURES)
smd_full.to_csv(STAGE1_DIR / "balance_criteo_full.csv")
arms_full = arm_summary(full, "treatment", OUTCOMES)
arms_full.to_csv(STAGE1_DIR / "arm_summary_full.csv")
ate_full = {y: diff_in_means(full[y], full["treatment"]) for y in OUTCOMES}
cov_full_mean = full[data.CRITEO_FEATURES].mean()
cov_full_sd = full[data.CRITEO_FEATURES].std(ddof=1)

print("building 1M-row stratified subsample ...", flush=True)
dev = data.make_dev_subsample(full)
assert len(dev) == data.DEV_N_ROWS
n1_full, n0_full = ate_full["conversion"].n_treated, ate_full["conversion"].n_control
del full

dev.to_parquet(data.DEV_PARQUET, compression="zstd", index=False)
digest = data.sha256_of(data.DEV_PARQUET)
data.DEV_PARQUET.with_name(data.DEV_PARQUET.name + ".sha256").write_text(
    f"{digest}  {data.DEV_PARQUET.name}\n"
)
size_mb = data.DEV_PARQUET.stat().st_size / 2**20
print(f"  {data.DEV_PARQUET.name}: {size_mb:.1f} MB, sha256 {digest[:12]}...",
      flush=True)

print("representativeness checks ...", flush=True)
ate_dev = {y: diff_in_means(dev[y], dev["treatment"]) for y in OUTCOMES}
ate_ok = {
    y: bool(ate_full[y].ci_lo <= ate_dev[y].ate <= ate_full[y].ci_hi)
    for y in OUTCOMES
}
cov_z = ((dev[data.CRITEO_FEATURES].mean() - cov_full_mean)
         / (cov_full_sd / np.sqrt(len(dev))))
max_abs_z = float(cov_z.abs().max())

folds = assign_folds(dev["row_id"].to_numpy())
fp = fold_fingerprint(folds)

p0 = ate_full["conversion"].mean_control
share1 = n1_full / (n1_full + n0_full)
def mde_at(n):
    return mde_two_proportions(p0, int(n * share1), int(n * (1 - share1)))
mde = {
    "p0_control_conversion": p0,
    "treated_share": share1,
    "mde_abs_at_1m": mde_at(data.DEV_N_ROWS),
    "mde_abs_at_full": mde_at(data.CRITEO_N_ROWS),
    "power_for_full_ate_at_1m": power_two_proportions(
        ate_full["conversion"].ate, p0,
        int(data.DEV_N_ROWS * share1), int(data.DEV_N_ROWS * (1 - share1))),
}

cert = {
    "created": date.today().isoformat(),
    "seed": data.DEV_SEED,
    "strata": data.DEV_STRATA,
    "criteria": {
        "smd_full_max_lt": 0.1,
        "cov_mean_abs_z_lt": 4.0,
        "ate_dev_within_full_ci": True,
    },
    "full": {
        "n": data.CRITEO_N_ROWS,
        "smd_max_abs": float(smd_full["abs_smd"].max()),
        "ate": {y: r.to_dict() for y, r in ate_full.items()},
    },
    "dev": {
        "n": data.DEV_N_ROWS,
        "parquet_sha256": digest,
        "parquet_mb": round(size_mb, 1),
        "ate": {y: r.to_dict() for y, r in ate_dev.items()},
        "cov_mean_z_vs_full": {k: float(v) for k, v in cov_z.items()},
        "cov_mean_max_abs_z": max_abs_z,
    },
    "verdicts": {
        "balance_full": bool(smd_full["abs_smd"].max() < 0.1),
        "cov_moments_representative": bool(max_abs_z < 4.0),
        "ate_representative": ate_ok,
    },
    "folds": {"n_folds": N_FOLDS, "seed": FOLD_SEED,
              "dev_fingerprint": fp,
              "dev_counts": np.bincount(folds, minlength=N_FOLDS).tolist()},
    "mde": mde,
}
(STAGE1_DIR / "subsample_cert.json").write_text(json.dumps(cert, indent=2) + "\n")
write_manifest()

print(json.dumps(cert["verdicts"], indent=2))
for y in OUTCOMES:
    r = ate_full[y]
    print(f"full ATE {y}: {r.ate:+.5%} (95% CI {r.ci_lo:+.5%}..{r.ci_hi:+.5%}, "
          f"p={r.p_value:.2e})")
print(f"max |SMD| full: {smd_full['abs_smd'].max():.4f} | "
      f"max cov |z|: {max_abs_z:.2f} | done in {time.time() - t0:.0f}s")
