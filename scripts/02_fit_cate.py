"""Stage 2 fitting pass (Tier 2): out-of-fold CATEs on the dev subsample,
uplift metrics with bootstrap CIs, and the H1 BLP test.

Everything follows HYPOTHESES.md amendment A1: 5 hashed folds, fixed learner
configs, known p = 0.85, B = 500 bootstrap, DR-learner proxy on `visit` as
the primary H1 test. Writes to results/stage2/:

  oof_cate.parquet   — row_id + one OOF-CATE column per outcome × learner
  metrics.json       — qini (with CIs) + causalml AUUC per outcome × learner
  gates_deciles.csv  — decile calibration for every outcome × learner
  clan_dr_visit.csv  — CLAN table for the primary proxy
  h1_blp.json        — the H1 verdict (raw p; Holm applied across H1–H3 later)

Run: uv run python scripts/02_fit_cate.py   (~1.5–3 h CPU, background it)
"""

import json
import time
from datetime import date

import numpy as np
import pandas as pd
from causalml.metrics import auuc_score

from how_wrong import data
from how_wrong.evaluate import (
    blp_test, bootstrap_ci, clan_table, gates_deciles,
)
from how_wrong.folds import assign_folds
from how_wrong.learners import LEARNER_SEED, cross_fit_cate, default_learners
from how_wrong.reproduce import RESULTS_DIR, write_manifest

STAGE2 = RESULTS_DIR / "stage2"
P_ASSIGN = 0.85          # known randomized assignment rate (A1)
OUTCOMES = ["visit", "conversion"]
LEARNER_NAMES = ["t_learner", "x_learner", "dr_learner", "causal_forest"]

t0 = time.time()
dev = data.load_criteo_dev()
X = dev[data.CRITEO_FEATURES].to_numpy(dtype="float64")
t_arr = dev["treatment"].to_numpy()
folds = assign_folds(dev["row_id"].to_numpy())
STAGE2.mkdir(parents=True, exist_ok=True)

oof_path = STAGE2 / "oof_cate.parquet"
if oof_path.exists():
    # fits are expensive and deterministic — reuse them; only the metrics
    # phase below is recomputed
    oof = pd.read_parquet(oof_path)
    assert (oof["row_id"].to_numpy() == dev["row_id"].to_numpy()).all()
    print("reusing existing oof_cate.parquet (delete it to refit)", flush=True)
else:
    oof = pd.DataFrame({"row_id": dev["row_id"]})
    for outcome in OUTCOMES:
        y = dev[outcome].to_numpy(dtype="float64")
        for i, name in enumerate(LEARNER_NAMES):
            tic = time.time()
            oof[f"{outcome}_{name}"] = cross_fit_cate(
                lambda: default_learners(P_ASSIGN, seed=LEARNER_SEED)[i],
                X, t_arr, y, folds,
            )
            print(f"{outcome}/{name}: {time.time() - tic:.0f}s "
                  f"(total {(time.time() - t0)/60:.0f} min)", flush=True)
    oof.to_parquet(oof_path, compression="zstd", index=False)

print("metrics + bootstrap CIs ...", flush=True)
metrics: dict = {"created": date.today().isoformat(), "B": 500,
                 "seed": 20260717}
gates_rows = []
for outcome in OUTCOMES:
    y = dev[outcome].to_numpy(dtype="float64")
    metrics[outcome] = {}
    for name in LEARNER_NAMES:
        s = oof[f"{outcome}_{name}"].to_numpy()
        tic = time.time()
        qini = bootstrap_ci(y, t_arr, s)
        auuc = float(auuc_score(
            pd.DataFrame({"y": y, "w": t_arr, "score": s}),
            outcome_col="y", treatment_col="w", normalize=True,
        )["score"])
        metrics[outcome][name] = {"qini": qini, "auuc_normalized": auuc}
        g = gates_deciles(y, t_arr, s).reset_index()
        g.insert(0, "outcome", outcome)
        g.insert(1, "learner", name)
        gates_rows.append(g)
        print(f"  {outcome}/{name}: qini={qini['point']:.6f} "
              f"[{qini['ci_lo']:.6f}, {qini['ci_hi']:.6f}] auuc={auuc:.4f} "
              f"({time.time() - tic:.0f}s)", flush=True)
(STAGE2 / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
pd.concat(gates_rows).to_csv(STAGE2 / "gates_deciles.csv", index=False)

print("H1 BLP tests ...", flush=True)
blp_all = {
    outcome: {name: blp_test(dev[outcome], t_arr,
                             oof[f"{outcome}_{name}"], P_ASSIGN)
              for name in LEARNER_NAMES}
    for outcome in OUTCOMES
}
primary = blp_all["visit"]["dr_learner"]
h1 = {
    "created": date.today().isoformat(),
    "spec": "HYPOTHESES.md amendment A1",
    "primary": {"outcome": "visit", "proxy": "dr_learner", "blp": primary},
    "exploratory": blp_all,
    "verdict_h1_raw": bool(primary["beta2_p_value"] < 0.05
                           and primary["beta2_het"] > 0),
    "note": "Holm correction across H1–H3 applied at project level; "
            "raw p recorded here.",
}
(STAGE2 / "h1_blp.json").write_text(json.dumps(h1, indent=2) + "\n")

clan = clan_table(dev[data.CRITEO_FEATURES], oof["visit_dr_learner"])
clan.to_csv(STAGE2 / "clan_dr_visit.csv")
write_manifest()

print(f"\nH1 (visit, dr_learner): beta2={primary['beta2_het']:.3f} "
      f"(se {primary['beta2_se']:.3f}), p={primary['beta2_p_value']:.2e} "
      f"-> raw verdict: {h1['verdict_h1_raw']}")
print(f"done in {(time.time() - t0)/60:.0f} min")
