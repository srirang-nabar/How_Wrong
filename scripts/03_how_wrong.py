"""Stage 3 experiment (Tier 2): how wrong without the experiment?

Runs the A2 protocol on the dev subsample: corrupt the RCT under both
mechanisms across the severity grids (R = 20 replicates each), estimate the
ATE four ways per corrupted dataset, and grade every estimate against the
uncorrupted RCT truth. Writes to results/stage3/:

  bias_grid.csv   — replicate-level estimates for every cell
  summary.json    — per-cell bias summaries + the H3 adjudication
  (manifest refreshed)

Run: uv run python scripts/03_how_wrong.py   (~2–3 h CPU, background it)
"""

import json
import time
from datetime import date

from how_wrong import data
from how_wrong.ate import diff_in_means
from how_wrong.confound import (
    GAMMA_GRID, GAMMA_STAR, PDROP_GRID, CONFOUND_FEATURES,
    recovery_stats, run_cell, summarize_bias,
)
from how_wrong.reproduce import RESULTS_DIR, write_manifest

import pandas as pd

STAGE3 = RESULTS_DIR / "stage3"
STAGE3.mkdir(parents=True, exist_ok=True)

t0 = time.time()
dev = data.load_criteo_dev()
truth = {y: diff_in_means(dev[y], dev["treatment"]).ate
         for y in ("visit", "conversion")}
print(f"truth (uncorrupted dev): visit {truth['visit']:+.5%}, "
      f"conversion {truth['conversion']:+.5%}", flush=True)

cells: list[pd.DataFrame] = []
grid_summaries = []

def do_cell(mechanism, severity, outcome, features, label):
    tic = time.time()
    cell = run_cell(dev, mechanism, severity, outcome, features)
    cell.insert(0, "arm", label)
    cells.append(cell)
    s = summarize_bias(cell, truth[outcome])
    print(f"{label} {mechanism} sev={severity} {outcome}: "
          f"naive {s['naive']['mean_bias']:+.5f} | "
          f"aipw {s['aipw']['mean_bias']:+.5f} "
          f"({time.time() - tic:.0f}s, total {(time.time()-t0)/60:.0f}m)",
          flush=True)
    return cell, s

for gamma in GAMMA_GRID:
    cell, s = do_cell("confounded_assignment", gamma, "visit",
                      data.CRITEO_FEATURES, "primary")
    grid_summaries.append({"mechanism": "confounded_assignment",
                           "severity": gamma, "outcome": "visit", "bias": s})
    if gamma == GAMMA_STAR:
        star_cell = cell
for p_drop in PDROP_GRID:
    _, s = do_cell("selective_attrition", p_drop, "visit",
                   data.CRITEO_FEATURES, "robustness")
    grid_summaries.append({"mechanism": "selective_attrition",
                           "severity": p_drop, "outcome": "visit", "bias": s})

hidden_features = [f for f in data.CRITEO_FEATURES
                   if f not in CONFOUND_FEATURES]
hidden_cell, hidden_s = do_cell("confounded_assignment", GAMMA_STAR, "visit",
                                hidden_features, "hidden_confounder")
conv_cell, conv_s = do_cell("confounded_assignment", GAMMA_STAR, "conversion",
                            data.CRITEO_FEATURES, "conversion_explore")

pd.concat(cells).to_csv(STAGE3 / "bias_grid.csv", index=False)

star_bias = summarize_bias(star_cell, truth["visit"])
star_recovery = recovery_stats(star_cell, truth["visit"])
a = star_bias["naive"]
verdict = bool(
    a["p_two_sided"] < 0.05
    and abs(a["mean_bias"]) >= 0.25 * abs(truth["visit"])
    and star_recovery["mean_recovery"] >= 0.5
    and star_recovery["p_one_sided_gt0"] < 0.05
)
summary = {
    "created": date.today().isoformat(),
    "spec": "HYPOTHESES.md amendment A2",
    "truth": truth,
    "grid": grid_summaries,
    "h3": {
        "gamma_star": GAMMA_STAR,
        "a_naive_bias": {"mean_bias": a["mean_bias"], "ci_lo": a["ci_lo"],
                         "ci_hi": a["ci_hi"],
                         "p_two_sided": a["p_two_sided"],
                         "materiality_threshold": 0.25 * abs(truth["visit"])},
        "b_recovery": star_recovery,
        "p_h3": max(a["p_two_sided"], star_recovery["p_one_sided_gt0"]),
        "verdict_h3_raw": verdict,
        "strong_support": bool(star_recovery["ci_lo"] > 0.5),
    },
    "exploratory": {
        "hidden_confounder": {"features_denied": CONFOUND_FEATURES,
                              "bias": hidden_s},
        "conversion_at_gamma_star": {"bias": conv_s},
    },
}
(STAGE3 / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
write_manifest()

h = summary["h3"]
print(f"\nH3 @ gamma*={GAMMA_STAR}: naive bias {a['mean_bias']:+.5f} "
      f"[{a['ci_lo']:+.5f}, {a['ci_hi']:+.5f}], recovery "
      f"{star_recovery['mean_recovery']:.2f} "
      f"[{star_recovery['ci_lo']:.2f}, {star_recovery['ci_hi']:.2f}]")
print(f"p_H3 = {h['p_h3']:.2e} -> raw verdict {verdict} "
      f"(strong: {h['strong_support']})")
print(f"done in {(time.time() - t0)/60:.0f} min")
