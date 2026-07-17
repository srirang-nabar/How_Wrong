"""Stage 4 policy pass (Tier 2): the three-curve money chart and the H2
adjudication, per HYPOTHESES.md amendment A3.

Scores: DR-learner OOF conversion CATE (committed, Stage 2) vs a
propensity-to-convert model (LightGBM ignoring treatment, OOF on the same
hashed folds). Primary metric: within-top-k diff-in-means × |top-k|;
robustness: Hájek/IPW with cross-fitted ê (C16). Writes results/stage4/:

  curves_conversion.csv / curves_visit.csv — three-curve data
  roi.csv                                  — ROI + sensitivity (exploratory)
  policy_results.json                      — H2 verdict + all tests

Run: uv run python scripts/04_policy.py   (~20 min)
"""

import json
import time
from datetime import date

import numpy as np
import pandas as pd

from how_wrong import data
from how_wrong.confound import _nuisance_clf, PROPENSITY_CLIP
from how_wrong.folds import assign_folds
from how_wrong.learners import _lgbm
from how_wrong.policy import (
    K_PRIMARY, policy_difference_test, roi_table, three_curves,
)
from how_wrong.reproduce import RESULTS_DIR, write_manifest

STAGE4 = RESULTS_DIR / "stage4"
STAGE4.mkdir(parents=True, exist_ok=True)
SEED = 20260717

t0 = time.time()
dev = data.load_criteo_dev()
oof = pd.read_parquet(RESULTS_DIR / "stage2" / "oof_cate.parquet")
assert (oof["row_id"].to_numpy() == dev["row_id"].to_numpy()).all()
X = dev[data.CRITEO_FEATURES].to_numpy("float64")
t_arr = dev["treatment"].to_numpy("float64")

print("propensity-to-convert model (OOF, ignoring treatment) ...", flush=True)
folds5 = assign_folds(dev["row_id"].to_numpy())
prop_score = np.full(len(dev), np.nan)
y_conv = dev["conversion"].to_numpy("float64")
for k in range(5):
    tr, te = folds5 != k, folds5 == k
    m = _lgbm(fast=False, seed=SEED).fit(X[tr], y_conv[tr])
    prop_score[te] = m.predict(X[te])
assert not np.isnan(prop_score).any()

print("cross-fitted e-hat for the IPW-robust variant ...", flush=True)
folds2 = assign_folds(dev["row_id"].to_numpy(), n_folds=2, seed=SEED)
e_hat = np.full(len(dev), np.nan)
for k in (0, 1):
    tr, te = folds2 != k, folds2 == k
    pm = _nuisance_clf(SEED, 4).fit(X[tr], t_arr[tr])
    e_hat[te] = pm.predict_proba(X[te])[:, 1]
e_hat = np.clip(e_hat, *PROPENSITY_CLIP)

cate_conv = oof["conversion_dr_learner"].to_numpy()
k_grid = np.round(np.arange(0.02, 1.0001, 0.02), 2)

print("three-curve data ...", flush=True)
curves_conv = three_curves(y_conv, t_arr,
                           {"cate": cate_conv, "propensity": prop_score},
                           k_grid)
curves_conv.to_csv(STAGE4 / "curves_conversion.csv", index=False)
y_vis = dev["visit"].to_numpy("float64")
curves_vis = three_curves(y_vis, t_arr,
                          {"cate": oof["visit_dr_learner"].to_numpy(),
                           "propensity": prop_score},
                          k_grid)
curves_vis.to_csv(STAGE4 / "curves_visit.csv", index=False)

print("H2 bootstrap tests ...", flush=True)
h2_primary, h2_robust = {}, {}
for k in K_PRIMARY:
    tic = time.time()
    key = f"k{int(k*100)}"
    h2_primary[key] = policy_difference_test(y_conv, t_arr, cate_conv,
                                             prop_score, k)
    h2_robust[key] = policy_difference_test(y_conv, t_arr, cate_conv,
                                            prop_score, k, e=e_hat)
    print(f"  k={k}: primary delta {h2_primary[key]['delta']:+.1f} "
          f"[{h2_primary[key]['ci_lo']:+.1f}, {h2_primary[key]['ci_hi']:+.1f}]"
          f" | robust delta {h2_robust[key]['delta']:+.1f} "
          f"({time.time()-tic:.0f}s)", flush=True)

p10, p30 = h2_primary["k10"], h2_primary["k30"]
verdict = bool(p10["delta"] > 0 and p10["ci_lo"] > 0
               and p30["delta"] > 0 and p30["ci_lo"] > 0)
robust_agrees = bool(
    (h2_robust["k10"]["ci_lo"] > 0) == (p10["ci_lo"] > 0)
    and (h2_robust["k30"]["ci_lo"] > 0) == (p30["ci_lo"] > 0))

roi = roi_table(curves_conv, "cate", len(dev), 40.0, 0.15)
roi.to_csv(STAGE4 / "roi.csv", index=False)
sens = []
for v in (20.0, 40.0, 80.0):
    for c in (0.05, 0.15, 0.50):
        r = roi_table(curves_conv, "cate", len(dev), v, c)
        best = r.loc[r["profit"].idxmax()]
        sens.append({"value": v, "cost": c, "k_opt": float(best["k"]),
                     "profit_at_opt": float(best["profit"])})

results = {
    "created": date.today().isoformat(),
    "spec": "HYPOTHESES.md amendment A3",
    "h2": {"primary": h2_primary, "ipw_robust": h2_robust,
           "p_h2": max(p10["p_boot"], p30["p_boot"]),
           "verdict_h2_raw": verdict, "robust_agrees": robust_agrees},
    "roi_assumptions": {"value_per_conversion": 40.0,
                        "cost_per_target": 0.15},
    "roi_sensitivity": sens,
}
(STAGE4 / "policy_results.json").write_text(
    json.dumps(results, indent=2) + "\n")
write_manifest()

print(f"\nH2: p = {results['h2']['p_h2']:.4f}, verdict {verdict}, "
      f"robust agrees: {robust_agrees}")
print(f"done in {(time.time()-t0)/60:.0f} min")
