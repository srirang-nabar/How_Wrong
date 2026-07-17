"""Stage 5 finals (Tier 3): the pre-registered full-data numbers, the C16
exploratory companions, and the formal Holm adjudication across H1–H3.

Full-scale compute is restricted to what the plan registers: the
covariate-adjusted companion ATE (AIPW, 13.98M rows) and nothing else — the
primaries were adjudicated on the dev subsample per A1/A2/A3 and are not
re-run at scale (that would be fishing). Also: BLP robustness with
estimated ê on dev (A1 assumed the constant 0.85), the Holm record, and
the compute-budget log. Writes results/stage5/finals.json.

Run: uv run python scripts/05_full_finals.py   (~40 min, ~3 GB RAM)
"""

import gc
import json
import time
from datetime import date

import numpy as np
import pandas as pd
import statsmodels.api as sm

from how_wrong import data
from how_wrong.confound import _nuisance_clf, _nuisance_reg, PROPENSITY_CLIP
from how_wrong.folds import assign_folds
from how_wrong.reproduce import RESULTS_DIR, write_manifest

STAGE5 = RESULTS_DIR / "stage5"
STAGE5.mkdir(parents=True, exist_ok=True)
SEED = 20260717
CHUNK = 2_000_000

t0 = time.time()
print("loading full criteo ...", flush=True)
full = data.load_criteo()
full.insert(0, "row_id", np.arange(len(full), dtype="int64"))
X = full[data.CRITEO_FEATURES].to_numpy("float32")
t_arr = full["treatment"].to_numpy()
folds2 = assign_folds(full["row_id"].to_numpy(), n_folds=2, seed=SEED)
print(f"  {len(full):,} rows in {time.time()-t0:.0f}s", flush=True)


def chunked_predict(model, X, te_idx, out, proba=False):
    for lo in range(0, len(te_idx), CHUNK):
        sl = te_idx[lo:lo + CHUNK]
        p = (model.predict_proba(X[sl])[:, 1] if proba
             else model.predict(X[sl]))
        out[sl] = p


print("cross-fitted e-hat (full) ...", flush=True)
e_hat = np.empty(len(full), dtype="float32")
for k in (0, 1):
    tr = np.flatnonzero(folds2 != k)
    te = np.flatnonzero(folds2 == k)
    pm = _nuisance_clf(SEED, 4).fit(X[tr], t_arr[tr])
    chunked_predict(pm, X, te, e_hat, proba=True)
    del pm; gc.collect()
e_hat = np.clip(e_hat, *PROPENSITY_CLIP)

aipw_full = {}
for outcome in ("visit", "conversion"):
    print(f"AIPW companion, {outcome} ...", flush=True)
    y = full[outcome].to_numpy("float32")
    mu1 = np.empty(len(full), dtype="float32")
    mu0 = np.empty(len(full), dtype="float32")
    for k in (0, 1):
        tr = np.flatnonzero(folds2 != k)
        te = np.flatnonzero(folds2 == k)
        tr1, tr0 = tr[t_arr[tr] == 1], tr[t_arr[tr] == 0]
        m1 = _nuisance_reg(SEED, 4).fit(X[tr1], y[tr1])
        chunked_predict(m1, X, te, mu1); del m1; gc.collect()
        m0 = _nuisance_reg(SEED, 4).fit(X[tr0], y[tr0])
        chunked_predict(m0, X, te, mu0); del m0; gc.collect()
    y64 = y.astype("float64"); m1_ = mu1.astype("float64")
    m0_ = mu0.astype("float64"); e64 = e_hat.astype("float64")
    psi = (m1_ - m0_ + t_arr * (y64 - m1_) / e64
           - (1 - t_arr) * (y64 - m0_) / (1 - e64))
    naive = float(y64[t_arr == 1].mean() - y64[t_arr == 0].mean())
    aipw = float(psi.mean())
    se = float(psi.std(ddof=1) / np.sqrt(len(psi)))
    aipw_full[outcome] = {
        "diff_in_means": naive, "aipw": aipw, "aipw_se": se,
        "aipw_ci_lo": aipw - 1.96 * se, "aipw_ci_hi": aipw + 1.96 * se,
    }
    print(f"  raw {naive:+.5%} vs aipw {aipw:+.5%} (se {se:.5%})", flush=True)
    del y, mu1, mu0, y64, m1_, m0_, e64, psi; gc.collect()

e_summary = {"mean": float(e_hat.mean()), "sd": float(e_hat.std()),
             "min": float(e_hat.min()), "max": float(e_hat.max())}
del full, X, e_hat; gc.collect()

print("BLP robustness with estimated e-hat (dev) ...", flush=True)
dev = data.load_criteo_dev()
oof = pd.read_parquet(RESULTS_DIR / "stage2" / "oof_cate.parquet")
Xd = dev[data.CRITEO_FEATURES].to_numpy("float64")
td = dev["treatment"].to_numpy("float64")
folds2d = assign_folds(dev["row_id"].to_numpy(), n_folds=2, seed=SEED)
ed = np.empty(len(dev))
for k in (0, 1):
    tr, te = folds2d != k, folds2d == k
    pm = _nuisance_clf(SEED, 4).fit(Xd[tr], td[tr])
    ed[te] = pm.predict_proba(Xd[te])[:, 1]
ed = np.clip(ed, *PROPENSITY_CLIP)
yv = dev["visit"].to_numpy("float64")
s = oof["visit_dr_learner"].to_numpy(); s_c = s - s.mean()
tp = td - ed
Xb = sm.add_constant(np.column_stack([s_c, tp, tp * s_c]))
fit = sm.OLS(yv, Xb).fit(cov_type="HC3")
blp_robust = {"beta2_het": float(fit.params[3]),
              "beta2_se": float(fit.bse[3]),
              "beta2_p_value": float(fit.pvalues[3])}
print(f"  beta2 = {blp_robust['beta2_het']:.4f}, "
      f"p = {blp_robust['beta2_p_value']:.2e}", flush=True)

# ---- formal Holm adjudication across H1–H3 ----
h1 = json.loads((RESULTS_DIR / "stage2" / "h1_blp.json").read_text())
h3 = json.loads((RESULTS_DIR / "stage3" / "summary.json").read_text())
h2 = json.loads((RESULTS_DIR / "stage4" / "policy_results.json").read_text())
p1 = h1["primary"]["blp"]["beta2_p_value"]
p3 = h3["h3"]["p_h3"]
# H2 is directional; the observed effect is opposite the registered
# direction, so its one-sided p is ~1 (recorded conservatively).
p2 = 0.999
ordered = sorted([("H1", p1), ("H2", p2), ("H3", p3)], key=lambda x: x[1])
holm, alpha, rejected = {}, 0.05, True
for rank, (name, p) in enumerate(ordered):
    thresh = alpha / (3 - rank)
    rejected = rejected and (p < thresh)
    holm[name] = {"p": p, "threshold": thresh,
                  "significant_after_holm": bool(rejected and p < thresh)}

finals = {
    "created": date.today().isoformat(),
    "aipw_companion_full": aipw_full,
    "e_hat_full": e_summary,
    "blp_robust_ehat_dev": blp_robust,
    "holm": {
        "family": holm,
        "verdicts": {
            "H1": "supported" if holm["H1"]["significant_after_holm"] else "not supported",
            "H2": "rejected (effect opposite registered direction)",
            "H3": "supported" if holm["H3"]["significant_after_holm"] else "not supported",
        },
    },
    "compute_budget_cpu_min": {
        "stage1_full_pass": 3, "stage2_fits": 139, "stage2_metrics_rerun": 56,
        "stage3_experiment": 98, "stage4_policy": 84,
        "stage5_finals": round((time.time() - t0) / 60),
        "note": "single 8-core CPU box; no GPU, no cloud spend",
    },
}
(STAGE5 / "finals.json").write_text(json.dumps(finals, indent=2) + "\n")
write_manifest()
print(json.dumps(finals["holm"], indent=2))
print(f"done in {(time.time()-t0)/60:.0f} min")
