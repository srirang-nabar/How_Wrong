"""Stage 5 gate: finals recorded and consistent, report pack tied to the
claims, fresh-machine Tier 1 evidence present.
"""

import json

import pytest

from how_wrong import data

pytestmark = pytest.mark.gate_stage5

RESULTS = data.PROJECT_ROOT / "results"
REPORT = data.PROJECT_ROOT / "report"


@pytest.fixture(scope="module")
def finals():
    path = RESULTS / "stage5" / "finals.json"
    if not path.exists():
        pytest.skip("stage 5 finals script has not run yet")
    return json.loads(path.read_text())


def test_holm_family_consistent_with_stage_artifacts(finals):
    h1 = json.loads((RESULTS / "stage2" / "h1_blp.json").read_text())
    h3 = json.loads((RESULTS / "stage3" / "summary.json").read_text())
    fam = finals["holm"]["family"]
    assert fam["H1"]["p"] == h1["primary"]["blp"]["beta2_p_value"]
    assert fam["H3"]["p"] == h3["h3"]["p_h3"]
    # Holm at alpha=0.05 over three: H3 (smallest) vs alpha/3, H1 vs alpha/2
    assert fam["H3"]["significant_after_holm"] is True
    assert fam["H1"]["significant_after_holm"] is True
    assert fam["H2"]["significant_after_holm"] is False
    assert finals["holm"]["verdicts"]["H2"].startswith("rejected")


def test_full_aipw_companion_recorded(finals):
    for outcome in ("visit", "conversion"):
        c = finals["aipw_companion_full"][outcome]
        # companion sits below raw diff-in-means (the C16 direction) but
        # remains positive and precisely estimated
        assert 0 < c["aipw"] < c["diff_in_means"]
        assert c["aipw_ci_lo"] > 0
    e = finals["e_hat_full"]
    assert e["max"] > 0.9 and e["min"] < 0.7  # merge artifact visible at scale


def test_blp_robustness_confirms_h1(finals):
    b = finals["blp_robust_ehat_dev"]
    assert b["beta2_het"] > 0
    assert b["beta2_p_value"] < 0.05 / 3  # survives strictest Holm slot


def test_compute_budget_documented(finals):
    assert finals["compute_budget_cpu_min"]["stage5_finals"] > 0


def test_report_pack_states_the_claims():
    report = (REPORT / "report.md").read_text()
    for needle in ("5.9×", "93.7%", "+6.08 pp", "β₂ = 0.384", "−253",
                   "treatment-block-ordered", "[0.64, 0.98]"):
        assert needle in report, f"report.md missing claim text: {needle}"
    qa = (REPORT / "interview_qa.md").read_text()
    assert "Neyman" in qa and "sure thing" in qa
    bullets = (REPORT / "resume_bullets.md").read_text()
    assert "93.7%" in bullets and "5.9×" in bullets


def test_fresh_machine_log():
    log = RESULTS / "fresh_machine_run.log"
    if not log.exists():
        pytest.skip("fresh-machine dry run not yet recorded")
    text = log.read_text()
    assert "passed" in text
    assert "failed" not in text.split("=")[-1]
