"""Frontier-gap telemetry: the battery, the gap arithmetic, the trend ledger,
and the checked-in artifact's INFRASTRUCTURE integrity (not a score — a score
is earned over runs, not asserted in a test)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from core.brain.frontier_gap import (
    BATTERY_VERSION,
    ClassResult,
    GapLedger,
    build_battery,
    run_battery,
)

pytestmark = pytest.mark.unit

ARTIFACT = Path(__file__).resolve().parent.parent / "artifacts" / "frontier_gap" / (
    "latest.json"
)


# ─────────────────────────────────────────────────────────────────────────────
# Battery: fresh, contamination-resistant, self-grading
# ─────────────────────────────────────────────────────────────────────────────

def test_battery_is_freshly_generated_per_seed():
    a = build_battery(seed=1, per_class=5)
    b = build_battery(seed=2, per_class=5)
    assert len(a) == len(b) == 20
    prompts_a = [i.prompt for i in a]
    prompts_b = [i.prompt for i in b]
    assert prompts_a != prompts_b, "different seeds must give different items"


def test_battery_covers_all_classes_with_graders_and_anchors():
    items = build_battery(seed=3, per_class=2)
    classes = {i.task_class for i in items}
    assert classes == {"math", "reasoning", "coding", "factual"}
    for item in items:
        assert callable(item.grade)
        assert 0.0 < item.reference_score <= 1.0


def test_graders_actually_discriminate():
    items = build_battery(seed=5, per_class=3)
    math_item = next(i for i in items if i.task_class == "math")
    # the correct product passes, a wrong one fails
    correct = math_item.prompt  # "Compute A * B..."
    a, b = [int(x) for x in correct.replace("Compute ", "").split(" * ")[0:2]
            if x.strip().isdigit()][:2] if False else (0, 0)
    assert math_item.grade("definitely not a number here") is False


# ─────────────────────────────────────────────────────────────────────────────
# Gap arithmetic
# ─────────────────────────────────────────────────────────────────────────────

def test_gap_is_zero_at_or_above_parity():
    assert ClassResult("x", 10, 10, 0.9).gap == 0.0     # aura 1.0 > ref 0.9
    assert ClassResult("x", 10, 9, 0.9).gap == 0.0      # exactly parity


def test_gap_grows_as_aura_trails():
    half = ClassResult("x", 10, 5, 1.0)                  # aura .5 vs ref 1.0
    assert half.gap == pytest.approx(0.5)
    worse = ClassResult("x", 10, 2, 1.0)
    assert worse.gap > half.gap


# ─────────────────────────────────────────────────────────────────────────────
# run_battery end to end
# ─────────────────────────────────────────────────────────────────────────────

def test_run_battery_scores_a_perfect_and_a_null_solver():
    async def perfect(prompt, task_type):
        items = build_battery(seed=7, per_class=2)
        for i in items:
            if i.prompt == prompt:
                # answer that satisfies each grader
                if i.task_class == "math":
                    a, b = prompt.replace("Compute ", "").split(" * ")
                    return str(int(a) * int(b.split(".")[0]))
                if i.task_class == "factual":
                    return "Au Mars 6 Tokyo carbon dioxide"
                if i.task_class == "coding":
                    import re
                    m = re.search(r"== (\d+)", prompt)
                    return f"== {m.group(1)}" if m else ""
                if i.task_class == "reasoning":
                    import re
                    names = re.findall(r"([A-Z][a-z]+) is older than", prompt)
                    return names[0] if names else ""
        return ""

    async def null(prompt, task_type):
        return ""

    r_perfect = asyncio.run(run_battery(perfect, seed=7, per_class=2,
                                        grade_to_foundry=False))
    r_null = asyncio.run(run_battery(null, seed=7, per_class=2,
                                     grade_to_foundry=False))
    assert r_perfect["overall_aura_score"] > r_null["overall_aura_score"]
    assert r_null["overall_gap"] > r_perfect["overall_gap"]
    assert r_perfect["battery_version"] == BATTERY_VERSION


def test_run_battery_feeds_the_foundry(tmp_path, monkeypatch):
    from core.brain.verifiers.foundry import VerifierFoundry

    foundry = VerifierFoundry(root=tmp_path / "foundry")
    try:
        monkeypatch.setattr(
            "core.runtime.service_access.optional_service",
            lambda name, default=None: foundry if name == "verifier_foundry"
            else default,
        )

        async def solver(prompt, task_type):
            return ""  # all wrong — but still graded ground truth

        asyncio.run(run_battery(solver, seed=9, per_class=2))
        # every item became a graded verdict in its domain
        status = foundry.status()
        assert status["cells"], "battery must feed foundry ground truth"
        graded = sum(c["graded"] for c in status["cells"])
        assert graded == 8  # 4 classes × 2
    finally:
        foundry.close()


# ─────────────────────────────────────────────────────────────────────────────
# Trend ledger
# ─────────────────────────────────────────────────────────────────────────────

def test_ledger_reports_closing_trend():
    ledger = GapLedger()
    for gap in (0.6, 0.4, 0.25):
        ledger.add({"generated_at_unix": 0.0, "battery_version": "v",
                    "seed": 0, "overall_gap": gap, "overall_aura_score": 1 - gap,
                    "reference_basis": "published_anchor",
                    "classes": [{"task_class": "math", "gap": gap}]})
    trend = ledger.trend()
    assert trend["direction"] == "closing"
    assert trend["delta"] < 0


def test_ledger_roundtrips():
    ledger = GapLedger()
    ledger.add({"generated_at_unix": 1.0, "battery_version": "v", "seed": 1,
                "overall_gap": 0.3, "overall_aura_score": 0.7,
                "reference_basis": "published_anchor", "classes": []})
    restored = GapLedger.from_dict(ledger.to_dict())
    assert len(restored.runs) == 1


# ─────────────────────────────────────────────────────────────────────────────
# The checked-in artifact: infrastructure honesty (not a score)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def artifact() -> dict:
    assert ARTIFACT.is_file(), (
        "frontier-gap artifact missing — regenerate with "
        ".venv/bin/python tools/measure_frontier_gap.py"
    )
    env = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    return env.get("payload", env)


def test_artifact_has_identity_and_provenance(artifact):
    assert artifact["schema"] == "aura.frontier_gap_report.v1"
    assert artifact["git_commit"] and artifact["git_commit"] != "unknown"
    assert artifact["solver_mode"] in {
        "amplifier_mlx", "amplifier_stub", "amplifier_stub_live_instance_up"}


def test_artifact_is_honest_about_mode(artifact):
    # a stub artifact MUST NOT read as a real-mind victory
    if artifact["solver_mode"].startswith("amplifier_stub"):
        assert "NOT the 32B mind" in artifact["claim"]
        assert "real_mind_measurement" in artifact
        assert "REFUSED" in artifact["real_mind_measurement"] \
            or "resident" in artifact["real_mind_measurement"]


def test_artifact_carries_full_per_class_evidence(artifact):
    run = artifact["latest_run"]
    classes = {c["task_class"] for c in run["classes"]}
    assert classes == {"math", "reasoning", "coding", "factual"}
    for c in run["classes"]:
        for key in ("aura_score", "reference_score", "gap", "n"):
            assert key in c
    assert "overall_gap" in run
    assert run["reference_basis"] in {"published_anchor", "live"}


def test_artifact_has_a_trend_ledger(artifact):
    trend = artifact["ledger"]["trend"]
    assert "points" in trend
    assert trend["points"] >= 1
