"""No catastrophic regressions (CP245).

The guard that makes enabling the loop / promoting an adapter reversible:
it protects each capability family independently, because an average hides
exactly the failure that matters -- a math gain masking a language loss.
"""
from __future__ import annotations

import pytest

from core.learning.capability_regression_battery import (
    CapabilityRegressionGuard,
    Probe,
    compose_from_battery,
    score_family,
)


def _probes():
    contains = lambda needle: (lambda ans: needle in ans.lower())
    return [
        Probe("language", "spell cat", contains("c-a-t")),
        Probe("math", "2+2", contains("4")),
        Probe("factual", "capital of France", contains("paris")),
        Probe("instruction_following", "reply OK", contains("ok")),
    ]


def _solver(answers):
    return lambda prompt: answers.get(prompt, "")


# ── Per-family scoring ──────────────────────────────────────────────────


def test_score_family_is_per_capability():
    scores = score_family(_probes(), _solver({
        "spell cat": "c-a-t", "2+2": "4",
        "capital of France": "paris", "reply OK": "ok",
    }))
    assert scores == {"language": 1.0, "math": 1.0, "factual": 1.0,
                      "instruction_following": 1.0}


# ── The core discipline: a gain masking a loss is CAUGHT ────────────────


def test_a_math_gain_masking_a_language_loss_is_flagged():
    """The failure averages hide. This is the whole reason the guard exists."""
    good = {"spell cat": "c-a-t", "2+2": "wrong", "capital of France": "paris",
            "reply OK": "ok"}
    guard = CapabilityRegressionGuard(_probes(), max_drop=0.02)
    guard.measure_baseline(_solver(good))

    # candidate: math now right (+), but language now WRONG (-)
    candidate = {"spell cat": "nope", "2+2": "4", "capital of France": "paris",
                 "reply OK": "ok"}
    report = guard.evaluate(_solver(candidate))
    assert "math" in report["improvements"]
    assert "language" in report["regressions"]
    # the average went UP (2/4 -> 3/4) but the change is NOT safe
    assert report["safe"] is False
    assert "language" in report["verdict"]


def test_a_clean_improvement_passes():
    base = {"spell cat": "c-a-t", "2+2": "wrong", "capital of France": "paris",
            "reply OK": "ok"}
    guard = CapabilityRegressionGuard(_probes())
    guard.measure_baseline(_solver(base))
    better = dict(base, **{"2+2": "4"})  # math up, nothing down
    report = guard.evaluate(_solver(better))
    assert report["safe"] is True
    assert report["regressions"] == []
    assert "math" in report["improvements"]


def test_no_change_is_safe():
    answers = {"spell cat": "c-a-t", "2+2": "4", "capital of France": "paris",
               "reply OK": "ok"}
    guard = CapabilityRegressionGuard(_probes())
    guard.measure_baseline(_solver(answers))
    report = guard.evaluate(_solver(answers))
    assert report["safe"] is True
    assert report["improvements"] == [] and report["regressions"] == []


def test_small_jitter_within_margin_is_not_a_regression():
    # 5 language probes; dropping 1 = -0.2, above a 0.02 margin -> caught.
    # But with a wider margin it is tolerated as noise.
    probes = [Probe("language", f"q{i}", (lambda a: "yes" in a)) for i in range(5)]
    guard = CapabilityRegressionGuard(probes, max_drop=0.25)
    guard.measure_baseline(lambda p: "yes")
    # one of five now fails
    fail_one = {f"q{i}": "yes" for i in range(5)}
    fail_one["q0"] = "no"
    report = guard.evaluate(lambda p: fail_one[p])
    assert report["safe"] is True  # -0.2 within the 0.25 margin


# ── Honest degradation and guards ───────────────────────────────────────


def test_solver_errors_score_as_wrong_not_crash():
    def broken(prompt):
        raise RuntimeError("model down")

    scores = score_family(_probes(), broken)
    assert all(v == 0.0 for v in scores.values())


def test_evaluate_requires_a_baseline_first():
    guard = CapabilityRegressionGuard(_probes())
    with pytest.raises(ValueError, match="no baseline"):
        guard.evaluate(_solver({}))


def test_invalid_configuration_is_refused():
    with pytest.raises(ValueError, match="not a protected family"):
        Probe("astrology", "q", lambda a: True)
    with pytest.raises(ValueError, match="protects nothing"):
        CapabilityRegressionGuard([])
    with pytest.raises(ValueError, match="max_drop"):
        CapabilityRegressionGuard(_probes(), max_drop=0.9)


# ── Reuses the existing sealed batteries ────────────────────────────────


def test_compose_from_battery_reuses_verifiable_tasks():
    from core.learning.verifiable_tasks import build_task_set

    tasks = build_task_set(
        domains=["arithmetic_chain"], depths=[2], per_cell=3, seed=1
    )
    probes = compose_from_battery(tasks, family="math")
    assert len(probes) == 3
    assert all(p.family == "math" for p in probes)
    # a probe grades its own gold correct
    gold = tasks[0]
    assert probes[0].grader(f"FINAL_ANSWER: {gold.expected}") is True
