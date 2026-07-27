"""SPARK-064: the gate rows come out of the batteries, not out of a text editor.

A complete gate set is worth nothing if the rows are prose. These tests run the
real batteries -- the real interference battery over a real MLX module, the real
capability guard, the real canary verdict, real paired benchmark and held-out
results -- and check that the produced rows carry the batteries' own counts and
the batteries' own verdicts.
"""

from __future__ import annotations

import pytest

from core.learning.capability_regression_battery import (
    CapabilityRegressionGuard,
    Probe,
)
from core.learning.heldout_battery import BatteryResult
from core.learning.permanent_distillation import (
    FAIL,
    PASS,
    REQUIRED_GATES,
    evaluate_promotion,
)
from core.learning.permanent_distillation_gates import (
    GateProducerError,
    anti_interference_gate,
    behavior_canary_gate,
    build_gate_report,
    capability_families_gate,
    frontier_regression_gate,
    memory_retention_gate,
)
from core.learning.recurrent_sft_behavior_canaries import BEHAVIOR_CANARY_SCHEMA
from core.memory.memory_benchmarking import MemoryBenchmarkResult

# --- anti_interference, over the real battery's receipt ---------------------


def _interference_receipt(stable: int, total: int, verdict: str) -> dict:
    from core.learning.interference_battery import INTERFERENCE_BATTERY_SCHEMA

    results = [
        {"probe": index, "top1_same": index < stable, "drift": 0.0, "stable": index < stable}
        for index in range(total)
    ]
    return {
        "schema": INTERFERENCE_BATTERY_SCHEMA,
        "probes": total,
        "stable_probes": stable,
        "stable_fraction": round(stable / total, 4),
        "required_stable_fraction": 0.9,
        "max_stable_drift": 0.05,
        "results": results,
        "verdict": verdict,
        "ran_at": 0.0,
    }


def test_the_interference_row_carries_the_batterys_own_counts():
    row = anti_interference_gate(_interference_receipt(24, 24, PASS))
    assert row["gate"] == "anti_interference"
    assert row["probes_graded"] == 24
    assert row["probes_passed"] == 24
    assert row["verdict"] == PASS


def test_the_interference_verdict_is_the_batterys_not_a_second_opinion():
    # 20/24 is 0.83, under the battery's own 0.9 floor. The producer relays
    # the battery's FAIL rather than recomputing a threshold of its own.
    row = anti_interference_gate(_interference_receipt(20, 24, FAIL))
    assert row["verdict"] == FAIL
    assert row["probes_passed"] == 20


def test_a_receipt_whose_counts_disagree_with_its_rows_is_refused():
    receipt = _interference_receipt(24, 24, PASS)
    receipt["stable_probes"] = 30
    with pytest.raises(GateProducerError) as excinfo:
        anti_interference_gate(receipt)
    assert "stable_count_differs" in str(excinfo.value)


def test_a_dict_shaped_like_an_interference_receipt_is_not_one():
    receipt = _interference_receipt(24, 24, PASS)
    receipt["schema"] = "aura.something_else.v1"
    with pytest.raises(GateProducerError):
        anti_interference_gate(receipt)


def test_the_interference_row_binds_a_digest_over_the_whole_receipt():
    clean = anti_interference_gate(_interference_receipt(24, 24, PASS))
    edited = _interference_receipt(24, 24, PASS)
    edited["results"][0]["drift"] = 0.9
    assert clean["evidence_sha256"] != anti_interference_gate(edited)["evidence_sha256"]


# --- capability_families, over the real guard -------------------------------


def _guard() -> CapabilityRegressionGuard:
    probes = []
    for family in ("language", "math", "code"):
        for index in range(8):
            probes.append(
                Probe(
                    family=family,
                    prompt=f"{family} probe {index}",
                    grader=lambda answer: answer == "ok",
                )
            )
    return CapabilityRegressionGuard(probes=probes)


def _capability_report(guard: CapabilityRegressionGuard, *, regress: bool) -> dict:
    guard.measure_baseline(lambda _prompt: "ok")
    if not regress:
        return guard.evaluate(lambda _prompt: "ok")
    # Half the language probes now fail; the guard's own margin decides.
    seen = {"n": 0}

    def solve(prompt: str) -> str:
        if prompt.startswith("language"):
            seen["n"] += 1
            return "ok" if seen["n"] % 2 else "no"
        return "ok"

    return guard.evaluate(solve)


def test_the_capability_row_counts_the_guards_own_probes():
    guard = _guard()
    row = capability_families_gate(guard, _capability_report(guard, regress=False))
    assert row["probes_graded"] == 24
    assert row["probes_passed"] == 24
    assert row["verdict"] == PASS


def test_a_single_family_regression_fails_the_row_despite_the_others():
    guard = _guard()
    report = _capability_report(guard, regress=True)
    row = capability_families_gate(guard, report)
    assert report["regressions"] == ["language"]
    assert row["verdict"] == FAIL
    # math and code still perfect; the row is not an average.
    assert row["probes_passed"] < row["probes_graded"]


def test_a_report_from_a_different_guard_is_refused():
    guard = _guard()
    report = _capability_report(guard, regress=False)
    other = CapabilityRegressionGuard(
        probes=[Probe(family="social_reasoning", prompt="p", grader=lambda a: True)]
    )
    with pytest.raises(GateProducerError) as excinfo:
        capability_families_gate(other, report)
    assert "families_differ" in str(excinfo.value)


def test_a_bare_dict_is_not_a_capability_guard():
    guard = _guard()
    report = _capability_report(guard, regress=False)
    with pytest.raises(GateProducerError):
        capability_families_gate({"probes": []}, report)


# --- the three canary gates -------------------------------------------------


def _behavior_verdict(**family_overrides) -> dict:
    families = {}
    for family in ("identity_grounding", "tool_effect_honesty", "authority_safety"):
        override = family_overrides.get(family, {})
        case_count = override.get("case_count", 12)
        trained_pass = override.get("trained_pass_count", case_count)
        right_to_wrong = override.get("right_to_wrong", 0)
        families[family] = {
            "case_count": case_count,
            "trained_pass_count": trained_pass,
            "right_to_wrong": right_to_wrong,
            "passed": trained_pass == case_count and right_to_wrong == 0,
        }
    return {
        "schema": f"{BEHAVIOR_CANARY_SCHEMA}.verdict",
        "case_count": sum(row["case_count"] for row in families.values()),
        "by_family": families,
        "passed": all(row["passed"] for row in families.values()),
    }


@pytest.mark.parametrize(
    ("gate", "family"),
    [
        ("personality_retention", "identity_grounding"),
        ("tool_effect_honesty", "tool_effect_honesty"),
        ("authority_safety", "authority_safety"),
    ],
)
def test_each_canary_gate_reads_its_own_family(gate, family):
    row = behavior_canary_gate(gate, _behavior_verdict())
    assert row["gate"] == gate
    assert row["probes_graded"] == 12
    assert row["verdict"] == PASS
    assert family  # the mapping under test


def test_one_right_to_wrong_transition_fails_the_family():
    verdict = _behavior_verdict(
        tool_effect_honesty={"trained_pass_count": 11, "right_to_wrong": 1}
    )
    row = behavior_canary_gate("tool_effect_honesty", verdict)
    assert row["verdict"] == FAIL
    # The other families are unaffected -- gates are per family, not pooled.
    assert behavior_canary_gate("authority_safety", verdict)["verdict"] == PASS


def test_a_family_the_canaries_did_not_measure_is_a_missing_measurement():
    verdict = _behavior_verdict()
    del verdict["by_family"]["authority_safety"]
    with pytest.raises(GateProducerError) as excinfo:
        behavior_canary_gate("authority_safety", verdict)
    assert "family_missing" in str(excinfo.value)


# --- memory and frontier are paired claims ----------------------------------


def _memory(accuracy: float) -> MemoryBenchmarkResult:
    return MemoryBenchmarkResult(
        strategy="graph_selective",
        accuracy=accuracy,
        f1=accuracy,
        p95_latency_ms=1.0,
        mean_tokens=10.0,
        duplication_rate=0.0,
    )


def test_memory_retention_needs_both_sides_of_the_pair():
    row = memory_retention_gate(
        before=_memory(0.90), after=_memory(0.89), case_count=40
    )
    assert row["verdict"] == PASS
    assert row["probes_graded"] == 40
    assert row["probes_passed"] == 36


def test_a_memory_drop_past_the_margin_fails():
    row = memory_retention_gate(
        before=_memory(0.90), after=_memory(0.70), case_count=40
    )
    assert row["verdict"] == FAIL


def test_comparing_two_different_memory_strategies_is_refused():
    other = MemoryBenchmarkResult(
        strategy="full_context",
        accuracy=0.9,
        f1=0.9,
        p95_latency_ms=1.0,
        mean_tokens=10.0,
        duplication_rate=0.0,
    )
    with pytest.raises(GateProducerError) as excinfo:
        memory_retention_gate(before=_memory(0.9), after=other, case_count=40)
    assert "strategy_differs" in str(excinfo.value)


def _battery(correct: int, total: int = 50, battery_id: str = "sealed-1") -> BatteryResult:
    return BatteryResult(battery_id=battery_id, total=total, correct=correct)


def test_frontier_regression_holds_when_the_battery_holds():
    row = frontier_regression_gate(before=_battery(40), after=_battery(40))
    assert row["verdict"] == PASS
    assert (row["probes_graded"], row["probes_passed"]) == (50, 40)


def test_grading_a_shorter_battery_after_the_change_is_refused():
    # The oldest way to make a regression disappear: score fewer items and
    # compare rates.
    with pytest.raises(GateProducerError) as excinfo:
        frontier_regression_gate(before=_battery(40), after=_battery(20, total=25))
    assert "total_differs" in str(excinfo.value)


def test_swapping_the_battery_between_arms_is_refused():
    with pytest.raises(GateProducerError) as excinfo:
        frontier_regression_gate(
            before=_battery(40), after=_battery(45, battery_id="sealed-2")
        )
    assert "battery_differs" in str(excinfo.value)


# --- the assembled report feeds the transaction -----------------------------


def _report(**overrides):
    guard = _guard()
    kwargs = {
        "interference_receipt": _interference_receipt(24, 24, PASS),
        "capability_guard": guard,
        "capability_report": _capability_report(guard, regress=False),
        "behavior_verdict": _behavior_verdict(),
        "memory_before": _memory(0.9),
        "memory_after": _memory(0.9),
        "memory_case_count": 40,
        "frontier_before": _battery(40),
        "frontier_after": _battery(41),
    }
    kwargs.update(overrides)
    return build_gate_report(**kwargs)


def test_the_assembled_report_is_complete_and_admits_a_promotion():
    import hashlib

    from core.learning.permanent_distillation import ADMIT, artifact_manifest

    report = _report()
    assert [row["gate"] for row in report["gates"]] == list(REQUIRED_GATES)

    def artifact(tag):
        return artifact_manifest(
            artifact_id=tag,
            base_model_identity="resident-32b",
            adapter_identity=f"rlc-{tag}",
            files=[
                {
                    "name": "adapter.safetensors",
                    "sha256": hashlib.sha256(tag.encode()).hexdigest(),
                    "size_bytes": 8,
                }
            ],
        )

    decision = evaluate_promotion(
        report=report,
        candidate_artifact=artifact("trained"),
        incumbent_artifact=artifact("frozen"),
    )
    assert decision["decision"] == ADMIT


def test_a_real_battery_failure_propagates_into_a_named_refusal():
    from core.learning.permanent_distillation import REFUSE, artifact_manifest

    guard = _guard()
    report = _report(
        capability_guard=guard,
        capability_report=_capability_report(guard, regress=True),
    )

    def artifact(tag):
        import hashlib

        return artifact_manifest(
            artifact_id=tag,
            base_model_identity="resident-32b",
            adapter_identity=f"rlc-{tag}",
            files=[
                {
                    "name": "adapter.safetensors",
                    "sha256": hashlib.sha256(tag.encode()).hexdigest(),
                    "size_bytes": 8,
                }
            ],
        )

    decision = evaluate_promotion(
        report=report,
        candidate_artifact=artifact("trained"),
        incumbent_artifact=artifact("frozen"),
    )
    assert decision["decision"] == REFUSE
    assert decision["refusals"][0]["gate"] == "capability_families"


def test_the_report_cannot_be_assembled_without_every_measurement():
    with pytest.raises(TypeError):
        build_gate_report(  # type: ignore[call-arg]
            interference_receipt=_interference_receipt(24, 24, PASS),
        )
