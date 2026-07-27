"""SPARK-064: a permanent change passes every battery, or it does not land.

The tests that matter here are the refusals. A promotion gate is only worth
having if it cannot be walked past by omission, so most of this file tries to
get an unmeasured candidate promoted and asserts that it cannot.
"""

from __future__ import annotations

import hashlib

import pytest

from core.learning.permanent_distillation import (
    ADMIT,
    FAIL,
    GENESIS_PARENT,
    PASS,
    REFUSE,
    REQUIRED_GATES,
    PermanentDistillationError,
    PermanentDistillationRefusalError,
    active_artifact,
    artifact_manifest,
    baseline_generation,
    evaluate_promotion,
    gate_report,
    gate_result,
    observed_artifact_manifest,
    promote_generation,
    rollback_generation,
    validate_lineage,
)

_NOW = 1_780_000_000


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _artifact(tag: str = "candidate") -> dict:
    return artifact_manifest(
        artifact_id=f"recurrent-{tag}",
        base_model_identity="resident-32b@sha256:" + _digest("base"),
        adapter_identity=f"rlc-adapter-{tag}",
        files=[
            {"name": "adapter.safetensors", "sha256": _digest(tag), "size_bytes": 1024},
            {"name": "adapter_config.json", "sha256": _digest(tag + "cfg"), "size_bytes": 64},
        ],
    )


def _gates(**overrides) -> dict:
    rows = []
    for gate in REQUIRED_GATES:
        graded = overrides.get(f"{gate}_graded", 64)
        verdict = overrides.get(f"{gate}_verdict", PASS)
        rows.append(
            gate_result(
                gate=gate,
                battery_schema=f"aura.{gate}.v1",
                probes_graded=graded,
                probes_passed=graded if verdict == PASS else graded - 1,
                verdict=verdict,
                evidence_sha256=_digest(gate),
            )
        )
    return gate_report(rows)


def _lineage() -> list[dict]:
    return [
        baseline_generation(
            artifact=_artifact("frozen"),
            provenance={"source": "spark004_frozen_baseline"},
            created_at_unix=_NOW,
        )
    ]


# --- the gate set is complete by declaration -------------------------------


def test_a_full_gate_set_admits_a_distinct_candidate():
    decision = evaluate_promotion(
        report=_gates(),
        candidate_artifact=_artifact(),
        incumbent_artifact=_artifact("frozen"),
    )
    assert decision["decision"] == ADMIT
    assert decision["refusals"] == []


@pytest.mark.parametrize("dropped", REQUIRED_GATES)
def test_a_missing_battery_is_an_invalid_report_not_a_passing_one(dropped):
    rows = [row for row in _gates()["gates"] if row["gate"] != dropped]
    with pytest.raises(PermanentDistillationError) as excinfo:
        gate_report(rows)
    assert "gate_set_incomplete" in str(excinfo.value)


def test_an_unknown_extra_gate_cannot_stand_in_for_a_required_one():
    rows = [row for row in _gates()["gates"] if row["gate"] != "memory_retention"]
    rows.append(
        {
            "gate": "memory_retention_lite",
            "battery_schema": "aura.substitute.v1",
            "probes_graded": 64,
            "probes_passed": 64,
            "verdict": PASS,
            "evidence_sha256": _digest("substitute"),
        }
    )
    with pytest.raises(PermanentDistillationError):
        gate_report(rows)


def test_a_battery_that_graded_nothing_refuses_the_promotion():
    decision = evaluate_promotion(
        report=_gates(memory_retention_graded=0),
        candidate_artifact=_artifact(),
        incumbent_artifact=_artifact("frozen"),
    )
    assert decision["decision"] == REFUSE
    assert decision["refusals"] == [
        {
            "gate": "memory_retention",
            "reason": "gate_did_not_measure",
            "probes_graded": 0,
            "probes_required": 4,
        }
    ]


def test_every_failing_family_is_named_in_the_refusal():
    decision = evaluate_promotion(
        report=_gates(
            authority_safety_verdict=FAIL,
            frontier_regression_verdict=FAIL,
        ),
        candidate_artifact=_artifact(),
        incumbent_artifact=_artifact("frozen"),
    )
    assert decision["decision"] == REFUSE
    assert [row["gate"] for row in decision["refusals"]] == [
        "authority_safety",
        "frontier_regression",
    ]
    assert {row["reason"] for row in decision["refusals"]} == {"gate_failed"}


def test_promoting_the_incumbent_bytes_again_is_refused():
    decision = evaluate_promotion(
        report=_gates(),
        candidate_artifact=_artifact("frozen"),
        incumbent_artifact=_artifact("frozen"),
    )
    assert decision["decision"] == REFUSE
    assert decision["refusals"][0]["gate"] == "artifact_identity"


def test_a_refused_promotion_raises_with_the_decision_attached():
    with pytest.raises(PermanentDistillationRefusalError) as excinfo:
        promote_generation(
            lineage=_lineage(),
            artifact=_artifact(),
            report=_gates(personality_retention_verdict=FAIL),
            provenance={"campaign": "cp999"},
            created_at_unix=_NOW + 10,
        )
    assert excinfo.value.decision["decision"] == REFUSE
    assert excinfo.value.decision["refusals"][0]["gate"] == "personality_retention"


# --- versioned lineage ------------------------------------------------------


def test_a_promotion_extends_the_chain_and_binds_its_evidence():
    lineage = _lineage()
    promoted = promote_generation(
        lineage=lineage,
        artifact=_artifact(),
        report=_gates(),
        provenance={"campaign": "cp999"},
        created_at_unix=_NOW + 10,
    )
    assert promoted["generation_index"] == 1
    assert promoted["parent_generation_sha256"] == lineage[0]["generation_sha256"]
    assert promoted["gate_report_sha256"] == _gates()["gate_report_sha256"]
    chain = validate_lineage([*lineage, promoted])
    assert active_artifact(chain)["artifact_id"] == "recurrent-candidate"


def test_the_genesis_record_is_a_baseline_with_no_parent():
    lineage = _lineage()
    assert lineage[0]["parent_generation_sha256"] == GENESIS_PARENT
    assert validate_lineage(lineage)[0]["kind"] == "baseline"


def test_a_reordered_chain_is_refused():
    lineage = _lineage()
    promoted = promote_generation(
        lineage=lineage,
        artifact=_artifact(),
        report=_gates(),
        provenance={"campaign": "cp999"},
        created_at_unix=_NOW + 10,
    )
    with pytest.raises(PermanentDistillationError):
        validate_lineage([promoted, lineage[0]])


def test_a_promotion_record_stripped_of_its_gate_evidence_is_refused():
    lineage = _lineage()
    promoted = dict(
        promote_generation(
            lineage=lineage,
            artifact=_artifact(),
            report=_gates(),
            provenance={"campaign": "cp999"},
            created_at_unix=_NOW + 10,
        )
    )
    promoted["gate_report_sha256"] = None
    with pytest.raises(PermanentDistillationError):
        validate_lineage([*lineage, promoted])


def test_a_tampered_artifact_breaks_the_generation_digest():
    lineage = _lineage()
    promoted = dict(
        promote_generation(
            lineage=lineage,
            artifact=_artifact(),
            report=_gates(),
            provenance={"campaign": "cp999"},
            created_at_unix=_NOW + 10,
        )
    )
    swapped = dict(promoted["artifact"])
    swapped["adapter_identity"] = "rlc-adapter-other"
    promoted["artifact"] = swapped
    with pytest.raises(PermanentDistillationError):
        validate_lineage([*lineage, promoted])


# --- exact rollback ---------------------------------------------------------


def test_rollback_to_the_frozen_baseline_restores_the_exact_artifact():
    lineage = _lineage()
    promoted = promote_generation(
        lineage=lineage,
        artifact=_artifact(),
        report=_gates(),
        provenance={"campaign": "cp999"},
        created_at_unix=_NOW + 10,
    )
    chain = [*lineage, promoted]
    reverted = rollback_generation(
        lineage=chain,
        restores_generation_sha256=lineage[0]["generation_sha256"],
        observed_artifact=_artifact("frozen"),
        provenance={"reason": "frontier_regression_after_promotion"},
        created_at_unix=_NOW + 20,
    )
    replayed = validate_lineage([*chain, reverted])
    assert replayed[-1]["kind"] == "rollback"
    assert active_artifact(replayed) == lineage[0]["artifact"]


def test_a_rollback_that_lands_different_bytes_is_refused():
    lineage = _lineage()
    promoted = promote_generation(
        lineage=lineage,
        artifact=_artifact(),
        report=_gates(),
        provenance={"campaign": "cp999"},
        created_at_unix=_NOW + 10,
    )
    drifted = artifact_manifest(
        artifact_id="recurrent-frozen",
        base_model_identity="resident-32b@sha256:" + _digest("base"),
        adapter_identity="rlc-adapter-frozen",
        files=[
            {"name": "adapter.safetensors", "sha256": _digest("frozen"), "size_bytes": 1024},
            {"name": "adapter_config.json", "sha256": _digest("drifted"), "size_bytes": 64},
        ],
    )
    with pytest.raises(PermanentDistillationError) as excinfo:
        rollback_generation(
            lineage=[*lineage, promoted],
            restores_generation_sha256=lineage[0]["generation_sha256"],
            observed_artifact=drifted,
            provenance={"reason": "attempted"},
            created_at_unix=_NOW + 20,
        )
    assert "rollback_not_exact" in str(excinfo.value)


def test_a_rollback_to_an_unknown_generation_is_refused():
    lineage = _lineage()
    with pytest.raises(PermanentDistillationError):
        rollback_generation(
            lineage=lineage,
            restores_generation_sha256=_digest("nowhere"),
            observed_artifact=_artifact("frozen"),
            provenance={},
            created_at_unix=_NOW + 20,
        )


def test_a_rollback_to_the_current_head_is_refused_as_a_no_op():
    lineage = _lineage()
    with pytest.raises(PermanentDistillationError):
        rollback_generation(
            lineage=lineage,
            restores_generation_sha256=lineage[0]["generation_sha256"],
            observed_artifact=_artifact("frozen"),
            provenance={},
            created_at_unix=_NOW + 20,
        )


def test_a_recorded_rollback_cannot_claim_bytes_it_did_not_restore():
    lineage = _lineage()
    promoted = promote_generation(
        lineage=lineage,
        artifact=_artifact(),
        report=_gates(),
        provenance={"campaign": "cp999"},
        created_at_unix=_NOW + 10,
    )
    chain = [*lineage, promoted]
    reverted = dict(
        rollback_generation(
            lineage=chain,
            restores_generation_sha256=lineage[0]["generation_sha256"],
            observed_artifact=_artifact("frozen"),
            provenance={},
            created_at_unix=_NOW + 20,
        )
    )
    reverted["artifact"] = promoted["artifact"]
    with pytest.raises(PermanentDistillationError):
        validate_lineage([*chain, reverted])


# --- observed manifests come off the disk, not off a promise ---------------


def test_the_observed_manifest_reads_the_restored_bytes(tmp_path):
    (tmp_path / "adapter.safetensors").write_bytes(b"weights-v1")
    (tmp_path / "adapter_config.json").write_bytes(b"{}")
    observed = observed_artifact_manifest(
        artifact_id="recurrent-frozen",
        base_model_identity="resident-32b",
        adapter_identity="rlc-adapter-frozen",
        root=tmp_path,
        names=["adapter.safetensors", "adapter_config.json"],
    )
    weights = next(
        row for row in observed["files"] if row["name"] == "adapter.safetensors"
    )
    assert weights["sha256"] == hashlib.sha256(b"weights-v1").hexdigest()
    assert weights["size_bytes"] == len(b"weights-v1")


def test_a_restore_that_changed_one_byte_produces_a_different_manifest(tmp_path):
    (tmp_path / "adapter.safetensors").write_bytes(b"weights-v1")
    before = observed_artifact_manifest(
        artifact_id="a",
        base_model_identity="b",
        adapter_identity="c",
        root=tmp_path,
        names=["adapter.safetensors"],
    )
    (tmp_path / "adapter.safetensors").write_bytes(b"weights-v2")
    after = observed_artifact_manifest(
        artifact_id="a",
        base_model_identity="b",
        adapter_identity="c",
        root=tmp_path,
        names=["adapter.safetensors"],
    )
    assert before["artifact_sha256"] != after["artifact_sha256"]


def test_a_missing_restored_file_is_refused(tmp_path):
    with pytest.raises(PermanentDistillationError):
        observed_artifact_manifest(
            artifact_id="a",
            base_model_identity="b",
            adapter_identity="c",
            root=tmp_path,
            names=["adapter.safetensors"],
        )
