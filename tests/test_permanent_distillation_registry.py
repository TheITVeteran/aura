"""SPARK-064: the stored lineage may grow, and it may not be edited."""

from __future__ import annotations

import hashlib
import json

import pytest

from core.learning.permanent_distillation import (
    PASS,
    REQUIRED_GATES,
    PermanentDistillationError,
    artifact_manifest,
    baseline_generation,
    gate_report,
    gate_result,
    promote_generation,
    rollback_generation,
)
from core.learning.permanent_distillation_registry import (
    append_generation,
    load_lineage,
    write_lineage,
)

_NOW = 1_780_000_000


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _artifact(tag: str) -> dict:
    return artifact_manifest(
        artifact_id=f"recurrent-{tag}",
        base_model_identity="resident-32b",
        adapter_identity=f"rlc-adapter-{tag}",
        files=[{"name": "adapter.safetensors", "sha256": _digest(tag), "size_bytes": 8}],
    )


def _gates() -> dict:
    return gate_report(
        [
            gate_result(
                gate=gate,
                battery_schema=f"aura.{gate}.v1",
                probes_graded=64,
                probes_passed=64,
                verdict=PASS,
                evidence_sha256=_digest(gate),
            )
            for gate in REQUIRED_GATES
        ]
    )


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
    return tmp_path / "lineage" / "permanent_distillation.json"


def test_a_written_lineage_reads_back_identically(registry):
    lineage = [
        baseline_generation(
            artifact=_artifact("frozen"), provenance={}, created_at_unix=_NOW
        )
    ]
    write_lineage(registry, lineage)
    assert load_lineage(registry) == lineage


def test_appending_a_promotion_extends_the_head(registry):
    baseline = baseline_generation(
        artifact=_artifact("frozen"), provenance={}, created_at_unix=_NOW
    )
    write_lineage(registry, [baseline])
    promoted = promote_generation(
        lineage=[baseline],
        artifact=_artifact("trained"),
        report=_gates(),
        provenance={"campaign": "cp999"},
        created_at_unix=_NOW + 10,
    )
    head = append_generation(registry, promoted)
    stored = load_lineage(registry)
    assert head == promoted["generation_sha256"]
    assert [row["kind"] for row in stored] == ["baseline", "promotion"]


def test_a_rollback_keeps_the_promotion_it_reverts(registry):
    baseline = baseline_generation(
        artifact=_artifact("frozen"), provenance={}, created_at_unix=_NOW
    )
    write_lineage(registry, [baseline])
    promoted = promote_generation(
        lineage=[baseline],
        artifact=_artifact("trained"),
        report=_gates(),
        provenance={"campaign": "cp999"},
        created_at_unix=_NOW + 10,
    )
    append_generation(registry, promoted)
    reverted = rollback_generation(
        lineage=[baseline, promoted],
        restores_generation_sha256=baseline["generation_sha256"],
        observed_artifact=_artifact("frozen"),
        provenance={"reason": "regression"},
        created_at_unix=_NOW + 20,
    )
    append_generation(registry, reverted)
    stored = load_lineage(registry)
    assert [row["kind"] for row in stored] == ["baseline", "promotion", "rollback"]
    assert stored[1] == promoted


def test_rewriting_a_stored_generation_is_refused(registry):
    baseline = baseline_generation(
        artifact=_artifact("frozen"), provenance={}, created_at_unix=_NOW
    )
    write_lineage(registry, [baseline])
    replacement = baseline_generation(
        artifact=_artifact("other"), provenance={}, created_at_unix=_NOW
    )
    promoted = promote_generation(
        lineage=[replacement],
        artifact=_artifact("trained"),
        report=_gates(),
        provenance={},
        created_at_unix=_NOW + 10,
    )
    with pytest.raises(PermanentDistillationError) as excinfo:
        write_lineage(registry, [replacement, promoted])
    assert "rewrites_history" in str(excinfo.value)


def test_truncating_the_lineage_is_refused(registry):
    baseline = baseline_generation(
        artifact=_artifact("frozen"), provenance={}, created_at_unix=_NOW
    )
    promoted = promote_generation(
        lineage=[baseline],
        artifact=_artifact("trained"),
        report=_gates(),
        provenance={},
        created_at_unix=_NOW + 10,
    )
    write_lineage(registry, [baseline, promoted])
    with pytest.raises(PermanentDistillationError) as excinfo:
        write_lineage(registry, [baseline])
    assert "truncated" in str(excinfo.value)


def test_an_edited_registry_file_does_not_load(registry):
    baseline = baseline_generation(
        artifact=_artifact("frozen"), provenance={}, created_at_unix=_NOW
    )
    write_lineage(registry, [baseline])
    document = json.loads(registry.read_text(encoding="utf-8"))
    document["generations"][0]["provenance"] = {"tampered": True}
    registry.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(PermanentDistillationError):
        load_lineage(registry)


def test_a_registry_head_that_disagrees_with_its_records_is_refused(registry):
    baseline = baseline_generation(
        artifact=_artifact("frozen"), provenance={}, created_at_unix=_NOW
    )
    write_lineage(registry, [baseline])
    document = json.loads(registry.read_text(encoding="utf-8"))
    document["head_generation_sha256"] = _digest("elsewhere")
    registry.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(PermanentDistillationError) as excinfo:
        load_lineage(registry)
    assert "head_differs" in str(excinfo.value)
