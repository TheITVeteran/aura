from __future__ import annotations

from pathlib import Path

import pytest

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from tools import watch_resident_v3_confirmatory_gate as watcher


def _pipeline_config(tmp_path: Path) -> tuple[Path, dict]:
    pipeline_root = tmp_path / "pipeline"
    pipeline_root.mkdir()
    material = {
        "schema": watcher.PIPELINE_CONFIG_SCHEMA,
        "output_root": str(pipeline_root),
        "recovery": {"mode": "migration_recovery"},
    }
    document = {**material, "config_sha256": watcher._document_sha(material)}
    path = tmp_path / "pipeline_config.json"
    path.write_bytes(canonical_json_bytes(document) + b"\n")
    return path, document


def _watcher_config(tmp_path: Path, pipeline: dict) -> dict:
    return {
        "config_sha256": "a" * 64,
        "pipeline_root": pipeline["output_root"],
        "pipeline_config": {"path": str(tmp_path / "pipeline_config.json")},
        "source_commit": "b" * 40,
        "confirmatory_design": {
            "exact_power": watcher._power_plan(),
            "arms": [
                "base_vanilla",
                "base_rlc",
                "adapter_vanilla",
                "adapter_rlc",
                "base_equal_compute",
                "adapter_equal_compute",
            ],
        },
    }


def _pipeline_verdict(pipeline: dict, *, decision: str, advance: bool) -> dict:
    material = {
        "schema": watcher.PIPELINE_VERDICT_SCHEMA,
        "decision": decision,
        "pipeline_completed": True,
        "directional_gain_gate_passed": advance,
        "reasoning_gain_proven": False,
        "frontier_level_proven": False,
        "external_attestation_present": False,
        "failure_points": [] if advance else ["adapter_rlc_not_better"],
        "required_next_gate": (
            "powered_external_frontier_campaign"
            if advance
            else "repair_and_preregister_recurrence_v3_directional_pilot"
        ),
        "config_sha256": pipeline["config_sha256"],
    }
    return {**material, "verdict_sha256": watcher._document_sha(material)}


def test_build_config_binds_exact_powered_design(tmp_path: Path) -> None:
    pipeline_path, _pipeline = _pipeline_config(tmp_path)

    config = watcher.build_config(
        pipeline_config_path=pipeline_path,
        output_root=tmp_path / "confirmatory",
        source_commit="c" * 40,
    )

    power = config["confirmatory_design"]["exact_power"]
    assert power["minimum_observations"] == 411
    assert power["planned_total_tasks"] == 2_877
    assert power["planned_total_cells"] == 17_262
    assert power["powered_for_zero_loss_noninferiority"] is True
    assert len(config["confirmatory_design"]["arms"]) == 6
    assert config["confirmatory_design"]["exact_power_scope"] == (
        "zero_loss_noninferiority_floor_only"
    )
    assert config["confirmatory_design"][
        "positive_interaction_power_simulation_required"
    ] is True
    material = dict(config)
    claimed = material.pop("config_sha256")
    assert claimed == watcher._document_sha(material)


def test_negative_directional_result_seals_repair_verdict(tmp_path: Path) -> None:
    _path, pipeline = _pipeline_config(tmp_path)
    config = _watcher_config(tmp_path, pipeline)
    pipeline_verdict = _pipeline_verdict(
        pipeline,
        decision="directional_gate_failed_frontier_gain_not_proven",
        advance=False,
    )

    verdict, handoff = watcher.evaluate(config, pipeline, pipeline_verdict)

    assert handoff is None
    assert verdict["decision"] == "directional_gate_not_admitted_to_confirmatory_campaign"
    assert verdict["failure_points"] == ["adapter_rlc_not_better"]
    assert verdict["reasoning_gain_proven"] is False
    assert verdict["frontier_level_proven"] is False


def test_positive_directional_result_requires_external_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _path, pipeline = _pipeline_config(tmp_path)
    config = _watcher_config(tmp_path, pipeline)
    pipeline_verdict = _pipeline_verdict(
        pipeline,
        decision="directional_gate_passed_external_frontier_proof_pending",
        advance=True,
    )
    monkeypatch.setattr(
        watcher,
        "_artifact_set",
        lambda _root: {
            "pilot_result": {
                "binding": {"path": "/evidence/pilot.json", "sha256": "d" * 64},
                "document": {
                    "pilot_advance_gate_passed": True,
                    "decision": "advance_to_powered_external_frontier_campaign",
                    "reasoning_gain_proven": False,
                    "frontier_gain_proven": False,
                },
            },
            "frozen_adapter": {
                "path": "/evidence/frozen",
                "certificate_sha256": "e" * 64,
            },
        },
    )

    verdict, handoff = watcher.evaluate(config, pipeline, pipeline_verdict)

    assert handoff is not None
    assert verdict["decision"] == "external_custody_required"
    assert verdict["external_attestation_present"] is False
    assert verdict["frontier_level_proven"] is False
    assert handoff["status"] == "external_custody_required"
    assert handoff["claim_authorized"] is False
    assert "post_evidence_independent_verifier_attestation" in handoff["required_inputs"]


def test_rehashed_positive_claim_is_rejected(tmp_path: Path) -> None:
    _path, pipeline = _pipeline_config(tmp_path)
    config = _watcher_config(tmp_path, pipeline)
    verdict = _pipeline_verdict(
        pipeline,
        decision="directional_gate_passed_external_frontier_proof_pending",
        advance=True,
    )
    verdict["reasoning_gain_proven"] = True
    material = dict(verdict)
    material.pop("verdict_sha256")
    verdict["verdict_sha256"] = watcher._document_sha(material)

    with pytest.raises(
        watcher.ConfirmatoryWatcherError,
        match="positive_pipeline_verdict_invalid",
    ):
        watcher.evaluate(config, pipeline, verdict)
