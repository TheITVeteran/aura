"""Contracts for unattended signed latent-cortex campaign advancement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.campaign_trust import CAMPAIGN_RUNNER, TASK_ISSUER
from tools import run_latent_cortex_campaign_controller as controller


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


class _Broker:
    def __init__(self, role: str, calls: list[tuple[str, dict[str, Any]]]) -> None:
        self.role = role
        self.calls = calls

    def attest_prepared_request(
        self,
        _policy: object,
        *,
        role: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        assert role == self.role
        self.calls.append((role, request))
        return {
            "schema": "fixture.attestation",
            "signed_payload": request["signed_payload"],
            "request_sha256": request["request_sha256"],
        }


def test_controller_drives_all_signed_phases_without_reconstructing_requests(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    state_dir = tmp_path / "state"
    answer_request = tmp_path / "campaign" / "answer-request.json"
    final_request = tmp_path / "campaign" / "final-request.json"
    answer_document = {
        "request_sha256": "a" * 64,
        "signed_payload": {"purpose": "campaign:answer-reveal"},
    }
    final_document = {
        "request_sha256": "b" * 64,
        "signed_payload": {"purpose": "campaign:final-run"},
    }
    _write(answer_request, answer_document)
    _write(final_request, final_document)
    calls: list[tuple[str, dict[str, Any]]] = []
    phase = {"value": "ready_for_inference"}

    def fake_status(_args: argparse.Namespace) -> dict[str, Any]:
        value = phase["value"]
        result: dict[str, Any] = {"phase": value}
        if value == "awaiting_answer_reveal_signature":
            result["request_path"] = str(answer_request)
        elif value == "awaiting_final_run_signature":
            result["request_path"] = str(final_request)
        elif value == "campaign_evidence_sealed":
            result["envelope_sha256"] = "c" * 64
        return result

    def fake_execute(**kwargs: Any) -> int:
        packet = Path(kwargs["packet_path"]).name
        if packet == "launch_packet.json":
            phase["value"] = "awaiting_answer_reveal_signature"
            return 5
        if packet == controller.advancement.ANSWER_RESUME_PACKET_FILE:
            phase["value"] = "awaiting_final_run_signature"
            return 6
        assert packet == controller.advancement.FINAL_RESUME_PACKET_FILE
        phase["value"] = "campaign_evidence_sealed"
        return 2

    def fake_admit(args: argparse.Namespace) -> dict[str, Any]:
        request_hash = json.loads(args.attestation.read_text())["request_sha256"]
        packet = (
            controller.advancement.ANSWER_RESUME_PACKET_FILE
            if request_hash == "a" * 64
            else controller.advancement.FINAL_RESUME_PACKET_FILE
        )
        return {"packet_path": str(bundle / packet)}

    monkeypatch.setattr(
        controller.advancement,
        "_context",
        lambda _bundle: {"launch_spec": {}},
    )
    monkeypatch.setattr(
        controller.advancement,
        "_policy",
        lambda _launch, observed_at=None: object(),
    )
    monkeypatch.setattr(controller.advancement, "status", fake_status)
    monkeypatch.setattr(controller.advancement, "admit", fake_admit)
    monkeypatch.setattr(controller, "_execute_packet", fake_execute)
    brokers = {
        TASK_ISSUER: _Broker(TASK_ISSUER, calls),
        CAMPAIGN_RUNNER: _Broker(CAMPAIGN_RUNNER, calls),
    }
    monkeypatch.setattr(
        controller,
        "_load_broker",
        lambda _path, *, policy, role: brokers[role],
    )
    monkeypatch.setattr(controller, "_stop_signal", 0)

    result = controller.run_controller(
        argparse.Namespace(
            bundle_dir=bundle,
            state_dir=state_dir,
            task_issuer_signer_config=tmp_path / "issuer.json",
            campaign_runner_signer_config=tmp_path / "runner.json",
            heartbeat_seconds=0.25,
            max_phase_executions=8,
        )
    )

    assert result["phase"] == "campaign_evidence_sealed"
    assert [role for role, _request in calls] == [TASK_ISSUER, CAMPAIGN_RUNNER]
    assert calls[0][1] == answer_document
    assert calls[1][1] == final_document
    status = json.loads((state_dir / "status.json").read_text())
    assert status["state"] == "complete"
    assert status["sequence"] == 3


def test_controller_records_fail_closed_phase_failure(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    state_dir = tmp_path / "state"
    monkeypatch.setattr(
        controller.advancement,
        "_context",
        lambda _bundle: {"launch_spec": {}},
    )
    monkeypatch.setattr(
        controller.advancement,
        "_policy",
        lambda _launch, observed_at=None: object(),
    )
    monkeypatch.setattr(
        controller,
        "_load_broker",
        lambda _path, *, policy, role: _Broker(role, []),
    )
    monkeypatch.setattr(
        controller.advancement,
        "status",
        lambda _args: {"phase": "unknown_untrusted_phase"},
    )
    monkeypatch.setattr(controller, "_stop_signal", 0)

    try:
        controller.run_controller(
            argparse.Namespace(
                bundle_dir=bundle,
                state_dir=state_dir,
                task_issuer_signer_config=tmp_path / "issuer.json",
                campaign_runner_signer_config=tmp_path / "runner.json",
                heartbeat_seconds=0.25,
                max_phase_executions=8,
            )
        )
    except controller.CampaignControllerError as exc:
        assert exc.code == "campaign_phase_unsupported:unknown_untrusted_phase"
    else:
        raise AssertionError("controller accepted an unknown phase")

    status = json.loads((state_dir / "status.json").read_text())
    assert status["state"] == "failed"
    assert status["reason"] == "campaign_phase_unsupported:unknown_untrusted_phase"
