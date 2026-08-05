from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from core.reality_reach.acceptance import (
    REQUIRED_SCALAR_ACCEPTANCE_CASES,
    AcceptanceCaseResult,
    AcceptanceEvidenceClass,
    AcceptanceVerdict,
    ConnectorAcceptanceCertificate,
)
from core.reality_reach.acceptance_mandate import (
    AcceptanceMandateError,
    AcceptanceMandateStore,
)
from core.reality_reach.acceptance_verifier import (
    persist_mandated_verification_receipt,
    verify_acceptance_against_mandate,
)
from core.reality_reach.trust_custody import AttachmentTrustStoreError
from core.runtime.audit_chain import canonical_json, sha256_hex


class _FakeKeychain:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, account: str) -> str | None:
        return self.values.get((service, account))

    def set_password(self, service: str, account: str, password: str) -> bool:
        self.values[(service, account)] = password
        return True


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


def _store(path: Path, backend: _FakeKeychain) -> AcceptanceMandateStore:
    return AcceptanceMandateStore.for_backend(backend, path)


def _provision(
    store: AcceptanceMandateStore,
    *,
    target: float = 2.0,
) -> Any:
    return store.provision(
        campaign_id="campaign.precommitted",
        connector_id="connector.fixture",
        adapter_id="adapter.fixture",
        expected_source_commit_sha256=_digest("source"),
        expected_physical_identity_sha256=_digest("device"),
        expected_evidence_class=AcceptanceEvidenceClass.SIMULATION,
        target=target,
        target_tolerance=0.05,
        scenario_id="",
    )


def _evidence() -> dict[str, dict[str, Any]]:
    prepare = {
        "preparation_id": "prep-1",
        "command_sha256": _digest("command"),
        "lease_sha256": _digest("lease"),
        "precondition_sha256": _digest("precondition"),
        "rollback_token_sha256": _digest("rollback-token"),
    }
    dispatch = {
        "state": "executed",
        "accepted": True,
        "transport_completed": True,
        "executed": True,
        "command_sha256": prepare["command_sha256"],
        "preparation_sha256": _digest(prepare),
    }
    return {
        "observation.fresh": {
            "status": "available",
            "value": 1.0,
            "source_event_id": "event-1",
        },
        "cancellation.pre_dispatch": {
            "state": "cancelled",
            "executed": False,
            "transport_completed": False,
        },
        "actuation.prepare": prepare,
        "actuation.dispatch": dispatch,
        "effect.independent_readback": {
            "state": "effect_verified",
            "independently_observed": True,
            "observation_sha256": _digest("effect"),
            "command_sha256": prepare["command_sha256"],
            "actuation_receipt_sha256": _digest(dispatch),
        },
        "restoration.rollback": {
            "state": "rolled_back",
            "independently_observed": True,
            "safe_state_observation_sha256": _digest("safe-state"),
            "command_sha256": prepare["command_sha256"],
            "actuation_receipt_sha256": _digest(dispatch),
        },
    }


def _certificate(
    mandate: Any,
    *,
    started_at_ns: int | None = None,
) -> tuple[ConnectorAcceptanceCertificate, dict[str, Any]]:
    evidence = _evidence()
    cases = tuple(
        AcceptanceCaseResult(
            case_id=case_id,
            verdict=AcceptanceVerdict.PASS,
            evidence_class=AcceptanceEvidenceClass.SIMULATION,
            required=True,
            evidence_sha256=_digest(evidence[case_id]),
            duration_ms=1.0,
        )
        for case_id in REQUIRED_SCALAR_ACCEPTANCE_CASES
    )
    started = started_at_ns or mandate.provisioned_at_ns + 1
    certificate = ConnectorAcceptanceCertificate(
        campaign_id=mandate.campaign_id,
        connector_id=mandate.connector_id,
        adapter_id=mandate.adapter_id,
        physical_identity_sha256=mandate.expected_physical_identity_sha256,
        source_commit_sha256=mandate.expected_source_commit_sha256,
        target=mandate.target,
        target_tolerance=mandate.target_tolerance,
        started_at_ns=started,
        completed_at_ns=started + 1,
        cases=cases,
    )
    return certificate, {"case_evidence": evidence}


def test_mandate_is_ciphertext_only_restartable_and_idempotent(tmp_path: Path) -> None:
    backend = _FakeKeychain()
    state_path = tmp_path / "mandates.json"
    store = _store(state_path, backend)

    first = _provision(store)
    second = _provision(store)

    encoded = state_path.read_text(encoding="utf-8")
    assert "campaign.precommitted" not in encoded
    assert "adapter.fixture" not in encoded
    assert first.created is True
    assert first.custody_sequence == 1
    assert second.created is False
    assert second.mandate_sha256 == first.mandate_sha256
    assert second.provisioned_at_ns == first.provisioned_at_ns
    restarted = AcceptanceMandateStore.for_backend(
        backend,
        state_path,
        create_if_missing=False,
    )
    assert restarted.get("campaign.precommitted").sha256 == first.mandate_sha256


def test_conflicting_reprovision_is_refused_without_advancing_state(tmp_path: Path) -> None:
    backend = _FakeKeychain()
    store = _store(tmp_path / "mandates.json", backend)
    first = _provision(store)

    with pytest.raises(AcceptanceMandateError, match="acceptance_mandate_conflict"):
        _provision(store, target=3.0)

    assert store.get(first.campaign_id).target == 2.0
    assert store.status()["custody"]["committed_sequence"] == 1


def test_valid_older_mandate_envelope_is_refused_as_rollback(tmp_path: Path) -> None:
    backend = _FakeKeychain()
    state_path = tmp_path / "mandates.json"
    store = _store(state_path, backend)
    _provision(store)
    first_envelope = state_path.read_bytes()
    store.provision(
        campaign_id="campaign.second",
        connector_id="connector.fixture",
        adapter_id="adapter.fixture",
        expected_source_commit_sha256=_digest("source"),
        expected_physical_identity_sha256=_digest("device"),
        expected_evidence_class=AcceptanceEvidenceClass.SIMULATION,
        target=2.0,
        target_tolerance=0.05,
        scenario_id="",
    )
    state_path.write_bytes(first_envelope)

    with pytest.raises(
        AttachmentTrustStoreError,
        match="rollback_or_replay_refused",
    ):
        store.get("campaign.precommitted")


def test_verifier_accepts_only_the_precommitted_question(tmp_path: Path) -> None:
    store = _store(tmp_path / "mandates.json", _FakeKeychain())
    _provision(store)
    mandate = store.get("campaign.precommitted")
    certificate, evidence = _certificate(mandate)

    receipt = verify_acceptance_against_mandate(certificate, evidence, mandate)

    assert receipt.accepted is True
    assert receipt.blockers == ()
    assert receipt.verification.accepted is True
    assert receipt.mandate_sha256 == mandate.sha256

    substituted = verify_acceptance_against_mandate(
        replace(
            certificate,
            adapter_id="adapter.substituted",
            target=3.0,
            target_tolerance=0.5,
        ),
        evidence,
        mandate,
    )
    assert substituted.accepted is False
    assert set(substituted.blockers) == {
        "mandate_adapter_mismatch",
        "mandate_target_mismatch",
        "mandate_target_tolerance_mismatch",
    }


@pytest.mark.parametrize(
    ("changes", "expected_blocker"),
    (
        ({"campaign_id": "campaign.substituted"}, "mandate_campaign_mismatch"),
        ({"connector_id": "connector.substituted"}, "mandate_connector_mismatch"),
        ({"scenario_id": "scenario.substituted"}, "mandate_scenario_mismatch"),
        (
            {"source_commit_sha256": _digest("other-source")},
            "mandate_source_commit_mismatch",
        ),
        (
            {"physical_identity_sha256": _digest("other-device")},
            "mandate_physical_identity_mismatch",
        ),
    ),
)
def test_verifier_rejects_identity_substitution(
    tmp_path: Path,
    changes: dict[str, Any],
    expected_blocker: str,
) -> None:
    store = _store(tmp_path / "mandates.json", _FakeKeychain())
    _provision(store)
    mandate = store.get("campaign.precommitted")
    certificate, evidence = _certificate(mandate)

    receipt = verify_acceptance_against_mandate(
        replace(certificate, **changes),
        evidence,
        mandate,
    )

    assert receipt.accepted is False
    assert expected_blocker in receipt.blockers


def test_verifier_rejects_evidence_class_and_case_set_substitution(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "mandates.json", _FakeKeychain())
    _provision(store)
    mandate = store.get("campaign.precommitted")
    certificate, evidence = _certificate(mandate)
    live_cases = tuple(
        replace(item, evidence_class=AcceptanceEvidenceClass.LIVE)
        for item in certificate.cases
    )

    wrong_class = verify_acceptance_against_mandate(
        replace(certificate, cases=live_cases),
        evidence,
        mandate,
    )
    wrong_cases = verify_acceptance_against_mandate(
        replace(certificate, cases=tuple(reversed(certificate.cases))),
        evidence,
        mandate,
    )

    assert wrong_class.accepted is False
    assert "mandate_evidence_class_mismatch" in wrong_class.blockers
    assert wrong_cases.accepted is False
    assert "mandate_required_case_set_mismatch" in wrong_cases.blockers


def test_campaign_evidence_cannot_predate_the_mandate(tmp_path: Path) -> None:
    store = _store(tmp_path / "mandates.json", _FakeKeychain())
    _provision(store)
    mandate = store.get("campaign.precommitted")
    certificate, evidence = _certificate(
        mandate,
        started_at_ns=mandate.provisioned_at_ns - 1,
    )

    receipt = verify_acceptance_against_mandate(certificate, evidence, mandate)

    assert receipt.accepted is False
    assert receipt.blockers == ("campaign_predates_mandate",)


def test_mandated_verification_receipt_is_private_and_create_once(tmp_path: Path) -> None:
    store = _store(tmp_path / "mandates.json", _FakeKeychain())
    _provision(store)
    mandate = store.get("campaign.precommitted")
    certificate, evidence = _certificate(mandate)
    receipt = verify_acceptance_against_mandate(certificate, evidence, mandate)
    receipt_path = tmp_path / "receipts" / "mandated.json"

    assert persist_mandated_verification_receipt(receipt, receipt_path) is True
    assert persist_mandated_verification_receipt(receipt, receipt_path) is False
    document = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert document["mandate_sha256"] == mandate.sha256
    assert document["accepted"] is True
    assert receipt_path.stat().st_mode & 0o077 == 0

    attacked = replace(receipt, blockers=("post_result_substitution",))
    with pytest.raises(
        RuntimeError,
        match="acceptance_mandated_verification_receipt_collision",
    ):
        persist_mandated_verification_receipt(attacked, receipt_path)


def test_physical_mandates_require_exact_evidence_topology(tmp_path: Path) -> None:
    store = _store(tmp_path / "mandates.json", _FakeKeychain())
    common = {
        "connector_id": "connector.fixture",
        "adapter_id": "adapter.fixture",
        "expected_source_commit_sha256": _digest("source"),
        "expected_physical_identity_sha256": _digest("device"),
        "target": 2.0,
        "target_tolerance": 0.05,
    }

    with pytest.raises(ValueError, match="requires live channels"):
        store.provision(
            campaign_id="campaign.live-no-readback",
            expected_evidence_class=AcceptanceEvidenceClass.LIVE,
            **common,
        )
    with pytest.raises(ValueError, match="requires simulated channels"):
        store.provision(
            campaign_id="campaign.hil-no-companion",
            expected_evidence_class=AcceptanceEvidenceClass.HARDWARE_IN_LOOP,
            scenario_id="hil.fixture",
            expected_live_channel_ids=("fixture.live",),
            **common,
        )
    receipt = store.provision(
        campaign_id="campaign.hil-complete",
        expected_evidence_class=AcceptanceEvidenceClass.HARDWARE_IN_LOOP,
        scenario_id="hil.fixture",
        expected_live_channel_ids=("fixture.live",),
        expected_simulated_channel_ids=("fixture.simulated",),
        **common,
    )
    mandate = store.get(receipt.campaign_id)
    assert mandate.expected_live_channel_ids == ("fixture.live",)
    assert mandate.expected_simulated_channel_ids == ("fixture.simulated",)
