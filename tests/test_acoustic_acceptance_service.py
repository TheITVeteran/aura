from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

import pytest

from core.embodiment.macos_acoustic_reality import (
    ACOUSTIC_FREQUENCY_HZ,
    ACOUSTIC_RESOURCE_ID,
    AcousticDeviceIdentity,
    MacOSAcousticRealityAdapter,
    MacOSAcousticScalarTransport,
)
from core.reality_reach.acceptance import AcceptanceError
from core.reality_reach.acceptance_mandate import (
    AcceptanceMandateError,
    AcceptanceMandateProvisionReceipt,
    AcceptanceVerificationMandate,
)
from core.reality_reach.acceptance_service import (
    AcousticA1MandateRequest,
    AcousticA1Request,
    RealityAcceptanceService,
)
from core.reality_reach.acoustic_acceptance import AcousticA1CampaignStore
from core.reality_reach.contracts import NumericDomain
from core.reality_reach.live import RealityReachService
from core.reality_reach.metrology import RealityMetrologyService
from core.reality_reach.scalar_adapter import ScalarResourceProfile
from core.runtime.audit_chain import canonical_json, sha256_hex


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


class _MandateStore:
    def __init__(self) -> None:
        self.mandates: dict[str, AcceptanceVerificationMandate] = {}

    def provision(self, **kwargs: Any) -> AcceptanceMandateProvisionReceipt:
        proposed = AcceptanceVerificationMandate(
            **kwargs,
            provisioned_at_ns=time.time_ns(),
            custody_sequence=len(self.mandates) + 1,
        )
        existing = self.mandates.get(proposed.campaign_id)
        if existing is not None and existing.contract_sha256 != proposed.contract_sha256:
            raise AcceptanceMandateError("acceptance_mandate_conflict")
        mandate = existing or proposed
        self.mandates[mandate.campaign_id] = mandate
        return AcceptanceMandateProvisionReceipt(
            campaign_id=mandate.campaign_id,
            mandate_sha256=mandate.sha256,
            contract_sha256=mandate.contract_sha256,
            custody_identity_sha256=_digest("fixture-custody"),
            provisioned_at_ns=mandate.provisioned_at_ns,
            created=existing is None,
            custody_sequence=mandate.custody_sequence,
        )

    def get(self, campaign_id: str) -> AcceptanceVerificationMandate:
        try:
            return self.mandates[campaign_id]
        except KeyError as exc:
            raise AcceptanceMandateError("acceptance_mandate_not_found") from exc

    def status(self) -> dict[str, Any]:
        return {"healthy": True, "mandate_count": len(self.mandates)}

    def close(self) -> None:
        return None


class _AcousticBackend:
    def __init__(self, *, nonlinear: bool) -> None:
        self.nonlinear = nonlinear
        self.calls = 0

    def identity(self) -> AcousticDeviceIdentity:
        return AcousticDeviceIdentity(
            input_name="Fixture microphone",
            output_name="Fixture speaker",
            input_channels=1,
            output_channels=2,
            sample_rate_hz=8_000,
        )

    def play_and_record(
        self,
        signal: Sequence[float],
        *,
        sample_rate_hz: int,
    ) -> Sequence[float]:
        self.calls += 1
        commanded_peak = max((abs(float(value)) for value in signal), default=0.0)
        if commanded_peak <= 1e-8:
            observed_dbfs = -80.0
        elif self.nonlinear:
            observed_dbfs = -80.0 + 60.0 * (commanded_peak / 0.08) ** 0.35
        else:
            observed_dbfs = 20.0 * math.log10(commanded_peak / math.sqrt(2.0))
        observed_peak = math.sqrt(2.0) * 10.0 ** (observed_dbfs / 20.0)
        return tuple(
            observed_peak
            * math.sin(2.0 * math.pi * ACOUSTIC_FREQUENCY_HZ * index / sample_rate_hz)
            for index in range(len(signal))
        )


async def _adapter(backend: _AcousticBackend) -> MacOSAcousticRealityAdapter:
    transport = MacOSAcousticScalarTransport(backend, duration_s=0.1)
    initial = await transport.read_scalar(ACOUSTIC_RESOURCE_ID)
    profile = ScalarResourceProfile(
        resource_id=ACOUSTIC_RESOURCE_ID,
        observable="reference_tone_level",
        unit="dbfs",
        domain=NumericDomain(-100.0, -10.0),
        resolution=0.5,
        writable=True,
        physical_identity_sha256=transport.physical_identity_sha256,
        owner="tests.acoustic_acceptance_service",
        protocol="macos_acoustic",
        safe_value=-80.0,
        tolerance=2.0,
        max_commands_per_minute=8,
        cooldown_s=0.25,
        stale_after_s=5.0,
        readback_distinct_from_command=True,
    )
    return MacOSAcousticRealityAdapter(
        transport,
        profile,
        initial_sample=initial,
    )


async def _governed_executor(**kwargs: Any) -> dict[str, Any]:
    context = {"will_receipt_id": "will.cp810.acoustic"}
    dispatch = dict(await kwargs["effect_handler"](context))
    verification = dict(await kwargs["effect_verifier"](context))
    verified = verification.get("effect_verified") is True
    return {
        **dispatch,
        **verification,
        "action_id": kwargs["action_id"],
        "request_digest": _digest("acoustic-governed-request"),
        "will_receipt_id": context["will_receipt_id"],
        "post_action_receipt_id": "post.cp810.acoustic",
        "post_action_output_hash": _digest("acoustic-governed-output"),
        "status": "success_verified" if verified else "verification_failed",
        "transport_succeeded": True,
        "effect_verified": verified,
        "receipt_persisted": True,
        "welfare_transaction_completed": True,
    }


async def _interrupted_after_effect_executor(**kwargs: Any) -> dict[str, Any]:
    await kwargs["effect_handler"]({"will_receipt_id": "will.cp810.interrupted"})
    raise RuntimeError("fixture_post_effect_interruption")


async def _service(tmp_path, *, nonlinear: bool) -> tuple[
    RealityAcceptanceService,
    _AcousticBackend,
]:
    backend = _AcousticBackend(nonlinear=nonlinear)
    adapter = await _adapter(backend)
    source_identity = {
        "identity_bound": True,
        "source_commit": "a" * 40,
        "source_dirty": False,
        "workspace_state_sha256": "b" * 64,
    }
    reality = RealityReachService((adapter,), session_id="acoustic.a1.fixture")
    metrology = RealityMetrologyService(
        reality,
        state_path=tmp_path / "metrology.json",
    )
    await metrology.start()
    service = RealityAcceptanceService(
        reality,
        metrology,
        mandate_store=_MandateStore(),
        governed_executor=_governed_executor,
        acoustic_campaign_store=AcousticA1CampaignStore(tmp_path / "campaigns"),
        pinned_source_identity=source_identity,
        source_identity_provider=lambda: source_identity,
    )
    return service, backend


@pytest.mark.asyncio
async def test_service_precommits_runs_persists_and_replays_positive_a1(tmp_path) -> None:
    service, backend = await _service(tmp_path, nonlinear=True)
    precommit = await service.precommit_acoustic_a1(
        AcousticA1MandateRequest(
            campaign_id="cp810.acoustic.positive",
            expected_source_commit_sha256=_digest("a" * 40),
        )
    )
    request = AcousticA1Request(
        campaign_id="cp810.acoustic.positive",
        expected_source_commit_sha256=_digest("a" * 40),
        mandate_sha256=precommit["mandate"]["mandate_sha256"],
    )

    result = await service.run_acoustic_a1(request)
    calls_after_first = backend.calls
    replay = await service.run_acoustic_a1(request)
    persisted = service._acoustic_campaign_store.load(request.campaign_id)

    assert result["accepted"] is True
    assert result["receipt"]["error_reduction"] >= 0.5
    assert result["published"] is True
    assert result["replayed"] is False
    assert replay["published"] is False
    assert replay["replayed"] is True
    assert backend.calls == calls_after_first
    assert persisted.accepted is True
    assert persisted.sha256 == result["campaign_record_sha256"]


@pytest.mark.asyncio
async def test_service_persists_honest_negative_a1_result(tmp_path) -> None:
    service, _backend = await _service(tmp_path, nonlinear=False)
    precommit = await service.precommit_acoustic_a1(
        AcousticA1MandateRequest(
            campaign_id="cp810.acoustic.negative",
            expected_source_commit_sha256=_digest("a" * 40),
        )
    )

    result = await service.run_acoustic_a1(
        AcousticA1Request(
            campaign_id="cp810.acoustic.negative",
            expected_source_commit_sha256=_digest("a" * 40),
            mandate_sha256=precommit["mandate"]["mandate_sha256"],
        )
    )

    assert result["accepted"] is False
    assert "acoustic_a1_error_reduction_below_threshold" in result["receipt"]["blockers"]
    assert result["published"] is True
    assert result["governance_evidence"]["effect_verified"] is False


@pytest.mark.asyncio
async def test_service_refuses_wrong_mandate_before_physical_dispatch(tmp_path) -> None:
    service, backend = await _service(tmp_path, nonlinear=True)
    precommit = await service.precommit_acoustic_a1(
        AcousticA1MandateRequest(
            campaign_id="cp810.acoustic.refusal",
            expected_source_commit_sha256=_digest("a" * 40),
        )
    )
    calls_before = backend.calls

    with pytest.raises(AcceptanceError, match="mandate_digest_mismatch"):
        await service.run_acoustic_a1(
            AcousticA1Request(
                campaign_id="cp810.acoustic.refusal",
                expected_source_commit_sha256=_digest("a" * 40),
                mandate_sha256=_digest("wrong-mandate"),
            )
        )

    assert precommit["required_error_reduction"] == 0.5
    assert backend.calls == calls_before


@pytest.mark.asyncio
async def test_service_preserves_completed_evidence_after_executor_interruption(
    tmp_path,
) -> None:
    service, backend = await _service(tmp_path, nonlinear=True)
    service._governed_executor = _interrupted_after_effect_executor
    precommit = await service.precommit_acoustic_a1(
        AcousticA1MandateRequest(
            campaign_id="cp810.acoustic.interrupted",
            expected_source_commit_sha256=_digest("a" * 40),
        )
    )
    request = AcousticA1Request(
        campaign_id="cp810.acoustic.interrupted",
        expected_source_commit_sha256=_digest("a" * 40),
        mandate_sha256=precommit["mandate"]["mandate_sha256"],
    )

    with pytest.raises(RuntimeError, match="fixture_post_effect_interruption"):
        await service.run_acoustic_a1(request)
    calls_after_interruption = backend.calls
    persisted = service._acoustic_campaign_store.load(request.campaign_id)
    service._governed_executor = _governed_executor
    replay = await service.run_acoustic_a1(request)

    assert persisted.receipt.accepted is True
    assert persisted.accepted is False
    assert persisted.governance_evidence["status"] == "interrupted_after_effect"
    assert replay["replayed"] is True
    assert replay["accepted"] is False
    assert backend.calls == calls_after_interruption
