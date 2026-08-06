"""Preregistered A1 acoustic verification, custody, and physical scoring."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.reality_reach.acceptance_contracts import AcceptanceEvidenceClass
from core.reality_reach.acceptance_mandate import AcceptanceVerificationMandate
from core.reality_reach.acceptance_preregistration import PreregisteredAcceptanceReceipt
from core.reality_reach.acceptance_transparency import (
    ACCEPTANCE_TRANSPARENCY_STATEMENT_SCHEMA,
    AcceptanceTransparencyError,
    build_acceptance_transparency_artifact_bundle,
    validate_acceptance_transparency_statement_envelope,
    verify_acceptance_transparency_artifact,
)
from core.reality_reach.acceptance_witness import (
    AcceptanceWitnessBundle,
    AcceptanceWitnessError,
    AcceptanceWitnessRole,
    verify_acceptance_witness_artifact_bundle,
)
from core.reality_reach.acoustic_acceptance_contracts import (
    _REKOR_UUID,
    _SHA256,
    ACOUSTIC_A1_CAMPAIGN_SCHEMA,
    ACOUSTIC_A1_CONNECTOR_ID,
    ACOUSTIC_A1_RECEIPT_SCHEMA,
    ACOUSTIC_A1_REQUIRED_CASES,
    EXTERNAL_ACOUSTIC_A1_VERIFICATION_SCHEMA,
    TRANSPARENT_ACOUSTIC_A1_VERIFICATION_SCHEMA,
    AcousticA1AcceptanceReceipt,
    AcousticA1CampaignRecord,
    AcousticAcceptanceConfig,
    AcousticAcceptanceError,
    AcousticTrial,
    AcousticTrialArm,
    AcousticTrialDriver,
    ExternallyWitnessedAcousticA1Receipt,
    _digest,
    _finite,
)
from core.runtime.audit_chain import canonical_json
from core.runtime.secure_path_custody import DirectoryCustody, SecurePathCustodyError
from core.runtime.state_ownership import state_root


def acoustic_a1_campaign_binding_blockers(
    record: AcousticA1CampaignRecord,
    mandate: AcceptanceVerificationMandate,
) -> tuple[str, ...]:
    """Replay the immutable A1 campaign question before external promotion."""

    if not isinstance(record, AcousticA1CampaignRecord):
        raise TypeError("record must be an AcousticA1CampaignRecord")
    if not isinstance(mandate, AcceptanceVerificationMandate):
        raise TypeError("mandate must be an AcceptanceVerificationMandate")
    config = AcousticAcceptanceConfig(campaign_id=record.campaign_id)
    blockers: list[str] = []
    expected_mandate = {
        "campaign_id": record.campaign_id,
        "connector_id": ACOUSTIC_A1_CONNECTOR_ID,
        "adapter_id": record.adapter_id,
        "expected_source_commit_sha256": record.source_commit_sha256,
        "expected_physical_identity_sha256": record.physical_identity_sha256,
        "expected_evidence_class": AcceptanceEvidenceClass.LIVE,
        "target": config.required_error_reduction,
        "target_tolerance": 0.0,
        "scenario_id": "",
        "expected_simulated_channel_ids": (),
        "required_cases": ACOUSTIC_A1_REQUIRED_CASES,
    }
    if mandate.sha256 != record.mandate_sha256:
        blockers.append("acoustic_a1_mandate_digest_mismatch")
    for field, value in expected_mandate.items():
        if getattr(mandate, field) != value:
            blockers.append(f"acoustic_a1_mandate_{field}_mismatch")
    if record.receipt.config_sha256 != config.sha256:
        blockers.append("acoustic_a1_config_digest_mismatch")
    if not record.accepted:
        blockers.append("acoustic_a1_producer_record_not_accepted")
    return tuple(sorted(set(blockers)))


def verify_acoustic_a1_with_external_witnesses(
    record: AcousticA1CampaignRecord,
    mandate: AcceptanceVerificationMandate,
    *,
    preregistration_receipt: PreregisteredAcceptanceReceipt | None,
    metrology_witness_bundle: AcceptanceWitnessBundle | Mapping[str, Any] | None,
    governance_witness_bundle: AcceptanceWitnessBundle | Mapping[str, Any] | None,
    metrology_witness_key_sha256: str,
    governance_witness_key_sha256: str,
    metrology_sequence: int = 1,
    governance_sequence: int = 1,
    metrology_previous_statement_sha256: str = "sha256:" + "0" * 64,
    governance_previous_statement_sha256: str = "sha256:" + "0" * 64,
    now_ns: int | None = None,
) -> ExternallyWitnessedAcousticA1Receipt:
    """Independently bind A1 physical and governance evidence to two roots."""

    blockers = list(acoustic_a1_campaign_binding_blockers(record, mandate))
    verified_preregistration = ""
    expected_acceptance_log_key_sha256 = ""
    if preregistration_receipt is None:
        blockers.append("acceptance_preregistration_missing")
    elif not isinstance(preregistration_receipt, PreregisteredAcceptanceReceipt):
        blockers.append("acceptance_preregistration_invalid")
    elif (
        not preregistration_receipt.accepted
        or preregistration_receipt.mandate.sha256 != mandate.sha256
        or preregistration_receipt.campaign_started_at_ns != record.started_at_ns
        or not preregistration_receipt.strictly_predates_campaign
    ):
        blockers.append("acceptance_preregistration_binding_invalid")
    else:
        verified_preregistration = preregistration_receipt.sha256
        if preregistration_receipt.trust_policy is None:
            blockers.append("acceptance_trust_policy_missing_or_invalid")
        else:
            policy = preregistration_receipt.trust_policy
            expected_acceptance_log_key_sha256 = policy.acceptance_log_key_sha256
            if (
                metrology_witness_key_sha256
                != policy.metrology_witness_key_sha256
            ):
                blockers.append("external_metrology_trust_root_not_preregistered")
            if (
                governance_witness_key_sha256
                != policy.governance_witness_key_sha256
            ):
                blockers.append("external_governance_trust_root_not_preregistered")

    verified_metrology_bundle = ""
    verified_governance_bundle = ""
    if metrology_witness_bundle is None:
        blockers.append("external_metrology_witness_missing")
    elif not metrology_witness_key_sha256:
        blockers.append("external_metrology_trust_root_missing")
    else:
        try:
            verified = verify_acceptance_witness_artifact_bundle(
                metrology_witness_bundle,
                expected_role=AcceptanceWitnessRole.METROLOGY,
                expected_public_key_sha256=metrology_witness_key_sha256,
                expected_campaign_id=record.campaign_id,
                expected_mandate_sha256=mandate.sha256,
                expected_artifact_sha256=record.sha256,
                expected_evidence_sha256=record.receipt.sha256,
                expected_sequence=metrology_sequence,
                expected_previous_statement_sha256=(
                    metrology_previous_statement_sha256
                ),
                campaign_completed_at_ns=record.completed_at_ns,
                now_ns=now_ns,
            )
            verified_metrology_bundle = verified.sha256
        except (AcceptanceWitnessError, TypeError, ValueError) as exc:
            blockers.append(
                exc.code
                if isinstance(exc, AcceptanceWitnessError)
                else "external_metrology_witness_invalid"
            )

    if governance_witness_bundle is None:
        blockers.append("external_governance_witness_missing")
    elif not governance_witness_key_sha256:
        blockers.append("external_governance_trust_root_missing")
    else:
        try:
            verified = verify_acceptance_witness_artifact_bundle(
                governance_witness_bundle,
                expected_role=AcceptanceWitnessRole.GOVERNANCE,
                expected_public_key_sha256=governance_witness_key_sha256,
                expected_campaign_id=record.campaign_id,
                expected_mandate_sha256=mandate.sha256,
                expected_artifact_sha256=record.sha256,
                expected_evidence_sha256=_digest(record.governance_evidence),
                expected_sequence=governance_sequence,
                expected_previous_statement_sha256=(
                    governance_previous_statement_sha256
                ),
                campaign_completed_at_ns=record.completed_at_ns,
                now_ns=now_ns,
            )
            verified_governance_bundle = verified.sha256
        except (AcceptanceWitnessError, TypeError, ValueError) as exc:
            blockers.append(
                exc.code
                if isinstance(exc, AcceptanceWitnessError)
                else "external_governance_witness_invalid"
            )
    if (
        verified_metrology_bundle
        and verified_governance_bundle
        and metrology_witness_key_sha256 == governance_witness_key_sha256
    ):
        blockers.append("external_witness_roots_not_distinct")
    return ExternallyWitnessedAcousticA1Receipt(
        campaign_id=record.campaign_id,
        campaign_record_sha256=record.sha256,
        mandate_sha256=mandate.sha256,
        preregistration_verification_sha256=verified_preregistration,
        metrology_witness_bundle_sha256=verified_metrology_bundle,
        governance_witness_bundle_sha256=verified_governance_bundle,
        metrology_witness_key_sha256=(
            metrology_witness_key_sha256 if verified_metrology_bundle else ""
        ),
        governance_witness_key_sha256=(
            governance_witness_key_sha256 if verified_governance_bundle else ""
        ),
        acceptance_log_key_sha256=expected_acceptance_log_key_sha256,
        blockers=tuple(sorted(set(blockers))),
    )


def build_acoustic_a1_transparency_statement(
    receipt: ExternallyWitnessedAcousticA1Receipt,
    *,
    sequence: int,
    previous_statement_sha256: str,
    previous_rekor_uuid: str | None,
    issued_at_unix: int,
) -> dict[str, Any]:
    """Commit one accepted A1 dual-witness verdict for public timestamping."""

    if not isinstance(receipt, ExternallyWitnessedAcousticA1Receipt):
        raise TypeError("receipt must be an ExternallyWitnessedAcousticA1Receipt")
    if not receipt.accepted:
        raise AcceptanceTransparencyError("acceptance_transparency_receipt_not_accepted")
    if type(sequence) is not int or sequence <= 0:
        raise AcceptanceTransparencyError("acceptance_transparency_sequence_invalid")
    if not _SHA256.fullmatch(previous_statement_sha256):
        raise AcceptanceTransparencyError(
            "acceptance_transparency_previous_statement_sha256_invalid"
        )
    if type(issued_at_unix) is not int or issued_at_unix <= 0:
        raise AcceptanceTransparencyError("acceptance_transparency_issued_at_invalid")
    zero = "sha256:" + "0" * 64
    if sequence == 1:
        if previous_statement_sha256 != zero or previous_rekor_uuid is not None:
            raise AcceptanceTransparencyError("acceptance_transparency_genesis_invalid")
    elif previous_statement_sha256 == zero or not _REKOR_UUID.fullmatch(
        str(previous_rekor_uuid or "")
    ):
        raise AcceptanceTransparencyError("acceptance_transparency_chain_invalid")
    body = {
        "schema": ACCEPTANCE_TRANSPARENCY_STATEMENT_SCHEMA,
        "domain": "aura.reality-reach.acoustic-a1",
        "campaign_id": receipt.campaign_id,
        "mandate_sha256": receipt.mandate_sha256,
        "external_verification_sha256": receipt.sha256,
        "acceptance_log_key_sha256": receipt.acceptance_log_key_sha256,
        "metrology_witness_bundle_sha256": receipt.metrology_witness_bundle_sha256,
        "governance_witness_bundle_sha256": receipt.governance_witness_bundle_sha256,
        "sequence": sequence,
        "previous_statement_sha256": previous_statement_sha256,
        "previous_rekor_uuid": previous_rekor_uuid,
        "issued_at_unix": issued_at_unix,
    }
    return {**body, "statement_sha256": _digest(body)}


def _validate_acoustic_a1_transparency_statement(
    raw: object,
    *,
    receipt: ExternallyWitnessedAcousticA1Receipt,
    expected_sequence: int,
    expected_previous_statement_sha256: str,
    expected_previous_rekor_uuid: str | None,
) -> dict[str, Any]:
    statement = validate_acceptance_transparency_statement_envelope(raw)
    issued_at = statement.get("issued_at_unix")
    if type(issued_at) is not int:
        raise AcceptanceTransparencyError(
            "acceptance_transparency_statement_binding_invalid"
        )
    expected = build_acoustic_a1_transparency_statement(
        receipt,
        sequence=expected_sequence,
        previous_statement_sha256=expected_previous_statement_sha256,
        previous_rekor_uuid=expected_previous_rekor_uuid,
        issued_at_unix=issued_at,
    )
    if statement != expected:
        raise AcceptanceTransparencyError(
            "acceptance_transparency_statement_binding_invalid"
        )
    return dict(statement)


def build_acoustic_a1_transparency_bundle(
    *,
    statement: Mapping[str, Any],
    producer_signature: bytes,
    producer_certificate_pem: bytes,
    rekor_uuid: str,
    rekor_entry: Mapping[str, Any],
    trusted_log_public_key_pem: bytes,
) -> dict[str, Any]:
    """Build a portable Rekor bundle for one accepted A1 statement."""

    statement_document = validate_acceptance_transparency_statement_envelope(statement)
    if statement_document.get("domain") != "aura.reality-reach.acoustic-a1":
        raise AcceptanceTransparencyError(
            "acceptance_transparency_statement_binding_invalid"
        )
    bundle = build_acceptance_transparency_artifact_bundle(
        statement=statement_document,
        producer_signature=producer_signature,
        producer_certificate_pem=producer_certificate_pem,
        rekor_uuid=rekor_uuid,
        rekor_entry=rekor_entry,
        trusted_log_public_key_pem=trusted_log_public_key_pem,
    )
    if not isinstance(bundle, dict):
        raise AcceptanceTransparencyError("acceptance_transparency_bundle_invalid")
    return dict(bundle)


@dataclass(frozen=True, slots=True)
class TransparentlyLoggedAcousticA1Receipt:
    external_verification: ExternallyWitnessedAcousticA1Receipt
    transparency_bundle_sha256: str
    trusted_log_key_sha256: str
    rekor_uuid: str
    rekor_log_index: int
    rekor_integrated_time: int
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(
            self.external_verification,
            ExternallyWitnessedAcousticA1Receipt,
        ):
            raise TypeError(
                "external_verification must be an "
                "ExternallyWitnessedAcousticA1Receipt"
            )
        for name in ("transparency_bundle_sha256", "trusted_log_key_sha256"):
            value = str(getattr(self, name) or "")
            if value and not _SHA256.fullmatch(value):
                raise ValueError(f"{name} must be empty or a sha256 digest")
        if self.rekor_uuid and not _REKOR_UUID.fullmatch(self.rekor_uuid):
            raise ValueError("rekor_uuid must be empty or a Rekor UUID")
        if type(self.rekor_log_index) is not int or self.rekor_log_index < -1:
            raise ValueError("rekor_log_index must be an integer >= -1")
        if type(self.rekor_integrated_time) is not int or self.rekor_integrated_time < 0:
            raise ValueError("rekor_integrated_time must be a non-negative integer")
        if len(self.blockers) != len(set(self.blockers)):
            raise ValueError("blockers must be unique")

    @property
    def accepted(self) -> bool:
        return bool(
            self.external_verification.accepted
            and self.transparency_bundle_sha256
            and not self.blockers
        )

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict(include_digest=False))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        document = {
            "schema": TRANSPARENT_ACOUSTIC_A1_VERIFICATION_SCHEMA,
            "external_verification": self.external_verification.to_dict(),
            "transparency_bundle_sha256": self.transparency_bundle_sha256,
            "trusted_log_key_sha256": self.trusted_log_key_sha256,
            "rekor_uuid": self.rekor_uuid,
            "rekor_log_index": self.rekor_log_index,
            "rekor_integrated_time": self.rekor_integrated_time,
            "blockers": list(self.blockers),
            "accepted": self.accepted,
        }
        if include_digest:
            document["transparent_verification_sha256"] = self.sha256
        return document


def verify_transparently_logged_acoustic_a1(
    receipt: ExternallyWitnessedAcousticA1Receipt,
    *,
    transparency_bundle: Mapping[str, Any] | None,
    trusted_log_public_key_pem: bytes | None,
    expected_sequence: int,
    expected_previous_statement_sha256: str,
    expected_previous_rekor_uuid: str | None,
    minimum_log_index: int | None = None,
    minimum_integrated_time: int | None = None,
) -> TransparentlyLoggedAcousticA1Receipt:
    """Require append-only public-log inclusion for A1 promotion."""

    if not isinstance(receipt, ExternallyWitnessedAcousticA1Receipt):
        raise TypeError("receipt must be an ExternallyWitnessedAcousticA1Receipt")
    artifact = verify_acceptance_transparency_artifact(
        transparency_bundle=transparency_bundle,
        trusted_log_public_key_pem=trusted_log_public_key_pem,
        statement_validator=lambda raw: _validate_acoustic_a1_transparency_statement(
            raw,
            receipt=receipt,
            expected_sequence=expected_sequence,
            expected_previous_statement_sha256=expected_previous_statement_sha256,
            expected_previous_rekor_uuid=expected_previous_rekor_uuid,
        ),
        minimum_log_index=minimum_log_index,
        minimum_integrated_time=minimum_integrated_time,
    )
    blockers = list(artifact.blockers)
    if (
        artifact.accepted
        and artifact.trusted_log_key_sha256 != receipt.acceptance_log_key_sha256
    ):
        blockers.append("acceptance_transparency_log_root_not_preregistered")
    return TransparentlyLoggedAcousticA1Receipt(
        external_verification=receipt,
        transparency_bundle_sha256=artifact.transparency_bundle_sha256,
        trusted_log_key_sha256=artifact.trusted_log_key_sha256,
        rekor_uuid=artifact.rekor_uuid,
        rekor_log_index=artifact.rekor_log_index,
        rekor_integrated_time=artifact.rekor_integrated_time,
        blockers=tuple(sorted(set(blockers))),
    )


def persist_transparently_logged_acoustic_a1_receipt(
    receipt: TransparentlyLoggedAcousticA1Receipt,
    path: str | Path,
) -> bool:
    """Create-once publish one transparency-bound A1 promotion verdict."""

    if not isinstance(receipt, TransparentlyLoggedAcousticA1Receipt):
        raise TypeError("receipt must be a TransparentlyLoggedAcousticA1Receipt")
    target = Path(path).expanduser().absolute()
    if not target.name or target.name in {".", ".."}:
        raise AcousticAcceptanceError("transparent_acoustic_a1_receipt_path_invalid")
    payload = canonical_json(receipt.to_dict())
    try:
        with DirectoryCustody.acquire(target.parent, create=True, private=True) as custody:
            published = bool(custody.write_bytes_once(target.name, payload, mode=0o600))
            fd = custody.open_file(target.name, os.O_RDONLY)
            try:
                if stat.S_IMODE(os.fstat(fd).st_mode) != 0o600:
                    raise AcousticAcceptanceError(
                        "transparent_acoustic_a1_receipt_mode_invalid"
                    )
            finally:
                os.close(fd)
            existing = custody.read_bytes(target.name, max_bytes=1024 * 1024)
    except SecurePathCustodyError as exc:
        raise AcousticAcceptanceError(
            "transparent_acoustic_a1_receipt_custody_invalid"
        ) from exc
    if existing != payload:
        raise AcousticAcceptanceError("transparent_acoustic_a1_receipt_collision")
    return published


def persist_externally_witnessed_acoustic_a1_receipt(
    receipt: ExternallyWitnessedAcousticA1Receipt,
    path: str | Path,
) -> bool:
    """Create-once publish one dual-root A1 promotion verdict."""

    if not isinstance(receipt, ExternallyWitnessedAcousticA1Receipt):
        raise TypeError("receipt must be an ExternallyWitnessedAcousticA1Receipt")
    target = Path(path).expanduser().absolute()
    if not target.name or target.name in {".", ".."}:
        raise AcousticAcceptanceError("external_acoustic_a1_receipt_path_invalid")
    payload = canonical_json(receipt.to_dict())
    try:
        with DirectoryCustody.acquire(target.parent, create=True, private=True) as custody:
            published = bool(custody.write_bytes_once(target.name, payload, mode=0o600))
            fd = custody.open_file(target.name, os.O_RDONLY)
            try:
                if stat.S_IMODE(os.fstat(fd).st_mode) != 0o600:
                    raise AcousticAcceptanceError(
                        "external_acoustic_a1_receipt_mode_invalid"
                    )
            finally:
                os.close(fd)
            existing = custody.read_bytes(target.name, max_bytes=1024 * 1024)
    except SecurePathCustodyError as exc:
        raise AcousticAcceptanceError(
            "external_acoustic_a1_receipt_custody_invalid"
        ) from exc
    if existing != payload:
        raise AcousticAcceptanceError("external_acoustic_a1_receipt_collision")
    return published


class AcousticA1CampaignStore:
    """Private create-once storage for positive and negative A1 outcomes."""

    def __init__(self, root: str | Path | None = None) -> None:
        self._root = Path(
            root
            or (state_root() / "data" / "reality_reach" / "acoustic_a1_acceptance")
        ).expanduser().absolute()

    @property
    def root(self) -> Path:
        return self._root

    @staticmethod
    def _filename(campaign_id: str) -> str:
        if not campaign_id or len(campaign_id) > 128:
            raise ValueError("campaign_id must be a bounded non-empty string")
        return _digest({"campaign_id": campaign_id}).removeprefix("sha256:") + ".json"

    @staticmethod
    def _verify_mode(custody: DirectoryCustody, filename: str) -> None:
        fd = custody.open_file(filename, os.O_RDONLY)
        try:
            if stat.S_IMODE(os.fstat(fd).st_mode) != 0o600:
                raise AcousticAcceptanceError("acoustic_a1_campaign_mode_invalid")
        finally:
            os.close(fd)

    def persist(self, record: AcousticA1CampaignRecord) -> bool:
        if not isinstance(record, AcousticA1CampaignRecord):
            raise TypeError("record must be an AcousticA1CampaignRecord")
        filename = self._filename(record.campaign_id)
        payload = canonical_json(record.to_dict())
        try:
            with DirectoryCustody.acquire(self._root, create=True, private=True) as custody:
                published = bool(custody.write_bytes_once(filename, payload, mode=0o600))
                self._verify_mode(custody, filename)
                existing = custody.read_bytes(filename, max_bytes=2 * 1024 * 1024)
        except SecurePathCustodyError as exc:
            raise AcousticAcceptanceError("acoustic_a1_campaign_custody_invalid") from exc
        if existing != payload:
            raise AcousticAcceptanceError("acoustic_a1_campaign_collision")
        return published

    def load(self, campaign_id: str) -> AcousticA1CampaignRecord:
        filename = self._filename(campaign_id)
        try:
            os.lstat(self._root)
        except FileNotFoundError as exc:
            raise AcousticAcceptanceError("acoustic_a1_campaign_unavailable") from exc
        except OSError as exc:
            raise AcousticAcceptanceError("acoustic_a1_campaign_custody_invalid") from exc
        try:
            with DirectoryCustody.acquire(self._root, create=False, private=True) as custody:
                self._verify_mode(custody, filename)
                payload = custody.read_bytes(filename, max_bytes=2 * 1024 * 1024)
        except FileNotFoundError as exc:
            raise AcousticAcceptanceError("acoustic_a1_campaign_unavailable") from exc
        except SecurePathCustodyError as exc:
            raise AcousticAcceptanceError("acoustic_a1_campaign_custody_invalid") from exc
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, ValueError) as exc:
            raise AcousticAcceptanceError("acoustic_a1_campaign_json_invalid") from exc
        record = AcousticA1CampaignRecord.from_dict(document)
        if record.campaign_id != campaign_id:
            raise AcousticAcceptanceError("acoustic_a1_campaign_identity_mismatch")
        return record


def _pava(values: Sequence[float]) -> tuple[float, ...]:
    blocks: list[tuple[float, int]] = []
    for value in values:
        blocks.append((_finite(value, name="pava_value"), 1))
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            right_mean, right_count = blocks.pop()
            left_mean, left_count = blocks.pop()
            count = left_count + right_count
            blocks.append(
                (
                    (left_mean * left_count + right_mean * right_count) / count,
                    count,
                )
            )
    fitted: list[float] = []
    for mean, count in blocks:
        fitted.extend([mean] * count)
    return tuple(fitted)


def _inverse_monotone(
    amplitudes: Sequence[float],
    observed: Sequence[float],
    target: float,
) -> float:
    if target <= observed[0]:
        return amplitudes[0]
    if target >= observed[-1]:
        return amplitudes[-1]
    for index in range(1, len(observed)):
        if target <= observed[index]:
            low_y = observed[index - 1]
            high_y = observed[index]
            if high_y <= low_y:
                return amplitudes[index]
            fraction = (target - low_y) / (high_y - low_y)
            return amplitudes[index - 1] + fraction * (
                amplitudes[index] - amplitudes[index - 1]
            )
    return amplitudes[-1]


def _nominal_open_loop_amplitude(target_dbfs: float, maximum: float) -> float:
    return float(
        min(maximum, max(0.0, math.sqrt(2.0) * 10.0 ** (target_dbfs / 20.0)))
    )


def _trial_order(seed: str, labels: Sequence[tuple[str, float, int]]) -> list[tuple[str, float, int]]:
    return sorted(
        labels,
        key=lambda item: hashlib.sha256(
            f"{seed}|{item[0]}|{item[1]:.9f}|{item[2]}".encode()
        ).digest(),
    )


async def run_acoustic_a1_acceptance(
    driver: AcousticTrialDriver,
    config: AcousticAcceptanceConfig,
) -> AcousticA1AcceptanceReceipt:
    """Run equal-work calibration/control arms and restore silence in ``finally``."""

    if not isinstance(driver, AcousticTrialDriver):
        raise TypeError("driver must satisfy AcousticTrialDriver")
    if not isinstance(config, AcousticAcceptanceConfig):
        raise TypeError("config must be an AcousticAcceptanceConfig")
    trials: list[AcousticTrial] = []
    order = 0

    async def capture(
        arm: AcousticTrialArm,
        amplitude: float,
        *,
        target: float | None,
        repeat: int,
    ) -> AcousticTrial:
        nonlocal order
        order += 1
        raw_trial_id = f"{config.campaign_id}.{arm.value}.{order:04d}"
        trial_id = raw_trial_id
        if len(trial_id) > 128:
            suffix = hashlib.sha256(raw_trial_id.encode()).hexdigest()[:24]
            trial_id = f"{config.campaign_id[:80]}.{arm.value}.{order:04d}.{suffix}"
        sample = await driver.measure_stimulus(amplitude, trial_id=trial_id)
        trial = AcousticTrial(
            trial_id=trial_id,
            arm=arm,
            order=order,
            repeat=repeat,
            amplitude=amplitude,
            target_dbfs=target,
            observed_dbfs=sample.value,
            source_event_id=sample.source_event_id,
            captured_at_ns=sample.captured_at_ns,
        )
        trials.append(trial)
        return trial

    baseline = math.nan
    restoration_error = math.inf
    experiment_error: BaseException | None = None
    try:
        calibration_groups: list[list[float]] = []
        for amplitude in config.calibration_amplitudes:
            group: list[float] = []
            for repeat in range(config.calibration_repeats):
                trial = await capture(
                    AcousticTrialArm.CALIBRATION,
                    amplitude,
                    target=None,
                    repeat=repeat,
                )
                group.append(trial.observed_dbfs)
            calibration_groups.append(group)
        medians = tuple(statistics.median(group) for group in calibration_groups)
        fitted = _pava(medians)
        baseline = fitted[0]
        calibration_span = fitted[-1] - baseline
        lower = baseline + config.baseline_margin_db
        upper = fitted[-1] - config.ceiling_margin_db
        if upper <= lower:
            raise AcousticAcceptanceError("acoustic_a1_calibration_span_insufficient")
        targets = tuple(
            lower + quantile * (upper - lower)
            for quantile in config.heldout_quantiles
        )
        schedule = _trial_order(
            f"{config.schedule_seed}|{config.sha256}",
            tuple(
                (arm.value, target, repeat)
                for target in targets
                for repeat in range(config.heldout_repeats)
                for arm in (AcousticTrialArm.OPEN_LOOP, AcousticTrialArm.CLOSED_LOOP)
            ),
        )
        maximum = config.calibration_amplitudes[-1]
        for arm_value, target, repeat in schedule:
            arm = AcousticTrialArm(arm_value)
            amplitude = (
                _nominal_open_loop_amplitude(target, maximum)
                if arm is AcousticTrialArm.OPEN_LOOP
                else _inverse_monotone(config.calibration_amplitudes, fitted, target)
            )
            await capture(arm, amplitude, target=target, repeat=repeat)
    except BaseException as exc:
        experiment_error = exc

    try:
        restored = await capture(
            AcousticTrialArm.RESTORATION,
            0.0,
            target=baseline if math.isfinite(baseline) else None,
            repeat=0,
        )
        if math.isfinite(baseline):
            restoration_error = abs(restored.observed_dbfs - baseline)
    except BaseException as restoration_exc:
        if experiment_error is not None:
            experiment_error.add_note(
                "acoustic silence restoration also failed: "
                f"{type(restoration_exc).__name__}: {restoration_exc}"
            )
            raise experiment_error from restoration_exc
        raise AcousticAcceptanceError("acoustic_a1_restoration_failed") from restoration_exc

    if experiment_error is not None:
        raise experiment_error

    open_errors = tuple(
        trial.absolute_error_db
        for trial in trials
        if trial.arm is AcousticTrialArm.OPEN_LOOP
        and trial.absolute_error_db is not None
    )
    closed_errors = tuple(
        trial.absolute_error_db
        for trial in trials
        if trial.arm is AcousticTrialArm.CLOSED_LOOP
        and trial.absolute_error_db is not None
    )
    if not open_errors or len(open_errors) != len(closed_errors):
        raise AcousticAcceptanceError("acoustic_a1_equal_work_contract_failed")
    open_mae = statistics.fmean(open_errors)
    closed_mae = statistics.fmean(closed_errors)
    reduction = 1.0 - closed_mae / max(open_mae, 1e-9)
    calibration_trials = tuple(
        trial for trial in trials if trial.arm is AcousticTrialArm.CALIBRATION
    )
    calibration_values = tuple(trial.observed_dbfs for trial in calibration_trials)
    calibration_span = max(calibration_values) - min(calibration_values)
    blockers: list[str] = []
    if calibration_span < config.minimum_signal_span_db:
        blockers.append("acoustic_a1_signal_span_below_threshold")
    if reduction < config.required_error_reduction:
        blockers.append("acoustic_a1_error_reduction_below_threshold")
    if restoration_error > config.restoration_tolerance_db:
        blockers.append("acoustic_a1_restoration_out_of_tolerance")
    heldout_targets = tuple(
        sorted(
            {
                float(trial.target_dbfs)
                for trial in trials
                if trial.target_dbfs is not None
                and trial.arm in {AcousticTrialArm.OPEN_LOOP, AcousticTrialArm.CLOSED_LOOP}
            }
        )
    )
    return AcousticA1AcceptanceReceipt(
        campaign_id=config.campaign_id,
        config_sha256=config.sha256,
        heldout_targets_dbfs=heldout_targets,
        trials=tuple(trials),
        open_loop_mae_db=open_mae,
        closed_loop_mae_db=closed_mae,
        error_reduction=reduction,
        restoration_error_db=restoration_error,
        calibration_span_db=calibration_span,
        blockers=tuple(blockers),
        completed_at_ns=time.time_ns(),
    )


__all__ = [
    "ACOUSTIC_A1_CONNECTOR_ID",
    "ACOUSTIC_A1_CAMPAIGN_SCHEMA",
    "ACOUSTIC_A1_RECEIPT_SCHEMA",
    "ACOUSTIC_A1_REQUIRED_CASES",
    "EXTERNAL_ACOUSTIC_A1_VERIFICATION_SCHEMA",
    "TRANSPARENT_ACOUSTIC_A1_VERIFICATION_SCHEMA",
    "AcousticA1AcceptanceReceipt",
    "AcousticA1CampaignRecord",
    "AcousticA1CampaignStore",
    "AcousticAcceptanceConfig",
    "AcousticAcceptanceError",
    "AcousticTrial",
    "AcousticTrialArm",
    "AcousticTrialDriver",
    "ExternallyWitnessedAcousticA1Receipt",
    "TransparentlyLoggedAcousticA1Receipt",
    "acoustic_a1_campaign_binding_blockers",
    "build_acoustic_a1_transparency_bundle",
    "build_acoustic_a1_transparency_statement",
    "persist_externally_witnessed_acoustic_a1_receipt",
    "persist_transparently_logged_acoustic_a1_receipt",
    "run_acoustic_a1_acceptance",
    "verify_acoustic_a1_with_external_witnesses",
    "verify_transparently_logged_acoustic_a1",
]
