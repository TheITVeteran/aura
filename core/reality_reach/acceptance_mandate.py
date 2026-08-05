"""Precommitted, rollback-resistant Reality Reach acceptance mandates.

Acceptance evidence is only meaningful when the verifier's source, device,
evidence class, target, tolerance, and required cases were fixed before the
campaign began.  This module stores that mandate under a dedicated macOS
Keychain custody root so a filesystem edit cannot rewrite the question after
the result is known.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.reality_reach.acceptance import (
    REQUIRED_SCALAR_ACCEPTANCE_CASES,
    AcceptanceEvidenceClass,
)
from core.reality_reach.trust_custody import (
    AttachmentTrustStore,
    KeychainAttachmentTrustStore,
)
from core.runtime.atomic_writer import interprocess_file_lock
from core.runtime.state_ownership import state_root
from core.security.zenith_secrets import KeychainBackend, require_keychain_backend

ACCEPTANCE_MANDATE_SCHEMA = "aura.reality_reach.acceptance_mandate.v1"
ACCEPTANCE_MANDATE_STATE_SCHEMA = "aura.reality_reach.acceptance_mandates.v1"
ACCEPTANCE_MANDATE_RECEIPT_SCHEMA = (
    "aura.reality_reach.acceptance_mandate_provision.v1"
)

_MANDATE_KEYCHAIN_SERVICE = "AuraRealityReachAcceptance"
_MANDATE_KEYRING_ACCOUNT = "acceptance-mandate-keyring-v1"
_MANDATE_ANCHOR_ACCOUNT = "acceptance-mandate-anchor-v1"
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_MANDATES = 4096
_MAX_REQUIRED_CASES = 128


class AcceptanceMandateError(RuntimeError):
    """A mandate is invalid, missing, or conflicts with a prior commitment."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError, OverflowError) as exc:
        raise AcceptanceMandateError("acceptance_mandate_json_invalid") from exc


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _identifier(value: object, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(f"{name} must be a canonical identifier")
    return normalized


def _sha256(value: object, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _DIGEST.fullmatch(normalized):
        raise ValueError(f"{name} must be a sha256 digest")
    return normalized


@dataclass(frozen=True, slots=True)
class AcceptanceVerificationMandate:
    """The exact acceptance question committed before physical execution."""

    campaign_id: str
    connector_id: str
    adapter_id: str
    expected_source_commit_sha256: str
    expected_physical_identity_sha256: str
    expected_evidence_class: AcceptanceEvidenceClass
    target: float
    target_tolerance: float
    scenario_id: str
    required_cases: tuple[str, ...]
    provisioned_at_ns: int
    custody_sequence: int

    def __post_init__(self) -> None:
        for name in ("campaign_id", "connector_id", "adapter_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name=name))
        for name in (
            "expected_source_commit_sha256",
            "expected_physical_identity_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        if not isinstance(self.expected_evidence_class, AcceptanceEvidenceClass):
            raise TypeError("expected_evidence_class must be an AcceptanceEvidenceClass")
        scenario_id = str(self.scenario_id or "").strip().lower()
        if scenario_id:
            scenario_id = _identifier(scenario_id, name="scenario_id")
        if (
            self.expected_evidence_class is AcceptanceEvidenceClass.HARDWARE_IN_LOOP
            and not scenario_id
        ):
            raise ValueError("HIL acceptance mandate requires a scenario_id")
        object.__setattr__(self, "scenario_id", scenario_id)
        target = float(self.target)
        tolerance = float(self.target_tolerance)
        if not math.isfinite(target) or not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("acceptance mandate target and tolerance must be finite")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "target_tolerance", tolerance)
        cases = tuple(
            _identifier(case_id, name="required_case_id")
            for case_id in self.required_cases
        )
        if not 1 <= len(cases) <= _MAX_REQUIRED_CASES:
            raise ValueError("acceptance mandate required case count is invalid")
        if len(cases) != len(set(cases)):
            raise ValueError("acceptance mandate required cases must be unique")
        object.__setattr__(self, "required_cases", cases)
        if (
            isinstance(self.provisioned_at_ns, bool)
            or not isinstance(self.provisioned_at_ns, int)
            or self.provisioned_at_ns <= 0
        ):
            raise ValueError("provisioned_at_ns must be a positive integer")
        if (
            isinstance(self.custody_sequence, bool)
            or not isinstance(self.custody_sequence, int)
            or self.custody_sequence <= 0
        ):
            raise ValueError("custody_sequence must be a positive integer")

    @property
    def contract_sha256(self) -> str:
        return _digest(self.contract_dict())

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict(include_digest=False))

    def contract_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "connector_id": self.connector_id,
            "adapter_id": self.adapter_id,
            "expected_source_commit_sha256": self.expected_source_commit_sha256,
            "expected_physical_identity_sha256": (
                self.expected_physical_identity_sha256
            ),
            "expected_evidence_class": self.expected_evidence_class.value,
            "target": self.target,
            "target_tolerance": self.target_tolerance,
            "scenario_id": self.scenario_id,
            "required_cases": list(self.required_cases),
        }

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        document = {
            "schema": ACCEPTANCE_MANDATE_SCHEMA,
            **self.contract_dict(),
            "provisioned_at_ns": self.provisioned_at_ns,
            "custody_sequence": self.custody_sequence,
            "contract_sha256": self.contract_sha256,
        }
        if include_digest:
            document["mandate_sha256"] = self.sha256
        return document

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> AcceptanceVerificationMandate:
        expected = {
            "schema",
            "campaign_id",
            "connector_id",
            "adapter_id",
            "expected_source_commit_sha256",
            "expected_physical_identity_sha256",
            "expected_evidence_class",
            "target",
            "target_tolerance",
            "scenario_id",
            "required_cases",
            "provisioned_at_ns",
            "custody_sequence",
            "contract_sha256",
            "mandate_sha256",
        }
        if not isinstance(document, Mapping) or set(document) != expected:
            raise AcceptanceMandateError("acceptance_mandate_schema_invalid")
        raw_cases = document.get("required_cases")
        if not isinstance(raw_cases, list):
            raise AcceptanceMandateError("acceptance_mandate_required_cases_invalid")
        try:
            mandate = cls(
                campaign_id=document["campaign_id"],
                connector_id=document["connector_id"],
                adapter_id=document["adapter_id"],
                expected_source_commit_sha256=document[
                    "expected_source_commit_sha256"
                ],
                expected_physical_identity_sha256=document[
                    "expected_physical_identity_sha256"
                ],
                expected_evidence_class=AcceptanceEvidenceClass(
                    document["expected_evidence_class"]
                ),
                target=document["target"],
                target_tolerance=document["target_tolerance"],
                scenario_id=document["scenario_id"],
                required_cases=tuple(raw_cases),
                provisioned_at_ns=document["provisioned_at_ns"],
                custody_sequence=document["custody_sequence"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AcceptanceMandateError("acceptance_mandate_invalid") from exc
        if not hmac.compare_digest(
            str(document.get("contract_sha256") or ""), mandate.contract_sha256
        ):
            raise AcceptanceMandateError("acceptance_mandate_contract_digest_invalid")
        if not hmac.compare_digest(
            str(document.get("mandate_sha256") or ""), mandate.sha256
        ):
            raise AcceptanceMandateError("acceptance_mandate_digest_invalid")
        return mandate


@dataclass(frozen=True, slots=True)
class AcceptanceMandateProvisionReceipt:
    campaign_id: str
    mandate_sha256: str
    contract_sha256: str
    custody_identity_sha256: str
    provisioned_at_ns: int
    created: bool
    custody_sequence: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "campaign_id",
            _identifier(self.campaign_id, name="campaign_id"),
        )
        for name in (
            "mandate_sha256",
            "contract_sha256",
            "custody_identity_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        if (
            isinstance(self.provisioned_at_ns, bool)
            or not isinstance(self.provisioned_at_ns, int)
            or self.provisioned_at_ns <= 0
        ):
            raise ValueError("provisioned_at_ns must be a positive integer")
        if not isinstance(self.created, bool):
            raise TypeError("created must be a bool")
        if (
            isinstance(self.custody_sequence, bool)
            or not isinstance(self.custody_sequence, int)
            or self.custody_sequence <= 0
        ):
            raise ValueError("custody_sequence must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": ACCEPTANCE_MANDATE_RECEIPT_SCHEMA,
            "campaign_id": self.campaign_id,
            "mandate_sha256": self.mandate_sha256,
            "contract_sha256": self.contract_sha256,
            "custody_identity_sha256": self.custody_identity_sha256,
            "provisioned_at_ns": self.provisioned_at_ns,
            "created": self.created,
            "custody_sequence": self.custody_sequence,
        }
        return {**body, "receipt_sha256": _digest(body)}


class AcceptanceMandateStore:
    """Create-once mandate registry over encrypted Keychain-anchored state."""

    def __init__(
        self,
        trust_store: AttachmentTrustStore,
        *,
        lock_path: Path,
    ) -> None:
        if not isinstance(trust_store, AttachmentTrustStore):
            raise TypeError("trust_store must implement AttachmentTrustStore")
        if not isinstance(lock_path, Path):
            raise TypeError("lock_path must be a Path")
        self._trust_store = trust_store
        self._lock_path = lock_path.expanduser().absolute()

    @staticmethod
    def default_state_path() -> Path:
        return (
            Path(state_root())
            / "data"
            / "reality_reach"
            / "acceptance_mandates.encrypted.json"
        )

    @classmethod
    def _system_store(
        cls,
        state_path: Path | None,
        *,
        backend: KeychainBackend,
        create_if_missing: bool,
    ) -> AcceptanceMandateStore:
        path = (state_path or cls.default_state_path()).expanduser().absolute()
        trust = KeychainAttachmentTrustStore(
            backend,
            path,
            service=_MANDATE_KEYCHAIN_SERVICE,
            keyring_account=_MANDATE_KEYRING_ACCOUNT,
            anchor_account=_MANDATE_ANCHOR_ACCOUNT,
            create_if_missing=create_if_missing,
        )
        return cls(trust, lock_path=path.with_name(path.name + ".mandate.lock"))

    @classmethod
    def provision_system(
        cls,
        state_path: Path | None = None,
    ) -> AcceptanceMandateStore:
        return cls._system_store(
            state_path,
            backend=require_keychain_backend(),
            create_if_missing=True,
        )

    @classmethod
    def from_system(
        cls,
        state_path: Path | None = None,
    ) -> AcceptanceMandateStore:
        return cls._system_store(
            state_path,
            backend=require_keychain_backend(),
            create_if_missing=False,
        )

    @classmethod
    def for_backend(
        cls,
        backend: KeychainBackend,
        state_path: Path,
        *,
        create_if_missing: bool = True,
    ) -> AcceptanceMandateStore:
        return cls._system_store(
            state_path,
            backend=backend,
            create_if_missing=create_if_missing,
        )

    @property
    def identity(self) -> Mapping[str, Any]:
        return dict(self._trust_store.identity)

    def _load_locked(self) -> dict[str, AcceptanceVerificationMandate]:
        raw = self._trust_store.load()
        if raw is None:
            return {}
        if not isinstance(raw, Mapping) or set(raw) != {"schema", "mandates"}:
            raise AcceptanceMandateError("acceptance_mandate_state_schema_invalid")
        if raw.get("schema") != ACCEPTANCE_MANDATE_STATE_SCHEMA:
            raise AcceptanceMandateError("acceptance_mandate_state_schema_invalid")
        raw_mandates = raw.get("mandates")
        if not isinstance(raw_mandates, Mapping) or len(raw_mandates) > _MAX_MANDATES:
            raise AcceptanceMandateError("acceptance_mandate_state_invalid")
        mandates: dict[str, AcceptanceVerificationMandate] = {}
        custody_sequences: set[int] = set()
        for campaign_id, document in raw_mandates.items():
            if not isinstance(campaign_id, str) or not isinstance(document, Mapping):
                raise AcceptanceMandateError("acceptance_mandate_state_invalid")
            mandate = AcceptanceVerificationMandate.from_dict(document)
            if campaign_id != mandate.campaign_id or campaign_id in mandates:
                raise AcceptanceMandateError("acceptance_mandate_state_identity_invalid")
            if mandate.custody_sequence in custody_sequences:
                raise AcceptanceMandateError("acceptance_mandate_state_sequence_invalid")
            mandates[campaign_id] = mandate
            custody_sequences.add(mandate.custody_sequence)
        committed = int(
            self._trust_store.status().get("committed_sequence") or 0
        )
        if mandates and max(custody_sequences) != committed:
            raise AcceptanceMandateError("acceptance_mandate_state_head_invalid")
        if not mandates and committed:
            raise AcceptanceMandateError("acceptance_mandate_state_head_invalid")
        return mandates

    def get(self, campaign_id: str) -> AcceptanceVerificationMandate:
        canonical_id = _identifier(campaign_id, name="campaign_id")
        with interprocess_file_lock(self._lock_path):
            mandate = self._load_locked().get(canonical_id)
        if mandate is None:
            raise AcceptanceMandateError("acceptance_mandate_not_found")
        return mandate

    def list(self) -> tuple[AcceptanceVerificationMandate, ...]:
        with interprocess_file_lock(self._lock_path):
            mandates = self._load_locked()
        return tuple(mandates[key] for key in sorted(mandates))

    def provision(
        self,
        *,
        campaign_id: str,
        connector_id: str,
        adapter_id: str,
        expected_source_commit_sha256: str,
        expected_physical_identity_sha256: str,
        expected_evidence_class: AcceptanceEvidenceClass,
        target: float,
        target_tolerance: float,
        scenario_id: str = "",
        required_cases: Sequence[str] = REQUIRED_SCALAR_ACCEPTANCE_CASES,
    ) -> AcceptanceMandateProvisionReceipt:
        created = False
        with interprocess_file_lock(self._lock_path):
            mandates = self._load_locked()
            next_sequence = (
                int(self._trust_store.status().get("committed_sequence") or 0) + 1
            )
            proposed = AcceptanceVerificationMandate(
                campaign_id=campaign_id,
                connector_id=connector_id,
                adapter_id=adapter_id,
                expected_source_commit_sha256=expected_source_commit_sha256,
                expected_physical_identity_sha256=expected_physical_identity_sha256,
                expected_evidence_class=expected_evidence_class,
                target=target,
                target_tolerance=target_tolerance,
                scenario_id=scenario_id,
                required_cases=tuple(required_cases),
                provisioned_at_ns=max(1, time.time_ns()),
                custody_sequence=next_sequence,
            )
            existing = mandates.get(proposed.campaign_id)
            if existing is not None:
                if not hmac.compare_digest(existing.contract_sha256, proposed.contract_sha256):
                    raise AcceptanceMandateError("acceptance_mandate_conflict")
                mandate = existing
            else:
                if len(mandates) >= _MAX_MANDATES:
                    raise AcceptanceMandateError("acceptance_mandate_capacity_exceeded")
                mandates[proposed.campaign_id] = proposed
                receipt = self._trust_store.save(
                    {
                        "schema": ACCEPTANCE_MANDATE_STATE_SCHEMA,
                        "mandates": {
                            key: mandates[key].to_dict() for key in sorted(mandates)
                        },
                    }
                )
                if int(receipt.get("sequence") or 0) != proposed.custody_sequence:
                    raise AcceptanceMandateError(
                        "acceptance_mandate_custody_sequence_mismatch"
                    )
                mandate = proposed
                created = True
        identity_sha256 = str(self._trust_store.identity.get("identity_sha256") or "")
        if not _DIGEST.fullmatch(identity_sha256):
            raise AcceptanceMandateError("acceptance_mandate_custody_identity_invalid")
        return AcceptanceMandateProvisionReceipt(
            campaign_id=mandate.campaign_id,
            mandate_sha256=mandate.sha256,
            contract_sha256=mandate.contract_sha256,
            custody_identity_sha256=identity_sha256,
            provisioned_at_ns=mandate.provisioned_at_ns,
            created=created,
            custody_sequence=mandate.custody_sequence,
        )

    def status(self) -> Mapping[str, Any]:
        custody = dict(self._trust_store.status())
        try:
            with interprocess_file_lock(self._lock_path):
                mandate_count = len(self._load_locked())
            error = ""
        except Exception as exc:  # noqa: BLE001 - bounded status boundary
            mandate_count = 0
            error = type(exc).__name__
        return {
            "healthy": bool(custody.get("healthy") and not error),
            "error_type": error,
            "mandate_count": mandate_count,
            "custody": custody,
        }

    def close(self) -> None:
        close = getattr(self._trust_store, "close", None)
        if callable(close):
            close()


__all__ = [
    "ACCEPTANCE_MANDATE_RECEIPT_SCHEMA",
    "ACCEPTANCE_MANDATE_SCHEMA",
    "ACCEPTANCE_MANDATE_STATE_SCHEMA",
    "AcceptanceMandateError",
    "AcceptanceMandateProvisionReceipt",
    "AcceptanceMandateStore",
    "AcceptanceVerificationMandate",
]
