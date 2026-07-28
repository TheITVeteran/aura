"""Create-once custody for causally evolving verified-transition campaigns.

The campaign open record binds only facts available before training: the
schedule, initial policy, trust policy, and schedule root.  Exact group plans
are admitted just in time under task-issuer signatures.  Every later plan must
continue from the policy produced by the preceding durable terminal record.

This module is intentionally independent of the static legacy campaign
ledger.  In particular, it never predeclares future policy hashes or future
group-manifest hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Never, cast

from core.brain.llm.latent_cortex.campaign_trust import (
    EVIDENCE_VERIFIER,
    TASK_ISSUER,
    VerifiedCampaignTrustPolicy,
    verify_role_attestation,
)
from core.learning.verified_transition_episode import canonical_json_bytes
from core.learning.verified_transition_group_admission import (
    validate_transition_group_manifest,
)
from core.runtime.file_read_gateway import read_stable_bytes
from core.runtime.file_write_gateway import FileWriteGateway

CAUSAL_CAMPAIGN_MANIFEST_SCHEMA = (
    "aura.verified_transition.causal_campaign_manifest.v2"
)
CAUSAL_CAMPAIGN_OPEN_SCHEMA = "aura.verified_transition.causal_campaign_open.v1"
CAUSAL_GROUP_START_SCHEMA = "aura.verified_transition.causal_group_start.v1"
CAUSAL_GROUP_TERMINAL_SCHEMA = (
    "aura.verified_transition.causal_group_terminal.v1"
)
CAUSAL_CAMPAIGN_CLOSE_PAYLOAD_SCHEMA = (
    "aura.verified_transition.causal_campaign_close_payload.v4"
)
CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA = (
    "aura.verified_transition.campaign_evidence_manifest.v3"
)
CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V4 = (
    "aura.verified_transition.campaign_evidence_manifest.v4"
)
EXTERNAL_EVIDENCE_VERIFICATION_RECEIPT_SCHEMA = (
    "aura.verified_transition.external_evidence_verification_receipt.v2"
)
CAUSAL_CAMPAIGN_RECEIPT_SCHEMA = (
    "aura.verified_transition.causal_campaign_receipt.v1"
)
LINEAGE_PLAN_SCHEMA = "aura.verified_transition.lineage_plan.v1"

_MAX_GROUPS = 100_000
_MAX_RECORD_BYTES = 16 * 1024 * 1024
_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "campaign_id",
        "provider_contract_sha256",
        "trust_policy_sha256",
        "campaign_schedule_root_sha256",
        "initial_policy_sha256",
        "group_count",
        "schedule",
        "planned_at_unix_ns",
        "manifest_sha256",
    }
)
_SCHEDULE_KEYS = frozenset(
    {"sequence", "task_id", "task_commitment_sha256"}
)
_OPEN_KEYS = frozenset(
    {
        "schema",
        "campaign_manifest",
        "campaign_manifest_attestation",
        "opened_at_unix_ns",
        "receipt_sha256",
    }
)
_LINEAGE_KEYS = frozenset(
    {
        "schema",
        "contract_sha256",
        "campaign_id",
        "campaign_schedule_root_sha256",
        "sequence",
        "task_commitment_sha256",
        "policy_before_sha256",
        "group_manifest_sha256",
    }
)
_START_KEYS = frozenset(
    {
        "schema",
        "campaign_manifest_sha256",
        "campaign_schedule_root_sha256",
        "sequence",
        "group_id",
        "task_commitment_sha256",
        "policy_before_sha256",
        "previous_terminal_sha256",
        "group_manifest",
        "group_manifest_attestation",
        "lineage_plan",
        "lineage_attestation",
        "admitted_at_unix_ns",
        "receipt_sha256",
    }
)
_TERMINAL_KEYS = frozenset(
    {
        "schema",
        "campaign_manifest_sha256",
        "campaign_schedule_root_sha256",
        "sequence",
        "group_id",
        "group_manifest_sha256",
        "group_start_sha256",
        "status",
        "reward_receipt_sha256",
        "group_admission_sha256",
        "update_receipt_sha256",
        "policy_before_sha256",
        "policy_after_sha256",
        "terminal_reason",
        "finished_at_unix_ns",
        "receipt_sha256",
    }
)
_CLOSE_PAYLOAD_KEYS = frozenset(
    {
        "schema",
        "campaign_id",
        "campaign_manifest_sha256",
        "campaign_open_sha256",
        "campaign_schedule_root_sha256",
        "trust_policy_sha256",
        "group_count",
        "group_start_sha256s",
        "group_terminal_sha256s",
        "group_statuses",
        "updated_count",
        "rejected_count",
        "aborted_count",
        "indeterminate_count",
        "initial_policy_sha256",
        "final_policy_sha256",
        "evidence_manifest",
        "external_evidence_verification_receipt",
        "completed_at_unix_ns",
        "payload_sha256",
    }
)
_EVIDENCE_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "contract_sha256",
        "campaign_schedule_root_sha256",
        "trust_policy_sha256",
        "campaign_ledger_root",
        "transition_artifact_root",
        "update_journal_root",
        "transaction_root",
        "completed_groups",
        "halt_reason",
        "group_packages",
        "updated_replay_sequences",
        "created_at_unix_ns",
        "manifest_sha256",
    }
)
_EVIDENCE_PACKAGE_KEYS_V3 = frozenset(
    {
        "sequence",
        "status",
        "package_artifact",
        "package_receipt_sha256",
        "group_manifest_sha256",
        "reward_receipt_sha256",
        "group_admission_sha256",
        "update_receipt_sha256",
        "trainer_step_receipt_sha256",
        "sample_receipt_sha256s",
        "evidence_receipt_sha256s",
    }
)
_EVIDENCE_PACKAGE_KEYS_V4 = _EVIDENCE_PACKAGE_KEYS_V3 | {
    "pre_measurement_sha256"
}
_EVIDENCE_ARTIFACT_KEYS = frozenset({"path", "sha256", "size_bytes"})
_EXTERNAL_VERIFICATION_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "evidence_manifest_sha256",
        "verifier_identity",
        "verified_package_count",
        "artifact_observation_root_sha256",
        "validation_profile",
        "verified_at_unix",
        "receipt_sha256",
    }
)
_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "close_payload",
        "evidence_verifier_attestation",
        "receipt_sha256",
    }
)
_STATUSES = frozenset({"updated", "rejected", "aborted", "indeterminate"})


class VerifiedTransitionCausalCampaignError(RuntimeError):
    """A causal campaign record is unsafe, incomplete, or inconsistent."""


def _fail(code: str) -> Never:
    raise VerifiedTransitionCausalCampaignError(code)


def _sha256(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{role}_invalid")
    return value


def _identifier(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 192
        or not value[0].isalnum()
        or any(
            not (character.isalnum() or character in "._:/;=+-")
            for character in value
        )
    ):
        _fail(f"{role}_invalid")
    return value


def validate_causal_campaign_evidence_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _EVIDENCE_MANIFEST_KEYS:
        _fail("causal_campaign_evidence_manifest_schema_invalid")
    document = _clone(value, role="causal_campaign_evidence_manifest")
    observed = _sha256(
        document.get("manifest_sha256"),
        role="causal_campaign_evidence_manifest",
    )
    unsigned = dict(document)
    unsigned.pop("manifest_sha256")
    packages = document.get("group_packages")
    updated = document.get("updated_replay_sequences")
    completed = document.get("completed_groups")
    if (
        document.get("schema")
        not in {
            CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA,
            CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V4,
        }
        or observed != hashlib.sha256(_json_bytes(unsigned)).hexdigest()
        or type(completed) is not int
        or completed < 0
        or not isinstance(packages, list)
        or len(packages) != completed
        or not isinstance(updated, list)
        or any(type(sequence) is not int for sequence in updated)
        or updated != sorted(set(updated))
        or type(document.get("created_at_unix_ns")) is not int
        or document["created_at_unix_ns"] <= 0
    ):
        _fail("causal_campaign_evidence_manifest_invalid")
    _sha256(document.get("contract_sha256"), role="causal_campaign_contract")
    _sha256(
        document.get("campaign_schedule_root_sha256"),
        role="causal_campaign_evidence_schedule_root",
    )
    _sha256(
        document.get("trust_policy_sha256"),
        role="causal_campaign_evidence_trust_policy",
    )
    for role in (
        "campaign_ledger_root",
        "transition_artifact_root",
        "update_journal_root",
        "transaction_root",
    ):
        artifact_root = document.get(role)
        if (
            not isinstance(artifact_root, str)
            or not Path(artifact_root).is_absolute()
            or Path(artifact_root).resolve(strict=False)
            != Path(artifact_root)
        ):
            _fail(f"causal_campaign_evidence_{role}_invalid")
    _identifier(document.get("halt_reason"), role="causal_campaign_halt_reason")
    package_keys = (
        _EVIDENCE_PACKAGE_KEYS_V4
        if document["schema"]
        == CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V4
        else _EVIDENCE_PACKAGE_KEYS_V3
    )
    for sequence, package in enumerate(packages):
        if (
            not isinstance(package, Mapping)
            or set(package) != package_keys
            or package.get("sequence") != sequence
            or package.get("status") not in {"updated", "rejected"}
            or not isinstance(package.get("sample_receipt_sha256s"), list)
            or not package["sample_receipt_sha256s"]
            or not isinstance(package.get("evidence_receipt_sha256s"), list)
            or len(package["sample_receipt_sha256s"])
            != len(package["evidence_receipt_sha256s"])
        ):
            _fail("causal_campaign_evidence_package_invalid")
        artifact = package.get("package_artifact")
        if (
            not isinstance(artifact, Mapping)
            or set(artifact) != _EVIDENCE_ARTIFACT_KEYS
            or not isinstance(artifact.get("path"), str)
            or not Path(artifact["path"]).is_absolute()
            or Path(artifact["path"]).resolve(strict=False)
            != Path(artifact["path"])
            or type(artifact.get("size_bytes")) is not int
            or artifact["size_bytes"] <= 0
        ):
            _fail("causal_campaign_evidence_artifact_binding_invalid")
        _sha256(
            artifact.get("sha256"),
            role="causal_campaign_evidence_package_artifact",
        )
        for role in (
            "package_receipt_sha256",
            "group_manifest_sha256",
            "reward_receipt_sha256",
            "trainer_step_receipt_sha256",
        ):
            _sha256(
                package.get(role),
                role=f"causal_campaign_evidence_{role}",
            )
        for role in ("group_admission_sha256", "update_receipt_sha256"):
            digest = package.get(role)
            if digest is not None:
                _sha256(
                    digest, role=f"causal_campaign_evidence_{role}"
                )
        if (
            document["schema"]
            == CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V4
        ):
            pre_measurement = package.get("pre_measurement_sha256")
            if pre_measurement is not None:
                _sha256(
                    pre_measurement,
                    role="causal_campaign_evidence_pre_measurement",
                )
        for digest in (
            package["sample_receipt_sha256s"]
            + package["evidence_receipt_sha256s"]
        ):
            _sha256(
                digest, role="causal_campaign_evidence_artifact"
            )
        if (
            (package["status"] == "updated")
            is not (
                package["group_admission_sha256"] is not None
                and package["update_receipt_sha256"] is not None
            )
        ):
            _fail("causal_campaign_evidence_package_status_invalid")
        if (
            document["schema"]
            == CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V4
            and (
                (package["status"] == "updated")
                is not (package["pre_measurement_sha256"] is not None)
            )
        ):
            _fail(
                "causal_campaign_evidence_pre_measurement_status_invalid"
            )
    if updated != [
        package["sequence"]
        for package in packages
        if package["status"] == "updated"
    ]:
        _fail("causal_campaign_evidence_updated_set_invalid")
    return cast(dict[str, Any], document)


def _integer(value: Any, *, role: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum or value > (1 << 63) - 1:
        _fail(f"{role}_invalid")
    return value


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def _seal(value: Mapping[str, Any], *, field_name: str) -> dict[str, Any]:
    sealed = dict(value)
    sealed[field_name] = _digest(sealed)
    return sealed


def _validate_seal(
    value: Mapping[str, Any], *, field_name: str, role: str
) -> str:
    observed = _sha256(value.get(field_name), role=f"{role}_{field_name}")
    unsigned = dict(value)
    unsigned.pop(field_name, None)
    if observed != _digest(unsigned):
        _fail(f"{role}_digest_mismatch")
    return observed


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError) as exc:
        raise VerifiedTransitionCausalCampaignError(
            "causal_campaign_document_invalid"
        ) from exc


def _clone(value: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    try:
        cloned = json.loads(canonical_json_bytes(dict(value)))
    except (TypeError, ValueError, RecursionError, OverflowError) as exc:
        raise VerifiedTransitionCausalCampaignError(f"{role}_invalid") from exc
    if not isinstance(cloned, dict):
        _fail(f"{role}_invalid")
    return cloned


@dataclass(frozen=True, slots=True)
class CausalCampaignScheduleEntry:
    """One knowable schedule commitment; no policy outcome is permitted."""

    sequence: int
    task_id: str
    task_commitment_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": _integer(self.sequence, role="causal_schedule_sequence"),
            "task_id": _identifier(self.task_id, role="causal_schedule_task"),
            "task_commitment_sha256": _sha256(
                self.task_commitment_sha256,
                role="causal_schedule_task_commitment",
            ),
        }


def build_causal_campaign_manifest(
    *,
    campaign_id: str,
    provider_contract_sha256: str,
    campaign_schedule_root_sha256: str,
    trust_policy_sha256: str,
    initial_policy_sha256: str,
    schedule: Sequence[CausalCampaignScheduleEntry],
    planned_at_unix_ns: int,
) -> dict[str, Any]:
    """Seal only schedule facts knowable before the first optimizer update."""

    entries = tuple(schedule)
    if not entries or len(entries) > _MAX_GROUPS:
        _fail("causal_campaign_group_count_invalid")
    normalized = [entry.to_dict() for entry in entries]
    if [row["sequence"] for row in normalized] != list(range(len(normalized))):
        _fail("causal_campaign_schedule_sequence_invalid")
    task_ids = [cast(str, row["task_id"]) for row in normalized]
    if len(set(task_ids)) != len(task_ids):
        _fail("causal_campaign_schedule_task_duplicate")
    return _seal(
        {
            "schema": CAUSAL_CAMPAIGN_MANIFEST_SCHEMA,
            "campaign_id": _identifier(campaign_id, role="causal_campaign_id"),
            "provider_contract_sha256": _sha256(
                provider_contract_sha256,
                role="causal_campaign_provider_contract",
            ),
            "trust_policy_sha256": _sha256(
                trust_policy_sha256, role="causal_campaign_trust_policy"
            ),
            "campaign_schedule_root_sha256": _sha256(
                campaign_schedule_root_sha256,
                role="causal_campaign_schedule_root",
            ),
            "initial_policy_sha256": _sha256(
                initial_policy_sha256, role="causal_campaign_initial_policy"
            ),
            "group_count": len(normalized),
            "schedule": normalized,
            "planned_at_unix_ns": _integer(
                planned_at_unix_ns,
                role="causal_campaign_planned_at",
                minimum=1,
            ),
        },
        field_name="manifest_sha256",
    )


def validate_causal_campaign_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _MANIFEST_KEYS:
        _fail("causal_campaign_manifest_schema_invalid")
    document = _clone(value, role="causal_campaign_manifest")
    if document.get("schema") != CAUSAL_CAMPAIGN_MANIFEST_SCHEMA:
        _fail("causal_campaign_manifest_version_invalid")
    _validate_seal(
        document,
        field_name="manifest_sha256",
        role="causal_campaign_manifest",
    )
    raw_schedule = document.get("schedule")
    if not isinstance(raw_schedule, list):
        _fail("causal_campaign_schedule_invalid")
    entries: list[CausalCampaignScheduleEntry] = []
    for raw in raw_schedule:
        if not isinstance(raw, Mapping) or set(raw) != _SCHEDULE_KEYS:
            _fail("causal_campaign_schedule_entry_schema_invalid")
        entries.append(
            CausalCampaignScheduleEntry(
                sequence=cast(int, raw.get("sequence")),
                task_id=cast(str, raw.get("task_id")),
                task_commitment_sha256=cast(
                    str, raw.get("task_commitment_sha256")
                ),
            )
        )
    expected = build_causal_campaign_manifest(
        campaign_id=cast(str, document.get("campaign_id")),
        provider_contract_sha256=cast(
            str, document.get("provider_contract_sha256")
        ),
        campaign_schedule_root_sha256=cast(
            str, document.get("campaign_schedule_root_sha256")
        ),
        trust_policy_sha256=cast(str, document.get("trust_policy_sha256")),
        initial_policy_sha256=cast(str, document.get("initial_policy_sha256")),
        schedule=entries,
        planned_at_unix_ns=cast(int, document.get("planned_at_unix_ns")),
    )
    if expected != document or document.get("group_count") != len(entries):
        _fail("causal_campaign_manifest_reconstruction_mismatch")
    return document


def validate_external_evidence_verification_receipt(
    value: Any,
    *,
    evidence_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an external verifier's replay receipt against exact packages."""

    evidence = validate_causal_campaign_evidence_manifest(evidence_manifest)
    if (
        not isinstance(value, Mapping)
        or set(value) != _EXTERNAL_VERIFICATION_RECEIPT_KEYS
    ):
        _fail("external_evidence_verification_receipt_schema_invalid")
    document = _clone(value, role="external_evidence_verification_receipt")
    observed = _sha256(
        document.get("receipt_sha256"),
        role="external_evidence_verification_receipt",
    )
    unsigned = dict(document)
    unsigned.pop("receipt_sha256")
    expected_observations = [
        {
            "sequence": package["sequence"],
            "package_artifact": package["package_artifact"],
            "package_receipt_sha256": package["package_receipt_sha256"],
            "sample_receipt_sha256s": package["sample_receipt_sha256s"],
            "evidence_receipt_sha256s": package[
                "evidence_receipt_sha256s"
            ],
            "reward_receipt_sha256": package["reward_receipt_sha256"],
            "group_admission_sha256": package[
                "group_admission_sha256"
            ],
            "update_receipt_sha256": package["update_receipt_sha256"],
            "trainer_step_receipt_sha256": package[
                "trainer_step_receipt_sha256"
            ],
            **(
                {
                    "pre_measurement_sha256": package[
                        "pre_measurement_sha256"
                    ]
                }
                if evidence["schema"]
                == CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V4
                else {}
            ),
        }
        for package in evidence["group_packages"]
    ]
    if (
        document.get("schema")
        != EXTERNAL_EVIDENCE_VERIFICATION_RECEIPT_SCHEMA
        or observed != hashlib.sha256(_json_bytes(unsigned)).hexdigest()
        or document.get("evidence_manifest_sha256")
        != evidence["manifest_sha256"]
        or not isinstance(document.get("verifier_identity"), str)
        or not document["verifier_identity"]
        or document.get("validation_profile")
        != (
            "recurrent_transition_causal_replay.v3"
            if evidence["schema"]
            == CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V4
            else "recurrent_transition_causal_replay.v2"
        )
        or document.get("verified_package_count")
        != len(expected_observations)
        or document.get("artifact_observation_root_sha256")
        != _digest({"artifact_observations": expected_observations})
        or type(document.get("verified_at_unix")) is not int
        or document["verified_at_unix"] <= 0
    ):
        _fail("external_evidence_verification_receipt_invalid")
    return document


def _private_root(path: str | Path, *, create: bool, gateway: FileWriteGateway) -> Path:
    lexical = Path(path).expanduser().absolute()
    if lexical.is_symlink():
        _fail("causal_campaign_root_symlink_rejected")
    if create:
        lexical = Path(
            gateway.ensure_directory(
                lexical, source="verified_transition_causal_campaign.ledger"
            )
        )
    try:
        resolved = lexical.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as exc:
        raise VerifiedTransitionCausalCampaignError(
            "causal_campaign_root_unavailable"
        ) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        _fail("causal_campaign_root_not_private_owned_directory")
    return resolved


@dataclass(slots=True)
class VerifiedTransitionCausalCampaignLedger:
    """Durable adapter implementing the production provider ledger protocol."""

    root: Path
    policy: VerifiedCampaignTrustPolicy
    gateway: FileWriteGateway
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        campaign_manifest: Mapping[str, Any],
        campaign_manifest_attestation: Mapping[str, Any],
        policy: VerifiedCampaignTrustPolicy,
        gateway: FileWriteGateway | None = None,
    ) -> VerifiedTransitionCausalCampaignLedger:
        manifest = validate_causal_campaign_manifest(campaign_manifest)
        if manifest["trust_policy_sha256"] != policy.policy_sha256:
            _fail("causal_campaign_trust_policy_mismatch")
        planned_at = cast(int, manifest["planned_at_unix_ns"])
        signed = verify_role_attestation(
            policy,
            campaign_manifest_attestation,
            role=TASK_ISSUER,
            expected_payload=manifest,
            not_after_unix=planned_at // 1_000_000_000,
        )
        if signed.get("signed_at_unix") != planned_at // 1_000_000_000:
            _fail("causal_campaign_manifest_signature_time_mismatch")
        resolved_gateway = gateway or FileWriteGateway()
        ledger = cls(
            root=_private_root(root, create=True, gateway=resolved_gateway),
            policy=policy,
            gateway=resolved_gateway,
        )
        opened = _seal(
            {
                "schema": CAUSAL_CAMPAIGN_OPEN_SCHEMA,
                "campaign_manifest": manifest,
                "campaign_manifest_attestation": _clone(
                    campaign_manifest_attestation,
                    role="causal_campaign_manifest_attestation",
                ),
                "opened_at_unix_ns": planned_at,
            },
            field_name="receipt_sha256",
        )
        ledger._write_once(
            "campaign.open.json", opened, source="causal_campaign.open"
        )
        return ledger

    @classmethod
    def open(
        cls,
        root: str | Path,
        *,
        policy: VerifiedCampaignTrustPolicy,
        gateway: FileWriteGateway | None = None,
    ) -> VerifiedTransitionCausalCampaignLedger:
        resolved_gateway = gateway or FileWriteGateway()
        ledger = cls(
            root=_private_root(root, create=False, gateway=resolved_gateway),
            policy=policy,
            gateway=resolved_gateway,
        )
        ledger._open_record()
        return ledger

    @staticmethod
    def _record_name(sequence: int, suffix: str) -> str:
        value = _integer(sequence, role="causal_campaign_sequence")
        if suffix not in {"started", "terminal"}:
            _fail("causal_campaign_record_suffix_invalid")
        return f"group-{value:08d}.{suffix}.json"

    def _path(self, name: str) -> Path:
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or "/" in name
            or "\\" in name
            or name in {".", ".."}
        ):
            _fail("causal_campaign_record_path_invalid")
        target = self.root / name
        if target.parent != self.root:
            _fail("causal_campaign_record_path_escape")
        return target

    def _exists(self, name: str) -> bool:
        target = self._path(name)
        if target.is_symlink():
            _fail("causal_campaign_record_symlink_rejected")
        if not target.exists():
            return False
        metadata = target.stat()
        if not stat.S_ISREG(metadata.st_mode):
            _fail("causal_campaign_record_not_regular")
        return True

    def _write_once(self, name: str, value: Mapping[str, Any], *, source: str) -> None:
        target = self._path(name)
        if not self.gateway.write_bytes_if_absent(
            target,
            _json_bytes(value),
            source=f"verified_transition_causal_campaign.{source}",
            durable=True,
        ):
            _fail(f"causal_campaign_record_already_exists:{name}")
        self._read(name)

    def _read(self, name: str) -> dict[str, Any]:
        target = self._path(name)
        try:
            before = target.lstat()
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.getuid()
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) & 0o077
            ):
                _fail("causal_campaign_record_not_private_owned_file")
            payload = read_stable_bytes(target, max_bytes=_MAX_RECORD_BYTES)
            after = target.lstat()
        except FileNotFoundError as exc:
            raise VerifiedTransitionCausalCampaignError(
                f"causal_campaign_record_missing:{name}"
            ) from exc
        except OSError as exc:
            raise VerifiedTransitionCausalCampaignError(
                f"causal_campaign_record_unreadable:{name}"
            ) from exc
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            _fail("causal_campaign_record_changed_during_read")
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VerifiedTransitionCausalCampaignError(
                "causal_campaign_record_json_invalid"
            ) from exc
        if not isinstance(document, dict) or _json_bytes(document) != payload:
            _fail("causal_campaign_record_noncanonical")
        return document

    def _open_record(self) -> tuple[dict[str, Any], dict[str, Any]]:
        opened = self._read("campaign.open.json")
        if set(opened) != _OPEN_KEYS or opened.get("schema") != CAUSAL_CAMPAIGN_OPEN_SCHEMA:
            _fail("causal_campaign_open_schema_invalid")
        _validate_seal(opened, field_name="receipt_sha256", role="causal_campaign_open")
        manifest = validate_causal_campaign_manifest(opened.get("campaign_manifest"))
        if (
            opened.get("opened_at_unix_ns") != manifest["planned_at_unix_ns"]
            or manifest["trust_policy_sha256"] != self.policy.policy_sha256
        ):
            _fail("causal_campaign_open_reconstruction_mismatch")
        planned_at = cast(int, manifest["planned_at_unix_ns"])
        signed = verify_role_attestation(
            self.policy,
            opened.get("campaign_manifest_attestation"),
            role=TASK_ISSUER,
            expected_payload=manifest,
            not_after_unix=planned_at // 1_000_000_000,
        )
        if signed.get("signed_at_unix") != planned_at // 1_000_000_000:
            _fail("causal_campaign_manifest_signature_time_mismatch")
        return opened, manifest

    def campaign_manifest(self) -> dict[str, Any]:
        """Return a validated copy of the immutable campaign-open manifest."""

        with self._lock:
            _opened, manifest = self._open_record()
            return _clone(manifest, role="causal_campaign_manifest")

    def _assert_policy(self, policy: VerifiedCampaignTrustPolicy) -> None:
        if (
            not isinstance(policy, VerifiedCampaignTrustPolicy)
            or policy.policy_sha256 != self.policy.policy_sha256
            or policy.root_key_id != self.policy.root_key_id
            or policy.document != self.policy.document
        ):
            _fail("causal_campaign_runtime_policy_substitution")

    def _assert_not_closed(self) -> None:
        if self._exists("campaign.closed.json"):
            _fail("causal_campaign_already_closed")

    def _schedule_entry(
        self, manifest: Mapping[str, Any], sequence: int
    ) -> dict[str, Any]:
        sequence = _integer(sequence, role="causal_campaign_sequence")
        if sequence >= cast(int, manifest["group_count"]):
            _fail("causal_campaign_group_not_scheduled")
        return cast(dict[str, Any], manifest["schedule"][sequence])

    def _terminal_identity(self, sequence: int) -> dict[str, Any]:
        terminal = self._read(self._record_name(sequence, "terminal"))
        if (
            set(terminal) != _TERMINAL_KEYS
            or terminal.get("schema") != CAUSAL_GROUP_TERMINAL_SCHEMA
            or terminal.get("sequence") != sequence
        ):
            _fail("causal_group_terminal_schema_invalid")
        _validate_seal(
            terminal,
            field_name="receipt_sha256",
            role="causal_group_terminal",
        )
        return terminal

    def _lineage_head(
        self, manifest: Mapping[str, Any], sequence: int
    ) -> tuple[str, str | None]:
        """Validate the constant-size predecessor link needed for one append."""

        sequence = _integer(sequence, role="causal_campaign_sequence")
        if sequence == 0:
            return cast(str, manifest["initial_policy_sha256"]), None
        prior_sequence = sequence - 1
        if prior_sequence == 0:
            prior_expected_policy = cast(str, manifest["initial_policy_sha256"])
            prior_previous_terminal = None
        else:
            predecessor = self._terminal_identity(prior_sequence - 1)
            predecessor_after = predecessor.get("policy_after_sha256")
            if predecessor.get("status") == "indeterminate" or predecessor_after is None:
                _fail("causal_campaign_policy_lineage_indeterminate")
            prior_expected_policy = _sha256(
                predecessor_after, role="causal_campaign_predecessor_policy"
            )
            prior_previous_terminal = cast(str, predecessor["receipt_sha256"])
        prior_start = self._read(self._record_name(prior_sequence, "started"))
        self._validate_start(
            sequence=prior_sequence,
            start=prior_start,
            manifest=manifest,
            expected_policy_before=prior_expected_policy,
            expected_previous_terminal_sha256=prior_previous_terminal,
        )
        prior_terminal = self._read(self._record_name(prior_sequence, "terminal"))
        self._validate_terminal(
            sequence=prior_sequence,
            terminal=prior_terminal,
            start=prior_start,
            manifest=manifest,
        )
        policy_after = prior_terminal.get("policy_after_sha256")
        if prior_terminal.get("status") == "indeterminate" or policy_after is None:
            _fail("causal_campaign_policy_lineage_indeterminate")
        return (
            _sha256(policy_after, role="causal_campaign_policy_lineage_head"),
            cast(str, prior_terminal["receipt_sha256"]),
        )

    def _validate_start(
        self,
        *,
        sequence: int,
        start: Mapping[str, Any],
        manifest: Mapping[str, Any],
        expected_policy_before: str,
        expected_previous_terminal_sha256: str | None,
    ) -> dict[str, Any]:
        if set(start) != _START_KEYS or start.get("schema") != CAUSAL_GROUP_START_SCHEMA:
            _fail("causal_group_start_schema_invalid")
        _validate_seal(start, field_name="receipt_sha256", role="causal_group_start")
        schedule = self._schedule_entry(manifest, sequence)
        expected_before = _sha256(
            expected_policy_before, role="causal_group_expected_policy_before"
        )
        previous_terminal = start.get("previous_terminal_sha256")
        if expected_previous_terminal_sha256 is None:
            if previous_terminal is not None:
                _fail("causal_group_previous_terminal_unexpected")
        elif _sha256(
            previous_terminal, role="causal_group_previous_terminal"
        ) != expected_previous_terminal_sha256:
            _fail("causal_group_previous_terminal_mismatch")
        group = validate_transition_group_manifest(start.get("group_manifest"))
        lineage = start.get("lineage_plan")
        group_attestation = start.get("group_manifest_attestation")
        lineage_attestation = start.get("lineage_attestation")
        admitted_at = _integer(
            start.get("admitted_at_unix_ns"),
            role="causal_group_admitted_at",
            minimum=1,
        )
        if not isinstance(lineage, Mapping) or set(lineage) != _LINEAGE_KEYS:
            _fail("causal_group_lineage_plan_schema_invalid")
        if (
            lineage.get("schema") != LINEAGE_PLAN_SCHEMA
            or start.get("campaign_manifest_sha256") != manifest["manifest_sha256"]
            or start.get("campaign_schedule_root_sha256")
            != manifest["campaign_schedule_root_sha256"]
            or start.get("sequence") != sequence
            or start.get("group_id") != group["group_id"]
            or start.get("task_commitment_sha256")
            != schedule["task_commitment_sha256"]
            or start.get("policy_before_sha256") != expected_before
            or group["task_id"] != schedule["task_id"]
            or group["planned_at_unix_ns"] >= admitted_at
            or any(
                entry["policy_sha256"] != expected_before
                for entry in group["entries"]
            )
            or lineage.get("campaign_id") != manifest["campaign_id"]
            or lineage.get("contract_sha256")
            != manifest["provider_contract_sha256"]
            or lineage.get("campaign_schedule_root_sha256")
            != manifest["campaign_schedule_root_sha256"]
            or lineage.get("sequence") != sequence
            or lineage.get("task_commitment_sha256")
            != schedule["task_commitment_sha256"]
            or lineage.get("policy_before_sha256") != expected_before
            or lineage.get("group_manifest_sha256") != group["manifest_sha256"]
        ):
            _fail("causal_group_start_reconstruction_mismatch")
        _sha256(lineage.get("contract_sha256"), role="causal_group_contract")
        planned_second = cast(int, group["planned_at_unix_ns"]) // 1_000_000_000
        admitted_second = admitted_at // 1_000_000_000
        verify_role_attestation(
            self.policy,
            group_attestation,
            role=TASK_ISSUER,
            expected_payload=group,
            not_after_unix=planned_second,
        )
        verify_role_attestation(
            self.policy,
            lineage_attestation,
            role=TASK_ISSUER,
            expected_payload=cast(Mapping[str, Any], lineage),
            not_after_unix=admitted_second,
        )
        return group

    def _validate_terminal(
        self,
        *,
        sequence: int,
        terminal: Mapping[str, Any],
        start: Mapping[str, Any],
        manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        if (
            set(terminal) != _TERMINAL_KEYS
            or terminal.get("schema") != CAUSAL_GROUP_TERMINAL_SCHEMA
        ):
            _fail("causal_group_terminal_schema_invalid")
        _validate_seal(
            terminal,
            field_name="receipt_sha256",
            role="causal_group_terminal",
        )
        group = validate_transition_group_manifest(start.get("group_manifest"))
        before = _sha256(
            terminal.get("policy_before_sha256"),
            role="causal_group_terminal_policy_before",
        )
        status_value = terminal.get("status")
        if status_value not in _STATUSES:
            _fail("causal_group_terminal_status_invalid")
        status_value = cast(str, status_value)
        raw_after = terminal.get("policy_after_sha256")
        after = (
            None
            if status_value == "indeterminate" and raw_after is None
            else _sha256(raw_after, role="causal_group_terminal_policy_after")
        )
        reward = terminal.get("reward_receipt_sha256")
        admission = terminal.get("group_admission_sha256")
        update = terminal.get("update_receipt_sha256")
        if reward is not None:
            _sha256(reward, role="causal_group_terminal_reward")
        if admission is not None:
            _sha256(admission, role="causal_group_terminal_admission")
        if update is not None:
            _sha256(update, role="causal_group_terminal_update")
        evidence_valid = (
            status_value == "updated"
            and reward is not None
            and admission is not None
            and update is not None
            and after is not None
            and before != after
        ) or (
            status_value == "rejected"
            and reward is not None
            and admission is None
            and update is None
            and before == after
        ) or (
            status_value == "aborted"
            and reward is None
            and admission is None
            and update is None
            and before == after
        ) or (
            status_value == "indeterminate"
            and update is None
            and after is None
        )
        if (
            not evidence_valid
            or terminal.get("campaign_manifest_sha256")
            != manifest["manifest_sha256"]
            or terminal.get("campaign_schedule_root_sha256")
            != manifest["campaign_schedule_root_sha256"]
            or terminal.get("sequence") != sequence
            or terminal.get("group_id") != group["group_id"]
            or terminal.get("group_manifest_sha256") != group["manifest_sha256"]
            or terminal.get("group_start_sha256") != start["receipt_sha256"]
            or before != start["policy_before_sha256"]
            or _integer(
                terminal.get("finished_at_unix_ns"),
                role="causal_group_terminal_finished_at",
                minimum=1,
            )
            < cast(int, start["admitted_at_unix_ns"])
        ):
            _fail("causal_group_terminal_reconstruction_mismatch")
        _identifier(
            terminal.get("terminal_reason"), role="causal_group_terminal_reason"
        )
        return dict(terminal)

    def _record_pairs_through(
        self,
        last_sequence: int,
        *,
        manifest: Mapping[str, Any] | None = None,
    ) -> tuple[tuple[dict[str, Any], dict[str, Any]], ...]:
        """Replay a policy chain once, without recursive or quadratic descent."""

        last = _integer(last_sequence, role="causal_campaign_sequence")
        campaign = (
            dict(manifest) if manifest is not None else self._open_record()[1]
        )
        self._schedule_entry(campaign, last)
        expected_policy = cast(str, campaign["initial_policy_sha256"])
        previous_terminal_sha256: str | None = None
        records: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for sequence in range(last + 1):
            start = self._read(self._record_name(sequence, "started"))
            self._validate_start(
                sequence=sequence,
                start=start,
                manifest=campaign,
                expected_policy_before=expected_policy,
                expected_previous_terminal_sha256=previous_terminal_sha256,
            )
            terminal = self._read(self._record_name(sequence, "terminal"))
            self._validate_terminal(
                sequence=sequence,
                terminal=terminal,
                start=start,
                manifest=campaign,
            )
            policy_after = terminal.get("policy_after_sha256")
            if sequence < last and (
                terminal.get("status") == "indeterminate" or policy_after is None
            ):
                _fail("causal_campaign_policy_lineage_indeterminate")
            if policy_after is not None:
                expected_policy = cast(str, policy_after)
            previous_terminal_sha256 = cast(str, terminal["receipt_sha256"])
            records.append((start, terminal))
        return tuple(records)

    def _record_pair(
        self, sequence: int
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        _opened, manifest = self._open_record()
        expected_policy, previous_terminal = self._lineage_head(manifest, sequence)
        start = self._read(self._record_name(sequence, "started"))
        self._validate_start(
            sequence=sequence,
            start=start,
            manifest=manifest,
            expected_policy_before=expected_policy,
            expected_previous_terminal_sha256=previous_terminal,
        )
        terminal = self._read(self._record_name(sequence, "terminal"))
        self._validate_terminal(
            sequence=sequence,
            terminal=terminal,
            start=start,
            manifest=manifest,
        )
        return start, terminal

    def admit_group_plan(
        self,
        *,
        sequence: int,
        campaign_id: str,
        campaign_schedule_root_sha256: str,
        policy_before_sha256: str,
        group_manifest: Mapping[str, Any],
        group_manifest_attestation: Mapping[str, Any],
        lineage_plan: Mapping[str, Any],
        lineage_attestation: Mapping[str, Any],
        policy: VerifiedCampaignTrustPolicy,
        admitted_at_unix_ns: int,
    ) -> Mapping[str, Any]:
        """Persist one externally signed exact plan before its sampling boundary."""

        with self._lock:
            self._assert_policy(policy)
            self._assert_not_closed()
            opened, manifest = self._open_record()
            sequence = _integer(sequence, role="causal_group_sequence")
            schedule = self._schedule_entry(manifest, sequence)
            expected_before, previous_terminal = self._lineage_head(
                manifest, sequence
            )
            group = validate_transition_group_manifest(group_manifest)
            admitted_at = _integer(
                admitted_at_unix_ns,
                role="causal_group_admitted_at",
                minimum=1,
            )
            if (
                campaign_id != manifest["campaign_id"]
                or campaign_schedule_root_sha256
                != manifest["campaign_schedule_root_sha256"]
                or _sha256(
                    policy_before_sha256, role="causal_group_policy_before"
                )
                != expected_before
                or self._exists(self._record_name(sequence, "started"))
                or self._exists(self._record_name(sequence, "terminal"))
            ):
                _fail("causal_group_plan_sequence_or_lineage_invalid")
            if group["planned_at_unix_ns"] >= admitted_at:
                _fail("causal_group_plan_post_disclosure")
            if not isinstance(lineage_plan, Mapping) or set(lineage_plan) != _LINEAGE_KEYS:
                _fail("causal_group_lineage_plan_schema_invalid")
            expected_lineage = {
                "schema": LINEAGE_PLAN_SCHEMA,
                "contract_sha256": _sha256(
                    lineage_plan.get("contract_sha256"),
                    role="causal_group_contract",
                ),
                "campaign_id": manifest["campaign_id"],
                "campaign_schedule_root_sha256": manifest[
                    "campaign_schedule_root_sha256"
                ],
                "sequence": sequence,
                "task_commitment_sha256": schedule["task_commitment_sha256"],
                "policy_before_sha256": expected_before,
                "group_manifest_sha256": group["manifest_sha256"],
            }
            if dict(lineage_plan) != expected_lineage:
                _fail("causal_group_lineage_plan_mismatch")
            start = _seal(
                {
                    "schema": CAUSAL_GROUP_START_SCHEMA,
                    "campaign_manifest_sha256": manifest["manifest_sha256"],
                    "campaign_schedule_root_sha256": manifest[
                        "campaign_schedule_root_sha256"
                    ],
                    "sequence": sequence,
                    "group_id": group["group_id"],
                    "task_commitment_sha256": schedule[
                        "task_commitment_sha256"
                    ],
                    "policy_before_sha256": expected_before,
                    "previous_terminal_sha256": previous_terminal,
                    "group_manifest": group,
                    "group_manifest_attestation": _clone(
                        group_manifest_attestation,
                        role="causal_group_manifest_attestation",
                    ),
                    "lineage_plan": expected_lineage,
                    "lineage_attestation": _clone(
                        lineage_attestation, role="causal_group_lineage_attestation"
                    ),
                    "admitted_at_unix_ns": admitted_at,
                },
                field_name="receipt_sha256",
            )
            self._validate_start(
                sequence=sequence,
                start=start,
                manifest=manifest,
                expected_policy_before=expected_before,
                expected_previous_terminal_sha256=previous_terminal,
            )
            self._write_once(
                self._record_name(sequence, "started"),
                start,
                source="group_start",
            )
            if opened["campaign_manifest"]["manifest_sha256"] != manifest[
                "manifest_sha256"
            ]:
                _fail("causal_campaign_open_changed_during_admission")
            return dict(start)

    def validate_started_group(
        self, *, sequence: int, group_manifest: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        with self._lock:
            self._assert_not_closed()
            _opened, manifest = self._open_record()
            sequence = _integer(sequence, role="causal_group_sequence")
            start = self._read(self._record_name(sequence, "started"))
            expected_before, previous_terminal = self._lineage_head(
                manifest, sequence
            )
            observed = self._validate_start(
                sequence=sequence,
                start=start,
                manifest=manifest,
                expected_policy_before=expected_before,
                expected_previous_terminal_sha256=previous_terminal,
            )
            if observed != validate_transition_group_manifest(group_manifest):
                _fail("causal_group_started_manifest_substitution")
            if self._exists(self._record_name(sequence, "terminal")):
                _fail("causal_group_already_terminal")
            return dict(start)

    def finish_group(
        self,
        *,
        sequence: int,
        status: str,
        reward_receipt_sha256: str | None,
        group_admission_sha256: str | None,
        update_receipt_sha256: str | None,
        terminal_reason: str,
        finished_at_unix_ns: int,
        policy_after_sha256: str | None = None,
    ) -> Mapping[str, Any]:
        """Seal one terminal; updated groups must disclose the actual new policy."""

        with self._lock:
            self._assert_not_closed()
            _opened, manifest = self._open_record()
            sequence = _integer(sequence, role="causal_group_sequence")
            start = self._read(self._record_name(sequence, "started"))
            expected_before, previous_terminal = self._lineage_head(
                manifest, sequence
            )
            group = self._validate_start(
                sequence=sequence,
                start=start,
                manifest=manifest,
                expected_policy_before=expected_before,
                expected_previous_terminal_sha256=previous_terminal,
            )
            if self._exists(self._record_name(sequence, "terminal")):
                _fail("causal_group_already_terminal")
            if status not in _STATUSES:
                _fail("causal_group_terminal_status_invalid")
            before = cast(str, start["policy_before_sha256"])
            if status == "updated":
                after = _sha256(
                    policy_after_sha256,
                    role="causal_group_terminal_policy_after",
                )
            elif status == "indeterminate":
                after = None
                if policy_after_sha256 is not None:
                    _fail("causal_group_indeterminate_policy_must_be_unknown")
            else:
                after = before
                if policy_after_sha256 is not None and policy_after_sha256 != before:
                    _fail("causal_group_nonupdated_policy_changed")
            terminal = _seal(
                {
                    "schema": CAUSAL_GROUP_TERMINAL_SCHEMA,
                    "campaign_manifest_sha256": manifest["manifest_sha256"],
                    "campaign_schedule_root_sha256": manifest[
                        "campaign_schedule_root_sha256"
                    ],
                    "sequence": sequence,
                    "group_id": group["group_id"],
                    "group_manifest_sha256": group["manifest_sha256"],
                    "group_start_sha256": start["receipt_sha256"],
                    "status": status,
                    "reward_receipt_sha256": reward_receipt_sha256,
                    "group_admission_sha256": group_admission_sha256,
                    "update_receipt_sha256": update_receipt_sha256,
                    "policy_before_sha256": before,
                    "policy_after_sha256": after,
                    "terminal_reason": _identifier(
                        terminal_reason, role="causal_group_terminal_reason"
                    ),
                    "finished_at_unix_ns": _integer(
                        finished_at_unix_ns,
                        role="causal_group_terminal_finished_at",
                        minimum=1,
                    ),
                },
                field_name="receipt_sha256",
            )
            self._validate_terminal(
                sequence=sequence,
                terminal=terminal,
                start=start,
                manifest=manifest,
            )
            self._write_once(
                self._record_name(sequence, "terminal"),
                terminal,
                source="group_terminal",
            )
            return dict(terminal)

    def group_records_unclosed(
        self, *, sequence: int
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        with self._lock:
            return self._record_pair(sequence)

    def group_start(self, *, sequence: int) -> Mapping[str, Any]:
        """Return the validated durable start, including for an open group."""

        with self._lock:
            _opened, manifest = self._open_record()
            expected_policy, previous_terminal = self._lineage_head(
                manifest, sequence
            )
            start = self._read(self._record_name(sequence, "started"))
            self._validate_start(
                sequence=sequence,
                start=start,
                manifest=manifest,
                expected_policy_before=expected_policy,
                expected_previous_terminal_sha256=previous_terminal,
            )
            return dict(start)

    def group_start_if_exists(
        self, *, sequence: int
    ) -> Mapping[str, Any] | None:
        """Return a validated start or ``None`` before first publication."""

        with self._lock:
            _opened, manifest = self._open_record()
            expected_policy, previous_terminal = self._lineage_head(
                manifest, sequence
            )
            record_name = self._record_name(sequence, "started")
            if not self._exists(record_name):
                return None
            start = self._read(record_name)
            self._validate_start(
                sequence=sequence,
                start=start,
                manifest=manifest,
                expected_policy_before=expected_policy,
                expected_previous_terminal_sha256=previous_terminal,
            )
            return dict(start)

    def group_terminal_if_exists(
        self, *, sequence: int
    ) -> Mapping[str, Any] | None:
        """Return a validated terminal or ``None`` for a valid open group."""

        with self._lock:
            _opened, manifest = self._open_record()
            expected_policy, previous_terminal = self._lineage_head(
                manifest, sequence
            )
            start = self._read(self._record_name(sequence, "started"))
            self._validate_start(
                sequence=sequence,
                start=start,
                manifest=manifest,
                expected_policy_before=expected_policy,
                expected_previous_terminal_sha256=previous_terminal,
            )
            if not self._exists(self._record_name(sequence, "terminal")):
                return None
            terminal = self._read(self._record_name(sequence, "terminal"))
            return self._validate_terminal(
                sequence=sequence,
                terminal=terminal,
                start=start,
                manifest=manifest,
            )

    def group_records(
        self, *, sequence: int, policy: VerifiedCampaignTrustPolicy
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        with self._lock:
            self.validate_closed(policy=policy)
            return self._record_pair(sequence)

    def _close_payload_snapshot(
        self,
        *,
        completed_at_unix_ns: int,
        policy: VerifiedCampaignTrustPolicy,
        evidence_manifest: Mapping[str, Any],
        external_evidence_verification_receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._assert_policy(policy)
        opened, manifest = self._open_record()
        starts: list[str | None] = []
        terminals: list[str | None] = []
        statuses: list[str] = []
        final_policy: str | None = cast(str, manifest["initial_policy_sha256"])
        latest = cast(int, manifest["planned_at_unix_ns"])
        group_count = cast(int, manifest["group_count"])
        tail_started = False
        records: list[tuple[dict[str, Any], dict[str, Any]] | None] = []
        for sequence in range(group_count):
            has_start = self._exists(self._record_name(sequence, "started"))
            has_terminal = self._exists(self._record_name(sequence, "terminal"))
            if not has_start:
                tail_started = True
                if has_terminal:
                    _fail("causal_campaign_terminal_without_start")
                records.append(None)
                continue
            if tail_started:
                _fail("causal_campaign_noncontiguous_started_tail")
            if not has_terminal:
                raise VerifiedTransitionCausalCampaignError(
                    f"causal_campaign_incomplete:sequence={sequence}"
                )
            records.append(self._record_pair(sequence))
        for record in records:
            if record is None:
                starts.append(None)
                terminals.append(None)
                statuses.append("aborted")
                continue
            start, terminal = record
            if final_policy is None or start["policy_before_sha256"] != final_policy:
                _fail("causal_campaign_policy_lineage_broken")
            final_policy = cast(str | None, terminal["policy_after_sha256"])
            starts.append(cast(str, start["receipt_sha256"]))
            terminals.append(cast(str, terminal["receipt_sha256"]))
            statuses.append(cast(str, terminal["status"]))
            latest = max(latest, cast(int, terminal["finished_at_unix_ns"]))
        completed_at = _integer(
            completed_at_unix_ns,
            role="causal_campaign_completed_at",
            minimum=1,
        )
        if completed_at < latest:
            _fail("causal_campaign_close_time_reversed")
        evidence = validate_causal_campaign_evidence_manifest(
            evidence_manifest
        )
        external_verification = (
            validate_external_evidence_verification_receipt(
                external_evidence_verification_receipt,
                evidence_manifest=evidence,
            )
        )
        completed_groups = next(
            (
                index
                for index, status in enumerate(statuses)
                if status in {"aborted", "indeterminate"}
            ),
            len(statuses),
        )
        if (
            evidence["contract_sha256"]
            != manifest["provider_contract_sha256"]
            or evidence["campaign_schedule_root_sha256"]
            != manifest["campaign_schedule_root_sha256"]
            or evidence["trust_policy_sha256"]
            != manifest["trust_policy_sha256"]
            or evidence["campaign_ledger_root"] != str(self.root)
            or evidence["completed_groups"] != completed_groups
            or [
                package["status"]
                for package in evidence["group_packages"]
            ]
            != statuses[:completed_groups]
        ):
            _fail("causal_campaign_evidence_manifest_campaign_mismatch")
        return _seal(
            {
                "schema": CAUSAL_CAMPAIGN_CLOSE_PAYLOAD_SCHEMA,
                "campaign_id": manifest["campaign_id"],
                "campaign_manifest_sha256": manifest["manifest_sha256"],
                "campaign_open_sha256": opened["receipt_sha256"],
                "campaign_schedule_root_sha256": manifest[
                    "campaign_schedule_root_sha256"
                ],
                "trust_policy_sha256": manifest["trust_policy_sha256"],
                "group_count": manifest["group_count"],
                "group_start_sha256s": starts,
                "group_terminal_sha256s": terminals,
                "group_statuses": statuses,
                "updated_count": statuses.count("updated"),
                "rejected_count": statuses.count("rejected"),
                "aborted_count": statuses.count("aborted"),
                "indeterminate_count": statuses.count("indeterminate"),
                "initial_policy_sha256": manifest["initial_policy_sha256"],
                "final_policy_sha256": final_policy,
                "evidence_manifest": evidence,
                "external_evidence_verification_receipt": (
                    external_verification
                ),
                "completed_at_unix_ns": completed_at,
            },
            field_name="payload_sha256",
        )

    def close_payload(
        self,
        *,
        completed_at_unix_ns: int,
        policy: VerifiedCampaignTrustPolicy,
        evidence_manifest: Mapping[str, Any],
        external_evidence_verification_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        with self._lock:
            self._assert_not_closed()
            return self._close_payload_snapshot(
                completed_at_unix_ns=completed_at_unix_ns,
                policy=policy,
                evidence_manifest=evidence_manifest,
                external_evidence_verification_receipt=(
                    external_evidence_verification_receipt
                ),
            )

    def close(
        self,
        *,
        close_payload: Mapping[str, Any],
        evidence_verifier_attestation: Mapping[str, Any],
        policy: VerifiedCampaignTrustPolicy,
    ) -> Mapping[str, Any]:
        with self._lock:
            self._assert_not_closed()
            if not isinstance(close_payload, Mapping) or set(close_payload) != _CLOSE_PAYLOAD_KEYS:
                _fail("causal_campaign_close_payload_schema_invalid")
            expected = self._close_payload_snapshot(
                completed_at_unix_ns=_integer(
                    close_payload.get("completed_at_unix_ns"),
                    role="causal_campaign_completed_at",
                    minimum=1,
                ),
                policy=policy,
                evidence_manifest=cast(
                    Mapping[str, Any],
                    close_payload.get("evidence_manifest"),
                ),
                external_evidence_verification_receipt=cast(
                    Mapping[str, Any],
                    close_payload.get(
                        "external_evidence_verification_receipt"
                    ),
                ),
            )
            if dict(close_payload) != expected:
                _fail("causal_campaign_close_payload_mismatch")
            completed_at = cast(int, expected["completed_at_unix_ns"])
            verify_role_attestation(
                policy,
                evidence_verifier_attestation,
                role=EVIDENCE_VERIFIER,
                expected_payload=expected,
                not_before_unix=(completed_at + 999_999_999) // 1_000_000_000,
            )
            receipt = _seal(
                {
                    "schema": CAUSAL_CAMPAIGN_RECEIPT_SCHEMA,
                    "close_payload": expected,
                    "evidence_verifier_attestation": _clone(
                        evidence_verifier_attestation,
                        role="causal_campaign_close_attestation",
                    ),
                },
                field_name="receipt_sha256",
            )
            self._write_once(
                "campaign.closed.json", receipt, source="campaign_close"
            )
            return dict(receipt)

    def validate_closed(
        self, *, policy: VerifiedCampaignTrustPolicy
    ) -> Mapping[str, Any]:
        with self._lock:
            self._assert_policy(policy)
            receipt = self._read("campaign.closed.json")
            if (
                set(receipt) != _RECEIPT_KEYS
                or receipt.get("schema") != CAUSAL_CAMPAIGN_RECEIPT_SCHEMA
            ):
                _fail("causal_campaign_receipt_schema_invalid")
            _validate_seal(
                receipt,
                field_name="receipt_sha256",
                role="causal_campaign_receipt",
            )
            close_payload = receipt.get("close_payload")
            if not isinstance(close_payload, Mapping) or set(close_payload) != _CLOSE_PAYLOAD_KEYS:
                _fail("causal_campaign_close_payload_schema_invalid")
            expected = self._close_payload_snapshot(
                completed_at_unix_ns=_integer(
                    close_payload.get("completed_at_unix_ns"),
                    role="causal_campaign_completed_at",
                    minimum=1,
                ),
                policy=policy,
                evidence_manifest=cast(
                    Mapping[str, Any],
                    close_payload.get("evidence_manifest"),
                ),
                external_evidence_verification_receipt=cast(
                    Mapping[str, Any],
                    close_payload.get(
                        "external_evidence_verification_receipt"
                    ),
                ),
            )
            if dict(close_payload) != expected:
                _fail("causal_campaign_close_reconstruction_mismatch")
            completed_at = cast(int, expected["completed_at_unix_ns"])
            verify_role_attestation(
                policy,
                receipt.get("evidence_verifier_attestation"),
                role=EVIDENCE_VERIFIER,
                expected_payload=expected,
                not_before_unix=(completed_at + 999_999_999) // 1_000_000_000,
            )
            return dict(receipt)

    def validate_closed_if_exists(
        self, *, policy: VerifiedCampaignTrustPolicy
    ) -> Mapping[str, Any] | None:
        """Return one fully replayed closure, or None when close never published."""

        with self._lock:
            self._assert_policy(policy)
            if not self._exists("campaign.closed.json"):
                return None
            return self.validate_closed(policy=policy)

    def validate_open_manifest(
        self,
        *,
        expected_manifest: Mapping[str, Any],
        policy: VerifiedCampaignTrustPolicy,
    ) -> Mapping[str, Any]:
        """Reopen and compare the exact immutable campaign manifest."""

        with self._lock:
            self._assert_policy(policy)
            _opened, manifest = self._open_record()
            expected = validate_causal_campaign_manifest(expected_manifest)
            if manifest != expected:
                _fail("causal_campaign_open_manifest_mismatch")
            return dict(manifest)


__all__ = [
    "CAUSAL_CAMPAIGN_CLOSE_PAYLOAD_SCHEMA",
    "CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA",
    "CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V4",
    "CAUSAL_CAMPAIGN_MANIFEST_SCHEMA",
    "CAUSAL_CAMPAIGN_OPEN_SCHEMA",
    "CAUSAL_CAMPAIGN_RECEIPT_SCHEMA",
    "CAUSAL_GROUP_START_SCHEMA",
    "CAUSAL_GROUP_TERMINAL_SCHEMA",
    "EXTERNAL_EVIDENCE_VERIFICATION_RECEIPT_SCHEMA",
    "CausalCampaignScheduleEntry",
    "VerifiedTransitionCausalCampaignError",
    "VerifiedTransitionCausalCampaignLedger",
    "build_causal_campaign_manifest",
    "validate_causal_campaign_manifest",
    "validate_causal_campaign_evidence_manifest",
    "validate_external_evidence_verification_receipt",
]
