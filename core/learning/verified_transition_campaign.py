"""Campaign-completeness custody for verified recurrent transition updates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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

CAMPAIGN_MANIFEST_SCHEMA = "aura.verified_transition.campaign_manifest.v1"
CAMPAIGN_OPEN_SCHEMA = "aura.verified_transition.campaign_open.v1"
CAMPAIGN_GROUP_START_SCHEMA = "aura.verified_transition.campaign_group_start.v1"
CAMPAIGN_GROUP_TERMINAL_SCHEMA = "aura.verified_transition.campaign_group_terminal.v1"
CAMPAIGN_CLOSE_PAYLOAD_SCHEMA = "aura.verified_transition.campaign_close_payload.v1"
CAMPAIGN_RECEIPT_SCHEMA = "aura.verified_transition.campaign_receipt.v1"
_MAX_GROUPS = 100_000


class VerifiedTransitionCampaignError(RuntimeError):
    """Stable campaign-custody failure."""


def _fail(code: str) -> Never:
    raise VerifiedTransitionCampaignError(code)


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
        or len(value) > 192
        or not value[0].isalnum()
        or any(not (character.isalnum() or character in "._:/;=+-") for character in value)
    ):
        _fail(f"{role}_invalid")
    return value


def _integer(value: Any, *, role: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum or value > (1 << 63) - 1:
        _fail(f"{role}_invalid")
    return value


def _digest(document: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(document))).hexdigest()


def _seal(document: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    sealed = dict(document)
    sealed[field] = _digest(sealed)
    return sealed


def _validate_seal(document: Mapping[str, Any], *, field: str, role: str) -> None:
    observed = _sha256(document.get(field), role=f"{role}_{field}")
    unsigned = dict(document)
    unsigned.pop(field, None)
    if observed != _digest(unsigned):
        _fail(f"{role}_digest_mismatch")


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(document),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError) as exc:
        raise VerifiedTransitionCampaignError("campaign_document_invalid") from exc


def _attestation_sha256(attestation: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(attestation))).hexdigest()


@dataclass(frozen=True, slots=True)
class TransitionCampaignGroup:
    sequence: int
    group_id: str
    task_id: str
    group_manifest_sha256: str
    group_manifest_attestation_sha256: str
    episode_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        episodes = [_identifier(item, role="campaign_episode_id") for item in self.episode_ids]
        if not episodes or len(set(episodes)) != len(episodes):
            _fail("campaign_group_episode_ids_invalid")
        return {
            "sequence": _integer(self.sequence, role="campaign_group_sequence"),
            "group_id": _identifier(self.group_id, role="campaign_group_id"),
            "task_id": _identifier(self.task_id, role="campaign_task_id"),
            "group_manifest_sha256": _sha256(
                self.group_manifest_sha256, role="campaign_group_manifest"
            ),
            "group_manifest_attestation_sha256": _sha256(
                self.group_manifest_attestation_sha256,
                role="campaign_group_manifest_attestation",
            ),
            "episode_ids": episodes,
        }


def campaign_group_from_manifest(
    sequence: int,
    group_manifest: Mapping[str, Any],
    group_manifest_attestation: Mapping[str, Any],
) -> TransitionCampaignGroup:
    manifest = validate_transition_group_manifest(group_manifest)
    return TransitionCampaignGroup(
        sequence=sequence,
        group_id=cast(str, manifest["group_id"]),
        task_id=cast(str, manifest["task_id"]),
        group_manifest_sha256=cast(str, manifest["manifest_sha256"]),
        group_manifest_attestation_sha256=_attestation_sha256(
            group_manifest_attestation
        ),
        episode_ids=tuple(
            cast(str, entry["episode_id"])
            for entry in cast(list[dict[str, Any]], manifest["entries"])
        ),
    )


def build_transition_campaign_manifest(
    *,
    campaign_id: str,
    groups: Sequence[TransitionCampaignGroup],
    trust_policy_sha256: str,
    planned_at_unix_ns: int,
) -> dict[str, Any]:
    """Bind every optimizer group before any campaign task is disclosed."""

    if not groups or len(groups) > _MAX_GROUPS:
        _fail("campaign_manifest_group_count_invalid")
    normalized = [group.to_dict() for group in groups]
    if [group["sequence"] for group in normalized] != list(range(len(normalized))):
        _fail("campaign_manifest_sequence_invalid")
    group_ids = [cast(str, group["group_id"]) for group in normalized]
    if len(set(group_ids)) != len(group_ids):
        _fail("campaign_manifest_duplicate_group")
    episode_ids = [
        cast(str, episode)
        for group in normalized
        for episode in cast(list[str], group["episode_ids"])
    ]
    if len(set(episode_ids)) != len(episode_ids):
        _fail("campaign_manifest_duplicate_episode")
    return _seal(
        {
            "schema": CAMPAIGN_MANIFEST_SCHEMA,
            "campaign_id": _identifier(campaign_id, role="campaign_id"),
            "trust_policy_sha256": _sha256(
                trust_policy_sha256, role="campaign_trust_policy"
            ),
            "group_count": len(normalized),
            "groups": normalized,
            "planned_at_unix_ns": _integer(
                planned_at_unix_ns, role="campaign_planned_at", minimum=1
            ),
        },
        field="manifest_sha256",
    )


def validate_transition_campaign_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "campaign_id",
        "trust_policy_sha256",
        "group_count",
        "groups",
        "planned_at_unix_ns",
        "manifest_sha256",
    }:
        _fail("campaign_manifest_schema_invalid")
    if value.get("schema") != CAMPAIGN_MANIFEST_SCHEMA:
        _fail("campaign_manifest_version_invalid")
    _validate_seal(value, field="manifest_sha256", role="campaign_manifest")
    raw_groups = value.get("groups")
    if not isinstance(raw_groups, list):
        _fail("campaign_manifest_groups_invalid")
    groups: list[TransitionCampaignGroup] = []
    for raw in raw_groups:
        if not isinstance(raw, Mapping) or set(raw) != {
            "sequence",
            "group_id",
            "task_id",
            "group_manifest_sha256",
            "group_manifest_attestation_sha256",
            "episode_ids",
        }:
            _fail("campaign_manifest_group_schema_invalid")
        episodes = raw.get("episode_ids")
        if not isinstance(episodes, list):
            _fail("campaign_manifest_episode_ids_invalid")
        groups.append(
            TransitionCampaignGroup(
                sequence=cast(int, raw.get("sequence")),
                group_id=cast(str, raw.get("group_id")),
                task_id=cast(str, raw.get("task_id")),
                group_manifest_sha256=cast(str, raw.get("group_manifest_sha256")),
                group_manifest_attestation_sha256=cast(
                    str, raw.get("group_manifest_attestation_sha256")
                ),
                episode_ids=tuple(cast(str, item) for item in episodes),
            )
        )
    expected = build_transition_campaign_manifest(
        campaign_id=cast(str, value.get("campaign_id")),
        groups=groups,
        trust_policy_sha256=cast(str, value.get("trust_policy_sha256")),
        planned_at_unix_ns=cast(int, value.get("planned_at_unix_ns")),
    )
    if expected != dict(value) or value.get("group_count") != len(groups):
        _fail("campaign_manifest_reconstruction_mismatch")
    return dict(value)


@dataclass(frozen=True, slots=True)
class VerifiedTransitionCampaignLedger:
    """Create-once campaign event inventory with externally witnessed closure."""

    root: Path
    gateway: FileWriteGateway

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        campaign_manifest: Mapping[str, Any],
        campaign_manifest_attestation: Mapping[str, Any],
        policy: VerifiedCampaignTrustPolicy,
        gateway: FileWriteGateway | None = None,
    ) -> VerifiedTransitionCampaignLedger:
        manifest = validate_transition_campaign_manifest(campaign_manifest)
        if manifest["trust_policy_sha256"] != policy.policy_sha256:
            _fail("campaign_manifest_policy_mismatch")
        planned_at_ns = cast(int, manifest["planned_at_unix_ns"])
        signed = verify_role_attestation(
            policy,
            campaign_manifest_attestation,
            role=TASK_ISSUER,
            expected_payload=manifest,
            not_after_unix=planned_at_ns // 1_000_000_000,
        )
        if signed.get("signed_at_unix") != planned_at_ns // 1_000_000_000:
            _fail("campaign_manifest_signature_time_mismatch")
        resolved_gateway = gateway or FileWriteGateway()
        path = Path(
            resolved_gateway.ensure_directory(
                Path(root), source="verified_transition_campaign.ledger"
            )
        )
        ledger = cls(root=path, gateway=resolved_gateway)
        opened = _seal(
            {
                "schema": CAMPAIGN_OPEN_SCHEMA,
                "campaign_manifest": manifest,
                "campaign_manifest_attestation": dict(campaign_manifest_attestation),
                "opened_at_unix_ns": planned_at_ns,
            },
            field="receipt_sha256",
        )
        if not resolved_gateway.write_bytes_if_absent(
            path / "campaign.open.json",
            _json_bytes(opened),
            source="verified_transition_campaign.open",
            durable=True,
        ):
            _fail("campaign_ledger_already_open")
        return ledger

    @classmethod
    def open(
        cls,
        root: str | Path,
        *,
        gateway: FileWriteGateway | None = None,
    ) -> VerifiedTransitionCampaignLedger:
        return cls(root=Path(root), gateway=gateway or FileWriteGateway())

    def _read(self, name: str) -> dict[str, Any]:
        try:
            payload = read_stable_bytes(self.root / name, max_bytes=16 * 1024 * 1024)
        except (FileNotFoundError, OSError) as exc:
            raise VerifiedTransitionCampaignError(
                f"campaign_ledger_record_missing:{name}"
            ) from exc
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VerifiedTransitionCampaignError("campaign_ledger_json_invalid") from exc
        if not isinstance(document, dict) or _json_bytes(document) != payload:
            _fail("campaign_ledger_record_noncanonical")
        return document

    def _manifest(self) -> dict[str, Any]:
        opened = self._read("campaign.open.json")
        if opened.get("schema") != CAMPAIGN_OPEN_SCHEMA:
            _fail("campaign_open_schema_invalid")
        _validate_seal(opened, field="receipt_sha256", role="campaign_open")
        return validate_transition_campaign_manifest(opened.get("campaign_manifest"))

    @staticmethod
    def _group_name(sequence: int, suffix: str) -> str:
        return f"group-{_integer(sequence, role='campaign_sequence'):08d}.{suffix}.json"

    def start_group(
        self,
        *,
        sequence: int,
        group_manifest: Mapping[str, Any],
        group_manifest_attestation: Mapping[str, Any],
        policy: VerifiedCampaignTrustPolicy,
        started_at_unix_ns: int,
    ) -> dict[str, Any]:
        campaign = self._manifest()
        sequence = _integer(sequence, role="campaign_start_sequence")
        if sequence >= cast(int, campaign["group_count"]):
            _fail("campaign_group_not_planned")
        if sequence > 0:
            self._read(self._group_name(sequence - 1, "terminal"))
        plan = cast(list[dict[str, Any]], campaign["groups"])[sequence]
        manifest = validate_transition_group_manifest(group_manifest)
        attestation_sha256 = _attestation_sha256(group_manifest_attestation)
        if (
            plan["group_id"] != manifest["group_id"]
            or plan["task_id"] != manifest["task_id"]
            or plan["group_manifest_sha256"] != manifest["manifest_sha256"]
            or plan["group_manifest_attestation_sha256"] != attestation_sha256
            or plan["episode_ids"]
            != [entry["episode_id"] for entry in manifest["entries"]]
        ):
            _fail("campaign_group_plan_mismatch")
        verify_role_attestation(
            policy,
            group_manifest_attestation,
            role=TASK_ISSUER,
            expected_payload=manifest,
            not_after_unix=cast(int, campaign["planned_at_unix_ns"])
            // 1_000_000_000,
        )
        started_at = _integer(
            started_at_unix_ns, role="campaign_group_started_at", minimum=1
        )
        if started_at < cast(int, campaign["planned_at_unix_ns"]):
            _fail("campaign_group_started_before_plan")
        receipt = _seal(
            {
                "schema": CAMPAIGN_GROUP_START_SCHEMA,
                "campaign_manifest_sha256": campaign["manifest_sha256"],
                "sequence": sequence,
                "group_id": manifest["group_id"],
                "group_manifest": manifest,
                "group_manifest_attestation": dict(group_manifest_attestation),
                "started_at_unix_ns": started_at,
            },
            field="receipt_sha256",
        )
        if not self.gateway.write_bytes_if_absent(
            self.root / self._group_name(sequence, "started"),
            _json_bytes(receipt),
            source="verified_transition_campaign.group_start",
            durable=True,
        ):
            _fail("campaign_group_already_started")
        return receipt

    def finish_group(
        self,
        *,
        sequence: int,
        status: str,
        group_admission_sha256: str | None,
        update_receipt_sha256: str | None,
        terminal_reason: str,
        finished_at_unix_ns: int,
    ) -> dict[str, Any]:
        campaign = self._manifest()
        sequence = _integer(sequence, role="campaign_finish_sequence")
        start = self._read(self._group_name(sequence, "started"))
        if start.get("schema") != CAMPAIGN_GROUP_START_SCHEMA:
            _fail("campaign_group_start_schema_invalid")
        _validate_seal(start, field="receipt_sha256", role="campaign_group_start")
        if status not in {"updated", "rejected", "aborted", "indeterminate"}:
            _fail("campaign_group_terminal_status_invalid")
        admission = (
            _sha256(group_admission_sha256, role="campaign_group_admission")
            if group_admission_sha256 is not None
            else None
        )
        update = (
            _sha256(update_receipt_sha256, role="campaign_group_update")
            if update_receipt_sha256 is not None
            else None
        )
        if status == "updated" and (admission is None or update is None):
            _fail("campaign_group_updated_evidence_missing")
        if status != "updated" and update is not None:
            _fail("campaign_group_nonupdated_has_update")
        finished_at = _integer(
            finished_at_unix_ns, role="campaign_group_finished_at", minimum=1
        )
        if finished_at < cast(int, start["started_at_unix_ns"]):
            _fail("campaign_group_time_reversed")
        receipt = _seal(
            {
                "schema": CAMPAIGN_GROUP_TERMINAL_SCHEMA,
                "campaign_manifest_sha256": campaign["manifest_sha256"],
                "sequence": sequence,
                "group_id": start["group_id"],
                "group_start_sha256": start["receipt_sha256"],
                "status": status,
                "group_admission_sha256": admission,
                "update_receipt_sha256": update,
                "terminal_reason": _identifier(
                    terminal_reason, role="campaign_group_terminal_reason"
                ),
                "finished_at_unix_ns": finished_at,
            },
            field="receipt_sha256",
        )
        if not self.gateway.write_bytes_if_absent(
            self.root / self._group_name(sequence, "terminal"),
            _json_bytes(receipt),
            source="verified_transition_campaign.group_terminal",
            durable=True,
        ):
            _fail("campaign_group_already_terminal")
        return receipt

    def validate_started_group(
        self,
        *,
        sequence: int,
        group_manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        campaign = self._manifest()
        sequence = _integer(sequence, role="campaign_started_sequence")
        if sequence >= cast(int, campaign["group_count"]):
            _fail("campaign_group_not_planned")
        start = self._read(self._group_name(sequence, "started"))
        if start.get("schema") != CAMPAIGN_GROUP_START_SCHEMA:
            _fail("campaign_group_start_schema_invalid")
        _validate_seal(start, field="receipt_sha256", role="campaign_group_start")
        manifest = validate_transition_group_manifest(group_manifest)
        plan = cast(list[dict[str, Any]], campaign["groups"])[sequence]
        if (
            start.get("campaign_manifest_sha256") != campaign["manifest_sha256"]
            or start.get("sequence") != sequence
            or start.get("group_id") != plan["group_id"]
            or start.get("group_manifest") != manifest
            or manifest["manifest_sha256"] != plan["group_manifest_sha256"]
        ):
            _fail("campaign_group_start_reconstruction_mismatch")
        if self.exists_group_terminal(sequence):
            _fail("campaign_group_already_terminal")
        return start

    def exists_group_terminal(self, sequence: int) -> bool:
        path = self.root / self._group_name(sequence, "terminal")
        if path.is_symlink():
            _fail("campaign_ledger_symlink_rejected")
        return path.is_file()

    def group_records(
        self,
        *,
        sequence: int,
        policy: VerifiedCampaignTrustPolicy,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return one independently checked start/terminal pair."""

        receipt = self.validate_closed(policy=policy)
        close_payload = cast(dict[str, Any], receipt["close_payload"])
        sequence = _integer(sequence, role="campaign_records_sequence")
        if sequence >= cast(int, close_payload["group_count"]):
            _fail("campaign_group_not_planned")
        start = self._read(self._group_name(sequence, "started"))
        terminal = self._read(self._group_name(sequence, "terminal"))
        if (
            start.get("receipt_sha256")
            != close_payload["group_start_sha256s"][sequence]
            or terminal.get("receipt_sha256")
            != close_payload["group_terminal_sha256s"][sequence]
        ):
            _fail("campaign_group_close_binding_mismatch")
        return start, terminal

    def close_payload(
        self,
        *,
        completed_at_unix_ns: int,
        policy: VerifiedCampaignTrustPolicy,
    ) -> dict[str, Any]:
        campaign = self._manifest()
        if campaign["trust_policy_sha256"] != policy.policy_sha256:
            _fail("campaign_manifest_policy_mismatch")
        starts: list[str] = []
        terminals: list[str] = []
        statuses: list[str] = []
        latest = cast(int, campaign["planned_at_unix_ns"])
        for sequence in range(cast(int, campaign["group_count"])):
            start = self._read(self._group_name(sequence, "started"))
            terminal = self._read(self._group_name(sequence, "terminal"))
            plan = cast(list[dict[str, Any]], campaign["groups"])[sequence]
            start_manifest = validate_transition_group_manifest(
                start.get("group_manifest")
            )
            start_attestation = cast(
                Mapping[str, Any], start.get("group_manifest_attestation")
            )
            if (
                start.get("schema") != CAMPAIGN_GROUP_START_SCHEMA
                or terminal.get("schema") != CAMPAIGN_GROUP_TERMINAL_SCHEMA
                or start.get("sequence") != sequence
                or terminal.get("sequence") != sequence
                or start.get("campaign_manifest_sha256")
                != campaign["manifest_sha256"]
                or terminal.get("campaign_manifest_sha256")
                != campaign["manifest_sha256"]
                or start.get("group_id") != plan["group_id"]
                or terminal.get("group_id") != plan["group_id"]
                or start_manifest["manifest_sha256"]
                != plan["group_manifest_sha256"]
                or _attestation_sha256(start_attestation)
                != plan["group_manifest_attestation_sha256"]
                or [entry["episode_id"] for entry in start_manifest["entries"]]
                != plan["episode_ids"]
                or terminal.get("group_start_sha256") != start.get("receipt_sha256")
            ):
                _fail("campaign_group_chain_invalid")
            _validate_seal(start, field="receipt_sha256", role="campaign_group_start")
            _validate_seal(
                terminal, field="receipt_sha256", role="campaign_group_terminal"
            )
            verify_role_attestation(
                policy,
                start_attestation,
                role=TASK_ISSUER,
                expected_payload=start_manifest,
                not_after_unix=cast(int, campaign["planned_at_unix_ns"])
                // 1_000_000_000,
            )
            status = terminal.get("status")
            admission = terminal.get("group_admission_sha256")
            update = terminal.get("update_receipt_sha256")
            if (
                status not in {"updated", "rejected", "aborted", "indeterminate"}
                or (status == "updated" and (admission is None or update is None))
                or (status != "updated" and update is not None)
                or (
                    admission is not None
                    and _sha256(admission, role="campaign_terminal_admission")
                    != admission
                )
                or (
                    update is not None
                    and _sha256(update, role="campaign_terminal_update") != update
                )
                or _integer(
                    terminal.get("finished_at_unix_ns"),
                    role="campaign_terminal_finished_at",
                    minimum=1,
                )
                < _integer(
                    start.get("started_at_unix_ns"),
                    role="campaign_start_started_at",
                    minimum=1,
                )
            ):
                _fail("campaign_group_terminal_invalid")
            starts.append(cast(str, start["receipt_sha256"]))
            terminals.append(cast(str, terminal["receipt_sha256"]))
            statuses.append(cast(str, status))
            latest = max(latest, cast(int, terminal["finished_at_unix_ns"]))
        completed_at = _integer(
            completed_at_unix_ns, role="campaign_completed_at", minimum=1
        )
        if completed_at < latest:
            _fail("campaign_close_time_reversed")
        return _seal(
            {
                "schema": CAMPAIGN_CLOSE_PAYLOAD_SCHEMA,
                "campaign_id": campaign["campaign_id"],
                "campaign_manifest_sha256": campaign["manifest_sha256"],
                "group_count": campaign["group_count"],
                "group_start_sha256s": starts,
                "group_terminal_sha256s": terminals,
                "group_statuses": statuses,
                "updated_count": statuses.count("updated"),
                "rejected_count": statuses.count("rejected"),
                "aborted_count": statuses.count("aborted"),
                "indeterminate_count": statuses.count("indeterminate"),
                "completed_at_unix_ns": completed_at,
            },
            field="payload_sha256",
        )

    def close(
        self,
        *,
        close_payload: Mapping[str, Any],
        evidence_verifier_attestation: Mapping[str, Any],
        policy: VerifiedCampaignTrustPolicy,
    ) -> dict[str, Any]:
        expected = self.close_payload(
            completed_at_unix_ns=_integer(
                close_payload.get("completed_at_unix_ns"),
                role="campaign_close_completed_at",
                minimum=1,
            ),
            policy=policy,
        )
        if expected != dict(close_payload):
            _fail("campaign_close_payload_mismatch")
        verify_role_attestation(
            policy,
            evidence_verifier_attestation,
            role=EVIDENCE_VERIFIER,
            expected_payload=expected,
            not_before_unix=(
                cast(int, expected["completed_at_unix_ns"]) + 999_999_999
            )
            // 1_000_000_000,
        )
        receipt = _seal(
            {
                "schema": CAMPAIGN_RECEIPT_SCHEMA,
                "close_payload": expected,
                "evidence_verifier_attestation": dict(evidence_verifier_attestation),
            },
            field="receipt_sha256",
        )
        if not self.gateway.write_bytes_if_absent(
            self.root / "campaign.closed.json",
            _json_bytes(receipt),
            source="verified_transition_campaign.close",
            durable=True,
        ):
            _fail("campaign_already_closed")
        return receipt

    def validate_closed(
        self, *, policy: VerifiedCampaignTrustPolicy
    ) -> dict[str, Any]:
        opened = self._read("campaign.open.json")
        campaign = validate_transition_campaign_manifest(opened.get("campaign_manifest"))
        verify_role_attestation(
            policy,
            cast(Mapping[str, Any], opened.get("campaign_manifest_attestation")),
            role=TASK_ISSUER,
            expected_payload=campaign,
            not_after_unix=cast(int, campaign["planned_at_unix_ns"])
            // 1_000_000_000,
        )
        receipt = self._read("campaign.closed.json")
        if receipt.get("schema") != CAMPAIGN_RECEIPT_SCHEMA:
            _fail("campaign_receipt_schema_invalid")
        _validate_seal(receipt, field="receipt_sha256", role="campaign_receipt")
        close_payload = cast(Mapping[str, Any], receipt.get("close_payload"))
        expected = self.close_payload(
            completed_at_unix_ns=_integer(
                close_payload.get("completed_at_unix_ns"),
                role="campaign_close_completed_at",
                minimum=1,
            ),
            policy=policy,
        )
        if expected != dict(close_payload):
            _fail("campaign_close_reconstruction_mismatch")
        verify_role_attestation(
            policy,
            cast(Mapping[str, Any], receipt.get("evidence_verifier_attestation")),
            role=EVIDENCE_VERIFIER,
            expected_payload=expected,
            not_before_unix=(
                cast(int, expected["completed_at_unix_ns"]) + 999_999_999
            )
            // 1_000_000_000,
        )
        return receipt


__all__ = [
    "CAMPAIGN_CLOSE_PAYLOAD_SCHEMA",
    "CAMPAIGN_GROUP_START_SCHEMA",
    "CAMPAIGN_GROUP_TERMINAL_SCHEMA",
    "CAMPAIGN_MANIFEST_SCHEMA",
    "CAMPAIGN_OPEN_SCHEMA",
    "CAMPAIGN_RECEIPT_SCHEMA",
    "TransitionCampaignGroup",
    "VerifiedTransitionCampaignError",
    "VerifiedTransitionCampaignLedger",
    "build_transition_campaign_manifest",
    "campaign_group_from_manifest",
    "validate_transition_campaign_manifest",
]
