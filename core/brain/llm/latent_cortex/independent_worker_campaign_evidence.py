"""Independent replay of detached worker evidence for paired campaigns.

This module reconstructs the campaign's worker-execution manifest from raw
broker, detached-supervisor, staged-journal, lifecycle, and import artifacts.
It never accepts a producer summary without deriving the same value itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Never

from core.brain.llm.latent_cortex.campaign_journal import (
    ARM_RESULT,
    STARTED,
    CampaignJournal,
    CampaignJournalError,
    CampaignPlan,
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.campaign_trust import VerifiedCampaignTrustPolicy
from core.brain.llm.latent_cortex.detached_campaign_evidence import (
    DetachedCampaignEvidenceError,
    verify_detached_broker_evidence,
)
from core.brain.llm.latent_cortex.paired_campaign import FULL_ARMS
from core.brain.llm.latent_cortex.worker_attempt_import import (
    IMPORT_INTENT_SCHEMA,
    IMPORT_RECEIPT_SCHEMA,
    WorkerAttemptImportError,
    verify_terminal_worker_stage,
)
from core.runtime.detached_subprocess_broker import BrokeredProcessResult
from core.runtime.file_read_gateway import read_stable_bytes

WORKER_EXECUTION_MANIFEST_SCHEMA = "aura.latent_cortex.worker_execution_manifest.v1"
BROKER_RESULT_SCHEMA = "aura.latent_cortex.brokered_worker_result.v1"
WORKER_ATTEMPT_DIR = "worker_attempts"
WORKER_EXECUTION_MANIFEST_FILE = "worker_execution_manifest.json"
JOURNAL_FILE = "campaign.jsonl"

_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_JOURNAL_BYTES = 1024 * 1024 * 1024


class IndependentWorkerCampaignEvidenceError(ValueError):
    """Stable fail-closed error for independently invalid worker evidence."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise IndependentWorkerCampaignEvidenceError(code)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: Any) -> str:
    try:
        return _sha256_bytes(canonical_json_bytes(value))
    except (TypeError, ValueError, OverflowError):
        _fail("worker_evidence_value_not_canonical")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json(raw: bytes, *, role: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                _fail(f"{role}_duplicate_key")
            value[key] = item
        return value

    def reject_constant(_value: str) -> Never:
        _fail(f"{role}_non_finite_number")

    def parse_float(raw_value: str) -> float:
        value = float(raw_value)
        if not math.isfinite(value):
            _fail(f"{role}_non_finite_number")
        return value

    try:
        value = json.loads(
            raw,
            object_pairs_hook=object_pairs,
            parse_float=parse_float,
            parse_constant=reject_constant,
        )
    except IndependentWorkerCampaignEvidenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, OverflowError):
        _fail(f"{role}_json_invalid")
    if not isinstance(value, dict):
        _fail(f"{role}_not_object")
    return value


def _read_canonical_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        observed = path.lstat()
    except OSError:
        _fail(f"{role}_unavailable")
    if (
        path.is_symlink()
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or observed.st_nlink != 1
        or observed.st_mode & 0o022
    ):
        _fail(f"{role}_storage_invalid")
    try:
        raw = read_stable_bytes(path, max_bytes=_MAX_JSON_BYTES)
    except (OSError, ValueError):
        _fail(f"{role}_unavailable")
    value = _strict_json(raw, role=role)
    if raw != canonical_json_bytes(value) + b"\n":
        _fail(f"{role}_noncanonical")
    return value


def _verified_hash_object(
    value: Any,
    *,
    hash_key: str,
    role: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{role}_invalid")
    body = {key: item for key, item in value.items() if key != hash_key}
    if not _is_sha256(value.get(hash_key)) or value.get(hash_key) != _sha256(body):
        _fail(f"{role}_hash_invalid")
    return value


def _worker_paths(campaign_dir: Path, arm: str, slot: int) -> dict[str, Path]:
    root = campaign_dir / WORKER_ATTEMPT_DIR / f"{arm}.attempt-{slot:02d}"
    return {
        "root": root,
        "stage": root / "stage.jsonl",
        "origin_dir": root / "supervisor-origin",
        "broker_result": root / "broker-result.json",
        "verified_stage": root / "verified-stage.json",
        "import_intent": root / "import-intent.json",
        "import_receipt": root / "import-receipt.json",
    }


def _broker_result(path: Path) -> tuple[BrokeredProcessResult, dict[str, Any]]:
    artifact = _read_canonical_json(path, role="worker_broker_result")
    fields = set(BrokeredProcessResult.__dataclass_fields__)
    body = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    if (
        set(artifact) != {"schema", *fields, "artifact_sha256"}
        or artifact.get("schema") != BROKER_RESULT_SCHEMA
        or artifact.get("artifact_sha256") != _sha256(body)
    ):
        _fail("worker_broker_result_invalid")
    try:
        result = BrokeredProcessResult(**{field: artifact[field] for field in fields})
    except (TypeError, ValueError):
        _fail("worker_broker_result_fields_invalid")
    return result, artifact


def _canonical_snapshot(
    path: Path,
    plan: CampaignPlan,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], bytes]:
    try:
        before = read_stable_bytes(path, max_bytes=_MAX_JOURNAL_BYTES)
        with CampaignJournal(path, plan) as journal:
            result_records = journal.result_records()
        after = read_stable_bytes(path, max_bytes=_MAX_JOURNAL_BYTES)
    except (OSError, ValueError, CampaignJournalError):
        _fail("canonical_campaign_journal_invalid")
    if before != after or not before.endswith(b"\n"):
        _fail("canonical_campaign_journal_changed")
    events: list[dict[str, Any]] = []
    for line in before.splitlines():
        event = _strict_json(line, role="canonical_campaign_event")
        if line != canonical_json_bytes(event):
            _fail("canonical_campaign_event_noncanonical")
        events.append(event)
    if not events:
        _fail("canonical_campaign_journal_empty")
    return tuple(events), result_records, before


def _verify_import_boundary(
    *,
    paths: dict[str, Path],
    plan: CampaignPlan,
    arm: str,
    slot: int,
    stage_manifest: dict[str, Any],
    stage_records: tuple[dict[str, Any], ...],
    canonical_records: dict[str, dict[str, Any]],
    canonical_events: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    intent = _verified_hash_object(
        _read_canonical_json(paths["import_intent"], role="worker_import_intent"),
        hash_key="intent_sha256",
        role="worker_import_intent",
    )
    receipt = _verified_hash_object(
        _read_canonical_json(paths["import_receipt"], role="worker_import_receipt"),
        hash_key="receipt_sha256",
        role="worker_import_receipt",
    )
    if (
        set(intent)
        != {
            "schema",
            "campaign_plan_sha256",
            "arm",
            "worker_attempt_slot",
            "stage_manifest_sha256",
            "prior_canonical_head_sha256",
            "cell_ids",
            "intent_sha256",
        }
        or intent.get("schema") != IMPORT_INTENT_SCHEMA
        or intent.get("campaign_plan_sha256") != plan.plan_sha256
        or intent.get("arm") != arm
        or intent.get("worker_attempt_slot") != slot
        or intent.get("stage_manifest_sha256") != stage_manifest["manifest_sha256"]
        or intent.get("cell_ids") != stage_manifest["cell_ids"]
    ):
        _fail("worker_import_intent_binding_invalid")

    event_indexes = {
        event.get("event_sha256"): index
        for index, event in enumerate(canonical_events)
        if isinstance(event.get("event_sha256"), str)
    }
    expected_imported: list[dict[str, Any]] = []
    first_started_index: int | None = None
    last_result_index: int | None = None
    for record in stage_records:
        canonical = canonical_records.get(record["cell_id"])
        if (
            canonical is None
            or canonical.get("attempt_id") != record.get("attempt_id")
            or canonical.get("result") != record.get("result")
        ):
            _fail("worker_import_canonical_result_differs")
        result_sha = canonical.get("arm_result_event_sha256")
        result_index = event_indexes.get(result_sha)
        if not isinstance(result_index, int) or result_index <= 0:
            _fail("worker_import_result_event_missing")
        started_index = result_index - 1
        started = canonical_events[started_index]
        result_event = canonical_events[result_index]
        if (
            started.get("event") != STARTED
            or result_event.get("event") != ARM_RESULT
            or started.get("cell_id") != record["cell_id"]
            or result_event.get("cell_id") != record["cell_id"]
            or started.get("attempt_id") != record["attempt_id"]
            or result_event.get("attempt_id") != record["attempt_id"]
            or result_event.get("payload") != {"result": record["result"]}
            or (last_result_index is not None and started_index != last_result_index + 1)
        ):
            _fail("worker_import_events_not_atomic_or_contiguous")
        if first_started_index is None:
            first_started_index = started_index
        last_result_index = result_index
        expected_imported.append(
            {
                "cell_id": record["cell_id"],
                "attempt_id": record["attempt_id"],
                "arm_result_event_sha256": result_sha,
                "result_origin_sha256": record["result"]["worker_origin"][
                    "origin_sha256"
                ],
                "stage_verification_sha256": _sha256(record["verification"]),
                "stage_commit_sha256": _sha256(record["commit"]),
            }
        )
    if first_started_index is None or last_result_index is None:
        _fail("worker_import_empty")
    if intent.get("prior_canonical_head_sha256") != canonical_events[
        first_started_index
    ].get("previous_event_sha256"):
        _fail("worker_import_prior_head_invalid")
    expected_receipt = {
        "schema": IMPORT_RECEIPT_SCHEMA,
        "campaign_plan_sha256": plan.plan_sha256,
        "arm": arm,
        "worker_attempt_slot": slot,
        "stage_manifest_sha256": stage_manifest["manifest_sha256"],
        "import_intent_sha256": intent["intent_sha256"],
        "imported": expected_imported,
        "final_canonical_head_sha256": canonical_events[last_result_index][
            "event_sha256"
        ],
    }
    expected_receipt["receipt_sha256"] = _sha256(expected_receipt)
    if receipt != expected_receipt:
        _fail("worker_import_receipt_binding_invalid")
    return intent, receipt, expected_imported


@dataclass(frozen=True)
class VerifiedWorkerCampaignEvidence:
    manifest_sha256: str
    detached_plan_sha256: str
    detached_plan_artifact_sha256: str
    detached_journal_head_sha256: str
    detached_attempts_artifact_sha256: str
    detached_classification_head_sha256: str
    detached_classifications_sha256: str
    imports_sha256: str
    excluded_attempts_sha256: str
    imported_attempt_count: int
    excluded_attempt_count: int


def _worker_look_for_slot(
    slot: int,
    *,
    attempt_slots: int,
    sequential_looks: list[Any] | None,
) -> int:
    if sequential_looks is None:
        return 0
    slots_per_look = attempt_slots // len(sequential_looks)
    return (slot - 1) // slots_per_look + 1


def _expected_worker_batches(
    arms: list[str],
    sequential_looks: list[Any] | None,
) -> set[tuple[int, str]]:
    look_values = (
        range(1, len(sequential_looks) + 1)
        if sequential_looks is not None
        else (0,)
    )
    return {(worker_look, arm) for worker_look in look_values for arm in arms}


def verify_worker_campaign_evidence(
    *,
    campaign_dir: Path,
    plan: CampaignPlan,
    policy: VerifiedCampaignTrustPolicy,
    expected_protocol_sha256: str,
) -> VerifiedWorkerCampaignEvidence:
    """Rebuild and verify all claim worker evidence from immutable artifacts."""

    metadata = plan.to_dict().get("metadata")
    execution = metadata.get("execution_config") if isinstance(metadata, dict) else None
    arms = metadata.get("arms") if isinstance(metadata, dict) else None
    attempt_slots = (
        execution.get("worker_origin_attempt_slots")
        if isinstance(execution, dict)
        else None
    )
    sequential_looks = (
        execution.get("sequential_look_observations_per_domain")
        if isinstance(execution, dict)
        else None
    )
    if (
        not isinstance(metadata, dict)
        or metadata.get("claim_eligible") is not True
        or not isinstance(execution, dict)
        or execution.get("worker_origin_protocol")
        != "detached_supervisor_staged_arm_import_v3"
        or not isinstance(arms, list)
        or not arms
        or len(set(arms)) != len(arms)
        or any(arm not in FULL_ARMS for arm in arms)
        or not isinstance(attempt_slots, int)
        or isinstance(attempt_slots, bool)
        or attempt_slots <= 0
        or attempt_slots > 64
        or (
            sequential_looks is not None
            and (
                not isinstance(sequential_looks, list)
                or not sequential_looks
                or attempt_slots % len(sequential_looks) != 0
            )
        )
        or not _is_sha256(expected_protocol_sha256)
    ):
        _fail("worker_evidence_plan_contract_invalid")

    manifest = _verified_hash_object(
        _read_canonical_json(
            campaign_dir / WORKER_EXECUTION_MANIFEST_FILE,
            role="worker_execution_manifest",
        ),
        hash_key="manifest_sha256",
        role="worker_execution_manifest",
    )
    expected_manifest_keys = {
        "schema",
        "campaign_name",
        "policy_sha256",
        "protocol_sha256",
        "plan_sha256",
        "detached_run_dir",
        "detached_plan_path",
        "detached_attempts_path",
        "detached_plan_artifact_sha256",
        "detached_plan_sha256",
        "detached_classification_head_sha256",
        "detached_classifications",
        "detached_classifications_sha256",
        "import_count",
        "imports",
        "imports_sha256",
        "excluded_count",
        "excluded_attempts",
        "excluded_attempts_sha256",
        "manifest_sha256",
    }
    if (
        set(manifest) != expected_manifest_keys
        or manifest.get("schema") != WORKER_EXECUTION_MANIFEST_SCHEMA
        or manifest.get("campaign_name") != plan.campaign_name
        or manifest.get("policy_sha256") != policy.policy_sha256
        or manifest.get("protocol_sha256") != expected_protocol_sha256
        or manifest.get("plan_sha256") != plan.plan_sha256
    ):
        _fail("worker_execution_manifest_binding_invalid")

    try:
        detached_run_dir = Path(manifest["detached_run_dir"]).resolve(strict=True)
        detached_plan_path = Path(manifest["detached_plan_path"]).resolve(strict=True)
        detached_attempts_path = Path(manifest["detached_attempts_path"]).resolve(
            strict=True
        )
    except (KeyError, OSError, TypeError):
        _fail("worker_detached_evidence_path_invalid")
    if (
        manifest.get("detached_run_dir") != str(detached_run_dir)
        or manifest.get("detached_plan_path") != str(detached_plan_path)
        or manifest.get("detached_attempts_path") != str(detached_attempts_path)
        or detached_plan_path != detached_run_dir / "detached_plan.json"
        or detached_attempts_path != detached_run_dir / "detached_attempts.jsonl"
    ):
        _fail("worker_detached_evidence_path_substitution")
    try:
        detached_plan_raw = read_stable_bytes(
            detached_plan_path, max_bytes=_MAX_JSON_BYTES
        )
        detached_attempts_raw = read_stable_bytes(
            detached_attempts_path, max_bytes=_MAX_JOURNAL_BYTES
        )
    except (OSError, ValueError):
        _fail("worker_detached_evidence_unavailable")
    if manifest.get("detached_plan_artifact_sha256") != _sha256_bytes(
        detached_plan_raw
    ):
        _fail("worker_detached_plan_artifact_differs")

    canonical_path = campaign_dir / JOURNAL_FILE
    canonical_events, result_records, canonical_raw = _canonical_snapshot(
        canonical_path,
        plan,
    )
    canonical_by_cell = {record["cell_id"]: record for record in result_records}

    attempts_root = campaign_dir / WORKER_ATTEMPT_DIR
    if not attempts_root.is_dir() or attempts_root.is_symlink():
        _fail("worker_attempt_root_invalid")
    root_stat = attempts_root.lstat()
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or root_stat.st_uid != os.geteuid()
        or stat.S_IMODE(root_stat.st_mode) & 0o077
    ):
        _fail("worker_attempt_root_not_private")
    expected_root_names = {
        f"{arm}.attempt-{slot:02d}"
        for arm in arms
        for slot in range(1, attempt_slots + 1)
    }
    observed_root_names = {entry.name for entry in attempts_root.iterdir()}
    if not observed_root_names.issubset(expected_root_names):
        _fail("worker_attempt_root_contains_unplanned_entry")

    imports: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    successful_batches: set[tuple[int, str]] = set()
    detached_plan_sha256: str | None = None
    classification_head_sha256: str | None = None
    detached_terminals: list[dict[str, Any]] | None = None
    detached_quarantines: list[dict[str, Any]] | None = None
    detached_journal_head_sha256: str | None = None
    classified_request_ids: set[str] = set()

    model_identity_sha256 = _sha256(metadata["model_identity"])
    adapter_identity_sha256 = _sha256(metadata["adapter_identity"])

    for arm in arms:
        for slot in range(1, attempt_slots + 1):
            paths = _worker_paths(campaign_dir, arm, slot)
            if not paths["root"].exists():
                continue
            root_observed = paths["root"].lstat()
            if (
                paths["root"].is_symlink()
                or not stat.S_ISDIR(root_observed.st_mode)
                or root_observed.st_uid != os.geteuid()
                or stat.S_IMODE(root_observed.st_mode) & 0o077
            ):
                _fail("worker_attempt_directory_not_private")
            allowed_entries = {
                path.name for name, path in paths.items() if name != "root"
            }
            allowed_entries.add(f"{paths['stage'].name}.lock")
            if not {entry.name for entry in paths["root"].iterdir()}.issubset(
                allowed_entries
            ):
                _fail("worker_attempt_contains_unplanned_entry")
            if not paths["broker_result"].exists():
                if any(paths["root"].iterdir()):
                    _fail("worker_attempt_activity_without_broker_result")
                continue

            broker_result, broker_artifact = _broker_result(paths["broker_result"])
            try:
                detached = verify_detached_broker_evidence(
                    run_dir=detached_run_dir,
                    broker_result=broker_result,
                    require_claim_eligible=False,
                )
            except DetachedCampaignEvidenceError as exc:
                raise IndependentWorkerCampaignEvidenceError(
                    f"worker_detached_replay_{exc.code}"
                ) from exc
            current_terminals = [asdict(item) for item in detached.terminal_summaries]
            current_quarantines = [
                asdict(item) for item in detached.quarantine_summaries
            ]
            snapshot = (
                detached.plan["plan_sha256"],
                detached.classification_head_sha256,
                current_terminals,
                current_quarantines,
                detached.journal_head_sha256,
            )
            if detached_plan_sha256 is None:
                (
                    detached_plan_sha256,
                    classification_head_sha256,
                    detached_terminals,
                    detached_quarantines,
                    detached_journal_head_sha256,
                ) = snapshot
            elif snapshot != (
                detached_plan_sha256,
                classification_head_sha256,
                detached_terminals,
                detached_quarantines,
                detached_journal_head_sha256,
            ):
                _fail("worker_attempts_span_detached_snapshots")

            matching = [
                item
                for item in current_terminals
                if item["request_id"] == broker_result.request_id
            ]
            if len(matching) != 1 or broker_result.request_id in classified_request_ids:
                _fail("worker_broker_terminal_not_unique")
            classified_request_ids.add(broker_result.request_id)
            terminal = matching[0]
            passed = broker_result.returncode == 0 and broker_result.status == "passed"
            if terminal["claim_eligible"] is not passed:
                _fail("worker_broker_terminal_eligibility_differs")
            summary = broker_result.worker_origin_lifecycle
            if not isinstance(summary, dict):
                _fail("worker_broker_lifecycle_summary_missing")
            worker_look = _worker_look_for_slot(
                slot,
                attempt_slots=attempt_slots,
                sequential_looks=sequential_looks,
            )
            common = {
                "arm": arm,
                "worker_attempt_slot": slot,
                "worker_look": worker_look,
                "broker_result_artifact_sha256": broker_artifact["artifact_sha256"],
                "broker_policy_sha256": broker_result.policy_sha256,
                "broker_request_id": broker_result.request_id,
                "broker_receipt_sha256": broker_result.receipt_sha256,
                "broker_response_hmac_sha256": broker_result.response_hmac_sha256,
                "worker_origin_lifecycle": summary,
                "detached_supervisor_attempt": detached.attempt,
                "detached_terminal_event_sha256": terminal["event_sha256"],
                "detached_classification_head_sha256": (
                    detached.classification_head_sha256
                ),
            }
            if not passed:
                if any(
                    paths[name].exists()
                    for name in ("verified_stage", "import_intent", "import_receipt")
                ):
                    _fail("excluded_worker_attempt_has_import_artifacts")
                excluded.append(
                    {
                        **common,
                        "classification": "terminal_excluded",
                        "status": broker_result.status,
                        "returncode": broker_result.returncode,
                        "reason": broker_result.error,
                    }
                )
                continue
            batch = (worker_look, arm)
            if batch in successful_batches:
                _fail("worker_batch_imported_more_than_once")
            lifecycle_value = summary.get("artifact_path")
            if not isinstance(lifecycle_value, str):
                _fail("worker_lifecycle_path_missing")
            try:
                lifecycle_path = Path(lifecycle_value).resolve(strict=True)
                expected_origin_dir = paths["origin_dir"].resolve(strict=True)
            except OSError:
                _fail("worker_lifecycle_path_unavailable")
            if lifecycle_path.parent != expected_origin_dir:
                _fail("worker_lifecycle_path_substitution")
            try:
                stage = verify_terminal_worker_stage(
                    stage_path=paths["stage"],
                    lifecycle_path=lifecycle_path,
                    plan=plan,
                    policy=policy,
                    broker_result=broker_result,
                    arm=arm,
                    worker_attempt_slot=slot,
                    expected_protocol_sha256=expected_protocol_sha256,
                    expected_detached_plan_sha256=detached.plan["plan_sha256"],
                    expected_broker_policy_sha256=broker_result.policy_sha256,
                    expected_model_identity_sha256=model_identity_sha256,
                    expected_adapter_identity_sha256=adapter_identity_sha256,
                )
            except WorkerAttemptImportError as exc:
                raise IndependentWorkerCampaignEvidenceError(
                    f"worker_stage_replay_{exc.code}"
                ) from exc
            persisted_stage = _read_canonical_json(
                paths["verified_stage"], role="verified_worker_stage"
            )
            if persisted_stage != stage.manifest:
                _fail("verified_worker_stage_differs_from_replay")
            intent, receipt, imported = _verify_import_boundary(
                paths=paths,
                plan=plan,
                arm=arm,
                slot=slot,
                stage_manifest=stage.manifest,
                stage_records=stage.records,
                canonical_records=canonical_by_cell,
                canonical_events=canonical_events,
            )
            imports.append(
                {
                    **common,
                    "classification": "terminal_imported",
                    "session_id": summary["session_id"],
                    "detached_plan_sha256": stage.manifest["detached_plan_sha256"],
                    "verified_stage_manifest_sha256": stage.manifest[
                        "manifest_sha256"
                    ],
                    "stage_sha256": stage.manifest["stage_sha256"],
                    "stage_journal_head_sha256": stage.manifest[
                        "stage_journal_head_sha256"
                    ],
                    "result_chain_head_sha256": stage.manifest[
                        "result_chain_head_sha256"
                    ],
                    "cell_ids": stage.manifest["cell_ids"],
                    "import_intent_sha256": intent["intent_sha256"],
                    "import_receipt_sha256": receipt["receipt_sha256"],
                    "canonical_imports": imported,
                }
            )
            successful_batches.add(batch)

    expected_batches = _expected_worker_batches(arms, sequential_looks)
    if (
        successful_batches != expected_batches
        or detached_plan_sha256 is None
        or classification_head_sha256 is None
        or detached_terminals is None
        or detached_quarantines is None
        or detached_journal_head_sha256 is None
    ):
        _fail("worker_evidence_arm_coverage_incomplete")
    terminal_request_ids = {item["request_id"] for item in detached_terminals}
    if terminal_request_ids != classified_request_ids:
        _fail("worker_detached_terminal_coverage_differs")

    detached_classifications = {
        "terminal_count": len(detached_terminals),
        "terminals": detached_terminals,
        "quarantine_count": len(detached_quarantines),
        "quarantines": detached_quarantines,
    }
    if (
        manifest.get("detached_plan_sha256") != detached_plan_sha256
        or manifest.get("detached_classification_head_sha256")
        != classification_head_sha256
        or manifest.get("detached_classifications") != detached_classifications
        or manifest.get("detached_classifications_sha256")
        != _sha256(detached_classifications)
        or manifest.get("import_count") != len(imports)
        or manifest.get("imports") != imports
        or manifest.get("imports_sha256") != _sha256(imports)
        or manifest.get("excluded_count") != len(excluded)
        or manifest.get("excluded_attempts") != excluded
        or manifest.get("excluded_attempts_sha256") != _sha256(excluded)
    ):
        _fail("worker_execution_manifest_differs_from_replay")

    try:
        canonical_origins = {
            record["result"]["worker_origin"]["origin_sha256"]
            for record in result_records
        }
        imported_origins = {
            cell["result_origin_sha256"]
            for entry in imports
            for cell in entry["canonical_imports"]
        }
        canonical_sessions = {
            record["result"]["worker_origin"]["signed_payload"]["session_id"]
            for record in result_records
        }
        excluded_sessions = {
            entry["worker_origin_lifecycle"]["session_id"] for entry in excluded
        }
        quarantined_sessions = {
            item["session_id"] for item in detached_quarantines
        }
    except (KeyError, TypeError):
        _fail("worker_canonical_origin_shape_invalid")
    if (
        canonical_origins != imported_origins
        or len(result_records) != len(plan.cell_ids)
        or canonical_sessions & (excluded_sessions | quarantined_sessions)
    ):
        _fail("worker_canonical_origin_set_invalid")

    try:
        final_detached_plan_raw = read_stable_bytes(
            detached_plan_path,
            max_bytes=_MAX_JSON_BYTES,
        )
        final_detached_attempts_raw = read_stable_bytes(
            detached_attempts_path,
            max_bytes=_MAX_JOURNAL_BYTES,
        )
        final_canonical_raw = read_stable_bytes(
            canonical_path,
            max_bytes=_MAX_JOURNAL_BYTES,
        )
    except (OSError, ValueError):
        _fail("worker_evidence_changed_during_replay")
    if (
        final_detached_plan_raw != detached_plan_raw
        or final_detached_attempts_raw != detached_attempts_raw
        or final_canonical_raw != canonical_raw
    ):
        _fail("worker_evidence_changed_during_replay")

    return VerifiedWorkerCampaignEvidence(
        manifest_sha256=manifest["manifest_sha256"],
        detached_plan_sha256=detached_plan_sha256,
        detached_plan_artifact_sha256=_sha256_bytes(detached_plan_raw),
        detached_journal_head_sha256=detached_journal_head_sha256,
        detached_attempts_artifact_sha256=_sha256_bytes(detached_attempts_raw),
        detached_classification_head_sha256=classification_head_sha256,
        detached_classifications_sha256=manifest["detached_classifications_sha256"],
        imports_sha256=manifest["imports_sha256"],
        excluded_attempts_sha256=manifest["excluded_attempts_sha256"],
        imported_attempt_count=len(imports),
        excluded_attempt_count=len(excluded),
    )


__all__ = [
    "IndependentWorkerCampaignEvidenceError",
    "VerifiedWorkerCampaignEvidence",
    "verify_worker_campaign_evidence",
]
