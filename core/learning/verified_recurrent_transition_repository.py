"""Durable production evidence and replay packages for recurrent training."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Never, cast

from core.brain.llm.latent_cortex.campaign_trust import EVIDENCE_VERIFIER
from core.learning.recurrent_grpo import (
    recurrent_policy_sample_from_receipt,
    validate_recurrent_policy_sample_receipt,
)
from core.learning.verified_recurrent_transition_evidence import (
    VerifiedRecurrentTransitionEvidence,
    build_verified_recurrent_transition_evidence,
    validate_verified_recurrent_transition_evidence,
)
from core.learning.verified_transition_episode import (
    TransitionArtifactStore,
    canonical_json_bytes,
)
from core.learning.verified_transition_group_admission import (
    build_verified_transition_group_admission,
    validate_transition_group_manifest,
)
from core.learning.verified_transition_reward import (
    TransitionRewardConfig,
    build_verified_transition_reward_batch,
)
from core.learning.verified_transition_trainer import (
    PreparedVerifiedTransitionGroup,
    VerifiedTransitionCampaignClosure,
)
from core.learning.verified_transition_update import VerifiedTransitionUpdateJournal
from core.runtime.atomic_writer import (
    atomic_write_bytes_if_absent,
    ensure_private_directory,
)
from core.runtime.file_read_gateway import read_stable_bytes

RECURRENT_REPLAY_PACKAGE_SCHEMA = (
    "aura.verified_transition.recurrent_replay_package.v1"
)
PRODUCTION_EVIDENCE_PRODUCER_ID = (
    "aura.verified_transition.recurrent_evidence_producer.v1"
)
DURABLE_REPLAY_LOADER_ID = (
    "aura.verified_transition.recurrent_replay_loader.v1"
)
CAMPAIGN_FINALIZER_ID = (
    "aura.verified_transition.recurrent_campaign_finalizer.v1"
)
INDEPENDENT_SCORER_ID = (
    "aura.verified_transition.recurrent_programmatic_scorer.v1"
)
TOKEN_CODEC_ID = "aura.verified_transition.recurrent_trace_codec.v1"

_PACKAGE_KEYS = frozenset(
    {
        "schema",
        "contract_sha256",
        "campaign_schedule_root_sha256",
        "sequence",
        "task_id",
        "tokenizer_bundle_sha256",
        "prompt_text",
        "prompt_tokens",
        "prompt_tokens_sha256",
        "sample_receipts_json",
        "sample_receipt_sha256s",
        "evidence_artifacts",
        "evidence_receipt_sha256s",
        "reward_artifact",
        "reward_receipt_sha256",
        "group_admission_artifact",
        "group_admission_sha256",
        "group_manifest",
        "group_manifest_attestation",
        "created_at_unix_ns",
        "receipt_sha256",
    }
)


class VerifiedRecurrentTransitionRepositoryError(RuntimeError):
    """Stable durable-package production or replay failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise VerifiedRecurrentTransitionRepositoryError(code)


def _sha256(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{role}_invalid")
    return value


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
        raise VerifiedRecurrentTransitionRepositoryError(
            "recurrent_replay_document_invalid"
        ) from exc


def _float_json(value: Mapping[str, Any]) -> str:
    return _json_bytes(value).decode("ascii")


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _tokens_sha256(tokens: Sequence[int]) -> str:
    return hashlib.sha256(
        json.dumps(list(tokens), separators=(",", ":"), allow_nan=False).encode(
            "ascii"
        )
    ).hexdigest()


def score_verified_recurrent_training_task(
    task: Any,
    response: str,
) -> dict[str, Any]:
    """Normalize one programmatic task verdict into the strict evidence schema."""

    if not isinstance(response, str):
        _fail("recurrent_scorer_response_invalid")
    grader = getattr(task, "grade", None)
    if not callable(grader):
        _fail("recurrent_scorer_task_grader_missing")
    verdict = grader(response)
    if not isinstance(verdict, Mapping) or type(verdict.get("correct")) is not bool:
        _fail("recurrent_scorer_verdict_invalid")
    parsed_value = verdict.get("parsed")
    parsed = parsed_value is not None
    reason = verdict.get("reason")
    if not isinstance(reason, str) or not reason:
        reason = "correct" if verdict["correct"] else "incorrect"
    normalized_material: Any = parsed_value if parsed else response
    try:
        normalized = json.dumps(
            normalized_material,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError) as exc:
        raise VerifiedRecurrentTransitionRepositoryError(
            "recurrent_scorer_normalized_answer_invalid"
        ) from exc
    return {
        "parsed": parsed,
        "correct": verdict["correct"],
        "reason": reason,
        "normalized_answer_sha256": hashlib.sha256(normalized).hexdigest(),
    }


def recurrent_trace_token_encoder(value: bytes) -> tuple[int, ...]:
    """Legacy interface binding; recurrent evidence uses verified token traces."""

    if not isinstance(value, bytes):
        _fail("recurrent_trace_encoder_input_invalid")
    return tuple(value)


def recurrent_trace_token_decoder(tokens: Sequence[int]) -> bytes:
    """Legacy interface binding; reject values outside its byte-only domain."""

    try:
        return bytes(tokens)
    except (TypeError, ValueError) as exc:
        raise VerifiedRecurrentTransitionRepositoryError(
            "recurrent_trace_decoder_input_invalid"
        ) from exc


def _private_root(path: str | Path) -> Path:
    lexical = Path(path).expanduser().absolute()
    if lexical.is_symlink():
        _fail("recurrent_replay_root_symlink_rejected")
    root = ensure_private_directory(lexical).resolve(strict=True)
    metadata = root.stat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        _fail("recurrent_replay_root_not_private_owned")
    return root


def _package_path(root: Path, sequence: int) -> Path:
    if type(sequence) is not int or not 0 <= sequence < 100_000:
        _fail("recurrent_replay_sequence_invalid")
    return root / f"group-{sequence:08d}.prepared.json"


def _publish_package(root: Path, package: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_recurrent_replay_package(package)
    path = _package_path(root, cast(int, validated["sequence"]))
    payload = _json_bytes(validated)
    if not atomic_write_bytes_if_absent(
        path,
        payload,
        mode=0o600,
        durable=True,
    ):
        observed = read_stable_bytes(path, max_bytes=256 * 1024 * 1024)
        if observed != payload:
            _fail("recurrent_replay_publication_conflict")
    return validated


def validate_recurrent_replay_package(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _PACKAGE_KEYS:
        _fail("recurrent_replay_package_schema_invalid")
    document = cast(dict[str, Any], json.loads(_json_bytes(value)))
    observed = _sha256(
        document.get("receipt_sha256"), role="recurrent_replay_receipt"
    )
    unsigned = dict(document)
    unsigned.pop("receipt_sha256")
    sample_json = document.get("sample_receipts_json")
    sample_sha256s = document.get("sample_receipt_sha256s")
    evidence = document.get("evidence_artifacts")
    evidence_sha256s = document.get("evidence_receipt_sha256s")
    prompt = document.get("prompt_tokens")
    if (
        document.get("schema") != RECURRENT_REPLAY_PACKAGE_SCHEMA
        or observed != _digest(unsigned)
        or type(document.get("sequence")) is not int
        or document["sequence"] < 0
        or not isinstance(document.get("task_id"), str)
        or not document["task_id"]
        or not isinstance(document.get("prompt_text"), str)
        or not isinstance(prompt, list)
        or not prompt
        or any(type(token) is not int or token < 0 for token in prompt)
        or document.get("prompt_tokens_sha256") != _tokens_sha256(prompt)
        or not isinstance(sample_json, list)
        or not isinstance(sample_sha256s, list)
        or not isinstance(evidence, list)
        or not isinstance(evidence_sha256s, list)
        or not len(sample_json)
        == len(sample_sha256s)
        == len(evidence)
        == len(evidence_sha256s)
        or type(document.get("created_at_unix_ns")) is not int
        or document["created_at_unix_ns"] <= 0
    ):
        _fail("recurrent_replay_package_invalid")
    for role in (
        "contract_sha256",
        "campaign_schedule_root_sha256",
        "tokenizer_bundle_sha256",
        "reward_receipt_sha256",
    ):
        _sha256(document.get(role), role=f"recurrent_replay_{role}")
    admission = document.get("group_admission_sha256")
    admission_artifact = document.get("group_admission_artifact")
    if (admission is None) is not (admission_artifact is None):
        _fail("recurrent_replay_admission_binding_invalid")
    if admission is not None:
        _sha256(admission, role="recurrent_replay_group_admission")
    for encoded, digest in zip(sample_json, sample_sha256s, strict=True):
        if (
            not isinstance(encoded, str)
            or hashlib.sha256(encoded.encode("ascii")).hexdigest()
            != _sha256(digest, role="recurrent_replay_sample")
        ):
            _fail("recurrent_replay_sample_binding_invalid")
        try:
            parsed = json.loads(encoded)
        except (UnicodeEncodeError, json.JSONDecodeError) as exc:
            raise VerifiedRecurrentTransitionRepositoryError(
                "recurrent_replay_sample_json_invalid"
            ) from exc
        if (
            not isinstance(parsed, Mapping)
            or _float_json(parsed) != encoded
        ):
            _fail("recurrent_replay_sample_json_noncanonical")
        validate_recurrent_policy_sample_receipt(parsed)
    for digest in evidence_sha256s:
        _sha256(digest, role="recurrent_replay_evidence")
    validate_transition_group_manifest(document.get("group_manifest"))
    if not isinstance(document.get("group_manifest_attestation"), Mapping):
        _fail("recurrent_replay_group_attestation_invalid")
    return document


def load_recurrent_replay_packages(request: Any) -> tuple[dict[str, Any], ...]:
    """Load every committed package as pure data in sequence order."""

    root = _private_root(request.replay_artifact_root)
    if (
        request.schema != "aura.verified_transition.restore_request.v2"
        or type(request.committed_steps) is not int
        or request.committed_steps != len(request.step_receipts)
    ):
        _fail("recurrent_replay_restore_request_invalid")
    packages = []
    for sequence, step in enumerate(request.step_receipts):
        path = _package_path(root, sequence)
        if path.is_symlink():
            _fail("recurrent_replay_package_symlink_rejected")
        payload = read_stable_bytes(path, max_bytes=256 * 1024 * 1024)
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VerifiedRecurrentTransitionRepositoryError(
                "recurrent_replay_package_json_invalid"
            ) from exc
        if not isinstance(document, Mapping) or _json_bytes(document) != payload:
            _fail("recurrent_replay_package_noncanonical")
        package = validate_recurrent_replay_package(document)
        if (
            package["sequence"] != sequence
            or package["contract_sha256"] != request.contract_sha256
            or package["campaign_schedule_root_sha256"]
            != request.campaign_schedule_root_sha256
            or package["task_id"] != step.get("task_id")
            or package["reward_receipt_sha256"]
            != step.get("reward_receipt_sha256")
            or package["group_manifest"]["manifest_sha256"]
            != step.get("group_manifest_sha256")
            or package["group_admission_sha256"]
            != step.get("group_admission_sha256")
        ):
            _fail("recurrent_replay_step_binding_mismatch")
        packages.append(package)
    return tuple(packages)


def produce_verified_recurrent_transition_group(
    request: Any,
) -> PreparedVerifiedTransitionGroup:
    """Build, replay, and publish one complete pre-mutation group."""

    if request.schema != "aura.verified_transition.production_request.v2":
        _fail("recurrent_evidence_production_request_invalid")
    roots = request.ledger_roots
    if not isinstance(roots, Mapping):
        _fail("recurrent_evidence_ledger_roots_invalid")
    store = TransitionArtifactStore(roots["transition_artifacts"])
    replay_root = _private_root(roots["replay_artifacts"])
    reward_config = TransitionRewardConfig()
    manifest = validate_transition_group_manifest(request.group_manifest)
    expected_reward_sha256 = hashlib.sha256(
        canonical_json_bytes(reward_config.to_dict())
    ).hexdigest()
    if manifest["reward_config_sha256"] != expected_reward_sha256:
        _fail("recurrent_evidence_reward_config_mismatch")
    created_at = max(time.time_ns(), manifest["planned_at_unix_ns"] + 1)
    evidence: list[VerifiedRecurrentTransitionEvidence] = []
    sample_json: list[str] = []
    sample_sha256s: list[str] = []
    evidence_artifacts: list[dict[str, Any]] = []
    for index, (sample, completion) in enumerate(
        zip(request.samples, request.completions, strict=True)
    ):
        receipt = validate_recurrent_policy_sample_receipt(sample.receipt())
        encoded = _float_json(receipt)
        sample_json.append(encoded)
        sample_sha256s.append(hashlib.sha256(encoded.encode("ascii")).hexdigest())
        item = build_verified_recurrent_transition_evidence(
            store,
            task=request.task,
            prompt_text=request.prompt_text,
            prompt_tokens=request.prompt_tokens,
            sample=sample,
            supplied_completion=completion,
            independent_scorer=request.independent_scorer,
            tokenizer_trace_adapter=request.tokenizer_trace_adapter,
            expected_tokenizer_bundle_sha256=request.tokenizer_bundle_sha256,
            campaign_trust_policy=request.campaign_trust_policy,
            created_at_unix_ns=created_at + index,
        )
        evidence.append(item)
        evidence_artifacts.append(store.put_json(item.document))
    reward = build_verified_transition_reward_batch(
        store,
        evidence,
        independent_scorer=request.independent_scorer,
        token_encoder=request.token_encoder,
        token_decoder=request.token_decoder,
        config=reward_config,
        created_at_unix_ns=created_at + len(evidence),
    )
    reward_artifact = store.put_json(reward)
    admission = None
    admission_artifact = None
    journal = None
    if reward["optimizer_admitted"] is True:
        admission = build_verified_transition_group_admission(
            store,
            reward,
            evidence,
            request.samples,
            request.prompt_tokens,
            group_manifest=manifest,
            group_manifest_attestation=request.group_manifest_attestation,
            independent_scorer=request.independent_scorer,
            token_encoder=request.token_encoder,
            token_decoder=request.token_decoder,
            created_at_unix_ns=created_at + len(evidence) + 1,
        )
        admission_artifact = store.put_json(admission)
        journal = VerifiedTransitionUpdateJournal.open(roots["updates"])
    start = request.campaign_ledger.group_start(sequence=request.sequence)
    body = {
        "schema": RECURRENT_REPLAY_PACKAGE_SCHEMA,
        "contract_sha256": request.contract_sha256,
        "campaign_schedule_root_sha256": request.campaign_schedule_root_sha256,
        "sequence": request.sequence,
        "task_id": request.task.task_id,
        "tokenizer_bundle_sha256": request.tokenizer_bundle_sha256,
        "prompt_text": request.prompt_text,
        "prompt_tokens": list(request.prompt_tokens),
        "prompt_tokens_sha256": _tokens_sha256(request.prompt_tokens),
        "sample_receipts_json": sample_json,
        "sample_receipt_sha256s": sample_sha256s,
        "evidence_artifacts": evidence_artifacts,
        "evidence_receipt_sha256s": [
            item.document["receipt_sha256"] for item in evidence
        ],
        "reward_artifact": reward_artifact,
        "reward_receipt_sha256": reward["receipt_sha256"],
        "group_admission_artifact": admission_artifact,
        "group_admission_sha256": (
            admission["receipt_sha256"] if admission is not None else None
        ),
        "group_manifest": manifest,
        "group_manifest_attestation": dict(
            request.group_manifest_attestation
        ),
        "created_at_unix_ns": created_at + len(evidence) + 2,
    }
    _publish_package(
        replay_root,
        {**body, "receipt_sha256": _digest(body)},
    )
    return PreparedVerifiedTransitionGroup(
        campaign_sequence=request.sequence,
        transition_store=store,
        reward_receipt=reward,
        transition_evidence=tuple(evidence),
        group_manifest=manifest,
        group_manifest_attestation=dict(request.group_manifest_attestation),
        independent_scorer=request.independent_scorer,
        token_encoder=request.token_encoder,
        token_decoder=request.token_decoder,
        campaign_ledger=request.campaign_ledger,
        campaign_trust_policy=request.campaign_trust_policy,
        group_admission_receipt=admission,
        update_journal=journal,
        campaign_manifest_sha256=cast(str, start["campaign_manifest_sha256"]),
        campaign_schedule_root_sha256=request.campaign_schedule_root_sha256,
    )


def reconstruct_recurrent_package_inputs(
    package: Mapping[str, Any],
    *,
    store: TransitionArtifactStore,
    task: Any,
    independent_scorer: Any,
    tokenizer_trace_adapter: Any,
    campaign_trust_policy: Any,
) -> tuple[tuple[Any, ...], tuple[VerifiedRecurrentTransitionEvidence, ...]]:
    """Rebuild samples and evidence from immutable package bytes."""

    validated = validate_recurrent_replay_package(package)
    samples = tuple(
        recurrent_policy_sample_from_receipt(json.loads(encoded))
        for encoded in validated["sample_receipts_json"]
    )
    evidence = []
    for binding, expected in zip(
        validated["evidence_artifacts"],
        validated["evidence_receipt_sha256s"],
        strict=True,
    ):
        document = store.read_json(binding, role="recurrent_transition_evidence")
        if document.get("receipt_sha256") != expected:
            _fail("recurrent_replay_evidence_binding_mismatch")
        evidence.append(
            validate_verified_recurrent_transition_evidence(
                store,
                document,
                task=task,
                independent_scorer=independent_scorer,
                tokenizer_trace_adapter=tokenizer_trace_adapter,
                expected_tokenizer_bundle_sha256=validated[
                    "tokenizer_bundle_sha256"
                ],
                campaign_trust_policy=campaign_trust_policy,
            )
        )
    return samples, tuple(evidence)


def finalize_verified_recurrent_transition_campaign(
    request: Any,
) -> VerifiedTransitionCampaignClosure:
    """Close a fully replayable campaign under an external verifier signature."""

    if request.schema != "aura.verified_transition.finalize_request.v2":
        _fail("recurrent_campaign_finalize_request_invalid")
    if type(request.completed_groups) is not int or request.completed_groups < 0:
        _fail("recurrent_campaign_completed_groups_invalid")
    latest_terminal = 0
    for sequence in range(request.completed_groups):
        terminal = request.campaign_ledger.group_terminal_if_exists(
            sequence=sequence
        )
        if not isinstance(terminal, Mapping):
            _fail("recurrent_campaign_terminal_missing")
        finished_at = terminal.get("finished_at_unix_ns")
        if type(finished_at) is not int or finished_at <= 0:
            _fail("recurrent_campaign_terminal_time_invalid")
        latest_terminal = max(latest_terminal, finished_at)
    completed_at = max(time.time_ns(), latest_terminal)
    payload = request.campaign_ledger.close_payload(
        completed_at_unix_ns=completed_at,
        policy=request.campaign_trust_policy,
    )
    signed_at = (completed_at + 999_999_999) // 1_000_000_000
    attestation = request.evidence_verifier_signer.attest(
        request.campaign_trust_policy,
        role=EVIDENCE_VERIFIER,
        payload=payload,
        signed_at_unix=signed_at,
        purpose="verified-recurrent-campaign-close",
    )
    request.campaign_ledger.close(
        close_payload=payload,
        evidence_verifier_attestation=attestation,
        policy=request.campaign_trust_policy,
    )
    return VerifiedTransitionCampaignClosure(
        campaign_ledger=request.campaign_ledger,
        campaign_trust_policy=request.campaign_trust_policy,
    )


__all__ = [
    "CAMPAIGN_FINALIZER_ID",
    "DURABLE_REPLAY_LOADER_ID",
    "INDEPENDENT_SCORER_ID",
    "PRODUCTION_EVIDENCE_PRODUCER_ID",
    "RECURRENT_REPLAY_PACKAGE_SCHEMA",
    "TOKEN_CODEC_ID",
    "VerifiedRecurrentTransitionRepositoryError",
    "load_recurrent_replay_packages",
    "finalize_verified_recurrent_transition_campaign",
    "produce_verified_recurrent_transition_group",
    "recurrent_trace_token_decoder",
    "recurrent_trace_token_encoder",
    "reconstruct_recurrent_package_inputs",
    "score_verified_recurrent_training_task",
    "validate_recurrent_replay_package",
]
