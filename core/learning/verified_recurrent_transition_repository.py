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

from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_TRUST_ROLES,
    EVIDENCE_VERIFIER,
    VerifiedCampaignTrustPolicy,
)
from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.learning.answer_channel_curriculum import (
    TASK_GENERATORS as ANSWER_CHANNEL_TASK_GENERATORS,
)
from core.learning.recurrence_curriculum import (
    TASK_GENERATORS as RECURRENCE_TASK_GENERATORS,
)
from core.learning.recurrent_grpo import (
    RecurrentGRPOConfig,
    VerifiedTrajectoryGroupConfig,
    exact_adjoint_verified_transition_group_value_and_grad,
    recurrent_policy_sample_from_receipt,
    recurrent_policy_tensor_map_sha256,
    validate_recurrent_policy_sample_receipt,
)
from core.learning.verified_recurrent_transition_evidence import (
    VerifiedRecurrentTransitionEvidence,
    build_verified_recurrent_transition_evidence,
    validate_verified_recurrent_transition_evidence,
)
from core.learning.verified_token_trace import (
    validate_tokenizer_bundle_identity,
    validate_verified_token_trace_structure,
)
from core.learning.verified_training_task import validate_public_training_task
from core.learning.verified_transition_causal_campaign import (
    CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA,
    CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V4,
    CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V5,
    EXTERNAL_EVIDENCE_VERIFICATION_RECEIPT_SCHEMA,
    EXTERNAL_EVIDENCE_VERIFICATION_RECEIPT_SCHEMA_V3,
    EXTERNAL_POLICY_STATE_REPLAY_RESULT_SCHEMA,
    VerifiedTransitionCausalCampaignLedger,
    validate_causal_campaign_evidence_manifest,
    validate_external_evidence_verification_receipt,
    validate_external_policy_state_replay_result,
)
from core.learning.verified_transition_episode import (
    TransitionArtifactStore,
    canonical_json_bytes,
)
from core.learning.verified_transition_group_admission import (
    build_verified_transition_group_admission,
    validate_transition_group_manifest,
    validate_verified_transition_group_admission,
)
from core.learning.verified_transition_measurement_chain import (
    load_pre_measurement_for_transaction,
    load_pre_measurement_state_tensors,
    recurrent_grpo_config_from_contract,
)
from core.learning.verified_transition_policy_state_replay import (
    replay_verified_policy_transition,
    validate_policy_state_replay_contract,
)
from core.learning.verified_transition_rejection_transaction import (
    VerifiedTransitionRejectionTransactionStore,
    build_rejected_transaction_trainer_step,
)
from core.learning.verified_transition_reward import (
    TransitionRewardConfig,
    bind_rewards_to_recurrent_samples,
    build_verified_transition_reward_batch,
    rewards_for_recurrent_samples,
    validate_verified_transition_reward_batch,
)
from core.learning.verified_transition_trainer import (
    PreparedVerifiedTransitionGroup,
    VerifiedTransitionCampaignClosure,
)
from core.learning.verified_transition_transaction import (
    VerifiedTransitionTransactionStore,
    build_transaction_trainer_step,
)
from core.learning.verified_transition_update import (
    VerifiedTransitionUpdateJournal,
    recover_committed_verified_transition_update,
    validate_verified_transition_update_receipt,
)
from core.runtime.atomic_writer import (
    atomic_write_bytes_if_absent,
    ensure_private_directory,
)
from core.runtime.file_read_gateway import read_stable_bytes

RECURRENT_REPLAY_PACKAGE_SCHEMA = "aura.verified_transition.recurrent_replay_package.v1"
PRODUCTION_EVIDENCE_PRODUCER_ID = "aura.verified_transition.recurrent_evidence_producer.v1"
DURABLE_REPLAY_LOADER_ID = "aura.verified_transition.recurrent_replay_loader.v1"
CAMPAIGN_FINALIZER_ID = "aura.verified_transition.recurrent_campaign_finalizer.v1"
INDEPENDENT_SCORER_ID = "aura.verified_transition.recurrent_programmatic_scorer.v1"
TOKEN_CODEC_ID = "aura.verified_transition.recurrent_trace_codec.v1"
EXTERNAL_POLICY_STATE_REPLAY_REQUEST_SCHEMA = (
    "aura.verified_transition.external_policy_state_replay_request.v1"
)
EXTERNAL_POLICY_STATE_REPLAY_REQUEST_PURPOSE = "verified-recurrent-policy-state-replay"
EXTERNAL_POLICY_STATE_REPLAY_BATCH_SCHEMA = (
    "aura.verified_transition.external_policy_state_replay_batch.v1"
)

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
        json.dumps(list(tokens), separators=(",", ":"), allow_nan=False).encode("ascii")
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


def _read_package(root: Path, sequence: int) -> dict[str, Any]:
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
    return validate_recurrent_replay_package(document)


def _package_artifact_binding(root: Path, sequence: int) -> dict[str, Any]:
    path = _package_path(root, sequence).resolve(strict=True)
    payload = read_stable_bytes(path, max_bytes=256 * 1024 * 1024)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _publish_policy_state_replay_result(
    root: Path,
    *,
    sequence: int,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    replay_root = Path(ensure_private_directory(root / "policy-state-replay")).resolve(strict=True)
    path = replay_root / f"transition-{sequence:08d}.json"
    payload = _json_bytes(result)
    if not atomic_write_bytes_if_absent(
        path,
        payload,
        mode=0o600,
        durable=True,
    ):
        observed = read_stable_bytes(
            path,
            max_bytes=256 * 1024 * 1024,
        )
        if observed != payload:
            _fail("external_policy_state_replay_publication_conflict")
    resolved = path.resolve(strict=True)
    metadata = resolved.stat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or metadata.st_nlink != 1
    ):
        _fail("external_policy_state_replay_publication_custody_invalid")
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _validate_policy_state_replay_batch(
    value: Any,
    *,
    request: Mapping[str, Any],
    contract: Mapping[str, Any],
    evidence_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("external_policy_state_replay_batch_invalid")
    document = cast(
        dict[str, Any],
        json.loads(canonical_json_bytes(value)),
    )
    transitions = document.get("transition_results")
    unsigned = dict(document)
    observed = unsigned.pop("result_sha256", None)
    expected_sequences = evidence_manifest["updated_replay_sequences"]
    if (
        set(document)
        != {
            "schema",
            "request_sha256",
            "policy_state_replay_contract_sha256",
            "evidence_manifest_sha256",
            "verifier_identity",
            "verified_at_unix",
            "transition_results",
            "transition_result_root_sha256",
            "completed_at_unix",
            "result_sha256",
        }
        or document.get("schema") != EXTERNAL_POLICY_STATE_REPLAY_BATCH_SCHEMA
        or document.get("request_sha256") != request["request_sha256"]
        or document.get("policy_state_replay_contract_sha256") != contract["contract_sha256"]
        or document.get("evidence_manifest_sha256") != evidence_manifest["manifest_sha256"]
        or document.get("verifier_identity") != request["verifier_identity"]
        or document.get("verified_at_unix") != request["verified_at_unix"]
        or not isinstance(transitions, list)
        or any(not isinstance(row, Mapping) for row in transitions)
        or [row.get("sequence") for row in transitions] != expected_sequences
        or document.get("transition_result_root_sha256")
        != hashlib.sha256(
            canonical_json_bytes(
                [
                    {
                        "sequence": row["sequence"],
                        "receipt_sha256": row.get("receipt_sha256"),
                    }
                    for row in transitions
                ]
            )
        ).hexdigest()
        or type(document.get("completed_at_unix")) is not int
        or document["completed_at_unix"] < request["verified_at_unix"]
        or observed != _digest(unsigned)
    ):
        _fail("external_policy_state_replay_batch_invalid")
    return document


def _reconstruct_external_training_task(
    task_commitment: Mapping[str, Any],
) -> Any:
    public = validate_public_training_task(task_commitment)
    parameters = public["public_parameters"]
    if public["task_type"] == "recurrence_training":
        generators = RECURRENCE_TASK_GENERATORS
    elif public["task_type"] == "answer_channel":
        generators = ANSWER_CHANNEL_TASK_GENERATORS
    else:
        _fail("external_recurrent_task_type_unsupported")
    generator = generators.get(parameters["family"])
    if generator is None:
        _fail("external_recurrent_task_generator_missing")
    task = generator(public["depth"], parameters["seed"])
    if (
        task.task_id != public["task_id"]
        or task.prompt != public["prompt"]
        or task.domain != public["domain"]
    ):
        _fail("external_recurrent_task_reconstruction_mismatch")
    return _ExternallyBoundTrainingTask(task, public)


class _ExternallyBoundTrainingTask:
    def __init__(self, source_task: Any, task_commitment: Mapping[str, Any]) -> None:
        self._source_task = source_task
        self._task_commitment = dict(task_commitment)

    def verified_transition_task_commitment(self) -> Mapping[str, Any]:
        return cast(
            dict[str, Any],
            json.loads(canonical_json_bytes(self._task_commitment)),
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source_task, name)


class _RecordedTokenizerTraceAdapter:
    """Replay only the exact tokenizer observations embedded in one group."""

    def __init__(self, evidence_documents: Sequence[Mapping[str, Any]]) -> None:
        self._bundle: dict[str, Any] | None = None
        self._prompts: dict[str, tuple[int, ...]] = {}
        self._outputs: dict[tuple[int, ...], str] = {}
        self._streams: dict[tuple[int, ...], tuple[str, ...]] = {}
        for evidence in evidence_documents:
            for role in ("parent_token_trace", "child_token_trace"):
                trace = validate_verified_token_trace_structure(evidence.get(role))
                bundle = validate_tokenizer_bundle_identity(trace["tokenizer_bundle"])
                if self._bundle is None:
                    self._bundle = bundle
                elif self._bundle != bundle:
                    _fail("external_recurrent_tokenizer_bundle_mixed")
                prompt = trace["prompt"]
                prompt_text = prompt["text"]
                prompt_tokens = tuple(prompt["token_ids"])
                generation = trace["generation"]
                output_tokens = tuple(generation["token_ids"])
                response = generation["response_text"]
                deltas = tuple(generation["streaming_deltas"])
                if prompt_text in self._prompts and self._prompts[prompt_text] != prompt_tokens:
                    _fail("external_recurrent_prompt_encoding_conflict")
                if output_tokens in self._outputs and (
                    self._outputs[output_tokens] != response
                    or self._streams[output_tokens] != deltas
                ):
                    _fail("external_recurrent_output_decoding_conflict")
                self._prompts[prompt_text] = prompt_tokens
                self._outputs[output_tokens] = response
                self._streams[output_tokens] = deltas
        if self._bundle is None:
            _fail("external_recurrent_tokenizer_observations_missing")

    @property
    def bundle_identity(self) -> Mapping[str, Any]:
        bundle = self._bundle
        if bundle is None:
            _fail("external_recurrent_tokenizer_observations_missing")
        return cast(dict[str, Any], json.loads(_json_bytes(bundle)))

    def encode_prompt(self, prompt_text: str) -> Sequence[int]:
        try:
            return self._prompts[prompt_text]
        except KeyError as exc:
            raise VerifiedRecurrentTransitionRepositoryError(
                "external_recurrent_prompt_observation_missing"
            ) from exc

    def decode_output(self, token_ids: Sequence[int]) -> str:
        try:
            return self._outputs[tuple(token_ids)]
        except KeyError as exc:
            raise VerifiedRecurrentTransitionRepositoryError(
                "external_recurrent_output_observation_missing"
            ) from exc

    def stream_decode_deltas(self, token_ids: Sequence[int]) -> Sequence[str]:
        try:
            return self._streams[tuple(token_ids)]
        except KeyError as exc:
            raise VerifiedRecurrentTransitionRepositoryError(
                "external_recurrent_stream_observation_missing"
            ) from exc


def campaign_trust_policy_from_verifier_material(
    value: Any,
) -> VerifiedCampaignTrustPolicy:
    """Reconstruct public trust material inside the verifier process."""

    if (
        not isinstance(value, Mapping)
        or set(value) != {"document", "policy_sha256", "root_key_id"}
        or not isinstance(value.get("document"), Mapping)
    ):
        _fail("external_recurrent_trust_policy_material_invalid")
    document = cast(
        dict[str, Any],
        json.loads(canonical_json_bytes(value["document"])),
    )
    policy_sha256 = _sha256(
        value.get("policy_sha256"),
        role="external_recurrent_trust_policy",
    )
    root_key_id = _sha256(
        value.get("root_key_id"),
        role="external_recurrent_trust_root",
    )
    root_signature = document.get("root_signature")
    if (
        policy_sha256 != hashlib.sha256(canonical_json_bytes(document)).hexdigest()
        or not isinstance(root_signature, Mapping)
        or root_signature.get("key_id") != root_key_id
    ):
        _fail("external_recurrent_trust_policy_material_mismatch")
    policy = VerifiedCampaignTrustPolicy(
        document=document,
        policy_sha256=policy_sha256,
        root_key_id=root_key_id,
    )
    for role in CAMPAIGN_TRUST_ROLES:
        pin = policy.role_pin(role)
        for field in (
            "implementation_sha256",
            "release_sha256",
            "custody_evidence_sha256",
            "key_id",
        ):
            _sha256(
                pin.get(field),
                role=f"external_recurrent_{role}_{field}",
            )
    return policy


def verify_recurrent_evidence_manifest_artifacts(
    evidence_manifest: Mapping[str, Any],
    *,
    campaign_trust_policy: VerifiedCampaignTrustPolicy,
    verifier_identity: str,
    verified_at_unix: int,
) -> dict[str, Any]:
    """Revalidate every bound causal receipt and accepted update externally."""

    evidence = validate_causal_campaign_evidence_manifest(evidence_manifest)
    manifest_has_pre_measurements = evidence["schema"] in {
        CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V4,
        CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V5,
    }
    manifest_has_policy_state_replay = (
        evidence["schema"] == CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V5
    )
    if (
        not isinstance(campaign_trust_policy, VerifiedCampaignTrustPolicy)
        or evidence["trust_policy_sha256"] != campaign_trust_policy.policy_sha256
        or not isinstance(verifier_identity, str)
        or not verifier_identity
        or verifier_identity != verifier_identity.strip()
        or type(verified_at_unix) is not int
        or verified_at_unix <= 0
    ):
        _fail("external_recurrent_verifier_identity_invalid")
    campaign_ledger = VerifiedTransitionCausalCampaignLedger.open(
        evidence["campaign_ledger_root"],
        policy=campaign_trust_policy,
    )
    campaign_manifest = campaign_ledger.campaign_manifest()
    if (
        campaign_manifest["provider_contract_sha256"] != evidence["contract_sha256"]
        or campaign_manifest["campaign_schedule_root_sha256"]
        != evidence["campaign_schedule_root_sha256"]
        or campaign_manifest["trust_policy_sha256"] != evidence["trust_policy_sha256"]
    ):
        _fail("external_recurrent_campaign_manifest_mismatch")
    transition_store = TransitionArtifactStore(evidence["transition_artifact_root"])
    update_journal: VerifiedTransitionUpdateJournal | None = None
    transaction_store: VerifiedTransitionTransactionStore | None = None
    rejection_store: VerifiedTransitionRejectionTransactionStore | None = None
    observations: list[dict[str, Any]] = []
    try:
        for row in evidence["group_packages"]:
            binding = row["package_artifact"]
            path = Path(binding["path"])
            if path.is_symlink():
                _fail("external_recurrent_package_symlink_rejected")
            payload = read_stable_bytes(
                path,
                max_bytes=256 * 1024 * 1024,
            )
            if (
                len(payload) != binding["size_bytes"]
                or hashlib.sha256(payload).hexdigest() != binding["sha256"]
            ):
                _fail("external_recurrent_package_binding_mismatch")
            try:
                parsed = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise VerifiedRecurrentTransitionRepositoryError(
                    "external_recurrent_package_json_invalid"
                ) from exc
            if not isinstance(parsed, Mapping) or _json_bytes(parsed) != payload:
                _fail("external_recurrent_package_noncanonical")
            package = validate_recurrent_replay_package(parsed)
            if (
                package["sequence"] != row["sequence"]
                or package["contract_sha256"] != evidence["contract_sha256"]
                or package["campaign_schedule_root_sha256"]
                != evidence["campaign_schedule_root_sha256"]
                or package["receipt_sha256"] != row["package_receipt_sha256"]
                or package["group_manifest"]["manifest_sha256"] != row["group_manifest_sha256"]
                or package["reward_receipt_sha256"] != row["reward_receipt_sha256"]
                or package["group_admission_sha256"] != row["group_admission_sha256"]
                or package["sample_receipt_sha256s"] != row["sample_receipt_sha256s"]
                or package["evidence_receipt_sha256s"] != row["evidence_receipt_sha256s"]
            ):
                _fail("external_recurrent_package_manifest_mismatch")

            evidence_documents = [
                transition_store.read_json(
                    artifact,
                    role="external_recurrent_evidence",
                )
                for artifact in package["evidence_artifacts"]
            ]
            if (
                not evidence_documents
                or [document.get("receipt_sha256") for document in evidence_documents]
                != package["evidence_receipt_sha256s"]
            ):
                _fail("external_recurrent_evidence_binding_mismatch")
            task_commitment = evidence_documents[0].get("task_commitment")
            if not isinstance(task_commitment, Mapping):
                _fail("external_recurrent_task_commitment_missing")
            task = _reconstruct_external_training_task(task_commitment)
            if task.prompt != package["prompt_text"]:
                _fail("external_recurrent_task_prompt_mismatch")
            tokenizer_adapter = _RecordedTokenizerTraceAdapter(evidence_documents)
            samples = tuple(
                recurrent_policy_sample_from_receipt(json.loads(encoded))
                for encoded in package["sample_receipts_json"]
            )
            replayed_evidence = tuple(
                validate_verified_recurrent_transition_evidence(
                    transition_store,
                    document,
                    task=task,
                    independent_scorer=(score_verified_recurrent_training_task),
                    tokenizer_trace_adapter=tokenizer_adapter,
                    expected_tokenizer_bundle_sha256=package["tokenizer_bundle_sha256"],
                    campaign_trust_policy=campaign_trust_policy,
                )
                for document in evidence_documents
            )
            reward = validate_verified_transition_reward_batch(
                transition_store,
                transition_store.read_json(
                    package["reward_artifact"],
                    role="external_recurrent_reward",
                ),
                replayed_evidence,
                independent_scorer=score_verified_recurrent_training_task,
                token_encoder=recurrent_trace_token_encoder,
                token_decoder=recurrent_trace_token_decoder,
            )
            if reward["receipt_sha256"] != package["reward_receipt_sha256"]:
                _fail("external_recurrent_reward_binding_mismatch")

            terminal = campaign_ledger.group_terminal_if_exists(sequence=row["sequence"])
            if (
                not isinstance(terminal, Mapping)
                or terminal.get("status") != row["status"]
                or terminal.get("group_manifest_sha256") != row["group_manifest_sha256"]
                or terminal.get("reward_receipt_sha256") != row["reward_receipt_sha256"]
                or terminal.get("group_admission_sha256") != row["group_admission_sha256"]
                or terminal.get("update_receipt_sha256") != row["update_receipt_sha256"]
            ):
                _fail("external_recurrent_terminal_binding_mismatch")

            if row["status"] == "updated":
                if (
                    reward["optimizer_admitted"] is not True
                    or package["group_admission_artifact"] is None
                    or package["group_admission_sha256"] is None
                    or row["update_receipt_sha256"] is None
                ):
                    _fail("external_recurrent_updated_evidence_missing")
                admission = validate_verified_transition_group_admission(
                    transition_store,
                    transition_store.read_json(
                        package["group_admission_artifact"],
                        role="external_recurrent_admission",
                    ),
                    reward,
                    replayed_evidence,
                    samples,
                    package["prompt_tokens"],
                    group_manifest=package["group_manifest"],
                    group_manifest_attestation=package["group_manifest_attestation"],
                    independent_scorer=(score_verified_recurrent_training_task),
                    token_encoder=recurrent_trace_token_encoder,
                    token_decoder=recurrent_trace_token_decoder,
                )
                admission_sha256 = admission["receipt_sha256"]
                if admission_sha256 != package["group_admission_sha256"]:
                    _fail("external_recurrent_admission_binding_mismatch")
                if update_journal is None:
                    update_journal = VerifiedTransitionUpdateJournal.open(
                        evidence["update_journal_root"]
                    )
                update = recover_committed_verified_transition_update(
                    update_journal,
                    admission_sha256,
                )
                if update["receipt_sha256"] != row["update_receipt_sha256"]:
                    _fail("external_recurrent_update_binding_mismatch")
                if transaction_store is None:
                    transaction_store = VerifiedTransitionTransactionStore.open(
                        evidence["transaction_root"]
                    )
                transaction = transaction_store.load(
                    sequence=row["sequence"],
                    admission_sha256=admission_sha256,
                    load_tensors=True,
                )
                if (
                    transaction is None
                    or transaction.adapter_tensors is None
                    or tuple(event["kind"] for event in transaction.events)
                    != (
                        "update_commit",
                        "campaign_terminal",
                        "trainer_checkpoint",
                    )
                    or transaction.events[0]["evidence"] != update
                    or transaction.events[1]["evidence"] != dict(terminal)
                ):
                    _fail("external_recurrent_transaction_chain_mismatch")
                pending = transaction.pending_step
                expected_pre_measurement_sha256 = (
                    row["pre_measurement_sha256"] if manifest_has_pre_measurements else None
                )
                validate_verified_transition_update_receipt(
                    update_journal,
                    update,
                    expected_pre_measurement_sha256=(expected_pre_measurement_sha256),
                    expected_campaign_sequence=(
                        row["sequence"] if manifest_has_pre_measurements else None
                    ),
                    expected_execution_spec_sha256=(
                        pending["execution_spec_sha256"] if manifest_has_pre_measurements else None
                    ),
                    expected_group_manifest_sha256=(
                        row["group_manifest_sha256"] if manifest_has_pre_measurements else None
                    ),
                )
                group_policy = package["group_manifest"]["entries"][0]["policy_sha256"]
                trainer_step = build_transaction_trainer_step(transaction)
                if (
                    pending["sequence"] != row["sequence"]
                    or pending["task_id"] != package["task_id"]
                    or pending["execution_spec_sha256"]
                    != package["group_manifest"]["entries"][0]["recurrent_execution_spec_sha256"]
                    or pending["campaign_manifest_sha256"] != campaign_manifest["manifest_sha256"]
                    or pending["campaign_schedule_root_sha256"]
                    != evidence["campaign_schedule_root_sha256"]
                    or pending["group_manifest_sha256"] != row["group_manifest_sha256"]
                    or pending["group_admission_sha256"] != admission_sha256
                    or pending["reward_receipt_sha256"] != row["reward_receipt_sha256"]
                    or pending["policy_before_sha256"] != group_policy
                    or pending["policy_before_sha256"] != update["policy_before_sha256"]
                    or pending["policy_after_sha256"] != update["policy_after_sha256"]
                    or (
                        manifest_has_pre_measurements
                        and pending.get("reservation_sha256") != update["reservation_sha256"]
                    )
                    or pending.get("pre_measurement_sha256") != expected_pre_measurement_sha256
                    or terminal.get("policy_before_sha256") != update["policy_before_sha256"]
                    or terminal.get("policy_after_sha256") != update["policy_after_sha256"]
                    or trainer_step["receipt_sha256"] != row["trainer_step_receipt_sha256"]
                    or recurrent_policy_tensor_map_sha256(
                        transaction.adapter_tensors,
                        pending["execution_spec_sha256"],
                    )
                    != update["policy_after_sha256"]
                ):
                    _fail("external_recurrent_transaction_causality_mismatch")
                structured_rewards = list(
                    rewards_for_recurrent_samples(
                        reward,
                        samples,
                        package["prompt_tokens"],
                    )
                )
                static = pending["trainer_step_static"]
                objective = update_journal.read(admission_sha256, "objective")["objective_receipt"]
                if (
                    static["structured_rewards"] != structured_rewards
                    or objective["advantage_report"] != static["advantage_report"]
                    or objective["completion_count"] != len(samples)
                    or objective["token_count"] != sum(len(sample.tokens) for sample in samples)
                    or objective["branch_indices"] != [sample.branch_index for sample in samples]
                ):
                    _fail("external_recurrent_objective_causality_mismatch")
                if manifest_has_policy_state_replay:
                    replay_binding = row["policy_state_replay_receipt_artifact"]
                    replay_path = Path(replay_binding["path"])
                    if replay_path.is_symlink():
                        _fail("external_policy_state_replay_artifact_symlink_rejected")
                    replay_payload = read_stable_bytes(
                        replay_path,
                        max_bytes=256 * 1024 * 1024,
                    )
                    if (
                        len(replay_payload) != replay_binding["size_bytes"]
                        or hashlib.sha256(replay_payload).hexdigest() != replay_binding["sha256"]
                    ):
                        _fail("external_policy_state_replay_artifact_binding_mismatch")
                    try:
                        replay_document = json.loads(replay_payload)
                    except (
                        UnicodeDecodeError,
                        json.JSONDecodeError,
                    ) as exc:
                        raise VerifiedRecurrentTransitionRepositoryError(
                            "external_policy_state_replay_artifact_json_invalid"
                        ) from exc
                    if (
                        not isinstance(replay_document, Mapping)
                        or _json_bytes(replay_document) != replay_payload
                    ):
                        _fail("external_policy_state_replay_artifact_noncanonical")
                    replay_result = validate_external_policy_state_replay_result(
                        replay_document,
                        policy_state_replay_contract=evidence["policy_state_replay_contract"],
                        expected_transition={
                            "provider_contract_sha256": evidence["contract_sha256"],
                            "campaign_schedule_root_sha256": evidence[
                                "campaign_schedule_root_sha256"
                            ],
                            "campaign_manifest_sha256": campaign_manifest["manifest_sha256"],
                            **{
                                key: row[key]
                                for key in (
                                    "sequence",
                                    "group_manifest_sha256",
                                    "group_admission_sha256",
                                    "update_receipt_sha256",
                                    "pre_measurement_sha256",
                                    "state_source_sha256",
                                    "post_state_transaction_stage_sha256",
                                    "objective_receipt_sha256",
                                    "policy_before_sha256",
                                    "policy_after_sha256",
                                    "policy_state_replay_receipt_artifact",
                                    "policy_state_replay_receipt_sha256",
                                )
                            },
                        },
                    )
                    if (
                        replay_result["verifier_identity"] != verifier_identity
                        or replay_result["verified_at_unix"] != verified_at_unix
                        or replay_result["objective_receipt_sha256"] != _digest(objective)
                        or replay_result["policy_before_sha256"] != pending["policy_before_sha256"]
                        or replay_result["policy_after_sha256"] != pending["policy_after_sha256"]
                        or replay_result["state_source_sha256"]
                        != load_pre_measurement_for_transaction(
                            evidence["transaction_root"],
                            sequence=row["sequence"],
                            admission_sha256=admission_sha256,
                            expected_receipt_sha256=row["pre_measurement_sha256"],
                        )["state_source"]["state_source_sha256"]
                        or replay_result["post_state_transaction_stage_sha256"]
                        != transaction.stage["receipt_sha256"]
                    ):
                        _fail("external_policy_state_replay_causality_mismatch")
            else:
                if (
                    reward["optimizer_admitted"] is not False
                    or package["group_admission_artifact"] is not None
                    or package["group_admission_sha256"] is not None
                    or row["update_receipt_sha256"] is not None
                ):
                    _fail("external_recurrent_rejection_causality_mismatch")
                if rejection_store is None:
                    rejection_store = VerifiedTransitionRejectionTransactionStore.open(
                        evidence["transaction_root"]
                    )
                rejection = rejection_store.load(
                    sequence=row["sequence"],
                    reward_sha256=row["reward_receipt_sha256"],
                )
                if (
                    rejection is None
                    or tuple(event["kind"] for event in rejection.events)
                    != ("campaign_terminal", "trainer_checkpoint")
                    or rejection.events[0]["evidence"] != dict(terminal)
                ):
                    _fail("external_recurrent_rejection_chain_mismatch")
                intent = rejection.intent
                trainer_step = build_rejected_transaction_trainer_step(rejection)
                static = intent["trainer_step_static"]
                structured_rewards = list(
                    bind_rewards_to_recurrent_samples(
                        reward,
                        samples,
                        package["prompt_tokens"],
                    )
                )
                if (
                    intent["sequence"] != row["sequence"]
                    or intent["task_id"] != package["task_id"]
                    or intent["execution_spec_sha256"]
                    != package["group_manifest"]["entries"][0]["recurrent_execution_spec_sha256"]
                    or intent["campaign_manifest_sha256"] != campaign_manifest["manifest_sha256"]
                    or intent["campaign_schedule_root_sha256"]
                    != evidence["campaign_schedule_root_sha256"]
                    or intent["group_manifest_sha256"] != row["group_manifest_sha256"]
                    or intent["reward_receipt_sha256"] != row["reward_receipt_sha256"]
                    or intent["policy_sha256"]
                    != package["group_manifest"]["entries"][0]["policy_sha256"]
                    or terminal.get("policy_before_sha256") != intent["policy_sha256"]
                    or terminal.get("policy_after_sha256") != intent["policy_sha256"]
                    or static["samples"] != [sample.receipt() for sample in samples]
                    or static["structured_rewards"] != structured_rewards
                    or trainer_step["receipt_sha256"] != row["trainer_step_receipt_sha256"]
                ):
                    _fail("external_recurrent_rejection_causality_mismatch")

            observation = {
                "sequence": row["sequence"],
                "package_artifact": binding,
                "package_receipt_sha256": row["package_receipt_sha256"],
                "sample_receipt_sha256s": row["sample_receipt_sha256s"],
                "evidence_receipt_sha256s": row["evidence_receipt_sha256s"],
                "reward_receipt_sha256": row["reward_receipt_sha256"],
                "group_admission_sha256": row["group_admission_sha256"],
                "update_receipt_sha256": row["update_receipt_sha256"],
                "trainer_step_receipt_sha256": row["trainer_step_receipt_sha256"],
            }
            if manifest_has_pre_measurements:
                observation["pre_measurement_sha256"] = row["pre_measurement_sha256"]
            if manifest_has_policy_state_replay:
                observation.update(
                    {
                        "group_manifest_sha256": row["group_manifest_sha256"],
                        "policy_before_sha256": row["policy_before_sha256"],
                        "policy_after_sha256": row["policy_after_sha256"],
                        "objective_receipt_sha256": row["objective_receipt_sha256"],
                        "state_source_sha256": row["state_source_sha256"],
                        "post_state_transaction_stage_sha256": row[
                            "post_state_transaction_stage_sha256"
                        ],
                        "policy_state_replay_receipt_artifact": row[
                            "policy_state_replay_receipt_artifact"
                        ],
                        "policy_state_replay_receipt_sha256": row[
                            "policy_state_replay_receipt_sha256"
                        ],
                    }
                )
            observations.append(observation)
    finally:
        transition_store.close()
    body = {
        "schema": (
            EXTERNAL_EVIDENCE_VERIFICATION_RECEIPT_SCHEMA_V3
            if manifest_has_policy_state_replay
            else EXTERNAL_EVIDENCE_VERIFICATION_RECEIPT_SCHEMA
        ),
        "evidence_manifest_sha256": evidence["manifest_sha256"],
        "verifier_identity": verifier_identity,
        "verified_package_count": len(observations),
        "artifact_observation_root_sha256": _digest({"artifact_observations": observations}),
        "validation_profile": (
            "recurrent_transition_causal_replay.v4"
            if manifest_has_policy_state_replay
            else (
                "recurrent_transition_causal_replay.v3"
                if manifest_has_pre_measurements
                else "recurrent_transition_causal_replay.v2"
            )
        ),
        "verified_at_unix": verified_at_unix,
        **(
            {
                "policy_state_replay_contract_sha256": evidence[
                    "policy_state_replay_contract_sha256"
                ],
                "verified_updated_transition_count": len(evidence["updated_replay_sequences"]),
                "policy_state_replay_receipt_root_sha256": evidence[
                    "policy_state_replay_receipt_root_sha256"
                ],
                "external_policy_state_replayed": True,
            }
            if manifest_has_policy_state_replay
            else {}
        ),
    }
    return validate_external_evidence_verification_receipt(
        {**body, "receipt_sha256": _digest(body)},
        evidence_manifest=evidence,
    )


def replay_recurrent_evidence_manifest_policy_states(
    evidence_manifest: Mapping[str, Any],
    *,
    policy_state_replay_contract: Mapping[str, Any],
    campaign_trust_policy: VerifiedCampaignTrustPolicy,
    verifier_identity: str,
    verified_at_unix: int,
    model: Any,
) -> tuple[dict[str, Any], ...]:
    """Independently recompute every admitted recurrent optimizer transition.

    The ordinary external replay first proves the immutable campaign chain.
    This second pass restores each sealed pre-measurement state into a caller-
    constructed model, recomputes the exact adjoint objective and gradients,
    replays one frozen Adam update, and compares both producer post-states.
    """

    evidence = validate_causal_campaign_evidence_manifest(evidence_manifest)
    if evidence["schema"] != CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V4:
        _fail("external_policy_state_replay_requires_v4_manifest")
    contract = validate_policy_state_replay_contract(
        policy_state_replay_contract,
        verify_files=True,
    )
    verify_recurrent_evidence_manifest_artifacts(
        evidence,
        campaign_trust_policy=campaign_trust_policy,
        verifier_identity=verifier_identity,
        verified_at_unix=verified_at_unix,
    )
    campaign_ledger = VerifiedTransitionCausalCampaignLedger.open(
        evidence["campaign_ledger_root"],
        policy=campaign_trust_policy,
    )
    campaign_manifest = campaign_ledger.campaign_manifest()
    if (
        campaign_manifest["initial_policy_sha256"] != contract["initial_policy_sha256"]
        or evidence["campaign_schedule_root_sha256"]
        != campaign_manifest["campaign_schedule_root_sha256"]
        or evidence["trust_policy_sha256"] != campaign_trust_policy.policy_sha256
    ):
        _fail("external_policy_state_replay_contract_campaign_mismatch")
    try:
        execution_spec = RLCExecutionSpec.from_dict(
            json.loads(contract["execution_spec"]["document_json"])
        )
        trajectory_config = VerifiedTrajectoryGroupConfig.from_dict(
            json.loads(contract["verified_trajectory_config_json"])
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VerifiedRecurrentTransitionRepositoryError(
            "external_policy_state_replay_objective_contract_invalid"
        ) from exc
    if execution_spec.sha256 != contract["execution_spec"]["semantic_sha256"]:
        _fail("external_policy_state_replay_execution_spec_mismatch")
    transition_store = TransitionArtifactStore(evidence["transition_artifact_root"])
    update_journal = VerifiedTransitionUpdateJournal.open(evidence["update_journal_root"])
    transaction_store = VerifiedTransitionTransactionStore.open(evidence["transaction_root"])
    results: list[dict[str, Any]] = []
    try:
        for row in evidence["group_packages"]:
            if row["status"] != "updated":
                continue
            binding = row["package_artifact"]
            payload = read_stable_bytes(
                Path(binding["path"]),
                max_bytes=256 * 1024 * 1024,
            )
            if (
                len(payload) != binding["size_bytes"]
                or hashlib.sha256(payload).hexdigest() != binding["sha256"]
            ):
                _fail("external_policy_state_replay_package_binding_mismatch")
            try:
                parsed = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise VerifiedRecurrentTransitionRepositoryError(
                    "external_policy_state_replay_package_json_invalid"
                ) from exc
            if not isinstance(parsed, Mapping) or _json_bytes(parsed) != payload:
                _fail("external_policy_state_replay_package_noncanonical")
            package = validate_recurrent_replay_package(parsed)
            evidence_documents = [
                transition_store.read_json(
                    artifact,
                    role="external_policy_state_replay_evidence",
                )
                for artifact in package["evidence_artifacts"]
            ]
            task_commitment = evidence_documents[0].get("task_commitment")
            if not isinstance(task_commitment, Mapping):
                _fail("external_policy_state_replay_task_missing")
            task = _reconstruct_external_training_task(task_commitment)
            tokenizer_adapter = _RecordedTokenizerTraceAdapter(evidence_documents)
            samples = tuple(
                recurrent_policy_sample_from_receipt(json.loads(encoded))
                for encoded in package["sample_receipts_json"]
            )
            replayed_evidence = tuple(
                validate_verified_recurrent_transition_evidence(
                    transition_store,
                    document,
                    task=task,
                    independent_scorer=(score_verified_recurrent_training_task),
                    tokenizer_trace_adapter=tokenizer_adapter,
                    expected_tokenizer_bundle_sha256=package["tokenizer_bundle_sha256"],
                    campaign_trust_policy=campaign_trust_policy,
                )
                for document in evidence_documents
            )
            reward = validate_verified_transition_reward_batch(
                transition_store,
                transition_store.read_json(
                    package["reward_artifact"],
                    role="external_policy_state_replay_reward",
                ),
                replayed_evidence,
                independent_scorer=score_verified_recurrent_training_task,
                token_encoder=recurrent_trace_token_encoder,
                token_decoder=recurrent_trace_token_decoder,
            )
            admission = validate_verified_transition_group_admission(
                transition_store,
                transition_store.read_json(
                    package["group_admission_artifact"],
                    role="external_policy_state_replay_admission",
                ),
                reward,
                replayed_evidence,
                samples,
                package["prompt_tokens"],
                group_manifest=package["group_manifest"],
                group_manifest_attestation=package["group_manifest_attestation"],
                independent_scorer=score_verified_recurrent_training_task,
                token_encoder=recurrent_trace_token_encoder,
                token_decoder=recurrent_trace_token_decoder,
            )
            admission_sha256 = admission["receipt_sha256"]
            transaction = transaction_store.load(
                sequence=row["sequence"],
                admission_sha256=admission_sha256,
                load_tensors=True,
            )
            if (
                transaction is None
                or transaction.adapter_tensors is None
                or transaction.optimizer_tensors is None
                or row["pre_measurement_sha256"] is None
            ):
                _fail("external_policy_state_replay_transaction_missing")
            pending = transaction.pending_step
            intent = load_pre_measurement_for_transaction(
                evidence["transaction_root"],
                sequence=row["sequence"],
                admission_sha256=admission_sha256,
                expected_receipt_sha256=row["pre_measurement_sha256"],
            )
            if (
                intent["execution_spec_sha256"] != execution_spec.sha256
                or intent["provider_contract_sha256"] != evidence["contract_sha256"]
                or intent["campaign_manifest_sha256"] != campaign_manifest["manifest_sha256"]
                or intent["campaign_schedule_root_sha256"]
                != evidence["campaign_schedule_root_sha256"]
                or intent["group_manifest_sha256"] != row["group_manifest_sha256"]
                or intent["group_admission_sha256"] != admission_sha256
                or intent["recurrent_grpo_config"] != contract["recurrent_grpo_config"]
                or intent["trajectory_source_binding"]["config"] != trajectory_config.to_dict()
            ):
                _fail("external_policy_state_replay_intent_mismatch")
            pre_adapter, pre_optimizer = load_pre_measurement_state_tensors(intent)
            objective = update_journal.read(
                admission_sha256,
                "objective",
            )["objective_receipt"]
            recurrent_config: RecurrentGRPOConfig = recurrent_grpo_config_from_contract(
                intent["recurrent_grpo_config"]
            )

            def objective_factory(
                active_model: Any,
                *,
                _prompt_tokens: Sequence[int] = package["prompt_tokens"],
                _samples: tuple[Any, ...] = samples,
                _admission: Mapping[str, Any] = admission,
                _reward: Mapping[str, Any] = reward,
                _evidence: tuple[Any, ...] = replayed_evidence,
                _manifest: Mapping[str, Any] = package["group_manifest"],
                _attestation: Mapping[str, Any] = package["group_manifest_attestation"],
                _config: RecurrentGRPOConfig = recurrent_config,
            ) -> Any:
                return exact_adjoint_verified_transition_group_value_and_grad(
                    active_model,
                    _prompt_tokens,
                    _samples,
                    _admission,
                    _reward,
                    _evidence,
                    transition_store=transition_store,
                    group_manifest=_manifest,
                    group_manifest_attestation=_attestation,
                    independent_scorer=(score_verified_recurrent_training_task),
                    token_encoder=recurrent_trace_token_encoder,
                    token_decoder=recurrent_trace_token_decoder,
                    spec=execution_spec,
                    bridge_tokens=(),
                    config=_config,
                    trajectory_group_config=trajectory_config,
                )

            replay_receipt = replay_verified_policy_transition(
                model=model,
                pre_adapter_tensors=pre_adapter,
                pre_optimizer_tensors=pre_optimizer,
                expected_post_adapter_tensors=transaction.adapter_tensors,
                expected_post_optimizer_tensors=transaction.optimizer_tensors,
                expected_objective_receipt=objective,
                objective_factory=objective_factory,
                optimizer_config=contract["optimizer_config"],
                execution_spec_sha256=execution_spec.sha256,
                expected_policy_before_sha256=pending["policy_before_sha256"],
                expected_policy_after_sha256=pending["policy_after_sha256"],
            )
            result_body = {
                "schema": EXTERNAL_POLICY_STATE_REPLAY_RESULT_SCHEMA,
                "policy_state_replay_contract_sha256": contract["contract_sha256"],
                "provider_contract_sha256": evidence["contract_sha256"],
                "campaign_schedule_root_sha256": evidence["campaign_schedule_root_sha256"],
                "campaign_manifest_sha256": campaign_manifest["manifest_sha256"],
                "sequence": row["sequence"],
                "group_manifest_sha256": row["group_manifest_sha256"],
                "group_admission_sha256": admission_sha256,
                "update_receipt_sha256": row["update_receipt_sha256"],
                "pre_measurement_sha256": row["pre_measurement_sha256"],
                "state_source_sha256": intent["state_source"]["state_source_sha256"],
                "post_state_transaction_stage_sha256": transaction.stage["receipt_sha256"],
                "execution_spec_sha256": execution_spec.sha256,
                "objective_receipt_sha256": _digest(objective),
                "policy_before_sha256": pending["policy_before_sha256"],
                "policy_after_sha256": pending["policy_after_sha256"],
                "verifier_identity": verifier_identity,
                "verified_at_unix": verified_at_unix,
                "policy_state_replay_receipt": replay_receipt,
            }
            results.append(
                {
                    **result_body,
                    "receipt_sha256": _digest(result_body),
                }
            )
    finally:
        transition_store.close()
    if [result["sequence"] for result in results] != evidence["updated_replay_sequences"]:
        _fail("external_policy_state_replay_result_set_mismatch")
    return tuple(results)


def validate_recurrent_replay_package(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _PACKAGE_KEYS:
        _fail("recurrent_replay_package_schema_invalid")
    document = cast(dict[str, Any], json.loads(_json_bytes(value)))
    observed = _sha256(document.get("receipt_sha256"), role="recurrent_replay_receipt")
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
        or not len(sample_json) == len(sample_sha256s) == len(evidence) == len(evidence_sha256s)
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
        if not isinstance(encoded, str) or hashlib.sha256(
            encoded.encode("ascii")
        ).hexdigest() != _sha256(digest, role="recurrent_replay_sample"):
            _fail("recurrent_replay_sample_binding_invalid")
        try:
            parsed = json.loads(encoded)
        except (UnicodeEncodeError, json.JSONDecodeError) as exc:
            raise VerifiedRecurrentTransitionRepositoryError(
                "recurrent_replay_sample_json_invalid"
            ) from exc
        if not isinstance(parsed, Mapping) or _float_json(parsed) != encoded:
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
        package = _read_package(root, sequence)
        if (
            package["sequence"] != sequence
            or package["contract_sha256"] != request.contract_sha256
            or package["campaign_schedule_root_sha256"] != request.campaign_schedule_root_sha256
            or package["task_id"] != step.get("task_id")
            or package["reward_receipt_sha256"] != step.get("reward_receipt_sha256")
            or package["group_manifest"]["manifest_sha256"] != step.get("group_manifest_sha256")
            or package["group_admission_sha256"] != step.get("group_admission_sha256")
        ):
            _fail("recurrent_replay_step_binding_mismatch")
        packages.append(package)
    return tuple(packages)


def _prepared_from_existing_package(
    request: Any,
    package: Mapping[str, Any],
    *,
    store: TransitionArtifactStore,
) -> PreparedVerifiedTransitionGroup:
    document = validate_recurrent_replay_package(package)
    sample_receipts = tuple(sample.receipt() for sample in request.samples)
    expected_sample_json = tuple(_float_json(receipt) for receipt in sample_receipts)
    if (
        document["contract_sha256"] != request.contract_sha256
        or document["campaign_schedule_root_sha256"] != request.campaign_schedule_root_sha256
        or document["sequence"] != request.sequence
        or document["task_id"] != request.task.task_id
        or document["tokenizer_bundle_sha256"] != request.tokenizer_bundle_sha256
        or document["prompt_text"] != request.prompt_text
        or document["prompt_tokens"] != list(request.prompt_tokens)
        or tuple(document["sample_receipts_json"]) != expected_sample_json
        or document["group_manifest"] != dict(request.group_manifest)
        or document["group_manifest_attestation"] != dict(request.group_manifest_attestation)
    ):
        _fail("recurrent_replay_orphan_request_mismatch")
    samples, evidence = reconstruct_recurrent_package_inputs(
        document,
        store=store,
        task=request.task,
        independent_scorer=request.independent_scorer,
        tokenizer_trace_adapter=request.tokenizer_trace_adapter,
        campaign_trust_policy=request.campaign_trust_policy,
    )
    decoded = tuple(
        item.document["child_token_trace"]["generation"]["response_text"] for item in evidence
    )
    if decoded != tuple(request.completions):
        _fail("recurrent_replay_orphan_completion_mismatch")
    reward = validate_verified_transition_reward_batch(
        store,
        store.read_json(document["reward_artifact"], role="recurrent_replay_reward"),
        evidence,
        independent_scorer=request.independent_scorer,
        token_encoder=request.token_encoder,
        token_decoder=request.token_decoder,
    )
    admission = None
    journal = None
    if reward["optimizer_admitted"] is True:
        admission = validate_verified_transition_group_admission(
            store,
            store.read_json(
                document["group_admission_artifact"],
                role="recurrent_replay_group_admission",
            ),
            reward,
            evidence,
            samples,
            document["prompt_tokens"],
            group_manifest=document["group_manifest"],
            group_manifest_attestation=document["group_manifest_attestation"],
            independent_scorer=request.independent_scorer,
            token_encoder=request.token_encoder,
            token_decoder=request.token_decoder,
        )
        journal = VerifiedTransitionUpdateJournal.open(request.ledger_roots["updates"])
    elif (
        document["group_admission_artifact"] is not None
        or document["group_admission_sha256"] is not None
    ):
        _fail("recurrent_replay_orphan_rejection_invalid")
    start = request.campaign_ledger.group_start(sequence=request.sequence)
    return PreparedVerifiedTransitionGroup(
        campaign_sequence=request.sequence,
        transition_store=store,
        reward_receipt=reward,
        transition_evidence=evidence,
        group_manifest=document["group_manifest"],
        group_manifest_attestation=document["group_manifest_attestation"],
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
    existing_path = _package_path(replay_root, request.sequence)
    if existing_path.exists() or existing_path.is_symlink():
        return _prepared_from_existing_package(
            request,
            _read_package(replay_root, request.sequence),
            store=store,
        )
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
        "evidence_receipt_sha256s": [item.document["receipt_sha256"] for item in evidence],
        "reward_artifact": reward_artifact,
        "reward_receipt_sha256": reward["receipt_sha256"],
        "group_admission_artifact": admission_artifact,
        "group_admission_sha256": (admission["receipt_sha256"] if admission is not None else None),
        "group_manifest": manifest,
        "group_manifest_attestation": dict(request.group_manifest_attestation),
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
                expected_tokenizer_bundle_sha256=validated["tokenizer_bundle_sha256"],
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
    if (
        type(request.completed_groups) is not int
        or request.completed_groups < 0
        or not isinstance(request.step_receipts, tuple)
        or len(request.step_receipts) != request.completed_groups
    ):
        _fail("recurrent_campaign_completed_groups_invalid")
    replay_root = _private_root(request.replay_artifact_root)
    packages = tuple(
        _read_package(replay_root, sequence) for sequence in range(request.completed_groups)
    )
    replay_by_sequence = {group.sequence: group for group in request.replay_groups}
    package_rows: list[dict[str, Any]] = []
    terminal_policy_lineage: list[tuple[str, str]] = []
    producer_replay_bindings: dict[int, dict[str, str]] = {}
    updated_pre_measurements: list[str | None] = []
    transaction_store: VerifiedTransitionTransactionStore | None = None
    update_journal: VerifiedTransitionUpdateJournal | None = None
    latest_terminal = 0
    for sequence, (step, package) in enumerate(zip(request.step_receipts, packages, strict=True)):
        terminal = request.campaign_ledger.group_terminal_if_exists(sequence=sequence)
        if not isinstance(terminal, Mapping):
            _fail("recurrent_campaign_terminal_missing")
        finished_at = terminal.get("finished_at_unix_ns")
        if type(finished_at) is not int or finished_at <= 0:
            _fail("recurrent_campaign_terminal_time_invalid")
        latest_terminal = max(latest_terminal, finished_at)
        status = (
            "updated"
            if step.get("step_kind") == "verified_optimizer_update"
            else "rejected"
            if step.get("step_kind") == "verified_rejected_group"
            else None
        )
        if (
            status is None
            or step.get("campaign_sequence") != sequence
            or package["sequence"] != sequence
            or package["contract_sha256"] != request.contract_sha256
            or package["campaign_schedule_root_sha256"] != request.campaign_schedule_root_sha256
            or package["group_manifest"]["manifest_sha256"] != step.get("group_manifest_sha256")
            or package["reward_receipt_sha256"] != step.get("reward_receipt_sha256")
            or package["group_admission_sha256"] != step.get("group_admission_sha256")
            or terminal.get("status") != status
            or terminal.get("group_manifest_sha256") != step.get("group_manifest_sha256")
            or terminal.get("reward_receipt_sha256") != step.get("reward_receipt_sha256")
            or terminal.get("group_admission_sha256") != step.get("group_admission_sha256")
            or terminal.get("update_receipt_sha256") != step.get("update_receipt_sha256")
        ):
            _fail("recurrent_campaign_evidence_package_mismatch")
        terminal_policy_before = terminal.get("policy_before_sha256")
        terminal_policy_after = terminal.get("policy_after_sha256")
        if not isinstance(terminal_policy_before, str) or not isinstance(
            terminal_policy_after, str
        ):
            _fail("recurrent_campaign_terminal_policy_lineage_missing")
        terminal_policy_lineage.append((terminal_policy_before, terminal_policy_after))
        update_sha256 = step.get("update_receipt_sha256")
        replay_group = replay_by_sequence.get(sequence)
        if status == "updated":
            if (
                replay_group is None
                or replay_group.reward_receipt.get("receipt_sha256")
                != package["reward_receipt_sha256"]
                or replay_group.group_admission_receipt.get("receipt_sha256")
                != package["group_admission_sha256"]
                or replay_group.update_receipt.get("receipt_sha256") != update_sha256
            ):
                _fail("recurrent_campaign_updated_replay_mismatch")
            if transaction_store is None:
                transaction_store = VerifiedTransitionTransactionStore.open(
                    request.transaction_root
                )
            transaction = transaction_store.load(
                sequence=sequence,
                admission_sha256=cast(
                    str,
                    package["group_admission_sha256"],
                ),
                load_tensors=False,
            )
            if transaction is None:
                _fail("recurrent_campaign_update_transaction_missing")
            pre_measurement_sha256 = transaction.pending_step.get("pre_measurement_sha256")
            if pre_measurement_sha256 is not None:
                _sha256(
                    pre_measurement_sha256,
                    role="recurrent_campaign_pre_measurement",
                )
                intent = load_pre_measurement_for_transaction(
                    request.transaction_root,
                    sequence=sequence,
                    admission_sha256=cast(
                        str,
                        package["group_admission_sha256"],
                    ),
                    expected_receipt_sha256=pre_measurement_sha256,
                )
                if update_journal is None:
                    update_journal = VerifiedTransitionUpdateJournal.open(
                        request.update_journal_root
                    )
                objective = update_journal.read(
                    cast(str, package["group_admission_sha256"]),
                    "objective",
                )["objective_receipt"]
                producer_replay_bindings[sequence] = {
                    "state_source_sha256": intent["state_source"]["state_source_sha256"],
                    "post_state_transaction_stage_sha256": (transaction.stage["receipt_sha256"]),
                    "objective_receipt_sha256": _digest(objective),
                }
            updated_pre_measurements.append(pre_measurement_sha256)
        elif replay_group is not None or update_sha256 is not None:
            _fail("recurrent_campaign_rejected_replay_mismatch")
        package_rows.append(
            {
                "sequence": sequence,
                "status": status,
                "package_artifact": _package_artifact_binding(replay_root, sequence),
                "package_receipt_sha256": package["receipt_sha256"],
                "group_manifest_sha256": package["group_manifest"]["manifest_sha256"],
                "reward_receipt_sha256": package["reward_receipt_sha256"],
                "group_admission_sha256": package["group_admission_sha256"],
                "update_receipt_sha256": update_sha256,
                "pre_measurement_sha256": (pre_measurement_sha256 if status == "updated" else None),
                "trainer_step_receipt_sha256": step["receipt_sha256"],
                "sample_receipt_sha256s": package["sample_receipt_sha256s"],
                "evidence_receipt_sha256s": package["evidence_receipt_sha256s"],
            }
        )
    if any(value is not None for value in updated_pre_measurements) and any(
        value is None for value in updated_pre_measurements
    ):
        _fail("recurrent_campaign_mixed_pre_measurement_versions")
    evidence_schema = (
        CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V4
        if updated_pre_measurements and all(value is not None for value in updated_pre_measurements)
        else CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA
    )
    if evidence_schema == CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA:
        for row in package_rows:
            row.pop("pre_measurement_sha256")
    updated_sequences = [row["sequence"] for row in package_rows if row["status"] == "updated"]
    if sorted(replay_by_sequence) != updated_sequences:
        _fail("recurrent_campaign_replay_group_set_mismatch")
    existing_close = request.campaign_ledger.validate_closed_if_exists(
        policy=request.campaign_trust_policy
    )
    replay_contract_value = getattr(
        request,
        "policy_state_replay_contract",
        None,
    )
    if replay_contract_value is not None and not updated_sequences:
        inactive_replay_contract = validate_policy_state_replay_contract(
            replay_contract_value,
            verify_files=False,
            verify_model=False,
        )
        if (
            inactive_replay_contract["initial_policy_sha256"]
            != request.campaign_ledger.campaign_manifest()["initial_policy_sha256"]
        ):
            _fail("recurrent_campaign_policy_state_replay_contract_mismatch")
    if existing_close is not None:
        existing_payload = existing_close.get("close_payload")
        existing_evidence = (
            existing_payload.get("evidence_manifest")
            if isinstance(existing_payload, Mapping)
            else None
        )
        if not isinstance(existing_evidence, Mapping):
            _fail("recurrent_campaign_existing_close_invalid")
        if existing_evidence.get("schema") == CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V5:
            if replay_contract_value is None:
                _fail("recurrent_campaign_existing_replay_contract_missing")
            replay_contract = validate_policy_state_replay_contract(
                replay_contract_value,
                verify_files=False,
                verify_model=False,
            )
            existing_packages = existing_evidence.get("group_packages")
            if (
                existing_evidence.get("contract_sha256") != request.contract_sha256
                or existing_evidence.get("campaign_schedule_root_sha256")
                != request.campaign_schedule_root_sha256
                or existing_evidence.get("trust_policy_sha256")
                != request.campaign_trust_policy.policy_sha256
                or existing_evidence.get("completed_groups") != request.completed_groups
                or existing_evidence.get("halt_reason") != request.halt_reason
                or existing_evidence.get("policy_state_replay_contract_sha256")
                != replay_contract["contract_sha256"]
                or existing_evidence.get("updated_replay_sequences") != updated_sequences
                or not isinstance(existing_packages, list)
                or len(existing_packages) != len(package_rows)
                or any(
                    {key: existing_row.get(key) for key in package_row} != package_row
                    for existing_row, package_row in zip(
                        existing_packages,
                        package_rows,
                        strict=True,
                    )
                    if isinstance(existing_row, Mapping)
                )
                or any(not isinstance(existing_row, Mapping) for existing_row in existing_packages)
            ):
                _fail("recurrent_campaign_existing_close_mismatch")
            return VerifiedTransitionCampaignClosure(
                campaign_ledger=request.campaign_ledger,
                campaign_trust_policy=request.campaign_trust_policy,
            )
        if replay_contract_value is not None and updated_sequences:
            _fail("recurrent_campaign_existing_close_missing_policy_replay")
        existing_created_at = existing_evidence.get("created_at_unix_ns")
        replay_body = {
            "schema": evidence_schema,
            "contract_sha256": request.contract_sha256,
            "campaign_schedule_root_sha256": (request.campaign_schedule_root_sha256),
            "trust_policy_sha256": (request.campaign_trust_policy.policy_sha256),
            "campaign_ledger_root": request.campaign_ledger_root,
            "transition_artifact_root": request.transition_artifact_root,
            "update_journal_root": request.update_journal_root,
            "transaction_root": request.transaction_root,
            "completed_groups": request.completed_groups,
            "halt_reason": request.halt_reason,
            "group_packages": package_rows,
            "updated_replay_sequences": updated_sequences,
            "created_at_unix_ns": existing_created_at,
        }
        reconstructed = validate_causal_campaign_evidence_manifest(
            {
                **replay_body,
                "manifest_sha256": _digest(replay_body),
            }
        )
        if dict(existing_evidence) != reconstructed:
            _fail("recurrent_campaign_existing_close_mismatch")
        return VerifiedTransitionCampaignClosure(
            campaign_ledger=request.campaign_ledger,
            campaign_trust_policy=request.campaign_trust_policy,
        )
    completed_at = max(time.time_ns(), latest_terminal)
    evidence_body = {
        "schema": evidence_schema,
        "contract_sha256": request.contract_sha256,
        "campaign_schedule_root_sha256": (request.campaign_schedule_root_sha256),
        "trust_policy_sha256": request.campaign_trust_policy.policy_sha256,
        "campaign_ledger_root": request.campaign_ledger_root,
        "transition_artifact_root": request.transition_artifact_root,
        "update_journal_root": request.update_journal_root,
        "transaction_root": request.transaction_root,
        "completed_groups": request.completed_groups,
        "halt_reason": request.halt_reason,
        "group_packages": package_rows,
        "updated_replay_sequences": updated_sequences,
        "created_at_unix_ns": completed_at,
    }
    evidence_manifest = validate_causal_campaign_evidence_manifest(
        {
            **evidence_body,
            "manifest_sha256": _digest(evidence_body),
        }
    )
    signed_at = (completed_at + 999_999_999) // 1_000_000_000
    if replay_contract_value is not None and updated_sequences:
        if evidence_schema != CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V4:
            _fail("recurrent_campaign_policy_state_replay_not_applicable")
        replay_contract = validate_policy_state_replay_contract(
            replay_contract_value,
            verify_files=True,
            verify_model=False,
        )
        campaign_manifest = request.campaign_ledger.campaign_manifest()
        if replay_contract["initial_policy_sha256"] != campaign_manifest["initial_policy_sha256"]:
            _fail("recurrent_campaign_policy_state_replay_contract_mismatch")
        replay_executor = getattr(
            request.evidence_verifier_signer,
            "replay_policy_states",
            None,
        )
        if not callable(replay_executor):
            _fail("recurrent_campaign_policy_state_replay_executor_required")
        replay_request_body = {
            "schema": EXTERNAL_POLICY_STATE_REPLAY_REQUEST_SCHEMA,
            "purpose": EXTERNAL_POLICY_STATE_REPLAY_REQUEST_PURPOSE,
            "evidence_manifest": evidence_manifest,
            "policy_state_replay_contract": replay_contract,
            "campaign_trust_policy": {
                "document": request.campaign_trust_policy.document,
                "policy_sha256": (request.campaign_trust_policy.policy_sha256),
                "root_key_id": request.campaign_trust_policy.root_key_id,
            },
            "verifier_identity": request.evidence_verifier_signer.identity,
            "verified_at_unix": signed_at,
        }
        replay_request = {
            **replay_request_body,
            "request_sha256": _digest(replay_request_body),
        }
        replay_batch = _validate_policy_state_replay_batch(
            replay_executor(
                request=replay_request,
                timeout_seconds=replay_contract["external_verifier_max_seconds"],
            ),
            request=replay_request,
            contract=replay_contract,
            evidence_manifest=evidence_manifest,
        )
        results_by_sequence = {
            result["sequence"]: result for result in replay_batch["transition_results"]
        }
        v5_rows: list[dict[str, Any]] = []
        for row, (policy_before, policy_after) in zip(
            package_rows,
            terminal_policy_lineage,
            strict=True,
        ):
            if row["status"] == "updated":
                result = results_by_sequence[row["sequence"]]
                producer_binding = producer_replay_bindings[row["sequence"]]
                artifact = _publish_policy_state_replay_result(
                    replay_root,
                    sequence=row["sequence"],
                    result=result,
                )
                expected_transition = {
                    "provider_contract_sha256": request.contract_sha256,
                    "campaign_schedule_root_sha256": (request.campaign_schedule_root_sha256),
                    "campaign_manifest_sha256": campaign_manifest["manifest_sha256"],
                    "sequence": row["sequence"],
                    "group_manifest_sha256": row["group_manifest_sha256"],
                    "group_admission_sha256": row["group_admission_sha256"],
                    "update_receipt_sha256": row["update_receipt_sha256"],
                    "pre_measurement_sha256": row["pre_measurement_sha256"],
                    "state_source_sha256": producer_binding["state_source_sha256"],
                    "post_state_transaction_stage_sha256": producer_binding[
                        "post_state_transaction_stage_sha256"
                    ],
                    "objective_receipt_sha256": producer_binding["objective_receipt_sha256"],
                    "policy_before_sha256": policy_before,
                    "policy_after_sha256": policy_after,
                    "policy_state_replay_receipt_artifact": artifact,
                    "policy_state_replay_receipt_sha256": result["receipt_sha256"],
                }
                validated_result = validate_external_policy_state_replay_result(
                    result,
                    policy_state_replay_contract=replay_contract,
                    expected_transition=expected_transition,
                )
                if (
                    validated_result["verifier_identity"] != replay_request["verifier_identity"]
                    or validated_result["verified_at_unix"] != signed_at
                ):
                    _fail("recurrent_campaign_policy_state_replay_identity_mismatch")
                replay_fields = {
                    "objective_receipt_sha256": producer_binding["objective_receipt_sha256"],
                    "state_source_sha256": producer_binding["state_source_sha256"],
                    "post_state_transaction_stage_sha256": producer_binding[
                        "post_state_transaction_stage_sha256"
                    ],
                    "policy_state_replay_receipt_artifact": artifact,
                    "policy_state_replay_receipt_sha256": result["receipt_sha256"],
                }
            else:
                replay_fields = {
                    "objective_receipt_sha256": None,
                    "state_source_sha256": None,
                    "post_state_transaction_stage_sha256": None,
                    "policy_state_replay_receipt_artifact": None,
                    "policy_state_replay_receipt_sha256": None,
                }
            v5_rows.append(
                {
                    **row,
                    "policy_before_sha256": policy_before,
                    "policy_after_sha256": policy_after,
                    **replay_fields,
                }
            )
        receipt_root = _digest(
            {
                "policy_state_replay_receipts": [
                    {
                        "sequence": row["sequence"],
                        "receipt_sha256": row["policy_state_replay_receipt_sha256"],
                    }
                    for row in v5_rows
                    if row["status"] == "updated"
                ]
            }
        )
        v5_body = {
            **evidence_body,
            "schema": CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA_V5,
            "group_packages": v5_rows,
            "policy_state_replay_contract": replay_contract,
            "policy_state_replay_contract_sha256": replay_contract["contract_sha256"],
            "policy_state_replay_receipt_root_sha256": receipt_root,
        }
        evidence_manifest = validate_causal_campaign_evidence_manifest(
            {
                **v5_body,
                "manifest_sha256": _digest(v5_body),
            }
        )
    verifier = getattr(
        request.evidence_verifier_signer,
        "verify_evidence_manifest",
        None,
    )
    if not callable(verifier):
        _fail("recurrent_campaign_external_verifier_required")
    external_verification_receipt = verifier(
        request.campaign_trust_policy,
        evidence_manifest=evidence_manifest,
        verified_at_unix=signed_at,
        purpose="verified-recurrent-campaign-evidence-replay",
    )
    payload = request.campaign_ledger.close_payload(
        completed_at_unix_ns=completed_at,
        policy=request.campaign_trust_policy,
        evidence_manifest=evidence_manifest,
        external_evidence_verification_receipt=(external_verification_receipt),
    )
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
    "EXTERNAL_POLICY_STATE_REPLAY_BATCH_SCHEMA",
    "EXTERNAL_POLICY_STATE_REPLAY_REQUEST_PURPOSE",
    "EXTERNAL_POLICY_STATE_REPLAY_REQUEST_SCHEMA",
    "TOKEN_CODEC_ID",
    "VerifiedRecurrentTransitionRepositoryError",
    "load_recurrent_replay_packages",
    "finalize_verified_recurrent_transition_campaign",
    "produce_verified_recurrent_transition_group",
    "recurrent_trace_token_decoder",
    "recurrent_trace_token_encoder",
    "reconstruct_recurrent_package_inputs",
    "replay_recurrent_evidence_manifest_policy_states",
    "score_verified_recurrent_training_task",
    "validate_recurrent_replay_package",
    "verify_recurrent_evidence_manifest_artifacts",
]
