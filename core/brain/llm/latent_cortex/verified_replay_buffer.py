"""Encrypted, independently reconstructable replay for verified local repairs.

Only an answer that the host already accepted through the RLC local-repair and
confidence-bound replacement contracts can enter this buffer.  The public
store contains commitments and authenticated ciphertext, never prompt or
answer prose.  Corrupt stores fail closed instead of being replaced with an
empty ledger.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import stat
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from cryptography.exceptions import InvalidTag

from core.brain.llm.latent_cortex.answer_replacement import (
    MAX_REPLACEMENT_OUTPUT_TOKENS,
    validate_answer_replacement_receipt,
)
from core.brain.llm.latent_cortex.atomic_decomposition import (
    validate_atomic_decomposition,
)
from core.brain.llm.latent_cortex.deterministic_verifier_router import (
    validate_deterministic_router_envelope,
)
from core.brain.llm.latent_cortex.local_repair import (
    validate_local_repair_receipt,
)
from core.brain.llm.latent_cortex.persistence import (
    get_latent_cortex_persistence,
)
from core.runtime.atomic_writer import interprocess_file_lock
from core.runtime.file_read_gateway import read_stable_bytes
from core.runtime.service_registry import get_runtime_service
from core.runtime.state_ownership import state_root

VERIFIED_REPLAY_PAYLOAD_SCHEMA = "aura.rlc.verified_replay_payload.v1"
VERIFIED_REPLAY_ENTRY_SCHEMA = "aura.rlc.verified_replay_entry.v1"
VERIFIED_REPLAY_STORE_SCHEMA = "aura.rlc.verified_replay_store.v1"
VERIFIED_REPLAY_RECEIPT_SCHEMA = "aura.rlc.verified_replay_receipt.v1"

DEFAULT_MAX_ENTRIES = 1_024
DEFAULT_MAX_STORE_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_PRIVATE_BYTES = 256 * 1024
HARD_MAX_ENTRIES = 16_384
HARD_MAX_STORE_BYTES = 256 * 1024 * 1024
HARD_MAX_PRIVATE_BYTES = 1024 * 1024

_ZERO_SHA256 = "0" * 64
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_EXACT_VERIFIERS = {
    "exact_integer_arithmetic",
    "python_ast",
    "json_parser",
}
_ERROR_CLASSES = {
    "exact_integer_arithmetic": "reasoning.exact_integer_arithmetic",
    "python_ast": "structured_generation.python_syntax",
    "json_parser": "structured_generation.json_syntax",
}
_DISPOSITION = {
    "classification": "private_local_verified_repair",
    "storage": "aes_256_gcm_authenticated_encryption",
    "export_allowed": False,
    "remote_sync_allowed": False,
    "training_authority": "none_pending_independent_transfer_validation",
    "retention": "bounded_hash_chained_fifo",
    "governance_scope": "local_internal_governed_scope",
}
_KEY_PROVENANCE_RE = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")


class ReplayEncryptionUnavailableError(RuntimeError):
    """No authenticated local encryption provider is available."""


class ReplayStoreCorruptError(RuntimeError):
    """The durable replay ledger cannot be authenticated or reconstructed."""


class ReplayProtector(Protocol):
    """Minimum authenticated-encryption surface used by the replay ledger."""

    @property
    def encryption_active(self) -> bool: ...

    @property
    def key_provenance(self) -> str: ...

    def encrypt(self, data: bytes) -> bytes: ...

    def decrypt(self, blob: bytes) -> bytes: ...


def _canonical_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("verified replay value is not canonical JSON") from exc
    return rendered.encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_sha(value: Any, *, field: str) -> str:
    rendered = str(value or "")
    if _SHA256_RE.fullmatch(rendered) is None:
        raise ValueError(f"{field} is not a SHA-256 commitment")
    return rendered


def _bounded_int(value: Any, *, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _env_int(name: str, default: int, *, low: int, high: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    return _bounded_int(value, name=name, low=low, high=high)


def default_verified_replay_path() -> Path:
    override = os.environ.get("AURA_RLC_VERIFIED_REPLAY_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    try:
        from core.config import config

        home = Path(config.paths.home_dir)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        home = state_root()
    return home / "private" / "rlc" / "verified_replay_buffer.v1.json"


def _runtime_protector() -> ReplayProtector:
    protector = get_runtime_service("black_hole", default=None)
    if (
        protector is None
        or getattr(protector, "encryption_active", False) is not True
        or not callable(getattr(protector, "encrypt", None))
        or not callable(getattr(protector, "decrypt", None))
    ):
        raise ReplayEncryptionUnavailableError("BlackHole authenticated encryption is unavailable")
    provenance = str(getattr(protector, "key_provenance", "") or "")
    if not provenance or len(provenance) > 64:
        raise ReplayEncryptionUnavailableError("BlackHole key provenance is unavailable")
    return protector


def _route_for_ordinal(
    routes: Mapping[str, Any],
    ordinal: int,
) -> dict[str, Any]:
    rows = routes.get("routes")
    if not isinstance(rows, list) or not 0 <= ordinal < len(rows):
        raise ValueError("verified replay route ordinal is absent")
    row = rows[ordinal]
    if not isinstance(row, Mapping):
        raise ValueError("verified replay route is invalid")
    return dict(row)


def _fragment(
    candidate: str,
    atom: Mapping[str, Any],
) -> str:
    start = atom.get("start")
    end = atom.get("end")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(candidate):
        raise ValueError("verified replay atom span is invalid")
    fragment = candidate[start:end]
    if _text_sha(fragment) != atom.get("text_sha256"):
        raise ValueError("verified replay atom source commitment differs")
    return fragment


def extract_verified_replay_payload(
    *,
    receipt: Mapping[str, Any],
    private_evidence: Mapping[str, Any],
    objective: str,
    output_text: str,
    output_tokens: Sequence[int],
    output_quality: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct one verified correction from host-validated RLC evidence."""

    if not isinstance(receipt, Mapping):
        raise TypeError("verified replay episode receipt must be a mapping")
    if not isinstance(private_evidence, Mapping):
        raise TypeError("verified replay private evidence must be a mapping")
    if not isinstance(objective, str) or not isinstance(output_text, str):
        raise TypeError("verified replay objective and output must be text")
    tokens = list(output_tokens)
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("verified replay output tokens are invalid")
    if (
        not isinstance(output_quality, Mapping)
        or output_quality.get("passed") is not True
        or output_quality.get("text_sha256") != _text_sha(output_text)
        or output_quality.get("objective_sha256") != _text_sha(objective)
    ):
        raise ValueError("verified replay output-quality authority is absent")

    graph = receipt.get("disagreement_graph")
    selector = receipt.get("diagnostic_action_selection")
    local_repair = receipt.get("local_repair")
    replacement = receipt.get("answer_replacement")
    if not all(
        isinstance(value, Mapping) for value in (graph, selector, local_repair, replacement)
    ):
        raise ValueError("verified replay repair evidence is incomplete")
    validated_local = validate_local_repair_receipt(
        local_repair,
        disagreement_graph=graph,
        diagnostic_selection=selector,
    )
    policy = replacement.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("verified replay replacement policy is absent")
    validated_replacement = validate_answer_replacement_receipt(
        replacement,
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=validated_local,
        private_evidence=private_evidence,
        expected_objective=objective,
        expected_selected_branch=int(receipt.get("selected_branch")),
        expected_enabled=policy.get("enabled") is True,
        expected_margin=policy.get("margin"),
        expected_max_output_tokens=policy.get("max_output_tokens"),
        expected_output_text=output_text,
        expected_output_tokens=tokens,
    )
    if (
        validated_replacement["decision"] != "replace"
        or validated_replacement["answer_selection_effect"] != "replaced"
        or validated_replacement["accepted_output"]["source"] != "repaired_candidate"
        or validated_replacement["accepted_output"]["text_sha256"] != _text_sha(output_text)
        or validated_replacement["accepted_output"]["tokens_sha256"] != _sha(tokens)
    ):
        raise ValueError("verified replay requires an applied, output-bound repair")

    request_id = str(validated_replacement["selected_request_id"])
    requests = [
        row
        for row in validated_local["requests"]
        if isinstance(row, Mapping) and row.get("request_id") == request_id
    ]
    transactions = [
        row
        for row in validated_local["transactions"]
        if isinstance(row, Mapping) and row.get("request_id") == request_id
    ]
    candidate_rows = [
        row
        for row in validated_replacement["candidates"]
        if isinstance(row, Mapping) and row.get("request_id") == request_id
    ]
    if len(requests) != 1 or len(transactions) != 1 or len(candidate_rows) != 1:
        raise ValueError("verified replay selected repair is not unique")
    request = requests[0]
    transaction = transactions[0]
    candidate_row = candidate_rows[0]
    if (
        transaction.get("status") != "repaired_candidate_admitted"
        or transaction.get("failed_verifier_passed") is not True
        or transaction.get("no_exact_refutations") is not True
        or candidate_row.get("dominates") is not True
        or candidate_row.get("same_verifier_class") is not True
    ):
        raise ValueError("verified replay repair did not pass causal admission")

    branch = int(request["branch"])
    if branch != int(receipt.get("selected_branch")):
        raise ValueError("verified replay repair did not affect the selected branch")
    branch_texts = private_evidence.get("branch_candidates")
    generated = private_evidence.get("generated_repairs")
    if not isinstance(branch_texts, Mapping) or not isinstance(generated, Mapping):
        raise ValueError("verified replay private sources are absent")
    original_candidate = branch_texts.get(str(branch))
    corrected_candidate = generated.get(request_id)
    if (
        not isinstance(original_candidate, str)
        or not isinstance(corrected_candidate, str)
        or corrected_candidate != output_text
    ):
        raise ValueError("verified replay private output source differs")

    decompositions = graph.get("candidate_decompositions")
    candidate_routes = selector.get("candidate_routes")
    if not isinstance(decompositions, Mapping) or not isinstance(
        candidate_routes,
        Mapping,
    ):
        raise ValueError("verified replay source decomposition is absent")
    original_decomposition = validate_atomic_decomposition(
        decompositions[str(branch)],
        candidate=original_candidate,
        objective=objective,
    )
    original_routes = validate_deterministic_router_envelope(
        candidate_routes[str(branch)],
        atomic_receipt=original_decomposition,
    )
    corrected_decomposition = validate_atomic_decomposition(
        transaction["replacement_decomposition"],
        candidate=corrected_candidate,
        objective=objective,
    )
    corrected_routes = validate_deterministic_router_envelope(
        transaction["replacement_routes"],
        atomic_receipt=corrected_decomposition,
    )
    ordinal = int(request["failed_atom_ordinal"])
    if not (
        0 <= ordinal < len(original_decomposition["atoms"])
        and ordinal < len(corrected_decomposition["atoms"])
    ):
        raise ValueError("verified replay failed atom ordinal is absent")
    earlier_exact_refutations = [
        row
        for row in original_routes["routes"][:ordinal]
        if row["verifier"] in _EXACT_VERIFIERS and row["outcome"] == "refuted"
    ]
    if earlier_exact_refutations:
        raise ValueError("verified replay repair is not the earliest exact causal error")
    original_route = _route_for_ordinal(original_routes, ordinal)
    corrected_route = _route_for_ordinal(corrected_routes, ordinal)
    required_verifier = str(request["required_verifier"])
    if (
        required_verifier not in _ERROR_CLASSES
        or original_route["verifier"] != required_verifier
        or corrected_route["verifier"] != required_verifier
        or original_route["outcome"] != "refuted"
        or corrected_route["outcome"] != "verified"
    ):
        raise ValueError("verified replay discriminating test is not exact")

    original_atom = original_decomposition["atoms"][ordinal]
    corrected_atom = corrected_decomposition["atoms"][ordinal]
    original_fragment = _fragment(original_candidate, original_atom)
    corrected_fragment = _fragment(corrected_candidate, corrected_atom)
    prefix_end = int(original_atom["start"])
    prefix = original_candidate[:prefix_end]
    if (
        corrected_candidate[:prefix_end] != prefix
        or corrected_candidate == original_candidate
        or corrected_fragment == original_fragment
    ):
        raise ValueError("verified replay corrected transition is not discriminating")

    provenance = {
        "episode_id": str(receipt.get("episode_id") or ""),
        "input_tokens_sha256": _require_sha(
            receipt.get("input_tokens_sha256"),
            field="input_tokens_sha256",
        ),
        "checkpoint_fingerprint": _require_sha(
            receipt.get("checkpoint_fingerprint"),
            field="checkpoint_fingerprint",
        ),
        "checkpoint_fingerprint_method": str(receipt.get("checkpoint_fingerprint_method") or ""),
        "checkpoint_file_count": int(receipt.get("checkpoint_file_count")),
        "worker_identity_sha256": _sha(receipt.get("worker_identity")),
        "disagreement_graph_sha256": _require_sha(
            graph.get("receipt_sha256"),
            field="disagreement_graph_sha256",
        ),
        "diagnostic_selection_sha256": _require_sha(
            selector.get("receipt_sha256"),
            field="diagnostic_selection_sha256",
        ),
        "local_repair_sha256": _require_sha(
            validated_local.get("receipt_sha256"),
            field="local_repair_sha256",
        ),
        "answer_replacement_sha256": _require_sha(
            validated_replacement.get("receipt_sha256"),
            field="answer_replacement_sha256",
        ),
        "request_id": _require_sha(request_id, field="request_id"),
        "transaction_sha256": _require_sha(
            transaction.get("transaction_sha256"),
            field="transaction_sha256",
        ),
        "objective_sha256": _text_sha(objective),
        "output_quality_sha256": _sha(dict(output_quality)),
    }
    if (
        not provenance["episode_id"]
        or provenance["checkpoint_fingerprint_method"] != "sha256"
        or provenance["checkpoint_file_count"] <= 0
    ):
        raise ValueError("verified replay runtime provenance is incomplete")

    payload = {
        "schema": VERIFIED_REPLAY_PAYLOAD_SCHEMA,
        "task_context": {
            "objective": objective,
            "objective_sha256": _text_sha(objective),
        },
        "initial_failure": {
            "candidate": original_candidate,
            "baseline_decode": str(private_evidence.get("baseline_text") or ""),
            "failed_atom": {
                "atom_id": str(original_atom["atom_id"]),
                "ordinal": ordinal,
                "kind": str(original_atom["kind"]),
                "start": int(original_atom["start"]),
                "end": int(original_atom["end"]),
                "text": original_fragment,
            },
        },
        "earliest_causal_error": {
            "basis": "earliest_exact_refutation_on_selected_branch",
            "branch": branch,
            "atom_ordinal": ordinal,
            "atom_id": str(original_atom["atom_id"]),
            "last_valid_atom_id": str(request["last_valid_atom_id"]),
            "invalidated_atom_ids": list(request["invalidated_atom_ids"]),
            "invalidated_transition_ids": list(request["invalidated_transition_ids"]),
            "required_verifier": required_verifier,
            "earlier_exact_refutation_count": 0,
        },
        "discriminating_test": {
            "verifier": required_verifier,
            "original_route": original_route,
            "corrected_route": corrected_route,
            "original_outcome": "refuted",
            "corrected_outcome": "verified",
            "same_verifier_class": True,
            "passed": True,
        },
        "corrected_transition": {
            "candidate": corrected_candidate,
            "preserved_prefix": prefix,
            "replacement_suffix": corrected_candidate[prefix_end:],
            "corrected_atom": {
                "atom_id": str(corrected_atom["atom_id"]),
                "ordinal": ordinal,
                "kind": str(corrected_atom["kind"]),
                "start": int(corrected_atom["start"]),
                "end": int(corrected_atom["end"]),
                "text": corrected_fragment,
            },
        },
        "verified_solution": {
            "text": output_text,
            "tokens": tokens,
            "tokens_sha256": _sha(tokens),
            "output_quality": dict(output_quality),
        },
        "error_class": _ERROR_CLASSES[required_verifier],
        "escape_strategy": {
            "name": "regenerate_invalidated_suffix_from_last_verified_prefix",
            "fresh_context_required": True,
            "prefix_preserved": True,
            "unrelated_work_preserved": True,
            "same_verifier_rechecked": True,
        },
        "provenance": provenance,
        "privacy_governance_disposition": dict(_DISPOSITION),
    }
    return validate_verified_replay_payload(payload)


def validate_verified_replay_payload(value: Any) -> dict[str, Any]:
    """Validate private replay semantics before encryption or after decryption."""

    fields = {
        "schema",
        "task_context",
        "initial_failure",
        "earliest_causal_error",
        "discriminating_test",
        "corrected_transition",
        "verified_solution",
        "error_class",
        "escape_strategy",
        "provenance",
        "privacy_governance_disposition",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("verified replay private payload fields differ")
    if value["schema"] != VERIFIED_REPLAY_PAYLOAD_SCHEMA:
        raise ValueError("verified replay private payload schema differs")
    task_context = value["task_context"]
    initial = value["initial_failure"]
    error = value["earliest_causal_error"]
    test = value["discriminating_test"]
    corrected = value["corrected_transition"]
    solution = value["verified_solution"]
    strategy = value["escape_strategy"]
    provenance = value["provenance"]
    if not all(
        isinstance(item, Mapping)
        for item in (
            task_context,
            initial,
            error,
            test,
            corrected,
            solution,
            strategy,
            provenance,
        )
    ):
        raise ValueError("verified replay private payload sections are invalid")
    objective = task_context.get("objective")
    if (
        set(task_context) != {"objective", "objective_sha256"}
        or not isinstance(objective, str)
        or not objective.strip()
        or len(objective) > 131_072
        or task_context.get("objective_sha256") != _text_sha(objective)
    ):
        raise ValueError("verified replay task context differs")
    if value["privacy_governance_disposition"] != _DISPOSITION:
        raise ValueError("verified replay privacy disposition differs")
    original_candidate = initial.get("candidate")
    baseline_decode = initial.get("baseline_decode")
    original_atom = initial.get("failed_atom")
    corrected_candidate = corrected.get("candidate")
    corrected_atom = corrected.get("corrected_atom")
    if (
        not isinstance(original_candidate, str)
        or not isinstance(baseline_decode, str)
        or not isinstance(corrected_candidate, str)
        or not isinstance(original_atom, Mapping)
        or not isinstance(corrected_atom, Mapping)
        or len(original_candidate) > 131_072
        or len(corrected_candidate) > 131_072
        or len(baseline_decode) > 131_072
    ):
        raise ValueError("verified replay private text is invalid")
    for atom, candidate in (
        (original_atom, original_candidate),
        (corrected_atom, corrected_candidate),
    ):
        if set(atom) != {"atom_id", "ordinal", "kind", "start", "end", "text"}:
            raise ValueError("verified replay atom fields differ")
        start = atom["start"]
        end = atom["end"]
        if (
            type(atom["ordinal"]) is not int
            or not isinstance(atom["atom_id"], str)
            or not isinstance(atom["kind"], str)
            or type(start) is not int
            or type(end) is not int
            or not isinstance(atom["text"], str)
            or not 0 <= start < end <= len(candidate)
            or candidate[start:end] != atom["text"]
        ):
            raise ValueError("verified replay atom source differs")
    if (
        error.get("basis") != "earliest_exact_refutation_on_selected_branch"
        or error.get("earlier_exact_refutation_count") != 0
        or error.get("atom_ordinal") != original_atom["ordinal"]
        or error.get("atom_id") != original_atom["atom_id"]
        or error.get("required_verifier") != test.get("verifier")
        or test.get("original_outcome") != "refuted"
        or test.get("corrected_outcome") != "verified"
        or test.get("same_verifier_class") is not True
        or test.get("passed") is not True
        or not isinstance(test.get("original_route"), Mapping)
        or not isinstance(test.get("corrected_route"), Mapping)
        or test["original_route"].get("outcome") != "refuted"
        or test["corrected_route"].get("outcome") != "verified"
        or test["original_route"].get("verifier") != test.get("verifier")
        or test["corrected_route"].get("verifier") != test.get("verifier")
    ):
        raise ValueError("verified replay discriminating test differs")
    prefix = corrected.get("preserved_prefix")
    suffix = corrected.get("replacement_suffix")
    if (
        not isinstance(prefix, str)
        or not isinstance(suffix, str)
        or prefix + suffix != corrected_candidate
        or not original_candidate.startswith(prefix)
        or original_candidate == corrected_candidate
        or original_atom["text"] == corrected_atom["text"]
        or corrected_atom["ordinal"] != original_atom["ordinal"]
    ):
        raise ValueError("verified replay corrected transition differs")
    tokens = solution.get("tokens")
    quality = solution.get("output_quality")
    if (
        solution.get("text") != corrected_candidate
        or not isinstance(tokens, list)
        or not tokens
        or len(tokens) > MAX_REPLACEMENT_OUTPUT_TOKENS
        or any(type(token) is not int or token < 0 for token in tokens)
        or solution.get("tokens_sha256") != _sha(tokens)
        or not isinstance(quality, Mapping)
        or quality.get("passed") is not True
        or quality.get("text_sha256") != _text_sha(corrected_candidate)
    ):
        raise ValueError("verified replay verified solution differs")
    if value["error_class"] != _ERROR_CLASSES.get(str(test.get("verifier"))) or strategy != {
        "name": "regenerate_invalidated_suffix_from_last_verified_prefix",
        "fresh_context_required": True,
        "prefix_preserved": True,
        "unrelated_work_preserved": True,
        "same_verifier_rechecked": True,
    }:
        raise ValueError("verified replay escape strategy differs")
    provenance_fields = {
        "episode_id",
        "input_tokens_sha256",
        "checkpoint_fingerprint",
        "checkpoint_fingerprint_method",
        "checkpoint_file_count",
        "worker_identity_sha256",
        "disagreement_graph_sha256",
        "diagnostic_selection_sha256",
        "local_repair_sha256",
        "answer_replacement_sha256",
        "request_id",
        "transaction_sha256",
        "objective_sha256",
        "output_quality_sha256",
    }
    if (
        set(provenance) != provenance_fields
        or not str(provenance.get("episode_id") or "")
        or provenance.get("checkpoint_fingerprint_method") != "sha256"
        or type(provenance.get("checkpoint_file_count")) is not int
        or provenance["checkpoint_file_count"] <= 0
        or any(
            _SHA256_RE.fullmatch(str(provenance.get(field) or "")) is None
            for field in provenance_fields
            - {
                "episode_id",
                "checkpoint_fingerprint_method",
                "checkpoint_file_count",
            }
        )
        or quality.get("objective_sha256") != provenance.get("objective_sha256")
        or task_context["objective_sha256"] != provenance.get("objective_sha256")
        or _sha(dict(quality)) != provenance.get("output_quality_sha256")
    ):
        raise ValueError("verified replay provenance differs")
    return json.loads(_canonical_bytes(dict(value)).decode("utf-8"))


def _commitments(payload: Mapping[str, Any]) -> dict[str, str]:
    fields = (
        "task_context",
        "initial_failure",
        "earliest_causal_error",
        "discriminating_test",
        "corrected_transition",
        "verified_solution",
        "error_class",
        "escape_strategy",
        "provenance",
        "privacy_governance_disposition",
    )
    return {f"{field}_sha256": _sha(payload[field]) for field in fields}


def build_verified_replay_entry(
    payload: Mapping[str, Any],
    *,
    protector: ReplayProtector,
    sequence: int,
    previous_entry_sha256: str,
    created_at_unix_ns: int | None = None,
    max_private_bytes: int = DEFAULT_MAX_PRIVATE_BYTES,
) -> dict[str, Any]:
    private = validate_verified_replay_payload(payload)
    _bounded_int(
        max_private_bytes,
        name="max_private_bytes",
        low=4_096,
        high=HARD_MAX_PRIVATE_BYTES,
    )
    if getattr(protector, "encryption_active", False) is not True or not callable(
        getattr(protector, "encrypt", None)
    ):
        raise ReplayEncryptionUnavailableError("replay protector is not active")
    provenance = str(getattr(protector, "key_provenance", "") or "")
    if not provenance or len(provenance) > 64:
        raise ReplayEncryptionUnavailableError("replay protector provenance is absent")
    if _KEY_PROVENANCE_RE.fullmatch(provenance) is None:
        raise ReplayEncryptionUnavailableError(
            "replay protector provenance contains invalid characters"
        )
    _bounded_int(sequence, name="sequence", low=1, high=2**63 - 1)
    prior = _require_sha(previous_entry_sha256, field="previous_entry_sha256")
    created = time.time_ns() if created_at_unix_ns is None else created_at_unix_ns
    _bounded_int(created, name="created_at_unix_ns", low=1, high=2**63 - 1)
    plaintext = _canonical_bytes(private)
    if len(plaintext) > max_private_bytes:
        raise ValueError("verified replay private payload exceeds its hard bound")
    ciphertext = bytes(protector.encrypt(plaintext))
    if len(ciphertext) < 28 or len(ciphertext) > max_private_bytes + 64:
        raise ValueError("verified replay ciphertext size is invalid")
    encoded = base64.b64encode(ciphertext).decode("ascii")
    payload_row = {
        "schema": VERIFIED_REPLAY_ENTRY_SCHEMA,
        "sequence": sequence,
        "created_at_unix_ns": created,
        "previous_entry_sha256": prior,
        "experience_sha256": hashlib.sha256(plaintext).hexdigest(),
        "private_plaintext_bytes": len(plaintext),
        "commitments": _commitments(private),
        "encryption": {
            "algorithm": "AES-256-GCM",
            "key_provenance": provenance,
            "ciphertext_b64": encoded,
            "ciphertext_sha256": hashlib.sha256(ciphertext).hexdigest(),
        },
        "privacy_governance_disposition": dict(_DISPOSITION),
    }
    return {**payload_row, "entry_sha256": _sha(payload_row)}


def validate_verified_replay_entry(
    value: Any,
    *,
    expected_previous_entry_sha256: str | None = None,
) -> dict[str, Any]:
    fields = {
        "schema",
        "sequence",
        "created_at_unix_ns",
        "previous_entry_sha256",
        "experience_sha256",
        "private_plaintext_bytes",
        "commitments",
        "encryption",
        "privacy_governance_disposition",
        "entry_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("verified replay entry fields differ")
    row = dict(value)
    payload = {key: row[key] for key in fields - {"entry_sha256"}}
    if row["schema"] != VERIFIED_REPLAY_ENTRY_SCHEMA or row["entry_sha256"] != _sha(payload):
        raise ValueError("verified replay entry commitment differs")
    _bounded_int(row["sequence"], name="sequence", low=1, high=2**63 - 1)
    _bounded_int(
        row["created_at_unix_ns"],
        name="created_at_unix_ns",
        low=1,
        high=2**63 - 1,
    )
    prior = _require_sha(
        row["previous_entry_sha256"],
        field="previous_entry_sha256",
    )
    if expected_previous_entry_sha256 is not None and prior != (expected_previous_entry_sha256):
        raise ValueError("verified replay entry chain differs")
    _require_sha(row["experience_sha256"], field="experience_sha256")
    _bounded_int(
        row["private_plaintext_bytes"],
        name="private_plaintext_bytes",
        low=2,
        high=HARD_MAX_PRIVATE_BYTES,
    )
    expected_commitment_fields = {
        f"{field}_sha256"
        for field in (
            "initial_failure",
            "task_context",
            "earliest_causal_error",
            "discriminating_test",
            "corrected_transition",
            "verified_solution",
            "error_class",
            "escape_strategy",
            "provenance",
            "privacy_governance_disposition",
        )
    }
    commitments = row["commitments"]
    if (
        not isinstance(commitments, Mapping)
        or set(commitments) != expected_commitment_fields
        or any(_SHA256_RE.fullmatch(str(item)) is None for item in commitments.values())
    ):
        raise ValueError("verified replay entry commitments are invalid")
    encryption = row["encryption"]
    if (
        not isinstance(encryption, Mapping)
        or set(encryption)
        != {
            "algorithm",
            "key_provenance",
            "ciphertext_b64",
            "ciphertext_sha256",
        }
        or encryption["algorithm"] != "AES-256-GCM"
        or not isinstance(encryption["key_provenance"], str)
        or _KEY_PROVENANCE_RE.fullmatch(encryption["key_provenance"]) is None
        or not isinstance(encryption["ciphertext_b64"], str)
        or _SHA256_RE.fullmatch(str(encryption["ciphertext_sha256"])) is None
        or row["privacy_governance_disposition"] != _DISPOSITION
    ):
        raise ValueError("verified replay encryption metadata is invalid")
    try:
        ciphertext = base64.b64decode(encryption["ciphertext_b64"], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("verified replay ciphertext encoding is invalid") from exc
    if (
        base64.b64encode(ciphertext).decode("ascii") != encryption["ciphertext_b64"]
        or len(ciphertext) < 28
        or len(ciphertext) > HARD_MAX_PRIVATE_BYTES + 64
        or hashlib.sha256(ciphertext).hexdigest() != encryption["ciphertext_sha256"]
    ):
        raise ValueError("verified replay ciphertext commitment differs")
    return json.loads(_canonical_bytes(row).decode("utf-8"))


def materialize_verified_replay_entry(
    value: Mapping[str, Any],
    *,
    protector: ReplayProtector,
) -> dict[str, Any]:
    entry = validate_verified_replay_entry(value)
    if getattr(protector, "encryption_active", False) is not True or not callable(
        getattr(protector, "decrypt", None)
    ):
        raise ReplayEncryptionUnavailableError("replay protector is not active")
    protector_provenance = str(getattr(protector, "key_provenance", "") or "")
    if protector_provenance != entry["encryption"]["key_provenance"]:
        raise ReplayEncryptionUnavailableError(
            "replay protector provenance differs from the encrypted entry"
        )
    try:
        ciphertext = base64.b64decode(
            entry["encryption"]["ciphertext_b64"],
            validate=True,
        )
        plaintext = bytes(protector.decrypt(ciphertext))
    except (InvalidTag, binascii.Error, RuntimeError, TypeError, ValueError) as exc:
        raise ReplayStoreCorruptError(
            "verified replay ciphertext could not be authenticated"
        ) from exc
    if (
        len(plaintext) != entry["private_plaintext_bytes"]
        or hashlib.sha256(plaintext).hexdigest() != entry["experience_sha256"]
    ):
        raise ReplayStoreCorruptError("verified replay plaintext commitment differs")
    try:
        decoded = json.loads(plaintext.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ReplayStoreCorruptError("verified replay plaintext is not canonical JSON") from exc
    payload = validate_verified_replay_payload(decoded)
    if _canonical_bytes(payload) != plaintext or _commitments(payload) != entry["commitments"]:
        raise ReplayStoreCorruptError("verified replay private commitments differ")
    return payload


def _empty_store(*, max_entries: int, max_store_bytes: int) -> dict[str, Any]:
    payload = {
        "schema": VERIFIED_REPLAY_STORE_SCHEMA,
        "revision": 0,
        "entry_count": 0,
        "entries": [],
        "head_entry_sha256": _ZERO_SHA256,
        "retired_count": 0,
        "retired_tail_entry_sha256": _ZERO_SHA256,
        "retired_accumulator_sha256": _ZERO_SHA256,
        "limits": {
            "max_entries": max_entries,
            "max_store_bytes": max_store_bytes,
        },
    }
    return {**payload, "store_sha256": _sha(payload)}


def validate_verified_replay_store(value: Any) -> dict[str, Any]:
    fields = {
        "schema",
        "revision",
        "entry_count",
        "entries",
        "head_entry_sha256",
        "retired_count",
        "retired_tail_entry_sha256",
        "retired_accumulator_sha256",
        "limits",
        "store_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("verified replay store fields differ")
    row = dict(value)
    payload = {key: row[key] for key in fields - {"store_sha256"}}
    if row["schema"] != VERIFIED_REPLAY_STORE_SCHEMA or row["store_sha256"] != _sha(payload):
        raise ValueError("verified replay store commitment differs")
    revision = _bounded_int(
        row["revision"],
        name="revision",
        low=0,
        high=2**63 - 1,
    )
    retired_count = _bounded_int(
        row["retired_count"],
        name="retired_count",
        low=0,
        high=2**63 - 1,
    )
    limits = row["limits"]
    if not isinstance(limits, Mapping) or set(limits) != {
        "max_entries",
        "max_store_bytes",
    }:
        raise ValueError("verified replay retention limits differ")
    max_entries = _bounded_int(
        limits["max_entries"],
        name="max_entries",
        low=1,
        high=HARD_MAX_ENTRIES,
    )
    max_store_bytes = _bounded_int(
        limits["max_store_bytes"],
        name="max_store_bytes",
        low=4_096,
        high=HARD_MAX_STORE_BYTES,
    )
    entries = row["entries"]
    if (
        not isinstance(entries, list)
        or len(entries) > max_entries
        or row["entry_count"] != len(entries)
    ):
        raise ValueError("verified replay entry inventory differs")
    retired_tail = _require_sha(
        row["retired_tail_entry_sha256"],
        field="retired_tail_entry_sha256",
    )
    retired_accumulator = _require_sha(
        row["retired_accumulator_sha256"],
        field="retired_accumulator_sha256",
    )
    if retired_count == 0 and (retired_tail != _ZERO_SHA256 or retired_accumulator != _ZERO_SHA256):
        raise ValueError("verified replay empty retirement anchor differs")
    prior = retired_tail if retired_count else _ZERO_SHA256
    expected_sequence = retired_count + 1
    validated_entries: list[dict[str, Any]] = []
    seen_experiences: set[str] = set()
    for entry in entries:
        validated = validate_verified_replay_entry(
            entry,
            expected_previous_entry_sha256=prior,
        )
        if validated["sequence"] != expected_sequence:
            raise ValueError("verified replay sequence differs")
        if validated["experience_sha256"] in seen_experiences:
            raise ValueError("verified replay store contains duplicate experience")
        seen_experiences.add(validated["experience_sha256"])
        validated_entries.append(validated)
        prior = validated["entry_sha256"]
        expected_sequence += 1
    expected_head = prior if validated_entries or retired_count else _ZERO_SHA256
    if row["head_entry_sha256"] != expected_head:
        raise ValueError("verified replay head commitment differs")
    if revision < retired_count + len(validated_entries):
        raise ValueError("verified replay revision predates its entries")
    normalized = {
        **row,
        "entries": validated_entries,
        "limits": {
            "max_entries": max_entries,
            "max_store_bytes": max_store_bytes,
        },
    }
    if len(_canonical_bytes(normalized)) > max_store_bytes:
        raise ValueError("verified replay store exceeds its declared byte bound")
    return json.loads(_canonical_bytes(normalized).decode("utf-8"))


def validate_verified_replay_receipt(value: Any) -> dict[str, Any]:
    """Validate the public persistence result before it gains host authority."""

    fields = {
        "schema",
        "status",
        "experience_sha256",
        "entry_sha256",
        "sequence",
        "store_revision",
        "store_sha256",
        "entry_count",
        "retired_count",
        "persistence_transaction_id",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("verified replay receipt fields differ")
    row = dict(value)
    payload = {key: row[key] for key in fields - {"receipt_sha256"}}
    if (
        row["schema"] != VERIFIED_REPLAY_RECEIPT_SCHEMA
        or row["status"] not in {"stored", "duplicate"}
        or row["receipt_sha256"] != _sha(payload)
    ):
        raise ValueError("verified replay receipt commitment differs")
    for field in ("experience_sha256", "entry_sha256", "store_sha256"):
        _require_sha(row[field], field=field)
    _bounded_int(row["sequence"], name="sequence", low=1, high=2**63 - 1)
    _bounded_int(
        row["store_revision"],
        name="store_revision",
        low=1,
        high=2**63 - 1,
    )
    _bounded_int(
        row["entry_count"],
        name="entry_count",
        low=1,
        high=HARD_MAX_ENTRIES,
    )
    _bounded_int(
        row["retired_count"],
        name="retired_count",
        low=0,
        high=2**63 - 1,
    )
    transaction_id = row["persistence_transaction_id"]
    if (
        not isinstance(transaction_id, str)
        or len(transaction_id) > 128
        or (row["status"] == "stored" and not transaction_id)
        or (row["status"] == "duplicate" and transaction_id)
    ):
        raise ValueError("verified replay persistence transaction differs")
    return json.loads(_canonical_bytes(row).decode("utf-8"))


class VerifiedReplayBuffer:
    """Atomic bounded store for encrypted verified-repair experiences."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        max_entries: int | None = None,
        max_store_bytes: int | None = None,
        max_private_bytes: int | None = None,
    ) -> None:
        requested = default_verified_replay_path() if path is None else Path(path)
        expanded = requested.expanduser()
        self.path = expanded.parent.resolve(strict=False) / expanded.name
        self.max_entries = _bounded_int(
            (
                _env_int(
                    "AURA_RLC_VERIFIED_REPLAY_MAX_ENTRIES",
                    DEFAULT_MAX_ENTRIES,
                    low=1,
                    high=HARD_MAX_ENTRIES,
                )
                if max_entries is None
                else max_entries
            ),
            name="max_entries",
            low=1,
            high=HARD_MAX_ENTRIES,
        )
        self.max_store_bytes = _bounded_int(
            (
                _env_int(
                    "AURA_RLC_VERIFIED_REPLAY_MAX_BYTES",
                    DEFAULT_MAX_STORE_BYTES,
                    low=4_096,
                    high=HARD_MAX_STORE_BYTES,
                )
                if max_store_bytes is None
                else max_store_bytes
            ),
            name="max_store_bytes",
            low=4_096,
            high=HARD_MAX_STORE_BYTES,
        )
        self.max_private_bytes = _bounded_int(
            (
                _env_int(
                    "AURA_RLC_VERIFIED_REPLAY_MAX_PRIVATE_BYTES",
                    DEFAULT_MAX_PRIVATE_BYTES,
                    low=4_096,
                    high=HARD_MAX_PRIVATE_BYTES,
                )
                if max_private_bytes is None
                else max_private_bytes
            ),
            name="max_private_bytes",
            low=4_096,
            high=HARD_MAX_PRIVATE_BYTES,
        )

    @property
    def lock_path(self) -> Path:
        return self.path.parent / ".aura_file_write_batch.lock"

    def _read_locked(self) -> dict[str, Any]:
        if not os.path.lexists(self.path):
            return _empty_store(
                max_entries=self.max_entries,
                max_store_bytes=self.max_store_bytes,
            )
        try:
            before = self.path.lstat()
            if (
                not stat.S_ISREG(before.st_mode)
                or (hasattr(os, "getuid") and before.st_uid != os.getuid())
                or stat.S_IMODE(before.st_mode) & 0o077
                or before.st_size <= 0
                or before.st_size > HARD_MAX_STORE_BYTES
            ):
                raise ReplayStoreCorruptError(
                    "verified replay path is not an owner-private regular file"
                )
            raw = read_stable_bytes(
                self.path,
                max_bytes=HARD_MAX_STORE_BYTES,
            )
            after = self.path.lstat()
            if not stat.S_ISREG(after.st_mode) or (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                stat.S_IMODE(after.st_mode),
                after.st_uid,
            ) != (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                stat.S_IMODE(before.st_mode),
                before.st_uid,
            ):
                raise ReplayStoreCorruptError("verified replay store changed while reading")
            decoded = json.loads(raw.decode("utf-8"))
            store = validate_verified_replay_store(decoded)
        except ReplayStoreCorruptError:
            raise
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise ReplayStoreCorruptError(
                "refusing to overwrite an unreadable verified replay store"
            ) from exc
        if _canonical_bytes(store) != raw:
            raise ReplayStoreCorruptError("verified replay store is not canonical JSON")
        return store

    @staticmethod
    def _retire(
        store: dict[str, Any],
        entry: Mapping[str, Any],
    ) -> None:
        entry_sha = str(entry["entry_sha256"])
        store["retired_accumulator_sha256"] = _sha(
            {
                "prior_accumulator_sha256": store["retired_accumulator_sha256"],
                "retired_entry_sha256": entry_sha,
            }
        )
        store["retired_tail_entry_sha256"] = entry_sha
        store["retired_count"] += 1

    def append(
        self,
        payload: Mapping[str, Any],
        *,
        protector: ReplayProtector,
        created_at_unix_ns: int | None = None,
    ) -> dict[str, Any]:
        private = validate_verified_replay_payload(payload)
        experience_sha = hashlib.sha256(_canonical_bytes(private)).hexdigest()
        with interprocess_file_lock(self.lock_path):
            store = self._read_locked()
            duplicate = next(
                (
                    entry
                    for entry in store["entries"]
                    if entry["experience_sha256"] == experience_sha
                ),
                None,
            )
            if duplicate is not None:
                receipt_payload = {
                    "schema": VERIFIED_REPLAY_RECEIPT_SCHEMA,
                    "status": "duplicate",
                    "experience_sha256": experience_sha,
                    "entry_sha256": duplicate["entry_sha256"],
                    "sequence": duplicate["sequence"],
                    "store_revision": store["revision"],
                    "store_sha256": store["store_sha256"],
                    "entry_count": store["entry_count"],
                    "retired_count": store["retired_count"],
                    "persistence_transaction_id": "",
                }
                return validate_verified_replay_receipt(
                    {
                        **receipt_payload,
                        "receipt_sha256": _sha(receipt_payload),
                    }
                )

            entry = build_verified_replay_entry(
                private,
                protector=protector,
                sequence=store["retired_count"] + len(store["entries"]) + 1,
                previous_entry_sha256=store["head_entry_sha256"],
                created_at_unix_ns=created_at_unix_ns,
                max_private_bytes=self.max_private_bytes,
            )
            entries = [*store["entries"], entry]
            next_store = {
                "schema": VERIFIED_REPLAY_STORE_SCHEMA,
                "revision": store["revision"] + 1,
                "entry_count": len(entries),
                "entries": entries,
                "head_entry_sha256": entry["entry_sha256"],
                "retired_count": store["retired_count"],
                "retired_tail_entry_sha256": store["retired_tail_entry_sha256"],
                "retired_accumulator_sha256": store["retired_accumulator_sha256"],
                "limits": {
                    "max_entries": self.max_entries,
                    "max_store_bytes": self.max_store_bytes,
                },
            }
            while len(next_store["entries"]) > self.max_entries:
                retired = next_store["entries"].pop(0)
                self._retire(next_store, retired)
            candidate: dict[str, Any] | None = None
            for _attempt in range(len(next_store["entries"])):
                next_store["entry_count"] = len(next_store["entries"])
                next_store["head_entry_sha256"] = next_store["entries"][-1]["entry_sha256"]
                proposed = {
                    **next_store,
                    "store_sha256": _sha(next_store),
                }
                encoded = _canonical_bytes(proposed)
                if len(encoded) <= self.max_store_bytes:
                    candidate = proposed
                    break
                if len(next_store["entries"]) <= 1:
                    raise ValueError("verified replay entry cannot fit within the store byte bound")
                retired = next_store["entries"].pop(0)
                self._retire(next_store, retired)
            if candidate is None:
                raise ValueError("verified replay retention could not satisfy its byte bound")

            validated = validate_verified_replay_store(candidate)
            persistence_receipt = get_latent_cortex_persistence().save_verified_replay_buffer(
                self.path,
                _canonical_bytes(validated),
            )
            if self.path.read_bytes() != _canonical_bytes(validated):
                raise RuntimeError("verified replay durable readback differs")
            receipt_payload = {
                "schema": VERIFIED_REPLAY_RECEIPT_SCHEMA,
                "status": "stored",
                "experience_sha256": experience_sha,
                "entry_sha256": entry["entry_sha256"],
                "sequence": entry["sequence"],
                "store_revision": validated["revision"],
                "store_sha256": validated["store_sha256"],
                "entry_count": validated["entry_count"],
                "retired_count": validated["retired_count"],
                "persistence_transaction_id": persistence_receipt.transaction_id,
            }
            return validate_verified_replay_receipt(
                {
                    **receipt_payload,
                    "receipt_sha256": _sha(receipt_payload),
                }
            )

    def load(self) -> dict[str, Any]:
        with interprocess_file_lock(self.lock_path):
            return self._read_locked()

    def materialize(
        self,
        *,
        protector: ReplayProtector,
    ) -> list[dict[str, Any]]:
        store = self.load()
        return [
            materialize_verified_replay_entry(entry, protector=protector)
            for entry in store["entries"]
        ]


def persist_runtime_verified_replay(
    *,
    receipt: Mapping[str, Any],
    private_evidence: Mapping[str, Any],
    objective: str,
    output_text: str,
    output_tokens: Sequence[int],
    output_quality: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract and persist one live replay using Aura's runtime trust roots."""

    payload = extract_verified_replay_payload(
        receipt=receipt,
        private_evidence=private_evidence,
        objective=objective,
        output_text=output_text,
        output_tokens=output_tokens,
        output_quality=output_quality,
    )
    return VerifiedReplayBuffer().append(
        payload,
        protector=_runtime_protector(),
    )


__all__ = [
    "DEFAULT_MAX_ENTRIES",
    "DEFAULT_MAX_PRIVATE_BYTES",
    "DEFAULT_MAX_STORE_BYTES",
    "ReplayEncryptionUnavailableError",
    "ReplayProtector",
    "ReplayStoreCorruptError",
    "VERIFIED_REPLAY_ENTRY_SCHEMA",
    "VERIFIED_REPLAY_PAYLOAD_SCHEMA",
    "VERIFIED_REPLAY_RECEIPT_SCHEMA",
    "VERIFIED_REPLAY_STORE_SCHEMA",
    "VerifiedReplayBuffer",
    "build_verified_replay_entry",
    "default_verified_replay_path",
    "extract_verified_replay_payload",
    "materialize_verified_replay_entry",
    "persist_runtime_verified_replay",
    "validate_verified_replay_entry",
    "validate_verified_replay_payload",
    "validate_verified_replay_receipt",
    "validate_verified_replay_store",
]
