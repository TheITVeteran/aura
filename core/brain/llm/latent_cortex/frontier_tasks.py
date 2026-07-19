"""Deterministic, blinded task registry for RLC frontier evaluation.

The candidate receives only :class:`PublicTaskRecord` objects. Exact answers,
generation seeds, and commitment nonces remain inside a verifier-only payload.
This module deliberately performs no model execution and no filesystem I/O so
campaign orchestration can freeze, sign, and transport its outputs separately.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Any, NoReturn, cast

from core.brain.frontier_evidence_v5 import canonical_json_bytes

REGISTRY_SCHEMA = "aura.latent_cortex.frontier_task_registry.v1"
TASK_SCHEMA = "aura.latent_cortex.frontier_task.v1"
PUBLIC_TASK_SCHEMA = "aura.latent_cortex.frontier_public_task.v1"
ANSWER_PAYLOAD_SCHEMA = "aura.latent_cortex.frontier_answer_payload.v1"
TASK_MANIFEST_SCHEMA = "aura.latent_cortex.frontier_task_manifest.v1"
TASK_COMMITMENT_SCHEMA = "aura.latent_cortex.frontier_task_commitment.v1"
SCORE_RESULT_SCHEMA = "aura.latent_cortex.frontier_score_result.v1"
REGISTRY_VERSION = "2026.07.18.1"
CURRENT_REGISTRY_VERSION = "2026.07.18.2"
SCORER_VERSION = "1"

FRONTIER_DOMAINS = (
    "novel_algorithms",
    "mathematics",
    "coding",
    "scientific_inference",
    "long_horizon_planning",
    "calibration",
    "misleading_premise",
)
EXCLUDED_TRAINING_FAMILIES = ("boolean", "khop", "modular")
CURRENT_EXCLUDED_TRAINING_FAMILIES = (
    "khop",
    "boolean",
    "modular",
    "register_trace",
    "stack_trace",
    "constraint_order",
    "causal_intervention",
    "bayes_update",
    "budget_plan",
    "symbolic_rewrite",
    "premise_audit",
    "code_trace",
)
SUPPORTED_REGISTRY_EXCLUSIONS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        REGISTRY_VERSION: EXCLUDED_TRAINING_FAMILIES,
        CURRENT_REGISTRY_VERSION: CURRENT_EXCLUDED_TRAINING_FAMILIES,
    }
)
_ACTIVE_REGISTRY_VERSION: ContextVar[str] = ContextVar(
    "aura_frontier_registry_version",
    default=REGISTRY_VERSION,
)

FINAL_ANSWER_MARKER = "FINAL_ANSWER:"
MAX_PROMPT_BYTES = 12_000
MAX_RESPONSE_BYTES = 32_000
MAX_ANSWER_PAYLOAD_BYTES = 8_000
MAX_MANIFEST_TASKS = 10_000
MAX_JSON_DEPTH = 8
MAX_JSON_NODES = 256
MAX_JSON_STRING_BYTES = 2_048
MAX_SEED = (1 << 63) - 1

_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{1,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN = re.compile(r"[a-z0-9]+")
_DIFFICULTIES = {1, 2, 3}


class FrontierTaskError(ValueError):
    """Stable fail-closed error for invalid task or answer material."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> NoReturn:
    error = FrontierTaskError(code)
    raise error


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(payload: Any) -> str:
    return _sha256_bytes(canonical_json_bytes(payload))


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field}_invalid")
    return value


def _require_identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field}_invalid")
    return value


def _require_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        _fail("generation_seed_invalid")
    return seed


def _require_difficulty(difficulty: object) -> int:
    if (
        isinstance(difficulty, bool)
        or not isinstance(difficulty, int)
        or difficulty not in _DIFFICULTIES
    ):
        _fail("difficulty_invalid")
    return difficulty


def _require_registry_version(version: object) -> str:
    if not isinstance(version, str) or version not in SUPPORTED_REGISTRY_EXCLUSIONS:
        _fail("registry_version_unsupported")
    return version


def _active_registry_contract() -> tuple[str, tuple[str, ...]]:
    version = _require_registry_version(_ACTIVE_REGISTRY_VERSION.get())
    return version, SUPPORTED_REGISTRY_EXCLUSIONS[version]


def _require_prompt(prompt: object) -> str:
    if (
        not isinstance(prompt, str)
        or not prompt
        or prompt != prompt.strip()
        or "\x00" in prompt
        or len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES
    ):
        _fail("prompt_invalid")
    return prompt


def _seeded_rng(domain: str, seed: int, difficulty: int, *, stream: str) -> random.Random:
    registry_version, _exclusions = _active_registry_contract()
    material = canonical_json_bytes(
        {
            "registry_version": registry_version,
            "domain": domain,
            "seed": seed,
            "difficulty": difficulty,
            "stream": stream,
        }
    )
    return random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))


def _strict_json_loads(payload: bytes, *, role: str) -> Any:
    if len(payload) > MAX_ANSWER_PAYLOAD_BYTES:
        _fail(f"{role}_too_large")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        _fail(f"{role}_not_utf8")

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{role}_duplicate_key")
            result[key] = value
        return result

    def parse_int(raw: str) -> int:
        digits = raw.removeprefix("-")
        if not digits or len(digits) > 19:
            _fail(f"{role}_integer_out_of_bounds")
        value = int(raw)
        if not -(1 << 63) <= value <= (1 << 63) - 1:
            _fail(f"{role}_integer_out_of_bounds")
        return value

    def reject_float(_raw: str) -> float:
        _fail(f"{role}_floating_point_forbidden")

    def reject_constant(_raw: str) -> None:
        _fail(f"{role}_non_finite_number")

    try:
        return json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_int=parse_int,
            parse_float=reject_float,
            parse_constant=reject_constant,
        )
    except FrontierTaskError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError, TypeError):
        _fail(f"{role}_invalid_json")


def _validate_json_tree(value: Any, *, depth: int = 0, nodes: list[int] | None = None) -> None:
    if nodes is None:
        nodes = [0]
    nodes[0] += 1
    if nodes[0] > MAX_JSON_NODES:
        _fail("final_answer_too_complex")
    if depth > MAX_JSON_DEPTH:
        _fail("final_answer_too_deep")
    if value is None or type(value) in {bool, int}:
        return
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_JSON_STRING_BYTES or "\x00" in value:
            _fail("final_answer_string_invalid")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_tree(item, depth=depth + 1, nodes=nodes)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key.encode("utf-8")) > 128:
                _fail("final_answer_key_invalid")
            _validate_json_tree(item, depth=depth + 1, nodes=nodes)
        return
    _fail("final_answer_type_invalid")


def parse_final_answer(response: Any) -> dict[str, Any]:
    """Parse one terminal structured answer while rejecting ambiguous output."""

    if not isinstance(response, str) or not response.strip():
        _fail("response_missing")
    if len(response.encode("utf-8")) > MAX_RESPONSE_BYTES or "\x00" in response:
        _fail("response_invalid")
    if response.count(FINAL_ANSWER_MARKER) != 1:
        _fail("final_answer_marker_count_invalid")
    lines = response.rstrip().splitlines()
    if not lines or not lines[-1].startswith(FINAL_ANSWER_MARKER):
        _fail("final_answer_not_terminal_line")
    if any(FINAL_ANSWER_MARKER in line for line in lines[:-1]):
        _fail("final_answer_marker_count_invalid")
    encoded = lines[-1].removeprefix(FINAL_ANSWER_MARKER).strip()
    if not encoded:
        _fail("final_answer_missing")
    parsed = _strict_json_loads(encoded.encode("utf-8"), role="final_answer")
    if not isinstance(parsed, dict):
        _fail("final_answer_not_object")
    _validate_json_tree(parsed)
    return cast(dict[str, Any], parsed)


@dataclass(frozen=True, slots=True, repr=False)
class BlindedAnswerPayload:
    """Verifier-only canonical bytes committed before model evaluation."""

    commitment_sha256: str
    _canonical_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _require_sha256(self.commitment_sha256, field="answer_commitment_sha256")
        if (
            not isinstance(self._canonical_bytes, bytes)
            or not self._canonical_bytes
            or len(self._canonical_bytes) > MAX_ANSWER_PAYLOAD_BYTES
            or _sha256_bytes(self._canonical_bytes) != self.commitment_sha256
        ):
            _fail("answer_payload_commitment_mismatch")
        decoded = _strict_json_loads(self._canonical_bytes, role="answer_payload")
        if not isinstance(decoded, dict):
            _fail("answer_payload_not_object")
        if canonical_json_bytes(decoded) != self._canonical_bytes:
            _fail("answer_payload_noncanonical")

    def __repr__(self) -> str:
        return (
            "BlindedAnswerPayload(commitment_sha256="
            f"'{self.commitment_sha256}', payload=<redacted>)"
        )

    def reveal_for_verifier(self) -> dict[str, Any]:
        """Return a fresh decoded copy for an isolated correctness verifier."""

        value = _strict_json_loads(self._canonical_bytes, role="answer_payload")
        if not isinstance(value, dict):
            _fail("answer_payload_not_object")
        return cast(dict[str, Any], value)


@dataclass(frozen=True, slots=True)
class ContaminationFingerprint:
    method: str
    sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.method, field="contamination_method")
        _require_sha256(self.sha256, field="contamination_sha256")

    def to_dict(self) -> dict[str, str]:
        return {"method": self.method, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class PublicTaskRecord:
    """Candidate-visible immutable task description with no seed or answer."""

    schema: str
    registry_version: str
    task_id: str
    task_payload_sha256: str
    domain: str
    generator_id: str
    generator_version: str
    difficulty: int
    prompt: str
    response_contract: str
    scorer_id: str
    scorer_version: str
    answer_commitment_sha256: str
    contamination_fingerprints: tuple[ContaminationFingerprint, ...]
    excluded_training_families: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema != PUBLIC_TASK_SCHEMA:
            _fail("public_task_schema_invalid")
        registry_version = _require_registry_version(self.registry_version)
        if self.domain not in FRONTIER_DOMAINS:
            _fail("public_task_domain_invalid")
        _require_identifier(self.generator_id, field="generator_id")
        _require_identifier(self.scorer_id, field="scorer_id")
        _require_difficulty(self.difficulty)
        _require_prompt(self.prompt)
        if not isinstance(self.response_contract, str) or not self.response_contract:
            _fail("response_contract_invalid")
        _require_sha256(self.answer_commitment_sha256, field="answer_commitment_sha256")
        _require_sha256(self.task_payload_sha256, field="task_payload_sha256")
        if self.task_id != f"rlc_frontier:{self.domain}:{self.task_payload_sha256}":
            _fail("task_id_noncanonical")
        if (
            not self.contamination_fingerprints
            or len({item.method for item in self.contamination_fingerprints})
            != len(self.contamination_fingerprints)
            or self.contamination_fingerprints
            != tuple(sorted(self.contamination_fingerprints, key=lambda item: item.method))
        ):
            _fail("contamination_fingerprints_invalid")
        if self.excluded_training_families != SUPPORTED_REGISTRY_EXCLUSIONS[registry_version]:
            _fail("training_family_exclusions_invalid")
        if _sha256_json(self._body()) != self.task_payload_sha256:
            _fail("task_payload_hash_mismatch")

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "registry_version": self.registry_version,
            "domain": self.domain,
            "generator_id": self.generator_id,
            "generator_version": self.generator_version,
            "difficulty": self.difficulty,
            "prompt": self.prompt,
            "response_contract": self.response_contract,
            "scorer_id": self.scorer_id,
            "scorer_version": self.scorer_version,
            "answer_commitment_sha256": self.answer_commitment_sha256,
            "contamination_fingerprints": [
                item.to_dict() for item in self.contamination_fingerprints
            ],
            "excluded_training_families": list(self.excluded_training_families),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._body(),
            "task_id": self.task_id,
            "task_payload_sha256": self.task_payload_sha256,
        }


@dataclass(frozen=True, slots=True)
class FrontierTask:
    """Full immutable issuer/verifier task; never send this object to a model."""

    schema: str
    public: PublicTaskRecord
    blinded_answer: BlindedAnswerPayload = field(repr=False)

    def __post_init__(self) -> None:
        if self.schema != TASK_SCHEMA:
            _fail("task_schema_invalid")
        if self.blinded_answer.commitment_sha256 != self.public.answer_commitment_sha256:
            _fail("task_answer_commitment_mismatch")

    @property
    def task_id(self) -> str:
        return self.public.task_id

    @property
    def domain(self) -> str:
        return self.public.domain

    def reveal_for_verifier(self) -> dict[str, Any]:
        return self.blinded_answer.reveal_for_verifier()

    def score(self, response: Any) -> ScoreResult:
        return score_task(self, response)


Task = FrontierTask


@dataclass(frozen=True, slots=True)
class ScoreResult:
    schema: str
    task_id: str
    domain: str
    scorer_id: str
    parsed: bool
    correct: bool
    reason: str
    normalized_answer_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task_id": self.task_id,
            "domain": self.domain,
            "scorer_id": self.scorer_id,
            "parsed": self.parsed,
            "correct": self.correct,
            "reason": self.reason,
            "normalized_answer_sha256": self.normalized_answer_sha256,
        }


@dataclass(frozen=True, slots=True)
class TaskManifest:
    schema: str
    registry_version: str
    tasks: tuple[PublicTaskRecord, ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        if self.schema != TASK_MANIFEST_SCHEMA:
            _fail("task_manifest_schema_invalid")
        _require_registry_version(self.registry_version)
        if not self.tasks or len(self.tasks) > MAX_MANIFEST_TASKS:
            _fail("task_manifest_size_invalid")
        task_ids = [task.task_id for task in self.tasks]
        if task_ids != sorted(task_ids) or len(set(task_ids)) != len(task_ids):
            _fail("task_manifest_task_order_invalid")
        if any(task.registry_version != self.registry_version for task in self.tasks):
            _fail("task_manifest_registry_version_mismatch")
        _require_sha256(self.manifest_sha256, field="manifest_sha256")
        if _sha256_json(self._body()) != self.manifest_sha256:
            _fail("task_manifest_hash_mismatch")

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "registry_version": self.registry_version,
            "task_count": len(self.tasks),
            "domains": sorted({task.domain for task in self.tasks}),
            "tasks": [task.to_dict() for task in self.tasks],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "manifest_sha256": self.manifest_sha256}

    def canonical_bytes(self) -> bytes:
        return bytes(canonical_json_bytes(self.to_dict()))


@dataclass(frozen=True, slots=True)
class TaskCommitment:
    schema: str
    registry_version: str
    manifest_sha256: str
    task_count: int
    domain_counts: tuple[tuple[str, int], ...]
    task_ids_sha256: str
    answer_commitments_sha256: str
    contamination_corpus_sha256: str
    commitment_sha256: str

    def __post_init__(self) -> None:
        if self.schema != TASK_COMMITMENT_SCHEMA:
            _fail("task_commitment_schema_invalid")
        _require_registry_version(self.registry_version)
        if type(self.task_count) is not int or self.task_count <= 0:
            _fail("task_commitment_count_invalid")
        if (
            not self.domain_counts
            or self.domain_counts != tuple(sorted(self.domain_counts))
            or len({domain for domain, _count in self.domain_counts}) != len(self.domain_counts)
            or sum(count for _domain, count in self.domain_counts) != self.task_count
        ):
            _fail("task_commitment_domain_counts_invalid")
        for domain, count in self.domain_counts:
            if domain not in FRONTIER_DOMAINS or type(count) is not int or count <= 0:
                _fail("task_commitment_domain_counts_invalid")
        for field_name in (
            "manifest_sha256",
            "task_ids_sha256",
            "answer_commitments_sha256",
            "contamination_corpus_sha256",
            "commitment_sha256",
        ):
            _require_sha256(getattr(self, field_name), field=field_name)
        if _sha256_json(self._body()) != self.commitment_sha256:
            _fail("task_commitment_hash_mismatch")

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "registry_version": self.registry_version,
            "manifest_sha256": self.manifest_sha256,
            "task_count": self.task_count,
            "domain_counts": [
                {"domain": domain, "count": count} for domain, count in self.domain_counts
            ],
            "task_ids_sha256": self.task_ids_sha256,
            "answer_commitments_sha256": self.answer_commitments_sha256,
            "contamination_corpus_sha256": self.contamination_corpus_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "commitment_sha256": self.commitment_sha256}


def _normalized_prompt(prompt: str) -> str:
    return " ".join(_TOKEN.findall(prompt.lower()))


def _contamination_fingerprints(
    *, domain: str, generator_id: str, generator_version: str, prompt: str
) -> tuple[ContaminationFingerprint, ...]:
    registry_version, exclusions = _active_registry_contract()
    normalized = _normalized_prompt(prompt)
    tokens = normalized.split()
    shingles = [" ".join(tokens[index : index + 5]) for index in range(max(0, len(tokens) - 4))]
    values = (
        ContaminationFingerprint("prompt", _sha256_bytes(prompt.encode("utf-8"))),
        ContaminationFingerprint("normalized_prompt", _sha256_bytes(normalized.encode("utf-8"))),
        ContaminationFingerprint("token_fivegram_set", _sha256_json(sorted(set(shingles)))),
        ContaminationFingerprint(
            "generator_lineage",
            _sha256_json(
                {
                    "registry_version": registry_version,
                    "domain": domain,
                    "generator_id": generator_id,
                    "generator_version": generator_version,
                    "excluded_training_families": list(exclusions),
                }
            ),
        ),
    )
    return tuple(sorted(values, key=lambda item: item.method))


def _make_task(
    *,
    domain: str,
    seed: int,
    difficulty: int,
    generator_id: str,
    generator_version: str,
    prompt: str,
    response_contract: str,
    expected: Mapping[str, Any],
) -> FrontierTask:
    registry_version, exclusions = _active_registry_contract()
    _require_seed(seed)
    _require_difficulty(difficulty)
    _require_prompt(prompt)
    _require_identifier(generator_id, field="generator_id")
    scorer_id = f"score_{domain}"
    _require_identifier(scorer_id, field="scorer_id")
    expected_copy = json.loads(canonical_json_bytes(expected).decode("ascii"))
    blind_nonce = _sha256_json(
        {
            "purpose": "answer_blind",
            "registry_version": registry_version,
            "domain": domain,
            "seed": seed,
            "difficulty": difficulty,
            "expected": expected_copy,
        }
    )
    private_body = {
        "schema": ANSWER_PAYLOAD_SCHEMA,
        "registry_version": registry_version,
        "domain": domain,
        "generator_id": generator_id,
        "generator_version": generator_version,
        "scorer_id": scorer_id,
        "scorer_version": SCORER_VERSION,
        "generation_seed": seed,
        "difficulty": difficulty,
        "blind_nonce": blind_nonce,
        "expected": expected_copy,
    }
    answer_bytes = canonical_json_bytes(private_body)
    if len(answer_bytes) > MAX_ANSWER_PAYLOAD_BYTES:
        _fail("answer_payload_too_large")
    blinded = BlindedAnswerPayload(_sha256_bytes(answer_bytes), answer_bytes)
    fingerprints = _contamination_fingerprints(
        domain=domain,
        generator_id=generator_id,
        generator_version=generator_version,
        prompt=prompt,
    )
    body = {
        "schema": PUBLIC_TASK_SCHEMA,
        "registry_version": registry_version,
        "domain": domain,
        "generator_id": generator_id,
        "generator_version": generator_version,
        "difficulty": difficulty,
        "prompt": prompt,
        "response_contract": response_contract,
        "scorer_id": scorer_id,
        "scorer_version": SCORER_VERSION,
        "answer_commitment_sha256": blinded.commitment_sha256,
        "contamination_fingerprints": [item.to_dict() for item in fingerprints],
        "excluded_training_families": list(exclusions),
    }
    task_hash = _sha256_json(body)
    public = PublicTaskRecord(
        schema=PUBLIC_TASK_SCHEMA,
        registry_version=registry_version,
        task_id=f"rlc_frontier:{domain}:{task_hash}",
        task_payload_sha256=task_hash,
        domain=domain,
        generator_id=generator_id,
        generator_version=generator_version,
        difficulty=difficulty,
        prompt=prompt,
        response_contract=response_contract,
        scorer_id=scorer_id,
        scorer_version=SCORER_VERSION,
        answer_commitment_sha256=blinded.commitment_sha256,
        contamination_fingerprints=fingerprints,
        excluded_training_families=exclusions,
    )
    return FrontierTask(schema=TASK_SCHEMA, public=public, blinded_answer=blinded)


def build_task_manifest(tasks: Iterable[FrontierTask]) -> TaskManifest:
    try:
        bounded_tasks = tuple(itertools.islice(iter(tasks), MAX_MANIFEST_TASKS + 1))
    except TypeError:
        _fail("task_manifest_input_invalid")
    if (
        not bounded_tasks
        or len(bounded_tasks) > MAX_MANIFEST_TASKS
        or any(not isinstance(task, FrontierTask) for task in bounded_tasks)
    ):
        _fail("task_manifest_size_invalid")
    records = tuple(sorted((task.public for task in bounded_tasks), key=lambda item: item.task_id))
    versions = {record.registry_version for record in records}
    if len(versions) != 1:
        _fail("task_manifest_registry_version_mismatch")
    registry_version = versions.pop()
    body = {
        "schema": TASK_MANIFEST_SCHEMA,
        "registry_version": registry_version,
        "task_count": len(records),
        "domains": sorted({task.domain for task in records}),
        "tasks": [task.to_dict() for task in records],
    }
    return TaskManifest(
        schema=TASK_MANIFEST_SCHEMA,
        registry_version=registry_version,
        tasks=records,
        manifest_sha256=_sha256_json(body),
    )


def build_task_commitment(manifest: TaskManifest) -> TaskCommitment:
    if not isinstance(manifest, TaskManifest):
        _fail("task_manifest_type_invalid")
    counts = tuple(sorted(Counter(task.domain for task in manifest.tasks).items()))
    task_ids_sha256 = _sha256_json([task.task_id for task in manifest.tasks])
    answer_commitments_sha256 = _sha256_json(
        [task.answer_commitment_sha256 for task in manifest.tasks]
    )
    contamination_corpus_sha256 = _sha256_json(
        [
            {
                "task_id": task.task_id,
                "fingerprints": [item.to_dict() for item in task.contamination_fingerprints],
            }
            for task in manifest.tasks
        ]
    )
    body = {
        "schema": TASK_COMMITMENT_SCHEMA,
        "registry_version": manifest.registry_version,
        "manifest_sha256": manifest.manifest_sha256,
        "task_count": len(manifest.tasks),
        "domain_counts": [{"domain": domain, "count": count} for domain, count in counts],
        "task_ids_sha256": task_ids_sha256,
        "answer_commitments_sha256": answer_commitments_sha256,
        "contamination_corpus_sha256": contamination_corpus_sha256,
    }
    return TaskCommitment(
        schema=TASK_COMMITMENT_SCHEMA,
        registry_version=manifest.registry_version,
        manifest_sha256=manifest.manifest_sha256,
        task_count=len(manifest.tasks),
        domain_counts=counts,
        task_ids_sha256=task_ids_sha256,
        answer_commitments_sha256=answer_commitments_sha256,
        contamination_corpus_sha256=contamination_corpus_sha256,
        commitment_sha256=_sha256_json(body),
    )


def _response_instruction(type_contract: str) -> str:
    return (
        "You may reason before the answer. End with exactly one line beginning "
        f"{FINAL_ANSWER_MARKER}, followed by one JSON object and no trailing text. "
        f"Required JSON keys and value types: {type_contract}."
    )


def _generate_novel_algorithms(seed: int, difficulty: int) -> FrontierTask:
    rng = _seeded_rng("novel_algorithms", seed, difficulty, stream="instance")
    size = 5 + difficulty
    values = rng.sample(range(11, 99), size)
    indexed = list(enumerate(values))
    first = sorted(indexed, key=lambda item: (item[1], item[0]))[(size - 1) // 2]
    order = [first]
    remaining = [item for item in indexed if item != first]
    while remaining:
        current_value = order[-1][1]
        chosen = min(
            remaining,
            key=lambda item: (abs(item[1] - current_value), item[1], item[0]),
        )
        order.append(chosen)
        remaining.remove(chosen)
    sequence = [value for _index, value in order]
    checksum = sum((index + 1) * value for index, value in enumerate(sequence))
    prompt = (
        "Fresh algorithm task. The input values, in original position order, are "
        f"{values}. Select the lower median by numeric value first. Then repeatedly "
        "select one remaining value by minimizing, in order: absolute distance from "
        "the most recently selected value; numeric value; original zero-based "
        "position. Return the complete selected-value sequence. Its checksum is the "
        "sum of one-based output position multiplied by value. "
        + _response_instruction("sequence (list of integers), checksum (integer)")
    )
    return _make_task(
        domain="novel_algorithms",
        seed=seed,
        difficulty=difficulty,
        generator_id="nearest_value_traversal",
        generator_version="1",
        prompt=prompt,
        response_contract='{"sequence":list[int],"checksum":int}',
        expected={"sequence": sequence, "checksum": checksum},
    )


def _generate_mathematics(seed: int, difficulty: int) -> FrontierTask:
    rng = _seeded_rng("mathematics", seed, difficulty, stream="instance")
    size = 7 + difficulty
    choose = 3 if difficulty < 3 else 4
    values = sorted(rng.sample(range(2, 34), size))
    gap = rng.randint(2, 4)
    separated = [
        combo
        for combo in itertools.combinations(values, choose)
        if all(right - left >= gap for left, right in zip(combo, combo[1:], strict=False))
    ]
    if not separated:
        _fail("mathematics_generator_invariant_failed")
    sums = sorted(sum(combo) for combo in separated)
    lower = sums[len(sums) // 4]
    upper = sums[min(len(sums) - 1, (3 * len(sums)) // 4)]
    valid = [combo for combo in separated if lower <= sum(combo) <= upper]
    if not valid:
        _fail("mathematics_generator_invariant_failed")
    witness = list(min(valid))
    prompt = (
        f"Fresh combinatorics task. From the set {values}, choose exactly {choose} "
        f"distinct values. Adjacent values in sorted chosen order must differ by at "
        f"least {gap}, and the chosen sum must be from {lower} through {upper}, "
        "inclusive. Count all valid subsets and give the lexicographically smallest "
        "valid subset in ascending order. "
        + _response_instruction("count (integer), witness (list of integers)")
    )
    return _make_task(
        domain="mathematics",
        seed=seed,
        difficulty=difficulty,
        generator_id="separated_subset_count",
        generator_version="1",
        prompt=prompt,
        response_contract='{"count":int,"witness":list[int]}',
        expected={"count": len(valid), "witness": witness},
    )


def _audit_events(events: list[tuple[str, int]]) -> tuple[list[list[str | int]], list[int]]:
    balances: dict[str, int] = {}
    pressure: list[int] = []
    for name, delta in events:
        balances[name] = balances.get(name, 0) + delta
        if balances[name] == 0:
            del balances[name]
        pressure.append(sum(abs(value) for value in balances.values()))
    state: list[list[str | int]] = []
    for name in sorted(balances):
        state.append([name, balances[name]])
    return state, pressure


def _generate_coding(seed: int, difficulty: int) -> FrontierTask:
    rng = _seeded_rng("coding", seed, difficulty, stream="instance")
    names = ("ax", "by", "cz", "du")
    cases: list[list[tuple[str, int]]] = []
    returns: list[dict[str, Any]] = []
    for _case in range(2):
        events = [
            (rng.choice(names), rng.choice((-3, -2, -1, 1, 2, 3))) for _ in range(4 + difficulty)
        ]
        cases.append(events)
        state, pressure = _audit_events(events)
        returns.append({"state": state, "pressure": pressure})
    code = """def audit(events):
    balances = {}
    pressure = []
    for name, delta in events:
        balances[name] = balances.get(name, 0) + delta
        if balances[name] == 0:
            del balances[name]
        pressure.append(sum(abs(v) for v in balances.values()))
    return sorted(balances.items()), pressure"""
    prompt = (
        "Fresh code-semantics task. Evaluate this exact Python function without "
        f"executing it:\n\n{code}\n\nThe two inputs, in order, are "
        f"{cases}. Return each result as an object whose state is a JSON list of "
        "[name, value] pairs and whose pressure is a list. Also report the tight "
        "worst-case time complexity in n events, assuming dictionary operations "
        "are O(1). "
        + _response_instruction(
            "returns (list of objects with state and pressure), time_complexity (string)"
        )
    )
    return _make_task(
        domain="coding",
        seed=seed,
        difficulty=difficulty,
        generator_id="stateful_python_trace",
        generator_version="1",
        prompt=prompt,
        response_contract='{"returns":list[{"state":list[[str,int]],"pressure":list[int]}],"time_complexity":"O(n^2)"}',
        expected={"returns": returns, "time_complexity": "O(n^2)"},
    )


def _generate_scientific_inference(seed: int, difficulty: int) -> FrontierTask:
    rng = _seeded_rng("scientific_inference", seed, difficulty, stream="instance")
    labels = rng.sample(["aeron", "brin", "cressa", "dovin", "elara", "faron"], 3)
    root, mediator, downstream = labels
    root_base = rng.randint(6, 12)
    mediator_base = rng.randint(20, 30)
    downstream_base = rng.randint(50, 70)
    root_step = rng.randint(2, 4)
    mediator_gain = rng.randint(2, 4)
    downstream_gain = rng.randint(2, 3)
    mediator_step = rng.randint(2, 5)
    downstream_step = rng.randint(2, 5)
    query_step = rng.randint(3, 5)
    predicted = downstream_base + query_step * mediator_gain * downstream_gain
    prompt = (
        "Fresh causal-inference task. Three measured variables have baseline values "
        f"{root}={root_base}, {mediator}={mediator_base}, {downstream}={downstream_base}. "
        "Independent interventions produced these changes relative to baseline: "
        f"setting {root} up by {root_step} changed {mediator} by "
        f"+{root_step * mediator_gain} and {downstream} by "
        f"+{root_step * mediator_gain * downstream_gain}; setting {mediator} up by "
        f"{mediator_step} left {root} unchanged and changed {downstream} by "
        f"+{mediator_step * downstream_gain}; setting {downstream} up by "
        f"{downstream_step} left both other variables unchanged. Assume deterministic "
        "linear effects and no hidden common cause. Identify root, mediator, and "
        f"downstream variables, then predict the absolute value of {downstream} when "
        f"{root} is set {query_step} above baseline. "
        + _response_instruction(
            "root (string), mediator (string), downstream (string), predicted_downstream (integer)"
        )
    )
    return _make_task(
        domain="scientific_inference",
        seed=seed,
        difficulty=difficulty,
        generator_id="interventional_chain_inference",
        generator_version="1",
        prompt=prompt,
        response_contract='{"root":str,"mediator":str,"downstream":str,"predicted_downstream":int}',
        expected={
            "root": root,
            "mediator": mediator,
            "downstream": downstream,
            "predicted_downstream": predicted,
        },
    )


def _best_plan(tasks: list[dict[str, Any]], horizon: int) -> tuple[list[str], int, int]:
    by_name = {task["name"]: task for task in tasks}
    names = sorted(by_name)
    candidates: list[tuple[int, int, tuple[str, ...]]] = []
    for length in range(1, len(names) + 1):
        for order in itertools.permutations(names, length):
            selected = set(order)
            elapsed = 0
            reward = 0
            completed: set[str] = set()
            valid = True
            for name in order:
                task = by_name[name]
                if not set(task["requires"]).issubset(completed):
                    valid = False
                    break
                if not set(task["requires"]).issubset(selected):
                    valid = False
                    break
                elapsed += task["duration"]
                if elapsed > task["deadline"] or elapsed > horizon:
                    valid = False
                    break
                reward += task["reward"]
                completed.add(name)
            if valid:
                candidates.append((-reward, elapsed, order))
    if not candidates:
        _fail("planning_generator_invariant_failed")
    negative_reward, elapsed, order = min(candidates)
    return list(order), -negative_reward, elapsed


def _generate_long_horizon_planning(seed: int, difficulty: int) -> FrontierTask:
    rng = _seeded_rng("long_horizon_planning", seed, difficulty, stream="instance")
    count = 4 + difficulty
    names = [chr(ord("A") + index) for index in range(count)]
    horizon = 9 + difficulty * 3
    tasks: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        prior = names[:index]
        requires: list[str] = []
        if prior and rng.random() < 0.55:
            requires.append(rng.choice(prior))
        duration = rng.randint(1, 4)
        deadline = rng.randint(max(duration, 4), horizon)
        reward = rng.randint(3, 12)
        tasks.append(
            {
                "name": name,
                "duration": duration,
                "deadline": deadline,
                "reward": reward,
                "requires": requires,
            }
        )
    order, reward, makespan = _best_plan(tasks, horizon)
    prompt = (
        "Fresh planning task. One crew executes at most one task at a time, starts "
        "at time 0, and may skip tasks. A selected task may start only after every "
        "required task has completed. It earns its reward only if completion is no "
        f"later than its own deadline and the overall horizon {horizon}. Tasks are "
        f"{tasks}. Maximize total reward; then minimize makespan; then choose the "
        "lexicographically smallest task-name sequence. Return the selected order, "
        "reward, and makespan. "
        + _response_instruction("order (list of strings), reward (integer), makespan (integer)")
    )
    return _make_task(
        domain="long_horizon_planning",
        seed=seed,
        difficulty=difficulty,
        generator_id="dependency_deadline_portfolio",
        generator_version="1",
        prompt=prompt,
        response_contract='{"order":list[str],"reward":int,"makespan":int}',
        expected={"order": order, "reward": reward, "makespan": makespan},
    )


def _confidence_band(probability: Fraction) -> str:
    percent = float(probability) * 100.0
    if percent < 50.0:
        return "below_50"
    if percent < 70.0:
        return "50_to_69"
    if percent < 90.0:
        return "70_to_89"
    return "90_to_100"


def _generate_calibration(seed: int, difficulty: int) -> FrontierTask:
    rng = _seeded_rng("calibration", seed, difficulty, stream="instance")
    prior = Fraction(rng.randint(2, 7), 10)
    likelihood_h = Fraction(rng.randint(5, 9), 10)
    likelihood_not_h = Fraction(rng.randint(1, 5), 10)
    posterior = (likelihood_h * prior) / (likelihood_h * prior + likelihood_not_h * (1 - prior))
    choice = "H" if posterior >= Fraction(1, 2) else "not_H"
    prompt = (
        "Fresh calibration task. Before evidence E, hypothesis H has probability "
        f"{prior.numerator}/{prior.denominator}. The likelihood of E is "
        f"{likelihood_h.numerator}/{likelihood_h.denominator} if H is true and "
        f"{likelihood_not_h.numerator}/{likelihood_not_h.denominator} if H is false. "
        "Using exact Bayes updating, return the more probable choice (H wins ties), "
        "the reduced posterior probability of H, and its band: below_50, 50_to_69, "
        "70_to_89, or 90_to_100. "
        + _response_instruction(
            "choice (string), posterior (reduced-fraction string), confidence_band (string)"
        )
    )
    return _make_task(
        domain="calibration",
        seed=seed,
        difficulty=difficulty,
        generator_id="bayesian_frequency_update",
        generator_version="1",
        prompt=prompt,
        response_contract='{"choice":"H"|"not_H","posterior":"p/q","confidence_band":str}',
        expected={
            "choice": choice,
            "posterior": f"{posterior.numerator}/{posterior.denominator}",
            "confidence_band": _confidence_band(posterior),
        },
    )


def _generate_misleading_premise(seed: int, difficulty: int) -> FrontierTask:
    rng = _seeded_rng("misleading_premise", seed, difficulty, stream="instance")
    count = 4 + difficulty
    rows: list[dict[str, int | str]] = []
    for index in range(count):
        impact = rng.randint(4, 12)
        reliability = rng.randint(2, 6)
        cost = rng.randint(3, 18)
        rows.append(
            {
                "name": chr(ord("P") + index),
                "impact": impact,
                "reliability": reliability,
                "cost": cost,
                "score": impact * reliability - cost,
            }
        )
    winner = min(rows, key=lambda row: (-int(row["score"]), str(row["name"])))
    assertion_is_valid = seed % 3 == 0
    if assertion_is_valid:
        asserted = winner
    else:
        asserted = rng.choice([row for row in rows if row["name"] != winner["name"]])
    public_rows = [
        {key: row[key] for key in ("name", "impact", "reliability", "cost")} for row in rows
    ]
    prompt = (
        "Fresh premise-audit task. For each project, score = impact multiplied by "
        f"reliability minus cost. The data are {public_rows}. The claim says project "
        f"{asserted['name']} has the highest score. Check the premise instead of "
        "accepting it. Ties are won by the lexicographically smaller name. Return "
        "whether the claim is valid, the actual winner, and its score. "
        + _response_instruction(
            "premise_valid (boolean), actual_winner (string), actual_score (integer)"
        )
    )
    return _make_task(
        domain="misleading_premise",
        seed=seed,
        difficulty=difficulty,
        generator_id="premise_audit_table",
        generator_version="1",
        prompt=prompt,
        response_contract='{"premise_valid":bool,"actual_winner":str,"actual_score":int}',
        expected={
            "premise_valid": assertion_is_valid,
            "actual_winner": winner["name"],
            "actual_score": winner["score"],
        },
    )


Generator = Callable[[int, int], FrontierTask]
DOMAIN_GENERATORS: Mapping[str, Generator] = MappingProxyType(
    {
        "novel_algorithms": _generate_novel_algorithms,
        "mathematics": _generate_mathematics,
        "coding": _generate_coding,
        "scientific_inference": _generate_scientific_inference,
        "long_horizon_planning": _generate_long_horizon_planning,
        "calibration": _generate_calibration,
        "misleading_premise": _generate_misleading_premise,
    }
)


def _expected_payload(task: FrontierTask) -> dict[str, Any]:
    payload = task.reveal_for_verifier()
    required = {
        "schema",
        "registry_version",
        "domain",
        "generator_id",
        "generator_version",
        "scorer_id",
        "scorer_version",
        "generation_seed",
        "difficulty",
        "blind_nonce",
        "expected",
    }
    if set(payload) != required:
        _fail("answer_payload_schema_invalid")
    if (
        payload["schema"] != ANSWER_PAYLOAD_SCHEMA
        or payload["registry_version"] != task.public.registry_version
        or payload["domain"] != task.domain
        or payload["generator_id"] != task.public.generator_id
        or payload["generator_version"] != task.public.generator_version
        or payload["scorer_id"] != task.public.scorer_id
        or payload["scorer_version"] != task.public.scorer_version
        or payload["difficulty"] != task.public.difficulty
    ):
        _fail("answer_payload_binding_invalid")
    _require_seed(payload["generation_seed"])
    _require_sha256(payload["blind_nonce"], field="answer_blind_nonce")
    if not isinstance(payload["expected"], dict):
        _fail("answer_payload_expected_invalid")
    return cast(dict[str, Any], payload["expected"])


def _is_int_list(value: Any, *, minimum: int = 0, maximum: int = 64) -> bool:
    return (
        isinstance(value, list)
        and minimum <= len(value) <= maximum
        and all(type(item) is int for item in value)
    )


def _score_novel_algorithms(answer: dict[str, Any], expected: dict[str, Any]) -> bool:
    return (
        set(answer) == {"sequence", "checksum"}
        and _is_int_list(answer["sequence"], minimum=1)
        and type(answer["checksum"]) is int
        and answer == expected
    )


def _score_mathematics(answer: dict[str, Any], expected: dict[str, Any]) -> bool:
    return (
        set(answer) == {"count", "witness"}
        and type(answer["count"]) is int
        and answer["count"] >= 0
        and _is_int_list(answer["witness"])
        and answer == expected
    )


def _valid_coding_returns(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    for row in value:
        if not isinstance(row, dict) or set(row) != {"state", "pressure"}:
            return False
        if not isinstance(row["state"], list) or not _is_int_list(row["pressure"]):
            return False
        for pair in row["state"]:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not isinstance(pair[0], str)
                or type(pair[1]) is not int
            ):
                return False
    return True


def _score_coding(answer: dict[str, Any], expected: dict[str, Any]) -> bool:
    return (
        set(answer) == {"returns", "time_complexity"}
        and _valid_coding_returns(answer["returns"])
        and answer["time_complexity"] == "O(n^2)"
        and answer == expected
    )


def _score_scientific_inference(answer: dict[str, Any], expected: dict[str, Any]) -> bool:
    return (
        set(answer) == {"root", "mediator", "downstream", "predicted_downstream"}
        and all(isinstance(answer[key], str) for key in ("root", "mediator", "downstream"))
        and type(answer["predicted_downstream"]) is int
        and answer == expected
    )


def _score_long_horizon_planning(answer: dict[str, Any], expected: dict[str, Any]) -> bool:
    return (
        set(answer) == {"order", "reward", "makespan"}
        and isinstance(answer["order"], list)
        and all(isinstance(item, str) for item in answer["order"])
        and type(answer["reward"]) is int
        and type(answer["makespan"]) is int
        and answer == expected
    )


def _score_calibration(answer: dict[str, Any], expected: dict[str, Any]) -> bool:
    if set(answer) != {"choice", "posterior", "confidence_band"}:
        return False
    if answer["choice"] not in {"H", "not_H"} or not isinstance(answer["posterior"], str):
        return False
    if re.fullmatch(r"[1-9][0-9]*/[1-9][0-9]*", answer["posterior"]) is None:
        return False
    numerator, denominator = map(int, answer["posterior"].split("/"))
    if math.gcd(numerator, denominator) != 1 or numerator > denominator:
        return False
    return answer == expected


def _score_misleading_premise(answer: dict[str, Any], expected: dict[str, Any]) -> bool:
    return (
        set(answer) == {"premise_valid", "actual_winner", "actual_score"}
        and type(answer["premise_valid"]) is bool
        and isinstance(answer["actual_winner"], str)
        and type(answer["actual_score"]) is int
        and answer == expected
    )


Scorer = Callable[[dict[str, Any], dict[str, Any]], bool]
_SCORERS: Mapping[str, Scorer] = MappingProxyType(
    {
        "score_novel_algorithms": _score_novel_algorithms,
        "score_mathematics": _score_mathematics,
        "score_coding": _score_coding,
        "score_scientific_inference": _score_scientific_inference,
        "score_long_horizon_planning": _score_long_horizon_planning,
        "score_calibration": _score_calibration,
        "score_misleading_premise": _score_misleading_premise,
    }
)


def score_task(task: FrontierTask, response: Any) -> ScoreResult:
    """Score one response exactly without exposing expected values in receipts."""

    if not isinstance(task, FrontierTask):
        _fail("task_type_invalid")
    try:
        answer = parse_final_answer(response)
    except FrontierTaskError as exc:
        return ScoreResult(
            schema=SCORE_RESULT_SCHEMA,
            task_id=task.task_id,
            domain=task.domain,
            scorer_id=task.public.scorer_id,
            parsed=False,
            correct=False,
            reason=exc.code,
            normalized_answer_sha256=None,
        )
    expected = _expected_payload(task)
    scorer = _SCORERS.get(task.public.scorer_id)
    if scorer is None:
        _fail("scorer_missing")
    correct = scorer(answer, expected)
    return ScoreResult(
        schema=SCORE_RESULT_SCHEMA,
        task_id=task.task_id,
        domain=task.domain,
        scorer_id=task.public.scorer_id,
        parsed=True,
        correct=correct,
        reason="correct" if correct else "incorrect_or_schema_mismatch",
        normalized_answer_sha256=_sha256_json(answer),
    )


@dataclass(frozen=True, slots=True)
class FrontierTaskRegistry:
    schema: str = REGISTRY_SCHEMA
    version: str = REGISTRY_VERSION
    domains: tuple[str, ...] = FRONTIER_DOMAINS

    def __post_init__(self) -> None:
        if self.schema != REGISTRY_SCHEMA:
            _fail("registry_schema_invalid")
        _require_registry_version(self.version)
        if self.domains != FRONTIER_DOMAINS:
            _fail("registry_domains_invalid")

    def generate(self, domain: str, *, seed: int, difficulty: int = 2) -> FrontierTask:
        if not isinstance(domain, str) or domain not in DOMAIN_GENERATORS:
            _fail("domain_unknown")
        token = _ACTIVE_REGISTRY_VERSION.set(self.version)
        try:
            return DOMAIN_GENERATORS[domain](
                _require_seed(seed),
                _require_difficulty(difficulty),
            )
        finally:
            _ACTIVE_REGISTRY_VERSION.reset(token)

    def battery(
        self,
        seeds: Sequence[int],
        *,
        domains: Sequence[str] = FRONTIER_DOMAINS,
        difficulty: int = 2,
    ) -> tuple[FrontierTask, ...]:
        _require_difficulty(difficulty)
        if not isinstance(domains, Sequence) or isinstance(domains, (str, bytes)) or not domains:
            _fail("battery_domains_invalid")
        domain_values = tuple(domains)
        if (
            any(not isinstance(domain, str) for domain in domain_values)
            or len(set(domain_values)) != len(domain_values)
            or any(domain not in DOMAIN_GENERATORS for domain in domain_values)
        ):
            _fail("battery_domains_invalid")
        if (
            not isinstance(seeds, Sequence)
            or isinstance(seeds, (str, bytes))
            or not seeds
            or len(seeds) * len(domain_values) > MAX_MANIFEST_TASKS
        ):
            _fail("battery_size_invalid")
        seed_values = tuple(_require_seed(seed) for seed in seeds)
        if len(set(seed_values)) != len(seed_values):
            _fail("battery_duplicate_task")
        tasks = tuple(
            self.generate(domain, seed=seed, difficulty=difficulty)
            for seed in seed_values
            for domain in domain_values
        )
        if len({task.task_id for task in tasks}) != len(tasks):
            _fail("battery_duplicate_task")
        return tasks


DEFAULT_REGISTRY = FrontierTaskRegistry()
CURRENT_REGISTRY = FrontierTaskRegistry(version=CURRENT_REGISTRY_VERSION)


def _registry(version: str) -> FrontierTaskRegistry:
    normalized = _require_registry_version(version)
    return CURRENT_REGISTRY if normalized == CURRENT_REGISTRY_VERSION else DEFAULT_REGISTRY


def generate_task(
    domain: str,
    *,
    seed: int,
    difficulty: int = 2,
    registry_version: str = REGISTRY_VERSION,
) -> FrontierTask:
    return _registry(registry_version).generate(domain, seed=seed, difficulty=difficulty)


def generate_task_battery(
    seeds: Sequence[int],
    *,
    domains: Sequence[str] = FRONTIER_DOMAINS,
    difficulty: int = 2,
    registry_version: str = REGISTRY_VERSION,
) -> tuple[FrontierTask, ...]:
    return _registry(registry_version).battery(
        seeds,
        domains=domains,
        difficulty=difficulty,
    )


__all__ = [
    "ANSWER_PAYLOAD_SCHEMA",
    "BlindedAnswerPayload",
    "ContaminationFingerprint",
    "CURRENT_EXCLUDED_TRAINING_FAMILIES",
    "CURRENT_REGISTRY",
    "CURRENT_REGISTRY_VERSION",
    "DEFAULT_REGISTRY",
    "DOMAIN_GENERATORS",
    "EXCLUDED_TRAINING_FAMILIES",
    "FINAL_ANSWER_MARKER",
    "FRONTIER_DOMAINS",
    "FrontierTask",
    "FrontierTaskError",
    "FrontierTaskRegistry",
    "PUBLIC_TASK_SCHEMA",
    "PublicTaskRecord",
    "REGISTRY_SCHEMA",
    "REGISTRY_VERSION",
    "SCORE_RESULT_SCHEMA",
    "ScoreResult",
    "SUPPORTED_REGISTRY_EXCLUSIONS",
    "TASK_COMMITMENT_SCHEMA",
    "TASK_MANIFEST_SCHEMA",
    "TASK_SCHEMA",
    "Task",
    "TaskCommitment",
    "TaskManifest",
    "build_task_commitment",
    "build_task_manifest",
    "generate_task",
    "generate_task_battery",
    "parse_final_answer",
    "score_task",
]
