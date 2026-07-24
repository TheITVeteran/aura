"""Adversarial curriculum for recurrent process-error localization.

The curriculum consumes paired, deterministic RLC replays.  A clean replay
must pass a closed-form task oracle; a replay with one bounded intervention
must fail it.  The first changed recurrent transition becomes the only label.
An adaptive inserter then searches the verified candidate pool for subtle
errors the current locator misses.  Frozen calibration and OOD sets are never
used to choose mutations or update weights.

No model-authored assertion is accepted as a label.  Every retained negative
is bound to exact task coordinates, execution controls, reflector observation
hashes, independent task grading, and a tamper-evident append-only store.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.brain.llm.latent_cortex.bidirectional_reflector import (
    REFLECTOR_OBSERVATION_SCHEMA,
    validate_reflector_observation,
)
from core.learning.mistake_locator import (
    REFLECTOR_SKETCH_V1,
    MistakeLocatorHead,
    MistakeTransitionExample,
    evaluate_mistake_locator,
)
from core.learning.recurrence_curriculum import TASK_GENERATORS
from core.runtime.atomic_writer import (
    atomic_write_bytes_if_absent,
    ensure_private_directory,
    interprocess_file_lock,
)
from core.runtime.audit_chain import AuditChain

CURRICULUM_SCHEMA = "aura.rlc.adversarial_verifier_curriculum.v1"
CAPTURE_SCHEMA = "aura.rlc.adversarial_trace_capture.v1"
PAIR_SCHEMA = "aura.rlc.verified_adversarial_pair.v1"
PAIR_INPUT_SCHEMA = "aura.rlc.adversarial_pair_input.v1"
NEGATIVE_SCHEMA = "aura.rlc.verified_adversarial_negative.v1"
EVALUATION_REQUEST_SCHEMA = "aura.rlc.adversarial_evaluation_request.v1"
EVALUATION_RECEIPT_SCHEMA = "aura.rlc.adversarial_evaluation_receipt.v1"
INSERTER_SCHEMA = "aura.rlc.adaptive_inserter_policy.v1"
MAX_TRACE_STEPS = 64
MAX_PAIRS = 20_000
MAX_SANDBOX_BYTES = 32 * 1024 * 1024
MIN_PAIRS_PER_RELATION = 8
_RELATIONS = frozenset({"train", "in_domain", "out_of_domain"})
_EXAMPLE_FIELDS = frozenset(
    {
        "example_id",
        "trace_id",
        "task_id",
        "domain_id",
        "relation",
        "mutation_family",
        "transition_index",
        "transition_count",
        "error_index",
        "prior_hidden",
        "candidate_hidden",
        "trace_receipt_sha256",
        "outcome_verifier_id",
    }
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _identifier(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value.strip()) > 128
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{name} is invalid")
    return value.strip()


def _vector(value: Any, *, width: int | None = None) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("adversarial trace vector is invalid")
    result = tuple(round(float(item), 8) for item in value)
    if (
        not result
        or len(result) > 128
        or (width is not None and len(result) != width)
        or any(not math.isfinite(item) for item in result)
    ):
        raise ValueError("adversarial trace vector is invalid")
    return result


def _vector_sha256(value: Sequence[float]) -> str:
    return hashlib.sha256(
        ",".join(f"{float(item):.8f}" for item in value).encode("ascii")
    ).hexdigest()


def _observation(value: Any, *, expected_step: int, width: int | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("adversarial trace observation is not a mapping")
    row = validate_reflector_observation(value)
    prior = _vector(row.get("prior_sketch"), width=width)
    proposal = _vector(row.get("proposal_sketch"), width=len(prior))
    if (
        row.get("schema") != REFLECTOR_OBSERVATION_SCHEMA
        or row.get("branch_step") != expected_step
        or row.get("sketch_width") != len(prior)
        or not _is_sha256(row.get("prior_reasoning_sha256"))
        or not _is_sha256(row.get("proposal_reasoning_sha256"))
        or row.get("prior_sketch") != list(prior)
        or row.get("proposal_sketch") != list(proposal)
        or row.get("prior_sketch_sha256") != _vector_sha256(prior)
        or row.get("proposal_sketch_sha256") != _vector_sha256(proposal)
    ):
        raise ValueError("adversarial trace observation commitment is invalid")
    row["prior_sketch"] = list(prior)
    row["proposal_sketch"] = list(proposal)
    return row


@dataclass(frozen=True, slots=True)
class AdversarialTraceCapture:
    """One deterministic RLC replay plus its independently graded answer."""

    capture_id: str
    relation: str
    family: str
    depth: int
    seed: int
    execution_seed: int
    model_stack_sha256: str
    schedule_sha256: str
    config_sha256: str
    source_manifest_sha256: str
    answer: str
    observations: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "capture_id", _identifier(self.capture_id, name="capture_id"))
        object.__setattr__(self, "family", _identifier(self.family, name="family"))
        if self.relation not in _RELATIONS:
            raise ValueError("adversarial capture relation is invalid")
        if self.family not in TASK_GENERATORS:
            raise ValueError("adversarial capture task family is unknown")
        if (
            type(self.depth) is not int
            or not 1 <= self.depth <= 32
            or type(self.seed) is not int
            or self.seed < 0
            or type(self.execution_seed) is not int
            or self.execution_seed < 0
            or not all(
                _is_sha256(value)
                for value in (
                    self.model_stack_sha256,
                    self.schedule_sha256,
                    self.config_sha256,
                    self.source_manifest_sha256,
                )
            )
            or not isinstance(self.answer, str)
            or not self.answer.strip()
            or not 1 <= len(self.observations) <= MAX_TRACE_STEPS
        ):
            raise ValueError("adversarial capture identity/control is invalid")
        rows: list[dict[str, Any]] = []
        width: int | None = None
        for index, value in enumerate(self.observations):
            row = _observation(value, expected_step=index, width=width)
            width = len(row["prior_sketch"])
            rows.append(row)
        object.__setattr__(self, "observations", tuple(rows))

    @property
    def task(self):
        return TASK_GENERATORS[self.family](self.depth, self.seed)

    @property
    def task_id(self) -> str:
        return self.task.task_id

    @property
    def correct(self) -> bool:
        return bool(self.task.grade(self.answer)["correct"])

    @property
    def answer_sha256(self) -> str:
        return hashlib.sha256(self.answer.encode("utf-8")).hexdigest()

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_dict(include_answer=False))

    @property
    def execution_fingerprint(self) -> str:
        return _sha256(
            {
                "relation": self.relation,
                "task_id": self.task_id,
                "execution_seed": self.execution_seed,
                "model_stack_sha256": self.model_stack_sha256,
                "schedule_sha256": self.schedule_sha256,
                "config_sha256": self.config_sha256,
                "source_manifest_sha256": self.source_manifest_sha256,
                "answer_sha256": self.answer_sha256,
                "observations": [row["observation_sha256"] for row in self.observations],
            }
        )

    def to_dict(self, *, include_answer: bool = True) -> dict[str, Any]:
        payload = {
            "schema": CAPTURE_SCHEMA,
            "capture_id": self.capture_id,
            "relation": self.relation,
            "family": self.family,
            "depth": self.depth,
            "seed": self.seed,
            "task_id": self.task_id,
            "execution_seed": self.execution_seed,
            "model_stack_sha256": self.model_stack_sha256,
            "schedule_sha256": self.schedule_sha256,
            "config_sha256": self.config_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "answer_sha256": self.answer_sha256,
            "outcome_correct": self.correct,
            "observation_sha256s": [row["observation_sha256"] for row in self.observations],
            "observations": [dict(row) for row in self.observations],
        }
        if include_answer:
            payload["answer"] = self.answer
        return payload

    def to_input_dict(self) -> dict[str, Any]:
        return {
            "schema": CAPTURE_SCHEMA,
            "capture_id": self.capture_id,
            "relation": self.relation,
            "family": self.family,
            "depth": self.depth,
            "seed": self.seed,
            "execution_seed": self.execution_seed,
            "model_stack_sha256": self.model_stack_sha256,
            "schedule_sha256": self.schedule_sha256,
            "config_sha256": self.config_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "answer": self.answer,
            "observations": [dict(row) for row in self.observations],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AdversarialTraceCapture:
        fields = {
            "schema",
            "capture_id",
            "relation",
            "family",
            "depth",
            "seed",
            "execution_seed",
            "model_stack_sha256",
            "schedule_sha256",
            "config_sha256",
            "source_manifest_sha256",
            "answer",
            "observations",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != fields
            or value.get("schema") != CAPTURE_SCHEMA
        ):
            raise ValueError("adversarial capture fields differ")
        return cls(**{key: value[key] for key in fields - {"schema"}})


@dataclass(frozen=True, slots=True)
class VerifiedAdversarialPair:
    """A clean pass and a controlled, independently proven failing replay."""

    pair_id: str
    mutation_family: str
    error_index: int
    clean: AdversarialTraceCapture
    mutant: AdversarialTraceCapture
    clean_repeat: AdversarialTraceCapture
    mutant_repeat: AdversarialTraceCapture
    mutator_id: str = "adaptive_inserter_v1"
    max_relative_delta: float = 0.35

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair_id", _identifier(self.pair_id, name="pair_id"))
        object.__setattr__(
            self,
            "mutation_family",
            _identifier(self.mutation_family, name="mutation_family"),
        )
        object.__setattr__(self, "mutator_id", _identifier(self.mutator_id, name="mutator_id"))
        if (
            not isinstance(self.clean, AdversarialTraceCapture)
            or not isinstance(self.mutant, AdversarialTraceCapture)
            or not isinstance(self.clean_repeat, AdversarialTraceCapture)
            or not isinstance(self.mutant_repeat, AdversarialTraceCapture)
            or type(self.error_index) is not int
            or not 0 <= self.error_index < len(self.clean.observations)
            or isinstance(self.max_relative_delta, bool)
            or not 0.0 < float(self.max_relative_delta) <= 1.0
        ):
            raise ValueError("adversarial pair configuration is invalid")
        controls = (
            "relation",
            "family",
            "depth",
            "seed",
            "execution_seed",
            "model_stack_sha256",
            "schedule_sha256",
            "config_sha256",
            "source_manifest_sha256",
        )
        captures = (self.clean, self.mutant, self.clean_repeat, self.mutant_repeat)
        if any(
            getattr(self.clean, name) != getattr(capture, name)
            for capture in captures[1:]
            for name in controls
        ):
            raise ValueError("adversarial replay controls differ")
        if len({capture.capture_id for capture in captures}) != len(captures):
            raise ValueError("adversarial replay identities are not unique")
        if len(self.clean.observations) != len(self.mutant.observations):
            raise ValueError("adversarial replay trace lengths differ")
        if (
            self.clean.execution_fingerprint != self.clean_repeat.execution_fingerprint
            or self.mutant.execution_fingerprint != self.mutant_repeat.execution_fingerprint
        ):
            raise ValueError("adversarial replay failed repeat-determinism")
        if (
            not self.clean.correct
            or not self.clean_repeat.correct
            or self.mutant.correct
            or self.mutant_repeat.correct
        ):
            raise ValueError("adversarial pair lacks clean-pass/mutant-fail evidence")
        first_changed: int | None = None
        for index, (clean_row, mutant_row) in enumerate(
            zip(self.clean.observations, self.mutant.observations, strict=True)
        ):
            changed = (
                clean_row["prior_reasoning_sha256"] != mutant_row["prior_reasoning_sha256"]
                or clean_row["proposal_reasoning_sha256"] != mutant_row["proposal_reasoning_sha256"]
                or clean_row["prior_sketch"] != mutant_row["prior_sketch"]
                or clean_row["proposal_sketch"] != mutant_row["proposal_sketch"]
            )
            if changed and first_changed is None:
                first_changed = index
        clean_target = self.clean.observations[self.error_index]
        mutant_target = self.mutant.observations[self.error_index]
        if (
            first_changed != self.error_index
            or clean_target["prior_reasoning_sha256"] != mutant_target["prior_reasoning_sha256"]
            or clean_target["prior_sketch"] != mutant_target["prior_sketch"]
            or clean_target["proposal_sketch"] == mutant_target["proposal_sketch"]
            or self.relative_delta > float(self.max_relative_delta)
        ):
            raise ValueError("adversarial pair does not prove one subtle first divergence")

    @property
    def relative_delta(self) -> float:
        clean = self.clean.observations[self.error_index]["proposal_sketch"]
        mutant = self.mutant.observations[self.error_index]["proposal_sketch"]
        delta = math.sqrt(
            sum(
                (float(left) - float(right)) ** 2 for left, right in zip(clean, mutant, strict=True)
            )
            / len(clean)
        )
        scale = max(
            1e-6,
            math.sqrt(sum(float(item) ** 2 for item in clean) / len(clean)),
        )
        return round(delta / scale, 10)

    @property
    def pair_sha256(self) -> str:
        return _sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PAIR_SCHEMA,
            "pair_id": self.pair_id,
            "mutation_family": self.mutation_family,
            "error_index": self.error_index,
            "mutator_id": self.mutator_id,
            "max_relative_delta": float(self.max_relative_delta),
            "relative_delta": self.relative_delta,
            "task_id": self.clean.task_id,
            "clean_capture_id": self.clean.capture_id,
            "mutant_capture_id": self.mutant.capture_id,
            "clean_receipt_sha256": self.clean.receipt_sha256,
            "mutant_receipt_sha256": self.mutant.receipt_sha256,
            "clean_repeat_receipt_sha256": self.clean_repeat.receipt_sha256,
            "mutant_repeat_receipt_sha256": self.mutant_repeat.receipt_sha256,
            "clean_execution_fingerprint": self.clean.execution_fingerprint,
            "mutant_execution_fingerprint": self.mutant.execution_fingerprint,
            "clean_answer_sha256": self.clean.answer_sha256,
            "mutant_answer_sha256": self.mutant.answer_sha256,
            "independent_oracle": "recurrence_curriculum_exact_json",
        }

    def examples(self) -> tuple[list[MistakeTransitionExample], list[MistakeTransitionExample]]:
        def rows(capture: AdversarialTraceCapture, *, error_index: int | None, family: str):
            return [
                MistakeTransitionExample(
                    example_id=f"{capture.capture_id}-step-{index}",
                    trace_id=capture.capture_id,
                    task_id=capture.task_id,
                    domain_id=capture.family,
                    relation=capture.relation,
                    mutation_family=family,
                    transition_index=index,
                    transition_count=len(capture.observations),
                    error_index=error_index,
                    prior_hidden=tuple(observation["prior_sketch"]),
                    candidate_hidden=tuple(observation["proposal_sketch"]),
                    trace_receipt_sha256=capture.receipt_sha256,
                    outcome_verifier_id="recurrence_curriculum_exact_json:v1",
                )
                for index, observation in enumerate(capture.observations)
            ]

        return (
            rows(self.clean, error_index=None, family="clean_sham"),
            rows(self.mutant, error_index=self.error_index, family=self.mutation_family),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> VerifiedAdversarialPair:
        fields = {
            "schema",
            "pair_id",
            "mutation_family",
            "error_index",
            "mutator_id",
            "max_relative_delta",
            "clean",
            "mutant",
            "clean_repeat",
            "mutant_repeat",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != fields
            or value.get("schema") != PAIR_INPUT_SCHEMA
        ):
            raise ValueError("adversarial pair input fields differ")
        return cls(
            pair_id=value["pair_id"],
            mutation_family=value["mutation_family"],
            error_index=value["error_index"],
            mutator_id=value["mutator_id"],
            max_relative_delta=value["max_relative_delta"],
            clean=AdversarialTraceCapture.from_dict(value["clean"]),
            mutant=AdversarialTraceCapture.from_dict(value["mutant"]),
            clean_repeat=AdversarialTraceCapture.from_dict(value["clean_repeat"]),
            mutant_repeat=AdversarialTraceCapture.from_dict(value["mutant_repeat"]),
        )

    def to_input_dict(self) -> dict[str, Any]:
        return {
            "schema": PAIR_INPUT_SCHEMA,
            "pair_id": self.pair_id,
            "mutation_family": self.mutation_family,
            "error_index": self.error_index,
            "mutator_id": self.mutator_id,
            "max_relative_delta": float(self.max_relative_delta),
            "clean": self.clean.to_input_dict(),
            "mutant": self.mutant.to_input_dict(),
            "clean_repeat": self.clean_repeat.to_input_dict(),
            "mutant_repeat": self.mutant_repeat.to_input_dict(),
        }


class AdaptiveInserterPolicy:
    """Small learned policy that concentrates mutations where the critic misses."""

    def __init__(self, *, learning_rate: float = 0.25) -> None:
        if not 0.01 <= float(learning_rate) <= 1.0:
            raise ValueError("adaptive inserter learning rate is invalid")
        self.learning_rate = float(learning_rate)
        self._weights: dict[str, float] = {}

    @staticmethod
    def _cell(pair: VerifiedAdversarialPair) -> str:
        progress = (pair.error_index + 0.5) / len(pair.clean.observations)
        band = "early" if progress <= 1 / 3 else "middle" if progress <= 2 / 3 else "late"
        return f"{pair.mutation_family}:{band}"

    def score(self, pair: VerifiedAdversarialPair, *, miss_probability: float) -> float:
        cell = self._cell(pair)
        return self._weights.get(cell, 0.0) + float(miss_probability) - 0.2 * pair.relative_delta

    def update(self, pair: VerifiedAdversarialPair, *, miss_probability: float) -> None:
        cell = self._cell(pair)
        reward = max(-1.0, min(1.0, float(miss_probability) - pair.relative_delta))
        self._weights[cell] = round(
            max(-8.0, min(8.0, self._weights.get(cell, 0.0) + self.learning_rate * reward)),
            10,
        )

    def manifest(self) -> dict[str, Any]:
        payload = {
            "schema": INSERTER_SCHEMA,
            "learning_rate": self.learning_rate,
            "weights": dict(sorted(self._weights.items())),
        }
        return {**payload, "policy_sha256": _sha256(payload)}


class VerifiedNegativeStore:
    """Private, append-only, tamper-evident store of verified locator misses."""

    def __init__(self, root: str | Path) -> None:
        self.root = ensure_private_directory(Path(root).expanduser())
        self.records = ensure_private_directory(self.root / "records")
        self.chain = AuditChain(self.root / "chain")

    def _chain_record_ids(self) -> set[str]:
        if not self.chain.path.exists():
            return set()
        result: set[str] = set()
        for line in self.chain.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                result.add(str(json.loads(line)["receipt_id"]))
        return result

    def append(
        self,
        pair: VerifiedAdversarialPair,
        examples: Sequence[MistakeTransitionExample],
        *,
        error_probability: float,
        threshold: float,
        round_index: int,
    ) -> str:
        if pair.clean.relation != "train":
            raise ValueError("held-out evidence cannot enter the retained-negative store")
        _clean, expected_examples = pair.examples()
        if isinstance(examples, (str, bytes)) or [_example_payload(row) for row in examples] != [
            _example_payload(row) for row in expected_examples
        ]:
            raise ValueError("retained-negative examples do not match the verified pair")
        if not 0.0 <= float(error_probability) < float(threshold) <= 0.95:
            raise ValueError("retained negative was not a verified locator miss")
        if type(round_index) is not int or not 0 <= round_index < 16:
            raise ValueError("retained-negative round index is invalid")
        rows = [
            {
                **row.to_dict(),
                "prior_hidden": list(row.prior_hidden),
                "candidate_hidden": list(row.candidate_hidden),
            }
            for row in examples
        ]
        payload = {
            "schema": NEGATIVE_SCHEMA,
            "pair": pair.to_dict(),
            "pair_sha256": pair.pair_sha256,
            "error_probability": round(float(error_probability), 10),
            "threshold": round(float(threshold), 10),
            "round_index": int(round_index),
            "examples": rows,
        }
        record_id = f"negative-{_sha256(payload)}"
        body = {**payload, "record_id": record_id}
        raw = _canonical_bytes(body)
        path = self.records / f"{record_id}.json"
        with interprocess_file_lock(self.root / ".negative-store.lock"):
            published = atomic_write_bytes_if_absent(path, raw, durable=True, mode=0o600)
            if not published and path.read_bytes() != raw:
                raise RuntimeError("retained-negative identity collision")
            if record_id not in self._chain_record_ids():
                self.chain.append(
                    receipt_id=record_id,
                    kind="adversarial_verified_negative",
                    body=body,
                    timestamp=float(round_index),
                )
                self.chain.flush()
        return record_id

    def verify(self) -> tuple[bool, list[dict[str, Any]]]:
        bodies: dict[str, dict[str, Any]] = {}
        problems: list[dict[str, Any]] = []
        for path in sorted(self.records.glob("negative-*.json")):
            try:
                body = json.loads(path.read_text(encoding="ascii"))
                record_id = str(body["record_id"])
                expected = (
                    f"negative-{_sha256({key: body[key] for key in body if key != 'record_id'})}"
                )
                if record_id != expected or path.name != f"{record_id}.json":
                    raise ValueError("retained-negative content commitment differs")
                pair = body.get("pair")
                rows = body.get("examples")
                if (
                    body.get("schema") != NEGATIVE_SCHEMA
                    or not isinstance(pair, Mapping)
                    or pair.get("schema") != PAIR_SCHEMA
                    or body.get("pair_sha256") != _sha256(pair)
                    or not isinstance(rows, list)
                    or not rows
                    or not 0.0
                    <= float(body.get("error_probability"))
                    < float(body.get("threshold"))
                    <= 0.95
                    or type(body.get("round_index")) is not int
                    or not 0 <= body["round_index"] < 16
                ):
                    raise ValueError("retained-negative semantic envelope is invalid")
                reconstructed: list[MistakeTransitionExample] = []
                for row in rows:
                    example = MistakeTransitionExample(**{key: row[key] for key in _EXAMPLE_FIELDS})
                    if {
                        **example.to_dict(),
                        "prior_hidden": list(example.prior_hidden),
                        "candidate_hidden": list(example.candidate_hidden),
                    } != row:
                        raise ValueError("retained-negative example commitment differs")
                    reconstructed.append(example)
                first = reconstructed[0]
                if (
                    first.relation != "train"
                    or first.trace_id != pair.get("mutant_capture_id")
                    or first.task_id != pair.get("task_id")
                    or first.error_index != pair.get("error_index")
                    or first.mutation_family != pair.get("mutation_family")
                    or first.trace_receipt_sha256 != pair.get("mutant_receipt_sha256")
                    or any(row.trace_id != first.trace_id for row in reconstructed)
                ):
                    raise ValueError("retained-negative pair/example lineage differs")
                bodies[record_id] = body
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                problems.append({"path": str(path), "reason": str(exc)})
        try:
            chain_ids = self._chain_record_ids()
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            chain_ids = set()
            problems.append({"path": str(self.chain.path), "reason": str(exc)})
        ok, chain_problems = self.chain.verify(
            body_loader=lambda record_id, _kind: bodies.get(record_id)
        )
        problems.extend(chain_problems)
        extras = sorted(set(bodies) - chain_ids)
        if extras:
            problems.append({"reason": "records_missing_from_audit_chain", "record_ids": extras})
        return ok and not problems, problems

    def close(self) -> None:
        self.chain.close()


def _example_payload(row: MistakeTransitionExample) -> dict[str, Any]:
    return {
        "example_id": row.example_id,
        "trace_id": row.trace_id,
        "task_id": row.task_id,
        "domain_id": row.domain_id,
        "relation": row.relation,
        "mutation_family": row.mutation_family,
        "transition_index": row.transition_index,
        "transition_count": row.transition_count,
        "error_index": row.error_index,
        "prior_hidden": list(row.prior_hidden),
        "candidate_hidden": list(row.candidate_hidden),
        "trace_receipt_sha256": row.trace_receipt_sha256,
        "outcome_verifier_id": row.outcome_verifier_id,
    }


def _head_from_payload(payload: Mapping[str, Any]) -> MistakeLocatorHead:
    import numpy as np

    content = {key: payload[key] for key in payload if key != "content_sha256"}
    if payload.get("content_sha256") != _sha256(content):
        raise ValueError("sandbox head payload commitment differs")
    head = MistakeLocatorHead(
        means=np.asarray(payload["means"], dtype=np.float64),
        scales=np.asarray(payload["scales"], dtype=np.float64),
        input_weights=np.asarray(payload["input_weights"], dtype=np.float64),
        input_bias=np.asarray(payload["input_bias"], dtype=np.float64),
        output_weights=np.asarray(payload["output_weights"], dtype=np.float64),
        output_bias=float(payload["output_bias"]),
        temperature=float(payload["temperature"]),
        threshold=float(payload["threshold"]),
        manifest_data=dict(payload["manifest"]),
    )
    head.validate()
    return head


def _sandbox_worker(request: Mapping[str, Any]) -> dict[str, Any]:
    if request.get("schema") != EVALUATION_REQUEST_SCHEMA:
        raise ValueError("sandbox evaluation request schema differs")
    head = _head_from_payload(request["head"])
    examples = [MistakeTransitionExample(**row) for row in request["examples"]]
    evaluation = evaluate_mistake_locator(head, examples)
    payload = {
        "schema": EVALUATION_RECEIPT_SCHEMA,
        "request_sha256": _sha256(request),
        "evaluation": evaluation,
        "sandbox_contract": {
            "network": "denied",
            "file_write": "denied",
            "head_frozen": True,
            "training_examples_available": False,
        },
    }
    return {**payload, "receipt_sha256": _sha256(payload)}


def sandboxed_evaluate(
    head: MistakeLocatorHead,
    examples: Sequence[MistakeTransitionExample],
    *,
    repo_root: str | Path,
    python: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a frozen head in a no-network, no-write macOS sandbox."""

    sandbox = Path("/usr/bin/sandbox-exec")
    if sys.platform != "darwin" or not sandbox.is_file():
        raise RuntimeError("native sandbox-exec is required for held-out evaluation")
    request = {
        "schema": EVALUATION_REQUEST_SCHEMA,
        "head": head.to_payload(),
        "examples": [_example_payload(row) for row in examples],
    }
    raw = _canonical_bytes(request)
    if len(raw) > MAX_SANDBOX_BYTES:
        raise ValueError("sandbox evaluation request exceeds size bound")
    executable = str(python or sys.executable)
    profile = "(version 1) (allow default) (deny network*) (deny file-write*)"
    environment = {
        "HOME": str(Path.home()),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(Path(repo_root).resolve()),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
    }
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    result = get_subprocess_gateway().run(
        [str(sandbox), "-p", profile, executable, "-B", "-m", __name__, "--sandbox-worker"],
        cwd=repo_root,
        env=environment,
        timeout=60.0,
        capture_output=True,
        input=raw.decode("ascii"),
        offline_tooling=True,
        source="training_tooling:adversarial_verifier_evaluation",
    )
    if result.returncode != 0 or len(result.stdout.encode("utf-8")) > 2 * 1024 * 1024:
        raise RuntimeError(f"sandbox evaluation failed: {result.stderr[-500:]}")
    try:
        receipt = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("sandbox evaluation returned malformed JSON") from exc
    payload = {key: receipt[key] for key in receipt if key != "receipt_sha256"}
    if (
        receipt.get("schema") != EVALUATION_RECEIPT_SCHEMA
        or receipt.get("request_sha256") != _sha256(request)
        or receipt.get("receipt_sha256") != _sha256(payload)
        or receipt.get("evaluation") != evaluate_mistake_locator(head, examples)
    ):
        raise RuntimeError("sandbox evaluation receipt failed independent reconstruction")
    return receipt


@dataclass(frozen=True, slots=True)
class AdversarialCurriculumResult:
    head: MistakeLocatorHead
    report: dict[str, Any]


class AdversarialVerifierCurriculum:
    """Co-adapt a subtle-error inserter and a frozen-OOD process locator."""

    def __init__(self, *, rounds: int = 3, seed: int = 0) -> None:
        if type(rounds) is not int or not 1 <= rounds <= 16 or type(seed) is not int or seed < 0:
            raise ValueError("adversarial curriculum configuration is invalid")
        self.rounds = rounds
        self.seed = seed

    @staticmethod
    def _validate_pool(
        pairs: Sequence[VerifiedAdversarialPair],
    ) -> dict[str, list[VerifiedAdversarialPair]]:
        if (
            isinstance(pairs, (str, bytes))
            or not MIN_PAIRS_PER_RELATION * 3 <= len(pairs) <= MAX_PAIRS
            or any(not isinstance(pair, VerifiedAdversarialPair) for pair in pairs)
            or len({pair.pair_id for pair in pairs}) != len(pairs)
            or len({pair.pair_sha256 for pair in pairs}) != len(pairs)
        ):
            raise ValueError("adversarial pair pool is invalid")
        by_relation: dict[str, list[VerifiedAdversarialPair]] = defaultdict(list)
        for pair in pairs:
            by_relation[pair.clean.relation].append(pair)
        if set(by_relation) != _RELATIONS or any(
            len(rows) < MIN_PAIRS_PER_RELATION for rows in by_relation.values()
        ):
            raise ValueError("adversarial pair relation support is incomplete")
        task_sets = {
            relation: {pair.clean.task_id for pair in rows}
            for relation, rows in by_relation.items()
        }
        if any(
            left & right
            for index, left in enumerate(task_sets.values())
            for right in list(task_sets.values())[index + 1 :]
        ):
            raise ValueError("adversarial train/calibration/OOD tasks overlap")
        train_domains = {pair.clean.family for pair in by_relation["train"]}
        in_domains = {pair.clean.family for pair in by_relation["in_domain"]}
        out_domains = {pair.clean.family for pair in by_relation["out_of_domain"]}
        if train_domains != in_domains or train_domains & out_domains:
            raise ValueError("adversarial train/calibration/OOD domains are invalid")
        train_mutations = {pair.mutation_family for pair in by_relation["train"]}
        in_mutations = {pair.mutation_family for pair in by_relation["in_domain"]}
        out_mutations = {pair.mutation_family for pair in by_relation["out_of_domain"]}
        if train_mutations != in_mutations or train_mutations & out_mutations:
            raise ValueError("adversarial train/calibration/OOD mutation families are invalid")
        for relation, rows in by_relation.items():
            if (
                len({pair.clean.task_id for pair in rows}) < MIN_PAIRS_PER_RELATION
                or len({pair.mutation_family for pair in rows}) < 2
                or len({pair.clean.family for pair in rows}) < 2
            ):
                raise ValueError(f"adversarial {relation} support is insufficient")
        return {
            key: sorted(value, key=lambda pair: pair.pair_id) for key, value in by_relation.items()
        }

    @staticmethod
    def _examples(rows: Sequence[VerifiedAdversarialPair]) -> list[MistakeTransitionExample]:
        result: list[MistakeTransitionExample] = []
        clean_receipts: set[str] = set()
        for pair in rows:
            clean, mutant = pair.examples()
            if pair.clean.receipt_sha256 not in clean_receipts:
                result.extend(clean)
                clean_receipts.add(pair.clean.receipt_sha256)
            result.extend(mutant)
        return result

    @staticmethod
    def _miss_probability(head: MistakeLocatorHead | None, pair: VerifiedAdversarialPair) -> float:
        if head is None:
            return 0.5
        _clean, mutant = pair.examples()
        error = mutant[pair.error_index]
        return round(1.0 - head.probability(error.prior_hidden, error.candidate_hidden), 10)

    def run(
        self,
        pairs: Sequence[VerifiedAdversarialPair],
        *,
        repo_root: str | Path,
        negative_store: VerifiedNegativeStore | None = None,
        python: str | Path | None = None,
    ) -> AdversarialCurriculumResult:
        by_relation = self._validate_pool(pairs)
        frozen_in = by_relation["in_domain"]
        frozen_out = by_relation["out_of_domain"]
        frozen_in_hash = _sha256([pair.pair_sha256 for pair in frozen_in])
        frozen_out_hash = _sha256([pair.pair_sha256 for pair in frozen_out])
        train_by_task: dict[str, list[VerifiedAdversarialPair]] = defaultdict(list)
        for pair in by_relation["train"]:
            train_by_task[pair.clean.task_id].append(pair)
        policy = AdaptiveInserterPolicy()
        head: MistakeLocatorHead | None = None
        round_rows: list[dict[str, Any]] = []
        retained: set[str] = set()
        for round_index in range(self.rounds):
            selected: list[VerifiedAdversarialPair] = []
            mutation_families = sorted(
                {pair.mutation_family for rows in train_by_task.values() for pair in rows}
            )
            for task_index, task_id in enumerate(sorted(train_by_task)):
                preferred_family = mutation_families[task_index % len(mutation_families)]
                family_candidates = [
                    pair
                    for pair in train_by_task[task_id]
                    if pair.mutation_family == preferred_family
                ]
                candidates = family_candidates or train_by_task[task_id]
                selected.append(
                    max(
                        candidates,
                        key=lambda pair: (
                            policy.score(
                                pair,
                                miss_probability=self._miss_probability(head, pair),
                            ),
                            pair.pair_id,
                        ),
                    )
                )
            train_examples = self._examples(selected)
            head = MistakeLocatorHead.fit(
                train_examples,
                self._examples(frozen_in),
                self._examples(frozen_out),
                hidden_width=16,
                seed=self.seed + round_index,
                steps=800,
                input_representation=REFLECTOR_SKETCH_V1,
            )
            misses = 0
            evaluated_candidates = by_relation["train"]
            for pair in evaluated_candidates:
                _clean, mutant = pair.examples()
                error = mutant[pair.error_index]
                probability = head.probability(error.prior_hidden, error.candidate_hidden)
                miss_probability = 1.0 - probability
                policy.update(pair, miss_probability=miss_probability)
                if probability < head.threshold:
                    misses += 1
                    if negative_store is not None:
                        retained.add(
                            negative_store.append(
                                pair,
                                mutant,
                                error_probability=probability,
                                threshold=head.threshold,
                                round_index=round_index,
                            )
                        )
            round_rows.append(
                {
                    "round": round_index,
                    "selected_pair_sha256": _sha256([pair.pair_sha256 for pair in selected]),
                    "selected_count": len(selected),
                    "evaluated_candidate_count": len(evaluated_candidates),
                    "verified_locator_misses": misses,
                    "train_dataset_sha256": head.manifest()["train_dataset_sha256"],
                    "head_admitted": head.admitted,
                    "policy": policy.manifest(),
                }
            )
        assert head is not None
        sandbox_receipt = sandboxed_evaluate(
            head,
            self._examples(frozen_out),
            repo_root=repo_root,
            python=python,
        )
        store_verified = True
        store_problems: list[dict[str, Any]] = []
        if negative_store is not None:
            store_verified, store_problems = negative_store.verify()
        payload = {
            "schema": CURRICULUM_SCHEMA,
            "rounds": self.rounds,
            "seed": self.seed,
            "pair_pool_sha256": _sha256(sorted(pair.pair_sha256 for pair in pairs)),
            "frozen_in_domain_sha256": frozen_in_hash,
            "frozen_out_of_domain_sha256": frozen_out_hash,
            "heldout_frozen_before_training": True,
            "heldout_used_for_weight_updates": False,
            "input_representation": REFLECTOR_SKETCH_V1,
            "independent_oracle": "recurrence_curriculum_exact_json",
            "source_manifest_commitment_required": True,
            "repeat_determinism_required": True,
            "external_trust_root_required_for_resident_promotion": True,
            "round_history": round_rows,
            "final_head_content_sha256": head.to_payload()["content_sha256"],
            "final_head_admitted": head.admitted,
            "sandbox_evaluation": sandbox_receipt,
            "retained_negative_count": len(retained),
            "retained_negative_store_verified": store_verified,
            "retained_negative_store_problems": store_problems,
            "claims": {
                "adversarial_curriculum_mechanics": True,
                "resident_32b_artifact_trained": False,
                "resident_source_manifest_signature_verified": False,
                "resident_32b_reasoning_gain": False,
                "frontier_capability": False,
            },
        }
        return AdversarialCurriculumResult(
            head=head,
            report={**payload, "report_sha256": _sha256(payload)},
        )


def _main() -> int:
    if sys.argv[1:] != ["--sandbox-worker"]:
        return 2
    request = json.loads(sys.stdin.read(MAX_SANDBOX_BYTES + 1))
    if len(_canonical_bytes(request)) > MAX_SANDBOX_BYTES:
        raise ValueError("sandbox request exceeds size bound")
    print(json.dumps(_sandbox_worker(request), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "AdversarialCurriculumResult",
    "AdversarialTraceCapture",
    "AdversarialVerifierCurriculum",
    "AdaptiveInserterPolicy",
    "VerifiedAdversarialPair",
    "VerifiedNegativeStore",
    "sandboxed_evaluate",
]
