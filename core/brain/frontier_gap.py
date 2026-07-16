"""Auditable gap-to-frontier measurement primitives.

Candidate correctness and frontier parity are separate scientific objects.
This module can always score a candidate against deterministic task truth, but
it computes a frontier gap only when a named, matched-budget reference artifact
passes the complete provenance contract.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import logging
import math
import random
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from core.brain.frontier_evidence_v5 import (
    MATCHED_BUDGET,
    PROTOCOL_MANIFEST,
    PROTOCOL_MANIFEST_SHA256,
    RUN_ENVELOPE_SCHEMA,
    analyze_gap_trend,
    identity_freeze_sha256,
    make_index_entry,
    require_sha256,
    validate_challenge_bundle,
    validate_correctness_receipt,
    validate_effective_runtime_manifest,
    validate_evidence_role_separation,
    validate_index_chain,
    validate_protocol_manifest,
    validate_run_envelope,
    validate_source_identity,
    validate_supervisor_observation,
    validate_task_spec,
    validate_trust_basis,
    validate_worker_receipt,
    verify_signed_envelope,
)
from core.brain.frontier_evidence_v5 import (
    canonical_json_bytes as _v5_canonical_json_bytes,
)
from core.brain.frontier_evidence_v5 import (
    sha256_json as _v5_sha256_json,
)
from core.runtime.dynamic_execution_gateway import get_dynamic_execution_gateway
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.FrontierGap")

SCHEMA_VERSION = 5
BATTERY_VERSION = "2026-07-15.v5"
REFERENCE_SCHEMA = "aura.frontier_reference.v4"
REFERENCE_MODEL_IDENTITY_SCHEMA = "aura.reference_model_identity.v1"
SOURCE_PROVENANCE_SCHEMA = "aura.source_provenance.v2"
SOURCE_STABILITY_SCHEMA = "aura.source_stability_window.v2"
MODEL_MANIFEST_SCHEMA = "aura.local_model_manifest.v1"
MODEL_STABILITY_SCHEMA = "aura.model_stability_window.v1"
TRUSTED_REFERENCE_BASIS = "trusted_signed_artifact"
CAPABILITY_EVIDENCE_CLASS = "aura_model_capability"
CONTROL_EVIDENCE_CLASS = "synthetic_pipeline_control"
REJECTED_EVIDENCE_CLASS = "aura_model_rejected_attempt"
SUPPORTED_EVIDENCE_CLASSES = frozenset(
    {
        CAPABILITY_EVIDENCE_CLASS,
        CONTROL_EVIDENCE_CLASS,
        REJECTED_EVIDENCE_CLASS,
    }
)
MAX_LEDGER_ENTRIES = 96
DISQUALIFYING_FALLBACK_MARKERS = (
    "no_candidates",
    "proof_refused_unverified",
    "courtroom_fallback",
    "generation_failed",
    "static_rule",
    "playbooks_injected",
    "solved_cache_hit",
)


@dataclass(frozen=True)
class BatteryItem:
    item_id: str
    task_class: str
    task_type: str
    prompt: str
    grade: Callable[[str], bool]
    grader_id: str
    grader_implementation_sha256: str
    expected_answer_commitment_sha256: str
    hidden_case_commitment_sha256: str


@dataclass(frozen=True)
class SolverObservation:
    answer: str
    verified: bool | None = None
    receipt: Mapping[str, Any] | None = None
    supervisor_observation: Mapping[str, Any] | None = None
    fallbacks_used: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReferenceEvidence:
    model_id: str
    source: str
    measured_at: str
    battery_version: str
    seed: int
    per_class: int
    scores: Mapping[str, float]
    budget: Mapping[str, Any]
    battery_manifest_sha256: str
    model_identity: Mapping[str, Any]
    item_receipts: tuple[Mapping[str, Any], ...]
    supervisor_observations: tuple[Mapping[str, Any], ...]
    evaluator_id: str
    evaluator_public_key_b64: str
    signature_b64: str
    exact_envelope: Mapping[str, Any]
    trust_basis: Mapping[str, Any]
    challenge: Mapping[str, Any]
    task_spec: Mapping[str, Any]
    effective_runtime_manifest: Mapping[str, Any]
    model_stability: Mapping[str, Any]
    run_envelope: Mapping[str, Any]
    outputs: tuple[Mapping[str, Any], ...]
    correctness_receipts: tuple[Mapping[str, Any], ...]
    challenge_nonce: bytes
    identity_freeze_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.exact_envelope))


def canonical_json_bytes(value: Any) -> bytes:
    """Return the single canonical byte representation used by evidence signatures."""

    return _v5_canonical_json_bytes(value)


def sha256_json(value: Any) -> str:
    return _v5_sha256_json(value)


def _challenge_bytes(seed: int, challenge_nonce: bytes | None) -> tuple[bytes, bool]:
    if challenge_nonce is not None:
        if not isinstance(challenge_nonce, bytes) or len(challenge_nonce) < 32:
            raise ValueError("challenge_nonce must contain at least 256 bits")
        return challenge_nonce, True
    return hashlib.sha256(f"diagnostic-control:{seed}".encode("ascii")).digest(), False


def battery_manifest(
    *, seed: int, per_class: int, challenge_nonce: bytes | None = None
) -> dict[str, Any]:
    items = build_battery(
        seed=seed,
        per_class=per_class,
        challenge_nonce=challenge_nonce,
    )
    manifest_items = [
        {
            "index": index,
            "item_id": item.item_id,
            "task_class": item.task_class,
            "task_type": item.task_type,
            "grader_id": item.grader_id,
            "prompt_sha256": hashlib.sha256(item.prompt.encode("utf-8")).hexdigest(),
            "grader_implementation_sha256": item.grader_implementation_sha256,
            "expected_answer_commitment_sha256": (
                item.expected_answer_commitment_sha256
            ),
            "hidden_case_commitment_sha256": item.hidden_case_commitment_sha256,
        }
        for index, item in enumerate(items)
    ]
    _, held_out = _challenge_bytes(seed, challenge_nonce)
    body = {
        "battery_version": BATTERY_VERSION,
        "seed": seed,
        "per_class": per_class,
        "held_out_challenge": held_out,
        "effective_n": len({item["prompt_sha256"] for item in manifest_items}),
        "items": manifest_items,
    }
    return {**body, "sha256": sha256_json(body)}


def _normalize_short_answer(text: str) -> str:
    normalized = re.sub(r"[^\w\s+-]", "", str(text or "").casefold())
    return " ".join(normalized.split())


def _exact_text_grader(expected: str) -> Callable[[str], bool]:
    normalized_expected = _normalize_short_answer(expected)
    return lambda text: _normalize_short_answer(text) == normalized_expected


def _exact_integer_grader(expected: int) -> Callable[[str], bool]:
    expected_text = str(expected)

    def grade(text: str) -> bool:
        candidate = str(text or "").strip()
        return bool(re.fullmatch(r"[+-]?\d+", candidate)) and candidate == expected_text

    return grade


def _extract_python(text: str) -> str:
    candidate = str(text or "").strip()
    fenced = re.fullmatch(r"```(?:python|py)?\s*\n?(.*?)\n?```", candidate, re.DOTALL | re.I)
    return fenced.group(1).strip() if fenced else candidate


def _code_grader(
    *,
    function_name: str,
    operation: str,
    hidden_cases: tuple[tuple[int, ...], ...],
) -> Callable[[str], bool]:
    allowed_builtins = {"sum": sum, "max": max, "min": min, "len": len}
    allowed_nodes = (
        ast.Module,
        ast.FunctionDef,
        ast.arguments,
        ast.arg,
        ast.Return,
        ast.Call,
        ast.Name,
        ast.Load,
    )
    expected_fn = allowed_builtins[operation]

    def grade(text: str) -> bool:
        source_code = _extract_python(text)
        try:
            tree = ast.parse(source_code, mode="exec")
        except (SyntaxError, TypeError, ValueError):
            return False
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            return False
        function = tree.body[0]
        if (
            function.name != function_name
            or function.decorator_list
            or function.args.vararg is not None
            or function.args.kwarg is not None
            or function.args.defaults
            or function.args.kw_defaults
            or len(function.args.args) != 1
            or function.args.args[0].arg != "xs"
            or len(function.body) != 1
            or not isinstance(function.body[0], ast.Return)
        ):
            return False
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return False
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in allowed_builtins:
                    return False
            if isinstance(node, ast.Name) and node.id not in {*allowed_builtins, "xs"}:
                return False
        namespace: dict[str, Any] = {}
        try:
            gateway = get_dynamic_execution_gateway()
            code_object = gateway.compile_source(
                source_code,
                filename="<frontier-hidden-code>",
                mode="exec",
                source="frontier_gap.restricted_hidden_grader",
            )
            gateway.execute_code_object(
                code_object,
                globals_dict={"__builtins__": allowed_builtins},
                locals_dict=namespace,
                source="frontier_gap.restricted_hidden_grader",
            )
            candidate = namespace.get(function_name)
            if not callable(candidate):
                return False
            return all(candidate(list(case)) == expected_fn(case) for case in hidden_cases)
        except (ArithmeticError, LookupError, RuntimeError, TypeError, ValueError):
            return False

    return grade


_GRADER_COMPONENTS: dict[str, tuple[Callable[..., Any], ...]] = {
    "exact_integer.v2": (_exact_integer_grader,),
    "exact_normalized_text.v2": (_normalize_short_answer, _exact_text_grader),
    "restricted_ast_hidden_execution.v2": (_extract_python, _code_grader),
}


def grader_implementation_sha256(grader_id: str) -> str:
    components = _GRADER_COMPONENTS.get(grader_id)
    if not components:
        raise ValueError(f"unknown grader implementation: {grader_id}")
    return hashlib.sha256(
        "\n\n".join(inspect.getsource(component) for component in components).encode(
            "utf-8"
        )
    ).hexdigest()


def _truth_commitment(*, nonce: bytes, label: str, value: Any) -> str:
    return sha256_json(
        {
            "challenge_nonce_sha256": hashlib.sha256(nonce).hexdigest(),
            "label": label,
            "value": value,
        }
    )


def _make_item(
    *,
    nonce: bytes,
    task_class: str,
    task_type: str,
    prompt: str,
    grade: Callable[[str], bool],
    grader_id: str,
    expected: Any,
    hidden_cases: Any = None,
) -> BatteryItem:
    grader_digest = grader_implementation_sha256(grader_id)
    expected_digest = _truth_commitment(
        nonce=nonce,
        label="expected_answer",
        value=expected,
    )
    hidden_digest = _truth_commitment(
        nonce=nonce,
        label="hidden_cases",
        value=hidden_cases if hidden_cases is not None else {"kind": "none"},
    )
    prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    item_id = sha256_json(
        {
            "battery_version": BATTERY_VERSION,
            "task_class": task_class,
            "task_type": task_type,
            "prompt_sha256": prompt_digest,
            "grader_implementation_sha256": grader_digest,
            "expected_answer_commitment_sha256": expected_digest,
            "hidden_case_commitment_sha256": hidden_digest,
        }
    )
    return BatteryItem(
        item_id=item_id,
        task_class=task_class,
        task_type=task_type,
        prompt=prompt,
        grade=grade,
        grader_id=grader_id,
        grader_implementation_sha256=grader_digest,
        expected_answer_commitment_sha256=expected_digest,
        hidden_case_commitment_sha256=hidden_digest,
    )


def _int_items(rng: random.Random, n: int, nonce: bytes) -> list[BatteryItem]:
    items: list[BatteryItem] = []
    seen: set[tuple[int, int]] = set()
    while len(items) < n:
        pair = (rng.randint(12, 999), rng.randint(12, 999))
        if pair in seen:
            continue
        seen.add(pair)
        a, b = pair
        answer = a * b
        items.append(
            _make_item(
                nonce=nonce,
                task_class="math",
                task_type="math",
                prompt=f"Compute {a} * {b}. Answer with just the integer.",
                grade=_exact_integer_grader(answer),
                grader_id="exact_integer.v2",
                expected=str(answer),
            )
        )
    return items


def _reasoning_items(rng: random.Random, n: int, nonce: bytes) -> list[BatteryItem]:
    items: list[BatteryItem] = []
    names = ["Ada", "Bao", "Cy", "Dita", "Evren", "Fen", "Gia", "Hale"]
    seen_prompts: set[str] = set()
    while len(items) < n:
        picked = rng.sample(names, 3)
        ages = rng.sample(range(20, 80), 3)
        order = sorted(zip(ages, picked, strict=True))
        oldest = order[-1][1]
        clues = (
            f"{order[2][1]} is older than {order[1][1]}. "
            f"{order[1][1]} is older than {order[0][1]}."
        )
        prompt = f"{clues} Who is oldest? Answer with just the name."
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        items.append(
            _make_item(
                nonce=nonce,
                task_class="reasoning",
                task_type="logic",
                prompt=prompt,
                grade=_exact_text_grader(oldest),
                grader_id="exact_normalized_text.v2",
                expected=_normalize_short_answer(oldest),
            )
        )
    return items


def _coding_items(rng: random.Random, n: int, nonce: bytes) -> list[BatteryItem]:
    items: list[BatteryItem] = []
    operations = (
        ("sum", "sum of"),
        ("max", "maximum of"),
        ("min", "minimum of"),
    )
    for index in range(n):
        operation, label = rng.choice(operations)
        function_name = f"{operation}_case_{index}_{rng.getrandbits(32):08x}"
        hidden_cases = tuple(
            tuple(rng.randint(-500, 500) for _ in range(rng.randint(2, 10)))
            for _case in range(7)
        )
        prompt = (
            f"Write a Python function `{function_name}(xs)` returning the {label} "
            "a non-empty finite list of integers. Return only one Python code block."
        )
        items.append(
            _make_item(
                nonce=nonce,
                task_class="coding",
                task_type="code",
                prompt=prompt,
                grade=_code_grader(
                    function_name=function_name,
                    operation=operation,
                    hidden_cases=hidden_cases,
                ),
                grader_id="restricted_ast_hidden_execution.v2",
                expected={"function_name": function_name, "operation": operation},
                hidden_cases=hidden_cases,
            )
        )
    return items


_FACTS = (
    ("What is the chemical symbol for gold?", "Au"),
    ("What planet is known as the Red Planet?", "Mars"),
    ("How many sides does a hexagon have?", "6"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What gas do plants primarily absorb for photosynthesis?", "carbon dioxide"),
    ("What is the chemical symbol for sodium?", "Na"),
    ("What is the largest ocean on Earth?", "Pacific Ocean"),
    ("How many degrees are in a right angle?", "90"),
    ("What is the capital of Kenya?", "Nairobi"),
    ("Which element has atomic number 8?", "oxygen"),
    ("What instrument measures atmospheric pressure?", "barometer"),
    ("What is the smallest prime number?", "2"),
    ("Which continent contains Peru?", "South America"),
    ("What is the SI unit of electric current?", "ampere"),
    ("What process changes liquid water into vapor?", "evaporation"),
    ("What is the capital of New Zealand?", "Wellington"),
)


def _factual_items(rng: random.Random, n: int, nonce: bytes) -> list[BatteryItem]:
    if n > len(_FACTS):
        raise ValueError(f"per_class cannot exceed {len(_FACTS)} for the held-out facts")
    return [
        _make_item(
            nonce=nonce,
            task_class="factual",
            task_type="factual",
            prompt=f"{question} Answer with only the answer.",
            grade=_exact_text_grader(answer),
            grader_id="exact_normalized_text.v2",
            expected=_normalize_short_answer(answer),
        )
        for question, answer in rng.sample(list(_FACTS), n)
    ]


_BATTERY_BUILDERS = {
    "math": _int_items,
    "reasoning": _reasoning_items,
    "coding": _coding_items,
    "factual": _factual_items,
}


def build_battery(
    *,
    seed: int,
    per_class: int = 5,
    challenge_nonce: bytes | None = None,
) -> list[BatteryItem]:
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    nonce, _ = _challenge_bytes(seed, challenge_nonce)
    derived_seed = int.from_bytes(
        hashlib.sha256(str(seed).encode("ascii") + b":" + nonce).digest(),
        "big",
    )
    rng = random.Random(derived_seed)
    items: list[BatteryItem] = []
    for builder in _BATTERY_BUILDERS.values():
        items.extend(builder(rng, per_class, nonce))
    item_ids = {item.item_id for item in items}
    prompt_hashes = {hashlib.sha256(item.prompt.encode()).hexdigest() for item in items}
    if len(item_ids) != len(items) or len(prompt_hashes) != len(items):
        raise RuntimeError("battery generation produced duplicated effective samples")
    return items


def _require_sha256(value: Any, *, field_name: str) -> str:
    digest = str(value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    return digest


def validate_reference_artifact(
    payload: Any,
    *,
    seed: int,
    per_class: int,
    trusted_evaluator_keys: Mapping[str, str] | None = None,
    trusted_worker_keys: Mapping[str, str] | None = None,
    trusted_verifiers: Mapping[str, Mapping[str, str]] | None = None,
    trusted_run_keys: Mapping[str, str] | None = None,
    trusted_release_keys: Mapping[str, str] | None = None,
    expected_identity_freeze_sha256: str | None = None,
    verification_time_unix: float | None = None,
    require_fresh_challenge: bool = False,
) -> ReferenceEvidence:
    """Validate a complete v5 reference without rewriting signed bytes."""

    raw = payload.get("payload", payload) if isinstance(payload, dict) else payload
    envelope, signed, evaluator_id = verify_signed_envelope(
        raw,
        schema=REFERENCE_SCHEMA,
        trusted_keys=trusted_evaluator_keys,
        role="frontier reference",
    )
    required = {
        "model_id",
        "source",
        "measured_at_unix",
        "battery_version",
        "seed",
        "per_class",
        "scores",
        "budget",
        "protocol_manifest",
        "trust_basis",
        "challenge",
        "task_spec",
        "effective_runtime_manifest",
        "model_stability",
        "source_identity_sha256",
        "reference_context_sha256",
        "outputs",
        "item_receipts",
        "supervisor_observations",
        "correctness_receipts",
        "run_envelope",
    }
    if set(signed) != required:
        raise ValueError("frontier reference signed fields are invalid")
    model_id = signed.get("model_id")
    source = signed.get("source")
    if (
        not isinstance(model_id, str)
        or not model_id
        or model_id != model_id.strip()
        or not isinstance(source, str)
        or not source
        or source != source.strip()
    ):
        raise ValueError("frontier reference identity text is noncanonical")
    if (
        signed.get("battery_version") != BATTERY_VERSION
        or signed.get("seed") != seed
        or signed.get("per_class") != per_class
    ):
        raise ValueError("frontier reference battery instance mismatch")
    if signed.get("budget") != MATCHED_BUDGET:
        raise ValueError("frontier reference budget is not the matched v5 budget")
    validate_protocol_manifest(signed.get("protocol_manifest"))
    trust_basis = validate_trust_basis(
        signed.get("trust_basis"),
        evaluator_keys=trusted_evaluator_keys,
        worker_keys=trusted_worker_keys,
        verifiers=trusted_verifiers,
        run_keys=trusted_run_keys,
        release_keys=trusted_release_keys,
    )
    runtime_manifest = validate_effective_runtime_manifest(
        signed.get("effective_runtime_manifest")
    )
    if runtime_manifest["subject_id"] != model_id:
        raise ValueError("reference runtime subject does not match model_id")
    raw_model_window = signed.get("model_stability")
    raw_model_before = (
        raw_model_window.get("before") if isinstance(raw_model_window, dict) else None
    )
    raw_model_digest = (
        raw_model_before.get("manifest_sha256")
        if isinstance(raw_model_before, dict)
        else ""
    )
    model_stability = _validate_model_stability(
        raw_model_window,
        measurement_subject=f"aura_model:{raw_model_digest}",
    )
    if runtime_manifest["base_model_manifest_sha256"] != model_stability["before"][
        "manifest_sha256"
    ]:
        raise ValueError("reference runtime is not bound to stable model material")
    reference_model_files = {
        entry["path"]: entry["sha256"]
        for entry in model_stability["before"]["files"]
    }
    expected_reference_adapters = sorted(
        reference_model_files[path]
        for path in model_stability["before"]["roles"]["adapters"]
    )
    if runtime_manifest["adapters_sha256"] != expected_reference_adapters:
        raise ValueError("reference runtime adapter identity omits stable model material")
    source_identity_digest = require_sha256(
        signed.get("source_identity_sha256"), field_name="reference source identity"
    )
    reference_context = require_sha256(
        signed.get("reference_context_sha256"), field_name="reference context"
    )
    expected_reference_context = sha256_json(
        {
            "model_id": model_id,
            "source": source,
            "source_identity_sha256": source_identity_digest,
            "effective_runtime_manifest_sha256": runtime_manifest["manifest_sha256"],
            "protocol_manifest_sha256": PROTOCOL_MANIFEST_SHA256,
        }
    )
    if reference_context != expected_reference_context:
        raise ValueError("reference context digest is not reproducible")
    challenge = validate_challenge_bundle(
        signed.get("challenge"),
        trusted_evaluator_keys=trusted_evaluator_keys,
        expected_identity_freeze_sha256=expected_identity_freeze_sha256,
        verification_time_unix=verification_time_unix,
        require_fresh=require_fresh_challenge,
    )
    manifest = battery_manifest(
        seed=seed,
        per_class=per_class,
        challenge_nonce=challenge["nonce"],
    )
    task_spec = validate_task_spec(
        signed.get("task_spec"),
        trusted_evaluator_keys=trusted_evaluator_keys,
        trusted_verifiers=trusted_verifiers,
        challenge=challenge,
        expected_items=manifest["items"],
        battery_version=BATTERY_VERSION,
        seed=seed,
        per_class=per_class,
    )
    if evaluator_id != challenge["evaluator_id"]:
        raise ValueError("reference signer differs from the challenge issuer")
    outputs = signed.get("outputs")
    items = build_battery(
        seed=seed,
        per_class=per_class,
        challenge_nonce=challenge["nonce"],
    )
    if not isinstance(outputs, list) or len(outputs) != len(items):
        raise ValueError("frontier reference outputs are incomplete")
    normalized_outputs: list[dict[str, Any]] = []
    for index, (output, item) in enumerate(zip(outputs, items, strict=True)):
        if not isinstance(output, dict) or set(output) != {
            "index",
            "item_id",
            "answer",
            "output_sha256",
        }:
            raise ValueError("frontier reference output fields are invalid")
        answer = output.get("answer")
        if (
            output.get("index") != index
            or output.get("item_id") != item.item_id
            or not isinstance(answer, str)
            or len(answer.encode("utf-8")) > 256 * 1024
        ):
            raise ValueError("frontier reference output does not match the battery")
        if output.get("output_sha256") != hashlib.sha256(answer.encode()).hexdigest():
            raise ValueError("frontier reference output digest mismatch")
        normalized_outputs.append(copy.deepcopy(output))
    run_raw, run_payload, _ = verify_signed_envelope(
        signed.get("run_envelope"),
        schema=RUN_ENVELOPE_SCHEMA,
        trusted_keys=trusted_run_keys,
        role="reference run coordinator",
    )
    run_id = require_sha256(run_payload.get("run_id"), field_name="reference run_id")
    run_nonce_sha256 = require_sha256(
        run_payload.get("run_nonce_sha256"), field_name="reference run nonce"
    )
    worker_raw = signed.get("item_receipts")
    if not isinstance(worker_raw, list) or len(worker_raw) != len(items):
        raise ValueError("frontier reference worker receipts are incomplete")
    worker_receipts: list[dict[str, Any]] = []
    request_ids: set[str] = set()
    for index, (receipt, item, output) in enumerate(
        zip(worker_raw, items, normalized_outputs, strict=True)
    ):
        validated = validate_worker_receipt(
            receipt,
            trusted_worker_keys=trusted_worker_keys,
            bindings={
                "run_id": run_id,
                "run_nonce_sha256": run_nonce_sha256,
                "item_id": item.item_id,
                "prompt_sha256": hashlib.sha256(item.prompt.encode()).hexdigest(),
                "output_sha256": output["output_sha256"],
                "source_identity_sha256": source_identity_digest,
                "runtime_manifest_sha256": runtime_manifest["manifest_sha256"],
                "model_stability_sha256": model_stability["window_sha256"],
                "protocol_manifest_sha256": PROTOCOL_MANIFEST_SHA256,
                "challenge_bundle_sha256": challenge["bundle_sha256"],
            },
        )
        if validated["payload"]["attempt_index"] != index:
            raise ValueError("frontier reference worker receipt index mismatch")
        request_id = validated["payload"]["request_id"]
        if request_id in request_ids:
            raise ValueError("frontier reference worker request identity is duplicated")
        request_ids.add(request_id)
        worker_receipts.append(validated)
    supervisor_raw = signed.get("supervisor_observations")
    if not isinstance(supervisor_raw, list) or len(supervisor_raw) != len(items):
        raise ValueError("frontier reference supervisor observations are incomplete")
    supervisor_observations: list[dict[str, Any]] = []
    for index, (observation, item, output, worker) in enumerate(
        zip(supervisor_raw, items, normalized_outputs, worker_receipts, strict=True)
    ):
        request_id = worker["payload"]["request_id"]
        validated_observation = validate_supervisor_observation(
            observation,
            bindings={
                "run_id": run_id,
                "run_nonce_sha256": run_nonce_sha256,
                "item_id": item.item_id,
                "request_id": request_id,
                "attempt_index": index,
                "prompt_sha256": hashlib.sha256(item.prompt.encode()).hexdigest(),
                "output_sha256": output["output_sha256"],
            },
        )
        if (
            float(validated_observation["observed_wall_time_s"]) + 0.25
            < float(worker["payload"]["elapsed_s"])
        ):
            raise ValueError("frontier reference worker time exceeds supervisor observation")
        supervisor_observations.append(validated_observation)
    correctness_raw = signed.get("correctness_receipts")
    if not isinstance(correctness_raw, list) or len(correctness_raw) != len(items):
        raise ValueError("frontier reference correctness receipts are incomplete")
    correctness_receipts: list[dict[str, Any]] = []
    for receipt, item, output in zip(
        correctness_raw, items, normalized_outputs, strict=True
    ):
        correctness_receipts.append(
            validate_correctness_receipt(
                receipt,
                trusted_verifiers=trusted_verifiers,
                verifier_identity=task_spec["verifier_identity"],
                bindings={
                    "run_id": run_id,
                    "item_id": item.item_id,
                    "output_sha256": output["output_sha256"],
                    "task_spec_sha256": task_spec["task_spec_sha256"],
                    "challenge_bundle_sha256": challenge["bundle_sha256"],
                    "expected_answer_commitment_sha256": (
                        item.expected_answer_commitment_sha256
                    ),
                    "hidden_case_commitment_sha256": (
                        item.hidden_case_commitment_sha256
                    ),
                },
            )
        )
    run = validate_run_envelope(
        run_raw,
        trusted_run_keys=trusted_run_keys,
        bindings={
            "run_id": run_id,
            "run_nonce_sha256": run_nonce_sha256,
            "task_spec_sha256": task_spec["task_spec_sha256"],
            "challenge_bundle_sha256": challenge["bundle_sha256"],
            "protocol_manifest_sha256": PROTOCOL_MANIFEST_SHA256,
            "source_identity_sha256": source_identity_digest,
            "runtime_manifest_sha256": runtime_manifest["manifest_sha256"],
            "reference_artifact_sha256": reference_context,
            "trust_basis_sha256": trust_basis["manifest_sha256"],
            "verifier_id": task_spec["verifier_id"],
        },
        worker_receipts=worker_receipts,
        supervisor_observations=supervisor_observations,
        correctness_receipts=correctness_receipts,
        outputs=normalized_outputs,
        challenge=challenge,
    )
    validate_evidence_role_separation(
        challenge=challenge,
        task_spec=task_spec,
        worker_receipts=worker_receipts,
        correctness_receipts=correctness_receipts,
        run_envelope=run,
    )
    correct_by_class = {task_class: 0 for task_class in _BATTERY_BUILDERS}
    count_by_class = {task_class: 0 for task_class in _BATTERY_BUILDERS}
    for item, receipt in zip(items, correctness_receipts, strict=True):
        count_by_class[item.task_class] += 1
        correct_by_class[item.task_class] += int(receipt["payload"]["correct"])
    scores = signed.get("scores")
    if not isinstance(scores, dict) or set(scores) != set(_BATTERY_BUILDERS):
        raise ValueError("frontier reference class coverage mismatch")
    normalized_scores: dict[str, float] = {}
    for task_class, value in scores.items():
        score = _finite_float(value, field_name=f"reference score {task_class}")
        expected_score = correct_by_class[task_class] / count_by_class[task_class]
        if not 0.0 <= score <= 1.0 or not math.isclose(
            score, expected_score, abs_tol=1e-12
        ):
            raise ValueError("frontier reference score is not verifier-reproducible")
        normalized_scores[task_class] = score
    measured_at = _finite_float(
        signed.get("measured_at_unix"), field_name="reference measurement time"
    )
    signer = envelope["signer"]
    return ReferenceEvidence(
        model_id=model_id,
        source=source,
        measured_at=str(measured_at),
        battery_version=BATTERY_VERSION,
        seed=seed,
        per_class=per_class,
        scores=normalized_scores,
        budget=copy.deepcopy(MATCHED_BUDGET),
        battery_manifest_sha256=manifest["sha256"],
        model_identity=runtime_manifest,
        item_receipts=tuple(item["envelope"] for item in worker_receipts),
        supervisor_observations=tuple(supervisor_observations),
        evaluator_id=evaluator_id,
        evaluator_public_key_b64=signer["public_key_b64"],
        signature_b64=signer["signature_b64"],
        exact_envelope=envelope,
        trust_basis=trust_basis,
        challenge={"commit": challenge["commit"], "reveal": challenge["reveal"]},
        task_spec=task_spec["envelope"],
        effective_runtime_manifest=runtime_manifest,
        model_stability=model_stability,
        run_envelope=run["envelope"],
        outputs=tuple(normalized_outputs),
        correctness_receipts=tuple(
            item["envelope"] for item in correctness_receipts
        ),
        challenge_nonce=bytes(challenge["nonce"]),
        identity_freeze_sha256=challenge["identity_freeze_sha256"],
    )


@dataclass
class ClassResult:
    task_class: str
    n: int
    candidate_correct: int
    reference_score: float | None = None

    @property
    def candidate_score(self) -> float:
        return self.candidate_correct / self.n if self.n else 0.0

    @property
    def gap(self) -> float | None:
        if self.reference_score is None:
            return None
        if self.reference_score <= 0.0:
            return 0.0 if self.candidate_score <= 0.0 else None
        return max(0.0, min(1.0, 1.0 - self.candidate_score / self.reference_score))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_class": self.task_class,
            "n": self.n,
            "candidate_correct": self.candidate_correct,
            "candidate_score": round(self.candidate_score, 4),
            "reference_score": (
                round(self.reference_score, 4)
                if self.reference_score is not None
                else None
            ),
            "gap": round(self.gap, 4) if self.gap is not None else None,
        }


def _coerce_observation(value: Any) -> SolverObservation:
    if isinstance(value, SolverObservation):
        return value
    if isinstance(value, str):
        return SolverObservation(answer=value)
    raise TypeError(f"solver returned unsupported observation: {type(value).__name__}")


async def run_battery(
    solve: Callable[[str, str], Awaitable[str | SolverObservation]],
    *,
    seed: int,
    per_class: int = 5,
    reference: ReferenceEvidence | None = None,
    challenge_nonce: bytes | None = None,
    grade_to_foundry: bool = True,
) -> dict[str, Any]:
    if reference is not None:
        if challenge_nonce is not None and challenge_nonce != reference.challenge_nonce:
            raise ValueError("explicit challenge nonce contradicts reference evidence")
        challenge_nonce = reference.challenge_nonce
    items = build_battery(
        seed=seed,
        per_class=per_class,
        challenge_nonce=challenge_nonce,
    )
    if reference is not None and (
        reference.seed != seed
        or reference.per_class != per_class
        or reference.battery_version != BATTERY_VERSION
    ):
        raise ValueError("reference evidence does not match this battery instance")
    by_class: dict[str, ClassResult] = {}
    item_evidence: list[dict[str, Any]] = []
    reference_scores = dict(reference.scores) if reference is not None else {}

    foundry = None
    if grade_to_foundry:
        try:
            from core.runtime.service_access import optional_service

            foundry = optional_service("verifier_foundry", default=None)
        except (ImportError, RuntimeError):
            foundry = None

    started = time.time()
    for item_index, item in enumerate(items):
        execution_error = ""
        try:
            observation = _coerce_observation(
                await solve(item.prompt, item.task_type)
            )
        except Exception as exc:  # noqa: BLE001 - bounded evaluation item boundary
            record_degradation(
                "frontier_gap",
                exc,
                severity="warning",
                action="battery item errored; retained as an invalid miss",
            )
            execution_error = f"{type(exc).__name__}: {str(exc)[:240]}"
            observation = SolverObservation(
                answer="",
                verified=False,
                diagnostics=(execution_error,),
            )
        correct = bool(item.grade(observation.answer))
        cell = by_class.get(item.task_class)
        if cell is None:
            cell = ClassResult(
                item.task_class,
                0,
                0,
                reference_scores.get(item.task_class),
            )
            by_class[item.task_class] = cell
        cell.n += 1
        cell.candidate_correct += int(correct)
        item_evidence.append(
            {
                "index": item_index,
                "item_id": item.item_id,
                "task_class": item.task_class,
                "task_type": item.task_type,
                "grader_id": item.grader_id,
                "prompt_sha256": hashlib.sha256(item.prompt.encode()).hexdigest(),
                "grader_implementation_sha256": item.grader_implementation_sha256,
                "expected_answer_commitment_sha256": (
                    item.expected_answer_commitment_sha256
                ),
                "hidden_case_commitment_sha256": (
                    item.hidden_case_commitment_sha256
                ),
                "answer": observation.answer,
                "output_sha256": hashlib.sha256(observation.answer.encode()).hexdigest(),
                "answer_sha256": hashlib.sha256(observation.answer.encode()).hexdigest(),
                "correct": correct,
                "execution_attested": observation.verified,
                "verified": observation.verified,
                "receipt": dict(observation.receipt or {}),
                "supervisor_observation": dict(
                    observation.supervisor_observation or {}
                ),
                "correctness_receipt": None,
                "fallbacks_used": list(observation.fallbacks_used),
                "diagnostics": list(observation.diagnostics),
                "execution_error": execution_error,
            }
        )
        if foundry is not None:
            try:
                verdict_id = foundry.record_verdict(
                    verifier=f"battery:{item.grader_id}",
                    domain=item.task_type,
                    hard_pass=correct,
                    score=1.0 if correct else 0.0,
                    checked=True,
                    task_key=item.task_class,
                )
                if verdict_id:
                    foundry.grade_verdict(
                        verdict_id,
                        truth_pass=correct,
                        source="frontier_battery",
                    )
            except (RuntimeError, AttributeError, TypeError, ValueError):
                pass

    classes = [cell.to_dict() for cell in by_class.values()]
    total = sum(cell.n for cell in by_class.values())
    overall_candidate = sum(
        cell.candidate_correct for cell in by_class.values()
    ) / max(1, total)
    gaps = [cell.gap for cell in by_class.values()]
    overall_gap = (
        sum(float(gap) for gap in gaps) / len(gaps)
        if gaps and all(gap is not None for gap in gaps)
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "measurement_scope": "four_class_deterministic_diagnostic",
        "general_frontier_claim_eligible": False,
        "battery_version": BATTERY_VERSION,
        "seed": seed,
        "per_class": per_class,
        "expected_item_count": len(items),
        "effective_n": len(
            {hashlib.sha256(item.prompt.encode()).hexdigest() for item in items}
        ),
        "classes": classes,
        "items": item_evidence,
        "overall_candidate_score": round(overall_candidate, 4),
        "overall_gap": round(overall_gap, 4) if overall_gap is not None else None,
        "reference_basis": TRUSTED_REFERENCE_BASIS if reference is not None else "unavailable",
        "reference": reference.to_dict() if reference is not None else None,
        "reference_artifact_sha256": (
            sha256_json(reference.to_dict()) if reference is not None else None
        ),
        "protocol_manifest": copy.deepcopy(PROTOCOL_MANIFEST),
        "task_spec": copy.deepcopy(dict(reference.task_spec)) if reference else None,
        "challenge": copy.deepcopy(dict(reference.challenge)) if reference else None,
        "challenge_id": (
            reference.challenge["commit"]["signed_payload"]["challenge_id"]
            if reference
            else None
        ),
        "correctness_receipts": [],
        "run_envelope": None,
        "effective_runtime_manifest": None,
        "source_identity": None,
        "comparison_stratum_sha256": (
            sha256_json(
                {
                    "battery_version": BATTERY_VERSION,
                    "per_class": per_class,
                    "reference_runtime_manifest_sha256": reference.effective_runtime_manifest[
                        "manifest_sha256"
                    ],
                    "protocol_manifest_sha256": PROTOCOL_MANIFEST_SHA256,
                }
            )
            if reference
            else None
        ),
        "budget": dict(MATCHED_BUDGET),
        "duration_s": round(time.time() - started, 2),
        "generated_at_unix": time.time(),
    }


def _finite_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite number")
    return normalized


def _validate_source_provenance(
    raw: Any,
    *,
    source_tree_resolver: Callable[[str], str],
    source_component_resolver: Callable[[str, str], str],
) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("schema") != SOURCE_PROVENANCE_SCHEMA:
        raise ValueError("capability source provenance schema is invalid")
    normalized = json.loads(canonical_json_bytes(raw))
    commit_sha = str(normalized.get("commit_sha") or "").lower()
    tree_sha = str(normalized.get("tree_sha") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{40}", commit_sha):
        raise ValueError("capability source commit is invalid")
    if not re.fullmatch(r"[0-9a-f]{40}", tree_sha):
        raise ValueError("capability source tree is invalid")
    try:
        resolved_tree = str(source_tree_resolver(commit_sha) or "").strip().lower()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("capability source commit cannot be resolved") from exc
    if resolved_tree != tree_sha:
        raise ValueError("capability source tree does not belong to its commit")
    if normalized.get("clean") is not True or normalized.get("issues") != []:
        raise ValueError("capability source provenance is not clean")
    empty_digest = hashlib.sha256(b"").hexdigest()
    for field_name in (
        "workspace_diff_sha256",
        "index_diff_sha256",
        "untracked_content_sha256",
        "workspace_state_sha256",
    ):
        _require_sha256(normalized.get(field_name), field_name=field_name)
    if normalized["workspace_diff_sha256"] != empty_digest:
        raise ValueError("clean source provenance contains a workspace diff")
    if normalized["index_diff_sha256"] != empty_digest:
        raise ValueError("clean source provenance contains an index diff")
    if normalized["untracked_content_sha256"] != empty_digest:
        raise ValueError("clean source provenance contains untracked content")
    expected_workspace = hashlib.sha256()
    expected_workspace.update(commit_sha.encode())
    expected_workspace.update(tree_sha.encode())
    expected_workspace.update(hashlib.sha256(b"").digest())
    if normalized["workspace_state_sha256"] != expected_workspace.hexdigest():
        raise ValueError("source workspace digest is not reproducible")
    components = normalized.get("execution_component_sha256")
    required_components = {
        "tools/measure_frontier_gap.py",
        "core/brain/frontier_gap.py",
        "core/brain/frontier_evidence_v5.py",
        "core/brain/reasoning_amplifier_v2.py",
        "core/brain/verifiers/registry.py",
        "core/brain/llm/mlx_client.py",
        "core/brain/llm/model_registry.py",
        "core/runtime/dynamic_execution_gateway.py",
    }
    if not isinstance(components, dict) or not required_components.issubset(components):
        raise ValueError("capability source component coverage is incomplete")
    for path, digest in components.items():
        if not isinstance(path, str) or not path or path.startswith("/") or ".." in path.split("/"):
            raise ValueError("capability source component path is invalid")
        normalized_digest = _require_sha256(digest, field_name=f"source component {path}")
        try:
            resolved_digest = str(source_component_resolver(commit_sha, path) or "").lower()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(f"source component cannot be resolved: {path}") from exc
        if resolved_digest != normalized_digest:
            raise ValueError(f"source component is not bound to its commit: {path}")
    source_identity = normalized.get("source_identity")
    if not isinstance(source_identity, dict):
        raise ValueError("capability source provenance lacks trusted repository identity")
    if (
        source_identity.get("commit_sha") != commit_sha
        or source_identity.get("tree_sha") != tree_sha
    ):
        raise ValueError("capability source provenance contradicts source identity")
    return normalized


def _validate_source_stability(
    raw: Any,
    *,
    source_tree_resolver: Callable[[str], str] | None,
    source_component_resolver: Callable[[str, str], str] | None,
) -> dict[str, Any]:
    if source_tree_resolver is None or source_component_resolver is None:
        raise ValueError("capability evidence requires independent source resolvers")
    if not isinstance(raw, dict) or raw.get("schema") != SOURCE_STABILITY_SCHEMA:
        raise ValueError("capability source stability schema is invalid")
    before = _validate_source_provenance(
        raw.get("before"),
        source_tree_resolver=source_tree_resolver,
        source_component_resolver=source_component_resolver,
    )
    after = _validate_source_provenance(
        raw.get("after"),
        source_tree_resolver=source_tree_resolver,
        source_component_resolver=source_component_resolver,
    )
    if canonical_json_bytes(before) != canonical_json_bytes(after):
        raise ValueError("capability source changed during measurement")
    body = {"before": before, "after": after}
    if raw.get("stable") is not True or raw.get("window_sha256") != sha256_json(body):
        raise ValueError("capability source stability receipt is invalid")
    return {
        "schema": SOURCE_STABILITY_SCHEMA,
        **body,
        "stable": True,
        "window_sha256": sha256_json(body),
    }


def _validate_model_manifest(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("schema") != MODEL_MANIFEST_SCHEMA:
        raise ValueError("candidate model manifest schema is invalid")
    normalized = json.loads(canonical_json_bytes(raw))
    model_path = normalized.get("model_path")
    files = normalized.get("files")
    roles = normalized.get("roles")
    if not isinstance(model_path, str) or not model_path.startswith("/"):
        raise ValueError("candidate model path is not absolute")
    if not isinstance(files, list) or not files:
        raise ValueError("candidate model manifest has no files")
    seen_paths: set[str] = set()
    total_bytes = 0
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("candidate model file entry is malformed")
        path = entry.get("path")
        size = entry.get("size")
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or ".." in path.split("/")
            or path in seen_paths
        ):
            raise ValueError("candidate model file path is invalid or duplicated")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("candidate model file size is invalid")
        _require_sha256(entry.get("sha256"), field_name=f"model file {path}")
        seen_paths.add(path)
        total_bytes += size
    if normalized.get("file_count") != len(files) or normalized.get("total_bytes") != total_bytes:
        raise ValueError("candidate model manifest totals are invalid")
    expected_roles = {"weights", "configuration", "tokenizer", "adapters"}
    if not isinstance(roles, dict) or set(roles) != expected_roles:
        raise ValueError("candidate model role map is invalid")
    for role, paths in roles.items():
        if not isinstance(paths, list) or len(paths) != len(set(paths)):
            raise ValueError(f"candidate model {role} role is malformed")
        if any(path not in seen_paths for path in paths):
            raise ValueError(f"candidate model {role} role references an unknown file")
    if not roles["weights"] or not roles["configuration"] or not roles["tokenizer"]:
        raise ValueError("candidate model manifest lacks required material")
    expected_digest = _require_sha256(
        normalized.get("manifest_sha256"), field_name="model manifest_sha256"
    )
    body = {key: value for key, value in normalized.items() if key != "manifest_sha256"}
    if sha256_json(body) != expected_digest:
        raise ValueError("candidate model manifest digest mismatch")
    return normalized


def _validate_model_stability(raw: Any, *, measurement_subject: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("schema") != MODEL_STABILITY_SCHEMA:
        raise ValueError("candidate model stability schema is invalid")
    before = _validate_model_manifest(raw.get("before"))
    after = _validate_model_manifest(raw.get("after"))
    if canonical_json_bytes(before) != canonical_json_bytes(after):
        raise ValueError("candidate model changed during measurement")
    body = {"before": before, "after": after}
    if raw.get("stable") is not True or raw.get("window_sha256") != sha256_json(body):
        raise ValueError("candidate model stability receipt is invalid")
    expected_subject = f"aura_model:{before['manifest_sha256']}"
    if measurement_subject != expected_subject:
        raise ValueError("measurement subject is not bound to the candidate model")
    return {
        "schema": MODEL_STABILITY_SCHEMA,
        **body,
        "stable": True,
        "window_sha256": sha256_json(body),
    }


def validate_capability_report(
    report: Any,
    *,
    trusted_evaluator_keys: Mapping[str, str] | None,
    trusted_worker_keys: Mapping[str, str] | None = None,
    trusted_verifiers: Mapping[str, Mapping[str, str]] | None = None,
    trusted_run_keys: Mapping[str, str] | None = None,
    trusted_release_keys: Mapping[str, str] | None = None,
    source_tree_resolver: Callable[[str], str] | None,
    source_component_resolver: Callable[[str, str], str] | None,
) -> dict[str, Any]:
    """Recompute a v5 claim from signed execution and correctness evidence."""

    if not isinstance(report, dict):
        raise ValueError("capability report must be an object")
    normalized = json.loads(canonical_json_bytes(report))
    if normalized.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("capability report schema version is invalid")
    if normalized.get("battery_version") != BATTERY_VERSION:
        raise ValueError("capability report battery version is invalid")
    if (
        normalized.get("measurement_scope") != "four_class_deterministic_diagnostic"
        or normalized.get("general_frontier_claim_eligible") is not False
    ):
        raise ValueError("capability report overstates the diagnostic battery scope")
    if normalized.get("evidence_class") != CAPABILITY_EVIDENCE_CLASS:
        raise ValueError("capability report evidence class is invalid")
    if normalized.get("capability_claim_eligible") is not True:
        raise ValueError("capability report is not claim eligible")
    if normalized.get("solver_mode") != "amplifier_mlx_worker_v5":
        raise ValueError("capability report did not use the isolated v5 model worker")
    if normalized.get("budget") != MATCHED_BUDGET:
        raise ValueError("capability report budget is not the pinned v5 protocol")
    validate_protocol_manifest(normalized.get("protocol_manifest"))
    trust_basis = validate_trust_basis(
        normalized.get("trust_basis"),
        evaluator_keys=trusted_evaluator_keys,
        worker_keys=trusted_worker_keys,
        verifiers=trusted_verifiers,
        run_keys=trusted_run_keys,
        release_keys=trusted_release_keys,
    )
    seed = normalized.get("seed")
    per_class = normalized.get("per_class")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("capability report seed is invalid")
    if isinstance(per_class, bool) or not isinstance(per_class, int) or per_class <= 0:
        raise ValueError("capability report per_class is invalid")

    source_stability = _validate_source_stability(
        normalized.get("source_stability"),
        source_tree_resolver=source_tree_resolver,
        source_component_resolver=source_component_resolver,
    )
    source_identity = validate_source_identity(
        normalized.get("source_identity"),
        trusted_release_keys=trusted_release_keys,
    )
    if (
        source_identity["commit_sha"] != source_stability["after"]["commit_sha"]
        or source_identity["tree_sha"] != source_stability["after"]["tree_sha"]
    ):
        raise ValueError("source identity is not bound to the measured source window")
    if canonical_json_bytes(normalized.get("source_provenance")) != canonical_json_bytes(
        source_stability["after"]
    ):
        raise ValueError("capability source summary is not bound to its stability window")
    runtime_manifest = validate_effective_runtime_manifest(
        normalized.get("effective_runtime_manifest")
    )
    raw_model_window = normalized.get("candidate_model")
    raw_model_before = (
        raw_model_window.get("before") if isinstance(raw_model_window, dict) else None
    )
    raw_model_digest = (
        raw_model_before.get("manifest_sha256")
        if isinstance(raw_model_before, dict)
        else ""
    )
    model_stability = _validate_model_stability(
        raw_model_window,
        measurement_subject=f"aura_model:{raw_model_digest}",
    )
    if runtime_manifest["base_model_manifest_sha256"] != model_stability["before"][
        "manifest_sha256"
    ]:
        raise ValueError("effective runtime is not bound to the stable model material")
    model_files = {
        entry["path"]: entry["sha256"]
        for entry in model_stability["before"]["files"]
    }
    expected_adapters = sorted(
        model_files[path]
        for path in model_stability["before"]["roles"]["adapters"]
    )
    if runtime_manifest["adapters_sha256"] != expected_adapters:
        raise ValueError("effective runtime adapter identity omits stable model material")
    expected_subject = f"aura_model:{runtime_manifest['manifest_sha256']}"
    if normalized.get("measurement_subject") != expected_subject:
        raise ValueError("measurement subject omits effective runtime identity")

    reference_raw = normalized.get("reference")
    if not isinstance(reference_raw, dict):
        raise ValueError("capability report lacks a signed reference")
    reference_signed = reference_raw.get("signed_payload")
    if not isinstance(reference_signed, dict):
        raise ValueError("capability reference signed payload is missing")
    reference_runtime = validate_effective_runtime_manifest(
        reference_signed.get("effective_runtime_manifest")
    )
    expected_freeze = identity_freeze_sha256(
        source_identity_sha256=source_identity["identity_sha256"],
        candidate_runtime_sha256=runtime_manifest["manifest_sha256"],
        reference_runtime_sha256=reference_runtime["manifest_sha256"],
    )
    reference = validate_reference_artifact(
        reference_raw,
        seed=seed,
        per_class=per_class,
        trusted_evaluator_keys=trusted_evaluator_keys,
        trusted_worker_keys=trusted_worker_keys,
        trusted_verifiers=trusted_verifiers,
        trusted_run_keys=trusted_run_keys,
        trusted_release_keys=trusted_release_keys,
        expected_identity_freeze_sha256=expected_freeze,
    )
    if normalized.get("reference_basis") != TRUSTED_REFERENCE_BASIS:
        raise ValueError("capability report lacks a trusted signed reference")
    reference_digest = sha256_json(reference.to_dict())
    if normalized.get("reference_artifact_sha256") != reference_digest:
        raise ValueError("capability report is not bound to its reference artifact")
    if canonical_json_bytes(normalized.get("task_spec")) != canonical_json_bytes(
        reference.task_spec
    ) or canonical_json_bytes(normalized.get("challenge")) != canonical_json_bytes(
        reference.challenge
    ):
        raise ValueError("candidate task or challenge differs from the reference run")
    challenge = validate_challenge_bundle(
        normalized["challenge"],
        trusted_evaluator_keys=trusted_evaluator_keys,
        expected_identity_freeze_sha256=expected_freeze,
    )
    manifest = battery_manifest(
        seed=seed,
        per_class=per_class,
        challenge_nonce=challenge["nonce"],
    )
    task_spec = validate_task_spec(
        normalized["task_spec"],
        trusted_evaluator_keys=trusted_evaluator_keys,
        trusted_verifiers=trusted_verifiers,
        challenge=challenge,
        expected_items=manifest["items"],
        battery_version=BATTERY_VERSION,
        seed=seed,
        per_class=per_class,
    )
    items = build_battery(
        seed=seed,
        per_class=per_class,
        challenge_nonce=challenge["nonce"],
    )
    evidence_items = normalized.get("items")
    if (
        normalized.get("expected_item_count") != len(items)
        or normalized.get("effective_n") != len(items)
        or not isinstance(evidence_items, list)
        or len(evidence_items) != len(items)
    ):
        raise ValueError("capability item evidence or effective sample count is incomplete")

    run_raw, run_payload, _ = verify_signed_envelope(
        normalized.get("run_envelope"),
        schema=RUN_ENVELOPE_SCHEMA,
        trusted_keys=trusted_run_keys,
        role="candidate run coordinator",
    )
    run_id = require_sha256(run_payload.get("run_id"), field_name="candidate run_id")
    run_nonce_sha256 = require_sha256(
        run_payload.get("run_nonce_sha256"), field_name="candidate run nonce"
    )
    worker_receipts: list[dict[str, Any]] = []
    supervisor_observations: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    seen_requests: set[str] = set()
    for index, (evidence, item) in enumerate(zip(evidence_items, items, strict=True)):
        if not isinstance(evidence, dict) or evidence.get("index") != index:
            raise ValueError("capability item index is invalid")
        expected_item_fields = {
            "item_id": item.item_id,
            "task_class": item.task_class,
            "task_type": item.task_type,
            "grader_id": item.grader_id,
            "prompt_sha256": hashlib.sha256(item.prompt.encode()).hexdigest(),
            "grader_implementation_sha256": item.grader_implementation_sha256,
            "expected_answer_commitment_sha256": (
                item.expected_answer_commitment_sha256
            ),
            "hidden_case_commitment_sha256": item.hidden_case_commitment_sha256,
        }
        if any(evidence.get(key) != value for key, value in expected_item_fields.items()):
            raise ValueError("capability item does not match the signed task specification")
        answer = evidence.get("answer")
        if not isinstance(answer, str) or not answer.strip() or len(answer.encode()) > 256 * 1024:
            raise ValueError("capability item answer is malformed")
        output_sha256 = hashlib.sha256(answer.encode()).hexdigest()
        if evidence.get("output_sha256") != output_sha256 or evidence.get(
            "answer_sha256"
        ) != output_sha256:
            raise ValueError("capability item output digest mismatch")
        worker = validate_worker_receipt(
            evidence.get("receipt"),
            trusted_worker_keys=trusted_worker_keys,
            bindings={
                "run_id": run_id,
                "run_nonce_sha256": run_nonce_sha256,
                "item_id": item.item_id,
                "prompt_sha256": expected_item_fields["prompt_sha256"],
                "output_sha256": output_sha256,
                "source_identity_sha256": source_identity["identity_sha256"],
                "runtime_manifest_sha256": runtime_manifest["manifest_sha256"],
                "model_stability_sha256": model_stability["window_sha256"],
                "protocol_manifest_sha256": PROTOCOL_MANIFEST_SHA256,
                "challenge_bundle_sha256": challenge["bundle_sha256"],
            },
        )
        if worker["payload"]["attempt_index"] != index:
            raise ValueError("capability worker attempt index mismatch")
        request_id = worker["payload"]["request_id"]
        if request_id in seen_requests:
            raise ValueError("capability worker request identity is duplicated")
        seen_requests.add(request_id)
        worker_receipts.append(worker)
        supervisor = validate_supervisor_observation(
            evidence.get("supervisor_observation"),
            bindings={
                "run_id": run_id,
                "run_nonce_sha256": run_nonce_sha256,
                "item_id": item.item_id,
                "request_id": worker["payload"]["request_id"],
                "attempt_index": index,
                "prompt_sha256": expected_item_fields["prompt_sha256"],
                "output_sha256": output_sha256,
            },
        )
        if (
            float(supervisor["observed_wall_time_s"]) + 0.25
            < float(worker["payload"]["elapsed_s"])
        ):
            raise ValueError("capability worker time exceeds supervisor observation")
        supervisor_observations.append(supervisor)
        outputs.append(
            {
                "index": index,
                "item_id": item.item_id,
                "answer": answer,
                "output_sha256": output_sha256,
            }
        )

    correctness_raw = normalized.get("correctness_receipts")
    if not isinstance(correctness_raw, list) or len(correctness_raw) != len(items):
        raise ValueError("capability independent correctness receipts are incomplete")
    correctness_receipts: list[dict[str, Any]] = []
    for index, (receipt, item, output) in enumerate(
        zip(correctness_raw, items, outputs, strict=True)
    ):
        correctness = validate_correctness_receipt(
            receipt,
            trusted_verifiers=trusted_verifiers,
            verifier_identity=task_spec["verifier_identity"],
            bindings={
                "run_id": run_id,
                "item_id": item.item_id,
                "output_sha256": output["output_sha256"],
                "task_spec_sha256": task_spec["task_spec_sha256"],
                "challenge_bundle_sha256": challenge["bundle_sha256"],
                "expected_answer_commitment_sha256": (
                    item.expected_answer_commitment_sha256
                ),
                "hidden_case_commitment_sha256": (
                    item.hidden_case_commitment_sha256
                ),
            },
        )
        evidence = evidence_items[index]
        if canonical_json_bytes(evidence.get("correctness_receipt")) != canonical_json_bytes(
            correctness["envelope"]
        ):
            raise ValueError("capability item omits or contradicts its correctness receipt")
        if evidence.get("correct") is not correctness["payload"]["correct"]:
            raise ValueError("candidate-local score contradicts independent correctness")
        correctness_receipts.append(correctness)

    run = validate_run_envelope(
        run_raw,
        trusted_run_keys=trusted_run_keys,
        bindings={
            "run_id": run_id,
            "run_nonce_sha256": run_nonce_sha256,
            "task_spec_sha256": task_spec["task_spec_sha256"],
            "challenge_bundle_sha256": challenge["bundle_sha256"],
            "protocol_manifest_sha256": PROTOCOL_MANIFEST_SHA256,
            "source_identity_sha256": source_identity["identity_sha256"],
            "runtime_manifest_sha256": runtime_manifest["manifest_sha256"],
            "reference_artifact_sha256": reference_digest,
            "trust_basis_sha256": trust_basis["manifest_sha256"],
            "verifier_id": task_spec["verifier_id"],
        },
        worker_receipts=worker_receipts,
        supervisor_observations=supervisor_observations,
        correctness_receipts=correctness_receipts,
        outputs=outputs,
        challenge=challenge,
    )
    validate_evidence_role_separation(
        challenge=challenge,
        task_spec=task_spec,
        worker_receipts=worker_receipts,
        correctness_receipts=correctness_receipts,
        run_envelope=run,
    )

    correct_by_class = {task_class: 0 for task_class in _BATTERY_BUILDERS}
    count_by_class = {task_class: 0 for task_class in _BATTERY_BUILDERS}
    for item, receipt in zip(items, correctness_receipts, strict=True):
        count_by_class[item.task_class] += 1
        correct_by_class[item.task_class] += int(receipt["payload"]["correct"])
    expected_classes = {
        task_class: ClassResult(
            task_class=task_class,
            n=count_by_class[task_class],
            candidate_correct=correct_by_class[task_class],
            reference_score=float(reference.scores[task_class]),
        ).to_dict()
        for task_class in _BATTERY_BUILDERS
    }
    observed_classes = normalized.get("classes")
    if not isinstance(observed_classes, list) or {
        str(cell.get("task_class")): cell for cell in observed_classes if isinstance(cell, dict)
    } != expected_classes:
        raise ValueError("capability class scores are not verifier-reproducible")
    expected_candidate = round(sum(correct_by_class.values()) / len(items), 4)
    expected_gap = round(
        sum(float(cell["gap"]) for cell in expected_classes.values())
        / len(expected_classes),
        4,
    )
    if _finite_float(
        normalized.get("overall_candidate_score"), field_name="overall candidate score"
    ) != expected_candidate:
        raise ValueError("capability candidate score is not verifier-reproducible")
    if _finite_float(normalized.get("overall_gap"), field_name="overall gap") != expected_gap:
        raise ValueError("capability gap is not verifier-reproducible")
    expected_stratum = sha256_json(
        {
            "battery_version": BATTERY_VERSION,
            "per_class": per_class,
            "reference_runtime_manifest_sha256": reference.effective_runtime_manifest[
                "manifest_sha256"
            ],
            "protocol_manifest_sha256": PROTOCOL_MANIFEST_SHA256,
        }
    )
    if normalized.get("comparison_stratum_sha256") != expected_stratum:
        raise ValueError("capability comparison stratum is invalid")
    if normalized.get("challenge_id") != challenge["challenge_id"]:
        raise ValueError("capability challenge identity is invalid")
    execution = normalized.get("execution")
    if not isinstance(execution, dict) or execution.get("attempted") != len(items) or execution.get(
        "completed"
    ) != len(items):
        raise ValueError("capability execution count is incomplete")
    for field_name in ("failed", "invalid", "empty", "disqualifying_fallbacks"):
        if execution.get(field_name) != 0:
            raise ValueError(f"capability execution has nonzero {field_name}")
    if normalized.get("eligibility_reasons") not in ([], None):
        raise ValueError("claim-eligible capability report contains rejection reasons")
    if normalized.get("reference_validation_error") not in (None, ""):
        raise ValueError("claim-eligible capability report contains a reference error")
    normalized["source_stability"] = source_stability
    normalized["source_provenance"] = source_stability["after"]
    normalized["source_identity"] = source_identity
    normalized["candidate_model"] = model_stability
    normalized["effective_runtime_manifest"] = runtime_manifest
    return normalized


@dataclass
class GapLedger:
    """Bounded hash-chained index over immutable content-addressed evidence."""

    evidence_class: str = CAPABILITY_EVIDENCE_CLASS
    capability_claim_eligible: bool = True
    runs: list[dict[str, Any]] = field(default_factory=list)
    max_entries: int = MAX_LEDGER_ENTRIES
    pruned_count: int = 0
    pruned_through_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.evidence_class not in SUPPORTED_EVIDENCE_CLASSES:
            raise ValueError(f"unsupported evidence class: {self.evidence_class}")
        expected = self.evidence_class == CAPABILITY_EVIDENCE_CLASS
        if self.capability_claim_eligible is not expected:
            raise ValueError("capability eligibility must match the ledger evidence class")
        if (
            isinstance(self.max_entries, bool)
            or not isinstance(self.max_entries, int)
            or not 1 <= self.max_entries <= MAX_LEDGER_ENTRIES
        ):
            raise ValueError("ledger retention bound is invalid")

    def add(
        self,
        report: dict[str, Any],
        *,
        evidence_blob_writer: Callable[[str, dict[str, Any]], None] | None = None,
        trusted_evaluator_keys: Mapping[str, str] | None = None,
        trusted_worker_keys: Mapping[str, str] | None = None,
        trusted_verifiers: Mapping[str, Mapping[str, str]] | None = None,
        trusted_run_keys: Mapping[str, str] | None = None,
        trusted_release_keys: Mapping[str, str] | None = None,
        source_tree_resolver: Callable[[str], str] | None = None,
        source_component_resolver: Callable[[str, str], str] | None = None,
    ) -> None:
        if report.get("evidence_class") != self.evidence_class:
            raise ValueError(
                f"cannot add {report.get('evidence_class')} evidence to "
                f"{self.evidence_class} ledger"
            )
        if report.get("capability_claim_eligible") is not self.capability_claim_eligible:
            raise ValueError("report capability eligibility does not match ledger")
        if evidence_blob_writer is None:
            raise ValueError("v5 evidence requires a content-addressed blob writer")
        if self.capability_claim_eligible:
            snapshot = validate_capability_report(
                report,
                trusted_evaluator_keys=trusted_evaluator_keys,
                trusted_worker_keys=trusted_worker_keys,
                trusted_verifiers=trusted_verifiers,
                trusted_run_keys=trusted_run_keys,
                trusted_release_keys=trusted_release_keys,
                source_tree_resolver=source_tree_resolver,
                source_component_resolver=source_component_resolver,
            )
        else:
            snapshot = json.loads(canonical_json_bytes(report))
            if snapshot.get("capability_claim_eligible") is not False:
                raise ValueError("non-capability evidence cannot be claim eligible")
            if (
                snapshot.get("schema_version") != SCHEMA_VERSION
                or snapshot.get("battery_version") != BATTERY_VERSION
                or not isinstance(snapshot.get("generated_at_unix"), (int, float))
                or isinstance(snapshot.get("generated_at_unix"), bool)
                or not math.isfinite(float(snapshot["generated_at_unix"]))
                or not isinstance(snapshot.get("overall_candidate_score"), (int, float))
                or isinstance(snapshot.get("overall_candidate_score"), bool)
                or not math.isfinite(float(snapshot["overall_candidate_score"]))
                or not isinstance(snapshot.get("effective_n"), int)
                or isinstance(snapshot.get("effective_n"), bool)
                or snapshot["effective_n"] <= 0
            ):
                raise ValueError("non-capability evidence summary is malformed")
            if "items" in snapshot and not isinstance(snapshot["items"], list):
                raise ValueError("non-capability item evidence is malformed")
        evidence_digest = sha256_json(snapshot)
        evidence_blob_writer(evidence_digest, snapshot)
        indexed_report = dict(snapshot)
        if not self.capability_claim_eligible:
            indexed_report["overall_gap"] = None
        previous = (
            self.runs[-1]["entry_sha256"]
            if self.runs
            else self.pruned_through_sha256
        )
        self.runs.append(
            make_index_entry(
                report=indexed_report,
                evidence_sha256=evidence_digest,
                previous_entry_sha256=previous,
            )
        )
        while len(self.runs) > self.max_entries:
            removed = self.runs.pop(0)
            self.pruned_count += 1
            self.pruned_through_sha256 = removed["entry_sha256"]

    def trend(self) -> dict[str, Any]:
        if not self.capability_claim_eligible:
            return {
                "points": len(self.runs),
                "measured_points": 0,
                "endpoint_delta": None,
                "claim_eligible": False,
                "direction": "claim_ineligible_evidence_class",
            }
        return analyze_gap_trend(self.runs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "evidence_class": self.evidence_class,
            "capability_claim_eligible": self.capability_claim_eligible,
            "retention": {
                "max_entries": self.max_entries,
                "pruned_count": self.pruned_count,
                "pruned_through_sha256": self.pruned_through_sha256,
                "retains_outputs_in_content_addressed_blobs": True,
            },
            "head_entry_sha256": (
                self.runs[-1]["entry_sha256"]
                if self.runs
                else self.pruned_through_sha256
            ),
            "runs": copy.deepcopy(self.runs),
            "trend": self.trend(),
        }

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        *,
        evidence_class: str | None = None,
        evidence_blob_resolver: Callable[[str], dict[str, Any]] | None = None,
        trusted_evaluator_keys: Mapping[str, str] | None = None,
        trusted_worker_keys: Mapping[str, str] | None = None,
        trusted_verifiers: Mapping[str, Mapping[str, str]] | None = None,
        trusted_run_keys: Mapping[str, str] | None = None,
        trusted_release_keys: Mapping[str, str] | None = None,
        source_tree_resolver: Callable[[str], str] | None = None,
        source_component_resolver: Callable[[str, str], str] | None = None,
        **_legacy_rejected: Any,
    ) -> GapLedger:
        if not isinstance(d, dict) or d.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("gap ledger schema is not v5")
        required = {
            "schema_version",
            "evidence_class",
            "capability_claim_eligible",
            "retention",
            "head_entry_sha256",
            "runs",
            "trend",
        }
        if set(d) != required:
            raise ValueError("gap ledger fields are malformed")
        stored_class = d.get("evidence_class")
        if stored_class not in SUPPORTED_EVIDENCE_CLASSES:
            raise ValueError("gap ledger evidence class is invalid")
        if evidence_class is not None and evidence_class != stored_class:
            raise ValueError("stored ledger class does not match requested class")
        expected_eligibility = stored_class == CAPABILITY_EVIDENCE_CLASS
        if d.get("capability_claim_eligible") is not expected_eligibility:
            raise ValueError("stored ledger eligibility contradicts its class")
        retention = d.get("retention")
        if not isinstance(retention, dict) or set(retention) != {
            "max_entries",
            "pruned_count",
            "pruned_through_sha256",
            "retains_outputs_in_content_addressed_blobs",
        }:
            raise ValueError("gap ledger retention metadata is malformed")
        if retention.get("retains_outputs_in_content_addressed_blobs") is not True:
            raise ValueError("gap ledger does not retain auditable evidence blobs")
        max_entries = retention.get("max_entries")
        pruned_count = retention.get("pruned_count")
        pruned_through = retention.get("pruned_through_sha256")
        if (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or not 1 <= max_entries <= MAX_LEDGER_ENTRIES
        ):
            raise ValueError("gap ledger retention bound is invalid")
        if (
            isinstance(pruned_count, bool)
            or not isinstance(pruned_count, int)
            or pruned_count < 0
        ):
            raise ValueError("gap ledger pruned count is invalid")
        if (pruned_count == 0) != (pruned_through is None):
            raise ValueError("gap ledger prune anchor is inconsistent")
        runs = validate_index_chain(
            d.get("runs"),
            max_entries=max_entries,
            initial_previous_sha256=pruned_through,
        )
        expected_head = runs[-1]["entry_sha256"] if runs else pruned_through
        if d.get("head_entry_sha256") != expected_head:
            raise ValueError("gap ledger head digest is invalid")
        if runs and evidence_blob_resolver is None:
            raise ValueError("v5 ledger restore requires an evidence blob resolver")
        previous = pruned_through
        for entry in runs:
            digest = entry["evidence_sha256"]
            try:
                snapshot = evidence_blob_resolver(digest) if evidence_blob_resolver else None
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise ValueError("evidence blob cannot be resolved") from exc
            if not isinstance(snapshot, dict) or sha256_json(snapshot) != digest:
                raise ValueError("evidence blob is missing or altered")
            if snapshot.get("evidence_class") != stored_class:
                raise ValueError("evidence blob class contradicts its index")
            if expected_eligibility:
                snapshot = validate_capability_report(
                    snapshot,
                    trusted_evaluator_keys=trusted_evaluator_keys,
                    trusted_worker_keys=trusted_worker_keys,
                    trusted_verifiers=trusted_verifiers,
                    trusted_run_keys=trusted_run_keys,
                    trusted_release_keys=trusted_release_keys,
                    source_tree_resolver=source_tree_resolver,
                    source_component_resolver=source_component_resolver,
                )
            indexed_report = dict(snapshot)
            if not expected_eligibility:
                indexed_report["overall_gap"] = None
            expected_entry = make_index_entry(
                report=indexed_report,
                evidence_sha256=digest,
                previous_entry_sha256=previous,
            )
            if canonical_json_bytes(entry) != canonical_json_bytes(expected_entry):
                raise ValueError("evidence index summary contradicts its immutable blob")
            previous = entry["entry_sha256"]
        ledger = cls(
            evidence_class=str(stored_class),
            capability_claim_eligible=expected_eligibility,
            runs=runs,
            max_entries=max_entries,
            pruned_count=pruned_count,
            pruned_through_sha256=pruned_through,
        )
        if canonical_json_bytes(d.get("trend")) != canonical_json_bytes(ledger.trend()):
            raise ValueError("gap ledger trend contradicts its evidence index")
        return ledger
