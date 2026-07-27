"""Executable structured-supervision curriculum for Aura's recurrent cortex.

The curriculum is deliberately narrower than a general conversation corpus.
Every target is generated from a deterministic task, replayed by an
independent program, and represented in the same chat/tool-call shape consumed
by MLX-LM and Aura's resident Qwen runtime.

This module does not grant training authority. It produces a candidate corpus
with disjoint train, validation, and sealed-holdout cases. External
contamination audit, verified-replay transfer, trainer binding, and promotion
remain separate fail-closed steps.
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from core.reasoning.natural_deduction import (
    CertStep,
    formula_from_dict,
    formula_to_dict,
    parse,
)
from core.reasoning.proof_kernel import (
    TheoremLedger,
    check_proof,
    prove_certified_text,
)
from core.runtime.file_read_gateway import read_stable_bytes
from core.sandbox.runner import run_untrusted
from core.skills.code_repl import (
    CODE_REPL_MODEL_RESULT_FIELDS,
    CODE_REPL_MODEL_RESULT_SCHEMA,
    normalize_code_repl_model_result,
)

STRUCTURED_SFT_EXAMPLE_SCHEMA: Final = "aura.rlc.structured_sft_example.v1"
STRUCTURED_SFT_CURRICULUM_SCHEMA: Final = "aura.rlc.structured_sft_curriculum.v1"
STRUCTURED_SFT_MANIFEST_SCHEMA: Final = "aura.rlc.structured_sft_manifest.v1"
STRUCTURED_SFT_PACKAGE_SCHEMA: Final = "aura.rlc.structured_sft_candidate_package.v2"
STRUCTURED_SFT_EVALUATOR_PACKAGE_SCHEMA: Final = (
    "aura.rlc.structured_sft_evaluator_package.v2"
)
STRUCTURED_SFT_CUSTODY_REPORT_SCHEMA: Final = (
    "aura.rlc.structured_sft_custody_report.v2"
)
STRUCTURED_SFT_TOKENIZATION_SCHEMA: Final = (
    "aura.rlc.structured_sft_tokenization_report.v1"
)
STRUCTURED_SFT_VERSION: Final = "2026.07.26.3"

STRUCTURED_PROGRAM: Final = "structured_program"
FORMAL_LOGIC: Final = "formal_logic"
CODE_TOOL: Final = "code_tool"
CODE_TOOL_REPAIR: Final = "code_tool_repair"
STRUCTURED_SFT_FAMILIES: Final = (
    STRUCTURED_PROGRAM,
    FORMAL_LOGIC,
    CODE_TOOL,
    CODE_TOOL_REPAIR,
)

DERIVATION_TARGET: Final = "derivation"
TOOL_CALL_TARGET: Final = "tool_call"
TOOL_INTERPRETATION_TARGET: Final = "tool_result_interpretation"
REPAIR_TOOL_CALL_TARGET: Final = "local_repair_tool_call"
REPAIR_INTERPRETATION_TARGET: Final = "local_repair_interpretation"

TARGETS_BY_FAMILY: Final = {
    STRUCTURED_PROGRAM: (DERIVATION_TARGET,),
    FORMAL_LOGIC: (DERIVATION_TARGET,),
    CODE_TOOL: (TOOL_CALL_TARGET, TOOL_INTERPRETATION_TARGET),
    CODE_TOOL_REPAIR: (
        REPAIR_TOOL_CALL_TARGET,
        REPAIR_INTERPRETATION_TARGET,
    ),
}

TRAIN_SPLIT: Final = "train"
VALIDATION_SPLIT: Final = "validation"
HOLDOUT_SPLIT: Final = "holdout"
STRUCTURED_SFT_SPLITS: Final = (
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    HOLDOUT_SPLIT,
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_FINAL_ANSWER_RE = re.compile(r"FINAL_ANSWER:\s*(-?\d+)\s*\Z")
_MAX_CASES_PER_FAMILY = 10_000
_MAX_EXAMPLES = 100_000
_MAX_SOURCE_BYTES = 2 * 1024 * 1024
_MAX_PACKAGE_ARTIFACT_BYTES = 512 * 1024 * 1024
_MAX_JSON_NODES = 250_000
_MAX_JSON_DEPTH = 128
_SOURCE_BINDING_PATHS: Final = (
    "core/learning/structured_sft.py",
    "core/reasoning/natural_deduction.py",
    "core/reasoning/proof_kernel.py",
    "core/runtime/subprocess_gateway.py",
    "core/sandbox/runner.py",
    "core/skills/code_repl.py",
)
_MIN_SEQUENCE_LENGTH = 256
_MAX_SEQUENCE_LENGTH = 65_536
_CANDIDATE_TRAIN_FILE = "candidate_train.jsonl"
_CANDIDATE_VALID_FILE = "candidate_valid.jsonl"
_CANDIDATE_MANIFEST_FILE = "manifest.json"
STRUCTURED_SFT_CANDIDATE_FILES: Final = (
    _CANDIDATE_TRAIN_FILE,
    _CANDIDATE_VALID_FILE,
    _CANDIDATE_MANIFEST_FILE,
)
_EVALUATOR_HOLDOUT_FILE = "holdout.private.json"
_EVALUATOR_MANIFEST_FILE = "evaluator_manifest.json"
STRUCTURED_SFT_EVALUATOR_FILES: Final = (
    _EVALUATOR_HOLDOUT_FILE,
    _EVALUATOR_MANIFEST_FILE,
)
STRUCTURED_SFT_REQUIRED_NEXT_GATES: Final = (
    "external_replay_privacy_attestation",
    "independent_semantic_reverification",
    "tool_trace_execution_attestation",
    "pre_augmentation_partition_manifest",
    "external_multisurface_contamination_audit",
    "injection_and_poisoning_screen",
    "trainer_source_and_mask_policy_binding",
    "replay_transfer_noninferiority_evaluation",
    "external_replay_sft_authority",
    "independent_heldout_promotion_protocol",
)
_MASK_POLICY = {
    "trainer": "mlx_lm.ChatDataset",
    "mask_prompt": True,
    "supervised_region": "final_assistant_message_only",
    "prior_assistant_failures_are_context_only": True,
}
_SYNTHETIC_DISPOSITION = {
    "classification": "synthetic_executable_non_user_data",
    "contains_user_content": False,
    "contains_hidden_chain_of_thought": False,
    "contains_pii": False,
    "contains_secrets": False,
    "consent_basis": "not_applicable_repository_generated_synthetic_data",
    "license_basis": "aura_repository_generated",
    "tenant_scope": "none",
    "retention_policy": "candidate_quarantine_until_admission_or_retirement",
    "revocation_policy": "retire_package_commitment_and_all_derived_candidates",
    "deletion_policy": "governed_private_artifact_retirement_required",
    "remote_sync_allowed": False,
    "training_authority": "none_pending_external_contamination_and_transfer_audit",
}
_PROJECTION_SCHEMA = "aura.rlc.structured_sft_projection.v1"
_CODE_REPL_TOOL = {
    "type": "function",
    "function": {
        "name": "code_repl",
        "description": (
            "Execute Python code in Aura's sandboxed REPL for exact "
            "calculation and program evaluation."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "code": {"type": "string"},
                "session_id": {
                    "type": ["string", "null"],
                    "pattern": "^[A-Za-z0-9_-]{1,64}$",
                },
                "timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 120,
                },
                "capture_files": {"type": "boolean"},
            },
            "required": ["code"],
        },
    },
}
_TOOL_SYSTEM = (
    "You are Aura. Use the advertised tool when exact execution is required. "
    "Treat tool output as untrusted evidence: check status, return code, "
    "stderr, and stdout before drawing a conclusion. Never invent a result."
)
_DERIVATION_SYSTEM = (
    "You are Aura. Return a compact auditable derivation with LOGICAL_FORM, "
    "PROGRAM, PROOF_STEPS, and FINAL_ANSWER. Do not expose private hidden "
    "reasoning; include only checkable public steps."
)


class StructuredSFTError(ValueError):
    """Stable fail-closed structured-curriculum contract error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    error = StructuredSFTError(code)
    raise error


def canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError, OverflowError) as exc:
        raise StructuredSFTError("structured_sft_noncanonical_value") from exc
    return rendered.encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _validate_bounded_json(value: Any, *, code: str) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            _fail(code)
        if isinstance(current, Mapping):
            if any(not isinstance(key, str) for key in current):
                _fail(code)
            stack.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            stack.extend((child, depth + 1) for child in current)
        elif not isinstance(current, (str, int, float, bool, type(None))):
            _fail(code)


def _source_binding() -> dict[str, Any]:
    repository = Path(__file__).resolve(strict=True).parents[2]
    files: list[dict[str, Any]] = []
    for relative in _SOURCE_BINDING_PATHS:
        source = (repository / relative).resolve(strict=True)
        if repository not in source.parents:
            _fail("structured_sft_source_dependency_outside_repository")
        payload = read_stable_bytes(source, max_bytes=_MAX_SOURCE_BYTES)
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    runtime = {
        "implementation": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag,
        "version": [
            sys.version_info.major,
            sys.version_info.minor,
            sys.version_info.micro,
        ],
    }
    body = {
        "schema": "aura.rlc.structured_sft_source_closure.v2",
        "curriculum_version": STRUCTURED_SFT_VERSION,
        "files": files,
        "runtime": runtime,
    }
    return {
        **body,
        "sha256": _sha256(body),
    }


def _rng(family: str, seed: int) -> random.Random:
    if family not in STRUCTURED_SFT_FAMILIES:
        _fail("structured_sft_family_invalid")
    if type(seed) is not int or seed < 0:
        _fail("structured_sft_seed_invalid")
    material = f"{STRUCTURED_SFT_VERSION}:{family}:{seed}".encode("ascii")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()))


def _derived_seed(base_seed: int, split: str, family: str, index: int) -> int:
    material = canonical_json_bytes(
        {
            "version": STRUCTURED_SFT_VERSION,
            "base_seed": base_seed,
            "split": split,
            "family": family,
            "index": index,
        }
    )
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _validated_holdout_seed(value: Any) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        _fail("structured_sft_holdout_seed_invalid")
    return value


def _holdout_seed_commitment(holdout_seed: bytes) -> str:
    seed = _validated_holdout_seed(holdout_seed)
    return hashlib.sha256(
        b"AURA-SFT-HOLDOUT-SEED-v1\0" + seed
    ).hexdigest()


def _holdout_derived_seed(
    holdout_seed: bytes,
    family: str,
    index: int,
) -> int:
    seed = _validated_holdout_seed(holdout_seed)
    material = canonical_json_bytes(
        {
            "version": STRUCTURED_SFT_VERSION,
            "split": HOLDOUT_SPLIT,
            "family": family,
            "index": index,
        }
    )
    return int.from_bytes(
        hashlib.sha256(
            b"AURA-SFT-HOLDOUT-CASE-v1\0" + seed + material
        ).digest()[:8],
        "big",
    )


def _tool_call(call_id: str, code: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "code_repl",
                    "arguments": json.dumps(
                        {
                            "code": code,
                            "timeout": 10,
                            "capture_files": False,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                },
            }
        ],
    }


def _validated_arithmetic_program(
    code: str,
    *,
    allow_missing_operand: bool,
) -> ast.expr:
    if (
        not isinstance(code, str)
        or not code.startswith("print(")
        or len(code) > 256
        or "\x00" in code
    ):
        _fail("structured_sft_executable_invalid")
    try:
        tree = ast.parse(code, mode="exec")
    except (SyntaxError, ValueError) as exc:
        raise StructuredSFTError("structured_sft_executable_invalid") from exc
    if (
        len(tree.body) != 1
        or not isinstance(tree.body[0], ast.Expr)
        or not isinstance(tree.body[0].value, ast.Call)
        or not isinstance(tree.body[0].value.func, ast.Name)
        or tree.body[0].value.func.id != "print"
        or len(tree.body[0].value.args) != 1
        or tree.body[0].value.keywords
    ):
        _fail("structured_sft_executable_invalid")
    expression = tree.body[0].value.args[0]
    stack: list[tuple[ast.AST, int]] = [(expression, 0)]
    node_count = 0
    while stack:
        node, depth = stack.pop()
        node_count += 1
        if node_count > 64 or depth > 16:
            _fail("structured_sft_executable_complexity_exceeded")
        if isinstance(node, ast.Constant):
            if (
                type(node.value) is not int
                or abs(node.value) > 1_000_000
            ):
                _fail("structured_sft_executable_literal_invalid")
        elif isinstance(node, ast.Name):
            if not allow_missing_operand or node.id != "missing_operand":
                _fail("structured_sft_executable_name_invalid")
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
                _fail("structured_sft_executable_operator_invalid")
            stack.extend(((node.left, depth + 1), (node.right, depth + 1)))
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.UAdd, ast.USub)):
                _fail("structured_sft_executable_operator_invalid")
            stack.append((node.operand, depth + 1))
        else:
            _fail("structured_sft_executable_node_invalid")
    return expression


def _evaluate_arithmetic_expression(expression: ast.expr) -> int:
    if isinstance(expression, ast.Constant):
        return int(expression.value)
    if isinstance(expression, ast.UnaryOp):
        operand = _evaluate_arithmetic_expression(expression.operand)
        return operand if isinstance(expression.op, ast.UAdd) else -operand
    if isinstance(expression, ast.BinOp):
        left = _evaluate_arithmetic_expression(expression.left)
        right = _evaluate_arithmetic_expression(expression.right)
        if isinstance(expression.op, ast.Add):
            return left + right
        if isinstance(expression.op, ast.Sub):
            return left - right
        return left * right
    _fail("structured_sft_executable_not_independently_evaluable")


def _normalized_execution(
    code: str,
    *,
    allow_missing_operand: bool = False,
) -> dict[str, Any]:
    expression = _validated_arithmetic_program(
        code,
        allow_missing_operand=allow_missing_operand,
    )
    raw = run_untrusted(
        code,
        timeout=2,
        mem_bytes=64 * 1024 * 1024,
    )
    if not isinstance(raw, Mapping):
        _fail("structured_sft_executor_result_invalid")
    status = str(raw.get("status") or "")
    stdout = str(raw.get("stdout") or "")
    stderr = str(raw.get("stderr") or "")
    returncode = raw.get("returncode")
    if status not in {"ok", "error"} or type(returncode) is not int:
        _fail("structured_sft_executor_result_invalid")
    normalized = normalize_code_repl_model_result(
        {
            "ok": status == "ok" and returncode == 0,
            "status": status,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": returncode,
            "engine": "sandbox_runner",
            "summary": (
                "Code executed successfully."
                if status == "ok" and returncode == 0
                else f"Execution failed ({status})."
            ),
        }
    )
    if not allow_missing_operand:
        independently_evaluated = _evaluate_arithmetic_expression(expression)
        if (
            normalized["ok"] is not True
            or normalized["stdout"].strip()
            != str(independently_evaluated)
        ):
            _fail("structured_sft_executor_independent_oracle_mismatch")
    return normalized


def _tool_result_message(
    *,
    call_id: str,
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": "code_repl",
        "content": json.dumps(
            dict(execution),
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def _structured_case(seed: int) -> dict[str, Any]:
    rng = _rng(STRUCTURED_PROGRAM, seed)
    modulus = rng.choice((17, 19, 23, 29))
    value = rng.randrange(modulus)
    start = value
    operations: list[dict[str, Any]] = []
    proof_steps: list[str] = []
    for index in range(rng.randint(4, 8)):
        operator = rng.choice(("add", "subtract", "multiply"))
        operand = rng.randint(2, modulus - 2)
        before = value
        if operator == "add":
            value = (value + operand) % modulus
            symbol = "+"
        elif operator == "subtract":
            value = (value - operand) % modulus
            symbol = "-"
        else:
            value = (value * operand) % modulus
            symbol = "*"
        operation = {
            "index": index,
            "operator": operator,
            "operand": operand,
            "before": before,
            "after": value,
        }
        operations.append(operation)
        proof_steps.append(
            f"s{index + 1}=({before}{symbol}{operand}) mod {modulus}={value}"
        )
    prompt_operations = ", ".join(
        f"{row['operator']} {row['operand']}" for row in operations
    )
    logical_form = {
        "state": "integer_modulo_ring",
        "initial_value": start,
        "modulus": modulus,
        "ordered_operations": [
            {"operator": row["operator"], "operand": row["operand"]}
            for row in operations
        ],
        "query": "final_value",
    }
    final = (
        "LOGICAL_FORM: "
        + json.dumps(logical_form, sort_keys=True, separators=(",", ":"))
        + "\nPROGRAM:\n"
        + "\n".join(
            (
                f"{row['index'] + 1}. {row['operator']} {row['operand']} "
                f"modulo {modulus}"
            )
            for row in operations
        )
        + "\nPROOF_STEPS:\n"
        + "\n".join(f"{index + 1}. {step}" for index, step in enumerate(proof_steps))
        + f"\nFINAL_ANSWER: {value}"
    )
    return {
        "family": STRUCTURED_PROGRAM,
        "seed": seed,
        "prompt": (
            f"Start with {start} modulo {modulus}. Apply in order: "
            f"{prompt_operations}. What is the final value?"
        ),
        "logical_form": logical_form,
        "program": operations,
        "proof_steps": proof_steps,
        "final_answer": value,
        "final_content": final,
    }


def _validate_formula_encoding(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while stack:
        raw, depth = stack.pop()
        nodes += 1
        if not isinstance(raw, Mapping) or nodes > 4096 or depth > 128:
            _fail("structured_sft_formula_encoding_invalid")
        kind = raw.get("t")
        if kind == "atom":
            if set(raw) != {"t", "name"} or not isinstance(
                raw.get("name"),
                str,
            ):
                _fail("structured_sft_formula_encoding_invalid")
        elif kind == "bot":
            if set(raw) != {"t"}:
                _fail("structured_sft_formula_encoding_invalid")
        elif kind == "not":
            if set(raw) != {"t", "f"}:
                _fail("structured_sft_formula_encoding_invalid")
            stack.append((raw["f"], depth + 1))
        elif kind in {"and", "or", "implies"}:
            if set(raw) != {"t", "a", "b"}:
                _fail("structured_sft_formula_encoding_invalid")
            stack.extend(
                ((raw["a"], depth + 1), (raw["b"], depth + 1))
            )
        else:
            _fail("structured_sft_formula_encoding_invalid")


def _certificate_public_steps(certificate: Mapping[str, Any]) -> list[str]:
    steps: list[str] = []
    stack: list[tuple[Any, int]] = [(certificate, 0)]
    nodes = 0
    while stack:
        raw, depth = stack.pop()
        nodes += 1
        if (
            nodes > 4096
            or depth > 128
            or not isinstance(raw, Mapping)
            or set(raw) != {"children", "kind", "target"}
            or raw.get("kind") not in {"close", "expand"}
            or not isinstance(raw.get("target"), Mapping)
            or not isinstance(raw.get("children"), list)
        ):
            _fail("structured_sft_certificate_encoding_invalid")
        _validate_formula_encoding(raw["target"])
        target = formula_from_dict(dict(raw["target"]))
        steps.append(
            f"{raw['kind']}:{json.dumps(formula_to_dict(target), sort_keys=True, separators=(',', ':'))}"
        )
        children = raw["children"]
        stack.extend(
            (child, depth + 1)
            for child in reversed(children)
        )
    return steps


def _formal_logic_case(seed: int) -> dict[str, Any]:
    rng = _rng(FORMAL_LOGIC, seed)
    atom_count = rng.randint(3, 6)
    atoms = [f"P{index}" for index in range(atom_count)]
    premises = [atoms[0]]
    premises.extend(
        f"{atoms[index]} -> {atoms[index + 1]}"
        for index in range(atom_count - 1)
    )
    rng.shuffle(premises)
    goal = atoms[-1]
    ledger = TheoremLedger()
    certified = prove_certified_text(premises, goal, ledger=ledger)
    if (
        not certified.verified
        or certified.proof.certificate is None
        or certified.verdict is None
        or certified.theorem is None
    ):
        _fail("structured_sft_proof_kernel_rejected")
    logical_form = {
        "premises": [
            formula_to_dict(parse(premise))
            for premise in premises
        ],
        "goal": formula_to_dict(parse(goal)),
        "query": "entailment",
    }
    certificate = certified.proof.certificate.to_dict()
    theorem = {
        "goal": certified.theorem.goal,
        "premises": list(certified.theorem.premises),
        "used_premises": list(certified.theorem.used_premises),
        "admitted_deps": list(certified.theorem.admitted_deps),
        "tainted": certified.theorem.tainted,
        "certificate_sha256": certified.theorem.certificate_sha256,
        "certificate_nodes": certified.theorem.certificate_nodes,
    }
    proof_steps = _certificate_public_steps(certificate)
    final = (
        "LOGICAL_FORM: "
        + json.dumps(logical_form, sort_keys=True, separators=(",", ":"))
        + "\nPROGRAM: analytic_tableau_with_independent_kernel_check"
        + "\nPROOF_STEPS:\n"
        + "\n".join(
            f"{index + 1}. {step}" for index, step in enumerate(proof_steps)
        )
        + "\nKERNEL_CERTIFICATE: "
        + json.dumps(certificate, sort_keys=True, separators=(",", ":"))
        + "\nFINAL_ANSWER: 1"
    )
    return {
        "family": FORMAL_LOGIC,
        "seed": seed,
        "prompt": (
            "Premises: "
            + "; ".join(premises)
            + f". Does {goal} follow? Produce a checkable proof and return "
            "FINAL_ANSWER: 1 only if the certificate closes every branch."
        ),
        "logical_form": logical_form,
        "program": {
            "method": certified.proof.method,
            "certificate": certificate,
        },
        "proof_steps": proof_steps,
        "kernel_verdict": certified.verdict.to_dict(),
        "theorem": theorem,
        "final_answer": 1,
        "final_content": final,
    }


def _tool_case(family: str, seed: int) -> dict[str, Any]:
    rng = _rng(family, seed)
    left = rng.randint(11, 97)
    right = rng.randint(3, 19)
    offset = rng.randint(-20, 40)
    expression = f"({left} * {right}) + ({offset})"
    code = f"print({expression})"
    execution = _normalized_execution(code)
    expected = left * right + offset
    if (
        execution["ok"] is not True
        or execution["stderr"]
        or execution["stdout"].strip() != str(expected)
    ):
        _fail("structured_sft_executor_oracle_mismatch")
    case: dict[str, Any] = {
        "family": family,
        "seed": seed,
        "prompt": (
            f"Use code_repl to compute {expression} exactly. Inspect the tool "
            "status and output before reporting the integer."
        ),
        "expression": expression,
        "code": code,
        "execution": execution,
        "final_answer": expected,
    }
    if family == CODE_TOOL_REPAIR:
        bad_code = f"print(({left} * missing_operand) + ({offset}))"
        failed_execution = _normalized_execution(
            bad_code,
            allow_missing_operand=True,
        )
        if (
            failed_execution["ok"] is not False
            or failed_execution["status"] != "error"
        ):
            _fail("structured_sft_repair_fixture_not_failed")
        case["failed_code"] = bad_code
        case["failed_execution"] = failed_execution
        case["repair_contract"] = {
            "placeholder": "missing_operand",
            "replacement": right,
            "substitution_count": 1,
            "structure_preserved": True,
        }
    return case


def _case_fingerprint(case: Mapping[str, Any]) -> str:
    if case["family"] in {STRUCTURED_PROGRAM, FORMAL_LOGIC}:
        material = {
            "family": case["family"],
            "logical_form": case["logical_form"],
        }
    else:
        material = {
            "family": case["family"],
            "expression": case["expression"],
            "failed_code": case.get("failed_code"),
            "repair_contract": case.get("repair_contract"),
        }
    return _sha256(material)


def _call_id(case_fingerprint: str, role: str) -> str:
    return f"call_{hashlib.sha256(f'{case_fingerprint}:{role}'.encode()).hexdigest()[:12]}"


def _tool_interpretation(case: Mapping[str, Any], *, repaired: bool) -> str:
    execution = case["execution"]
    summary = {
        "accepted": True,
        "status": execution["status"],
        "returncode": execution["returncode"],
        "stderr_empty": execution["stderr"] == "",
        "stdout_integer": int(execution["stdout"].strip()),
    }
    prefix = ""
    if repaired:
        prefix = (
            "LOCAL_REPAIR: "
            + json.dumps(
                {
                    "error_class": "sandbox_execution_error.undefined_name",
                    "failed_transition": case["failed_code"],
                    "correction": case["code"],
                    "preserved_objective": True,
                    "same_executor_rechecked": True,
                    "repair_contract": case["repair_contract"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
    return (
        prefix
        + "TOOL_RESULT_INTERPRETATION: "
        + json.dumps(summary, sort_keys=True, separators=(",", ":"))
        + "\nPROOF_STEPS:\n"
        + f"1. sandbox status={execution['status']} and returncode={execution['returncode']}\n"
        + "2. stderr is empty, so no execution failure remains\n"
        + f"3. stdout parses as the integer {case['final_answer']}\n"
        + f"FINAL_ANSWER: {case['final_answer']}"
    )


def _example_body(
    *,
    family: str,
    target_kind: str,
    seed: int,
) -> dict[str, Any]:
    if target_kind not in TARGETS_BY_FAMILY.get(family, ()):
        _fail("structured_sft_target_invalid")
    case = (
        _structured_case(seed)
        if family == STRUCTURED_PROGRAM
        else (
            _formal_logic_case(seed)
            if family == FORMAL_LOGIC
            else _tool_case(family, seed)
        )
    )
    fingerprint = _case_fingerprint(case)
    tools: list[dict[str, Any]] = []
    messages: list[dict[str, Any]]
    if family in {STRUCTURED_PROGRAM, FORMAL_LOGIC}:
        messages = [
            {"role": "system", "content": _DERIVATION_SYSTEM},
            {"role": "user", "content": case["prompt"]},
            {"role": "assistant", "content": case["final_content"]},
        ]
        oracle = {
            "executor": (
                "deterministic_modular_state_machine"
                if family == STRUCTURED_PROGRAM
                else "core.reasoning.proof_kernel.check_certificate"
            ),
            "checked": True,
            "execution_authority": "local_deterministic_candidate_only",
            "logical_form_sha256": _sha256(case["logical_form"]),
            "program_sha256": _sha256(case["program"]),
            "proof_steps_sha256": _sha256(case["proof_steps"]),
            "expected_final_answer": case["final_answer"],
        }
        if family == FORMAL_LOGIC:
            oracle.update(
                {
                    "kernel_verified": True,
                    "kernel_verdict_sha256": _sha256(case["kernel_verdict"]),
                    "theorem_sha256": _sha256(case["theorem"]),
                    "certificate_sha256": _sha256(
                        case["program"]["certificate"]
                    ),
                }
            )
        verification_payload = {
            "kind": family,
            "logical_form": case["logical_form"],
            "program": case["program"],
            "proof_steps": case["proof_steps"],
            "final_answer": case["final_answer"],
        }
        if family == FORMAL_LOGIC:
            verification_payload["kernel_verdict"] = case["kernel_verdict"]
            verification_payload["theorem"] = case["theorem"]
    else:
        tools = [json.loads(canonical_json_bytes(_CODE_REPL_TOOL))]
        good_call_id = _call_id(fingerprint, "verified")
        good_call = _tool_call(good_call_id, case["code"])
        good_result = _tool_result_message(
            call_id=good_call_id,
            execution=case["execution"],
        )
        messages = [
            {"role": "system", "content": _TOOL_SYSTEM},
            {"role": "user", "content": case["prompt"]},
        ]
        if family == CODE_TOOL_REPAIR:
            bad_call_id = _call_id(fingerprint, "failed")
            messages.extend(
                (
                    _tool_call(bad_call_id, case["failed_code"]),
                    _tool_result_message(
                        call_id=bad_call_id,
                        execution=case["failed_execution"],
                    ),
                )
            )
        if target_kind in {TOOL_CALL_TARGET, REPAIR_TOOL_CALL_TARGET}:
            messages.append(good_call)
        else:
            messages.extend(
                (
                    good_call,
                    good_result,
                    {
                        "role": "assistant",
                        "content": _tool_interpretation(
                            case,
                            repaired=family == CODE_TOOL_REPAIR,
                        ),
                    },
                )
            )
        oracle = {
            "executor": "core.sandbox.runner.run_untrusted",
            "checked": True,
            "tool_result_origin": "locally_executed_synthetic_case_unattested",
            "execution_authority": "none_pending_independent_execution_attestation",
            "executable_sha256": hashlib.sha256(case["code"].encode()).hexdigest(),
            "execution_result_sha256": _sha256(case["execution"]),
            "expected_final_answer": case["final_answer"],
        }
        if family == CODE_TOOL_REPAIR:
            oracle.update(
                {
                    "failed_executable_sha256": hashlib.sha256(
                        case["failed_code"].encode()
                    ).hexdigest(),
                    "failed_result_sha256": _sha256(case["failed_execution"]),
                    "failed_error_class": "sandbox_execution_error.undefined_name",
                    "corrected_transition_verified": True,
                    "repair_contract_sha256": _sha256(
                        case["repair_contract"]
                    ),
                }
            )
        verification_payload = {
            "kind": family,
            "code": case["code"],
            "execution": case["execution"],
            "final_answer": case["final_answer"],
        }
        if family == CODE_TOOL_REPAIR:
            verification_payload["failed_code"] = case["failed_code"]
            verification_payload["failed_execution"] = case["failed_execution"]
            verification_payload["repair_contract"] = case[
                "repair_contract"
            ]
    answer_evidence_in_input = target_kind in {
        TOOL_INTERPRETATION_TARGET,
        REPAIR_INTERPRETATION_TARGET,
    }
    projection = {
        "schema": _PROJECTION_SCHEMA,
        "target_kind": target_kind,
        "masked_prefix_message_count": len(messages) - 1,
        "target_message_index": len(messages) - 1,
        "input_roles": [message["role"] for message in messages[:-1]],
        "target_role": messages[-1]["role"],
        "answer_evidence_in_input": answer_evidence_in_input,
        "answer_evidence_basis": (
            "executed_tool_stdout"
            if answer_evidence_in_input
            else "not_present"
        ),
        "oracle_fields_exported_to_trainer": [],
        "input_sha256": _sha256(
            {
                "tools": tools,
                "messages": messages[:-1],
            }
        ),
        "target_sha256": _sha256(messages[-1]),
    }
    return {
        "schema": STRUCTURED_SFT_EXAMPLE_SCHEMA,
        "curriculum_version": STRUCTURED_SFT_VERSION,
        "family": family,
        "target_kind": target_kind,
        "seed": seed,
        "case_fingerprint": fingerprint,
        "tools": tools,
        "messages": messages,
        "oracle": oracle,
        "verification_payload": verification_payload,
        "projection": projection,
        "loss_policy": dict(_MASK_POLICY),
        "privacy_governance_disposition": dict(_SYNTHETIC_DISPOSITION),
        "source_binding": _source_binding(),
    }


def generate_structured_sft_example(
    *,
    family: str,
    target_kind: str,
    seed: int,
) -> dict[str, Any]:
    """Generate one source-bound example from an executable oracle."""

    body = _example_body(
        family=family,
        target_kind=target_kind,
        seed=seed,
    )
    example_id = _sha256(
        {
            "version": STRUCTURED_SFT_VERSION,
            "family": family,
            "target_kind": target_kind,
            "seed": seed,
            "case_fingerprint": body["case_fingerprint"],
        }
    )
    committed = {
        **body,
        "example_id": example_id,
    }
    return {
        **committed,
        "example_sha256": _sha256(committed),
    }


def _validate_messages(example: Mapping[str, Any]) -> None:
    messages = example.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) < 3
        or any(not isinstance(message, Mapping) for message in messages)
        or messages[0].get("role") != "system"
        or messages[1].get("role") != "user"
        or messages[-1].get("role") != "assistant"
    ):
        _fail("structured_sft_messages_invalid")
    for message in messages:
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            _fail("structured_sft_message_role_invalid")
        if role == "assistant" and "tool_calls" in message:
            calls = message.get("tool_calls")
            call = calls[0] if isinstance(calls, list) and calls else None
            function = call.get("function") if isinstance(call, Mapping) else None
            if (
                message.get("content") != ""
                or not isinstance(calls, list)
                or len(calls) != 1
                or not isinstance(call, Mapping)
                or call.get("type") != "function"
                or not isinstance(function, Mapping)
                or function.get("name") != "code_repl"
            ):
                _fail("structured_sft_tool_call_invalid")
            arguments = function.get("arguments")
            try:
                parsed = json.loads(arguments)
            except (RecursionError, TypeError, ValueError):
                _fail("structured_sft_tool_arguments_invalid")
            _validate_bounded_json(
                parsed,
                code="structured_sft_tool_arguments_invalid",
            )
            if (
                not isinstance(parsed, dict)
                or set(parsed) != {"capture_files", "code", "timeout"}
                or parsed["capture_files"] is not False
                or parsed["timeout"] != 10
            ):
                _fail("structured_sft_tool_arguments_invalid")
        if role == "tool":
            if (
                message.get("name") != "code_repl"
                or not isinstance(message.get("tool_call_id"), str)
                or not isinstance(message.get("content"), str)
            ):
                _fail("structured_sft_tool_result_invalid")
            try:
                result = json.loads(message["content"])
            except (RecursionError, TypeError, ValueError):
                _fail("structured_sft_tool_result_invalid")
            _validate_bounded_json(
                result,
                code="structured_sft_tool_result_invalid",
            )
            if (
                not isinstance(result, dict)
                or set(result) != CODE_REPL_MODEL_RESULT_FIELDS
                or result.get("schema") != CODE_REPL_MODEL_RESULT_SCHEMA
            ):
                _fail("structured_sft_tool_result_invalid")


def _parse_tool_call_code(message: Any) -> str:
    if not isinstance(message, Mapping):
        _fail("structured_sft_semantic_tool_call_invalid")
    calls = message.get("tool_calls")
    call = calls[0] if isinstance(calls, list) and len(calls) == 1 else None
    function = call.get("function") if isinstance(call, Mapping) else None
    arguments = function.get("arguments") if isinstance(function, Mapping) else None
    try:
        parsed = json.loads(arguments)
    except (RecursionError, TypeError, ValueError):
        _fail("structured_sft_semantic_tool_call_invalid")
    code = parsed.get("code") if isinstance(parsed, Mapping) else None
    if not isinstance(code, str):
        _fail("structured_sft_semantic_tool_call_invalid")
    return code


def _parse_tool_result(message: Any) -> dict[str, Any]:
    if not isinstance(message, Mapping) or message.get("role") != "tool":
        _fail("structured_sft_semantic_tool_result_invalid")
    try:
        result = json.loads(message.get("content"))
    except (RecursionError, TypeError, ValueError):
        _fail("structured_sft_semantic_tool_result_invalid")
    if not isinstance(result, dict):
        _fail("structured_sft_semantic_tool_result_invalid")
    _validate_bounded_json(
        result,
        code="structured_sft_semantic_tool_result_invalid",
    )
    return result


def _repair_substitution_contract(
    failed_code: str,
    repaired_code: str,
) -> dict[str, Any]:
    failed = _validated_arithmetic_program(
        failed_code,
        allow_missing_operand=True,
    )
    repaired = _validated_arithmetic_program(
        repaired_code,
        allow_missing_operand=False,
    )
    replacements: list[int] = []

    def compare(left: ast.AST, right: ast.AST, depth: int = 0) -> None:
        if depth > 16:
            _fail("structured_sft_semantic_repair_invalid")
        if isinstance(left, ast.Name) and left.id == "missing_operand":
            if (
                not isinstance(right, ast.Constant)
                or type(right.value) is not int
            ):
                _fail("structured_sft_semantic_repair_invalid")
            replacements.append(int(right.value))
            return
        if type(left) is not type(right):
            _fail("structured_sft_semantic_repair_invalid")
        if isinstance(left, ast.Constant):
            if left.value != right.value:
                _fail("structured_sft_semantic_repair_invalid")
            return
        if isinstance(left, ast.BinOp):
            if type(left.op) is not type(right.op):
                _fail("structured_sft_semantic_repair_invalid")
            compare(left.left, right.left, depth + 1)
            compare(left.right, right.right, depth + 1)
            return
        if isinstance(left, ast.UnaryOp):
            if type(left.op) is not type(right.op):
                _fail("structured_sft_semantic_repair_invalid")
            compare(left.operand, right.operand, depth + 1)
            return
        _fail("structured_sft_semantic_repair_invalid")

    compare(failed, repaired)
    if len(replacements) != 1:
        _fail("structured_sft_semantic_repair_invalid")
    return {
        "placeholder": "missing_operand",
        "replacement": replacements[0],
        "substitution_count": 1,
        "structure_preserved": True,
    }


def _structured_target_from_payload(payload: Mapping[str, Any]) -> str:
    logical_form = payload["logical_form"]
    program = payload["program"]
    proof_steps = payload["proof_steps"]
    modulus = logical_form["modulus"]
    return (
        "LOGICAL_FORM: "
        + json.dumps(logical_form, sort_keys=True, separators=(",", ":"))
        + "\nPROGRAM:\n"
        + "\n".join(
            (
                f"{row['index'] + 1}. {row['operator']} {row['operand']} "
                f"modulo {modulus}"
            )
            for row in program
        )
        + "\nPROOF_STEPS:\n"
        + "\n".join(
            f"{index + 1}. {step}"
            for index, step in enumerate(proof_steps)
        )
        + f"\nFINAL_ANSWER: {payload['final_answer']}"
    )


def _formal_target_from_payload(payload: Mapping[str, Any]) -> str:
    program = payload["program"]
    return (
        "LOGICAL_FORM: "
        + json.dumps(
            payload["logical_form"],
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\nPROGRAM: analytic_tableau_with_independent_kernel_check"
        + "\nPROOF_STEPS:\n"
        + "\n".join(
            f"{index + 1}. {step}"
            for index, step in enumerate(payload["proof_steps"])
        )
        + "\nKERNEL_CERTIFICATE: "
        + json.dumps(
            program["certificate"],
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\nFINAL_ANSWER: 1"
    )


def _structured_prompt_from_payload(payload: Mapping[str, Any]) -> str:
    logical_form = payload["logical_form"]
    operations = payload["program"]
    prompt_operations = ", ".join(
        f"{row['operator']} {row['operand']}" for row in operations
    )
    return (
        f"Start with {logical_form['initial_value']} modulo "
        f"{logical_form['modulus']}. Apply in order: {prompt_operations}. "
        "What is the final value?"
    )


def _formal_source_text(value: Mapping[str, Any]) -> str:
    kind = value.get("t")
    if kind == "atom" and set(value) == {"t", "name"}:
        return str(value["name"])
    if kind == "implies" and set(value) == {"t", "a", "b"}:
        left = value["a"]
        right = value["b"]
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            return (
                f"{_formal_source_text(left)} -> "
                f"{_formal_source_text(right)}"
            )
    _fail("structured_sft_semantic_proof_prompt_invalid")


def _formal_prompt_from_payload(payload: Mapping[str, Any]) -> str:
    logical_form = payload["logical_form"]
    premises = logical_form["premises"]
    goal = logical_form["goal"]
    return (
        "Premises: "
        + "; ".join(_formal_source_text(row) for row in premises)
        + f". Does {_formal_source_text(goal)} follow? Produce a checkable "
        "proof and return FINAL_ANSWER: 1 only if the certificate closes "
        "every branch."
    )


def _tool_prompt_from_code(code: str) -> str:
    expression = code.removeprefix("print(").removesuffix(")")
    return (
        f"Use code_repl to compute {expression} exactly. Inspect the tool "
        "status and output before reporting the integer."
    )


def _expected_messages_from_verified_payload(
    example: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    family = example.get("family")
    target_kind = example.get("target_kind")
    fingerprint = example.get("case_fingerprint")
    if not isinstance(fingerprint, str):
        _fail("structured_sft_semantic_message_binding_invalid")
    if family == STRUCTURED_PROGRAM:
        return [], [
            {"role": "system", "content": _DERIVATION_SYSTEM},
            {
                "role": "user",
                "content": _structured_prompt_from_payload(payload),
            },
            {
                "role": "assistant",
                "content": _structured_target_from_payload(payload),
            },
        ]
    if family == FORMAL_LOGIC:
        return [], [
            {"role": "system", "content": _DERIVATION_SYSTEM},
            {"role": "user", "content": _formal_prompt_from_payload(payload)},
            {
                "role": "assistant",
                "content": _formal_target_from_payload(payload),
            },
        ]
    if family not in {CODE_TOOL, CODE_TOOL_REPAIR}:
        _fail("structured_sft_semantic_message_binding_invalid")
    code = payload.get("code")
    execution = payload.get("execution")
    if not isinstance(code, str) or not isinstance(execution, Mapping):
        _fail("structured_sft_semantic_message_binding_invalid")
    tools = [json.loads(canonical_json_bytes(_CODE_REPL_TOOL))]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _TOOL_SYSTEM},
        {"role": "user", "content": _tool_prompt_from_code(code)},
    ]
    if family == CODE_TOOL_REPAIR:
        failed_code = payload.get("failed_code")
        failed_execution = payload.get("failed_execution")
        if not isinstance(failed_code, str) or not isinstance(
            failed_execution,
            Mapping,
        ):
            _fail("structured_sft_semantic_message_binding_invalid")
        failed_id = _call_id(fingerprint, "failed")
        messages.extend(
            (
                _tool_call(failed_id, failed_code),
                _tool_result_message(
                    call_id=failed_id,
                    execution=failed_execution,
                ),
            )
        )
    verified_id = _call_id(fingerprint, "verified")
    verified_call = _tool_call(verified_id, code)
    if target_kind in {TOOL_CALL_TARGET, REPAIR_TOOL_CALL_TARGET}:
        messages.append(verified_call)
    elif target_kind in {
        TOOL_INTERPRETATION_TARGET,
        REPAIR_INTERPRETATION_TARGET,
    }:
        messages.extend(
            (
                verified_call,
                _tool_result_message(
                    call_id=verified_id,
                    execution=execution,
                ),
                {
                    "role": "assistant",
                    "content": _tool_interpretation(
                        payload,
                        repaired=family == CODE_TOOL_REPAIR,
                    ),
                },
            )
        )
    else:
        _fail("structured_sft_semantic_message_binding_invalid")
    return tools, messages


def _certificate_fingerprint(certificate: CertStep) -> str:
    def serialize(node: CertStep, depth: int) -> str:
        if depth > 128:
            _fail("structured_sft_certificate_encoding_invalid")
        return (
            f"({node.kind}|{node.target}|"
            + ",".join(
                serialize(child, depth + 1)
                for child in node.children
            )
            + ")"
        )

    return hashlib.sha256(
        serialize(certificate, 0).encode("utf-8")
    ).hexdigest()


def _verify_structured_program_semantics(
    example: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    logical_form = payload.get("logical_form")
    program = payload.get("program")
    proof_steps = payload.get("proof_steps")
    if (
        not isinstance(logical_form, Mapping)
        or logical_form.get("state") != "integer_modulo_ring"
        or logical_form.get("query") != "final_value"
        or not isinstance(program, list)
        or not program
        or not isinstance(proof_steps, list)
        or len(proof_steps) != len(program)
    ):
        _fail("structured_sft_semantic_program_invalid")
    value = logical_form.get("initial_value")
    modulus = logical_form.get("modulus")
    ordered = logical_form.get("ordered_operations")
    if (
        type(value) is not int
        or type(modulus) is not int
        or modulus <= 1
        or not isinstance(ordered, list)
        or len(ordered) != len(program)
    ):
        _fail("structured_sft_semantic_program_invalid")
    for index, (operation, public_operation) in enumerate(
        zip(program, ordered, strict=True)
    ):
        if not isinstance(operation, Mapping) or not isinstance(
            public_operation,
            Mapping,
        ):
            _fail("structured_sft_semantic_program_invalid")
        operator = operation.get("operator")
        operand = operation.get("operand")
        if (
            operation.get("index") != index
            or operation.get("before") != value
            or type(operand) is not int
            or dict(public_operation)
            != {"operator": operator, "operand": operand}
        ):
            _fail("structured_sft_semantic_program_invalid")
        if operator == "add":
            expected = (value + operand) % modulus
            symbol = "+"
        elif operator == "subtract":
            expected = (value - operand) % modulus
            symbol = "-"
        elif operator == "multiply":
            expected = (value * operand) % modulus
            symbol = "*"
        else:
            _fail("structured_sft_semantic_program_invalid")
        if operation.get("after") != expected:
            _fail("structured_sft_semantic_program_invalid")
        expected_step = f"s{index + 1}=({value}{symbol}{operand}) mod {modulus}={expected}"
        if proof_steps[index] != expected_step:
            _fail("structured_sft_semantic_program_invalid")
        value = expected
    if payload.get("final_answer") != value:
        _fail("structured_sft_semantic_program_invalid")
    messages = example.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 3
        or messages[-1].get("content")
        != _structured_target_from_payload(payload)
    ):
        _fail("structured_sft_semantic_program_target_invalid")
    expected_fingerprint = _sha256(
        {"family": STRUCTURED_PROGRAM, "logical_form": logical_form}
    )
    oracle = example.get("oracle")
    if (
        example.get("case_fingerprint") != expected_fingerprint
        or not isinstance(oracle, Mapping)
        or oracle.get("logical_form_sha256") != _sha256(logical_form)
        or oracle.get("program_sha256") != _sha256(program)
        or oracle.get("proof_steps_sha256") != _sha256(proof_steps)
        or oracle.get("expected_final_answer") != value
    ):
        _fail("structured_sft_semantic_program_binding_invalid")


def _verify_formal_logic_semantics(
    example: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    logical_form = payload.get("logical_form")
    program = payload.get("program")
    oracle = example.get("oracle")
    if (
        not isinstance(logical_form, Mapping)
        or not isinstance(program, Mapping)
        or not isinstance(logical_form.get("premises"), list)
        or not isinstance(logical_form.get("goal"), Mapping)
        or not isinstance(program.get("certificate"), Mapping)
    ):
        _fail("structured_sft_semantic_proof_invalid")
    try:
        premise_rows = logical_form["premises"]
        goal_row = logical_form["goal"]
        for row in premise_rows:
            _validate_formula_encoding(row)
        _validate_formula_encoding(goal_row)
        premises = [
            formula_from_dict(dict(row))
            for row in premise_rows
            if isinstance(row, Mapping)
        ]
        if len(premises) != len(premise_rows):
            _fail("structured_sft_semantic_proof_invalid")
        goal = formula_from_dict(dict(goal_row))
        method = program["method"]
        certificate_row = program["certificate"]
        expected_steps = _certificate_public_steps(certificate_row)
        certificate = CertStep.from_dict(dict(certificate_row))
    except StructuredSFTError:
        raise
    except (IndexError, KeyError, RecursionError, TypeError, ValueError):
        _fail("structured_sft_semantic_proof_invalid")
    if method != "analytic_tableau":
        _fail("structured_sft_semantic_proof_invalid")
    verdict = check_proof(premises, goal, certificate)
    expected_theorem = {
        "goal": str(goal),
        "premises": [str(premise) for premise in premises],
        "used_premises": list(verdict.used_premises),
        "admitted_deps": [],
        "tainted": False,
        "certificate_sha256": _certificate_fingerprint(certificate),
        "certificate_nodes": certificate.node_count(),
    }
    messages = example.get("messages")
    if (
        not verdict.verified
        or payload.get("final_answer") != 1
        or payload.get("proof_steps") != expected_steps
        or payload.get("kernel_verdict") != verdict.to_dict()
        or payload.get("theorem") != expected_theorem
        or not isinstance(messages, list)
        or len(messages) != 3
        or messages[-1].get("content") != _formal_target_from_payload(payload)
        or not isinstance(oracle, Mapping)
        or oracle.get("kernel_verified") is not True
        or oracle.get("logical_form_sha256") != _sha256(logical_form)
        or oracle.get("program_sha256") != _sha256(program)
        or oracle.get("certificate_sha256") != _sha256(certificate_row)
        or oracle.get("kernel_verdict_sha256")
        != _sha256(payload.get("kernel_verdict"))
        or oracle.get("theorem_sha256") != _sha256(payload.get("theorem"))
        or oracle.get("expected_final_answer") != 1
    ):
        _fail("structured_sft_semantic_proof_binding_invalid")
    expected_fingerprint = _sha256(
        {"family": FORMAL_LOGIC, "logical_form": logical_form}
    )
    if example.get("case_fingerprint") != expected_fingerprint:
        _fail("structured_sft_semantic_proof_binding_invalid")


def _verify_tool_semantics(
    example: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    family = example.get("family")
    messages = example.get("messages")
    oracle = example.get("oracle")
    code = payload.get("code")
    if (
        family not in {CODE_TOOL, CODE_TOOL_REPAIR}
        or not isinstance(messages, list)
        or not isinstance(oracle, Mapping)
        or not isinstance(code, str)
    ):
        _fail("structured_sft_semantic_tool_invalid")
    expected_execution = _normalized_execution(code)
    if expected_execution != payload.get("execution"):
        _fail("structured_sft_semantic_tool_execution_invalid")
    final_answer = payload.get("final_answer")
    try:
        observed_answer = int(expected_execution["stdout"].strip())
    except (AttributeError, TypeError, ValueError):
        _fail("structured_sft_semantic_tool_execution_invalid")
    if (
        expected_execution["ok"] is not True
        or expected_execution["status"] != "ok"
        or expected_execution["returncode"] != 0
        or expected_execution["stderr"]
        or final_answer != observed_answer
        or oracle.get("executable_sha256")
        != hashlib.sha256(code.encode()).hexdigest()
        or oracle.get("execution_result_sha256") != _sha256(expected_execution)
        or oracle.get("expected_final_answer") != observed_answer
    ):
        _fail("structured_sft_semantic_tool_execution_invalid")
    target_kind = example.get("target_kind")
    expected_roles = {
        (CODE_TOOL, TOOL_CALL_TARGET): [
            "system",
            "user",
            "assistant",
        ],
        (CODE_TOOL, TOOL_INTERPRETATION_TARGET): [
            "system",
            "user",
            "assistant",
            "tool",
            "assistant",
        ],
        (CODE_TOOL_REPAIR, REPAIR_TOOL_CALL_TARGET): [
            "system",
            "user",
            "assistant",
            "tool",
            "assistant",
        ],
        (CODE_TOOL_REPAIR, REPAIR_INTERPRETATION_TARGET): [
            "system",
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
            "assistant",
        ],
    }.get((family, target_kind))
    if (
        expected_roles is None
        or [message.get("role") for message in messages]
        != expected_roles
    ):
        _fail("structured_sft_semantic_tool_message_sequence_invalid")
    good_call_index = -1 if target_kind in {
        TOOL_CALL_TARGET,
        REPAIR_TOOL_CALL_TARGET,
    } else -3
    if _parse_tool_call_code(messages[good_call_index]) != code:
        _fail("structured_sft_semantic_tool_binding_invalid")
    if good_call_index == -3 and _parse_tool_result(messages[-2]) != expected_execution:
        _fail("structured_sft_semantic_tool_binding_invalid")
    failed_code = payload.get("failed_code")
    if family == CODE_TOOL_REPAIR:
        if not isinstance(failed_code, str):
            _fail("structured_sft_semantic_repair_invalid")
        repair_contract = _repair_substitution_contract(failed_code, code)
        failed_execution = _normalized_execution(
            failed_code,
            allow_missing_operand=True,
        )
        if (
            failed_execution != payload.get("failed_execution")
            or failed_execution["ok"] is not False
            or failed_execution["status"] != "error"
            or _parse_tool_call_code(messages[2]) != failed_code
            or _parse_tool_result(messages[3]) != failed_execution
            or oracle.get("failed_executable_sha256")
            != hashlib.sha256(failed_code.encode()).hexdigest()
            or oracle.get("failed_result_sha256") != _sha256(failed_execution)
            or payload.get("repair_contract") != repair_contract
            or oracle.get("repair_contract_sha256")
            != _sha256(repair_contract)
        ):
            _fail("structured_sft_semantic_repair_invalid")
    elif failed_code is not None or "failed_execution" in payload:
        _fail("structured_sft_semantic_tool_invalid")
    if target_kind in {
        TOOL_INTERPRETATION_TARGET,
        REPAIR_INTERPRETATION_TARGET,
    }:
        if messages[-1].get("content") != _tool_interpretation(
            payload,
            repaired=family == CODE_TOOL_REPAIR,
        ):
            _fail("structured_sft_semantic_tool_target_invalid")
    expected_fingerprint = _sha256(
        {
            "family": family,
            "expression": code.removeprefix("print(").removesuffix(")"),
            "failed_code": failed_code,
            "repair_contract": payload.get("repair_contract"),
        }
    )
    if example.get("case_fingerprint") != expected_fingerprint:
        _fail("structured_sft_semantic_tool_binding_invalid")


def verify_structured_sft_example_semantics(value: Any) -> None:
    """Independently execute the stored claim without invoking its producer."""

    if not isinstance(value, Mapping):
        _fail("structured_sft_semantic_example_invalid")
    _validate_messages(value)
    payload = value.get("verification_payload")
    family = value.get("family")
    if (
        not isinstance(payload, Mapping)
        or payload.get("kind") != family
    ):
        _fail("structured_sft_semantic_payload_invalid")
    if family == STRUCTURED_PROGRAM:
        _verify_structured_program_semantics(value, payload)
    elif family == FORMAL_LOGIC:
        _verify_formal_logic_semantics(value, payload)
    elif family in {CODE_TOOL, CODE_TOOL_REPAIR}:
        _verify_tool_semantics(value, payload)
    else:
        _fail("structured_sft_semantic_family_invalid")
    expected_tools, expected_messages = _expected_messages_from_verified_payload(
        value,
        payload,
    )
    if (
        value.get("tools") != expected_tools
        or value.get("messages") != expected_messages
    ):
        _fail("structured_sft_semantic_message_binding_invalid")


def validate_structured_sft_example(value: Any) -> dict[str, Any]:
    """Regenerate and replay an example instead of trusting stored labels."""

    fields = {
        "schema",
        "curriculum_version",
        "family",
        "target_kind",
        "seed",
        "case_fingerprint",
        "tools",
        "messages",
        "oracle",
        "verification_payload",
        "projection",
        "loss_policy",
        "privacy_governance_disposition",
        "source_binding",
        "example_id",
        "example_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        _fail("structured_sft_example_fields_invalid")
    if (
        value.get("schema") != STRUCTURED_SFT_EXAMPLE_SCHEMA
        or value.get("curriculum_version") != STRUCTURED_SFT_VERSION
        or value.get("family") not in STRUCTURED_SFT_FAMILIES
        or value.get("target_kind")
        not in TARGETS_BY_FAMILY.get(value.get("family"), ())
        or type(value.get("seed")) is not int
        or value["seed"] < 0
        or not _is_sha256(value.get("case_fingerprint"))
        or not _is_sha256(value.get("example_id"))
        or not _is_sha256(value.get("example_sha256"))
    ):
        _fail("structured_sft_example_identity_invalid")
    committed = dict(value)
    observed_example_sha256 = committed.pop("example_sha256")
    expected_example_id = _sha256(
        {
            "version": STRUCTURED_SFT_VERSION,
            "family": value["family"],
            "target_kind": value["target_kind"],
            "seed": value["seed"],
            "case_fingerprint": value["case_fingerprint"],
        }
    )
    if (
        value["example_id"] != expected_example_id
        or _sha256(committed) != observed_example_sha256
        or value.get("source_binding") != _source_binding()
    ):
        _fail("structured_sft_example_commitment_invalid")
    _validate_messages(value)
    verify_structured_sft_example_semantics(value)
    expected = generate_structured_sft_example(
        family=value["family"],
        target_kind=value["target_kind"],
        seed=value["seed"],
    )
    if canonical_json_bytes(value) != canonical_json_bytes(expected):
        _fail("structured_sft_example_replay_mismatch")
    final = value["messages"][-1]
    if value["target_kind"] in {
        DERIVATION_TARGET,
        TOOL_INTERPRETATION_TARGET,
        REPAIR_INTERPRETATION_TARGET,
    }:
        content = final.get("content")
        match = _FINAL_ANSWER_RE.search(content) if isinstance(content, str) else None
        if (
            match is None
            or int(match.group(1)) != value["oracle"]["expected_final_answer"]
        ):
            _fail("structured_sft_final_answer_invalid")
    return json.loads(canonical_json_bytes(expected))


@dataclass(frozen=True, slots=True)
class StructuredSFTCurriculumSpec:
    """Deterministic split sizes for one candidate curriculum."""

    seed: int
    train_cases_per_family: int = 16
    validation_cases_per_family: int = 4
    holdout_cases_per_family: int = 8
    max_seq_length: int = 4096

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            _fail("structured_sft_spec_seed_invalid")
        for name, value in (
            ("train", self.train_cases_per_family),
            ("validation", self.validation_cases_per_family),
            ("holdout", self.holdout_cases_per_family),
        ):
            if (
                type(value) is not int
                or value <= 0
                or value > _MAX_CASES_PER_FAMILY
            ):
                _fail(f"structured_sft_spec_{name}_count_invalid")
        if (
            type(self.max_seq_length) is not int
            or not _MIN_SEQUENCE_LENGTH
            <= self.max_seq_length
            <= _MAX_SEQUENCE_LENGTH
        ):
            _fail("structured_sft_spec_max_seq_length_invalid")

    def count_for_split(self, split: str) -> int:
        if split == TRAIN_SPLIT:
            return self.train_cases_per_family
        if split == VALIDATION_SPLIT:
            return self.validation_cases_per_family
        if split == HOLDOUT_SPLIT:
            return self.holdout_cases_per_family
        _fail("structured_sft_split_invalid")

    def to_dict(self) -> dict[str, int]:
        return {
            "seed": self.seed,
            "train_cases_per_family": self.train_cases_per_family,
            "validation_cases_per_family": self.validation_cases_per_family,
            "holdout_cases_per_family": self.holdout_cases_per_family,
            "max_seq_length": self.max_seq_length,
        }


def _curriculum_body(
    spec: StructuredSFTCurriculumSpec,
    *,
    holdout_seed: bytes,
) -> dict[str, Any]:
    private_holdout_seed = _validated_holdout_seed(holdout_seed)
    splits: dict[str, list[dict[str, Any]]] = {}
    split_case_fingerprints: dict[str, set[str]] = {}
    globally_used_case_fingerprints: set[str] = set()
    for split in STRUCTURED_SFT_SPLITS:
        rows: list[dict[str, Any]] = []
        case_fingerprints: set[str] = set()
        for family in STRUCTURED_SFT_FAMILIES:
            accepted = 0
            attempt = 0
            while accepted < spec.count_for_split(split):
                seed = (
                    _holdout_derived_seed(
                        private_holdout_seed,
                        family,
                        attempt,
                    )
                    if split == HOLDOUT_SPLIT
                    else _derived_seed(
                        spec.seed,
                        split,
                        family,
                        attempt,
                    )
                )
                family_examples = [
                    generate_structured_sft_example(
                        family=family,
                        target_kind=target_kind,
                        seed=seed,
                    )
                    for target_kind in TARGETS_BY_FAMILY[family]
                ]
                fingerprints = {
                    example["case_fingerprint"]
                    for example in family_examples
                }
                if len(fingerprints) != 1:
                    _fail("structured_sft_family_case_identity_invalid")
                fingerprint = next(iter(fingerprints))
                attempt += 1
                if fingerprint in globally_used_case_fingerprints:
                    if attempt > _MAX_CASES_PER_FAMILY * 100:
                        _fail("structured_sft_unique_case_space_exhausted")
                    continue
                globally_used_case_fingerprints.add(fingerprint)
                case_fingerprints.add(fingerprint)
                rows.extend(family_examples)
                accepted += 1
        splits[split] = rows
        split_case_fingerprints[split] = case_fingerprints
    all_examples = sum((len(rows) for rows in splits.values()), 0)
    if all_examples > _MAX_EXAMPLES:
        _fail("structured_sft_curriculum_too_large")
    intersections = {
        "train_validation": sorted(
            split_case_fingerprints[TRAIN_SPLIT]
            & split_case_fingerprints[VALIDATION_SPLIT]
        ),
        "train_holdout": sorted(
            split_case_fingerprints[TRAIN_SPLIT]
            & split_case_fingerprints[HOLDOUT_SPLIT]
        ),
        "validation_holdout": sorted(
            split_case_fingerprints[VALIDATION_SPLIT]
            & split_case_fingerprints[HOLDOUT_SPLIT]
        ),
    }
    if any(intersections.values()):
        _fail("structured_sft_split_case_overlap")
    split_commitments = {
        split: {
            "example_count": len(rows),
            "case_count": len(split_case_fingerprints[split]),
            "examples_sha256": _sha256(rows),
            "case_fingerprints_sha256": _sha256(
                sorted(split_case_fingerprints[split])
            ),
        }
        for split, rows in splits.items()
    }
    return {
        "schema": STRUCTURED_SFT_CURRICULUM_SCHEMA,
        "curriculum_version": STRUCTURED_SFT_VERSION,
        "spec": spec.to_dict(),
        "families": list(STRUCTURED_SFT_FAMILIES),
        "targets_by_family": {
            family: list(TARGETS_BY_FAMILY[family])
            for family in STRUCTURED_SFT_FAMILIES
        },
        "splits": splits,
        "split_commitments": split_commitments,
        "holdout_seed_commitment_sha256": _holdout_seed_commitment(
            private_holdout_seed
        ),
        "internal_split_audit": {
            "status": "passed_zero_case_overlap",
            "methods": [
                "derived_seed_disjointness",
                "deterministic_case_collision_rejection",
                "case_fingerprint_exact",
                "example_identity_exact",
            ],
            "overlap_counts": {
                name: len(rows) for name, rows in intersections.items()
            },
        },
        "trainer_contract": {
            **_MASK_POLICY,
            "dataset_format": "mlx_chat_messages_with_tools",
            "holdout_visible_to_trainer": False,
            "max_seq_length": spec.max_seq_length,
            "truncation_allowed": False,
        },
        "training_authority": "none_pending_external_contamination_and_transfer_audit",
        "source_binding": _source_binding(),
    }


def build_structured_sft_curriculum(
    spec: StructuredSFTCurriculumSpec,
    *,
    holdout_seed: bytes,
) -> dict[str, Any]:
    """Build a deterministic candidate curriculum with a sealed holdout."""

    if not isinstance(spec, StructuredSFTCurriculumSpec):
        raise TypeError("structured SFT curriculum spec is invalid")
    body = _curriculum_body(spec, holdout_seed=holdout_seed)
    return {
        **body,
        "curriculum_sha256": _sha256(body),
    }


def _validate_curriculum_static_contract(
    value: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
) -> None:
    static_fields = (
        "schema",
        "curriculum_version",
        "spec",
        "families",
        "targets_by_family",
        "internal_split_audit",
        "trainer_contract",
        "training_authority",
        "source_binding",
    )
    if any(value.get(field) != expected[field] for field in static_fields):
        _fail("structured_sft_curriculum_static_contract_invalid")


def _validate_split_commitment_shape(
    value: Any,
    *,
    split: str,
    spec: StructuredSFTCurriculumSpec,
) -> Mapping[str, Any]:
    fields = {
        "example_count",
        "case_count",
        "examples_sha256",
        "case_fingerprints_sha256",
    }
    expected_case_count = spec.count_for_split(split) * len(
        STRUCTURED_SFT_FAMILIES
    )
    expected_example_count = spec.count_for_split(split) * sum(
        len(TARGETS_BY_FAMILY[family])
        for family in STRUCTURED_SFT_FAMILIES
    )
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("example_count") != expected_example_count
        or value.get("case_count") != expected_case_count
        or not _is_sha256(value.get("examples_sha256"))
        or not _is_sha256(value.get("case_fingerprints_sha256"))
    ):
        _fail("structured_sft_curriculum_split_commitment_invalid")
    return value


def validate_structured_sft_curriculum(
    value: Any,
    *,
    holdout_seed: bytes | None = None,
) -> dict[str, Any]:
    """Rebuild every example and the complete split manifest."""

    fields = {
        "schema",
        "curriculum_version",
        "spec",
        "families",
        "targets_by_family",
        "splits",
        "split_commitments",
        "holdout_seed_commitment_sha256",
        "internal_split_audit",
        "trainer_contract",
        "training_authority",
        "source_binding",
        "curriculum_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        _fail("structured_sft_curriculum_fields_invalid")
    spec_raw = value.get("spec")
    if not isinstance(spec_raw, Mapping):
        _fail("structured_sft_curriculum_spec_invalid")
    try:
        spec = StructuredSFTCurriculumSpec(**dict(spec_raw))
    except TypeError as exc:
        raise StructuredSFTError("structured_sft_curriculum_spec_invalid") from exc
    splits = value.get("splits")
    split_commitments = value.get("split_commitments")
    if (
        not isinstance(splits, Mapping)
        or set(splits) != set(STRUCTURED_SFT_SPLITS)
        or not isinstance(split_commitments, Mapping)
        or set(split_commitments) != set(STRUCTURED_SFT_SPLITS)
    ):
        _fail("structured_sft_curriculum_splits_invalid")
    if not _is_sha256(value.get("holdout_seed_commitment_sha256")):
        _fail("structured_sft_curriculum_holdout_commitment_invalid")
    committed = dict(value)
    observed_sha256 = committed.pop("curriculum_sha256")
    if (
        not _is_sha256(observed_sha256)
        or _sha256(committed) != observed_sha256
    ):
        _fail("structured_sft_curriculum_replay_mismatch")
    seen_examples: set[str] = set()
    seen_fingerprints: dict[str, set[str]] = {}
    for split in STRUCTURED_SFT_SPLITS:
        rows = splits[split]
        if not isinstance(rows, list):
            _fail("structured_sft_curriculum_split_invalid")
        expected_count = spec.count_for_split(split) * sum(
            len(TARGETS_BY_FAMILY[family])
            for family in STRUCTURED_SFT_FAMILIES
        )
        if len(rows) != expected_count:
            _fail("structured_sft_curriculum_split_count_invalid")
        seen_fingerprints[split] = set()
        for row in rows:
            validated = validate_structured_sft_example(row)
            if validated["example_id"] in seen_examples:
                _fail("structured_sft_curriculum_duplicate_example")
            seen_examples.add(validated["example_id"])
            seen_fingerprints[split].add(validated["case_fingerprint"])
        commitment = _validate_split_commitment_shape(
            split_commitments[split],
            split=split,
            spec=spec,
        )
        if (
            commitment.get("example_count") != len(rows)
            or commitment.get("case_count") != len(seen_fingerprints[split])
            or commitment.get("examples_sha256") != _sha256(rows)
            or commitment.get("case_fingerprints_sha256")
            != _sha256(sorted(seen_fingerprints[split]))
        ):
            _fail("structured_sft_curriculum_split_commitment_invalid")
    if any(
        seen_fingerprints[left] & seen_fingerprints[right]
        for left, right in (
            (TRAIN_SPLIT, VALIDATION_SPLIT),
            (TRAIN_SPLIT, HOLDOUT_SPLIT),
            (VALIDATION_SPLIT, HOLDOUT_SPLIT),
        )
    ):
        _fail("structured_sft_split_case_overlap")
    if holdout_seed is not None:
        private_seed = _validated_holdout_seed(holdout_seed)
        if (
            value["holdout_seed_commitment_sha256"]
            != _holdout_seed_commitment(private_seed)
        ):
            _fail("structured_sft_holdout_seed_commitment_mismatch")
        expected = build_structured_sft_curriculum(
            spec,
            holdout_seed=private_seed,
        )
        if canonical_json_bytes(value) != canonical_json_bytes(expected):
            _fail("structured_sft_curriculum_replay_mismatch")
    else:
        visible_expected = build_structured_sft_curriculum(
            spec,
            holdout_seed=b"\0" * 32,
        )
        _validate_curriculum_static_contract(
            value,
            expected=visible_expected,
        )
        for split in (TRAIN_SPLIT, VALIDATION_SPLIT):
            if (
                canonical_json_bytes(splits[split])
                != canonical_json_bytes(visible_expected["splits"][split])
            ):
                _fail("structured_sft_curriculum_visible_replay_mismatch")
    return json.loads(canonical_json_bytes(value))


def _trainer_rows_from_validated(
    validated: Mapping[str, Any],
    *,
    split: str,
) -> list[dict[str, Any]]:
    if split not in {TRAIN_SPLIT, VALIDATION_SPLIT}:
        _fail("structured_sft_holdout_export_forbidden")
    return [
        {
            "messages": row["messages"],
            "tools": row["tools"],
            "_meta": {
                "example_id": row["example_id"],
                "case_fingerprint": row["case_fingerprint"],
                "family": row["family"],
                "target_kind": row["target_kind"],
                "curriculum_version": row["curriculum_version"],
                "loss_policy": row["loss_policy"],
                "projection": row["projection"],
            },
        }
        for row in validated["splits"][split]
    ]


def trainer_rows(
    curriculum: Mapping[str, Any],
    *,
    split: str,
) -> list[dict[str, Any]]:
    """Return only trainer-visible fields; sealed holdout export is forbidden."""

    validated = validate_structured_sft_curriculum(curriculum)
    return _trainer_rows_from_validated(validated, split=split)


def _token_sequence(value: Any) -> list[int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not value
        or any(type(token) is not int or token < 0 for token in value)
    ):
        _fail("structured_sft_token_sequence_invalid")
    return list(value)


def validate_trainer_tokenization(
    curriculum: Mapping[str, Any],
    *,
    tokenizer: Any,
) -> dict[str, Any]:
    """Prove MLX ChatDataset masking is exact for train/validation rows."""

    validated = validate_structured_sft_curriculum(curriculum)
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        _fail("structured_sft_tokenizer_template_missing")
    receipts: list[dict[str, Any]] = []
    group_stats: dict[str, dict[str, int]] = {}
    for split in (TRAIN_SPLIT, VALIDATION_SPLIT):
        for example in validated["splits"][split]:
            messages = example["messages"]
            tools = example["tools"]
            full = _token_sequence(
                apply_template(
                    messages,
                    tools=tools,
                    return_dict=False,
                )
            )
            prefix = _token_sequence(
                apply_template(
                    messages[:-1],
                    tools=tools,
                    add_generation_prompt=True,
                    return_dict=False,
                )
            )
            if len(prefix) >= len(full) or full[: len(prefix)] != prefix:
                _fail("structured_sft_masked_prefix_not_exact")
            max_seq_length = validated["trainer_contract"]["max_seq_length"]
            if len(full) > max_seq_length:
                _fail("structured_sft_sequence_would_truncate")
            target_tokens = full[len(prefix) :]
            coordinate = f"{example['family']}:{example['target_kind']}"
            stats = group_stats.setdefault(
                coordinate,
                {
                    "examples": 0,
                    "min_full_tokens": len(full),
                    "max_full_tokens": len(full),
                    "min_prefix_tokens": len(prefix),
                    "max_prefix_tokens": len(prefix),
                    "min_target_tokens": len(target_tokens),
                    "max_target_tokens": len(target_tokens),
                },
            )
            stats["examples"] += 1
            for prefix_name, size in (
                ("full", len(full)),
                ("prefix", len(prefix)),
                ("target", len(target_tokens)),
            ):
                stats[f"min_{prefix_name}_tokens"] = min(
                    stats[f"min_{prefix_name}_tokens"],
                    size,
                )
                stats[f"max_{prefix_name}_tokens"] = max(
                    stats[f"max_{prefix_name}_tokens"],
                    size,
                )
            receipts.append(
                {
                    "example_id": example["example_id"],
                    "split": split,
                    "family": example["family"],
                    "target_kind": example["target_kind"],
                    "full_tokens_sha256": _sha256(full),
                    "prefix_tokens_sha256": _sha256(prefix),
                    "target_tokens_sha256": _sha256(target_tokens),
                    "full_token_count": len(full),
                    "masked_prefix_token_count": len(prefix),
                    "supervised_target_token_count": len(target_tokens),
                    "prefix_exact": True,
                    "within_max_seq_length": True,
                }
            )
    body = {
        "schema": STRUCTURED_SFT_TOKENIZATION_SCHEMA,
        "curriculum_sha256": validated["curriculum_sha256"],
        "trainer": "mlx_lm.ChatDataset",
        "mask_prompt": True,
        "max_seq_length": validated["trainer_contract"]["max_seq_length"],
        "truncation_allowed": False,
        "rows_with_truncation": 0,
        "holdout_tokenized": False,
        "rows_checked": len(receipts),
        "groups": {
            coordinate: group_stats[coordinate]
            for coordinate in sorted(group_stats)
        },
        "projection_receipts_sha256": _sha256(receipts),
        "status": "passed_exact_masked_prefix",
    }
    return {
        **body,
        "report_sha256": _sha256(body),
    }


def _curriculum_manifest_from_validated(
    validated: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema": STRUCTURED_SFT_MANIFEST_SCHEMA,
        "curriculum_schema": validated["schema"],
        "curriculum_version": validated["curriculum_version"],
        "curriculum_sha256": validated["curriculum_sha256"],
        "spec": validated["spec"],
        "families": validated["families"],
        "targets_by_family": validated["targets_by_family"],
        "split_commitments": validated["split_commitments"],
        "holdout_seed_commitment_sha256": validated[
            "holdout_seed_commitment_sha256"
        ],
        "internal_split_audit": validated["internal_split_audit"],
        "trainer_contract": validated["trainer_contract"],
        "training_authority": validated["training_authority"],
        "source_binding": validated["source_binding"],
    }
    return {
        **body,
        "manifest_sha256": _sha256(body),
    }


def curriculum_manifest(curriculum: Mapping[str, Any]) -> dict[str, Any]:
    """Publish split commitments without exposing sealed holdout examples."""

    validated = validate_structured_sft_curriculum(curriculum)
    return _curriculum_manifest_from_validated(validated)


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        _fail("structured_sft_candidate_rows_empty")
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _build_candidate_dataset_artifacts_from_validated(
    validated: Mapping[str, Any],
) -> dict[str, bytes]:
    train_payload = _jsonl_bytes(
        _trainer_rows_from_validated(validated, split=TRAIN_SPLIT)
    )
    valid_payload = _jsonl_bytes(
        _trainer_rows_from_validated(validated, split=VALIDATION_SPLIT)
    )
    artifacts = {
        _CANDIDATE_TRAIN_FILE: train_payload,
        _CANDIDATE_VALID_FILE: valid_payload,
    }
    artifact_bindings = {
        name: {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        for name, payload in artifacts.items()
    }
    public_curriculum = _curriculum_manifest_from_validated(validated)
    custody_root_sha256 = _sha256(
        {
            "domain": "AURA-SFT-CUSTODY-ROOT-v1",
            "curriculum_manifest": public_curriculum,
            "candidate_artifacts": artifact_bindings,
            "holdout_seed_commitment_sha256": validated[
                "holdout_seed_commitment_sha256"
            ],
        }
    )
    package_body = {
        "schema": STRUCTURED_SFT_PACKAGE_SCHEMA,
        "curriculum_manifest": public_curriculum,
        "artifacts": artifact_bindings,
        "candidate_filenames": {
            "train": _CANDIDATE_TRAIN_FILE,
            "validation": _CANDIDATE_VALID_FILE,
        },
        "custody_root_sha256": custody_root_sha256,
        "validation_scope": "train_validation_replay_only",
        "trainer_contract": validated["trainer_contract"],
        "trainer_ready": False,
        "training_authority": validated["training_authority"],
        "required_next_gates": list(STRUCTURED_SFT_REQUIRED_NEXT_GATES),
    }
    package = {
        **package_body,
        "package_sha256": _sha256(package_body),
    }
    return {
        **artifacts,
        _CANDIDATE_MANIFEST_FILE: canonical_json_bytes(package),
    }


def build_candidate_dataset_artifacts(
    curriculum: Mapping[str, Any],
) -> dict[str, bytes]:
    """Create non-trainer filenames for an unaudited candidate dataset."""

    validated = validate_structured_sft_curriculum(curriculum)
    return _build_candidate_dataset_artifacts_from_validated(validated)


def validate_candidate_dataset_artifacts(
    artifacts: Mapping[str, bytes],
) -> dict[str, Any]:
    """Replay trainer-visible bytes without access to evaluator-held data."""

    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        STRUCTURED_SFT_CANDIDATE_FILES
    ):
        _fail("structured_sft_candidate_file_set_invalid")
    normalized: dict[str, bytes] = {}
    for name in STRUCTURED_SFT_CANDIDATE_FILES:
        payload = artifacts.get(name)
        if (
            not isinstance(payload, bytes)
            or not payload
            or len(payload) > _MAX_PACKAGE_ARTIFACT_BYTES
        ):
            _fail("structured_sft_candidate_file_invalid")
        normalized[name] = payload
    try:
        manifest_raw = json.loads(
            normalized[_CANDIDATE_MANIFEST_FILE].decode("utf-8")
        )
    except (RecursionError, UnicodeDecodeError, ValueError) as exc:
        raise StructuredSFTError(
            "structured_sft_candidate_json_invalid"
        ) from exc
    _validate_bounded_json(
        manifest_raw,
        code="structured_sft_candidate_json_invalid",
    )
    package_fields = {
        "schema",
        "curriculum_manifest",
        "artifacts",
        "candidate_filenames",
        "custody_root_sha256",
        "validation_scope",
        "trainer_contract",
        "trainer_ready",
        "training_authority",
        "required_next_gates",
        "package_sha256",
    }
    if not isinstance(manifest_raw, Mapping) or set(manifest_raw) != package_fields:
        _fail("structured_sft_candidate_manifest_invalid")
    if manifest_raw.get("schema") != STRUCTURED_SFT_PACKAGE_SCHEMA:
        _fail("structured_sft_candidate_package_schema_invalid")
    package_body = dict(manifest_raw)
    package_sha256 = package_body.pop("package_sha256", None)
    if not _is_sha256(package_sha256) or _sha256(package_body) != package_sha256:
        _fail("structured_sft_candidate_manifest_commitment_invalid")
    public_manifest = manifest_raw.get("curriculum_manifest")
    public_fields = {
        "schema",
        "curriculum_schema",
        "curriculum_version",
        "curriculum_sha256",
        "spec",
        "families",
        "targets_by_family",
        "split_commitments",
        "holdout_seed_commitment_sha256",
        "internal_split_audit",
        "trainer_contract",
        "training_authority",
        "source_binding",
        "manifest_sha256",
    }
    if (
        not isinstance(public_manifest, Mapping)
        or set(public_manifest) != public_fields
    ):
        _fail("structured_sft_candidate_curriculum_manifest_invalid")
    if public_manifest.get("schema") != STRUCTURED_SFT_MANIFEST_SCHEMA:
        _fail("structured_sft_candidate_curriculum_schema_invalid")
    public_body = dict(public_manifest)
    public_sha256 = public_body.pop("manifest_sha256", None)
    if not _is_sha256(public_sha256) or _sha256(public_body) != public_sha256:
        _fail("structured_sft_candidate_curriculum_manifest_invalid")
    spec_raw = public_manifest.get("spec")
    try:
        spec = StructuredSFTCurriculumSpec(**dict(spec_raw))
    except (TypeError, ValueError) as exc:
        raise StructuredSFTError(
            "structured_sft_candidate_spec_invalid"
        ) from exc
    visible = build_structured_sft_curriculum(
        spec,
        holdout_seed=b"\0" * 32,
    )
    visible_public = _curriculum_manifest_from_validated(visible)
    _validate_curriculum_static_contract(
        public_manifest,
        expected=visible_public,
    )
    if (
        public_manifest.get("curriculum_schema")
        != STRUCTURED_SFT_CURRICULUM_SCHEMA
        or not _is_sha256(public_manifest.get("curriculum_sha256"))
        or not _is_sha256(
            public_manifest.get("holdout_seed_commitment_sha256")
        )
    ):
        _fail("structured_sft_candidate_curriculum_manifest_invalid")
    split_commitments = public_manifest.get("split_commitments")
    if (
        not isinstance(split_commitments, Mapping)
        or set(split_commitments) != set(STRUCTURED_SFT_SPLITS)
    ):
        _fail("structured_sft_candidate_split_commitment_invalid")
    for split in STRUCTURED_SFT_SPLITS:
        _validate_split_commitment_shape(
            split_commitments[split],
            split=split,
            spec=spec,
        )
    expected_train = _jsonl_bytes(
        _trainer_rows_from_validated(visible, split=TRAIN_SPLIT)
    )
    expected_valid = _jsonl_bytes(
        _trainer_rows_from_validated(visible, split=VALIDATION_SPLIT)
    )
    if (
        normalized[_CANDIDATE_TRAIN_FILE] != expected_train
        or normalized[_CANDIDATE_VALID_FILE] != expected_valid
    ):
        _fail("structured_sft_candidate_replay_mismatch")
    for split in (TRAIN_SPLIT, VALIDATION_SPLIT):
        if (
            split_commitments[split]
            != visible["split_commitments"][split]
        ):
            _fail("structured_sft_candidate_split_commitment_invalid")
    expected_bindings = {
        name: {
            "sha256": hashlib.sha256(normalized[name]).hexdigest(),
            "size_bytes": len(normalized[name]),
        }
        for name in (_CANDIDATE_TRAIN_FILE, _CANDIDATE_VALID_FILE)
    }
    if manifest_raw.get("artifacts") != expected_bindings:
        _fail("structured_sft_candidate_artifact_binding_invalid")
    expected_root = _sha256(
        {
            "domain": "AURA-SFT-CUSTODY-ROOT-v1",
            "curriculum_manifest": public_manifest,
            "candidate_artifacts": expected_bindings,
            "holdout_seed_commitment_sha256": public_manifest.get(
                "holdout_seed_commitment_sha256"
            ),
        }
    )
    if (
        manifest_raw.get("custody_root_sha256") != expected_root
        or manifest_raw.get("candidate_filenames")
        != {
            "train": _CANDIDATE_TRAIN_FILE,
            "validation": _CANDIDATE_VALID_FILE,
        }
        or manifest_raw.get("validation_scope")
        != "train_validation_replay_only"
        or manifest_raw.get("trainer_contract")
        != public_manifest.get("trainer_contract")
        or manifest_raw.get("trainer_ready") is not False
        or manifest_raw.get("training_authority")
        != public_manifest.get("training_authority")
        or manifest_raw.get("required_next_gates")
        != list(STRUCTURED_SFT_REQUIRED_NEXT_GATES)
    ):
        _fail("structured_sft_candidate_custody_binding_invalid")
    return json.loads(canonical_json_bytes(manifest_raw))


@dataclass(frozen=True, slots=True)
class StructuredSFTCustodyBundles:
    """Trainer and evaluator bundles that share commitments, not data."""

    candidate_artifacts: dict[str, bytes]
    evaluator_artifacts: dict[str, bytes]


def _build_evaluator_dataset_artifacts_from_validated(
    validated: Mapping[str, Any],
    *,
    holdout_seed: bytes,
    candidate_manifest: Mapping[str, Any],
) -> dict[str, bytes]:
    seed = _validated_holdout_seed(holdout_seed)
    holdout_body = {
        "schema": "aura.rlc.structured_sft_holdout.v1",
        "holdout_seed_hex": seed.hex(),
        "candidate_package_sha256": candidate_manifest["package_sha256"],
        "custody_root_sha256": candidate_manifest["custody_root_sha256"],
        "curriculum_manifest": candidate_manifest["curriculum_manifest"],
        "split_commitment": validated["split_commitments"][HOLDOUT_SPLIT],
        "examples": validated["splits"][HOLDOUT_SPLIT],
    }
    holdout_payload = canonical_json_bytes(holdout_body)
    holdout_binding = {
        "sha256": hashlib.sha256(holdout_payload).hexdigest(),
        "size_bytes": len(holdout_payload),
    }
    manifest_body = {
        "schema": STRUCTURED_SFT_EVALUATOR_PACKAGE_SCHEMA,
        "candidate_package_sha256": candidate_manifest["package_sha256"],
        "custody_root_sha256": candidate_manifest["custody_root_sha256"],
        "holdout_seed_commitment_sha256": validated[
            "holdout_seed_commitment_sha256"
        ],
        "artifact": {
            "filename": _EVALUATOR_HOLDOUT_FILE,
            **holdout_binding,
        },
        "trainer_access": "not_enforced_shared_process_identity",
        "custody_scope": "separate_artifact_root_shared_uid",
    }
    evaluator_manifest = {
        **manifest_body,
        "evaluator_package_sha256": _sha256(manifest_body),
    }
    return {
        _EVALUATOR_HOLDOUT_FILE: holdout_payload,
        _EVALUATOR_MANIFEST_FILE: canonical_json_bytes(evaluator_manifest),
    }


def build_structured_sft_custody_bundles(
    spec: StructuredSFTCurriculumSpec,
    *,
    holdout_seed: bytes,
) -> StructuredSFTCustodyBundles:
    """Build cryptographically linked, physically separable custody bundles."""

    seed = _validated_holdout_seed(holdout_seed)
    curriculum = build_structured_sft_curriculum(
        spec,
        holdout_seed=seed,
    )
    validated = validate_structured_sft_curriculum(
        curriculum,
        holdout_seed=seed,
    )
    candidate = _build_candidate_dataset_artifacts_from_validated(validated)
    candidate_manifest = validate_candidate_dataset_artifacts(candidate)
    evaluator = _build_evaluator_dataset_artifacts_from_validated(
        validated,
        holdout_seed=seed,
        candidate_manifest=candidate_manifest,
    )
    validate_structured_sft_custody_pair(candidate, evaluator)
    return StructuredSFTCustodyBundles(candidate, evaluator)


def validate_evaluator_dataset_artifacts(
    artifacts: Mapping[str, bytes],
    *,
    candidate_artifacts: Mapping[str, bytes],
) -> dict[str, Any]:
    """Verify evaluator bytes, secret derivation, and candidate binding."""

    pair = validate_structured_sft_custody_pair(
        candidate_artifacts,
        artifacts,
    )
    return pair["evaluator_manifest"]


def validate_structured_sft_custody_pair(
    candidate_artifacts: Mapping[str, bytes],
    evaluator_artifacts: Mapping[str, bytes],
) -> dict[str, Any]:
    """Prove custody linkage and exact split disjointness."""

    candidate = validate_candidate_dataset_artifacts(candidate_artifacts)
    if (
        not isinstance(evaluator_artifacts, Mapping)
        or set(evaluator_artifacts) != set(STRUCTURED_SFT_EVALUATOR_FILES)
    ):
        _fail("structured_sft_evaluator_file_set_invalid")
    normalized_evaluator: dict[str, bytes] = {}
    for name in STRUCTURED_SFT_EVALUATOR_FILES:
        payload = evaluator_artifacts.get(name)
        if (
            not isinstance(payload, bytes)
            or not payload
            or len(payload) > _MAX_PACKAGE_ARTIFACT_BYTES
        ):
            _fail("structured_sft_evaluator_file_invalid")
        normalized_evaluator[name] = payload
    try:
        holdout = json.loads(
            normalized_evaluator[_EVALUATOR_HOLDOUT_FILE].decode("utf-8")
        )
        evaluator_manifest = json.loads(
            normalized_evaluator[_EVALUATOR_MANIFEST_FILE].decode("utf-8")
        )
    except (RecursionError, UnicodeDecodeError, ValueError) as exc:
        raise StructuredSFTError(
            "structured_sft_evaluator_json_invalid"
        ) from exc
    _validate_bounded_json(
        holdout,
        code="structured_sft_evaluator_json_invalid",
    )
    _validate_bounded_json(
        evaluator_manifest,
        code="structured_sft_evaluator_json_invalid",
    )
    holdout_fields = {
        "schema",
        "holdout_seed_hex",
        "candidate_package_sha256",
        "custody_root_sha256",
        "curriculum_manifest",
        "split_commitment",
        "examples",
    }
    evaluator_manifest_fields = {
        "schema",
        "candidate_package_sha256",
        "custody_root_sha256",
        "holdout_seed_commitment_sha256",
        "artifact",
        "trainer_access",
        "custody_scope",
        "evaluator_package_sha256",
    }
    if (
        not isinstance(holdout, Mapping)
        or set(holdout) != holdout_fields
        or not isinstance(evaluator_manifest, Mapping)
        or set(evaluator_manifest) != evaluator_manifest_fields
    ):
        _fail("structured_sft_evaluator_schema_invalid")
    try:
        seed = bytes.fromhex(holdout["holdout_seed_hex"])
    except (KeyError, TypeError, ValueError) as exc:
        raise StructuredSFTError(
            "structured_sft_evaluator_json_invalid"
        ) from exc
    seed = _validated_holdout_seed(seed)
    spec = StructuredSFTCurriculumSpec(
        **dict(candidate["curriculum_manifest"]["spec"])
    )
    expected_curriculum = build_structured_sft_curriculum(
        spec,
        holdout_seed=seed,
    )
    expected_public = _curriculum_manifest_from_validated(
        expected_curriculum
    )
    if expected_public != candidate["curriculum_manifest"]:
        _fail("structured_sft_evaluator_curriculum_mismatch")
    expected_evaluator = _build_evaluator_dataset_artifacts_from_validated(
        expected_curriculum,
        holdout_seed=seed,
        candidate_manifest=candidate,
    )
    if any(
        normalized_evaluator.get(name) != expected_evaluator[name]
        for name in STRUCTURED_SFT_EVALUATOR_FILES
    ):
        _fail("structured_sft_evaluator_replay_mismatch")
    evaluator_body = dict(evaluator_manifest)
    evaluator_sha256 = evaluator_body.pop(
        "evaluator_package_sha256",
        None,
    )
    if (
        not _is_sha256(evaluator_sha256)
        or _sha256(evaluator_body) != evaluator_sha256
    ):
        _fail("structured_sft_evaluator_manifest_invalid")
    candidate_bytes = b"\n".join(
        candidate_artifacts[name] for name in STRUCTURED_SFT_CANDIDATE_FILES
    )
    if seed in candidate_bytes or seed.hex().encode("ascii") in candidate_bytes:
        _fail("structured_sft_holdout_seed_leaked_to_candidate")
    visible_examples = [
        *expected_curriculum["splits"][TRAIN_SPLIT],
        *expected_curriculum["splits"][VALIDATION_SPLIT],
    ]
    holdout_examples = expected_curriculum["splits"][HOLDOUT_SPLIT]
    visible_ids = {row["example_id"] for row in visible_examples}
    holdout_ids = {row["example_id"] for row in holdout_examples}
    visible_fingerprints = {
        row["case_fingerprint"] for row in visible_examples
    }
    holdout_fingerprints = {
        row["case_fingerprint"] for row in holdout_examples
    }
    if (
        visible_ids & holdout_ids
        or visible_fingerprints & holdout_fingerprints
    ):
        _fail("structured_sft_custody_split_overlap")
    body = {
        "schema": STRUCTURED_SFT_CUSTODY_REPORT_SCHEMA,
        "candidate_package_sha256": candidate["package_sha256"],
        "evaluator_package_sha256": evaluator_manifest[
            "evaluator_package_sha256"
        ],
        "custody_root_sha256": candidate["custody_root_sha256"],
        "holdout_seed_commitment_sha256": candidate[
            "curriculum_manifest"
        ]["holdout_seed_commitment_sha256"],
        "visible_example_count": len(visible_examples),
        "holdout_example_count": len(holdout_examples),
        "example_id_overlap_count": 0,
        "case_fingerprint_overlap_count": 0,
        "candidate_contains_holdout_seed": False,
        "status": "passed_artifact_noncontainment_shared_uid",
        "access_isolation_enforced": False,
        "evaluator_manifest": evaluator_manifest,
    }
    return {
        **body,
        "custody_report_sha256": _sha256(body),
    }


__all__ = [
    "CODE_TOOL",
    "CODE_TOOL_REPAIR",
    "DERIVATION_TARGET",
    "FORMAL_LOGIC",
    "HOLDOUT_SPLIT",
    "REPAIR_INTERPRETATION_TARGET",
    "REPAIR_TOOL_CALL_TARGET",
    "STRUCTURED_PROGRAM",
    "STRUCTURED_SFT_CURRICULUM_SCHEMA",
    "STRUCTURED_SFT_CUSTODY_REPORT_SCHEMA",
    "STRUCTURED_SFT_EVALUATOR_FILES",
    "STRUCTURED_SFT_EVALUATOR_PACKAGE_SCHEMA",
    "STRUCTURED_SFT_EXAMPLE_SCHEMA",
    "STRUCTURED_SFT_FAMILIES",
    "STRUCTURED_SFT_MANIFEST_SCHEMA",
    "STRUCTURED_SFT_PACKAGE_SCHEMA",
    "STRUCTURED_SFT_REQUIRED_NEXT_GATES",
    "STRUCTURED_SFT_SPLITS",
    "STRUCTURED_SFT_TOKENIZATION_SCHEMA",
    "STRUCTURED_SFT_VERSION",
    "STRUCTURED_SFT_CANDIDATE_FILES",
    "StructuredSFTCustodyBundles",
    "StructuredSFTCurriculumSpec",
    "StructuredSFTError",
    "TOOL_CALL_TARGET",
    "TOOL_INTERPRETATION_TARGET",
    "TRAIN_SPLIT",
    "VALIDATION_SPLIT",
    "build_structured_sft_curriculum",
    "build_candidate_dataset_artifacts",
    "build_structured_sft_custody_bundles",
    "canonical_json_bytes",
    "curriculum_manifest",
    "generate_structured_sft_example",
    "trainer_rows",
    "validate_candidate_dataset_artifacts",
    "validate_evaluator_dataset_artifacts",
    "validate_structured_sft_custody_pair",
    "validate_structured_sft_curriculum",
    "validate_structured_sft_example",
    "validate_trainer_tokenization",
    "verify_structured_sft_example_semantics",
]
