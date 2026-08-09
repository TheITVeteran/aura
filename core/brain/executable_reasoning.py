"""Model-authored, sandbox-grounded computation for difficult reasoning turns.

The symbolic sandbox used to be downstream of answer generation: Aura could
repair code that she happened to write, but she did not deliberately write a
program to solve a structured problem.  This module supplies that missing
operation.  It converts a public objective into a pure-compute program, runs it
inside the existing kernel sandbox, and returns the program's stdout as a
candidate answer.  The caller remains responsible for normal answer
verification and promotion.

No task answers or benchmark-family solvers live here.  The resident model
authors each program from the same public information available to ordinary
inference.  Generated source and raw diagnostics remain ephemeral; receipts
carry hashes, sizes, containment evidence, and bounded status only.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from core.brain.generation_provenance import attributed_text, generation_metadata_of

GenerateFn = Callable[[str, float], Awaitable[Any]]

EXECUTABLE_REASONING_SCHEMA = "aura.executable_reasoning.v1"
_MAX_OBJECTIVE_CHARS = 12_000
_MAX_STDOUT_CHARS = 32_000
_MIN_GENERATION_WINDOW_S = 2.0

_NO_EXECUTION = re.compile(
    r"\b(?:without\s+(?:executing|running)|do\s+not\s+(?:execute|run)|"
    r"must\s+not\s+(?:execute|run))\b",
    re.IGNORECASE,
)
_STRUCTURED_COMPUTE = re.compile(
    r"\b(?:calculate|compute|count|evaluate|trace|simulate|predict|checksum|"
    r"maximize|minimize|optimal|schedule|sequence|posterior|probability|"
    r"combinator|intervention|constraint|score|winner|median|algorithm)\w*\b",
    re.IGNORECASE,
)
_STRUCTURED_INPUT = re.compile(r"(?:\[[^\]]+\]|\{[^}]+\}|\b\d+(?:\.\d+)?\b)")


@dataclass(frozen=True, slots=True)
class ExecutableReasoningResult:
    candidate: str
    succeeded: bool
    receipt: dict[str, Any]


def should_use_executable_reasoning(
    objective: str,
    *,
    task_type: str,
    explicitly_enabled: bool = False,
) -> bool:
    """Return whether pure computation is a plausible reasoning operation.

    Explicit user constraints against execution always win.  The default path
    is intentionally semantic rather than tied to benchmark domain names:
    math is computational by definition; other domains need both a structured
    operation and structured input.  A caller may explicitly enable the organ
    for an already-classified hard task, but cannot override a no-execution
    instruction.
    """

    text = str(objective or "").strip()
    if not text or _NO_EXECUTION.search(text):
        return False
    if explicitly_enabled:
        return True
    normalized_type = str(task_type or "").strip().lower()
    if normalized_type == "math":
        return True
    return bool(
        normalized_type in {"code", "logic", "planning", "factual"}
        and _STRUCTURED_COMPUTE.search(text)
        and _STRUCTURED_INPUT.search(text)
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _program_prompt(objective: str, response_contract: str) -> str:
    contract = str(response_contract or "").strip()
    output_rule = (
        "Print exactly one line beginning FINAL_ANSWER: followed by one JSON "
        f"object satisfying this public response contract: {contract}"
        if contract
        else "Print only the final answer that should be returned to the user."
    )
    return (
        "Solve the task by authoring a self-contained pure-Python scratch program.\n"
        "The program will run in an isolated, network-denied sandbox with no files, "
        "input, shell, subprocesses, reflection, or external packages. Standard pure "
        "computation with builtins and safe modules such as math, itertools, fractions, "
        "collections, statistics, decimal, and json is allowed.\n"
        "Derive the answer from the task; do not guess it or embed an unexplained final "
        "constant. Prefer exhaustive search or an independent invariant when practical. "
        "Never assert a literal expected final answer, expected full sequence, or checksum. "
        "Assertions may check only generic invariants computed from the candidate itself, "
        "such as length, uniqueness, constraints, or an independently recomputed score.\n"
        f"{output_rule}\n"
        "Return exactly one fenced Python code block and no prose.\n\n"
        "TASK:\n"
        f"{str(objective or '').strip()[:_MAX_OBJECTIVE_CHARS]}"
    )


def _restart_prompt(
    objective: str,
    diagnostic: str,
    response_contract: str,
    prior_program_sha256: str,
) -> str:
    """Request a disjoint retry without exposing the failed source or answer."""

    contract = str(response_contract or "").strip()
    return (
        "A previous pure-Python reasoning attempt failed its sandbox check. Start over "
        "from the original task using a different derivation or algorithm. The failed "
        "source is deliberately withheld so it cannot anchor this attempt. Return exactly "
        "one fenced Python code block. Do not use files, network, shell, subprocesses, "
        "input, reflection, or external packages. Never assert a literal expected final "
        "answer, expected full sequence, or checksum; assertions may check only generic "
        "invariants computed from the candidate itself. "
        + (
            "It must print exactly one terminal FINAL_ANSWER JSON object satisfying "
            f"{contract}. "
            if contract
            else "It must print only the final user-facing answer. "
        )
        + f"\n\nPRIOR_PROGRAM_SHA256: {prior_program_sha256}"
        + f"\nFAILURE_CLASS: {diagnostic[:512]}"
        + f"\n\nORIGINAL_TASK:\n{str(objective or '').strip()[:_MAX_OBJECTIVE_CHARS]}"
    )


def _contract_valid(candidate: str, response_contract: str) -> bool:
    if not response_contract:
        return bool(candidate.strip())
    try:
        from core.brain.llm.latent_cortex.frontier_tasks import parse_final_answer
        from core.brain.llm.latent_cortex.response_contracts import (
            parse_response_contract,
            validate_response_payload,
        )

        contract = parse_response_contract(response_contract)
        payload = parse_final_answer(candidate)
        return bool(validate_response_payload(payload, contract)["valid"])
    except (KeyError, TypeError, ValueError):
        return False


def _execution_failure_class(execution: Any) -> str:
    """Return bounded failure data without feeding runtime text back as instructions."""

    if getattr(execution, "refused", False):
        warnings = getattr(execution, "warnings", []) or []
        labels = [re.sub(r"[^a-zA-Z0-9_.:-]+", "_", str(item))[:80] for item in warnings]
        return "sandbox_refused:" + ",".join(labels[:4])
    if getattr(execution, "timed_out", False):
        return "sandbox_timed_out"
    raw = (
        str(getattr(execution, "traceback", "") or "")
        or str(getattr(execution, "stderr", "") or "")
    )
    if "AssertionError" in raw:
        return "sandbox_execution_failed:AssertionError"
    if "SyntaxError" in raw:
        return "sandbox_execution_failed:SyntaxError"
    if "TypeError" in raw:
        return "sandbox_execution_failed:TypeError"
    if "ValueError" in raw:
        return "sandbox_execution_failed:ValueError"
    if "IndexError" in raw:
        return "sandbox_execution_failed:IndexError"
    if "KeyError" in raw:
        return "sandbox_execution_failed:KeyError"
    return "sandbox_execution_failed:RuntimeError"


async def derive_executable_candidate(
    *,
    objective: str,
    task_type: str,
    generate: GenerateFn,
    sandbox: Any,
    deadline: float,
    response_contract: str = "",
    explicitly_enabled: bool = False,
) -> ExecutableReasoningResult:
    """Generate, execute, and receipt one bounded program-of-thought attempt."""

    started = time.monotonic()
    base_receipt: dict[str, Any] = {
        "schema": EXECUTABLE_REASONING_SCHEMA,
        "status": "not_applicable",
        "task_type": str(task_type or ""),
        "objective_sha256": _sha256(str(objective or "")),
        "response_contract_sha256": _sha256(str(response_contract or "")),
        "generation_calls": 0,
        "program_chars": 0,
        "program_sha256": "",
        "candidate_chars": 0,
        "candidate_sha256": "",
        "contract_valid": False,
    }
    if not should_use_executable_reasoning(
        objective,
        task_type=task_type,
        explicitly_enabled=explicitly_enabled,
    ):
        return ExecutableReasoningResult("", False, base_receipt)

    remaining = deadline - time.monotonic()
    if remaining < _MIN_GENERATION_WINDOW_S:
        return ExecutableReasoningResult(
            "", False, {**base_receipt, "status": "deadline_exhausted"}
        )

    from core.brain.verifiers.code_engine import extract_code_blocks
    generation_calls = 0
    attempts: list[dict[str, Any]] = []
    generated_metadata: dict[str, Any] = {}
    execution: Any = None
    program = ""
    diagnostic = ""
    for attempt_index in range(2):
        remaining = deadline - time.monotonic()
        if remaining < _MIN_GENERATION_WINDOW_S:
            break
        prompt = (
            _program_prompt(objective, response_contract)
            if attempt_index == 0
            else _restart_prompt(
                objective,
                diagnostic or "prior_attempt_failed",
                response_contract,
                _sha256(program),
            )
        )
        try:
            generated = await asyncio.wait_for(
                generate(prompt, 0.2 if attempt_index == 0 else 0.6),
                timeout=max(_MIN_GENERATION_WINDOW_S, remaining),
            )
        except (TimeoutError, RuntimeError, AttributeError, TypeError, ValueError):
            generation_calls += 1
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "status": "program_generation_failed",
                    "program_sha256": "",
                }
            )
            break
        generation_calls += 1
        generated_metadata = generation_metadata_of(generated)
        blocks = extract_code_blocks(str(generated or "").strip())
        if len(blocks) != 1:
            diagnostic = f"program_shape_invalid:block_count={len(blocks)}"
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "status": "program_shape_invalid",
                    "program_sha256": "",
                    "program_block_count": len(blocks),
                }
            )
            program = ""
            continue
        program = blocks[0]
        try:
            execution = await sandbox.run(program)
        except (TimeoutError, OSError, RuntimeError, AttributeError, TypeError, ValueError):
            diagnostic = "sandbox_infrastructure_failure"
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "status": "sandbox_failed",
                    "program_sha256": _sha256(program),
                    "program_chars": len(program),
                }
            )
            break
        execution_receipt = execution.to_dict()
        attempt_status = "executed" if getattr(execution, "ok", False) else (
            "refused" if getattr(execution, "refused", False)
            else "timed_out" if getattr(execution, "timed_out", False)
            else "execution_failed"
        )
        attempts.append(
            {
                "attempt": attempt_index + 1,
                "status": attempt_status,
                "program_sha256": _sha256(program),
                "program_chars": len(program),
                "sandbox": execution_receipt,
            }
        )
        if getattr(execution, "ok", False):
            break
        diagnostic = _execution_failure_class(execution)

    if execution is None:
        return ExecutableReasoningResult(
            "",
            False,
            {
                **base_receipt,
                "status": attempts[-1]["status"] if attempts else "deadline_exhausted",
                "generation_calls": generation_calls,
                "attempts": attempts,
                "elapsed_s": round(time.monotonic() - started, 6),
            },
        )

    sandbox_receipt = execution.to_dict()
    candidate = str(getattr(execution, "stdout", "") or "").strip()
    if len(candidate) > _MAX_STDOUT_CHARS:
        candidate = ""
    contract_valid = bool(getattr(execution, "ok", False)) and _contract_valid(
        candidate, response_contract
    )
    status = "candidate_ready" if contract_valid else (
        "sandbox_execution_failed" if not getattr(execution, "ok", False)
        else "candidate_contract_invalid"
    )
    receipt = {
        **base_receipt,
        "status": status,
        "generation_calls": generation_calls,
        "program_chars": len(program),
        "program_sha256": _sha256(program),
        "candidate_chars": len(candidate),
        "candidate_sha256": _sha256(candidate) if candidate else "",
        "contract_valid": contract_valid,
        "sandbox": sandbox_receipt,
        "attempts": attempts,
        "elapsed_s": round(time.monotonic() - started, 6),
    }
    if not contract_valid:
        return ExecutableReasoningResult("", False, receipt)

    candidate = attributed_text(
        candidate,
        {
            **generated_metadata,
            "response_path": "executable_reasoning",
            "model_native_output": False,
            "sandbox_grounded": True,
            "executable_reasoning_receipt_sha256": _sha256(str(sorted(receipt.items()))),
        },
    )
    return ExecutableReasoningResult(candidate, True, receipt)


__all__ = [
    "EXECUTABLE_REASONING_SCHEMA",
    "ExecutableReasoningResult",
    "derive_executable_candidate",
    "should_use_executable_reasoning",
]
