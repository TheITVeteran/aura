"""Deterministic compute feedback for recurrent cognitive actions.

The latent worker can choose to formalize or simulate, but it cannot execute
host code. This broker performs one bounded service-side check and returns its
result as typed, non-authoritative evidence for one continuation episode.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from core.brain.llm.latent_cortex.cognitive_context import (
    MAX_COGNITIVE_CONTEXT_CHARS,
    normalize_cognitive_context,
)
from core.brain.llm.latent_cortex.epistemic_state import OperationKind
from core.brain.prompts.sanitizer import ContextGuard

COMPUTE_ACQUISITION_SCHEMA = "aura.rlc.compute_acquisition.v1"
_SUPPORTED = {OperationKind.FORMALIZE, OperationKind.SIMULATE}


@dataclass(frozen=True, slots=True)
class ComputeAcquisition:
    context: tuple[dict[str, Any], ...]
    receipt: dict[str, Any]


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=lambda item: f"<{type(item).__module__}.{type(item).__qualname__}>",
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bounded(value: Any, limit: int = MAX_COGNITIVE_CONTEXT_CHARS) -> str:
    return " ".join(str(value or "").split()).strip()[:limit].rstrip()


def _timeout(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("cognitive compute timeout is invalid")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("cognitive compute timeout is invalid")
    return max(0.1, min(12.0, number))


def _public_guard(receipt: Any) -> dict[str, Any]:
    return {
        "policy_version": str(getattr(receipt, "policy_version", "")),
        "quarantined": bool(getattr(receipt, "quarantined", False)),
        "fail_closed": bool(getattr(receipt, "fail_closed", False)),
        "categories": sorted(
            {
                str(getattr(item, "category", ""))
                for item in list(getattr(receipt, "detections", []) or [])
                if str(getattr(item, "category", ""))
            }
        ),
    }


async def acquire_cognitive_compute(
    *,
    objective: str,
    first_text: str,
    action: OperationKind | str,
    timeout_s: float,
) -> ComputeAcquisition:
    """Run one verifier/sandbox pass and emit a content-addressed observation."""

    operation = action if isinstance(action, OperationKind) else OperationKind(action)
    if operation not in _SUPPORTED:
        raise ValueError("cognitive compute action is unsupported")
    objective = str(objective or "").strip()
    candidate = str(first_text or "").strip()
    if not objective or not candidate:
        raise ValueError("cognitive compute objective and candidate are required")
    bounded_timeout = _timeout(timeout_s)

    from core.brain.reasoning_amplifier_v2 import classify_task_type
    from core.brain.verifiers import VerificationResult, verify_candidate

    task_type = classify_task_type(objective)
    if operation is OperationKind.FORMALIZE:
        verification = await verify_candidate(
            candidate,
            task_type=task_type,
            context={
                "objective": objective,
                "read_only_evaluation": True,
                "task_key": _canonical_sha256(
                    {"objective": objective, "candidate": candidate}
                ),
            },
        )
    else:
        # The sandbox vets syntax/safety and executes once below. Calling the
        # code verifier here could execute module-level assertions a second time.
        verification = VerificationResult(
            domain=task_type,
            ok=True,
            checked=False,
            engine="symbolic_sandbox",
        )
    verifier = verification.to_dict()
    sandbox: dict[str, Any] | None = None
    if operation is OperationKind.SIMULATE:
        from core.brain.symbolic_sandbox import get_symbolic_sandbox
        from core.brain.verifiers.code_engine import extract_code_blocks

        blocks = extract_code_blocks(candidate)
        if blocks:
            sandbox = (
                await get_symbolic_sandbox().run(
                    blocks[0], timeout_override=bounded_timeout
                )
            ).to_dict()

    measured = bool(verification.checked or sandbox is not None)
    status = "measured" if measured else "unmeasured"
    observations: list[str] = []
    if verification.checked:
        observations.append(
            "Verifier result: "
            f"ok={verification.ok}, score={verification.score:.3f}, "
            f"engine={verification.engine or 'unknown'}."
        )
        observations.extend(str(item) for item in verification.evidence[:3])
        observations.extend(str(item) for item in verification.issues[:3])
    if sandbox is not None:
        observations.append(
            "Sandbox result: "
            f"ok={bool(sandbox.get('ok'))}, refused={bool(sandbox.get('refused'))}, "
            f"timed_out={bool(sandbox.get('timed_out'))}, "
            f"isolation={sandbox.get('isolation', {}).get('isolation_level', 'unavailable')}."
        )
        stdout = _bounded(sandbox.get("stdout"), 160)
        stderr = _bounded(sandbox.get("stderr"), 160)
        if stdout:
            observations.append(f"stdout: {stdout}")
        if stderr:
            observations.append(f"stderr: {stderr}")

    raw_text = _bounded(" ".join(observations))
    guarded = ContextGuard.guard(
        raw_text,
        role="retrieved",
        request_id=f"rlc-compute:{operation.value}",
    ) if raw_text else None
    guard = _public_guard(guarded.receipt) if guarded is not None else {}
    clean = _bounded(guarded.text) if guarded is not None else ""
    admitted = bool(
        measured
        and clean
        and not guard.get("quarantined")
        and not guard.get("fail_closed")
    )
    private = {
        "schema": COMPUTE_ACQUISITION_SCHEMA,
        "action": operation.value,
        "objective_sha256": hashlib.sha256(objective.encode()).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
        "task_type": task_type,
        "status": status,
        "verifier": verifier,
        "sandbox": sandbox,
        "guard": guard,
        "admitted": admitted,
    }
    receipt_sha256 = _canonical_sha256(private)
    receipt = {**private, "receipt_sha256": receipt_sha256}
    if not admitted:
        return ComputeAcquisition((), receipt)

    source = f"capability.symbolic_{operation.value}"
    content_sha256 = hashlib.sha256(clean.encode()).hexdigest()
    identity = hashlib.sha256(
        f"{source}:{content_sha256}:{receipt_sha256}".encode()
    ).hexdigest()
    item = {
        "source": source,
        "text": clean,
        "context_role": "evidence_observation",
        "instruction_authority": False,
        "evidence_id": f"evidence-{identity[:24]}",
        "content_sha256": content_sha256,
        "retrieval_receipt_sha256": receipt_sha256,
        "evidence_kind": "governed_tool_observation",
        "evidence_origin": "core.brain.cortex_compute_acquisition",
        "source_version": COMPUTE_ACQUISITION_SCHEMA,
    }
    return ComputeAcquisition(tuple(normalize_cognitive_context([item])), receipt)


__all__ = [
    "COMPUTE_ACQUISITION_SCHEMA",
    "ComputeAcquisition",
    "acquire_cognitive_compute",
]
