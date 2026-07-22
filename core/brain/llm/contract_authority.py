"""Authenticated selection of privileged MLX output contracts.

CP126 ``841bf5f7``. The worker chose its prompt builder, sampling regime,
validator and output normalizer straight from booleans on the IPC job:
``strict_answer_contract``, ``strict_value_contract``,
``proof_evaluation_contract``, ``operator_evidence_contract``,
``capability_inventory_contract``, plus schema, health and cache-bypass
behaviour. Nothing established WHO selected them or whether that component
was entitled to.

What this does and does not buy
------------------------------

The parent process spawns the worker, so anything able to forge a job on
that queue already runs inside the parent. This is therefore **not** a new
security perimeter, and it is not claimed as one. What it does provide is
real and was missing:

* **Provenance.** Every privileged contract selection names the principal
  that made it, so a receipt says which subsystem asked for proof-mode
  sampling rather than only that proof mode happened.
* **Confused-deputy resistance.** A component that copies a job dict
  forward, or sets a flag it has no business setting, no longer silently
  changes validation and normalization: the selection has to be signed by
  the lane that owns the worker.
* **Tamper evidence.** A job mutated in flight between signing and
  submission fails verification instead of quietly taking effect.

Design
------

The key is generated per worker spawn and handed to the child at fork. It
never leaves the process pair and is not persisted, so there is no key
distribution or rotation problem: a new worker means a new key, and a
worker's key is meaningless to any other worker.

Enforcement is **fail-closed but scoped**: a worker holding a key refuses
any job that asserts a privileged contract without a valid signature.
Ordinary generation is untouched — it selects no privileged contract, so it
needs no signature. A worker with no key (a bare unit-test harness) does not
enforce, which is what keeps the spawn contract backward compatible.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from typing import Any

# Job fields whose truthiness changes prompt construction, sampling,
# validation or output normalization. Selecting any of them is privileged.
PRIVILEGED_CONTRACT_FIELDS: tuple[str, ...] = (
    "strict_answer_contract",
    "strict_value_contract",
    "proof_evaluation_contract",
    "operator_evidence_contract",
    "capability_inventory_contract",
    "health_probe",
    "disable_prompt_cache",
    "schema",
)

# Contracts that each select a DIFFERENT prompt builder, validator and
# normalizer. Asserting more than one is a contradiction the if/elif ladder
# would silently resolve by source order.
EXCLUSIVE_CONTRACT_FIELDS: tuple[str, ...] = (
    "strict_answer_contract",
    "strict_value_contract",
    "proof_evaluation_contract",
    "operator_evidence_contract",
)

AUTH_FIELD = "contract_auth"


def new_contract_key() -> bytes:
    """A fresh per-spawn signing key."""
    return secrets.token_bytes(32)


def selected_privileged_fields(job: dict[str, Any]) -> list[str]:
    """Privileged fields this job actually asserts, in a stable order."""
    selected: list[str] = []
    for field in PRIVILEGED_CONTRACT_FIELDS:
        value = job.get(field)
        # `schema` is privileged when present and non-empty, not when False.
        if field == "schema":
            if value:
                selected.append(field)
            continue
        if bool(value):
            selected.append(field)
    return selected


def exclusive_conflict(job: dict[str, Any]) -> list[str]:
    """Mutually exclusive output contracts asserted together, if any."""
    active = [f for f in EXCLUSIVE_CONTRACT_FIELDS if bool(job.get(f))]
    return active if len(active) > 1 else []


def _canonical_payload(job: dict[str, Any], principal: str) -> bytes:
    """Exactly what is signed: this job's id, principal, and selections.

    The job id binds the signature to one request so a valid signature
    cannot be lifted onto a different job. Selections are sorted so the
    same choice always produces the same bytes.
    """
    return json.dumps(
        {
            "id": str(job.get("id") or ""),
            "action": str(job.get("action") or ""),
            "principal": principal,
            "fields": sorted(selected_privileged_fields(job)),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sign_job(job: dict[str, Any], key: bytes | None, *, principal: str) -> dict[str, Any]:
    """Stamp a job with the authority for the contracts it selects.

    A job selecting nothing privileged is returned unchanged — ordinary
    generation carries no authority because it needs none. Mutates and
    returns the job so it can wrap a submission in one line.
    """
    if not key:
        return job
    if not selected_privileged_fields(job):
        return job
    job[AUTH_FIELD] = {
        "principal": str(principal or "unknown"),
        "signature": hmac.new(
            key, _canonical_payload(job, str(principal or "unknown")), hashlib.sha256
        ).hexdigest(),
    }
    return job


def verify_job(job: dict[str, Any], key: bytes | None) -> str:
    """Why this job's contract selection must be REFUSED, or "" if it is sound.

    Order matters: a contradictory selection is refused whether or not it is
    signed, because a signature over a contradiction still leaves the worker
    with no single contract to honour.
    """
    conflict = exclusive_conflict(job)
    if conflict:
        return "ambiguous_output_contract:" + ",".join(conflict)

    selected = selected_privileged_fields(job)
    if not selected:
        return ""
    if not key:
        # No key means this worker was never given an authority to check
        # against (bare test harness). Enforcing here would refuse every
        # privileged job in a process that has no way to sign one.
        return ""

    auth = job.get(AUTH_FIELD)
    if not isinstance(auth, dict):
        return "unauthenticated_contract_selection:" + ",".join(selected)
    principal = str(auth.get("principal") or "")
    signature = str(auth.get("signature") or "")
    if not principal or not signature:
        return "incomplete_contract_authority:" + ",".join(selected)
    expected = hmac.new(
        key, _canonical_payload(job, principal), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected, signature):
        # Covers a forged signature, a job mutated after signing, and a
        # signature lifted from a different job id or field selection.
        return "invalid_contract_authority:" + ",".join(selected)
    return ""


def authority_receipt(job: dict[str, Any]) -> dict[str, Any]:
    """What to record about how this job's contracts were authorized."""
    selected = selected_privileged_fields(job)
    auth = job.get(AUTH_FIELD)
    return {
        "privileged_fields": selected,
        "principal": (
            str(auth.get("principal")) if isinstance(auth, dict) else ""
        ),
        "authenticated": bool(isinstance(auth, dict) and auth.get("signature")),
    }


__all__ = [
    "AUTH_FIELD",
    "EXCLUSIVE_CONTRACT_FIELDS",
    "PRIVILEGED_CONTRACT_FIELDS",
    "authority_receipt",
    "exclusive_conflict",
    "new_contract_key",
    "selected_privileged_fields",
    "sign_job",
    "verify_job",
]
