"""Aura may not use her own testimony as proof that she is working.

The live Latent Cortex facade can check that a worker receipt is
well-formed and internally consistent. That is a real check and it is not
independence: the worker and the facade are not separate trust domains, so
a receipt that says "I did well" is graded by something that shares the
worker's state and its incentives.

The repo already says this plainly — a successful episode does not
independently establish which effective model and adapter answered, that a
separate verifier inspected the candidate, that the answer beat vanilla,
that held-out correctness improved, or that a frontier comparison
happened. This module is the thing that stops those from being ASSUMED.

What it is not: it does not invent signing. Ed25519 worker-capture
identity, adapter identity hashes and signed campaign payloads already
exist. What was missing is the judgement — a single place that decides
whether the evidence in hand is sufficient for the claim being made, and
**refuses when it is not**.

The refusal is the feature. Every failure this codebase keeps finding has
the same shape: the absence of a check reported as a passed check. So the
default verdict is INSUFFICIENT, upgraded only by evidence that is
present, attributed, and independent. There is no path through this module
that returns "certified" because nothing objected.

Three independence properties are checked, not assumed:

1. **The subject is not the verifier.** If the identity that produced the
   candidate is the identity that graded it, the grade is testimony.
2. **The verifier is pinned.** A verifier whose code hash is unknown could
   have been the thing that changed between a failing run and a passing
   one.
3. **The raw candidate was regraded.** A supplied SCORE is the worker's
   claim about its output. A supplied OUTPUT can be graded again by
   someone else, which is the only version that survives disagreement.
"""
from __future__ import annotations

import enum
import hashlib
import hmac
import json
import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from core.governance_context import local_internal_governed_scope
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.lockdep import checked_lock
from core.runtime.state_ownership import state_root
from core.runtime.turn_outcome import VerificationGrade

__all__ = [
    "ClaimClass",
    "VerdictStatus",
    "VerifierExecution",
    "EvidenceBundle",
    "Verdict",
    "adjudicate",
    "required_evidence",
    "EvidenceAdjudicationError",
]


class EvidenceAdjudicationError(ValueError):
    """The bundle itself is malformed — not a verdict, a rejected submission."""


class ClaimClass(str, enum.Enum):
    """What is being claimed. The bar scales with the consequence.

    Ordinary conversation is deliberately absent from the strict tiers:
    demanding a counterfactual control for every reply would make the
    check something operators route around, and a check people route
    around protects nothing.
    """

    #: A single conversational turn worked. Cheapest tier.
    TURN_QUALITY = "turn_quality"
    #: A durable learning update — changes behaviour beyond this session.
    DURABLE_LEARNING = "durable_learning"
    #: A high-stakes action with real-world effect.
    HIGH_STAKES_ACTION = "high_stakes_action"
    #: An adapter is promoted into the resident model's lane.
    ADAPTER_PROMOTION = "adapter_promotion"
    #: Aura modified her own code and wants it kept.
    SELF_IMPROVEMENT = "self_improvement"
    #: A published number about how good Aura is.
    BENCHMARK_CLAIM = "benchmark_claim"


class VerdictStatus(str, enum.Enum):
    CERTIFIED = "certified"
    #: The evidence does not establish the claim. NOT the same as refuted:
    #: the claim may well be true and nobody has shown it.
    INSUFFICIENT = "insufficient"
    #: The evidence actively contradicts the claim.
    REFUTED = "refuted"


class VerifierExecution(str, enum.Enum):
    """Where the verifier ran. A deployment fact, recorded not assumed."""

    #: Same process as the subject. Shares memory, imports, monkeypatches
    #: and any corruption the subject has. Never promotion-grade.
    IN_PROCESS = "in_process"
    #: Separate process, same host.
    SEPARATE_PROCESS = "separate_process"
    #: Separate trust domain entirely.
    SEPARATE_TRUST_DOMAIN = "separate_trust_domain"


#: Evidence item names. Strings would drift; these are the contract.
NEEDS_SUBJECT_IDENTITY = "subject_identity"
NEEDS_MODEL_HASHES = "model_hashes"
NEEDS_SIGNED_ATTESTATION = "signed_attestation"
NEEDS_RAW_CANDIDATE = "raw_candidate"
NEEDS_INDEPENDENT_VERIFIER = "independent_verifier"
NEEDS_PINNED_VERIFIER = "pinned_verifier"
NEEDS_OUT_OF_PROCESS_VERIFIER = "out_of_process_verifier"
NEEDS_VANILLA_CONTROL = "matched_vanilla_control"
NEEDS_EQUAL_COMPUTE_CONTROL = "equal_compute_control"
NEEDS_HELD_OUT = "held_out_correctness"

_REQUIREMENTS: dict[ClaimClass, tuple[str, ...]] = {
    ClaimClass.TURN_QUALITY: (
        NEEDS_SUBJECT_IDENTITY,
        NEEDS_RAW_CANDIDATE,
        NEEDS_INDEPENDENT_VERIFIER,
    ),
    ClaimClass.DURABLE_LEARNING: (
        NEEDS_SUBJECT_IDENTITY,
        NEEDS_RAW_CANDIDATE,
        NEEDS_INDEPENDENT_VERIFIER,
        NEEDS_PINNED_VERIFIER,
    ),
    ClaimClass.HIGH_STAKES_ACTION: (
        NEEDS_SUBJECT_IDENTITY,
        NEEDS_RAW_CANDIDATE,
        NEEDS_INDEPENDENT_VERIFIER,
        NEEDS_PINNED_VERIFIER,
        NEEDS_OUT_OF_PROCESS_VERIFIER,
    ),
    # From here the claim outlives the episode, so a control is required:
    # "it worked" without "better than not doing it" is not a finding.
    ClaimClass.ADAPTER_PROMOTION: (
        NEEDS_SUBJECT_IDENTITY,
        NEEDS_MODEL_HASHES,
        NEEDS_SIGNED_ATTESTATION,
        NEEDS_RAW_CANDIDATE,
        NEEDS_INDEPENDENT_VERIFIER,
        NEEDS_PINNED_VERIFIER,
        NEEDS_OUT_OF_PROCESS_VERIFIER,
        NEEDS_VANILLA_CONTROL,
        NEEDS_EQUAL_COMPUTE_CONTROL,
        NEEDS_HELD_OUT,
    ),
    ClaimClass.SELF_IMPROVEMENT: (
        NEEDS_SUBJECT_IDENTITY,
        NEEDS_SIGNED_ATTESTATION,
        NEEDS_RAW_CANDIDATE,
        NEEDS_INDEPENDENT_VERIFIER,
        NEEDS_PINNED_VERIFIER,
        NEEDS_OUT_OF_PROCESS_VERIFIER,
        NEEDS_VANILLA_CONTROL,
        NEEDS_HELD_OUT,
    ),
    ClaimClass.BENCHMARK_CLAIM: (
        NEEDS_SUBJECT_IDENTITY,
        NEEDS_MODEL_HASHES,
        NEEDS_SIGNED_ATTESTATION,
        NEEDS_RAW_CANDIDATE,
        NEEDS_INDEPENDENT_VERIFIER,
        NEEDS_PINNED_VERIFIER,
        NEEDS_OUT_OF_PROCESS_VERIFIER,
        NEEDS_VANILLA_CONTROL,
        NEEDS_EQUAL_COMPUTE_CONTROL,
        NEEDS_HELD_OUT,
    ),
}

#: Grade awarded when a claim class certifies. A control-backed result is
#: counterfactual evidence; one without a control is a postcondition check.
_CERTIFIED_GRADE: dict[ClaimClass, VerificationGrade] = {
    ClaimClass.TURN_QUALITY: VerificationGrade.POSTCONDITION_VERIFIED,
    ClaimClass.DURABLE_LEARNING: VerificationGrade.POSTCONDITION_VERIFIED,
    ClaimClass.HIGH_STAKES_ACTION: VerificationGrade.POSTCONDITION_VERIFIED,
    ClaimClass.ADAPTER_PROMOTION: VerificationGrade.COUNTERFACTUALLY_VERIFIED,
    ClaimClass.SELF_IMPROVEMENT: VerificationGrade.COUNTERFACTUALLY_VERIFIED,
    ClaimClass.BENCHMARK_CLAIM: VerificationGrade.EXTERNALLY_VERIFIED,
}


def required_evidence(claim: ClaimClass) -> tuple[str, ...]:
    """What this claim class needs. Public so callers can prepare it."""
    return _REQUIREMENTS[claim]


@dataclass(frozen=True)
class ControlResult:
    """A comparison arm. Without one, "it worked" has nothing to beat."""

    name: str
    score: float
    #: Compute spent on the control. An arm given less compute than the
    #: treatment is not a control for the treatment — it is a control for
    #: having less compute, which is a different and much easier finding.
    compute_tokens: int = 0


@dataclass(frozen=True)
class EvidenceBundle:
    """Everything offered in support of one claim.

    Fields default to absent. Absent is the normal state and produces
    INSUFFICIENT — never a pass.
    """

    claim: ClaimClass
    #: Who produced the candidate.
    subject_identity: str | None = None
    #: Exact base model / adapter / tokenizer / runtime hashes.
    model_hashes: Mapping[str, str] | None = None
    #: A signature over the attestation, verified by the caller's identity
    #: layer before it gets here.
    signed_attestation: str | None = None
    attestation_verified: bool = False
    #: The candidate OUTPUT, not a score for it.
    raw_candidate: str | None = None
    #: Who graded it, and how.
    verifier_identity: str | None = None
    verifier_code_hash: str | None = None
    verifier_execution: VerifierExecution = VerifierExecution.IN_PROCESS
    #: The grade the independent verifier reached on the raw candidate.
    verifier_score: float | None = None
    treatment_compute_tokens: int = 0
    controls: tuple[ControlResult, ...] = ()
    held_out_score: float | None = None
    held_out_baseline: float | None = None
    notes: Mapping[str, Any] = field(default_factory=dict)

    def control(self, name: str) -> ControlResult | None:
        for entry in self.controls:
            if entry.name == name:
                return entry
        return None


@dataclass(frozen=True)
class Verdict:
    """The adjudication. Signed, so it cannot be edited into a pass."""

    claim: ClaimClass
    status: VerdictStatus
    grade: VerificationGrade
    satisfied: tuple[str, ...]
    missing: tuple[str, ...]
    contradictions: tuple[str, ...]
    subject_identity: str | None
    verifier_identity: str | None
    at: float
    signature: str

    @property
    def is_certified(self) -> bool:
        return self.status is VerdictStatus.CERTIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim.value,
            "status": self.status.value,
            "grade": self.grade.value,
            "satisfied": list(self.satisfied),
            "missing": list(self.missing),
            "contradictions": list(self.contradictions),
            "subject_identity": self.subject_identity,
            "verifier_identity": self.verifier_identity,
            "at": self.at,
            "signature": self.signature,
        }


# --------------------------------------------------------------------------
# The checks. Each returns True when the named evidence is genuinely present.
# --------------------------------------------------------------------------


def _has_subject_identity(bundle: EvidenceBundle) -> bool:
    return bool(str(bundle.subject_identity or "").strip())


def _has_model_hashes(bundle: EvidenceBundle) -> bool:
    """Every component that could change the answer must be pinned.

    A partial set is worse than none: it reads as pinned while leaving the
    thing that actually changed unrecorded.
    """
    hashes = bundle.model_hashes or {}
    required = {"base_model", "adapter", "tokenizer", "runtime"}
    return required.issubset(hashes) and all(
        str(hashes.get(key) or "").strip() for key in required
    )


def _has_signed_attestation(bundle: EvidenceBundle) -> bool:
    """A signature nobody checked is a string.

    ``attestation_verified`` must be True: this module does not verify the
    signature itself (the identity layer owns that key material), and
    accepting the mere presence of a signature would be trusting the
    claimant to have checked their own credentials.
    """
    return bool(str(bundle.signed_attestation or "").strip()) and bool(
        bundle.attestation_verified
    )


def _has_raw_candidate(bundle: EvidenceBundle) -> bool:
    return bool(str(bundle.raw_candidate or "").strip())


def _has_independent_verifier(bundle: EvidenceBundle) -> bool:
    """The subject may not grade itself.

    The single most important line in this file. When these identities
    match, the "verification" is the worker's opinion of the worker.
    """
    verifier = str(bundle.verifier_identity or "").strip()
    subject = str(bundle.subject_identity or "").strip()
    if not verifier or bundle.verifier_score is None:
        return False
    return verifier != subject


def _has_pinned_verifier(bundle: EvidenceBundle) -> bool:
    return bool(str(bundle.verifier_code_hash or "").strip())


def _has_out_of_process_verifier(bundle: EvidenceBundle) -> bool:
    return bundle.verifier_execution in (
        VerifierExecution.SEPARATE_PROCESS,
        VerifierExecution.SEPARATE_TRUST_DOMAIN,
    )


def _has_vanilla_control(bundle: EvidenceBundle) -> bool:
    return bundle.control("vanilla") is not None


def _has_equal_compute_control(bundle: EvidenceBundle) -> bool:
    """A control given less compute is a control for compute, not for the treatment.

    Ten percent tolerance: exact equality is not achievable across two real
    generations, and demanding it would make the requirement unsatisfiable
    rather than strict.
    """
    control = bundle.control("equal_compute") or bundle.control("vanilla")
    if control is None or bundle.treatment_compute_tokens <= 0:
        return False
    if control.compute_tokens <= 0:
        return False
    ratio = control.compute_tokens / float(bundle.treatment_compute_tokens)
    return 0.9 <= ratio <= 1.1


def _has_held_out(bundle: EvidenceBundle) -> bool:
    return bundle.held_out_score is not None and bundle.held_out_baseline is not None


_CHECKS = {
    NEEDS_SUBJECT_IDENTITY: _has_subject_identity,
    NEEDS_MODEL_HASHES: _has_model_hashes,
    NEEDS_SIGNED_ATTESTATION: _has_signed_attestation,
    NEEDS_RAW_CANDIDATE: _has_raw_candidate,
    NEEDS_INDEPENDENT_VERIFIER: _has_independent_verifier,
    NEEDS_PINNED_VERIFIER: _has_pinned_verifier,
    NEEDS_OUT_OF_PROCESS_VERIFIER: _has_out_of_process_verifier,
    NEEDS_VANILLA_CONTROL: _has_vanilla_control,
    NEEDS_EQUAL_COMPUTE_CONTROL: _has_equal_compute_control,
    NEEDS_HELD_OUT: _has_held_out,
}


def _contradictions(bundle: EvidenceBundle) -> tuple[str, ...]:
    """Evidence that ARGUES AGAINST the claim.

    Separate from "missing" on purpose. Missing evidence means nobody
    showed it; a contradiction means somebody showed the opposite, and
    reporting those as the same thing is how a refuted result gets retried
    until it passes.
    """
    found: list[str] = []
    vanilla = bundle.control("vanilla")
    if (
        vanilla is not None
        and bundle.verifier_score is not None
        and bundle.verifier_score <= vanilla.score
    ):
        found.append(
            f"treatment_did_not_beat_vanilla:{bundle.verifier_score:.4f}<={vanilla.score:.4f}"
        )
    if (
        bundle.held_out_score is not None
        and bundle.held_out_baseline is not None
        and bundle.held_out_score <= bundle.held_out_baseline
    ):
        found.append(
            f"held_out_did_not_improve:{bundle.held_out_score:.4f}"
            f"<={bundle.held_out_baseline:.4f}"
        )
    if (
        bundle.subject_identity
        and bundle.verifier_identity
        and bundle.subject_identity == bundle.verifier_identity
    ):
        found.append("subject_graded_itself")
    return tuple(found)


def adjudicate(bundle: EvidenceBundle) -> Verdict:
    """Decide whether the evidence establishes the claim. Refuses by default.

    There is no argument that makes this return CERTIFIED without the
    required evidence being present, and no ordering of checks that skips
    the contradiction pass.
    """
    if not isinstance(bundle, EvidenceBundle):
        raise EvidenceAdjudicationError("bundle must be an EvidenceBundle")
    if not isinstance(bundle.claim, ClaimClass):
        raise EvidenceAdjudicationError("bundle.claim must be a ClaimClass")

    requirements = required_evidence(bundle.claim)
    satisfied: list[str] = []
    missing: list[str] = []
    for name in requirements:
        try:
            ok = bool(_CHECKS[name](bundle))
        except (AttributeError, TypeError, ValueError) as exc:
            # A check that cannot run has not passed. Recorded as missing,
            # never skipped — a skipped check is an absent check reported
            # as a passed one.
            record_degradation(
                "independent_evidence",
                exc,
                severity="warning",
                action=f"treated evidence check {name!r} as unmet after it failed to run",
            )
            ok = False
        (satisfied if ok else missing).append(name)

    contradictions = _contradictions(bundle)
    if contradictions:
        status = VerdictStatus.REFUTED
        grade = VerificationGrade.NONE
    elif missing:
        status = VerdictStatus.INSUFFICIENT
        grade = VerificationGrade.ASSERTED if satisfied else VerificationGrade.NONE
    else:
        status = VerdictStatus.CERTIFIED
        grade = _CERTIFIED_GRADE[bundle.claim]

    at = time.time()
    verdict = Verdict(
        claim=bundle.claim,
        status=status,
        grade=grade,
        satisfied=tuple(satisfied),
        missing=tuple(missing),
        contradictions=contradictions,
        subject_identity=bundle.subject_identity,
        verifier_identity=bundle.verifier_identity,
        at=at,
        signature="",
    )
    return _signed(verdict)


# --------------------------------------------------------------------------
# Verdict signing. A verdict that can be edited into a pass is a suggestion.
# --------------------------------------------------------------------------

_KEY_LOCK = checked_lock("independent_evidence")
_KEY: bytes | None = None


def _verdict_key() -> bytes | None:
    """Local signing key, created once at mode 600 under this runtime's root.

    Under ``state_root()``, so a test run signs with a test key and cannot
    mint verdicts the live instance would accept.
    """
    global _KEY
    with _KEY_LOCK:
        if _KEY is not None:
            return _KEY
        path = state_root() / "keys" / "independent_evidence.key"
        try:
            if path.exists():
                material = path.read_bytes()
                _KEY = material if len(material) == 32 else None
                return _KEY
            candidate = os.urandom(32)
            with local_internal_governed_scope(
                "independent_evidence.signing_key",
                domain="file_write",
            ):
                material = get_file_write_gateway().provision_private_bytes(
                    path,
                    candidate,
                    expected_size=32,
                    mode=0o600,
                    source="independent_evidence.signing_key",
                )
            _KEY = material
        except (OSError, ValueError) as exc:
            record_degradation(
                "independent_evidence",
                exc,
                severity="warning",
                action="issued an unsigned verdict after the local key was unavailable",
            )
            _KEY = None
        return _KEY


def _verdict_body(verdict: Verdict) -> bytes:
    return json.dumps(
        {
            "claim": verdict.claim.value,
            "status": verdict.status.value,
            "grade": verdict.grade.value,
            "satisfied": list(verdict.satisfied),
            "missing": list(verdict.missing),
            "contradictions": list(verdict.contradictions),
            "subject_identity": verdict.subject_identity,
            "verifier_identity": verdict.verifier_identity,
            "at": verdict.at,
        },
        sort_keys=True,
    ).encode("utf-8")


def _signed(verdict: Verdict) -> Verdict:
    key = _verdict_key()
    if key is None:
        return verdict
    signature = hmac.new(key, _verdict_body(verdict), hashlib.sha256).hexdigest()
    return Verdict(**{**verdict.__dict__, "signature": signature})


def verdict_signature_valid(verdict: Verdict) -> bool:
    """Whether a verdict still says what it was signed saying.

    An unsigned verdict is NOT valid. A missing signature is the state an
    attacker would produce by deleting one, and treating it as acceptable
    would make the signature optional in practice.
    """
    key = _verdict_key()
    if key is None or not verdict.signature:
        return False
    expected = hmac.new(
        key, _verdict_body(Verdict(**{**verdict.__dict__, "signature": ""})), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(verdict.signature, expected)


def summarize(verdicts: Iterable[Verdict]) -> dict[str, Any]:
    """Health-surface view of what has actually been established."""
    counts = {status.value: 0 for status in VerdictStatus}
    missing_tally: dict[str, int] = {}
    for verdict in verdicts:
        counts[verdict.status.value] = counts.get(verdict.status.value, 0) + 1
        for name in verdict.missing:
            missing_tally[name] = missing_tally.get(name, 0) + 1
    return {
        "verdicts": counts,
        "most_common_missing_evidence": sorted(
            missing_tally.items(), key=lambda item: (-item[1], item[0])
        )[:5],
    }
