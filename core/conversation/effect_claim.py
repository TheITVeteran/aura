"""Effect claims that cannot be rendered without the evidence for them.

The honesty checks in claimed_effect.py read finished prose and look for
sentences that assert a completed effect. That is detection, and detection over
natural language cannot be closed: Aura can form indefinitely many sentences
describing the world, and any finite set of patterns recognises a subset of
them. The inequality is permanent —

    {claims she can generate} ⊃ {claims the auditor recognises}

— so every phrasing added to the auditor narrows the gap and none of them shuts
it. Raised against this codebase on 2026-08-10, and correct.

This module is the other half. Rather than inspecting a sentence after it
exists, an EffectClaim is a typed proposition that carries its own evidence,
and the renderer refuses to produce completed-effect text for a claim with no
receipt behind it. For the path that renders effects, an unevidenced claim is
not caught — it is unrepresentable.

    render(claim) is defined only when claim.evidence_ids is non-empty
    and claim.tense is COMPLETED

That is an architectural invariant rather than a growing list of detectors, and
it is enforced by construction: EffectClaim.completed() raises without
evidence, so no caller can build one.

Scope, stated plainly because the distinction is the whole point: this closes
the loop where Aura RENDERS an effect from a receipt. Free-form text generated
by the model remains audited by claimed_effect.py rather than constrained by
this, and that surface is still a detector. What this removes is the class of
false claim that the runtime itself was composing — the "Done — wrote X"
sentences assembled from a result payload, which is exactly where a receipt was
already in hand and was not being checked against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "EffectClaim",
    "assertions_from_receipts",
    "EffectTense",
    "UnevidencedClaimError",
    "claims_from_receipts",
    "render_effect_claims",
]


class UnevidencedClaimError(ValueError):
    """A completed effect was asserted with no receipt behind it."""


class EffectTense(StrEnum):
    COMPLETED = "completed"
    ATTEMPTED = "attempted"
    REFUSED = "refused"


#: How each action reads in a sentence, and which receipt field names its
#: object. Keyed by the action vocabulary, which is finite and declared — that
#: is what makes this closable where language is not.
_EFFECT_VOCABULARY: dict[str, tuple[str, tuple[str, ...]]] = {
    "write_text_file": ("wrote", ("path",)),
    "render_text_pdf": ("rendered", ("path",)),
    "create_folder": ("created the folder", ("path",)),
    "move_file": ("moved", ("destination", "path")),
    "list_directory": ("read", ("path",)),
    "open_app": ("opened", ("opened", "app")),
    "open_url": ("opened", ("url",)),
    "set_clipboard": ("put text on the clipboard", ()),
    "write_in_app": ("wrote a document in", ("app",)),
    "create_note": ("created a note in", ("app",)),
    "system_control": ("changed a system setting", ()),
    "move_aura_bubble": ("moved her bubble", ()),
    "run_command": ("ran a command", ()),
    "run_applescript": ("ran a script", ()),
    "fetch_topic_image": ("fetched an image", ("path",)),
}


@dataclass(frozen=True, slots=True)
class EffectClaim:
    """A statement about an effect, bound to the receipts that support it."""

    action: str
    tense: EffectTense
    obj: str = ""
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.tense is EffectTense.COMPLETED and not self.evidence_ids:
            raise UnevidencedClaimError(
                f"completed effect {self.action!r} has no evidence; a completed "
                "claim without a receipt cannot be constructed"
            )

    @classmethod
    def completed(
        cls, action: str, *, obj: str = "", evidence_ids: tuple[str, ...]
    ) -> EffectClaim:
        """The only way to assert a finished effect. Raises without evidence."""
        return cls(
            action=action,
            tense=EffectTense.COMPLETED,
            obj=obj,
            evidence_ids=tuple(evidence_ids),
        )

    @classmethod
    def attempted(cls, action: str, *, obj: str = "") -> EffectClaim:
        """A step that ran without verifying. Never renders as done."""
        return cls(action=action, tense=EffectTense.ATTEMPTED, obj=obj)

    def render(self) -> str:
        """Words for this claim, or "" when it has nothing it may assert."""
        if self.tense is not EffectTense.COMPLETED:
            return ""
        phrase, _fields = _EFFECT_VOCABULARY.get(self.action, ("", ()))
        if not phrase:
            return ""
        return f"{phrase} {self.obj}".strip()


def claims_from_receipts(receipts: Any) -> list[EffectClaim]:
    """Turn step receipts into claims, completed only where they verified.

    receipt["ok"] is the strong signal: the tool returned success AND
    _verify_step_effect confirmed the effect. Anything short of that becomes an
    ATTEMPTED claim, which renders as nothing.
    """

    if not isinstance(receipts, list):
        return []
    claims: list[EffectClaim] = []
    for index, receipt in enumerate(receipts):
        if not isinstance(receipt, dict):
            continue
        action = str(receipt.get("action") or "").strip()
        if action not in _EFFECT_VOCABULARY:
            continue
        inner = receipt.get("result")
        payload = inner if isinstance(inner, dict) else receipt
        _phrase, fields = _EFFECT_VOCABULARY[action]
        obj = ""
        for name in fields:
            value = str(payload.get(name) or "").strip()
            if value:
                obj = value
                break
        if receipt.get("ok"):
            identifier = str(
                receipt.get("durable_receipt_id") or f"step:{receipt.get('index', index)}"
            )
            claims.append(
                EffectClaim.completed(action, obj=obj, evidence_ids=(identifier,))
            )
        else:
            claims.append(EffectClaim.attempted(action, obj=obj))
    return claims


def assertions_from_receipts(receipts: Any) -> list[Any]:
    """The same effects as Assertions on the shared epistemic substrate.

    EffectClaim came first and covers desktop effects only. Assertion is the
    general form — subject, provenance, evidence, confidence, time, source and
    verification for anything checkable — and having two types for one idea was
    the fragmentation this work was criticised for. Effects are lowered onto
    the substrate here rather than kept beside it.
    """

    from core.epistemics.assertion import Assertion, SourceKind, Verification

    lowered: list[Any] = []
    for claim in claims_from_receipts(receipts):
        rendered = claim.render()
        completed = claim.tense is EffectTense.COMPLETED
        lowered.append(
            Assertion(
                subject=claim.obj or claim.action,
                claim=f"I {rendered}" if rendered else f"I ran {claim.action}",
                source=SourceKind.RECEIPTED if completed else SourceKind.GENERATED,
                provenance=f"desktop_task step receipt ({claim.action})",
                evidence=claim.evidence_ids,
                verification=(
                    Verification.VERIFIED if completed else Verification.UNVERIFIED
                ),
            )
        )
    return lowered


def render_effect_claims(receipts: Any) -> str:
    """One sentence naming every effect that actually verified, or "".

    This is what the desktop reply says it did. It cannot overstate the work,
    because a claim it would have to overstate cannot be built.
    """

    rendered = [claim.render() for claim in claims_from_receipts(receipts)]
    rendered = [text for text in rendered if text]
    if not rendered:
        return ""
    if len(rendered) == 1:
        return f"{rendered[0]}."
    return ", ".join(rendered[:-1]) + f", and {rendered[-1]}."
