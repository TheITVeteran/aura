"""Bidirectional, lesion-checked coupling between the RLC and the organism.

SPARK-067 asks for epistemic state and recurrent control to be connected
*causally* to agency/Will, memory, tools, personality/self-model, affect/body,
global workspace, reasoning amplifiers, goals, and learning — and it names the
thing that must not be accepted instead: **metadata-only coupling**.

That phrase is doing a lot of work, so this module makes it mechanical. A field
copied from the epistemic state into another subsystem's receipt is not a
coupling. It is a field that was copied. The subsystem behaved identically
before and after; nothing downstream of it can tell the difference; and the
receipt showing the field present reads exactly like evidence. So:

- Every seam declares its evidence **kind**. `metadata` is a legal thing to
  record and an illegal thing to count — a seam whose forward or reverse
  evidence is metadata is refused as `metadata_only_coupling`, by name.
- Every seam needs **both directions**. A recurrent controller that reads
  memory but cannot change what memory keeps is coupled to memory the way a
  thermometer is coupled to weather. One direction is `unidirectional_coupling`.
- Every seam needs a **lesion that removes the effect**. If cutting the seam
  leaves the measured behavior where it was, whatever was measured was not
  flowing through that seam. Coupling asserted without a lesion is refused
  before it can be reported as proven.
- The subsystem set is **complete by declaration**. Nine seams, exactly; a
  matrix missing one is invalid rather than partially passing.

No model imports: an independent verifier decides whether the organism is
coupled without booting the organism.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final, Never

COUPLING_MATRIX_SCHEMA: Final = "aura.rlc.coupling_matrix.v1"
SEAM_SCHEMA: Final = "aura.rlc.coupling_seam.v1"

AGENCY_WILL: Final = "agency_will"
MEMORY: Final = "memory"
TOOLS: Final = "tools"
PERSONALITY_SELF_MODEL: Final = "personality_self_model"
AFFECT_BODY: Final = "affect_body"
GLOBAL_WORKSPACE: Final = "global_workspace"
REASONING_AMPLIFIERS: Final = "reasoning_amplifiers"
GOALS: Final = "goals"
LEARNING: Final = "learning"

COUPLED_SUBSYSTEMS: Final = (
    AGENCY_WILL,
    MEMORY,
    TOOLS,
    PERSONALITY_SELF_MODEL,
    AFFECT_BODY,
    GLOBAL_WORKSPACE,
    REASONING_AMPLIFIERS,
    GOALS,
    LEARNING,
)

FORWARD: Final = "rlc_to_subsystem"
REVERSE: Final = "subsystem_to_rlc"
DIRECTIONS: Final = (FORWARD, REVERSE)

BEHAVIORAL: Final = "behavioral"
METADATA: Final = "metadata"
EVIDENCE_KINDS: Final = (BEHAVIORAL, METADATA)

COUPLED: Final = "COUPLED"
REFUSED: Final = "REFUSED"

# A lesion must remove most of the effect. Anything less and the seam was not
# where the behavior was flowing.
_REQUIRED_LESION_REMOVAL: Final = 0.5
_MINIMUM_OBSERVATIONS: Final = 32

_SHA256_PATTERN: Final = re.compile(r"\A[0-9a-f]{64}\Z")
_EFFECT_FIELDS: Final = frozenset(
    {"direction", "kind", "baseline_statistic", "observed_statistic", "observations", "evidence_sha256"}
)
_LESION_FIELDS: Final = frozenset(
    {"intact_statistic", "lesioned_statistic", "baseline_statistic", "observations", "evidence_sha256"}
)
_SEAM_FIELDS: Final = frozenset({"schema", "subsystem", "forward", "reverse", "lesion", "seam_sha256"})


class CouplingMatrixError(ValueError):
    """A coupling seam or matrix is invalid."""


def _fail(code: str) -> Never:
    raise CouplingMatrixError(str(code or "coupling_matrix_invalid"))


def _sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        raise CouplingMatrixError("coupling_matrix_noncanonical_value") from exc
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_PATTERN.match(value))


def _finite(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(code)
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        _fail(code)
    return round(number, 9)


def _count(value: Any, code: str) -> int:
    if type(value) is not int or value < 0:
        _fail(code)
    return value


def coupling_effect(
    *,
    direction: str,
    kind: str,
    baseline_statistic: float,
    observed_statistic: float,
    observations: int,
    evidence_sha256: str,
) -> dict[str, Any]:
    """One measured direction of one seam."""

    if direction not in DIRECTIONS:
        _fail("coupling_matrix_direction_unknown")
    if kind not in EVIDENCE_KINDS:
        _fail("coupling_matrix_evidence_kind_unknown")
    if not _is_sha256(evidence_sha256):
        _fail("coupling_matrix_evidence_invalid")
    return {
        "direction": direction,
        "kind": kind,
        "baseline_statistic": _finite(
            baseline_statistic, "coupling_matrix_statistic_invalid"
        ),
        "observed_statistic": _finite(
            observed_statistic, "coupling_matrix_statistic_invalid"
        ),
        "observations": _count(observations, "coupling_matrix_observations_invalid"),
        "evidence_sha256": evidence_sha256,
    }


def lesion_result(
    *,
    baseline_statistic: float,
    intact_statistic: float,
    lesioned_statistic: float,
    observations: int,
    evidence_sha256: str,
) -> dict[str, Any]:
    """What happened to the effect when the seam was cut."""

    if not _is_sha256(evidence_sha256):
        _fail("coupling_matrix_lesion_evidence_invalid")
    return {
        "baseline_statistic": _finite(
            baseline_statistic, "coupling_matrix_statistic_invalid"
        ),
        "intact_statistic": _finite(
            intact_statistic, "coupling_matrix_statistic_invalid"
        ),
        "lesioned_statistic": _finite(
            lesioned_statistic, "coupling_matrix_statistic_invalid"
        ),
        "observations": _count(observations, "coupling_matrix_observations_invalid"),
        "evidence_sha256": evidence_sha256,
    }


def coupling_seam(
    *,
    subsystem: str,
    forward: Mapping[str, Any],
    reverse: Mapping[str, Any],
    lesion: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble one seam, refusing incoherent direction labels."""

    if subsystem not in COUPLED_SUBSYSTEMS:
        _fail("coupling_matrix_subsystem_unknown")
    for value, expected in ((forward, FORWARD), (reverse, REVERSE)):
        if not isinstance(value, Mapping) or set(value) != _EFFECT_FIELDS:
            _fail("coupling_matrix_effect_fields_differ")
        if value.get("direction") != expected:
            _fail("coupling_matrix_direction_mismatch")
    if not isinstance(lesion, Mapping) or set(lesion) != _LESION_FIELDS:
        _fail("coupling_matrix_lesion_fields_differ")

    body = {
        "schema": SEAM_SCHEMA,
        "subsystem": subsystem,
        "forward": coupling_effect(
            direction=forward["direction"],
            kind=forward["kind"],
            baseline_statistic=forward["baseline_statistic"],
            observed_statistic=forward["observed_statistic"],
            observations=forward["observations"],
            evidence_sha256=forward["evidence_sha256"],
        ),
        "reverse": coupling_effect(
            direction=reverse["direction"],
            kind=reverse["kind"],
            baseline_statistic=reverse["baseline_statistic"],
            observed_statistic=reverse["observed_statistic"],
            observations=reverse["observations"],
            evidence_sha256=reverse["evidence_sha256"],
        ),
        "lesion": lesion_result(
            baseline_statistic=lesion["baseline_statistic"],
            intact_statistic=lesion["intact_statistic"],
            lesioned_statistic=lesion["lesioned_statistic"],
            observations=lesion["observations"],
            evidence_sha256=lesion["evidence_sha256"],
        ),
    }
    return {**body, "seam_sha256": _sha256(body)}


def _seam_refusals(seam: Mapping[str, Any]) -> list[dict[str, Any]]:
    refusals: list[dict[str, Any]] = []

    for label, effect in (("forward", seam["forward"]), ("reverse", seam["reverse"])):
        if effect["kind"] == METADATA:
            # A field copied across a boundary is a field that was copied. It
            # is legal to record and illegal to count as coupling.
            refusals.append(
                {
                    "reason": "metadata_only_coupling",
                    "direction": label,
                    "kind": METADATA,
                }
            )
            continue
        if effect["observations"] < _MINIMUM_OBSERVATIONS:
            refusals.append(
                {
                    "reason": "insufficient_observations",
                    "direction": label,
                    "observations": effect["observations"],
                    "required": _MINIMUM_OBSERVATIONS,
                }
            )
        if effect["observed_statistic"] == effect["baseline_statistic"]:
            refusals.append(
                {
                    "reason": "no_measured_effect",
                    "direction": label,
                    "statistic": effect["observed_statistic"],
                }
            )

    lesion = seam["lesion"]
    intact_effect = abs(lesion["intact_statistic"] - lesion["baseline_statistic"])
    lesioned_effect = abs(lesion["lesioned_statistic"] - lesion["baseline_statistic"])
    if intact_effect <= 0.0:
        refusals.append({"reason": "lesion_had_no_intact_effect_to_remove"})
    elif lesioned_effect > intact_effect * (1.0 - _REQUIRED_LESION_REMOVAL):
        # Cutting the seam left the behavior where it was, so the behavior was
        # not flowing through the seam.
        refusals.append(
            {
                "reason": "lesion_did_not_remove_effect",
                "intact_effect": round(intact_effect, 9),
                "lesioned_effect": round(lesioned_effect, 9),
                "required_removal_fraction": _REQUIRED_LESION_REMOVAL,
            }
        )
    if lesion["observations"] < _MINIMUM_OBSERVATIONS:
        refusals.append(
            {
                "reason": "insufficient_observations",
                "direction": "lesion",
                "observations": lesion["observations"],
                "required": _MINIMUM_OBSERVATIONS,
            }
        )

    return refusals


def coupling_matrix(seams: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Judge the whole organism-wide coupling claim, seam by seam.

    Every subsystem must be present exactly once. A matrix that omits a
    subsystem is invalid rather than passing on the ones it bothered to
    measure.
    """

    if not isinstance(seams, Sequence) or isinstance(seams, (str, bytes)):
        _fail("coupling_matrix_invalid")

    rows: dict[str, dict[str, Any]] = {}
    for raw in seams:
        if not isinstance(raw, Mapping) or set(raw) != _SEAM_FIELDS:
            _fail("coupling_matrix_seam_fields_differ")
        normalized = coupling_seam(
            subsystem=raw["subsystem"],
            forward=raw["forward"],
            reverse=raw["reverse"],
            lesion=raw["lesion"],
        )
        if dict(raw) != normalized:
            _fail("coupling_matrix_seam_differs")
        if normalized["subsystem"] in rows:
            _fail("coupling_matrix_seam_duplicate")
        rows[normalized["subsystem"]] = normalized

    if set(rows) != set(COUPLED_SUBSYSTEMS):
        _fail("coupling_matrix_incomplete")

    verdicts: list[dict[str, Any]] = []
    for subsystem in COUPLED_SUBSYSTEMS:
        refusals = _seam_refusals(rows[subsystem])
        verdicts.append(
            {
                "subsystem": subsystem,
                "verdict": REFUSED if refusals else COUPLED,
                "seam_sha256": rows[subsystem]["seam_sha256"],
                "refusals": refusals,
            }
        )

    uncoupled = [row["subsystem"] for row in verdicts if row["verdict"] != COUPLED]
    body = {
        "schema": COUPLING_MATRIX_SCHEMA,
        "verdict": REFUSED if uncoupled else COUPLED,
        "subsystems_required": list(COUPLED_SUBSYSTEMS),
        "seams": verdicts,
        "uncoupled_subsystems": uncoupled,
    }
    return {**body, "matrix_sha256": _sha256(body)}


__all__ = [
    "AFFECT_BODY",
    "AGENCY_WILL",
    "BEHAVIORAL",
    "COUPLED",
    "COUPLED_SUBSYSTEMS",
    "COUPLING_MATRIX_SCHEMA",
    "DIRECTIONS",
    "EVIDENCE_KINDS",
    "FORWARD",
    "GLOBAL_WORKSPACE",
    "GOALS",
    "LEARNING",
    "MEMORY",
    "METADATA",
    "PERSONALITY_SELF_MODEL",
    "REASONING_AMPLIFIERS",
    "REFUSED",
    "REVERSE",
    "SEAM_SCHEMA",
    "TOOLS",
    "CouplingMatrixError",
    "coupling_effect",
    "coupling_matrix",
    "coupling_seam",
    "lesion_result",
]
