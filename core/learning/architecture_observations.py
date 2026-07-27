"""Derive architecture findings from measurements the RLC already takes.

`architecture_meta_controller` refuses a proposal with no finding behind it.
Left there, the findings come from wherever the caller feels like, and "the
depth is saturating" becomes an opinion typed into a dict.

This produces them from two evidence streams that already exist and are already
recorded per episode:

- **`trajectory_dynamics`** (`core/learning/intrinsic_recurrence.py`) answers
  whether a recurrent loop is converging, spinning, or still moving. Its
  `at_fixed_point`, `contracting`, `oscillating`, `diverged`, and `final_delta`
  fields are exactly what a depth finding is about: a loop that stopped moving
  stopped computing, whatever the compute budget says, and a loop still moving
  hard at the last pass was cut short.
- **action transitions** (`value_of_computation`) record which cognitive
  operator was actually selected at each step. In a dense checkpoint there is
  no MoE router to measure, so "expert" and "router" mean what they actually
  mean here: the operator inventory and the policy that selects among it. A
  vocabulary entry that is never selected is a dead expert; one taking most of
  the traffic is overloaded; a selection distribution with almost no entropy
  is a collapsed router.

Two rules make these findings rather than assertions:

- **The episode count is derived from the evidence**, never passed in. An
  observation claiming 512 episodes has 512 reports behind it or it does not
  exist. This is what the meta-controller's 64-episode floor is protecting,
  and it can only protect it if the count is real.
- **Malformed evidence is refused, not skipped.** A dynamics report that is
  not measurable, or a transition that is not an action transition, stops the
  derivation. Silently dropping unreadable episodes would shrink the
  denominator and inflate every rate computed from it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Final, Never

from core.learning.architecture_meta_controller import architecture_observation

ARCHITECTURE_OBSERVATION_PRODUCER: Final = "aura.rlc.architecture_observations.v1"

# A loop whose final relative delta is below this has stopped moving. It is the
# same 0.01 `trajectory_dynamics` uses for `at_fixed_point`; naming a second
# number here would let the two disagree about the same event.
_FIXED_POINT_DELTA: Final = 0.01
# Still moving this much at the last pass means the budget, not the
# computation, ended the loop.
_STARVED_DELTA: Final = 0.10

# Defaults for what counts as a finding. They are thresholds carried *in* the
# observation, so a reader sees the bar the statistic was judged against.
_DEFAULT_SATURATION_THRESHOLD: Final = 0.20
_DEFAULT_STARVATION_THRESHOLD: Final = 0.20
_DEFAULT_DEAD_EXPERT_THRESHOLD: Final = 0.0
_DEFAULT_OVERLOAD_THRESHOLD: Final = 0.60
_DEFAULT_COLLAPSE_THRESHOLD: Final = 0.35


class ArchitectureObservationError(ValueError):
    """The evidence behind an architecture observation is unusable."""


def _fail(code: str) -> Never:
    raise ArchitectureObservationError(str(code or "architecture_observation_invalid"))


def _evidence_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        raise ArchitectureObservationError(
            "architecture_observation_noncanonical_evidence"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _dynamics_rows(reports: Any) -> list[dict[str, Any]]:
    from core.learning.intrinsic_recurrence import INTRINSIC_RECURRENCE_SCHEMA

    if not isinstance(reports, Sequence) or isinstance(reports, (str, bytes)):
        _fail("architecture_observation_dynamics_invalid")
    if not reports:
        _fail("architecture_observation_dynamics_empty")
    rows: list[dict[str, Any]] = []
    for raw in reports:
        if not isinstance(raw, Mapping) or raw.get("schema") != INTRINSIC_RECURRENCE_SCHEMA:
            _fail("architecture_observation_dynamics_invalid")
        if raw.get("diverged"):
            # A non-finite loop is a different defect with a different repair;
            # letting it into a depth rate would hide it inside an average.
            _fail("architecture_observation_dynamics_diverged")
        if not raw.get("measurable"):
            # Fewer than two iterations: nothing about depth is askable of it.
            _fail("architecture_observation_dynamics_not_measurable")
        delta = raw.get("final_delta")
        if isinstance(delta, bool) or not isinstance(delta, (int, float)):
            _fail("architecture_observation_dynamics_invalid")
        rows.append(dict(raw))
    return rows


def depth_observations(
    reports: Sequence[Mapping[str, Any]],
    *,
    saturation_threshold: float = _DEFAULT_SATURATION_THRESHOLD,
    starvation_threshold: float = _DEFAULT_STARVATION_THRESHOLD,
) -> list[dict[str, Any]]:
    """Two depth observations from a run of `trajectory_dynamics` reports.

    Saturation and starvation are opposite failures of the same knob and are
    measured over the same episodes, so both are always produced. Reporting
    only the one that happens to be over threshold would hide the case where
    a depth change fixes one by causing the other.
    """

    rows = _dynamics_rows(reports)
    episodes = len(rows)

    saturated = sum(
        1
        for row in rows
        if bool(row.get("at_fixed_point")) or float(row["final_delta"]) < _FIXED_POINT_DELTA
    )
    starved = sum(1 for row in rows if float(row["final_delta"]) >= _STARVED_DELTA)

    digest = _evidence_sha256(
        {
            "producer": ARCHITECTURE_OBSERVATION_PRODUCER,
            "episodes": episodes,
            "final_deltas": [round(float(row["final_delta"]), 6) for row in rows],
            "at_fixed_point": [bool(row.get("at_fixed_point")) for row in rows],
        }
    )
    return [
        architecture_observation(
            failure_mode="depth_saturation",
            episodes=episodes,
            statistic=round(saturated / episodes, 9),
            threshold=float(saturation_threshold),
            evidence_sha256=digest,
        ),
        architecture_observation(
            failure_mode="depth_starvation",
            episodes=episodes,
            statistic=round(starved / episodes, 9),
            threshold=float(starvation_threshold),
            evidence_sha256=digest,
        ),
    ]


def _selection_counts(transitions: Any) -> tuple[dict[str, int], int]:
    from core.brain.llm.latent_cortex.epistemic_state import OperationKind
    from core.brain.llm.latent_cortex.value_of_computation import (
        ACTION_TRANSITION_SCHEMA,
    )

    if not isinstance(transitions, Sequence) or isinstance(transitions, (str, bytes)):
        _fail("architecture_observation_transitions_invalid")
    if not transitions:
        _fail("architecture_observation_transitions_empty")

    counts = {action.value: 0 for action in OperationKind}
    total = 0
    for raw in transitions:
        if not isinstance(raw, Mapping) or raw.get("schema") != ACTION_TRANSITION_SCHEMA:
            _fail("architecture_observation_transitions_invalid")
        action = raw.get("action")
        if action not in counts:
            _fail("architecture_observation_transition_action_unknown")
        counts[action] += 1
        total += 1
    return counts, total


def operator_observations(
    transitions: Sequence[Mapping[str, Any]],
    *,
    dead_expert_threshold: float = _DEFAULT_DEAD_EXPERT_THRESHOLD,
    overload_threshold: float = _DEFAULT_OVERLOAD_THRESHOLD,
    collapse_threshold: float = _DEFAULT_COLLAPSE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Expert and router observations from real action transitions.

    `router_collapse`'s statistic is *one minus* the normalized selection
    entropy, so that — like every other failure mode here — a larger number is
    worse and `statistic > threshold` is a finding. A uniform policy scores 0;
    a policy that always picks the same operator scores 1.
    """

    counts, total = _selection_counts(transitions)
    vocabulary = len(counts)
    dead = sum(1 for value in counts.values() if value == 0)
    busiest = max(counts.values())

    entropy = 0.0
    for value in counts.values():
        if value:
            share = value / total
            entropy -= share * math.log(share)
    normalized_entropy = entropy / math.log(vocabulary) if vocabulary > 1 else 0.0

    digest = _evidence_sha256(
        {
            "producer": ARCHITECTURE_OBSERVATION_PRODUCER,
            "transitions": total,
            "counts": dict(sorted(counts.items())),
        }
    )
    return [
        architecture_observation(
            failure_mode="dead_expert",
            episodes=total,
            statistic=round(dead / vocabulary, 9),
            threshold=float(dead_expert_threshold),
            evidence_sha256=digest,
        ),
        architecture_observation(
            failure_mode="overloaded_expert",
            episodes=total,
            statistic=round(busiest / total, 9),
            threshold=float(overload_threshold),
            evidence_sha256=digest,
        ),
        architecture_observation(
            failure_mode="router_collapse",
            episodes=total,
            # One minus normalized entropy: bigger is worse, like every other
            # statistic in this vocabulary.
            statistic=round(1.0 - normalized_entropy, 9),
            threshold=float(collapse_threshold),
            evidence_sha256=digest,
        ),
    ]


def observe_architecture(
    *,
    dynamics_reports: Sequence[Mapping[str, Any]],
    action_transitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Every observation the two evidence streams support, in one call.

    `router_misroute` is deliberately absent: nothing currently measures a
    routing decision against a known-correct one, and emitting a zero for it
    would report an unmeasured surface as a healthy one. It shows up in the
    findings report's `surfaces_unmeasured` instead, which is the honest place
    for it.
    """

    return [
        *depth_observations(dynamics_reports),
        *operator_observations(action_transitions),
    ]


__all__ = [
    "ARCHITECTURE_OBSERVATION_PRODUCER",
    "ArchitectureObservationError",
    "depth_observations",
    "observe_architecture",
    "operator_observations",
]
