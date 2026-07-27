"""Durable, contamination-gated accounting for a verified STaR flywheel.

SPARK-063 asks for generate → verify → filter → train → retest on fresh
holdouts → iterate, with durable manifests, and for tool-assisted and latent
traces to enter only after evidence gates. The mechanics of that loop are easy.
The honesty of it is not, and there is one failure that ruins it silently:

**Iteration k's holdout becomes iteration k+1's training data.** The curve goes
up, every individual step looks defensible, and the improvement is memorization
being scored as generalization. Nobody has to cheat for this to happen — a
flywheel that keeps every verified trace and keeps sampling fresh holdouts from
the same pool will do it on its own by the third iteration.

So this ledger's job is arithmetic and set disjointness, checked across the
whole lineage rather than within one iteration:

- A holdout must be disjoint from *this* iteration's training set, from *every
  prior* iteration's training set, and from every holdout already used. A
  holdout that has been scored before is not a fresh holdout, and a task that
  has ever been trained on is not a holdout at all.
- The counts must be a funnel: trained ≤ filtered ≤ verified ≤ generated. A
  filter that admits more than it received is a bookkeeping error at best.
- Every filter rejection is attributed to a named reason, and the reasons must
  account for exactly the drop. "We filtered 400 and 250 remain" with 100
  reasons given is refused rather than rounded.
- A trace class subject to an evidence gate cannot appear in a training set
  until that gate is recorded as passed *in an earlier or the same* iteration.
  Tool-assisted and latent traces are gated by default.

No Aura runtime imports: the lineage must be replayable by an independent
verifier that is not permitted to run cognition.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final, Never

ITERATION_SCHEMA: Final = "aura.rlc.star_iteration.v1"
GATE_SCHEMA: Final = "aura.rlc.star_trace_gate.v1"

GENESIS_PARENT: Final = "0" * 64

DIRECT: Final = "direct"
TOOL_ASSISTED: Final = "tool_assisted"
LATENT: Final = "latent"
TRACE_CLASSES: Final = (DIRECT, TOOL_ASSISTED, LATENT)

# Classes that may not enter a training set on their own merit. A direct trace
# is graded by the same verifier that grades the answer; a tool-assisted or
# latent trace carries provenance the verifier does not see, so it needs its
# own evidence before it is allowed to teach anything.
GATED_TRACE_CLASSES: Final = (TOOL_ASSISTED, LATENT)

_SHA256_PATTERN: Final = re.compile(r"\A[0-9a-f]{64}\Z")
_MAX_FINGERPRINTS: Final = 200_000
_MAX_ITERATIONS: Final = 4096
_GATE_FIELDS: Final = frozenset(
    {"trace_class", "passed", "evidence_sha256", "gate_description"}
)


class StarIterationError(ValueError):
    """A STaR iteration, gate, or lineage contract is invalid."""


class StarContaminationError(StarIterationError):
    """A holdout overlaps training data, here or in an earlier iteration."""

    def __init__(self, detail: Mapping[str, Any]) -> None:
        super().__init__("star_iteration_holdout_contaminated")
        self.detail: dict[str, Any] = dict(detail)


def _fail(code: str) -> Never:
    raise StarIterationError(str(code or "star_iteration_invalid"))


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
        raise StarIterationError("star_iteration_noncanonical_value") from exc
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_PATTERN.match(value))


def _count(value: Any, code: str) -> int:
    if type(value) is not int or value < 0:
        _fail(code)
    return value


def _fingerprints(value: Any, code: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail(code)
    if len(value) > _MAX_FINGERPRINTS:
        _fail(code)
    seen: set[str] = set()
    for item in value:
        if not _is_sha256(item):
            _fail(code)
        if item in seen:
            _fail(code)
        seen.add(item)
    return sorted(seen)


def trace_gate(
    *,
    trace_class: str,
    passed: bool,
    evidence_sha256: str,
    gate_description: str,
) -> dict[str, Any]:
    """Record whether a gated trace class has earned its way into training."""

    if trace_class not in TRACE_CLASSES:
        _fail("star_iteration_trace_class_unknown")
    if trace_class not in GATED_TRACE_CLASSES:
        _fail("star_iteration_trace_class_is_not_gated")
    if type(passed) is not bool:
        _fail("star_iteration_gate_verdict_invalid")
    if not _is_sha256(evidence_sha256):
        _fail("star_iteration_gate_evidence_invalid")
    if (
        not isinstance(gate_description, str)
        or not gate_description.strip()
        or len(gate_description) > 512
    ):
        _fail("star_iteration_gate_description_invalid")
    return {
        "trace_class": trace_class,
        "passed": passed,
        "evidence_sha256": evidence_sha256,
        "gate_description": gate_description,
    }


def star_iteration(
    *,
    iteration_index: int,
    parent_iteration_sha256: str,
    generated: int,
    verified: int,
    filtered: int,
    filter_reasons: Mapping[str, int],
    training_fingerprints: Sequence[str],
    training_trace_classes: Sequence[str],
    holdout_fingerprints: Sequence[str],
    holdout_score: float,
    trace_gates: Sequence[Mapping[str, Any]],
    created_at_unix: int,
) -> dict[str, Any]:
    """One iteration of the flywheel, with its arithmetic checked."""

    index = _count(iteration_index, "star_iteration_index_invalid")
    if parent_iteration_sha256 != GENESIS_PARENT and not _is_sha256(
        parent_iteration_sha256
    ):
        _fail("star_iteration_parent_invalid")

    generated_count = _count(generated, "star_iteration_counts_invalid")
    verified_count = _count(verified, "star_iteration_counts_invalid")
    filtered_count = _count(filtered, "star_iteration_counts_invalid")
    training = _fingerprints(
        training_fingerprints, "star_iteration_training_fingerprints_invalid"
    )
    holdout = _fingerprints(
        holdout_fingerprints, "star_iteration_holdout_fingerprints_invalid"
    )

    # The funnel only narrows. A stage that emitted more than it consumed is a
    # bookkeeping error, and a bookkeeping error here becomes a capability
    # claim later.
    if not len(training) <= filtered_count <= verified_count <= generated_count:
        _fail("star_iteration_funnel_widens")

    if not isinstance(filter_reasons, Mapping):
        _fail("star_iteration_filter_reasons_invalid")
    reasons: dict[str, int] = {}
    for reason, dropped in filter_reasons.items():
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 128:
            _fail("star_iteration_filter_reasons_invalid")
        reasons[reason] = _count(dropped, "star_iteration_filter_reasons_invalid")
    # Every trace that fell out between verification and training is attributed.
    if sum(reasons.values()) != verified_count - len(training):
        _fail("star_iteration_filter_reasons_unaccounted")

    if not isinstance(training_trace_classes, Sequence) or isinstance(
        training_trace_classes, (str, bytes)
    ):
        _fail("star_iteration_trace_classes_invalid")
    classes: list[str] = []
    for name in training_trace_classes:
        if name not in TRACE_CLASSES:
            _fail("star_iteration_trace_class_unknown")
        if name in classes:
            _fail("star_iteration_trace_class_duplicate")
        classes.append(name)
    if not classes and training:
        _fail("star_iteration_trace_classes_invalid")
    classes.sort()

    if not isinstance(trace_gates, Sequence) or isinstance(trace_gates, (str, bytes)):
        _fail("star_iteration_gates_invalid")
    gates: dict[str, dict[str, Any]] = {}
    for raw in trace_gates:
        if not isinstance(raw, Mapping) or set(raw) != _GATE_FIELDS:
            _fail("star_iteration_gate_fields_differ")
        row = trace_gate(
            trace_class=raw["trace_class"],
            passed=raw["passed"],
            evidence_sha256=raw["evidence_sha256"],
            gate_description=raw["gate_description"],
        )
        if row["trace_class"] in gates:
            _fail("star_iteration_gate_duplicate")
        gates[row["trace_class"]] = row

    score = holdout_score
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        _fail("star_iteration_holdout_score_invalid")
    score = round(float(score), 9)
    if score != score or not 0.0 <= score <= 1.0:
        _fail("star_iteration_holdout_score_invalid")
    if not holdout:
        _fail("star_iteration_holdout_missing")

    # Within-iteration contamination: a task cannot be both taught and tested.
    overlap = sorted(set(training) & set(holdout))
    if overlap:
        raise StarContaminationError(
            {
                "scope": "same_iteration",
                "iteration_index": index,
                "overlap_count": len(overlap),
                "overlap_sample": overlap[:8],
            }
        )

    if type(created_at_unix) is not int or created_at_unix <= 0:
        _fail("star_iteration_time_invalid")

    body = {
        "schema": ITERATION_SCHEMA,
        "iteration_index": index,
        "parent_iteration_sha256": parent_iteration_sha256,
        "generated": generated_count,
        "verified": verified_count,
        "filtered": filtered_count,
        "trained": len(training),
        "filter_reasons": dict(sorted(reasons.items())),
        "training_fingerprints": training,
        "training_trace_classes": classes,
        "holdout_fingerprints": holdout,
        "holdout_size": len(holdout),
        "holdout_score": score,
        "trace_gates": [gates[name] for name in sorted(gates)],
        "created_at_unix": created_at_unix,
    }
    return {**body, "iteration_sha256": _sha256(body)}


def validate_star_lineage(records: Any) -> list[dict[str, Any]]:
    """Replay a flywheel and refuse the contaminations a single step cannot see.

    Most of what this catches is invisible from inside one iteration: a holdout
    that was trained on three iterations ago, a holdout scored twice, a gated
    trace class that started training before its gate passed.
    """

    if (
        not isinstance(records, Sequence)
        or isinstance(records, (str, bytes))
        or not records
        or len(records) > _MAX_ITERATIONS
    ):
        _fail("star_iteration_lineage_invalid")

    replayed: list[dict[str, Any]] = []
    trained_ever: set[str] = set()
    holdout_ever: set[str] = set()
    gates_passed: set[str] = set()
    previous = GENESIS_PARENT

    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping) or raw.get("schema") != ITERATION_SCHEMA:
            _fail("star_iteration_invalid")
        normalized = star_iteration(
            iteration_index=raw.get("iteration_index"),
            parent_iteration_sha256=raw.get("parent_iteration_sha256"),
            generated=raw.get("generated"),
            verified=raw.get("verified"),
            filtered=raw.get("filtered"),
            filter_reasons=raw.get("filter_reasons"),
            training_fingerprints=raw.get("training_fingerprints"),
            training_trace_classes=raw.get("training_trace_classes"),
            holdout_fingerprints=raw.get("holdout_fingerprints"),
            holdout_score=raw.get("holdout_score"),
            trace_gates=raw.get("trace_gates"),
            created_at_unix=raw.get("created_at_unix"),
        )
        if dict(raw) != normalized:
            _fail("star_iteration_differs")
        if (
            normalized["iteration_index"] != index
            or normalized["parent_iteration_sha256"] != previous
        ):
            _fail("star_iteration_lineage_chain_differs")

        holdout = set(normalized["holdout_fingerprints"])
        training = set(normalized["training_fingerprints"])

        # A task ever trained on is not a holdout, no matter how many
        # iterations ago that was.
        stale = sorted(holdout & trained_ever)
        if stale:
            raise StarContaminationError(
                {
                    "scope": "earlier_training",
                    "iteration_index": index,
                    "overlap_count": len(stale),
                    "overlap_sample": stale[:8],
                }
            )

        # A holdout scored before is a holdout the policy has already been
        # selected against.
        reused = sorted(holdout & holdout_ever)
        if reused:
            raise StarContaminationError(
                {
                    "scope": "reused_holdout",
                    "iteration_index": index,
                    "overlap_count": len(reused),
                    "overlap_sample": reused[:8],
                }
            )

        # A gate passing in this iteration admits the class from this iteration
        # onward; a class already in training with no passed gate is refused.
        for row in normalized["trace_gates"]:
            if row["passed"]:
                gates_passed.add(row["trace_class"])
            else:
                gates_passed.discard(row["trace_class"])
        ungated = [
            name
            for name in normalized["training_trace_classes"]
            if name in GATED_TRACE_CLASSES and name not in gates_passed
        ]
        if ungated:
            _fail("star_iteration_trace_class_not_admitted")

        trained_ever |= training
        holdout_ever |= holdout
        previous = normalized["iteration_sha256"]
        replayed.append(normalized)

    return replayed


def lineage_trend(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize a validated lineage without editorializing about it.

    The trend is reported alongside the disjointness that makes it meaningful;
    a score series without that context is exactly what this module exists to
    stop anyone from quoting.
    """

    replayed = validate_star_lineage(records)
    scores = [row["holdout_score"] for row in replayed]
    return {
        "iterations": len(replayed),
        "holdout_scores": scores,
        "first_holdout_score": scores[0],
        "last_holdout_score": scores[-1],
        "score_delta": round(scores[-1] - scores[0], 9),
        "monotonic": all(
            later >= earlier for earlier, later in zip(scores, scores[1:], strict=False)
        ),
        "total_trained": sum(row["trained"] for row in replayed),
        "distinct_holdout_tasks": len(
            {
                fingerprint
                for row in replayed
                for fingerprint in row["holdout_fingerprints"]
            }
        ),
        "holdouts_disjoint_from_all_training": True,
    }


__all__ = [
    "DIRECT",
    "GATED_TRACE_CLASSES",
    "GATE_SCHEMA",
    "GENESIS_PARENT",
    "ITERATION_SCHEMA",
    "LATENT",
    "TOOL_ASSISTED",
    "TRACE_CLASSES",
    "StarContaminationError",
    "StarIterationError",
    "lineage_trend",
    "star_iteration",
    "trace_gate",
    "validate_star_lineage",
]
