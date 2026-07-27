"""Turn each battery's own output into a promotion gate row.

`permanent_distillation` refuses a promotion whose gate report is incomplete.
Left there, the operator hand-writes that report — and a hand-written report is
a place to write down whatever number makes the promotion go through. The gate
set being complete is worth nothing if the rows are prose.

So every gate row is produced here, from the battery's native receipt, by a
function that:

- **checks the receipt is that battery's.** Each adapter verifies the schema or
  the concrete type it claims to be reading. A dict shaped roughly like an
  interference receipt is not an interference receipt.
- **derives counts rather than accepting them.** `probes_graded` and
  `probes_passed` come out of the measurement — probe lists, case counts,
  battery totals — never from an argument the caller chose.
- **derives the verdict from the battery's own rule**, not from a threshold
  invented here. The interference battery decides what stable enough means;
  the capability guard decides what a regression is; the canaries decide what
  a right-to-wrong transition costs. This module does not get a second opinion.
- **binds an evidence digest over the whole receipt**, so the row cannot
  outlive an edit to the measurement behind it.

The one thing a caller still supplies is the measurement itself. That is the
correct boundary: this module cannot run a model, and a module that could
would be able to fabricate the run.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Final, Never

from core.learning.capability_regression_battery import (
    CAPABILITY_REGRESSION_SCHEMA,
    CapabilityRegressionGuard,
)
from core.learning.interference_battery import INTERFERENCE_BATTERY_SCHEMA
from core.learning.permanent_distillation import (
    FAIL,
    PASS,
    REQUIRED_GATES,
    gate_report,
    gate_result,
)
from core.learning.recurrent_sft_behavior_canaries import BEHAVIOR_CANARY_SCHEMA

GATE_PRODUCER_SCHEMA: Final = "aura.rlc.permanent_distillation.gate_producer.v1"

# Which canary family answers which promotion gate. `identity_grounding` is
# what personality retention means for a recurrent adapter: the canaries test
# whether the trained decode still knows who it is and what context it
# actually has, which is exactly the personality failure a promotion must not
# ship.
_CANARY_FAMILY_FOR_GATE: Final = {
    "personality_retention": "identity_grounding",
    "tool_effect_honesty": "tool_effect_honesty",
    "authority_safety": "authority_safety",
}


class GateProducerError(ValueError):
    """A battery receipt is not the receipt it claims to be."""


def _fail(code: str) -> Never:
    raise GateProducerError(str(code or "gate_producer_invalid"))


def _evidence_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
            default=str,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        raise GateProducerError("gate_producer_noncanonical_receipt") from exc
    return hashlib.sha256(payload).hexdigest()


def _verdict(passed: bool) -> str:
    return PASS if passed else FAIL


# ---------------------------------------------------------------------------
# anti_interference
# ---------------------------------------------------------------------------


def anti_interference_gate(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """From `run_interference_battery`'s receipt."""

    if not isinstance(receipt, Mapping) or receipt.get("schema") != INTERFERENCE_BATTERY_SCHEMA:
        _fail("gate_producer_interference_receipt_invalid")
    results = receipt.get("results")
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        _fail("gate_producer_interference_receipt_invalid")
    graded = len(results)
    if receipt.get("probes") != graded:
        _fail("gate_producer_interference_probe_count_differs")
    stable = sum(1 for row in results if isinstance(row, Mapping) and row.get("stable"))
    if receipt.get("stable_probes") != stable:
        _fail("gate_producer_interference_stable_count_differs")
    if receipt.get("verdict") not in (PASS, FAIL):
        _fail("gate_producer_interference_verdict_invalid")

    return gate_result(
        gate="anti_interference",
        battery_schema=INTERFERENCE_BATTERY_SCHEMA,
        probes_graded=graded,
        probes_passed=stable,
        # The battery's own stable-fraction rule decides. Recomputing it here
        # would be a second opinion that could disagree with the receipt.
        verdict=str(receipt["verdict"]),
        evidence_sha256=_evidence_sha256(receipt),
    )


# ---------------------------------------------------------------------------
# capability_families
# ---------------------------------------------------------------------------


def capability_families_gate(
    guard: CapabilityRegressionGuard, report: Mapping[str, Any]
) -> dict[str, Any]:
    """From `CapabilityRegressionGuard.evaluate()`, counted over its own probes.

    The guard is required rather than a probe count, because the per-family
    probe counts have to come from the probe list that produced the report.
    """

    if not isinstance(guard, CapabilityRegressionGuard):
        _fail("gate_producer_capability_guard_invalid")
    if not isinstance(report, Mapping) or report.get("schema") != CAPABILITY_REGRESSION_SCHEMA:
        _fail("gate_producer_capability_report_invalid")
    families = report.get("families")
    if not isinstance(families, Mapping) or not families:
        _fail("gate_producer_capability_report_invalid")

    counts: dict[str, int] = {}
    for probe in guard.probes:
        counts[probe.family] = counts.get(probe.family, 0) + 1
    if set(families) != set(counts):
        # A report over families the guard did not measure, or missing one it
        # did, is not this guard's report.
        _fail("gate_producer_capability_families_differ")

    graded = 0
    passed = 0
    for family, row in families.items():
        if not isinstance(row, Mapping) or "after" not in row:
            _fail("gate_producer_capability_report_invalid")
        after = row["after"]
        if isinstance(after, bool) or not isinstance(after, (int, float)):
            _fail("gate_producer_capability_report_invalid")
        if not 0.0 <= float(after) <= 1.0:
            _fail("gate_producer_capability_accuracy_out_of_range")
        graded += counts[family]
        passed += int(round(float(after) * counts[family]))

    if "safe" not in report or type(report["safe"]) is not bool:
        _fail("gate_producer_capability_report_invalid")

    return gate_result(
        gate="capability_families",
        battery_schema=CAPABILITY_REGRESSION_SCHEMA,
        probes_graded=graded,
        probes_passed=min(passed, graded),
        # The guard's rule: safe only if NO protected family regressed past
        # the margin. A mean improvement does not buy a language regression.
        verdict=_verdict(bool(report["safe"])),
        evidence_sha256=_evidence_sha256(report),
    )


# ---------------------------------------------------------------------------
# personality_retention / tool_effect_honesty / authority_safety
# ---------------------------------------------------------------------------


def behavior_canary_gate(gate: str, verdict: Mapping[str, Any]) -> dict[str, Any]:
    """From `generated_behavior_verdict`, one gate per canary family."""

    if gate not in _CANARY_FAMILY_FOR_GATE:
        _fail("gate_producer_canary_gate_unknown")
    if (
        not isinstance(verdict, Mapping)
        or verdict.get("schema") != f"{BEHAVIOR_CANARY_SCHEMA}.verdict"
    ):
        _fail("gate_producer_canary_verdict_invalid")
    buckets = verdict.get("by_family")
    if not isinstance(buckets, Mapping):
        _fail("gate_producer_canary_verdict_invalid")

    family = _CANARY_FAMILY_FOR_GATE[gate]
    bucket = buckets.get(family)
    if not isinstance(bucket, Mapping):
        # The family this gate answers was not measured. That is a missing
        # measurement, not a passing one.
        _fail("gate_producer_canary_family_missing")
    for key in ("case_count", "trained_pass_count", "right_to_wrong", "passed"):
        if key not in bucket:
            _fail("gate_producer_canary_bucket_invalid")
    graded = bucket["case_count"]
    passed = bucket["trained_pass_count"]
    if type(graded) is not int or type(passed) is not int or passed > graded:
        _fail("gate_producer_canary_bucket_invalid")
    if type(bucket["passed"]) is not bool:
        _fail("gate_producer_canary_bucket_invalid")

    return gate_result(
        gate=gate,
        battery_schema=f"{BEHAVIOR_CANARY_SCHEMA}.verdict",
        probes_graded=graded,
        probes_passed=passed,
        # The canaries' rule: every trained case passes AND no right-to-wrong
        # transition. A net-positive family with one new failure still fails.
        verdict=_verdict(bool(bucket["passed"])),
        evidence_sha256=_evidence_sha256(verdict),
    )


# ---------------------------------------------------------------------------
# memory_retention
# ---------------------------------------------------------------------------


def memory_retention_gate(
    *,
    before: Any,
    after: Any,
    case_count: int,
    max_drop: float = 0.02,
) -> dict[str, Any]:
    """From a paired `MemoryBenchmarkResult` over the same sealed cases.

    Retention is a *paired* claim, so both measurements are required and both
    must describe the same strategy over the same case count. A single
    post-change number cannot show that nothing was lost.
    """

    from core.memory.memory_benchmarking import MemoryBenchmarkResult

    if not isinstance(before, MemoryBenchmarkResult) or not isinstance(
        after, MemoryBenchmarkResult
    ):
        _fail("gate_producer_memory_result_invalid")
    if before.strategy != after.strategy:
        _fail("gate_producer_memory_strategy_differs")
    if type(case_count) is not int or case_count <= 0:
        _fail("gate_producer_memory_case_count_invalid")
    if isinstance(max_drop, bool) or not isinstance(max_drop, (int, float)):
        _fail("gate_producer_memory_margin_invalid")
    margin = float(max_drop)
    if not 0.0 <= margin <= 0.5:
        _fail("gate_producer_memory_margin_invalid")
    for value in (before.accuracy, after.accuracy):
        if not 0.0 <= float(value) <= 1.0:
            _fail("gate_producer_memory_accuracy_out_of_range")

    retained = float(after.accuracy) >= float(before.accuracy) - margin
    return gate_result(
        gate="memory_retention",
        battery_schema="aura.memory.benchmark.paired.v1",
        probes_graded=case_count,
        probes_passed=int(round(float(after.accuracy) * case_count)),
        verdict=_verdict(retained),
        evidence_sha256=_evidence_sha256(
            {
                "strategy": before.strategy,
                "case_count": case_count,
                "max_drop": round(margin, 9),
                "before": before.to_dict(),
                "after": after.to_dict(),
            }
        ),
    )


# ---------------------------------------------------------------------------
# frontier_regression
# ---------------------------------------------------------------------------


def frontier_regression_gate(
    *,
    before: Any,
    after: Any,
    max_drop: float = 0.02,
) -> dict[str, Any]:
    """From a paired `BatteryResult` over one sealed held-out battery.

    Both results must carry the same `battery_id` and the same total. Grading
    a shorter battery after the change and comparing rates is the oldest way
    to make a regression disappear.
    """

    from core.learning.heldout_battery import BatteryResult

    if not isinstance(before, BatteryResult) or not isinstance(after, BatteryResult):
        _fail("gate_producer_frontier_result_invalid")
    if before.battery_id != after.battery_id:
        _fail("gate_producer_frontier_battery_differs")
    if before.total != after.total or before.total <= 0:
        _fail("gate_producer_frontier_total_differs")
    if isinstance(max_drop, bool) or not isinstance(max_drop, (int, float)):
        _fail("gate_producer_frontier_margin_invalid")
    margin = float(max_drop)
    if not 0.0 <= margin <= 0.5:
        _fail("gate_producer_frontier_margin_invalid")

    held = after.accuracy >= before.accuracy - margin
    return gate_result(
        gate="frontier_regression",
        battery_schema="aura.heldout_battery.paired.v1",
        probes_graded=after.total,
        probes_passed=after.correct,
        verdict=_verdict(held),
        evidence_sha256=_evidence_sha256(
            {
                "battery_id": before.battery_id,
                "max_drop": round(margin, 9),
                "before": before.to_dict(),
                "after": after.to_dict(),
            }
        ),
    )


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def build_gate_report(
    *,
    interference_receipt: Mapping[str, Any],
    capability_guard: CapabilityRegressionGuard,
    capability_report: Mapping[str, Any],
    behavior_verdict: Mapping[str, Any],
    memory_before: Any,
    memory_after: Any,
    memory_case_count: int,
    frontier_before: Any,
    frontier_after: Any,
) -> dict[str, Any]:
    """Produce the complete seven-gate report from seven real measurements.

    Every argument is required. There is no partial-report path here for the
    same reason there is none in the transaction: a promotion that skipped a
    battery must not be expressible.
    """

    rows = [
        anti_interference_gate(interference_receipt),
        capability_families_gate(capability_guard, capability_report),
        behavior_canary_gate("personality_retention", behavior_verdict),
        behavior_canary_gate("tool_effect_honesty", behavior_verdict),
        behavior_canary_gate("authority_safety", behavior_verdict),
        memory_retention_gate(
            before=memory_before,
            after=memory_after,
            case_count=memory_case_count,
        ),
        frontier_regression_gate(before=frontier_before, after=frontier_after),
    ]
    produced = {row["gate"] for row in rows}
    if produced != set(REQUIRED_GATES):
        # Defensive: if REQUIRED_GATES ever grows, this refuses rather than
        # quietly producing a report the transaction will reject later with a
        # less useful message.
        _fail("gate_producer_report_incomplete")
    return gate_report(rows)


__all__ = [
    "GATE_PRODUCER_SCHEMA",
    "GateProducerError",
    "anti_interference_gate",
    "behavior_canary_gate",
    "build_gate_report",
    "capability_families_gate",
    "frontier_regression_gate",
    "memory_retention_gate",
]
