"""SPARK-063: the flywheel's holdout has to still be a holdout.

The interesting failures are the ones a single iteration cannot see -- a task
trained on three iterations ago reappearing as a holdout, a holdout scored
twice, a gated trace class that started teaching before its gate passed. Those
are what these tests go after.
"""

from __future__ import annotations

import hashlib

import pytest

from core.learning.star_iteration_ledger import (
    GENESIS_PARENT,
    LATENT,
    TOOL_ASSISTED,
    StarContaminationError,
    StarIterationError,
    lineage_trend,
    star_iteration,
    trace_gate,
    validate_star_lineage,
)

_NOW = 1_780_000_000


def _fp(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _tasks(prefix: str, count: int) -> list[str]:
    return [_fp(f"{prefix}-{index}") for index in range(count)]


def _iteration(
    index: int,
    parent: str,
    *,
    training: list[str],
    holdout: list[str],
    score: float = 0.5,
    classes: list[str] | None = None,
    gates: list[dict] | None = None,
    generated: int | None = None,
    verified: int | None = None,
    filtered: int | None = None,
    reasons: dict | None = None,
) -> dict:
    trained = len(training)
    verified_count = verified if verified is not None else trained + 10
    return star_iteration(
        iteration_index=index,
        parent_iteration_sha256=parent,
        generated=generated if generated is not None else verified_count + 20,
        verified=verified_count,
        filtered=filtered if filtered is not None else verified_count,
        filter_reasons=(
            reasons if reasons is not None else {"verifier_rejected": verified_count - trained}
        ),
        training_fingerprints=training,
        training_trace_classes=classes if classes is not None else ["direct"],
        holdout_fingerprints=holdout,
        holdout_score=score,
        trace_gates=gates or [],
        created_at_unix=_NOW + index,
    )


def _clean_lineage(count: int = 3) -> list[dict]:
    records: list[dict] = []
    parent = GENESIS_PARENT
    for index in range(count):
        record = _iteration(
            index,
            parent,
            training=_tasks(f"train{index}", 20),
            holdout=_tasks(f"holdout{index}", 12),
            score=0.4 + 0.1 * index,
        )
        records.append(record)
        parent = record["iteration_sha256"]
    return records


# --- the funnel arithmetic --------------------------------------------------


def test_a_clean_iteration_records_its_funnel():
    record = _iteration(
        0, GENESIS_PARENT, training=_tasks("t", 20), holdout=_tasks("h", 10)
    )
    assert record["trained"] == 20
    assert record["verified"] == 30
    assert record["generated"] == 50


def test_a_funnel_that_widens_is_refused():
    with pytest.raises(StarIterationError) as excinfo:
        _iteration(
            0,
            GENESIS_PARENT,
            training=_tasks("t", 20),
            holdout=_tasks("h", 10),
            verified=5,
        )
    assert "funnel_widens" in str(excinfo.value)


def test_unattributed_filter_losses_are_refused():
    with pytest.raises(StarIterationError) as excinfo:
        _iteration(
            0,
            GENESIS_PARENT,
            training=_tasks("t", 20),
            holdout=_tasks("h", 10),
            verified=30,
            reasons={"verifier_rejected": 3},
        )
    assert "filter_reasons_unaccounted" in str(excinfo.value)


def test_filter_reasons_may_be_split_across_named_causes():
    record = _iteration(
        0,
        GENESIS_PARENT,
        training=_tasks("t", 20),
        holdout=_tasks("h", 10),
        verified=30,
        reasons={"verifier_rejected": 6, "duplicate_prompt": 4},
    )
    assert record["filter_reasons"] == {"duplicate_prompt": 4, "verifier_rejected": 6}


def test_an_iteration_with_no_holdout_is_refused():
    with pytest.raises(StarIterationError) as excinfo:
        _iteration(0, GENESIS_PARENT, training=_tasks("t", 5), holdout=[])
    assert "holdout_missing" in str(excinfo.value)


# --- contamination inside one iteration -------------------------------------


def test_a_task_cannot_be_taught_and_tested_in_the_same_iteration():
    shared = _tasks("shared", 4)
    with pytest.raises(StarContaminationError) as excinfo:
        _iteration(
            0,
            GENESIS_PARENT,
            training=_tasks("t", 10) + shared,
            holdout=_tasks("h", 6) + shared,
        )
    assert excinfo.value.detail["scope"] == "same_iteration"
    assert excinfo.value.detail["overlap_count"] == 4


# --- contamination only the lineage can see ---------------------------------


def test_a_clean_lineage_replays():
    replayed = validate_star_lineage(_clean_lineage())
    assert [row["iteration_index"] for row in replayed] == [0, 1, 2]


def test_a_holdout_trained_on_three_iterations_ago_is_caught():
    records = _clean_lineage(3)
    leaked = records[0]["training_fingerprints"][:5]
    parent = records[-1]["iteration_sha256"]
    records.append(
        _iteration(
            3,
            parent,
            training=_tasks("train3", 20),
            holdout=_tasks("holdout3", 8) + leaked,
        )
    )
    with pytest.raises(StarContaminationError) as excinfo:
        validate_star_lineage(records)
    assert excinfo.value.detail["scope"] == "earlier_training"
    assert excinfo.value.detail["iteration_index"] == 3
    assert excinfo.value.detail["overlap_count"] == 5


def test_scoring_the_same_holdout_twice_is_caught():
    records = _clean_lineage(2)
    parent = records[-1]["iteration_sha256"]
    records.append(
        _iteration(
            2,
            parent,
            training=_tasks("train2", 20),
            holdout=records[0]["holdout_fingerprints"],
        )
    )
    with pytest.raises(StarContaminationError) as excinfo:
        validate_star_lineage(records)
    assert excinfo.value.detail["scope"] == "reused_holdout"


def test_a_reordered_lineage_is_refused():
    records = _clean_lineage(3)
    with pytest.raises(StarIterationError):
        validate_star_lineage([records[1], records[0], records[2]])


def test_an_edited_iteration_breaks_its_digest():
    records = _clean_lineage(2)
    tampered = dict(records[1])
    tampered["holdout_score"] = 0.99
    with pytest.raises(StarIterationError):
        validate_star_lineage([records[0], tampered])


# --- gated trace classes ----------------------------------------------------


def test_a_tool_assisted_trace_cannot_train_before_its_gate_passes():
    record = _iteration(
        0,
        GENESIS_PARENT,
        training=_tasks("t", 10),
        holdout=_tasks("h", 6),
        classes=["direct", TOOL_ASSISTED],
    )
    with pytest.raises(StarIterationError) as excinfo:
        validate_star_lineage([record])
    assert "trace_class_not_admitted" in str(excinfo.value)


def test_a_passed_gate_admits_the_class_from_that_iteration_on():
    gate = trace_gate(
        trace_class=TOOL_ASSISTED,
        passed=True,
        evidence_sha256=_fp("tool-gate"),
        gate_description="tool receipts replay against the governed ingress",
    )
    first = _iteration(
        0,
        GENESIS_PARENT,
        training=_tasks("t0", 10),
        holdout=_tasks("h0", 6),
        classes=["direct", TOOL_ASSISTED],
        gates=[gate],
    )
    second = _iteration(
        1,
        first["iteration_sha256"],
        training=_tasks("t1", 10),
        holdout=_tasks("h1", 6),
        classes=["direct", TOOL_ASSISTED],
    )
    assert len(validate_star_lineage([first, second])) == 2


def test_a_failing_gate_withdraws_a_previously_admitted_class():
    passed = trace_gate(
        trace_class=LATENT,
        passed=True,
        evidence_sha256=_fp("latent-gate"),
        gate_description="latent traces reconstruct their own slot lineage",
    )
    failed = trace_gate(
        trace_class=LATENT,
        passed=False,
        evidence_sha256=_fp("latent-gate-2"),
        gate_description="latent slot lineage no longer reconstructs",
    )
    first = _iteration(
        0,
        GENESIS_PARENT,
        training=_tasks("t0", 10),
        holdout=_tasks("h0", 6),
        classes=["direct", LATENT],
        gates=[passed],
    )
    second = _iteration(
        1,
        first["iteration_sha256"],
        training=_tasks("t1", 10),
        holdout=_tasks("h1", 6),
        classes=["direct", LATENT],
        gates=[failed],
    )
    with pytest.raises(StarIterationError) as excinfo:
        validate_star_lineage([first, second])
    assert "trace_class_not_admitted" in str(excinfo.value)


def test_a_direct_trace_has_no_gate_to_declare():
    with pytest.raises(StarIterationError) as excinfo:
        trace_gate(
            trace_class="direct",
            passed=True,
            evidence_sha256=_fp("x"),
            gate_description="direct",
        )
    assert "is_not_gated" in str(excinfo.value)


# --- the trend is only reported over a validated lineage --------------------


def test_the_trend_reports_the_series_and_its_disjointness():
    trend = lineage_trend(_clean_lineage(4))
    assert trend["iterations"] == 4
    assert trend["first_holdout_score"] == 0.4
    assert trend["last_holdout_score"] == 0.7
    assert trend["monotonic"] is True
    assert trend["distinct_holdout_tasks"] == 48
    assert trend["holdouts_disjoint_from_all_training"] is True


def test_a_trend_cannot_be_taken_over_a_contaminated_lineage():
    records = _clean_lineage(2)
    records.append(
        _iteration(
            2,
            records[-1]["iteration_sha256"],
            training=_tasks("train2", 20),
            holdout=records[1]["training_fingerprints"][:6],
            score=0.99,
        )
    )
    with pytest.raises(StarContaminationError):
        lineage_trend(records)
