"""SPARK-063: the fingerprints come from the tasks that actually ran.

These use real `HeldoutTask`s minted by the real `generate_battery`, and grade
the holdout with the real `grade_response`, so the disjointness the ledger
enforces is disjointness over the identities a flywheel iteration would really
produce.
"""

from __future__ import annotations

import pytest

from core.learning.heldout_battery import BatterySpec, generate_battery
from core.learning.selfplay_flywheel import EVAL_SEED_FLOOR
from core.learning.star_iteration_ledger import (
    GENESIS_PARENT,
    StarContaminationError,
    validate_star_lineage,
)
from core.learning.star_iteration_producer import (
    StarProducerError,
    mint_disjoint_holdout,
    star_iteration_from_flywheel,
    task_fingerprint,
)

_NOW = 1_780_000_000


def _tasks(seed: int, size: int = 12):
    return generate_battery(BatterySpec(seed=seed, size=size))


def _responses(tasks, *, correct: int):
    return {
        task.task_id: (task.answer if index < correct else "definitely wrong")
        for index, task in enumerate(tasks)
    }


def _iteration(
    index: int,
    parent: str,
    *,
    train_seed: int,
    holdout_seed: int,
    correct: int = 6,
    train_size: int = 12,
    holdout_size: int = 12,
    seen: set[str] | None = None,
):
    training = _tasks(train_seed, train_size)
    # Independently seeded batteries collide 15.5% of the time, so a holdout
    # has to be minted under exclusion rather than merely from another seed.
    holdout = mint_disjoint_holdout(
        seed=holdout_seed,
        size=holdout_size,
        excluded_fingerprints=(seen or set()) | {task_fingerprint(t) for t in training},
    )
    return star_iteration_from_flywheel(
        iteration_index=index,
        parent_iteration_sha256=parent,
        generated=len(training) + 20,
        verified=len(training) + 4,
        filter_reasons={"verifier_rejected": 4},
        training_tasks=training,
        training_trace_classes=["direct"],
        holdout_tasks=holdout,
        holdout_responses=_responses(holdout, correct=correct),
        trace_gates=[],
        created_at_unix=_NOW + index,
    )


# --- fingerprints are derived from real tasks -------------------------------


def test_a_fingerprint_is_the_full_digest_of_the_canonical_prompt():
    import hashlib

    from core.learning.heldout_battery import battery_fingerprints

    tasks = _tasks(EVAL_SEED_FLOOR + 1, 4)
    for task in tasks:
        assert task_fingerprint(task) == hashlib.sha256(
            task.prompt.encode()
        ).hexdigest()
    # The battery's own 16-hex fingerprints are prefixes of these, so the two
    # agree on identity without the ledger inheriting the truncation.
    truncated = battery_fingerprints(tasks)
    assert {task_fingerprint(task)[:16] for task in tasks} == truncated


def test_two_different_batteries_do_not_share_fingerprints():
    left = {task_fingerprint(task) for task in _tasks(11, 16)}
    right = {task_fingerprint(task) for task in _tasks(12, 16)}
    assert not left & right


def test_a_task_without_a_prompt_is_refused():
    class _Bare:
        prompt = ""

    with pytest.raises(StarProducerError):
        task_fingerprint(_Bare())


def test_the_same_prompt_twice_in_one_split_is_a_sampling_defect():
    tasks = _tasks(21, 4)
    with pytest.raises(StarProducerError) as excinfo:
        star_iteration_from_flywheel(
            iteration_index=0,
            parent_iteration_sha256=GENESIS_PARENT,
            generated=40,
            verified=len(tasks) * 2,
            filter_reasons={},
            training_tasks=[*tasks, *tasks],
            training_trace_classes=["direct"],
            holdout_tasks=_tasks(EVAL_SEED_FLOOR + 3, 4),
            holdout_responses={},
            trace_gates=[],
            created_at_unix=_NOW,
        )
    assert "duplicate_task" in str(excinfo.value)


# --- the holdout score is computed, never supplied --------------------------


def test_the_holdout_score_is_graded_from_the_responses():
    record = _iteration(
        0, GENESIS_PARENT, train_seed=5, holdout_seed=EVAL_SEED_FLOOR + 5, correct=9
    )
    assert record["holdout_score"] == 0.75
    assert record["holdout_size"] == 12


def test_a_perfect_and_a_failed_run_score_differently():
    perfect = _iteration(
        0, GENESIS_PARENT, train_seed=5, holdout_seed=EVAL_SEED_FLOOR + 5, correct=12
    )
    failed = _iteration(
        0, GENESIS_PARENT, train_seed=5, holdout_seed=EVAL_SEED_FLOOR + 5, correct=0
    )
    assert perfect["holdout_score"] == 1.0
    assert failed["holdout_score"] == 0.0


def test_an_ungraded_holdout_item_is_refused_rather_than_scored():
    training = _tasks(7, 8)
    holdout = _tasks(EVAL_SEED_FLOOR + 7, 8)
    responses = _responses(holdout, correct=8)
    responses.pop(holdout[0].task_id)
    with pytest.raises(StarProducerError) as excinfo:
        star_iteration_from_flywheel(
            iteration_index=0,
            parent_iteration_sha256=GENESIS_PARENT,
            generated=40,
            verified=len(training) + 2,
            filter_reasons={"verifier_rejected": 2},
            training_tasks=training,
            training_trace_classes=["direct"],
            holdout_tasks=holdout,
            holdout_responses=responses,
            trace_gates=[],
            created_at_unix=_NOW,
        )
    assert "holdout_response_missing" in str(excinfo.value)


# --- the flywheel's seed floor is checked, not trusted -----------------------


def test_a_training_task_minted_from_the_eval_seed_space_is_refused():
    class _Seeded:
        def __init__(self, task, seed):
            self.prompt = task.prompt
            self.answer = task.answer
            self.answer_kind = task.answer_kind
            self.task_id = task.task_id
            self.domain = task.domain
            self.seed = seed

    training = [_Seeded(task, EVAL_SEED_FLOOR + 1) for task in _tasks(9, 6)]
    holdout = _tasks(EVAL_SEED_FLOOR + 9, 6)
    with pytest.raises(StarProducerError) as excinfo:
        star_iteration_from_flywheel(
            iteration_index=0,
            parent_iteration_sha256=GENESIS_PARENT,
            generated=40,
            verified=len(training) + 2,
            filter_reasons={"verifier_rejected": 2},
            training_tasks=training,
            training_trace_classes=["direct"],
            holdout_tasks=holdout,
            holdout_responses=_responses(holdout, correct=3),
            trace_gates=[],
            created_at_unix=_NOW,
        )
    assert "eval_seed_space" in str(excinfo.value)


def test_a_holdout_task_minted_from_the_practice_seed_space_is_refused():
    class _Seeded:
        def __init__(self, task, seed):
            self.prompt = task.prompt
            self.answer = task.answer
            self.answer_kind = task.answer_kind
            self.task_id = task.task_id
            self.domain = task.domain
            self.seed = seed

    training = _tasks(9, 6)
    holdout = [_Seeded(task, 42) for task in _tasks(EVAL_SEED_FLOOR + 9, 6)]
    with pytest.raises(StarProducerError) as excinfo:
        star_iteration_from_flywheel(
            iteration_index=0,
            parent_iteration_sha256=GENESIS_PARENT,
            generated=40,
            verified=len(training) + 2,
            filter_reasons={"verifier_rejected": 2},
            training_tasks=training,
            training_trace_classes=["direct"],
            holdout_tasks=holdout,
            holdout_responses={task.task_id: task.answer for task in holdout},
            trace_gates=[],
            created_at_unix=_NOW,
        )
    assert "practice_seed_space" in str(excinfo.value)


# --- the leak the whole ledger exists to catch, from real batteries ---------


def test_a_clean_multi_iteration_run_validates():
    records = []
    parent = GENESIS_PARENT
    seen: set[str] = set()
    for index in range(4):
        record = _iteration(
            index,
            parent,
            train_seed=100 + index,
            holdout_seed=EVAL_SEED_FLOOR + 100 + index,
            correct=6 + index,
            seen=seen,
        )
        seen |= set(record["training_fingerprints"]) | set(
            record["holdout_fingerprints"]
        )
        records.append(record)
        parent = record["iteration_sha256"]
    replayed = validate_star_lineage(records)
    scores = [row["holdout_score"] for row in replayed]
    assert scores == pytest.approx([0.5, 7 / 12, 8 / 12, 0.75], abs=1e-9)
    assert scores == sorted(scores)


def test_reusing_an_earlier_training_battery_as_a_holdout_is_caught():
    # The exact leak: iteration 0 trains on seed 200; iteration 2 samples the
    # same battery as its "fresh" holdout.
    records = []
    parent = GENESIS_PARENT
    seen: set[str] = set()
    for index in range(2):
        record = _iteration(
            index,
            parent,
            train_seed=200 + index,
            holdout_seed=EVAL_SEED_FLOOR + 200 + index,
            seen=seen,
        )
        seen |= set(record["training_fingerprints"]) | set(
            record["holdout_fingerprints"]
        )
        records.append(record)
        parent = record["iteration_sha256"]

    leaked = _tasks(200, 12)
    records.append(
        star_iteration_from_flywheel(
            iteration_index=2,
            parent_iteration_sha256=parent,
            generated=40,
            verified=14,
            filter_reasons={"verifier_rejected": 2},
            training_tasks=_tasks(202, 12),
            training_trace_classes=["direct"],
            holdout_tasks=leaked,
            holdout_responses=_responses(leaked, correct=12),
            trace_gates=[],
            created_at_unix=_NOW + 2,
        )
    )
    with pytest.raises(StarContaminationError) as excinfo:
        validate_star_lineage(records)
    assert excinfo.value.detail["scope"] == "earlier_training"
    assert excinfo.value.detail["iteration_index"] == 2
    assert excinfo.value.detail["overlap_count"] == 12


def test_a_holdout_reused_between_iterations_is_caught():
    records = []
    parent = GENESIS_PARENT
    trainings = [_tasks(300 + index, 12) for index in range(2)]
    # Disjoint from BOTH trainings, so the only defect left is the reuse.
    shared_holdout = mint_disjoint_holdout(
        seed=EVAL_SEED_FLOOR + 300,
        size=12,
        excluded_fingerprints={
            task_fingerprint(task) for batch in trainings for task in batch
        },
    )
    for index in range(2):
        training = trainings[index]
        record = star_iteration_from_flywheel(
            iteration_index=index,
            parent_iteration_sha256=parent,
            generated=40,
            verified=14,
            filter_reasons={"verifier_rejected": 2},
            training_tasks=training,
            training_trace_classes=["direct"],
            # Both iterations score the SAME holdout battery.
            holdout_tasks=shared_holdout,
            holdout_responses=_responses(shared_holdout, correct=6),
            trace_gates=[],
            created_at_unix=_NOW + index,
        )
        records.append(record)
        parent = record["iteration_sha256"]
    with pytest.raises(StarContaminationError) as excinfo:
        validate_star_lineage(records)
    assert excinfo.value.detail["scope"] == "reused_holdout"


def test_a_within_iteration_leak_is_caught_at_construction():
    shared = _tasks(EVAL_SEED_FLOOR + 400, 8)
    with pytest.raises(StarContaminationError) as excinfo:
        star_iteration_from_flywheel(
            iteration_index=0,
            parent_iteration_sha256=GENESIS_PARENT,
            generated=40,
            verified=10,
            filter_reasons={"verifier_rejected": 2},
            training_tasks=shared,
            training_trace_classes=["direct"],
            holdout_tasks=shared,
            holdout_responses=_responses(shared, correct=8),
            trace_gates=[],
            created_at_unix=_NOW,
        )
    assert excinfo.value.detail["scope"] == "same_iteration"


# --- the defect this producer had to work around ----------------------------


def test_independently_seeded_batteries_really_do_collide():
    """The seed floor separates seeds, not content.

    `generate_battery` deduplicates within one battery and has no cross-battery
    exclusion, and the template space is small. This measures how often two
    independently seeded batteries share at least one prompt -- it is not rare,
    and it is exactly the leak SPARK-063's ledger exists to catch. CP385 hit
    the same defect in the recurrence curriculum and repaired it there by
    constructing the holdout under an exclusion set.

    If this ever reads 0%, the underlying generator gained a much larger task
    space or cross-battery exclusion, and `mint_disjoint_holdout`'s reason for
    existing should be re-checked rather than assumed.
    """
    import itertools

    def fingerprints(seed):
        return {task_fingerprint(task) for task in _tasks(seed, 12)}

    cache = {seed: fingerprints(seed) for seed in range(3, 40)}
    pairs = list(itertools.combinations(cache, 2))
    colliding = sum(1 for left, right in pairs if cache[left] & cache[right])
    assert colliding > 0, (
        "independently seeded batteries no longer collide; re-check whether "
        "mint_disjoint_holdout is still needed"
    )
    assert colliding / len(pairs) > 0.05


def test_a_minted_holdout_is_disjoint_from_everything_excluded():
    training = _tasks(500, 24)
    excluded = {task_fingerprint(task) for task in training}
    holdout = mint_disjoint_holdout(
        seed=EVAL_SEED_FLOOR + 500, size=16, excluded_fingerprints=excluded
    )
    assert len(holdout) == 16
    assert not {task_fingerprint(task) for task in holdout} & excluded


def test_minting_is_deterministic_for_the_same_exclusion():
    excluded = {task_fingerprint(task) for task in _tasks(501, 12)}
    first = mint_disjoint_holdout(
        seed=EVAL_SEED_FLOOR + 501, size=10, excluded_fingerprints=excluded
    )
    second = mint_disjoint_holdout(
        seed=EVAL_SEED_FLOOR + 501, size=10, excluded_fingerprints=excluded
    )
    assert [task_fingerprint(t) for t in first] == [
        task_fingerprint(t) for t in second
    ]


def test_minting_refuses_rather_than_returning_a_short_battery():
    # Exclude exactly what the minter can reach inside its attempt budget:
    # four attempts, each drawing size*2 from seed+attempt.
    size, attempts = 40, 4
    reachable = set()
    for attempt in range(attempts):
        reachable |= {
            task_fingerprint(task)
            for task in _tasks(EVAL_SEED_FLOOR + 600 + attempt, size * 2)
        }
    with pytest.raises(StarProducerError) as excinfo:
        mint_disjoint_holdout(
            seed=EVAL_SEED_FLOOR + 600,
            size=size,
            excluded_fingerprints=reachable,
            max_attempts=attempts,
        )
    assert "exhausted_disjoint_tasks" in str(excinfo.value)


def test_a_minted_holdout_survives_the_ledgers_own_contamination_check():
    training = _tasks(700, 12)
    holdout = mint_disjoint_holdout(
        seed=EVAL_SEED_FLOOR + 700,
        size=12,
        excluded_fingerprints={task_fingerprint(t) for t in training},
    )
    record = star_iteration_from_flywheel(
        iteration_index=0,
        parent_iteration_sha256=GENESIS_PARENT,
        generated=40,
        verified=16,
        filter_reasons={"verifier_rejected": 4},
        training_tasks=training,
        training_trace_classes=["direct"],
        holdout_tasks=holdout,
        holdout_responses=_responses(holdout, correct=7),
        trace_gates=[],
        created_at_unix=_NOW,
    )
    assert validate_star_lineage([record])[0]["holdout_score"] == pytest.approx(
        7 / 12, abs=1e-9
    )
