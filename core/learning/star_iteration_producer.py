"""Turn a real flywheel iteration into a contamination-checked ledger record.

`star_iteration_ledger` catches the leak no single iteration can see. It can
only catch it if the fingerprints it is given are the fingerprints of the tasks
that were actually trained on and actually scored — which means the fingerprints
have to be derived from the task objects, not assembled by whoever is writing
the record.

So this producer takes the real objects: `HeldoutTask`s for both splits, the
graded responses, and the verified/filtered accounting from the generation
stage. It derives everything else.

Two details that matter more than they look:

- **Fingerprints are full-width here.** `heldout_battery.battery_fingerprints`
  truncates to 16 hex characters, which is fine for its own leak check over a
  few hundred prompts and is not fine for a cross-iteration disjointness claim
  accumulated over a long campaign — a 64-bit space starts colliding by
  birthday at a few billion, but truncation also means a collision reads as
  *contamination that is not there*, which would stop a clean campaign. The
  ledger's fingerprints are the full SHA-256 of the same canonical prompt, so
  the two agree on identity without inheriting the truncation.

- **The flywheel's seed floor is checked, not trusted.** `selfplay_flywheel`
  keeps practice seeds strictly below `EVAL_SEED_FLOOR` and mints eval
  batteries at or above it. That structural separation is real and it is also
  exactly the thing that quietly breaks when someone reuses a seed, so the
  producer verifies the split it was handed rather than assuming the convention
  held.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Final, Never

from core.learning.star_iteration_ledger import (
    GENESIS_PARENT,
    star_iteration,
)

STAR_PRODUCER_SCHEMA: Final = "aura.rlc.star_iteration_producer.v1"


class StarProducerError(ValueError):
    """A flywheel iteration cannot be recorded honestly."""


def _fail(code: str) -> Never:
    raise StarProducerError(str(code or "star_producer_invalid"))


def task_fingerprint(task: Any) -> str:
    """Full-width identity of one task, from its canonical prompt.

    Same input as `heldout_battery.battery_fingerprints`, without the 16-hex
    truncation: a truncated fingerprint that collided would report
    contamination that is not there and stop a clean campaign.
    """

    prompt = getattr(task, "prompt", None)
    if not isinstance(prompt, str) or not prompt.strip():
        _fail("star_producer_task_prompt_missing")
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _fingerprints(tasks: Any, code: str) -> list[str]:
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)):
        _fail(code)
    if not tasks:
        _fail(code)
    seen: list[str] = []
    known: set[str] = set()
    for task in tasks:
        fingerprint = task_fingerprint(task)
        if fingerprint in known:
            # The same prompt twice in one split is a sampling defect, and
            # deduplicating it silently would make the counts disagree with
            # the tasks.
            _fail("star_producer_duplicate_task")
        known.add(fingerprint)
        seen.append(fingerprint)
    return seen


def _seeds_respect_the_floor(tasks: Sequence[Any], *, training: bool) -> None:
    from core.learning.selfplay_flywheel import EVAL_SEED_FLOOR

    for task in tasks:
        seed = getattr(task, "seed", None)
        if seed is None:
            continue
        if type(seed) is not int:
            _fail("star_producer_task_seed_invalid")
        if training and seed >= EVAL_SEED_FLOOR:
            _fail("star_producer_training_task_from_eval_seed_space")
        if not training and seed < EVAL_SEED_FLOOR:
            _fail("star_producer_holdout_task_from_practice_seed_space")


def star_iteration_from_flywheel(
    *,
    iteration_index: int,
    parent_iteration_sha256: str,
    generated: int,
    verified: int,
    filter_reasons: Mapping[str, int],
    training_tasks: Sequence[Any],
    training_trace_classes: Sequence[str],
    holdout_tasks: Sequence[Any],
    holdout_responses: Mapping[str, str],
    trace_gates: Sequence[Mapping[str, Any]],
    created_at_unix: int,
) -> dict[str, Any]:
    """Record one flywheel iteration, grading the holdout here.

    The holdout score is *computed* from the responses against the sealed
    answers. A caller cannot supply a score, because a supplied score is the
    one number in this record that nothing else constrains.
    """

    from core.learning.heldout_battery import grade_response

    training = _fingerprints(training_tasks, "star_producer_training_tasks_invalid")
    holdout = _fingerprints(holdout_tasks, "star_producer_holdout_tasks_invalid")
    _seeds_respect_the_floor(training_tasks, training=True)
    _seeds_respect_the_floor(holdout_tasks, training=False)

    if not isinstance(holdout_responses, Mapping):
        _fail("star_producer_holdout_responses_invalid")
    correct = 0
    for task in holdout_tasks:
        task_id = getattr(task, "task_id", None)
        if not isinstance(task_id, str) or not task_id:
            _fail("star_producer_holdout_task_id_missing")
        if task_id not in holdout_responses:
            # An ungraded holdout item is not a wrong answer and it is not a
            # right one. Scoring it either way would be inventing evidence.
            _fail("star_producer_holdout_response_missing")
        correct += int(bool(grade_response(task, holdout_responses[task_id])))

    return star_iteration(
        iteration_index=iteration_index,
        parent_iteration_sha256=parent_iteration_sha256,
        generated=generated,
        verified=verified,
        filtered=verified,
        filter_reasons=filter_reasons,
        training_fingerprints=training,
        training_trace_classes=training_trace_classes,
        holdout_fingerprints=holdout,
        holdout_score=round(correct / len(holdout), 9),
        trace_gates=trace_gates,
        created_at_unix=created_at_unix,
    )


def mint_disjoint_holdout(
    *,
    seed: int,
    size: int,
    excluded_fingerprints: Sequence[str] | set[str],
    max_attempts: int = 64,
) -> list[Any]:
    """Draw a holdout battery that is disjoint from everything already seen.

    **This exists because the seed floor is not enough.** `generate_battery`
    deduplicates *within* one battery but has no cross-battery exclusion, and
    the template space is small: measured over seeds 3-39, **15.5% of seed
    pairs share at least one prompt**. So a practice battery below
    `EVAL_SEED_FLOOR` and an eval battery above it can contain the same task,
    and the flywheel's seed convention separates seeds rather than content.

    That is the same defect CP385 hit — independently generated train and
    holdout batteries sharing a prompt — repaired there for the recurrence
    curriculum by constructing the holdout under an exclusion set. This is the
    same repair for the held-out battery path.

    Draws are deterministic per attempt, so a given (seed, size, exclusion)
    always yields the same battery. If enough disjoint tasks cannot be found
    the call fails rather than returning a short or contaminated battery.
    """

    from core.learning.heldout_battery import BatterySpec, generate_battery

    if type(seed) is not int or seed < 0:
        _fail("star_producer_holdout_seed_invalid")
    if type(size) is not int or size <= 0:
        _fail("star_producer_holdout_size_invalid")
    if type(max_attempts) is not int or max_attempts < 1:
        _fail("star_producer_holdout_attempts_invalid")

    excluded = set(excluded_fingerprints)
    chosen: list[Any] = []
    taken: set[str] = set()
    for attempt in range(max_attempts):
        # Oversample: most draws will survive the filter, and a fixed
        # multiplier keeps the work bounded.
        batch = generate_battery(BatterySpec(seed=seed + attempt, size=size * 2))
        for task in batch:
            fingerprint = task_fingerprint(task)
            if fingerprint in excluded or fingerprint in taken:
                continue
            taken.add(fingerprint)
            chosen.append(task)
            if len(chosen) == size:
                return chosen
    _fail("star_producer_holdout_exhausted_disjoint_tasks")


def open_star_lineage(**kwargs: Any) -> dict[str, Any]:
    """First iteration of a fresh flywheel run."""

    kwargs.setdefault("iteration_index", 0)
    kwargs.setdefault("parent_iteration_sha256", GENESIS_PARENT)
    return star_iteration_from_flywheel(**kwargs)


__all__ = [
    "STAR_PRODUCER_SCHEMA",
    "StarProducerError",
    "open_star_lineage",
    "star_iteration_from_flywheel",
    "task_fingerprint",
]
