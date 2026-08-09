"""The arms that can refute the commitment ratchet.

A mechanism that only ever runs its own arm is not being tested. This
codebase has paid for that lesson repeatedly — a promotion gate wired to the
policy that removed the floor, four subsystems gated on a verifier nothing
supplied, `0 >= 0` passing as parity — so the falsification apparatus is
built at the same time as the mechanism, not after it succeeds.

Five arms. Each one is a specific way the ratchet could be fooling us:

  VANILLA      no ratchet at all. The floor. If the ratchet does not beat
               this, there is nothing to discuss.

  DEPTH_ONLY   the same number of extra passes, conditioned on NOTHING.
               This is the arm that isolates the claim. If depth alone
               matches the ratchet, then the constraints are decoration and
               we have rediscovered "more compute", which we already know
               does not work here.

  SHUFFLE      the real constraints, permuted across steps — so pass 3 is
               conditioned on what pass 7 committed. Same constraints, same
               count, same context cost, wrong ORDER. The ratchet's whole
               claim is that a commitment narrows the problem for the passes
               that FOLLOW it. If shuffle matches real, the ordering carries
               no information and the mechanism is not the mechanism.
               THIS IS THE ARM THAT KILLS IT.

  RANDOM       constraints drawn from the same vocabulary about the same
               objective, but not derived from evidence. Predicts WORSE than
               vanilla: a wrong commitment is irreversible too. If random
               matches real, the ratchet is a prompt-length effect and the
               content of the constraints is irrelevant.

  ORACLE       constraints derived from the known-correct answer. Not a
               deployable arm — a ceiling. It answers "is the constraint
               CHANNEL capable of carrying a gain at all", so a null result
               in the real arm can be attributed to extraction rather than
               to the whole idea.

The verdict function is deliberately hard to pass and states its own null.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.brain.llm.latent_cortex.commitment_ratchet import (
    CommitmentRatchet,
    Constraint,
    ConstraintKind,
)

ABLATION_SCHEMA = "aura.rlc.commitment_ablation.v1"

#: A gain smaller than this over the vanilla floor is not a gain worth
#: promoting, whatever its p-value: it is inside the noise a decode
#: temperature change produces.
MIN_ABSOLUTE_GAIN = 0.02

#: How much the real arm must beat SHUFFLE by for ordering to be doing the
#: work. Chosen equal to the gain floor: if ordering contributes less than
#: the whole claimed effect, the claim is that a bag of constraints works,
#: which is a different and weaker claim than the one this module makes.
MIN_ORDERING_MARGIN = 0.02


class Arm(StrEnum):
    VANILLA = "vanilla"
    RATCHET = "ratchet"
    DEPTH_ONLY = "depth_only"
    SHUFFLE = "shuffle"
    RANDOM = "random"
    ORACLE = "oracle"


@dataclass(frozen=True)
class ArmResult:
    """One arm's measured score over a task set."""

    arm: Arm
    scored: int
    passed: int
    #: Total constraints committed across the set. Zero here in an arm that
    #: should have committed some is a broken run, not a null result.
    commitments: int = 0
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def rate(self) -> float:
        return self.passed / self.scored if self.scored else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm.value,
            "scored": self.scored,
            "passed": self.passed,
            "rate": round(self.rate, 6),
            "commitments": self.commitments,
            "notes": dict(self.notes),
        }


def shuffled_constraints(
    constraints: Sequence[Constraint], *, seed: int
) -> list[Constraint]:
    """Same constraints, same count — permuted step assignment.

    The permutation is over the STEP each constraint is attached to, not
    over the constraint list, because attaching them in a different order to
    the same steps is the intervention. A permutation that happens to be the
    identity is re-drawn: an ablation arm that silently equals the treatment
    arm produces a false null.
    """
    if len(constraints) < 2:
        return list(constraints)
    rng = random.Random(seed)
    steps = [constraint.step for constraint in constraints]
    for _ in range(16):
        permuted = list(steps)
        rng.shuffle(permuted)
        if permuted != steps:
            break
    else:
        return list(constraints)
    return [
        Constraint(
            kind=constraint.kind,
            subject=constraint.subject,
            args=constraint.args,
            source=f"shuffled:{constraint.source}",
            step=step,
        )
        for constraint, step in zip(constraints, permuted, strict=True)
    ]


_RANDOM_SUBJECTS = ("number", "list", "name", "boolean", "date")


def random_constraints(
    *, count: int, seed: int, step: int = 0
) -> list[Constraint]:
    """Evidence-free constraints from the same vocabulary.

    Predicts WORSE than vanilla, and that prediction is the point: an
    irreversible wrong commitment should hurt. An arm that predicts "no
    change" cannot distinguish a working mechanism from an inert one.
    """
    rng = random.Random(seed)
    out: list[Constraint] = []
    for index in range(max(0, count)):
        kind = rng.choice(
            [
                ConstraintKind.ANSWER_TYPE,
                ConstraintKind.MUST_MENTION,
                ConstraintKind.EXCLUDES,
                ConstraintKind.CARDINALITY,
            ]
        )
        if kind is ConstraintKind.ANSWER_TYPE:
            constraint = Constraint(kind=kind, subject=rng.choice(_RANDOM_SUBJECTS))
        elif kind is ConstraintKind.CARDINALITY:
            constraint = Constraint(
                kind=kind, subject="items", args=(float(rng.randint(1, 5)),)
            )
        else:
            constraint = Constraint(
                kind=kind, subject=f"token{rng.randint(1000, 9999)}"
            )
        out.append(
            Constraint(
                kind=constraint.kind,
                subject=constraint.subject,
                args=constraint.args,
                source="random",
                step=step + index,
            )
        )
    return out


def oracle_constraints(answer: str, *, step: int = 0) -> list[Constraint]:
    """The ceiling: constraints that are true of the known answer.

    Not deployable — it reads the answer. It exists so a null in the real arm
    can be attributed. If ORACLE also fails to beat vanilla, the constraint
    CHANNEL cannot carry a gain and no extractor will save it; if oracle
    wins big and real does not, extraction is the gap and the architecture
    is sound.
    """
    text = str(answer or "").strip()
    if not text:
        return []
    out = [
        Constraint(
            kind=ConstraintKind.MUST_EQUAL, subject=text[:120],
            source="oracle", step=step,
        )
    ]
    for answer_type in ("number", "boolean", "date", "list"):
        probe = Constraint(kind=ConstraintKind.ANSWER_TYPE, subject=answer_type)
        if probe.check(text) is True:
            out.append(
                Constraint(
                    kind=ConstraintKind.ANSWER_TYPE, subject=answer_type,
                    source="oracle", step=step,
                )
            )
            break
    return out


# ─────────────────────────────────────────────────────────── the verdict


def adjudicate(results: Mapping[str, ArmResult] | Mapping[Arm, ArmResult]) -> dict[str, Any]:
    """Did the ratchet earn its place, or is it decoration?

    Refuses by default. There is no combination of missing arms that returns
    SUPPORTED — an absent comparison is an absent comparison, and this
    codebase's recurring defect is exactly the absence of a check being read
    as a passed check.
    """
    table: dict[Arm, ArmResult] = {}
    for key, value in results.items():
        arm = key if isinstance(key, Arm) else Arm(str(key))
        table[arm] = value

    missing = [
        arm.value
        for arm in (Arm.VANILLA, Arm.RATCHET, Arm.DEPTH_ONLY, Arm.SHUFFLE)
        if arm not in table
    ]
    if missing:
        return _verdict(
            "INCONCLUSIVE",
            f"required arms not run: {missing}",
            table,
            checks={},
        )

    vanilla = table[Arm.VANILLA]
    ratchet = table[Arm.RATCHET]
    depth = table[Arm.DEPTH_ONLY]
    shuffle = table[Arm.SHUFFLE]

    if ratchet.scored == 0 or vanilla.scored == 0:
        return _verdict("INCONCLUSIVE", "an arm scored nothing", table, checks={})
    if ratchet.commitments == 0:
        # The treatment arm did not do the thing. Whatever it scored, it was
        # not this mechanism scoring it.
        return _verdict(
            "INCONCLUSIVE",
            "the ratchet arm committed zero constraints; it ran as depth_only",
            table,
            checks={},
        )

    checks = {
        "beats_vanilla": ratchet.rate - vanilla.rate >= MIN_ABSOLUTE_GAIN,
        "beats_depth_only": ratchet.rate - depth.rate >= MIN_ABSOLUTE_GAIN,
        "beats_shuffle": ratchet.rate - shuffle.rate >= MIN_ORDERING_MARGIN,
    }
    if Arm.RANDOM in table:
        # Not required, but if it was run and random did as well as real,
        # the content of the constraints is not what is working.
        checks["beats_random"] = (
            ratchet.rate - table[Arm.RANDOM].rate >= MIN_ABSOLUTE_GAIN
        )

    if all(checks.values()):
        return _verdict("SUPPORTED", "ratchet beat every control arm", table, checks)

    failed = [name for name, ok in checks.items() if not ok]
    if not checks["beats_shuffle"]:
        return _verdict(
            "REFUTED",
            "shuffling the constraints across steps cost nothing, so the "
            "ordering carries no information and the mechanism is not the "
            "mechanism",
            table,
            checks,
        )
    return _verdict("REFUTED", f"failed: {failed}", table, checks)


def _verdict(
    status: str, reason: str, table: Mapping[Arm, ArmResult], checks: Mapping[str, bool]
) -> dict[str, Any]:
    return {
        "schema": ABLATION_SCHEMA,
        "verdict": status,
        "reason": reason,
        "checks": dict(checks),
        "arms": {arm.value: result.to_dict() for arm, result in table.items()},
        "thresholds": {
            "min_absolute_gain": MIN_ABSOLUTE_GAIN,
            "min_ordering_margin": MIN_ORDERING_MARGIN,
        },
        # Stated up front, so nobody has to reconstruct what would have
        # counted as failure after seeing the numbers.
        "null_hypothesis": (
            "the ratchet's score is explained by extra passes and extra "
            "prompt text, not by the order in which commitments constrain "
            "later passes"
        ),
    }


def run_arm(
    arm: Arm,
    tasks: Sequence[Mapping[str, Any]],
    *,
    solve: Callable[[str, str], str],
    seed: int = 0,
) -> ArmResult:
    """Score one arm over a task set.

    ``solve(objective, conditioning_block) -> answer`` is the caller's model
    call. Every arm goes through the SAME callable with the same budget; the
    only thing that differs is the conditioning block, which is the variable
    under test.
    """
    passed = 0
    scored = 0
    commitments = 0
    for index, task in enumerate(tasks):
        objective = str(task.get("objective") or "")
        expected = str(task.get("answer") or "")
        if not objective or not expected:
            continue
        block, committed = _conditioning_for(arm, task, seed=seed + index)
        commitments += committed
        try:
            answer = solve(objective, block)
        except (RuntimeError, ValueError, TypeError, OSError):
            scored += 1
            continue
        scored += 1
        if _matches(answer, expected):
            passed += 1
    return ArmResult(arm=arm, scored=scored, passed=passed, commitments=commitments)


def _conditioning_for(
    arm: Arm, task: Mapping[str, Any], *, seed: int
) -> tuple[str, int]:
    if arm is Arm.VANILLA:
        return "", 0
    if arm is Arm.DEPTH_ONLY:
        # Extra passes, no commitments. The block is deliberately contentless
        # rather than absent, so prompt length is held roughly constant and
        # the comparison is about information, not tokens.
        return "[continuing to work on this problem]", 0

    constraints = list(task.get("constraints") or ())
    if arm is Arm.SHUFFLE:
        constraints = shuffled_constraints(constraints, seed=seed)
    elif arm is Arm.RANDOM:
        constraints = random_constraints(count=len(constraints) or 2, seed=seed)
    elif arm is Arm.ORACLE:
        constraints = oracle_constraints(str(task.get("answer") or ""))

    ratchet = CommitmentRatchet(task.get("pool") or ())
    committed = 0
    for constraint in sorted(constraints, key=lambda item: item.step):
        if ratchet.commit(constraint).committed:
            committed += 1
    # Explicitly ON: this harness exists to measure the prompt-conditioned
    # arm, including the fact that it loses.
    return ratchet.conditioning_block(include_exclusions=True), committed


def _matches(answer: Any, expected: str) -> bool:
    from core.brain.llm.latent_cortex.commitment_ratchet import _normalize

    got, want = _normalize(answer), _normalize(expected)
    return bool(want) and (got == want or want in got)


__all__ = [
    "ABLATION_SCHEMA",
    "MIN_ABSOLUTE_GAIN",
    "MIN_ORDERING_MARGIN",
    "Arm",
    "ArmResult",
    "adjudicate",
    "oracle_constraints",
    "random_constraints",
    "run_arm",
    "shuffled_constraints",
]
