"""core/evaluation/matched_budget.py — an unmatched comparison is not a comparison.

Operationally: this measures whether the arms of an A/B were allowed to spend
the same resources, and refuses to let a verdict be computed when they were not.

The defect it exists to prevent has a committed instance. `artifacts/current/
agi_live/` compared baselines running at DNU_BASELINE_MAX_TOKENS=160 against a
`full_aura` with a 240s budget, effectively unbounded tokens and a
deterministic symbolic solver, on coding and planning tasks that cannot be
answered inside 160 tokens at all. The baselines ran out of tokens before
emitting an <answer> tag and were scored `no_answer`. Three structurally
different baselines returned an identical 0.1667, which is the signature of a
shared handicap rather than three measurements agreeing — and the harness
reported a clean 100%-versus-16.67% result, because nothing in it had any
concept of what each arm had been allowed to do.

The audit that caught it was written by a human who got suspicious. That is not
a control.

Two mechanisms here, and both are refusals rather than warnings:

1.  **Budget parity.** Every arm declares its budget. Differences on
    OUTCOME-DETERMINING dimensions (tokens, wall clock, tool access, solver
    access, model, retries) make the comparison void. Not "flagged" — void. An
    asterisk is what the retracted bundle had.

2.  **Honest denominators.** Every attempt is counted, including the ones that
    are tempting not to count: crashes, timeouts, retries, fallbacks to another
    lane, and human intervention. A success rate whose denominator excludes the
    runs that went badly is not a success rate. `AttemptLedger` makes the
    denominator the number of attempts, and reports separately how many
    completed without help.

A deliberate asymmetry: a difference that HANDICAPS the treatment is still a
parity violation. It is tempting to allow "we gave the baseline more" as
conservative, and it is still an uncontrolled variable — it just flatters a
different conclusion.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

#: Outcome states an attempt can end in. `success` and `failure` are the only
#: two that mean the system produced an answer that was graded. The rest exist
#: because they were previously invisible: a crashed run that is silently
#: retried until it works reports as a success, and a fallback to a simpler
#: lane reports as the architecture succeeding when the architecture is exactly
#: what did not run.
AttemptOutcome = Literal[
    "success",
    "failure",
    "crash",
    "timeout",
    "refused",
    "no_answer",
]

#: Budget dimensions that change what an arm can achieve. A difference on any
#: of these voids the comparison.
OUTCOME_DETERMINING = (
    "model_id",
    "max_output_tokens",
    "max_wall_clock_s",
    "max_retries",
    "tools",
    "solver_available",
    "memory_available",
)


@dataclass(frozen=True)
class ConditionBudget:
    """What one arm of a comparison was allowed to spend.

    ``None`` means unbounded, and unbounded is a value like any other: an arm
    with ``max_output_tokens=None`` does not match an arm with 160, which is
    the entire point.
    """

    condition: str
    model_id: str
    max_output_tokens: int | None = None
    max_wall_clock_s: float | None = None
    max_retries: int = 0
    tools: frozenset[str] = frozenset()
    solver_available: bool = False
    memory_available: bool = False
    #: Dimensions this comparison deliberately varies — the independent
    #: variable. An ablation of memory MUST differ on `memory_available`, so
    #: parity checking would otherwise make ablations impossible. Naming it
    #: here is the declaration that it is the thing under test.
    varied: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        unknown = set(self.varied) - set(OUTCOME_DETERMINING)
        if unknown:
            raise ValueError(
                f"cannot vary unknown budget dimension(s): {sorted(unknown)}; "
                f"known dimensions are {list(OUTCOME_DETERMINING)}"
            )
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive or None (unbounded)")
        if self.max_wall_clock_s is not None and self.max_wall_clock_s <= 0:
            raise ValueError("max_wall_clock_s must be positive or None (unbounded)")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")

    def dimension(self, name: str) -> Any:
        value = getattr(self, name)
        return sorted(value) if isinstance(value, (frozenset, set)) else value

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            **{name: self.dimension(name) for name in OUTCOME_DETERMINING},
            "varied": sorted(self.varied),
        }


@dataclass(frozen=True)
class ParityViolation:
    dimension: str
    values: dict[str, Any]

    def describe(self) -> str:
        rendered = ", ".join(f"{arm}={value!r}" for arm, value in sorted(self.values.items()))
        return f"{self.dimension}: {rendered}"


@dataclass(frozen=True)
class BudgetParityReport:
    conditions: tuple[str, ...]
    violations: tuple[ParityViolation, ...]
    declared_varied: tuple[str, ...]

    @property
    def matched(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "aura.budget_parity.v1",
            "conditions": list(self.conditions),
            "declared_varied": list(self.declared_varied),
            "violations": [
                {"dimension": v.dimension, "values": v.values} for v in self.violations
            ],
            "matched": self.matched,
        }

    def refusal_reason(self) -> str:
        if self.matched:
            return ""
        return (
            "comparison void — the arms were not allowed the same resources: "
            + "; ".join(v.describe() for v in self.violations)
            + ". Declare the dimension in `varied` if it is the independent "
            "variable, or equalise it. An unmatched comparison does not become "
            "valid by being reported with a caveat."
        )


class UnmatchedBudgetsError(ValueError):
    """Raised instead of returning a verdict the arms did not earn."""

    def __init__(self, report: BudgetParityReport):
        super().__init__(report.refusal_reason())
        self.report = report


def check_budget_parity(budgets: Sequence[ConditionBudget]) -> BudgetParityReport:
    """Compare declared budgets across arms. Differences are violations."""
    if len(budgets) < 2:
        raise ValueError("budget parity needs at least two conditions to compare")
    names = [budget.condition for budget in budgets]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate condition names: {names}")

    # A dimension is exempt only when EVERY arm agrees it is the variable under
    # test. One arm quietly declaring it would let any handicap through.
    declared = {
        dimension
        for dimension in OUTCOME_DETERMINING
        if all(dimension in budget.varied for budget in budgets)
    }

    violations: list[ParityViolation] = []
    for dimension in OUTCOME_DETERMINING:
        if dimension in declared:
            continue
        values = {budget.condition: budget.dimension(dimension) for budget in budgets}
        if len({repr(value) for value in values.values()}) > 1:
            violations.append(ParityViolation(dimension=dimension, values=values))

    return BudgetParityReport(
        conditions=tuple(names),
        violations=tuple(violations),
        declared_varied=tuple(sorted(declared)),
    )


def require_budget_parity(budgets: Sequence[ConditionBudget]) -> BudgetParityReport:
    """Return the report, or raise rather than let a void comparison proceed."""
    report = check_budget_parity(budgets)
    if not report.matched:
        raise UnmatchedBudgetsError(report)
    return report


@dataclass
class Attempt:
    """One attempt at one task under one condition. Counted whatever happened."""

    task_id: str
    condition: str
    outcome: AttemptOutcome
    score: float = 0.0
    retries: int = 0
    human_intervention: bool = False
    #: Which lane actually answered. A run that fell back to a simpler model or
    #: a static response did not exercise the architecture, and reporting it as
    #: the architecture succeeding is the single easiest way to fake this
    #: measurement.
    lane: str = ""
    fell_back: bool = False
    detail: dict[str, Any] = field(default_factory=dict)

    def counts_as_clean_success(self) -> bool:
        return (
            self.outcome == "success"
            and not self.fell_back
            and not self.human_intervention
            and self.retries == 0
        )


@dataclass
class AttemptLedger:
    """Every attempt, with a denominator nobody can shrink.

    The temptation this removes: reporting `successes / graded_attempts` and
    quietly dropping crashes, timeouts and runs that needed a human. That
    number can approach 1.0 for a system that almost never works.
    """

    attempts: list[Attempt] = field(default_factory=list)

    def record(self, attempt: Attempt) -> None:
        self.attempts.append(attempt)

    def for_condition(self, condition: str) -> list[Attempt]:
        return [a for a in self.attempts if a.condition == condition]

    def summary(self, condition: str) -> dict[str, Any]:
        attempts = self.for_condition(condition)
        total = len(attempts)
        if not total:
            return {"condition": condition, "attempts": 0}
        outcomes: dict[str, int] = {}
        for attempt in attempts:
            outcomes[attempt.outcome] = outcomes.get(attempt.outcome, 0) + 1
        successes = outcomes.get("success", 0)
        clean = sum(1 for a in attempts if a.counts_as_clean_success())
        return {
            "condition": condition,
            "attempts": total,
            "outcomes": outcomes,
            # Denominator is ATTEMPTS, always.
            "success_rate": successes / total,
            # And separately: how often it worked without help of any kind.
            "clean_success_rate": clean / total,
            # The mean of the graded score, which is NOT success_rate whenever
            # the grade is continuous. A directional or partial-credit metric
            # scored 0.0-1.0 has a success_rate of 1.000 across every arm —
            # "the attempt completed" — while the thing under test lives
            # entirely in the score. Reporting the first as if it were the
            # second makes every arm look identical and hides the effect.
            "mean_score": sum(a.score for a in attempts) / total,
            "fell_back": sum(1 for a in attempts if a.fell_back),
            "needed_human": sum(1 for a in attempts if a.human_intervention),
            "total_retries": sum(a.retries for a in attempts),
            "lanes": sorted({a.lane for a in attempts if a.lane}),
        }

    def to_dict(self) -> dict[str, Any]:
        conditions = sorted({a.condition for a in self.attempts})
        return {
            "schema": "aura.attempt_ledger.v1",
            "total_attempts": len(self.attempts),
            "conditions": {name: self.summary(name) for name in conditions},
        }


def compare(
    budgets: Sequence[ConditionBudget],
    ledger: AttemptLedger,
) -> dict[str, Any]:
    """A comparison report, or a refusal. Never a number the arms did not earn.

    Refusal is the product here. A caller that wants the number anyway has to
    equalise the budgets or declare what it is varying, which is the same thing
    as designing the experiment.
    """
    report = check_budget_parity(budgets)
    payload: dict[str, Any] = {
        "schema": "aura.matched_comparison.v1",
        "budget_parity": report.to_dict(),
        "attempts": ledger.to_dict(),
    }
    if not report.matched:
        payload["verdict"] = "void"
        payload["reason"] = report.refusal_reason()
        return payload

    summaries = {b.condition: ledger.summary(b.condition) for b in budgets}
    missing = [name for name, s in summaries.items() if not s.get("attempts")]
    if missing:
        payload["verdict"] = "void"
        payload["reason"] = (
            f"no attempts recorded for {missing}; a condition with no attempts "
            "is not a condition that was measured"
        )
        return payload

    payload["verdict"] = "computed"
    payload["success_rate"] = {n: s["success_rate"] for n, s in summaries.items()}
    payload["clean_success_rate"] = {n: s["clean_success_rate"] for n, s in summaries.items()}
    return payload


def equalise(budgets: Iterable[ConditionBudget]) -> list[ConditionBudget]:
    """Tighten every arm to the most restrictive value on each shared dimension.

    A convenience for building a fair comparison rather than diagnosing an
    unfair one: whatever the stingiest arm was given, everyone gets. Dimensions
    declared as `varied` are left alone, because those are the experiment.
    """
    budgets = list(budgets)
    if not budgets:
        return []
    varied = {d for b in budgets for d in b.varied}
    tightest: dict[str, Any] = {}
    for dimension in OUTCOME_DETERMINING:
        if dimension in varied:
            continue
        values = [getattr(b, dimension) for b in budgets]
        if dimension in {"max_output_tokens", "max_wall_clock_s"}:
            bounded = [v for v in values if v is not None]
            tightest[dimension] = min(bounded) if bounded else None
        elif dimension == "max_retries":
            tightest[dimension] = min(values)
        elif dimension in {"solver_available", "memory_available"}:
            tightest[dimension] = all(values)
        elif dimension == "tools":
            tightest[dimension] = frozenset.intersection(*[frozenset(v) for v in values])
        # model_id is not equalisable: a comparison across models is a
        # different experiment, and silently picking one would hide that.
    return [replace(b, **tightest) for b in budgets]


def paired_separation(
    ledger: AttemptLedger,
    treatment: str,
    control: str,
    *,
    iterations: int = 5000,
    seed: int = 20260806,
) -> dict[str, Any]:
    """Paired bootstrap over per-task scores. Says when a delta is unresolvable.

    Paired because both arms answer the SAME tasks, so the per-task difference
    removes task difficulty as a source of variance — the largest one in a
    small battery, and the reason an unpaired comparison over eight items says
    almost nothing.

    A confidence interval spanning zero is reported as `unresolved`, never as a
    result. This lives here rather than in one ablation tool because the first
    version of that tool published a −0.125 delta from eight tasks, where
    0.125 is the smallest non-zero delta eight tasks can express: the number
    was one task's coin flip and it read like a finding. Any comparison of two
    arms over shared tasks needs this guard, so it belongs beside the ledger
    that holds the scores rather than in whichever tool needed it first.
    """
    import random

    treatment_scores = {a.task_id: a.score for a in ledger.for_condition(treatment)}
    control_scores = {a.task_id: a.score for a in ledger.for_condition(control)}
    shared = sorted(set(treatment_scores) & set(control_scores))
    diffs = [treatment_scores[t] - control_scores[t] for t in shared]
    if not diffs:
        return {"resolved": False, "reason": "no shared tasks between the arms"}

    observed = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        means.append(sum(sample) / len(sample))
    means.sort()
    low = means[int(0.025 * len(means))]
    high = means[min(len(means) - 1, int(0.975 * len(means)))]
    resolved = low > 0.0 or high < 0.0

    return {
        "n_tasks": len(diffs),
        "observed_delta": round(observed, 4),
        "ci95": [round(low, 4), round(high, 4)],
        "resolved": resolved,
        "smallest_resolvable_delta": round(1.0 / len(diffs), 4),
        "verdict": (
            ("treatment_better" if observed > 0 else "treatment_worse")
            if resolved
            else "unresolved"
        ),
        "reason": (
            ""
            if resolved
            else f"the 95% interval [{low:.4f}, {high:.4f}] spans zero; "
            f"with {len(diffs)} tasks the smallest non-zero delta expressible is "
            f"{1.0 / len(diffs):.4f}, so this run cannot tell an effect from one "
            "task changing its mind. Add tasks."
        ),
    }


__all__ = [
    "Attempt",
    "AttemptLedger",
    "AttemptOutcome",
    "BudgetParityReport",
    "ConditionBudget",
    "OUTCOME_DETERMINING",
    "ParityViolation",
    "UnmatchedBudgetsError",
    "check_budget_parity",
    "compare",
    "paired_separation",
    "equalise",
    "require_budget_parity",
]
