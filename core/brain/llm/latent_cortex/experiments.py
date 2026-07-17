"""The falsification harness: how we test this without fooling ourselves.

Implements the spec's Experiments 1–5 as runnable, seeded, self-verifying
protocols. Every experiment produces a graded ``Claim``:

    PROVEN      — effect present with non-overlapping Wilson intervals on
                  ≥2 task families and adequate n
    SUPPORTED   — treatment's Wilson lower bound beats control's upper
                  bound on ≥1 family (or a strictly monotone scaling trend)
    CONJECTURE  — insufficient evidence either way (small n, mixed signal)
    REFUTED     — adequate n and treatment failed to beat control

The graders are deliberately conservative; an exciting anecdote grades as
CONJECTURE, never SUPPORTED. Verdicts can be recorded to the Verifier
Foundry so the reliability of this harness itself is tracked like every
other verifier in the system.

Task generators are exact and self-verifying (graph reachability, nested
boolean evaluation, modular-arithmetic chains), each with a controllable
DEPTH knob — the compositional-depth ladder Experiment 2 climbs.

Experiment 6 (frontier comparison) is a protocol, not code that can run
here: it needs blind fresh tasks and an external comparison system under
equal information/tool/compute access. `frontier_comparison_protocol()`
returns the checklist so an operator run can't quietly skip a control.
"""
from __future__ import annotations

import hashlib
import logging
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.brain.verifiers.foundry import wilson_lower_bound, wilson_upper_bound

logger = logging.getLogger("Aura.LatentCortex.Experiments")

PROVEN = "PROVEN"
SUPPORTED = "SUPPORTED"
CONJECTURE = "CONJECTURE"
REFUTED = "REFUTED"

_MIN_N_FOR_VERDICT = 20  # below this, everything is CONJECTURE
_BOOTSTRAP_RESAMPLES = 10_000


# ── Self-verifying task generators ──────────────────────────────────────


@dataclass
class Task:
    prompt: str
    answer: str
    depth: int
    family: str
    seed: int

    def verify(self, text: str) -> bool:
        """Exact-answer check on the FINAL claim in the output.

        The last number/token mentioned wins — chain-of-thought before it is
        fine; hedging two different answers is not."""
        tokens = [t.strip(".,:;!()[]") for t in str(text or "").split()]
        candidates = [t for t in tokens if t and (t == self.answer or t.lstrip("-").isdigit())]
        return bool(candidates) and candidates[-1] == self.answer


def khop_reachability(depth: int, seed: int, n_nodes: int = 12) -> Task:
    """Follow a functional graph for ``depth`` hops; answer the landing node."""
    rng = random.Random(seed * 1_000_003 + depth)
    successor = {i: rng.randrange(n_nodes) for i in range(n_nodes)}
    start = rng.randrange(n_nodes)
    node = start
    for _ in range(depth):
        node = successor[node]
    edges = ", ".join(f"{a}->{b}" for a, b in sorted(successor.items()))
    prompt = (
        f"A directed graph has exactly one outgoing edge per node: {edges}. "
        f"Start at node {start} and follow exactly {depth} edges. "
        "Answer with the final node number only."
    )
    return Task(prompt=prompt, answer=str(node), depth=depth, family="khop", seed=seed)


def nested_boolean(depth: int, seed: int) -> Task:
    """Evaluate a nested and/or/not expression; answer 1 (true) or 0 (false)."""
    rng = random.Random(seed * 2_000_003 + depth)

    def build(d: int) -> tuple[str, bool]:
        if d <= 0:
            v = rng.random() < 0.5
            return ("1" if v else "0"), v
        op = rng.choice(("and", "or", "not"))
        if op == "not":
            s, v = build(d - 1)
            return f"(not {s})", (not v)
        left, lv = build(d - 1)
        right, rv = build(max(0, d - 1 - rng.randrange(2)))
        value = (lv and rv) if op == "and" else (lv or rv)
        return f"({left} {op} {right})", value

    expr, value = build(depth)
    prompt = (
        f"Evaluate this boolean expression where 1=true and 0=false: {expr}. "
        "Answer with a single digit, 1 or 0."
    )
    return Task(prompt=prompt, answer="1" if value else "0", depth=depth, family="boolean", seed=seed)


def modular_chain(depth: int, seed: int, mod: int = 17) -> Task:
    """Apply ``depth`` sequential +/× operations mod m; answer the result."""
    rng = random.Random(seed * 3_000_017 + depth)
    value = rng.randrange(mod)
    steps = [f"start with {value}"]
    for _ in range(depth):
        op, operand = rng.choice(("+", "*")), rng.randrange(1, mod)
        value = (value + operand) % mod if op == "+" else (value * operand) % mod
        steps.append(f"{op} {operand}, then take mod {mod}")
    prompt = (
        "Compute step by step: " + "; ".join(steps) + ". "
        f"All arithmetic is modulo {mod}. Answer with the final number only."
    )
    return Task(prompt=prompt, answer=str(value), depth=depth, family="modular", seed=seed)


TASK_FAMILIES: dict[str, Callable[[int, int], Task]] = {
    "khop": khop_reachability,
    "boolean": nested_boolean,
    "modular": modular_chain,
}


def task_battery(families: list[str], depths: list[int], per_cell: int, seed: int = 0) -> list[Task]:
    tasks: list[Task] = []
    for family in families:
        gen = TASK_FAMILIES[family]
        for depth in depths:
            for i in range(per_cell):
                tasks.append(gen(depth, seed * 7919 + i))
    return tasks


# ── Claims ──────────────────────────────────────────────────────────────


@dataclass
class ArmResult:
    """One experimental arm: n trials, k successes, plus cost accounting."""

    name: str
    n: int = 0
    successes: int = 0
    layer_apps: int = 0

    @property
    def accuracy(self) -> float:
        return self.successes / self.n if self.n else 0.0

    @property
    def lb(self) -> float:
        return wilson_lower_bound(self.successes, self.n)

    @property
    def ub(self) -> float:
        return wilson_upper_bound(self.successes, self.n)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n": self.n,
            "successes": self.successes,
            "accuracy": round(self.accuracy, 4),
            "wilson_lb": round(self.lb, 4),
            "wilson_ub": round(self.ub, 4),
            "layer_apps": self.layer_apps,
        }


@dataclass(frozen=True)
class PairedObservation:
    """One task evaluated by treatment and control under measured compute."""

    task_id: str
    family: str
    treatment_success: bool
    control_success: bool
    treatment_layer_apps: int | None = None
    control_layer_apps: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "treatment_success": self.treatment_success,
            "control_success": self.control_success,
            "treatment_layer_apps": self.treatment_layer_apps,
            "control_layer_apps": self.control_layer_apps,
        }


def _coerce_solver_outcome(value: Any) -> tuple[bool, int | None]:
    if isinstance(value, tuple) and len(value) == 2:
        success, layer_apps = value
        if not isinstance(success, bool):
            raise ValueError("solver success must be boolean")
        if type(layer_apps) is not int or layer_apps < 0:
            raise ValueError("solver layer-app receipt must be a non-negative integer")
        return success, layer_apps
    if isinstance(value, bool):
        return value, None
    raise ValueError("solver must return bool or (bool, non-negative layer_apps)")


def _exact_paired_pvalue_greater(wins: int, losses: int) -> float:
    """Exact one-sided McNemar/binomial p for treatment wins > losses."""
    discordant = wins + losses
    if discordant <= 0:
        return 1.0
    numerator = sum(math.comb(discordant, k) for k in range(wins, discordant + 1))
    return min(1.0, numerator / (2**discordant))


def _paired_bootstrap_interval(
    differences: list[int], *, alpha: float, seed: int = 20260715
) -> tuple[float, float]:
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
        or not 0.0 < alpha <= 0.5
    ):
        raise ValueError("bootstrap alpha must be inside (0, 0.5]")
    if not differences:
        return 0.0, 0.0
    if len(set(differences)) == 1:
        value = float(differences[0])
        return value, value
    import numpy as np

    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty((_BOOTSTRAP_RESAMPLES,), dtype=np.float64)
    for start in range(0, _BOOTSTRAP_RESAMPLES, 500):
        count = min(500, _BOOTSTRAP_RESAMPLES - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, pvalue) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * pvalue))
        adjusted[name] = running
    return adjusted


def grade_paired_treatment_vs_control(
    experiment: str,
    statement: str,
    observations_by_family: dict[str, list[PairedObservation]],
    *,
    alpha: float = 0.05,
    minimum_effect: float = 0.0,
    compute_tolerance: float = 0.05,
    require_compute: bool = True,
) -> Claim:
    """Paired, multiplicity-corrected capability comparison."""
    for name, value in (
        ("alpha", alpha),
        ("minimum_effect", minimum_effect),
        ("compute_tolerance", compute_tolerance),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(
            float(value)
        ):
            raise ValueError(f"{name} must be a finite number")
    if not 0.0 < alpha <= 0.5:
        raise ValueError("alpha must be inside (0, 0.5]")
    if not 0.0 <= minimum_effect < 1.0:
        raise ValueError("minimum_effect must be inside [0, 1)")
    if not 0.0 <= compute_tolerance <= 1.0:
        raise ValueError("compute_tolerance must be inside [0, 1]")
    if type(require_compute) is not bool:
        raise ValueError("require_compute must be boolean")

    family_stats: dict[str, dict[str, Any]] = {}
    raw_pvalues: dict[str, float] = {}
    all_differences: list[int] = []
    invalid_compute: list[str] = []
    underpowered: list[str] = []
    seen_task_ids: set[str] = set()
    family_bound_alpha = alpha / max(1, len(observations_by_family))
    for family, observations in observations_by_family.items():
        if not isinstance(family, str) or not family.strip():
            raise ValueError("paired evidence family names must be non-empty strings")
        if not isinstance(observations, list):
            raise ValueError(f"paired evidence for {family} must be a list")
        for observation in observations:
            if not isinstance(observation, PairedObservation):
                raise ValueError(f"paired evidence for {family} contains an invalid row")
            if not observation.task_id or observation.task_id in seen_task_ids:
                raise ValueError("paired evidence task ids must be non-empty and unique")
            seen_task_ids.add(observation.task_id)
            if observation.family != family:
                raise ValueError(
                    f"paired evidence family mismatch: {observation.family!r} != {family!r}"
                )
            if type(observation.treatment_success) is not bool or type(
                observation.control_success
            ) is not bool:
                raise ValueError("paired evidence outcomes must be boolean")
            for cost in (
                observation.treatment_layer_apps,
                observation.control_layer_apps,
            ):
                if cost is not None and (type(cost) is not int or cost < 0):
                    raise ValueError(
                        "paired evidence compute must be non-negative integers or null"
                    )
        differences = [
            int(obs.treatment_success) - int(obs.control_success) for obs in observations
        ]
        wins = differences.count(1)
        losses = differences.count(-1)
        missing_compute = any(
            obs.treatment_layer_apps is None or obs.control_layer_apps is None
            for obs in observations
        )
        nonpositive_compute = any(
            obs.treatment_layer_apps is not None
            and obs.control_layer_apps is not None
            and (obs.treatment_layer_apps <= 0 or obs.control_layer_apps <= 0)
            for obs in observations
        )
        mismatched = [
            obs.task_id
            for obs in observations
            if obs.treatment_layer_apps is not None
            and obs.control_layer_apps is not None
            and (
                abs(obs.treatment_layer_apps - obs.control_layer_apps)
                / max(1, obs.control_layer_apps)
            )
            > compute_tolerance
        ]
        if require_compute and (missing_compute or nonpositive_compute or mismatched):
            invalid_compute.append(family)
        effect = sum(differences) / len(differences) if differences else 0.0
        ci_low, ci_high = _paired_bootstrap_interval(
            differences,
            alpha=family_bound_alpha,
        )
        pvalue = _exact_paired_pvalue_greater(wins, losses)
        if len(observations) < _MIN_N_FOR_VERDICT:
            underpowered.append(family)
        else:
            raw_pvalues[family] = pvalue
        family_stats[family] = {
            "n": len(observations),
            "treatment_wins": wins,
            "control_wins": losses,
            "ties": len(observations) - wins - losses,
            "paired_effect": round(effect, 6),
            "effect_interval": [round(ci_low, 6), round(ci_high, 6)],
            "effect_bound_alpha": family_bound_alpha,
            "one_sided_exact_p": pvalue,
            "missing_compute": missing_compute,
            "nonpositive_compute": nonpositive_compute,
            "compute_mismatch_task_ids": mismatched,
        }
        all_differences.extend(differences)

    adjusted = _holm_adjust(raw_pvalues)
    positive_families = [
        family
        for family, stats in family_stats.items()
        if family in adjusted
        and adjusted[family] < alpha
        and stats["effect_interval"][0] > minimum_effect
        and family not in invalid_compute
    ]
    regressed_families = [
        family
        for family, stats in family_stats.items()
        if stats["effect_interval"][1] < -minimum_effect
    ]
    pooled_wins = all_differences.count(1)
    pooled_losses = all_differences.count(-1)
    pooled_effect = (
        sum(all_differences) / len(all_differences) if all_differences else 0.0
    )
    pooled_low, pooled_high = _paired_bootstrap_interval(
        all_differences,
        alpha=alpha,
        seed=20260716,
    )
    pooled_p = _exact_paired_pvalue_greater(pooled_wins, pooled_losses)
    evidence = {
        "method": (
            "paired exact McNemar/binomial + Holm correction + "
            "alpha-derived one-sided percentile bounds"
        ),
        "alpha": alpha,
        "minimum_effect": minimum_effect,
        "compute_tolerance": compute_tolerance,
        "bootstrap_resamples": _BOOTSTRAP_RESAMPLES,
        "families": family_stats,
        "holm_adjusted_p": adjusted,
        "positive_families": positive_families,
        "regressed_families": regressed_families,
        "underpowered_families": underpowered,
        "invalid_compute_families": invalid_compute,
        "pooled": {
            "n": len(all_differences),
            "treatment_wins": pooled_wins,
            "control_wins": pooled_losses,
            "paired_effect": round(pooled_effect, 6),
            "effect_interval": [round(pooled_low, 6), round(pooled_high, 6)],
            "effect_bound_alpha": alpha,
            "one_sided_exact_p": pooled_p,
        },
    }
    pooled_positive = (
        len(all_differences) >= _MIN_N_FOR_VERDICT
        and pooled_p < alpha
        and pooled_low > minimum_effect
    )
    required_positive = max(2, math.ceil(len(family_stats) * 2 / 3))
    evidence["required_positive_families"] = required_positive
    if invalid_compute or underpowered:
        tier = CONJECTURE
    elif (
        len(positive_families) >= required_positive
        and pooled_positive
        and not regressed_families
    ):
        tier = PROVEN
    elif positive_families and pooled_positive and not regressed_families:
        tier = SUPPORTED
    elif regressed_families or (all_differences and pooled_high <= 0.0):
        tier = REFUTED
    else:
        tier = CONJECTURE
    return Claim(experiment=experiment, statement=statement, tier=tier, evidence=evidence)


@dataclass
class Claim:
    experiment: str
    statement: str
    tier: str
    evidence: dict[str, Any] = field(default_factory=dict)
    graded_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "statement": self.statement,
            "tier": self.tier,
            "evidence": self.evidence,
            "graded_at": self.graded_at,
        }


def grade_treatment_vs_control(
    experiment: str,
    statement: str,
    treatment_by_family: dict[str, ArmResult],
    control_by_family: dict[str, ArmResult],
) -> Claim:
    """The conservative comparison grader shared by Experiments 1, 4, 5."""
    wins, losses, small = [], [], []
    for family, treat in treatment_by_family.items():
        control = control_by_family.get(family)
        if control is None or treat.n < _MIN_N_FOR_VERDICT or control.n < _MIN_N_FOR_VERDICT:
            small.append(family)
            continue
        if treat.lb > control.ub:
            wins.append(family)
        elif treat.accuracy <= control.accuracy:
            losses.append(family)
    evidence = {
        "treatment": {f: a.to_dict() for f, a in treatment_by_family.items()},
        "control": {f: a.to_dict() for f, a in control_by_family.items()},
        "separated_families": wins,
        "not_better_families": losses,
        "underpowered_families": small,
    }
    evidence["aggregate_only"] = True
    evidence["limitation"] = (
        "aggregate Wilson intervals lack paired task outcomes and cannot earn PROVEN"
    )
    if wins:
        tier = SUPPORTED
    elif small and not losses:
        tier = CONJECTURE
    elif losses and not wins:
        tier = REFUTED
    else:
        tier = CONJECTURE
    return Claim(experiment=experiment, statement=statement, tier=tier, evidence=evidence)


# ── Experiment 1: recurrence utility sweep ──────────────────────────────


def run_recurrence_sweep(
    solve: Callable[[Task, int], bool | tuple[bool, int]],
    tasks: list[Task],
    step_grid: list[int],
    *,
    baseline: Callable[[Task], bool | tuple[bool, int]] | None = None,
) -> dict[str, Any]:
    """Accuracy as a function of forced recurrence depth.

    ``solve(task, steps)`` runs one latent episode at exactly ``steps``
    recurrent steps and returns verified success. ``baseline(task)`` is the
    equal-FLOP conventional arm (longer CoT / best-of-N), supplied by the
    caller so its compute accounting is visible in the report, not implied.
    """
    if not step_grid or sorted(set(step_grid)) != step_grid or any(step < 1 for step in step_grid):
        raise ValueError("step_grid must be sorted, unique, and positive")
    curve: list[dict[str, Any]] = []
    outcomes_by_step: dict[int, list[tuple[bool, int | None]]] = {}
    for steps in step_grid:
        arm = ArmResult(name=f"steps={steps}")
        for task in tasks:
            success, cost = _coerce_solver_outcome(solve(task, steps))
            arm.n += 1
            arm.successes += int(success)
            arm.layer_apps += int(cost or 0)
            outcomes_by_step.setdefault(steps, []).append((success, cost))
        curve.append(arm.to_dict())
    result: dict[str, Any] = {"curve": curve}
    baseline_outcomes: list[tuple[bool, int | None]] = []
    if baseline is not None:
        base = ArmResult(name="equal_flop_baseline")
        for task in tasks:
            success, cost = _coerce_solver_outcome(baseline(task))
            base.n += 1
            base.successes += int(success)
            base.layer_apps += int(cost or 0)
            baseline_outcomes.append((success, cost))
        result["baseline"] = base.to_dict()
    accs = [c["accuracy"] for c in curve]
    result["monotone_gain"] = len(accs) >= 2 and all(
        b >= a - 1e-9 for a, b in zip(accs, accs[1:], strict=False)
    ) and accs[-1] > accs[0]
    if baseline is None:
        claim = Claim(
            experiment="exp1_recurrence_sweep",
            statement="additional recurrent steps improve equal-compute accuracy",
            tier=CONJECTURE,
            evidence={
                "curve": curve,
                "n_tasks": len(tasks),
                "limitation": "equal-compute baseline missing",
            },
        )
    else:
        deepest = outcomes_by_step[step_grid[-1]]
        paired: dict[str, list[PairedObservation]] = {}
        for index, (task, treatment, control) in enumerate(
            zip(tasks, deepest, baseline_outcomes, strict=True)
        ):
            paired.setdefault(task.family, []).append(
                PairedObservation(
                    task_id=f"{task.family}:{task.depth}:{task.seed}:{index}",
                    family=task.family,
                    treatment_success=treatment[0],
                    control_success=control[0],
                    treatment_layer_apps=treatment[1],
                    control_layer_apps=control[1],
                )
            )
        claim = grade_paired_treatment_vs_control(
            "exp1_recurrence_sweep",
            "additional recurrent steps improve equal-compute accuracy",
            paired,
        )
        if not result["monotone_gain"] and claim.tier in {PROVEN, SUPPORTED}:
            claim.tier = CONJECTURE
            claim.evidence["voided"] = "deepest arm won but recurrence curve was not monotone"
    result["claim"] = claim.to_dict()
    return result


# ── Experiment 2: depth extrapolation ───────────────────────────────────


def run_depth_extrapolation(
    solve: Callable[[Task, int], bool | tuple[bool, int]],
    family: str,
    depths: list[int],
    step_grid: list[int],
    *,
    per_depth: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    """T_required(depth): the minimum recurrence at which each depth is solved.

    The signature of genuine latent computation is T_required growing with
    problem depth while remaining solvable — compute buys composition."""
    gen = TASK_FAMILIES[family]
    t_required: dict[int, int | None] = {}
    matrix: dict[int, dict[int, float]] = {}
    for depth in depths:
        tasks = [gen(depth, seed * 31 + i) for i in range(per_depth)]
        matrix[depth] = {}
        t_required[depth] = None
        for steps in step_grid:
            wins = sum(
                int(_coerce_solver_outcome(solve(task, steps))[0]) for task in tasks
            )
            acc = wins / len(tasks)
            matrix[depth][steps] = round(acc, 4)
            if acc >= 0.5 and t_required[depth] is None:
                t_required[depth] = steps
    solved = [d for d in depths if t_required[d] is not None]
    pairs = [(d, t_required[d]) for d in solved]
    increasing = all(
        t2 >= t1 for (_, t1), (_, t2) in zip(pairs, pairs[1:], strict=False)
    )
    scaling = len(solved) >= 3 and increasing and len(set(t for _, t in pairs)) > 1
    tier = CONJECTURE if per_depth * len(depths) < _MIN_N_FOR_VERDICT else (
        SUPPORTED if scaling else (REFUTED if len(solved) >= 3 else CONJECTURE)
    )
    return {
        "family": family,
        "matrix": matrix,
        "t_required": t_required,
        "claim": Claim(
            experiment="exp2_depth_extrapolation",
            statement="required recurrence scales with compositional depth",
            tier=tier,
            evidence={"t_required": {str(k): v for k, v in t_required.items()}},
        ).to_dict(),
    }


# ── Experiment 3: slot causality ────────────────────────────────────────


def run_slot_causality(
    solve_with_ablation: Callable[[Task, int | None], bool],
    tasks: list[Task],
    slot_indices: list[int],
) -> dict[str, Any]:
    """Ablate slots one at a time; restore must recover performance.

    ``solve_with_ablation(task, slot)`` runs an episode with slot ``slot``
    destroyed pre-persist (None ⇒ intact). Causal workspace ⇒ intact runs
    beat ablated runs, and per-slot damage is measurable."""
    intact = ArmResult(name="intact")
    for task in tasks:
        intact.n += 1
        intact.successes += int(bool(solve_with_ablation(task, None)))
    per_slot: dict[int, ArmResult] = {}
    paired_claims: dict[int, Claim] = {}
    for slot in slot_indices:
        arm = ArmResult(name=f"ablated_slot_{slot}")
        observations: dict[str, list[PairedObservation]] = {}
        for index, task in enumerate(tasks):
            ablated_success = bool(solve_with_ablation(task, slot))
            intact_success = bool(solve_with_ablation(task, None))
            arm.n += 1
            arm.successes += int(ablated_success)
            observations.setdefault(task.family, []).append(
                PairedObservation(
                    task_id=f"{task.family}:{task.depth}:{task.seed}:{index}:slot{slot}",
                    family=task.family,
                    treatment_success=intact_success,
                    control_success=ablated_success,
                )
            )
        per_slot[slot] = arm
        paired_claims[slot] = grade_paired_treatment_vs_control(
            "exp3_slot_causality",
            f"slot {slot} carries causally necessary computation",
            observations,
            require_compute=False,
        )
    damaged = [
        slot
        for slot, claim in paired_claims.items()
        if claim.tier in {PROVEN, SUPPORTED}
    ]
    tier = CONJECTURE if intact.n < _MIN_N_FOR_VERDICT else (
        SUPPORTED if damaged else REFUTED
    )
    return {
        "intact": intact.to_dict(),
        "ablated": {s: a.to_dict() for s, a in per_slot.items()},
        "causally_necessary_slots": damaged,
        "claim": Claim(
            experiment="exp3_slot_causality",
            statement="thought slots carry causally necessary intermediate computation",
            tier=tier,
            evidence={"damaged_slots": damaged, "intact_accuracy": intact.accuracy},
        ).to_dict(),
        "paired_slot_claims": {
            slot: claim.to_dict() for slot, claim in paired_claims.items()
        },
    }


# ── Experiment 4: virtual width vs equal-FLOP sampling ──────────────────


def run_virtual_width(
    solve_branches: Callable[[Task, int], tuple[bool, int]],
    solve_sampling: Callable[[Task, int], tuple[bool, int]],
    tasks_by_family: dict[str, list[Task]],
    k: int,
) -> dict[str, Any]:
    """K latent branches vs K textual samples at (verified-)equal FLOPs.

    Both callbacks return (success, layer_apps_spent) so the equal-compute
    premise is CHECKED, not assumed: a >10% compute mismatch voids the claim."""
    treatment: dict[str, ArmResult] = {}
    control: dict[str, ArmResult] = {}
    paired: dict[str, list[PairedObservation]] = {}
    for family, tasks in tasks_by_family.items():
        t_arm, c_arm = ArmResult(name=f"branches_k{k}"), ArmResult(name=f"sampling_k{k}")
        for index, task in enumerate(tasks):
            ok_b, cost_b = solve_branches(task, k)
            ok_s, cost_s = solve_sampling(task, k)
            t_arm.n += 1
            t_arm.successes += int(bool(ok_b))
            t_arm.layer_apps += int(cost_b)
            c_arm.n += 1
            c_arm.successes += int(bool(ok_s))
            c_arm.layer_apps += int(cost_s)
            paired.setdefault(family, []).append(
                PairedObservation(
                    task_id=f"{family}:{task.depth}:{task.seed}:{index}",
                    family=family,
                    treatment_success=bool(ok_b),
                    control_success=bool(ok_s),
                    treatment_layer_apps=int(cost_b),
                    control_layer_apps=int(cost_s),
                )
            )
        treatment[family], control[family] = t_arm, c_arm
    claim = grade_paired_treatment_vs_control(
        "exp4_virtual_width",
        "latent branches beat equal-FLOP self-consistency sampling",
        paired,
    )
    return {
        "treatment": {f: a.to_dict() for f, a in treatment.items()},
        "control": {f: a.to_dict() for f, a in control.items()},
        "claim": claim.to_dict(),
    }


def extract_final_numeric_claim(text: str) -> str:
    """The candidate's final numeric claim, by the SAME rule Task.verify uses.

    Self-consistency voting needs answer extraction that cannot peek at the
    ground truth: the last numeric token wins, hedging loses."""
    tokens = [t.strip(".,:;!()[]") for t in str(text or "").split()]
    numeric = [t for t in tokens if t and t.lstrip("-").isdigit()]
    return numeric[-1] if numeric else ""


def majority_answer(answers: list[str]) -> str:
    """Most common non-empty extracted answer; deterministic tie-break."""
    filtered = [answer for answer in answers if answer]
    if not filtered:
        return ""
    counts: dict[str, int] = {}
    for answer in filtered:
        counts[answer] = counts.get(answer, 0) + 1
    top = max(counts.values())
    return sorted(answer for answer, count in counts.items() if count == top)[0]


# ── Factorial ablations: which mechanism carries any gain ───────────────

FACTORIAL_ARMS: tuple[str, ...] = (
    "recurrence_only",
    "branches_only",
    "latent_opt_only",
    "fast_weights_only",
    "recurrence_branches",
    "recurrence_verifier",
    "full_stack",
)


def run_factorial_ablations(
    solve_arm: Callable[[Task, str], tuple[bool, int]],
    tasks_by_family: dict[str, list[Task]],
    *,
    arms: tuple[str, ...] = FACTORIAL_ARMS,
) -> dict[str, Any]:
    """Attribute any gain to a mechanism: every arm paired against vanilla.

    ``solve_arm(task, arm)`` runs one configuration ("vanilla" is the
    ordinary-decoding control; the treatment arms enable one mechanism or a
    named combination). Each arm earns its own paired claim vs vanilla on
    the SAME tasks, so "the full stack helps" can be decomposed into which
    ingredient actually carried the effect — the RSL gap analysis's
    mechanism-attribution obligation."""
    arm_names = ("vanilla", *arms)
    results: dict[str, dict[str, ArmResult]] = {
        arm: {family: ArmResult(name=arm) for family in tasks_by_family}
        for arm in arm_names
    }
    outcomes: dict[str, dict[str, list[tuple[bool, int]]]] = {
        arm: {family: [] for family in tasks_by_family} for arm in arm_names
    }
    for family, tasks in tasks_by_family.items():
        for task in tasks:
            for arm in arm_names:
                success, cost = solve_arm(task, arm)
                row = results[arm][family]
                row.n += 1
                row.successes += int(bool(success))
                row.layer_apps += int(cost)
                outcomes[arm][family].append((bool(success), int(cost)))
    claims: dict[str, dict[str, Any]] = {}
    for arm in arms:
        paired: dict[str, list[PairedObservation]] = {}
        for family, tasks in tasks_by_family.items():
            for index, task in enumerate(tasks):
                treatment = outcomes[arm][family][index]
                control = outcomes["vanilla"][family][index]
                paired.setdefault(family, []).append(
                    PairedObservation(
                        task_id=f"{family}:{task.depth}:{task.seed}:{index}",
                        family=family,
                        treatment_success=treatment[0],
                        control_success=control[0],
                        treatment_layer_apps=treatment[1],
                        control_layer_apps=control[1],
                    )
                )
        claims[arm] = grade_paired_treatment_vs_control(
            f"ablation_{arm}",
            f"mechanism arm '{arm}' beats vanilla decoding on the same tasks",
            paired,
            # Mechanism arms intentionally spend different compute than
            # vanilla — attribution is about direction, not FLOP parity;
            # Experiments 1/4 own the equal-compute claims.
            require_compute=False,
        ).to_dict()
    attribution = [
        arm
        for arm in arms
        if claims[arm]["tier"] in {PROVEN, SUPPORTED}
    ]
    return {
        "arms": {
            arm: {family: row.to_dict() for family, row in families.items()}
            for arm, families in results.items()
        },
        "claims": claims,
        "attribution": attribution,
    }


# ── Experiment 5: latent optimization vs random control ─────────────────


def _latent_opt_arm_order(family: str, task: Task, index: int) -> tuple[str, str, str]:
    commitment = f"latent-opt-order-v1:{family}:{task.depth}:{task.seed}:{index}".encode()
    digest = hashlib.sha256(commitment).digest()
    family_offset = hashlib.sha256(f"{family}:latent-opt-order-v1".encode()).digest()[0] & 1
    gradient_first = (index + family_offset) % 2 == 0
    pair = ["gradient", "control"] if gradient_first else ["control", "gradient"]
    pair.insert((index + digest[1]) % 3, "off")
    return pair[0], pair[1], pair[2]


def run_latent_opt_control(
    solve_arm: Callable[[Task, str], bool | tuple[bool, int]],
    tasks_by_family: dict[str, list[Task]],
) -> dict[str, Any]:
    """Arms: 'off', 'gradient', 'control' (matched-magnitude random).

    The claim is only about DIRECTION: gradient must beat the random control,
    not merely beat doing nothing. That is the spec's essential control."""
    arms = ("off", "gradient", "control")
    results: dict[str, dict[str, ArmResult]] = {a: {} for a in arms}
    per_task: dict[str, dict[str, list[tuple[bool, int | None]]]] = {
        arm: {} for arm in arms
    }
    execution_order: list[dict[str, Any]] = []
    for family, tasks in tasks_by_family.items():
        family_results = {arm: ArmResult(name=arm) for arm in arms}
        for index, task in enumerate(tasks):
            order = _latent_opt_arm_order(family, task, index)
            task_id = f"{family}:{task.depth}:{task.seed}:{index}"
            execution_order.append({"task_id": task_id, "arms": list(order)})
            for arm in order:
                success, cost = _coerce_solver_outcome(solve_arm(task, arm))
                r = family_results[arm]
                r.n += 1
                r.successes += int(success)
                r.layer_apps += int(cost or 0)
                per_task[arm].setdefault(family, []).append((success, cost))
        for arm, result in family_results.items():
            results[arm][family] = result
    paired: dict[str, list[PairedObservation]] = {}
    for family, tasks in tasks_by_family.items():
        for index, task in enumerate(tasks):
            gradient = per_task["gradient"][family][index]
            control = per_task["control"][family][index]
            paired.setdefault(family, []).append(
                PairedObservation(
                    task_id=f"{family}:{task.depth}:{task.seed}:{index}",
                    family=family,
                    treatment_success=gradient[0],
                    control_success=control[0],
                    treatment_layer_apps=gradient[1],
                    control_layer_apps=control[1],
                )
            )
    claim = grade_paired_treatment_vs_control(
        "exp5_latent_opt",
        "gradient direction (not mere perturbation) improves outcomes",
        paired,
    )
    return {
        "arms": {a: {f: r.to_dict() for f, r in fam.items()} for a, fam in results.items()},
        "execution_order": execution_order,
        "claim": claim.to_dict(),
    }


# ── Experiment 6: frontier comparison protocol ──────────────────────────


def frontier_comparison_protocol() -> dict[str, Any]:
    """The operator checklist for the only claim that finally counts."""
    return {
        "preconditions": [
            "architecture and schedule library FROZEN before task generation",
            "fresh blind tasks generated after freeze (no benchmark reuse)",
            "checkpoint SHA recorded and republished with results",
        ],
        "controls": [
            "equal problem information for both systems",
            "equal tool and verification access",
            "equal-latency AND equal-compute result columns",
            "no benchmark-specific answer caches",
        ],
        "domains": [
            "novel algorithmic reasoning",
            "mathematics",
            "coding",
            "scientific inference",
            "long-horizon planning",
            "calibration",
            "robustness to misleading premises",
        ],
        "report": "publish per-domain Wilson intervals; the weakest domain is the headline",
    }


# ── Foundry recording ───────────────────────────────────────────────────


def record_claim_to_foundry(claim: Claim | dict[str, Any], domain: str) -> bool:
    """Log an experiment verdict into the Verifier Foundry reliability ledger."""
    body = claim.to_dict() if isinstance(claim, Claim) else dict(claim)
    try:
        from core.brain.verifiers.foundry import get_verifier_foundry

        foundry = get_verifier_foundry()
        verdict_id = foundry.record_verdict(
            verifier=f"latent_cortex.{body.get('experiment', 'unknown')}",
            domain=domain,
            hard_pass=body.get("tier") in (PROVEN, SUPPORTED),
            score=0.9 if body.get("tier") == PROVEN else 0.6,
            checked=True,
            meta={"statement": body.get("statement"), "tier": body.get("tier")},
        )
        return bool(verdict_id)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        from core.runtime.errors import record_degradation

        record_degradation(
            "latent_cortex",
            exc,
            action="kept experiment claim local after foundry recording failed",
        )
        return False


# ── Experiment R: are role anchors causal cognitive labor? ──────────────

ROLE_ARMS: tuple[str, ...] = (
    "distinct_roles",
    "lesioned_uniform_role",
    "swapped_roles",
)


def run_role_lesion(
    solve_arm: "Callable[[Task, str], tuple[bool, int, float]]",
    tasks_by_family: dict[str, list[Task]],
    *,
    divergence_margin: float = 0.02,
) -> dict[str, Any]:
    """Lesion/swap the branch role anchors and measure what they carry.

    ``solve_arm(task, arm)`` runs one arm and returns
    (success, layer_apps, branch_divergence) where branch_divergence is
    1 − mean pairwise branch-summary cosine at exchanges (NaN when the
    episode had no exchange telemetry). Arms:

    - distinct_roles: the default role rotation (treatment);
    - lesioned_uniform_role: every branch gets the SAME anchor — role
      diversity removed, everything else identical;
    - swapped_roles: the same distinct anchors, permuted across branch
      indices — if roles are causal, outcomes should track the anchors,
      not the branch index.

    Claims: a paired behavioral claim (distinct vs lesioned), a
    mechanistic divergence claim (distinct trajectories diverge more than
    lesioned ones by at least ``divergence_margin``), and a swap-parity
    observation (swapped ≈ distinct implies anchor-causality, not
    index-causality). Divergence claims cap at SUPPORTED: internal
    geometry cannot earn PROVEN.
    """
    if (
        isinstance(divergence_margin, bool)
        or not isinstance(divergence_margin, (int, float))
        or not math.isfinite(float(divergence_margin))
        or not 0.0 <= float(divergence_margin) < 1.0
    ):
        raise ValueError("divergence_margin must be a finite number in [0, 1)")
    outcomes: dict[str, dict[str, list[tuple[bool, int, float]]]] = {
        arm: {} for arm in ROLE_ARMS
    }
    for arm in ROLE_ARMS:
        for family, tasks in tasks_by_family.items():
            rows = outcomes[arm].setdefault(family, [])
            for task in tasks:
                ok, cost, divergence = solve_arm(task, arm)
                rows.append((bool(ok), int(cost), float(divergence)))

    paired: dict[str, list[PairedObservation]] = {}
    for family, tasks in tasks_by_family.items():
        for index, task in enumerate(tasks):
            treatment = outcomes["distinct_roles"][family][index]
            control = outcomes["lesioned_uniform_role"][family][index]
            paired.setdefault(family, []).append(
                PairedObservation(
                    task_id=f"{family}:{task.depth}:{task.seed}:{index}",
                    family=family,
                    treatment_success=treatment[0],
                    control_success=control[0],
                    treatment_layer_apps=treatment[1],
                    control_layer_apps=control[1],
                )
            )
    behavioral = grade_paired_treatment_vs_control(
        "expR_role_diversity",
        "distinct role anchors beat a lesioned uniform-role ensemble",
        paired,
        # Arms run identical topology; compute parity is structural here.
        require_compute=False,
    )

    def _mean_divergence(arm: str) -> tuple[float, int]:
        values = [
            divergence
            for rows in outcomes[arm].values()
            for _, _, divergence in rows
            if math.isfinite(divergence)
        ]
        if not values:
            return float("nan"), 0
        return sum(values) / len(values), len(values)

    distinct_div, distinct_n = _mean_divergence("distinct_roles")
    lesioned_div, lesioned_n = _mean_divergence("lesioned_uniform_role")
    swapped_div, swapped_n = _mean_divergence("swapped_roles")
    divergence_evidence = {
        "distinct_mean_divergence": distinct_div,
        "lesioned_mean_divergence": lesioned_div,
        "swapped_mean_divergence": swapped_div,
        "samples": {
            "distinct_roles": distinct_n,
            "lesioned_uniform_role": lesioned_n,
            "swapped_roles": swapped_n,
        },
        "divergence_margin": float(divergence_margin),
        "limitation": (
            "internal trajectory geometry; decorrelation jitter fires on "
            "near-collapse ensembles and partially masks lesioning"
        ),
    }
    enough = min(distinct_n, lesioned_n) >= _MIN_N_FOR_VERDICT
    if not enough or not (
        math.isfinite(distinct_div) and math.isfinite(lesioned_div)
    ):
        divergence_tier = CONJECTURE
    elif distinct_div - lesioned_div >= float(divergence_margin):
        divergence_tier = SUPPORTED
    elif lesioned_div >= distinct_div:
        divergence_tier = REFUTED
    else:
        divergence_tier = CONJECTURE
    mechanistic = Claim(
        experiment="expR_role_divergence",
        statement=(
            "distinct role anchors produce more divergent branch "
            "trajectories than a lesioned uniform-role ensemble"
        ),
        tier=divergence_tier,
        evidence=divergence_evidence,
    )

    swap_parity: dict[str, Any] = {
        "note": (
            "swapped ≈ distinct on both accuracy and divergence implies the "
            "ANCHOR, not the branch index, carries the role"
        )
    }
    for family in tasks_by_family:
        distinct_acc = sum(
            1 for ok, _, _ in outcomes["distinct_roles"][family] if ok
        ) / max(1, len(outcomes["distinct_roles"][family]))
        swapped_acc = sum(
            1 for ok, _, _ in outcomes["swapped_roles"][family] if ok
        ) / max(1, len(outcomes["swapped_roles"][family]))
        swap_parity[family] = {
            "distinct_accuracy": round(distinct_acc, 4),
            "swapped_accuracy": round(swapped_acc, 4),
        }

    return {
        "arms": {
            arm: {
                family: {
                    "n": len(rows),
                    "successes": sum(1 for ok, _, _ in rows if ok),
                    "layer_apps": sum(cost for _, cost, _ in rows),
                }
                for family, rows in by_family.items()
            }
            for arm, by_family in outcomes.items()
        },
        "behavioral_claim": behavioral.to_dict(),
        "divergence_claim": mechanistic.to_dict(),
        "swap_parity": swap_parity,
    }


__all__ = [
    "ROLE_ARMS",
    "run_role_lesion",
    "ArmResult",
    "CONJECTURE",
    "Claim",
    "PairedObservation",
    "PROVEN",
    "REFUTED",
    "SUPPORTED",
    "TASK_FAMILIES",
    "Task",
    "FACTORIAL_ARMS",
    "extract_final_numeric_claim",
    "frontier_comparison_protocol",
    "grade_treatment_vs_control",
    "grade_paired_treatment_vs_control",
    "khop_reachability",
    "majority_answer",
    "modular_chain",
    "nested_boolean",
    "record_claim_to_foundry",
    "run_depth_extrapolation",
    "run_factorial_ablations",
    "run_latent_opt_control",
    "run_recurrence_sweep",
    "run_slot_causality",
    "run_virtual_width",
    "task_battery",
]
