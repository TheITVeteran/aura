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

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from core.brain.verifiers.foundry import wilson_lower_bound, wilson_upper_bound

logger = logging.getLogger("Aura.LatentCortex.Experiments")

PROVEN = "PROVEN"
SUPPORTED = "SUPPORTED"
CONJECTURE = "CONJECTURE"
REFUTED = "REFUTED"

_MIN_N_FOR_VERDICT = 20  # below this, everything is CONJECTURE


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
    if len(wins) >= 2:
        tier = PROVEN
    elif len(wins) == 1:
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
    solve: Callable[[Task, int], bool],
    tasks: list[Task],
    step_grid: list[int],
    *,
    baseline: Callable[[Task], bool] | None = None,
) -> dict[str, Any]:
    """Accuracy as a function of forced recurrence depth.

    ``solve(task, steps)`` runs one latent episode at exactly ``steps``
    recurrent steps and returns verified success. ``baseline(task)`` is the
    equal-FLOP conventional arm (longer CoT / best-of-N), supplied by the
    caller so its compute accounting is visible in the report, not implied.
    """
    curve: list[dict[str, Any]] = []
    for steps in step_grid:
        arm = ArmResult(name=f"steps={steps}")
        for task in tasks:
            arm.n += 1
            arm.successes += int(bool(solve(task, steps)))
        curve.append(arm.to_dict())
    result: dict[str, Any] = {"curve": curve}
    if baseline is not None:
        base = ArmResult(name="equal_flop_baseline")
        for task in tasks:
            base.n += 1
            base.successes += int(bool(baseline(task)))
        result["baseline"] = base.to_dict()
    accs = [c["accuracy"] for c in curve]
    result["monotone_gain"] = all(b >= a - 1e-9 for a, b in zip(accs, accs[1:])) and accs[-1] > accs[0]
    result["claim"] = Claim(
        experiment="exp1_recurrence_sweep",
        statement="additional recurrent steps systematically improve accuracy",
        tier=(
            CONJECTURE
            if len(tasks) < _MIN_N_FOR_VERDICT
            else (SUPPORTED if result["monotone_gain"] else REFUTED)
        ),
        evidence={"curve": curve, "n_tasks": len(tasks)},
    ).to_dict()
    return result


# ── Experiment 2: depth extrapolation ───────────────────────────────────


def run_depth_extrapolation(
    solve: Callable[[Task, int], bool],
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
            wins = sum(int(bool(solve(t, steps))) for t in tasks)
            acc = wins / len(tasks)
            matrix[depth][steps] = round(acc, 4)
            if acc >= 0.5 and t_required[depth] is None:
                t_required[depth] = steps
    solved = [d for d in depths if t_required[d] is not None]
    pairs = [(d, t_required[d]) for d in solved]
    increasing = all(t2 >= t1 for (_, t1), (_, t2) in zip(pairs, pairs[1:]))
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
    for slot in slot_indices:
        arm = ArmResult(name=f"ablated_slot_{slot}")
        for task in tasks:
            arm.n += 1
            arm.successes += int(bool(solve_with_ablation(task, slot)))
        per_slot[slot] = arm
    damaged = [s for s, a in per_slot.items() if a.ub < intact.lb]
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
    for family, tasks in tasks_by_family.items():
        t_arm, c_arm = ArmResult(name=f"branches_k{k}"), ArmResult(name=f"sampling_k{k}")
        for task in tasks:
            ok_b, cost_b = solve_branches(task, k)
            ok_s, cost_s = solve_sampling(task, k)
            t_arm.n += 1
            t_arm.successes += int(bool(ok_b))
            t_arm.layer_apps += int(cost_b)
            c_arm.n += 1
            c_arm.successes += int(bool(ok_s))
            c_arm.layer_apps += int(cost_s)
        treatment[family], control[family] = t_arm, c_arm
    claim = grade_treatment_vs_control(
        "exp4_virtual_width",
        "latent branches beat equal-FLOP self-consistency sampling",
        treatment,
        control,
    )
    total_t = sum(a.layer_apps for a in treatment.values())
    total_c = sum(a.layer_apps for a in control.values())
    if total_c and abs(total_t - total_c) / total_c > 0.10:
        claim.tier = CONJECTURE
        claim.evidence["voided"] = (
            f"compute mismatch {total_t} vs {total_c} layer-apps exceeds 10%"
        )
    return {
        "treatment": {f: a.to_dict() for f, a in treatment.items()},
        "control": {f: a.to_dict() for f, a in control.items()},
        "claim": claim.to_dict(),
    }


# ── Experiment 5: latent optimization vs random control ─────────────────


def run_latent_opt_control(
    solve_arm: Callable[[Task, str], bool],
    tasks_by_family: dict[str, list[Task]],
) -> dict[str, Any]:
    """Arms: 'off', 'gradient', 'control' (matched-magnitude random).

    The claim is only about DIRECTION: gradient must beat the random control,
    not merely beat doing nothing. That is the spec's essential control."""
    arms = ("off", "gradient", "control")
    results: dict[str, dict[str, ArmResult]] = {a: {} for a in arms}
    for family, tasks in tasks_by_family.items():
        for arm in arms:
            r = ArmResult(name=arm)
            for task in tasks:
                r.n += 1
                r.successes += int(bool(solve_arm(task, arm)))
            results[arm][family] = r
    claim = grade_treatment_vs_control(
        "exp5_latent_opt",
        "gradient direction (not mere perturbation) improves outcomes",
        results["gradient"],
        results["control"],
    )
    return {
        "arms": {a: {f: r.to_dict() for f, r in fam.items()} for a, fam in results.items()},
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


__all__ = [
    "ArmResult",
    "CONJECTURE",
    "Claim",
    "PROVEN",
    "REFUTED",
    "SUPPORTED",
    "TASK_FAMILIES",
    "Task",
    "frontier_comparison_protocol",
    "grade_treatment_vs_control",
    "khop_reachability",
    "modular_chain",
    "nested_boolean",
    "record_claim_to_foundry",
    "run_depth_extrapolation",
    "run_latent_opt_control",
    "run_recurrence_sweep",
    "run_slot_causality",
    "run_virtual_width",
    "task_battery",
]
