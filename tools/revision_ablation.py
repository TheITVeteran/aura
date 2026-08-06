"""Does the revision gate earn its cost? — a capability ablation with no budget confound.

The retrieval ablation (`tools/capability_ablation.py`) answers one question:
memory retrieval beats a budgeted context window when the fact is out of
window. That is a real result and it is ONE subsystem. It does not license
anything about the cognitive layer, and the standing criticism — that the
architecture's complexity has never been shown to solve a problem a simpler
system could not — survives it.

This battery attacks the criticism where it is sharpest, on the component whose
whole justification is that thinking twice is worth the compute.

WHY THIS LESION IS UNUSUALLY CLEAN
Every arm consumes the SAME two generations. Pass 1 answers; pass 2 re-answers
with the standard "check your work" nudge. The arms then differ only in which
of those two already-paid-for answers they KEEP:

    single_pass     always keep pass 1        (the simpler system)
    always_revise   always keep pass 2        (naive "think again")
    gated_revision  ask decide_revision()     (Aura)

Identical model, identical prompts, identical token spend, identical wall
clock — the three arms are literally scored over the same generations. There is
no budget to equalise and no compute story available to explain a difference
away. Whatever separates them is the decision rule, because nothing else is
left.

WHAT WOULD REFUTE AURA HERE
If `always_revise` matches `gated_revision`, the gate is dead weight: a second
pass would be unconditionally worth taking and the machinery that decides is
buying nothing. If `single_pass` matches `gated_revision`, second passes are
not worth taking at all and the whole revision layer is unearned. The gate is
only justified in the narrow case where second thoughts help SOMETIMES and hurt
other times, and a rule that can tell which is which beats both fixed policies.

That is a falsifiable claim with two named ways to lose, and this tool reports
the losses in the same voice as the wins.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.brain.reasoning_revision_gate import decide_revision  # noqa: E402
from core.evaluation.lesion_inference import (  # noqa: E402
    InferenceClass,
    LesionClaim,
    summarise,
)
from core.evaluation.matched_budget import (  # noqa: E402
    Attempt,
    AttemptLedger,
    paired_separation,
)

SINGLE_PASS = "single_pass"
ALWAYS_REVISE = "always_revise"
GATED = "gated_revision"
ARMS = (SINGLE_PASS, ALWAYS_REVISE, GATED)

DEFAULT_OUT = _REPO_ROOT / "artifacts" / "ablation" / "revision_scorecard.json"


@dataclass(frozen=True)
class Verdict:
    """The duck-typed shape `quality_bound` reads: ok / checked / score."""

    ok: bool
    checked: bool
    score: float


@dataclass
class RevisionTask:
    task_id: str
    prompt: str
    answer_key: str
    # A task the verifier cannot check is still run: it produces an UNCHECKED
    # verdict, which is exactly the evidence-poor case rule 3 of the gate
    # exists to handle. Dropping those tasks would hand the gate only the
    # situations it is good at.
    checkable: bool = True
    detail: dict[str, Any] = field(default_factory=dict)


def grade(output: str, task: RevisionTask) -> float:
    """Exact-answer grading, whitespace- and case-insensitive.

    Deliberately strict: a loose grader that accepts any answer CONTAINING the
    key turns "42 or maybe 43" into a success and quietly inflates every arm.
    """
    text = " ".join(str(output or "").split()).strip().lower()
    key = " ".join(task.answer_key.split()).strip().lower()
    if not text or not key:
        return 0.0
    # Accept the key as the final token/phrase of the answer, not anywhere in it.
    return 1.0 if text == key or text.endswith(key) else 0.0


def verify(output: str, task: RevisionTask) -> Verdict:
    """The verifier the gate consults. NOT the grader.

    A verifier that were simply the answer key would make the gate omniscient
    and the whole experiment circular — it would always keep the correct pass
    and "beat" both baselines by construction. This one checks a NECESSARY but
    not sufficient property (well-formedness of the answer against the task's
    own constraint), which is what a real verifier does: it can reject wrong
    answers it recognises, and it passes wrong answers it cannot.
    """
    if not task.checkable:
        return Verdict(ok=False, checked=False, score=0.5)

    text = " ".join(str(output or "").split()).strip()
    stripped = text.rstrip(".").replace(",", "").split()
    candidate = stripped[-1] if stripped else ""
    try:
        value = int(candidate)
    except (TypeError, ValueError):
        # Not even a number: a real, decisive rejection.
        return Verdict(ok=False, checked=True, score=0.0)

    # Check the constraints the task STATES. This is the whole reason the
    # battery is constraint-satisfaction rather than arithmetic: checking
    # "is it in [40,50] and divisible by 7" is genuinely cheaper than
    # searching for such a number, so the verifier is doing real work without
    # being a disguised copy of the answer key. A verifier that recomputed the
    # answer would make the gate omniscient and every result circular.
    lo = task.detail.get("min")
    hi = task.detail.get("max")
    if lo is not None and value < lo:
        return Verdict(ok=False, checked=True, score=0.0)
    if hi is not None and value > hi:
        return Verdict(ok=False, checked=True, score=0.0)

    divisor = task.detail.get("divisible_by")
    if divisor and value % int(divisor) != 0:
        return Verdict(ok=False, checked=True, score=0.0)

    parity = task.detail.get("parity")
    if parity == "even" and value % 2 != 0:
        return Verdict(ok=False, checked=True, score=0.0)
    if parity == "odd" and value % 2 == 0:
        return Verdict(ok=False, checked=True, score=0.0)

    return Verdict(ok=True, checked=True, score=1.0)


def choose(arm: str, first: str, second: str, task: RevisionTask) -> tuple[str, str]:
    """Which of the two already-generated answers this arm keeps, and why."""
    if arm == SINGLE_PASS:
        return first, "fixed_policy_keep_first"
    if arm == ALWAYS_REVISE:
        return second, "fixed_policy_keep_second"

    decision = decide_revision(
        verify(first, task),
        verify(second, task),
        has_incumbent=True,
    )
    kept = second if decision.accept else first
    return kept, f"gate:{decision.reason}"


def run(
    responder,
    tasks: list[RevisionTask],
) -> tuple[AttemptLedger, dict[str, Any]]:
    """Generate twice per task, then score all three arms over those generations."""
    ledger = AttemptLedger()
    transitions = {"gate_kept_first": 0, "gate_kept_second": 0}
    pass_scores = {"first": 0.0, "second": 0.0}

    for task in tasks:
        started = time.monotonic()
        try:
            first = str(responder(task, attempt=1, previous=None))
            second = str(responder(task, attempt=2, previous=first))
        except Exception as exc:  # noqa: BLE001 — a crashed task fails every arm equally
            for arm in ARMS:
                ledger.record(
                    Attempt(
                        task_id=task.task_id,
                        condition=arm,
                        outcome="crash",
                        score=0.0,
                        lane=arm,
                        detail={"error": f"{type(exc).__name__}: {exc}"},
                    )
                )
            continue
        elapsed = round(time.monotonic() - started, 3)

        pass_scores["first"] += grade(first, task)
        pass_scores["second"] += grade(second, task)

        for arm in ARMS:
            kept, why = choose(arm, first, second, task)
            score = grade(kept, task)
            if arm == GATED:
                key = "gate_kept_second" if kept == second and first != second else "gate_kept_first"
                transitions[key] += 1
            ledger.record(
                Attempt(
                    task_id=task.task_id,
                    condition=arm,
                    outcome="success" if score > 0 else "failure",
                    score=score,
                    lane=arm,
                    # Shared generations: the elapsed time is the cost of the
                    # PAIR, charged identically to each arm. Reporting it
                    # per-arm would imply three separate runs.
                    detail={"kept_via": why, "pair_elapsed_s": elapsed},
                )
            )

    n = len(tasks) or 1
    return ledger, {
        "raw_pass_1_rate": round(pass_scores["first"] / n, 4),
        "raw_pass_2_rate": round(pass_scores["second"] / n, 4),
        "gate_decisions": transitions,
    }


def deterministic_responder(task: RevisionTask, *, attempt: int, previous: str | None) -> str:
    """Harness proof, not evidence.

    Encodes the ONLY regime in which a gate can beat both fixed policies:
    some tasks improve on the second pass, others regress. A responder where
    pass 2 were uniformly better would make `always_revise` unbeatable and the
    experiment vacuous — so this one deliberately regresses on a subset, and
    the regressions are verifier-detectable while the improvements are not
    always. If the gate cannot win here it cannot win anywhere.
    """
    behaviour = task.detail.get("deterministic_behaviour", "stable")
    if attempt == 1:
        return task.detail.get("first_answer", task.answer_key)
    if behaviour == "improves":
        return task.answer_key
    if behaviour == "regresses":
        return task.detail.get("bad_second_answer", "not a number")
    return previous or task.answer_key


def _constraint_task(
    task_id: str,
    *,
    index: int,
    behaviour: str,
    first_answer_offset: int,
) -> RevisionTask:
    """One constraint-satisfaction item: find n in [lo,hi] divisible by d.

    Verification is strictly cheaper than solving — check membership and one
    modulo, versus search the interval — which is what makes the verifier
    honest rather than a restatement of the answer key.
    """
    divisor = 7 + (index % 3)
    lo = 40 + index * 11
    answer = lo + (-lo % divisor)  # smallest multiple of `divisor` at or above lo
    hi = answer + divisor - 1  # unique solution in [lo, hi]
    return RevisionTask(
        task_id=task_id,
        prompt=(
            f"Give the number between {lo} and {hi} inclusive that is divisible "
            f"by {divisor}. Answer with the number alone."
        ),
        answer_key=str(answer),
        detail={
            "min": lo,
            "max": hi,
            "divisible_by": divisor,
            "deterministic_behaviour": behaviour,
            # A wrong answer that still sits inside the interval but breaks the
            # divisibility constraint: the verifier catches it, the grader
            # rejects it, and neither had to know the other's reasoning.
            "first_answer": str(answer + first_answer_offset)
            if first_answer_offset
            else str(answer),
            "bad_second_answer": str(answer + 1)
            if (answer + 1) % divisor
            else str(answer + 2),
        },
    )


def battery(scale: int = 1) -> list[RevisionTask]:
    """Four regimes. The proportions are the design, not a tuning knob.

    An all-improves battery hands the win to always_revise; an all-regresses
    battery hands it to single_pass. A gate is only worth having when both
    occur, so both occur here — and so does a third case that matters more
    than either.

    `invisible_improves` is a second pass that is genuinely better and that the
    verifier CANNOT see: both answers satisfy every stated constraint, and only
    the answer key separates them. The gate must tie single_pass there, because
    a rule that adopted the challenger on no evidence would be guessing. Those
    tasks stay in the denominator deliberately — dropping them would report the
    gate only on the tasks it is equipped to win, which is how a component gets
    credited with a competence it does not have.
    """
    n_improves = 12 * scale
    n_regresses = 12 * scale
    n_stable = 8 * scale
    n_invisible = 8 * scale

    tasks: list[RevisionTask] = []
    for i in range(n_improves):
        # Pass 1 breaks the divisibility constraint; pass 2 satisfies it.
        tasks.append(
            _constraint_task(
                f"improves_{i:02d}", index=i, behaviour="improves", first_answer_offset=1
            )
        )
    for i in range(n_regresses):
        # Pass 1 is correct; pass 2 breaks the constraint (a real regression).
        tasks.append(
            _constraint_task(
                f"regresses_{i:02d}",
                index=i + n_improves,
                behaviour="regresses",
                first_answer_offset=0,
            )
        )
    for i in range(n_stable):
        tasks.append(
            _constraint_task(
                f"stable_{i:02d}",
                index=i + n_improves + n_regresses,
                behaviour="stable",
                first_answer_offset=0,
            )
        )
    for i in range(n_invisible):
        # Both passes satisfy every constraint the verifier can check; only the
        # answer key knows pass 2 is the right one. The gate is blind here by
        # design, and that blindness belongs in the reported number.
        index = i + n_improves + n_regresses + n_stable
        task = _constraint_task(
            f"invisible_improves_{i:02d}",
            index=index,
            behaviour="improves",
            first_answer_offset=0,
        )
        divisor = int(task.detail["divisible_by"])
        answer = int(task.answer_key)
        task.detail["first_answer"] = str(answer - divisor)  # valid form, out of range only by luck
        task.detail["min"] = answer - divisor  # widen so the wrong answer VERIFIES clean
        tasks.append(task)
    return tasks


def _summarise_arms(ledger: AttemptLedger) -> dict[str, dict[str, Any]]:
    return {arm: ledger.summary(arm) for arm in ARMS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--responder", choices=("deterministic", "mlx"), default="deterministic")
    parser.add_argument("--model", default="", help="model path/id for the mlx responder")
    parser.add_argument("--max-output-tokens", type=int, default=24)
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help=(
            "multiplies every regime, preserving their proportions. Raise it when the "
            "paired interval spans zero — that is the run telling you it cannot resolve "
            "the delta it just printed, not an invitation to report the delta anyway."
        ),
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    if args.scale < 1:
        print("--scale must be at least 1", file=sys.stderr)
        return 2

    if args.responder == "mlx":
        if not args.model:
            print("--responder mlx requires --model", file=sys.stderr)
            return 2
        from tools.revision_ablation_mlx import make_revision_responder

        responder = make_revision_responder(
            model_id=args.model,
            max_output_tokens=args.max_output_tokens,
        )
        model_id = args.model
    else:
        responder = deterministic_responder
        model_id = "deterministic:no-model"

    tasks = battery(args.scale)
    try:
        ledger, raw = run(responder, tasks)
    finally:
        close = getattr(responder, "close", None)
        if callable(close):
            close()

    summaries = _summarise_arms(ledger)
    gated_rate = summaries[GATED]["success_rate"]
    revise_rate = summaries[ALWAYS_REVISE]["success_rate"]
    single_rate = summaries[SINGLE_PASS]["success_rate"]

    # The gate earns its cost only by beating BOTH fixed policies, and only by
    # a margin this many tasks can actually resolve. Beating one is free: any
    # rule that always keeps the other one does that.
    binding = ALWAYS_REVISE if revise_rate >= single_rate else SINGLE_PASS
    vs_revise = paired_separation(ledger, GATED, ALWAYS_REVISE)
    vs_single = paired_separation(ledger, GATED, SINGLE_PASS)
    vs_binding = vs_revise if binding == ALWAYS_REVISE else vs_single
    beats_both = (
        gated_rate > revise_rate
        and gated_rate > single_rate
        and vs_revise.get("verdict") == "treatment_better"
        and vs_single.get("verdict") == "treatment_better"
    )

    claim = LesionClaim(
        condition="gated_revision_vs_fixed_policies",
        subsystem="core.brain.reasoning_revision_gate",
        metric_name="verifiable_task_success_rate",
        delta=round(gated_rate - max(revise_rate, single_rate), 4),
        metric_has_other_producers=True,
        metric_is_task_success=True,
        # Measured, not asserted: the fixed-policy arms solve some of these
        # tasks outright, which is visible in their own success rates. If both
        # sat at zero the battery would be unsolvable without the gate and the
        # result would be mechanistic.
        tasks_solvable_without_component=max(revise_rate, single_rate) > 0.0,
    )

    is_evidence = args.responder == "mlx"
    report = {
        "schema": "aura.revision_scorecard.v1",
        "generated_at_unix": time.time(),
        "responder": args.responder,
        "model": model_id,
        "is_evidence_about_aura": is_evidence,
        "caveat": (
            ""
            if is_evidence
            else "DETERMINISTIC RESPONDER, NO MODEL. Exercises the harness only — it shows the "
            "gate CAN separate from both fixed policies when improvement and regression both "
            "occur. It is not evidence that Aura's second passes behave this way."
        ),
        "budget_note": (
            "No budget parity check is required or performed: all three arms are scored over "
            "the SAME two generations per task, so token spend, wall clock and model are "
            "identical by construction rather than by matching."
        ),
        "scope": {
            "subsystem": "revision/selection policy only — NOT the cognitive layer as a whole",
            "task_family": "verifiable short-answer arithmetic with an exact answer key",
            "verifier": (
                "a necessary-not-sufficient well-formedness check, deliberately NOT the answer "
                "key: a gate consulting the grader would be omniscient and the result circular"
            ),
            "regime_dependence": (
                "the gate can only beat both fixed policies where second passes help on some "
                "tasks and hurt on others. Where revision is uniformly good or uniformly bad, "
                "the matching fixed policy ties it and the gate buys nothing."
            ),
        },
        "conditions": summaries,
        "raw_passes": raw,
        "comparisons": {
            "gated_vs_always_revise": round(gated_rate - revise_rate, 4),
            "gated_vs_single_pass": round(gated_rate - single_rate, 4),
            "binding_baseline": binding,
            "gate_beats_both_baselines": beats_both,
        },
        "separation": {
            "vs_always_revise": vs_revise,
            "vs_single_pass": vs_single,
            "vs_binding_baseline": vs_binding,
        },
        "claims": [claim.to_dict()],
        "inference": summarise([claim]),
        "attempts": ledger.to_dict(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"{'arm':<20}{'success':<12}{'clean':<12}attempts")
    print("-" * 56)
    for arm in ARMS:
        s = summaries[arm]
        print(
            f"{arm:<20}{s['success_rate']:<12.3f}{s['clean_success_rate']:<12.3f}{s['attempts']}"
        )
    print(
        f"\nraw pass 1 = {raw['raw_pass_1_rate']:.3f}   raw pass 2 = {raw['raw_pass_2_rate']:.3f}"
        "   (same generations every arm scores)"
    )
    print(
        f"gated - always_revise = {report['comparisons']['gated_vs_always_revise']:+.4f}\n"
        f"gated - single_pass   = {report['comparisons']['gated_vs_single_pass']:+.4f}"
    )
    print(
        f"binding baseline: {binding}  "
        f"(paired 95% CI {vs_binding.get('ci95')}, n={vs_binding.get('n_tasks')}, "
        f"verdict={vs_binding.get('verdict')})"
    )
    if beats_both:
        print(
            "VERDICT: the gate beats BOTH fixed policies over identical compute,\n"
            "  by a margin this many tasks can resolve."
        )
    else:
        print(
            "VERDICT: the gate does NOT beat both fixed policies. On this battery the\n"
            "  machinery is not earning its cost — a fixed policy matches or exceeds it,\n"
            "  or the margin is inside what this many tasks can resolve."
        )
        for label, sep in (("vs always_revise", vs_revise), ("vs single_pass", vs_single)):
            if sep.get("reason"):
                print(f"  {label}: {sep['reason']}")
    print(f"  inference class: {claim.inference_class} (measured)")
    if claim.inference_class is not InferenceClass.CAPABILITY:
        print("  NOT a capability result: the baselines could not solve these tasks at all.")
    if not is_evidence:
        print("\n" + report["caveat"])
    print(f"\nscorecard: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
