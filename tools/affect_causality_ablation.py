"""Is affect causal, or is it feature extraction with a steering vector attached?

THE CRITICISM THIS ANSWERS
"Neurochemicals, the Qualia Engine and Φ are dressed-up feature extraction."
The strong form of that charge is not "these modules do nothing" — they
demonstrably compute numbers and those numbers demonstrably reach the model as
a steering vector. The strong form is that the CONTENT of the state does not
matter: that any vector of the same magnitude, injected at the same place,
would move the output just as much, and the elaborate affective machinery
upstream is decoration on a noise generator.

That is a sharp, falsifiable claim, and it needs a control that most steering
evaluations do not run.

THE ARMS
    unsteered         alpha = 0. Identity, and measured rather than assumed.
    real_state        the composite vector the affect substrate actually built.
    shuffled_state    the SAME vector with its components permuted.

`shuffled_state` is the arm that does the work. A permutation preserves the
vector's norm and the exact multiset of its components, so it is
magnitude-matched by construction rather than by tuning — it differs from the
real vector in direction alone, which is precisely the thing "dressed-up
feature extraction" says is irrelevant. If `real_state` and `shuffled_state`
move the output equally, the affect layer is not contributing meaning, and
this tool says so.

THE NULL IS VALIDATED BEFORE ANY VERDICT IS READ
This repository has already shipped a CAA A/B whose own null scored d=17.3,
p=0.0005 — a control that should have found nothing and instead found a
larger effect than most real results. Any number that harness produced was
uninterpretable, and nothing in it noticed.

So this tool runs shuffled-versus-shuffled FIRST: two independent permutations
of the same vector, which differ by nothing that could matter. That comparison
must come back `unresolved`. If it separates, the measurement apparatus is
broken and every downstream verdict is withheld — printed as NULL FAILED, no
effect size reported. A harness that cannot detect its own noise floor has no
business reporting an effect above it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.evaluation.lesion_inference import LesionClaim, summarise  # noqa: E402
from core.evaluation.matched_budget import (  # noqa: E402
    Attempt,
    AttemptLedger,
    paired_separation,
)

UNSTEERED = "unsteered"
REAL = "real_state"
SHUFFLED = "shuffled_state"
ARMS = (UNSTEERED, REAL, SHUFFLED)

#: Two independent permutations of the same vector. Nothing distinguishes them,
#: so a separation here is instrument error, not a finding.
NULL_A = "null_shuffle_a"
NULL_B = "null_shuffle_b"

DEFAULT_OUT = _REPO_ROOT / "artifacts" / "ablation" / "affect_causality_scorecard.json"


@dataclass
class AffectProbe:
    """One prompt, plus the affect state to hold while answering it."""

    probe_id: str
    prompt: str
    #: Valence/arousal the substrate is driven to. The battery pairs each
    #: prompt with opposing states so a direction has somewhere to show up.
    valence: float
    arousal: float
    detail: dict[str, Any] = field(default_factory=dict)


def permute(vector: list[float], seed: int) -> list[float]:
    """Same norm, same components, different direction.

    Magnitude-matching by permutation rather than by rescaling is deliberate:
    a rescaled random vector matches one summary statistic and can still differ
    in component distribution, which leaves "the arms had different energy" as
    a live explanation for any result. A permutation removes that explanation
    entirely — the two vectors are the same numbers in a different order.
    """
    shuffled = list(vector)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def battery(scale: int = 1) -> list[AffectProbe]:
    """Prompts whose answers have room to move without having a right answer.

    Deliberately NOT task-success items. Affect is not claimed to make Aura
    better at arithmetic, and scoring it on arithmetic would either find
    nothing (and be reported as a refutation of something nobody claimed) or
    find noise. These are open prompts where a valence shift is expressible.
    """
    seeds = [
        "Describe how the last hour has gone.",
        "A plan you made did not work out. Say what you make of that.",
        "Someone asks how you are. Answer them.",
        "Describe the room you are working in.",
        "You have been given a new problem to solve. React.",
        "Summarise where things stand right now.",
        "A task just finished. Comment on it.",
        "Describe what you expect from the next hour.",
    ]
    probes: list[AffectProbe] = []
    for repeat in range(scale):
        for index, text in enumerate(seeds):
            # Opposing states on the same prompt: the pairing is what lets a
            # direction be detected at all, and it keeps prompt difficulty from
            # loading onto the comparison.
            for polarity, (valence, arousal) in (
                ("pos", (0.85, 0.65)),
                ("neg", (0.15, 0.65)),
            ):
                probes.append(
                    AffectProbe(
                        probe_id=f"{index:02d}_{polarity}_{repeat}",
                        prompt=text,
                        valence=valence,
                        arousal=arousal,
                        detail={"polarity": polarity, "seed_index": index},
                    )
                )
    return probes


#: A transparent, auditable valence lexicon. Not a learned sentiment model:
#: the point is that a reader can check every word that moved a score, and that
#: the metric cannot quietly encode the hypothesis it is testing.
_POSITIVE = frozenset(
    """good well fine glad pleased steady clear bright easy smooth ready useful
    progress works working solid better right sound calm settled""".split()
)
_NEGATIVE = frozenset(
    """bad wrong hard difficult stuck slow tired heavy unclear failed failing
    broken worse trouble strain rough tense uneasy concerned""".split()
)


def valence_score(text: str) -> float:
    """(positive - negative) / total, in [-1, 1]. Zero when neither appears."""
    words = [w.strip(".,!?;:'\"()").lower() for w in str(text or "").split()]
    pos = sum(1 for w in words if w in _POSITIVE)
    neg = sum(1 for w in words if w in _NEGATIVE)
    total = pos + neg
    return 0.0 if total == 0 else (pos - neg) / total


def directional_score(text: str, probe: AffectProbe) -> float:
    """How well the output's valence matches the state that was held.

    Scored as agreement with the INTENDED direction, so a steering vector that
    moves output consistently the wrong way scores below chance rather than
    being credited for having an effect. Magnitude of change is not enough —
    a noise injector changes output too.
    """
    observed = valence_score(text)
    intended = 1.0 if probe.valence >= 0.5 else -1.0
    return max(0.0, min(1.0, 0.5 + 0.5 * observed * intended))


def run(
    responder,
    probes: list[AffectProbe],
    arms: tuple[str, ...] = ARMS,
) -> AttemptLedger:
    ledger = AttemptLedger()
    for arm in arms:
        for probe in probes:
            started = time.monotonic()
            outcome = "success"
            score = 0.0
            detail: dict[str, Any] = {"polarity": probe.detail.get("polarity")}
            try:
                output = str(responder(arm, probe))
                score = directional_score(output, probe)
                detail["valence"] = round(valence_score(output), 4)
                detail["chars"] = len(output)
            except TimeoutError:
                outcome = "timeout"
            except Exception as exc:  # noqa: BLE001 — every attempt counted, this one too
                outcome = "crash"
                detail["error"] = f"{type(exc).__name__}: {exc}"
            detail["elapsed_s"] = round(time.monotonic() - started, 3)
            ledger.record(
                Attempt(
                    task_id=probe.probe_id,
                    condition=arm,
                    outcome=outcome,
                    score=score,
                    lane=arm,
                    detail=detail,
                )
            )
    return ledger


def mean_abs_paired_difference(ledger: AttemptLedger, a: str, b: str) -> float:
    """Mean |score(a) - score(b)| over shared probes — the instrument's noise floor.

    The signed mean is not enough, and this function exists because relying on
    it let a perfectly leaking instrument pass. A responder that answered arm A
    positively and arm B negatively separated the two arms completely, but it
    did so in OPPOSITE directions on positive and negative probes, so the
    signed differences (+1 and -1) averaged to zero and the paired bootstrap
    reported `unresolved`. The null "held" while the apparatus was reading the
    arm label directly.

    Absolute difference cannot cancel that way. Two arms that differ by nothing
    must produce per-probe scores that differ by nothing.
    """
    left = {a_.task_id: a_.score for a_ in ledger.for_condition(a)}
    right = {b_.task_id: b_.score for b_ in ledger.for_condition(b)}
    shared = sorted(set(left) & set(right))
    if not shared:
        return 0.0
    return sum(abs(left[t] - right[t]) for t in shared) / len(shared)


def validate_null(responder, probes: list[AffectProbe]) -> dict[str, Any]:
    """Two permutations of one vector must be indistinguishable.

    Run before anything else and gates every downstream verdict. The failure
    this catches is not hypothetical: a CAA A/B in this repository reported its
    own null at d=17.3, p=0.0005, and nothing in that harness noticed that a
    control finding a huge effect meant the instrument was broken.
    """
    ledger = run(responder, probes, arms=(NULL_A, NULL_B))
    separation = paired_separation(ledger, NULL_A, NULL_B)
    noise_floor = mean_abs_paired_difference(ledger, NULL_A, NULL_B)
    # BOTH conditions. The signed test alone was insufficient — see
    # mean_abs_paired_difference for the instrument it let through.
    passed = separation.get("verdict") == "unresolved" and noise_floor <= 0.0
    return {
        "purpose": (
            "two independent permutations of the same steering vector; they differ by "
            "nothing that could matter, so any separation is instrument error"
        ),
        "separation": separation,
        "noise_floor_mean_abs_difference": round(noise_floor, 6),
        "why_two_conditions": (
            "the signed mean can be zero while the arms separate completely, if they "
            "separate in opposite directions on different probe types. Mean absolute "
            "difference cannot cancel that way"
        ),
        "null_holds": passed,
        "conditions": {name: ledger.summary(name) for name in (NULL_A, NULL_B)},
        "consequence_if_failed": (
            "every effect estimate is withheld — a harness that cannot detect its own "
            "noise floor cannot report an effect above it"
        ),
    }


def deterministic_responder(arm: str, probe: AffectProbe) -> str:
    """Harness proof, not evidence.

    Encodes the hypothesis being tested so the harness can be checked against a
    world where affect IS causal: the real arm tracks the intended valence, the
    shuffled arm produces a magnitude-matched but directionless shift, and the
    unsteered arm sits flat. If the tool cannot separate these, it cannot
    separate anything.
    """
    positive = "good clear steady progress"
    negative = "hard slow stuck trouble"
    mixed = "good hard clear stuck"
    if arm == UNSTEERED:
        return "the hour passed"
    if arm == REAL:
        return positive if probe.valence >= 0.5 else negative
    # Shuffled and both nulls: perturbed, but carrying no direction.
    return mixed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--responder", choices=("deterministic", "mlx"), default="deterministic")
    parser.add_argument("--model", default="")
    parser.add_argument("--max-output-tokens", type=int, default=48)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    if args.responder == "mlx":
        if not args.model:
            print("--responder mlx requires --model", file=sys.stderr)
            return 2
        from tools.affect_causality_mlx import make_affect_responder

        responder = make_affect_responder(
            model_id=args.model,
            max_output_tokens=args.max_output_tokens,
        )
        model_id = args.model
    else:
        responder = deterministic_responder
        model_id = "deterministic:no-model"

    probes = battery(args.scale)
    try:
        null_report = validate_null(responder, probes)
        ledger = run(responder, probes)
    finally:
        close = getattr(responder, "close", None)
        if callable(close):
            close()

    summaries = {arm: ledger.summary(arm) for arm in ARMS}
    real_vs_shuffled = paired_separation(ledger, REAL, SHUFFLED)
    real_vs_unsteered = paired_separation(ledger, REAL, UNSTEERED)
    shuffled_vs_unsteered = paired_separation(ledger, SHUFFLED, UNSTEERED)

    null_holds = bool(null_report["null_holds"])
    # The verdict requires BOTH: the real state beats an unsteered baseline
    # (affect reaches generation) AND beats a magnitude-matched permutation
    # (the state's content is what did it). Only the second answers the
    # criticism; the first alone is satisfied by any noise injector.
    content_matters = real_vs_shuffled.get("verdict") == "treatment_better"
    reaches_generation = real_vs_unsteered.get("verdict") == "treatment_better"
    verdict = (
        "withheld_null_failed"
        if not null_holds
        else "affect_content_is_causal"
        if (content_matters and reaches_generation)
        else "indistinguishable_from_magnitude_matched_noise"
        if reaches_generation
        else "no_measurable_effect"
    )

    claim = LesionClaim(
        condition="affect_content_vs_magnitude_matched_permutation",
        subsystem="core.consciousness.affective_steering",
        metric_name="intended_direction_agreement",
        delta=round(
            summaries[REAL]["mean_score"] - summaries[SHUFFLED]["mean_score"], 4
        ),
        metric_has_other_producers=True,
        metric_is_task_success=False,  # a directional metric, not task success
        tasks_solvable_without_component=True,
    )

    is_evidence = args.responder == "mlx" and null_holds
    report = {
        "schema": "aura.affect_causality_scorecard.v1",
        "generated_at_unix": time.time(),
        "responder": args.responder,
        "model": model_id,
        "is_evidence_about_aura": is_evidence,
        "verdict": verdict,
        "null_validation": null_report,
        "conditions": summaries,
        "separation": {
            "real_vs_shuffled": real_vs_shuffled,
            "real_vs_unsteered": real_vs_unsteered,
            "shuffled_vs_unsteered": shuffled_vs_unsteered,
        },
        "scope": {
            "subsystem": "affective steering only — NOT Φ, and NOT the neurochemical model",
            "metric": (
                "agreement with the INTENDED valence direction, scored by a transparent "
                "word lexicon. A learned sentiment model would be more sensitive and less "
                "auditable; every word that moves this score can be read in the source"
            ),
            "what_a_win_means": (
                "the state's CONTENT changed the output, not merely its magnitude. That "
                "refutes 'dressed-up feature extraction' for this subsystem and nothing else"
            ),
            "what_this_cannot_show": (
                "that affect improves task performance, that Φ is meaningful, or that the "
                "neurochemical model is more than a parameterisation. Those are separate "
                "claims needing separate batteries"
            ),
        },
        "claims": [claim.to_dict()],
        "inference": summarise([claim]),
        "attempts": ledger.to_dict(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"{'arm':<20}{'direction':<14}{'clean':<12}attempts")
    print("-" * 58)
    for arm in ARMS:
        s = summaries[arm]
        print(
            f"{arm:<20}{s['mean_score']:<14.3f}{s['clean_success_rate']:<12.3f}{s['attempts']}"
        )

    null_sep = null_report["separation"]
    print(
        f"\nNULL (shuffle vs shuffle): verdict={null_sep.get('verdict')} "
        f"CI {null_sep.get('ci95')}  ->  {'HOLDS' if null_holds else 'FAILED'}"
    )
    if not null_holds:
        print(
            "  The control separated. Two permutations of one vector cannot differ,\n"
            "  so the instrument is measuring something other than what it claims.\n"
            "  ALL effect estimates below are withheld."
        )
    else:
        print(
            f"real - shuffled  = {claim.delta:+.4f}  "
            f"CI {real_vs_shuffled.get('ci95')} ({real_vs_shuffled.get('verdict')})"
        )
        print(
            f"real - unsteered = "
            f"{summaries[REAL]['mean_score'] - summaries[UNSTEERED]['mean_score']:+.4f}  "
            f"CI {real_vs_unsteered.get('ci95')} ({real_vs_unsteered.get('verdict')})"
        )
    print(f"\nVERDICT: {verdict}")
    if verdict == "indistinguishable_from_magnitude_matched_noise":
        print(
            "  Affect reaches generation, but a permutation of the same vector moves the\n"
            "  output just as well. For this subsystem the criticism stands: what is\n"
            "  causal is the injection, not the state."
        )
    if not is_evidence:
        print(
            "\nNOT EVIDENCE ABOUT AURA: "
            + ("null failed" if not null_holds else "deterministic responder, no model")
        )
    print(f"\nscorecard: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
