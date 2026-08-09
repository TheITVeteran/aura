#!/usr/bin/env python3
"""A/B the exclusion policy against i.i.d. best-of-N on a real model.

Same model, same draw budget, same verifier. The only difference is whether
a refuted answer is excluded from later draws.

    python tools/run_exclusion_ab.py --draws 6 --tasks 12

The verifier is SOUND and NON-ORACLE: it recomputes arithmetic and checks
declared formats. It never sees the gold answer, so a refutation is a real
deterministic contradiction rather than "not the answer we wanted". Tasks it
cannot decide contribute UNDECIDED, which excludes nothing — those tasks
measure nothing about the policy and are reported separately rather than
being quietly folded into the pass rate.

Exit 0 if exclusion wins, 1 if it does not, 2 if the run could not decide.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.commitment_ratchet import (  # noqa: E402
    CommitmentRatchet,
    Constraint,
    ConstraintKind,
)
from core.brain.llm.latent_cortex.sequential_exclusion import (  # noqa: E402
    DrawOutcome,
)

DEFAULT_MODEL = "models/Qwen2.5-1.5B-Instruct-4bit"

#: Difficulty BANDS, because guessing the difficulty failed twice.
#:
#: Two-digit multiplication: the model solved every item on draw 1. Zero
#: refutations, nothing to exclude — a ceiling, which looks exactly like a
#: null unless the harness says INCONCLUSIVE, which it now does.
#:
#: Three-digit x three-digit: the model solved 1 of 24 in BOTH arms. That is
#: the support premise failing — p* is near zero, the answer is outside the
#: model's reach, and no search policy over an unreachable answer helps.
#:
#: A search improvement can only show in the band between those. The band is
#: chosen by MEASURING the i.i.d. rate rather than by picking numbers that
#: feel right, because the two failures above were both from picking numbers
#: that felt right.
_BANDS: dict[str, tuple[int, int, int, int]] = {
    "easy": (11, 99, 2, 9),
    "moderate": (11, 99, 11, 29),
    "hard": (101, 499, 11, 49),
    "extreme": (113, 987, 113, 987),
}


def build_tasks(band: str, count: int, seed: int) -> tuple[tuple[str, str, dict[str, Any]], ...]:
    import random

    lo_a, hi_a, lo_b, hi_b = _BANDS[band]
    rng = random.Random(seed)
    rows: list[tuple[str, str, dict[str, Any]]] = []
    seen: set[tuple[int, int]] = set()
    while len(rows) < count:
        a = rng.randint(lo_a, hi_a)
        b = rng.randint(lo_b, hi_b)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        rows.append(
            (
                f"What is {a} * {b}? Reply with only the number.",
                str(a * b),
                {"arith": (a, "*", b)},
            )
        )
    return tuple(rows)


_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _first_number(text: str) -> float | None:
    match = _NUM_RE.search(str(text or ""))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _arith_truth(spec: tuple[int, str, int]) -> float:
    left, op, right = spec
    return {
        "*": left * right,
        "+": left + right,
        "-": left - right,
        "/": left / right,
    }[op]


def make_verifier(spec: dict[str, Any]):
    """Sound, non-oracle: it recomputes, it does not consult the gold."""

    def _verify(objective: str, candidate: str) -> tuple[DrawOutcome, str]:
        value = _first_number(candidate)
        if value is None:
            return DrawOutcome.UNDECIDED, "no number in the answer"
        truth = _arith_truth(spec["arith"])
        if abs(value - truth) < 1e-9:
            return DrawOutcome.ACCEPTED, f"recomputed {truth:g}"
        return DrawOutcome.REFUTED, f"recomputed {truth:g}, answer said {value:g}"

    return _verify


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def run_task(
    model: Any,
    tokenizer: Any,
    prompt: str,
    spec: dict[str, Any],
    *,
    draws: int,
    temperature: float,
    exclude: bool,
    mode: str = "prompt",
) -> dict[str, Any]:
    """One task under one policy.

    ``mode`` decides HOW the restriction to A \\ R is implemented, and the
    distinction turned out to be the whole result:

      prompt     the excluded answers are described in the context. This is
                 what the first two A/B runs measured, and it LOST — twice.
                 It is not the theorem: it does not remove mass, it adds
                 tokens, and for numeric answers those tokens anchor the very
                 values they are meant to exclude. Compliance 0.43 then 0.64,
                 coverage negative in both.

      rejection  draw from the UNCONDITIONED model and discard any draw that
                 lands on an already-refuted answer. This is the theorem
                 exactly: samples from p restricted to A \\ R, renormalised,
                 with no perturbation of p itself. Rejected draws are charged
                 to the budget, so the comparison stays honest.
    """
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    verify = make_verifier(spec)
    ratchet = CommitmentRatchet()
    seen: list[str] = []
    solved = False
    solved_at: int | None = None
    noncompliant = 0

    rejected = 0
    verified_calls = 0
    max_generations = draws * 4  # bounded: rejection must not spin forever
    generations = 0
    while verified_calls < draws and generations < max_generations:
        generations += 1
        index = verified_calls
        use_prompt_block = exclude and mode == "prompt"
        block = (
            ratchet.conditioning_block(include_exclusions=True)
            if use_prompt_block
            else ""
        )
        # AFTER the question, not before it. A prefix competes with the
        # instruction for the model's attention; the constraint has to be the
        # last thing it reads before generating.
        content = f"{prompt}\n\n{block}" if block else prompt
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
        )
        completion = str(
            generate(
                model,
                tokenizer,
                prompt=text,
                max_tokens=32,
                sampler=make_sampler(temp=temperature),
                verbose=False,
            )
            or ""
        )
        answer = completion.strip().split("\n")[0][:120]
        normalised = _normalise(answer)
        already_excluded = exclude and any(
            _normalise(tooth.subject) == normalised for tooth in ratchet.teeth
        )
        if already_excluded:
            if mode == "rejection":
                # The draw landed in R. Discard it WITHOUT spending a
                # verifier call, and redraw.
                #
                # Which unit the budget is denominated in decides the whole
                # result, and the first version got it wrong. Charging a
                # rejected draw against the draw budget made rejection lose,
                # correctly: a duplicate of an already-refuted answer costs
                # nothing to verify here (the verifier is arithmetic), so
                # removing it saved nothing and cost a generation.
                #
                # The theorem's budget is VERIFIER CALLS — the expensive,
                # bounded resource in any real deployment, where verifying
                # means running a test, calling a tool, or paying a model.
                # Both arms now get the same number of verifier calls, and
                # rejected draws are counted separately so the generation
                # overhead stays visible rather than hidden.
                rejected += 1
                continue
            noncompliant += 1
        seen.append(normalised)

        verified_calls += 1
        outcome, _detail = verify(prompt, answer)
        if outcome is DrawOutcome.ACCEPTED:
            solved = True
            solved_at = index + 1
            break
        if outcome is DrawOutcome.REFUTED and exclude:
            ratchet.commit(
                Constraint(
                    kind=ConstraintKind.EXCLUDES,
                    subject=answer,
                    source="ab",
                    step=index,
                )
            )

    return {
        "solved": solved,
        "solved_at": solved_at,
        "verifier_calls": verified_calls,
        "generations": generations,
        "draws_used": len(seen),
        "distinct": len(set(seen)),
        "noncompliant": noncompliant,
        "rejected": rejected,
        "commitments": ratchet.turns,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--draws", type=int, default=6)
    parser.add_argument("--tasks", type=int, default=24)
    parser.add_argument("--band", default="moderate", choices=sorted(_BANDS))
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        from mlx_lm import load
    except ImportError:
        print(json.dumps({"error": "mlx_lm unavailable"}, indent=2))
        return 2

    started = time.time()
    path = args.model
    model, tokenizer = load(
        str(REPO_ROOT / path) if not Path(path).is_absolute() else path
    )

    tasks = build_tasks(args.band, max(1, args.tasks), args.seed)
    arms: dict[str, list[dict[str, Any]]] = {
        "iid": [], "exclusion": [], "rejection": []
    }
    for prompt, _gold, spec in tasks:
        for name, exclude, mode in (
            ("iid", False, "prompt"),
            ("exclusion", True, "prompt"),
            ("rejection", True, "rejection"),
        ):
            arms[name].append(
                run_task(
                    model, tokenizer, prompt, spec,
                    draws=args.draws, temperature=args.temperature,
                    exclude=exclude, mode=mode,
                )
            )

    def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        solved = sum(1 for row in rows if row["solved"])
        return {
            "tasks": len(rows),
            "solved": solved,
            "solve_rate": round(solved / len(rows), 4) if rows else 0.0,
            "mean_distinct": round(
                sum(row["distinct"] for row in rows) / len(rows), 3
            ) if rows else 0.0,
            "mean_draws_used": round(
                sum(row["draws_used"] for row in rows) / len(rows), 3
            ) if rows else 0.0,
            "mean_verifier_calls": round(
                sum(row.get("verifier_calls", 0) for row in rows) / len(rows), 3
            ) if rows else 0.0,
            "mean_generations": round(
                sum(row.get("generations", 0) for row in rows) / len(rows), 3
            ) if rows else 0.0,
            "noncompliant_draws": sum(row["noncompliant"] for row in rows),
            "rejected_draws": sum(row.get("rejected", 0) for row in rows),
            "commitments": sum(row["commitments"] for row in rows),
        }

    iid = _summary(arms["iid"])
    excl = _summary(arms["exclusion"])
    reject = _summary(arms["rejection"])
    # The best faithful implementation is what the verdict is about. Prompt
    # conditioning is a proxy for the restriction; rejection sampling IS it.
    best_name, best = max(
        (("exclusion", excl), ("rejection", reject)),
        key=lambda pair: pair[1]["solve_rate"],
    )
    delta = best["solve_rate"] - iid["solve_rate"]
    coverage_delta = best["mean_distinct"] - iid["mean_distinct"]

    if iid["solve_rate"] >= 0.9 or iid["solve_rate"] <= 0.05:
        verdict = "INCONCLUSIVE"
        reading = (
            f"i.i.d. solve rate {iid['solve_rate']:.0%} in band {args.band!r}: "
            "the band is saturated (ceiling) or unreachable (floor), so no "
            "search policy could show a difference. This measured the task "
            "set, not the policy."
        )
    elif best["commitments"] == 0:
        verdict = "INCONCLUSIVE"
        reading = (
            "the exclusion arm committed nothing, so it ran as i.i.d.; this "
            "measured no policy"
        )
    elif delta >= 0.05:
        verdict = "EXCLUSION_WINS"
        reading = (
            f"{best_name}: solve rate {iid['solve_rate']:.0%} -> {best['solve_rate']:.0%} "
            f"(+{delta:.0%}) on the same model, same {args.draws}-draw budget, "
            f"same verifier; distinct answers examined "
            f"{iid['mean_distinct']:.1f} -> {best['mean_distinct']:.1f}"
        )
    elif delta <= -0.05:
        verdict = "EXCLUSION_LOSES"
        reading = (
            f"exclusion scored WORSE ({delta:+.0%}). The policy is not free: a "
            "wrong exclusion is irreversible, and conditioning costs context."
        )
    else:
        verdict = "NO_DIFFERENCE"
        reading = (
            f"solve rate moved {delta:+.0%}, inside noise for {len(tasks)} tasks. "
            f"Coverage moved {coverage_delta:+.1f} distinct answers, so the "
            "mechanism engaged without changing the outcome."
        )

    report = {
        "schema": "aura.rlc.exclusion_ab.v1",
        "model": args.model,
        "draws": args.draws,
        "temperature": args.temperature,
        "tasks": len(tasks),
        "band": args.band,
        "verifier": "sound_non_oracle_recomputation",
        "iid": iid,
        "exclusion_prompt_conditioned": excl,
        "exclusion_rejection_sampled": reject,
        "best_exclusion_arm": best_name,
        "solve_rate_delta": round(delta, 4),
        "coverage_delta": round(coverage_delta, 3),
        "prompt_conditioning_compliance": round(
            1.0 - (excl["noncompliant_draws"] / max(1, sum(r["draws_used"] for r in arms["exclusion"]))),
            4,
        ),
        "rejection_rate": round(
            reject["rejected_draws"] / max(1, sum(r["draws_used"] for r in arms["rejection"])),
            4,
        ),
        "verdict": verdict,
        "reading": reading,
        "duration_s": round(time.time() - started, 1),
    }
    payload = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if verdict == "EXCLUSION_WINS" else (2 if verdict == "INCONCLUSIVE" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
