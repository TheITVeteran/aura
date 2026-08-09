#!/usr/bin/env python3
"""Measure the premise the commitment search rests on, on a real model.

The whole argument for sequential exclusion is that independent sampling
from a peaked answer distribution wastes most of its budget re-deriving one
answer — that best-of-N is really best-of-D for some D much smaller than N.
That is an empirical claim about a specific model on specific tasks, and it
has never been measured here, because branch candidate texts are
worker-private and no campaign artifact retained them.

So measure it. Small model, small task set, bounded run, no effect on
anything else on the host.

    python tools/measure_candidate_peakedness.py --draws 8

What it reports:

  distinct/draws     how many of N draws were distinct answers. This is the
                     number that decides whether the exclusion argument is
                     worth anything at all;
  peakedness         Herfindahl concentration of the empirical answer mass;
  predicted gain     expected distinct candidates under exclusion vs i.i.d.,
                     computed from the SAME samples by the same function the
                     runtime uses.

A low peakedness here REFUTES the premise for this model and task set, and
that is a real outcome worth having. The script says so rather than
reporting a number and leaving the reader to decide what it meant.
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

from core.brain.llm.latent_cortex.sequential_exclusion import (  # noqa: E402
    estimate_mass_profile,
    expected_distinct_iid,
    peakedness,
    predict_distinct_advantage,
)

DEFAULT_MODEL = "models/Qwen2.5-1.5B-Instruct-4bit"

#: Short-answer tasks: the answer surface is small enough that "same answer"
#: is decidable without a judge, which is what makes distinctness measurable
#: at all. A long-form set would need a semantic equivalence model, and then
#: the measurement would be about that model.
TASKS: tuple[tuple[str, str], ...] = (
    ("A shop sells pens at 3 for $2. How much do 12 pens cost? Answer with just the amount.", "8"),
    ("What is 17 * 23? Answer with just the number.", "391"),
    ("If a train leaves at 2pm and takes 3 hours, what time does it arrive? Just the time.", "5pm"),
    ("How many days are in February in a leap year? Just the number.", "29"),
    ("What is the capital of Australia? Just the city name.", "Canberra"),
    ("A rectangle is 7 by 4. What is its area? Just the number.", "28"),
    ("What is 144 divided by 12? Just the number.", "12"),
    ("How many sides does a hexagon have? Just the number.", "6"),
)


def _normalize_answer(text: str) -> str:
    """Reduce a completion to its answer surface for distinctness counting.

    Deliberately aggressive: trailing punctuation, case and whitespace are
    not different answers. Anything this over-merges makes the measured
    peakedness HIGHER, which favours the hypothesis — so the script also
    reports the unmerged count, and the honest reading uses both.
    """
    body = str(text or "").strip().split("\n")[0]
    body = re.sub(r"[^\w\s.$/:-]", " ", body)
    return re.sub(r"\s+", " ", body).strip().lower()


def sample_candidates(
    model: Any, tokenizer: Any, prompt: str, *, draws: int, temperature: float
) -> list[str]:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    out: list[str] = []
    for index in range(draws):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        completion = generate(
            model,
            tokenizer,
            prompt=text,
            max_tokens=48,
            sampler=make_sampler(temp=temperature),
            verbose=False,
        )
        out.append(str(completion or ""))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--tasks", type=int, default=len(TASKS))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        from mlx_lm import load
    except ImportError:
        print(json.dumps({"error": "mlx_lm unavailable"}, indent=2))
        return 2

    started = time.time()
    model, tokenizer = load(str(REPO_ROOT / args.model) if not Path(args.model).is_absolute() else args.model)

    rows: list[dict[str, Any]] = []
    for prompt, expected in TASKS[: max(1, args.tasks)]:
        raw = sample_candidates(
            model, tokenizer, prompt, draws=args.draws, temperature=args.temperature
        )
        normalised = [_normalize_answer(text) for text in raw]
        masses = estimate_mass_profile(normalised)
        prediction = predict_distinct_advantage(normalised, draws=args.draws)
        rows.append(
            {
                "prompt": prompt[:80],
                "expected": expected,
                "draws": args.draws,
                "distinct_normalised": len(set(normalised)),
                "distinct_raw": len({text.strip() for text in raw}),
                "peakedness": round(peakedness(masses), 4),
                "top_mass": round(masses[0], 4) if masses else 0.0,
                "expected_distinct_iid": round(
                    expected_distinct_iid(masses, args.draws), 3
                ),
                "predicted_advantage": round(prediction.advantage, 3),
                "answers": sorted(set(normalised))[:6],
            }
        )

    total_draws = sum(row["draws"] for row in rows)
    total_distinct = sum(row["distinct_normalised"] for row in rows)
    mean_peak = sum(row["peakedness"] for row in rows) / len(rows)
    mean_predicted_iid = sum(row["expected_distinct_iid"] for row in rows) / len(rows)

    # The verdict is stated here, not left to the reader. A premise that is
    # only true if you squint at the table is not a premise.
    if mean_peak >= 0.40:
        verdict = "PREMISE_HOLDS"
        reading = (
            f"answers concentrate hard (mean peakedness {mean_peak:.2f}); i.i.d. "
            f"sampling examines ~{mean_predicted_iid:.1f} distinct answers per "
            f"{args.draws} draws, so exclusion has real coverage to gain"
        )
    elif mean_peak >= 0.20:
        verdict = "PREMISE_WEAK"
        reading = (
            f"mean peakedness {mean_peak:.2f}: some concentration, but the gain "
            "from exclusion will be modest on this model and task set"
        )
    else:
        verdict = "PREMISE_REFUTED"
        reading = (
            f"mean peakedness {mean_peak:.2f}: sampling already covers the answer "
            "space, so removing refuted mass buys little. The exclusion argument "
            "does not apply here, whatever it does elsewhere."
        )

    report = {
        "schema": "aura.rlc.peakedness_measurement.v1",
        "model": args.model,
        "temperature": args.temperature,
        "draws_per_task": args.draws,
        "tasks": len(rows),
        "total_draws": total_draws,
        "total_distinct": total_distinct,
        "distinct_fraction": round(total_distinct / total_draws, 4),
        "mean_peakedness": round(mean_peak, 4),
        "mean_expected_distinct_iid": round(mean_predicted_iid, 3),
        "verdict": verdict,
        "reading": reading,
        "duration_s": round(time.time() - started, 1),
        "per_task": rows,
    }
    payload = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if verdict != "PREMISE_REFUTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
