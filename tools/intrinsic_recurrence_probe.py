#!/usr/bin/env python3
"""Does making the 32B ITSELF recurrent move accuracy? (CP226)

The previous ladder measured a model that TALKS to a recurrent workspace:
the answer tokens traversed the middle block exactly once at every depth,
so 25/29/25/25 across an 8x compute range was the only result that
architecture could produce. This probe runs the corrected shape -- the real
token stream re-enters ``layers[16:48]`` T times, giving 64 / 96 / 128 /
160 effective layers on the same weights.

Two controls make the numbers mean something:

* **T=1 is the base model.** It is bit-identical by construction, so if
  T=1 does not land on the vanilla baseline the harness is broken and no
  other cell may be read. Checked against a live vanilla arm, not asserted.
* **Untrained retrofit is expected to COST accuracy.** A checkpoint never
  pretrained to iterate has no reason for its middle block to be a stable
  map. A drop at T>1 is evidence about stability, not a refutation of
  recurrence; the question this probe answers is whether the degradation is
  recoverable-looking (graceful, stabilizer-responsive) or catastrophic.

Never claims a gain. It reports four cells, their controls, and the norm
trajectory that explains them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.learning.intrinsic_recurrence import (  # noqa: E402
    RecurrentDepthPlan,
    make_recurrent_caches,
    recurrent_hidden_states,
    trajectory_dynamics,
)
from core.runtime.mlx_memory_guard import mlx_memory_envelope  # noqa: E402
from tools.rlc_accuracy_ladder import (  # noqa: E402
    CalibrationError,
    HarnessError,
    _head_logits,
    _load_tasks,
    _score,
    _tally,
    calibrate_scoring,
    run_vanilla_arm,
)

BRIDGE = "\n\nFINAL_ANSWER: "


def _greedy_recurrent(model, tokenizer, task, plan, *, max_tokens, envelope):
    """Greedy decode where the ANSWER's own computation is T layers deep."""
    import mlx.core as mx

    from core.brain.llm.latent_cortex.answer_contract import is_contract_complete

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": task.prompt}],
        add_generation_prompt=True,
        tokenize=False,
    ) + BRIDGE
    ids = tokenizer.encode(rendered)
    caches = make_recurrent_caches(model, plan)
    hidden, trajectory = recurrent_hidden_states(
        model, mx.array([ids]), plan, caches=caches
    )
    dynamics = trajectory_dynamics(trajectory)
    pieces: list[str] = []
    token = int(mx.argmax(_head_logits(model, hidden)[0, -1]))
    eos = tokenizer.eos_token_id
    for step in range(max_tokens):
        if token == eos:
            break
        pieces.append(tokenizer.decode([token]))
        text = "".join(pieces)
        if "}" in text and is_contract_complete(text):
            break
        hidden, _ = recurrent_hidden_states(
            model, mx.array([[token]]), plan, caches=caches
        )
        token = int(mx.argmax(_head_logits(model, hidden)[0, -1]))
        if envelope is not None and step % 16 == 15:
            envelope.reclaim(force=True)
    if envelope is not None:
        envelope.reclaim(force=True)
    return "".join(pieces), dynamics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iterations", default="1,2,3,4")
    parser.add_argument("--prelude-end", type=int, default=16)
    parser.add_argument("--coda-start", type=int, default=48)
    parser.add_argument("--anchor-injection", type=float, default=0.0)
    parser.add_argument("--renormalize", action="store_true")
    parser.add_argument("--per-cell", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--task-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--families",
        default="khop,modular,sorting,parity,graph_reach,expression",
    )
    args = parser.parse_args()

    # Calibrate BEFORE loading 20GB of weights: a broken scorer discovered
    # after the model is resident has already wasted the run.
    calibration = calibrate_scoring()
    if not calibration["passed"]:
        raise CalibrationError(f"scoring is not trustworthy: {calibration}")
    print(f"[calibration] {calibration['checks']} fixtures PASS", flush=True)

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    tasks = _load_tasks(families, args.task_depth, args.per_cell, args.seed)
    iterations = [int(t) for t in args.iterations.split(",") if t.strip()]
    started = time.time()

    from mlx_lm import load
    from core.runtime.model_lane_control import standalone_model_lane

    with standalone_model_lane(
        owner_id=f"intrinsic-recurrence-probe:{Path(args.out).name}",
        model_path=args.model,
        purpose="evaluation",
        preemptible=False,
        metadata={"tool": "intrinsic_recurrence_probe", "operator_launched": True},
    ), mlx_memory_envelope(fraction=0.55) as envelope:
        print(f"[envelope] {envelope.to_receipt()}", flush=True)
        model, tokenizer = load(args.model)
        total_layers = len(model.model.layers)
        print(f"[model] {total_layers} layers loaded", flush=True)

        cells = []
        for count in iterations:
            plan = RecurrentDepthPlan(
                prelude_end=args.prelude_end,
                coda_start=args.coda_start,
                iterations=count,
                anchor_injection=args.anchor_injection,
                renormalize=args.renormalize,
            )
            outcomes, deltas, fixed_points, diverged = [], [], 0, 0
            cell_started = time.time()
            for task in tasks:
                text, dynamics = _greedy_recurrent(
                    model, tokenizer, task, plan,
                    max_tokens=args.max_tokens, envelope=envelope,
                )
                outcomes.append(_score(task, text))
                if dynamics.get("diverged"):
                    diverged += 1
                elif dynamics.get("measurable"):
                    deltas.append(dynamics["final_delta"])
                    fixed_points += int(dynamics["at_fixed_point"])
            tally = _tally(outcomes)
            cell = {
                "plan": plan.to_receipt(total_layers),
                "tally": tally,
                "mean_final_delta": (
                    round(sum(deltas) / len(deltas), 6) if deltas else None
                ),
                "at_fixed_point": fixed_points,
                "diverged": diverged,
                # Logits decoded from a non-finite state are noise. Report
                # the tally so the failure is visible, but refuse to let it
                # be read as an accuracy.
                "trustworthy": diverged == 0,
                "seconds": round(time.time() - cell_started, 1),
            }
            cells.append(cell)
            print(
                f"[T={count}] depth={cell['plan']['effective_depth']} "
                f"REASONING={tally['reasoning_accuracy']:.0%} "
                f"strict={tally['accuracy']:.0%} "
                f"answered={tally['answered_at_all']:.0%} "
                f"final_delta={cell['mean_final_delta']} "
                f"{'DIVERGED=' + str(diverged) + ' ' if diverged else ''}"
                f"({cell['seconds']}s)",
                flush=True,
            )

        print("[control] vanilla arm", flush=True)
        vanilla = run_vanilla_arm(
            model, tokenizer, tasks, max_tokens=args.max_tokens,
            samples=1, envelope=envelope, bridge_text=BRIDGE,
        )

    # T=1 IS the base model by construction. If it does not reproduce the
    # vanilla arm, the recurrent forward is wrong and every other cell in
    # this file is meaningless -- so say so in the receipt rather than
    # letting a reader compare numbers that were never comparable.
    base_cell = next(
        (c for c in cells if c["plan"]["iterations"] == 1), None
    )
    identity_gap = None
    if base_cell is not None:
        identity_gap = abs(
            base_cell["tally"]["reasoning_accuracy"]
            - vanilla["greedy"]["reasoning_accuracy"]
        )

    receipt = {
        "schema": "aura.intrinsic_recurrence_probe.v1",
        "model": args.model,
        "families": families,
        "per_cell": args.per_cell,
        "calibration": calibration,
        "cells": cells,
        "vanilla_control": vanilla,
        "identity_check": {
            "t1_vs_vanilla_gap": identity_gap,
            # T=1 is bit-identical in exact arithmetic; sampling n=24 and
            # fp16 decode leave a little slack, but a large gap is a bug.
            "trustworthy": bool(
                identity_gap is not None and identity_gap <= 0.10
            ),
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(receipt, indent=2))
    if not receipt["identity_check"]["trustworthy"]:
        print(
            f"[VOID] T=1 ({base_cell['tally']['reasoning_accuracy']:.0%}) does "
            f"not reproduce vanilla ({vanilla['greedy']['reasoning_accuracy']:.0%})"
            " -- the recurrent forward is wrong; no cell here may be read",
            flush=True,
        )
        return 2
    print(f"[receipt] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (HarnessError, CalibrationError) as failure:
        print(f"[HARNESS FAILURE] {failure}", flush=True)
        raise SystemExit(3) from failure
