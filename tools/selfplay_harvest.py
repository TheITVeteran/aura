#!/usr/bin/env python3
"""tools/selfplay_harvest.py — verifier-grounded self-play preference data.

The RLVR recipe, run locally against Aura's own exact checkers: sample K
attempts per task from the CURRENT model at temperature, grade every attempt
with the battery's exact verifier, and emit (correct, incorrect) contrasts for
the SAME prompt as DPO pairs. The pairs flow through the canonical
VerifiablePreferenceHarness — the same store the live reasoning amplifier
feeds — so the compounding loop consumes self-play data and lived-conversation
data through one door.

Leak discipline (both layers enforced):
  * training tasks come from seeds BELOW 1000; the held-out gate batteries use
    seeds ≥ 1000 (weight_compounding.battery_seed_base) — disjoint by design;
  * the compounding harvest additionally drops any row that contains a sealed
    battery prompt (fingerprint seal), so even a misconfigured run cannot
    train on its own eval.

The summary this prints (correct-rate at temperature) is itself a capability
signal worth watching across generations.

Usage:
  python tools/selfplay_harvest.py --model <path> --store <preferences.jsonl>
      [--tasks 48] [--attempts 4] [--seed-start 1] [--temp 0.8]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.heldout_battery import (  # noqa: E402
    BatterySpec,
    generate_battery,
    grade_response,
)
from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402

EVAL_SEED_FLOOR = 1000  # weight_compounding.battery_seed_base — never harvest at/above


def _build_prompt(tokenizer, user_prompt: str) -> str:
    apply = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply):
        try:
            return apply(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except (TypeError, ValueError):
            pass
    return user_prompt


def harvest(args: argparse.Namespace) -> dict:
    if args.seed_start >= EVAL_SEED_FLOOR:
        raise SystemExit(
            f"--seed-start must stay below {EVAL_SEED_FLOOR}: seeds at/above it are "
            "reserved for the sealed held-out gate batteries"
        )

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    from core.learning.verifiable_preference_harness import (
        Attempt,
        VerifiablePreferenceHarness,
    )

    # Spread tasks across several seeds so one seed's rng stream doesn't shape
    # the whole training set; all seeds stay below the eval floor.
    per_seed = 16
    tasks = []
    seed = args.seed_start
    while len(tasks) < args.tasks and seed < EVAL_SEED_FLOOR:
        tasks.extend(generate_battery(BatterySpec(seed=seed, size=per_seed)))
        seed += 1
    tasks = tasks[: args.tasks]

    model, tokenizer = load(args.model)
    sampler = make_sampler(temp=args.temp, top_p=0.95)
    harness = VerifiablePreferenceHarness(store_path=Path(args.store))

    stats = {
        "tasks": len(tasks),
        "attempts_per_task": args.attempts,
        "correct_attempts": 0,
        "total_attempts": 0,
        "tasks_with_contrast": 0,
        "pairs_emitted": 0,
    }
    started = time.time()
    for i, task in enumerate(tasks, start=1):
        prompt = _build_prompt(tokenizer, task.prompt)
        attempts: list[Attempt] = []
        for _ in range(args.attempts):
            out = generate(
                model, tokenizer, prompt=prompt,
                max_tokens=args.max_tokens, sampler=sampler, verbose=False,
            )
            text = out if isinstance(out, str) else str(out)
            ok = grade_response(task, text)
            attempts.append(
                Attempt(candidate=text, verified=ok, checked=True, confidence=1.0 if ok else 0.0)
            )
        correct = sum(1 for a in attempts if a.verified)
        stats["correct_attempts"] += correct
        stats["total_attempts"] += len(attempts)
        if 0 < correct < len(attempts):
            stats["tasks_with_contrast"] += 1
        stats["pairs_emitted"] += harness.ingest(task.prompt, attempts, domain=task.domain)
        if i % 10 == 0 or i == len(tasks):
            rate = stats["correct_attempts"] / max(1, stats["total_attempts"])
            print(
                f"[selfplay] {i}/{len(tasks)} tasks | correct-rate {rate:.1%} | "
                f"pairs {stats['pairs_emitted']} | {time.time() - started:.0f}s",
                flush=True,
            )

    stats["correct_rate"] = round(
        stats["correct_attempts"] / max(1, stats["total_attempts"]), 4
    )
    stats["elapsed_s"] = round(time.time() - started, 1)
    stats["model"] = str(args.model)
    stats["store"] = str(args.store)
    stats["seed_range"] = [args.seed_start, seed - 1]
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--store", required=True, help="preference store to append pairs into")
    parser.add_argument("--tasks", type=int, default=48)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", default="", help="optional stats JSON path")
    args = parser.parse_args()

    with standalone_model_lane(
        owner_id="selfplay-harvest",
        model_path=args.model,
        purpose="benchmark",
        metadata={"tool": "selfplay_harvest"},
    ):
        stats = harvest(args)
    print(json.dumps(stats, indent=2, sort_keys=True))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
