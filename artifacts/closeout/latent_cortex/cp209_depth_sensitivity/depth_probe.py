"""Depth-sensitivity probe: does recurrence change anything on our curriculum?

If answer CE is flat across recurrent depths on the UNTRAINED model, the
tasks do not require latent recurrence, the monotonicity hinge is
satisfiable by the identity map, and no amount of depth training can
produce a reasoning gain. This is the load-bearing assumption of the whole
program and it has never been measured.

Runs on the 1.5B so it cannot contend with the resident 32B training lane.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/Users/bryan/.aura/live-source")
sys.path.insert(0, str(REPO))

import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec  # noqa: E402
from core.learning import recurrence_curriculum as curriculum  # noqa: E402
from core.learning.recurrence_native_objective_v2 import live_path_loss  # noqa: E402

MODEL = str(REPO / "models/Qwen2.5-1.5B-Instruct-4bit")
DEPTHS = (1, 2, 4, 8)
FAMILIES = ["khop", "boolean", "modular", "register_trace", "code_trace"]
TASK_DEPTH = 8  # the HARDEST tier: most likely to need serial computation


def render(tokenizer, task):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": task.prompt}],
        add_generation_prompt=True,
        tokenize=True,
    )
    answer = tokenizer.encode(task.answer, add_special_tokens=False)
    return list(prompt), list(answer)


def main() -> int:
    print(f"loading {MODEL}", flush=True)
    model, tokenizer = load(MODEL)
    spec = RLCExecutionSpec(
        n_slots=4,
        branch_roles=("constructive_solution",),
        recurrent_steps=max(DEPTHS),
        exchange_interval=1,
    )
    tasks = curriculum.task_battery(FAMILIES, [TASK_DEPTH], 3, seed=99991)
    print(f"{len(tasks)} tasks at task-depth {TASK_DEPTH}\n", flush=True)

    per_depth: dict[int, list[float]] = {depth: [] for depth in DEPTHS}
    for task in tasks:
        prompt_tokens, answer_tokens = render(tokenizer, task)
        row = []
        for depth in DEPTHS:
            value = live_path_loss(
                model,
                prompt_tokens,
                answer_tokens,
                spec=spec.with_depth(depth),
            )
            mx.eval(value)
            loss = float(value)
            per_depth[depth].append(loss)
            row.append(f"d{depth}={loss:.4f}")
        print(f"  {task.family:16s} {' '.join(row)}", flush=True)

    print("\n=== mean answer CE by recurrent depth (UNTRAINED base) ===")
    means = {}
    for depth in DEPTHS:
        values = per_depth[depth]
        means[depth] = sum(values) / len(values)
        print(f"  depth {depth}: {means[depth]:.5f}")

    spread = max(means.values()) - min(means.values())
    relative = spread / max(means[DEPTHS[0]], 1e-9)
    print(f"\nabsolute spread across depths : {spread:.5f}")
    print(f"relative to depth-1 loss      : {100*relative:.2f}%")
    best = min(means, key=means.get)
    print(f"best depth                    : {best}")
    verdict = (
        "RECURRENCE IS INERT — depth carries no signal on these tasks"
        if relative < 0.02
        else "recurrence changes the loss; depth is a real lever"
    )
    print(f"\nVERDICT: {verdict}")
    out = Path(__file__).with_name("depth_probe_result.json")
    out.write_text(
        json.dumps(
            {
                "model": MODEL,
                "task_depth": TASK_DEPTH,
                "families": FAMILIES,
                "n_tasks": len(tasks),
                "mean_ce_by_depth": {str(k): v for k, v in means.items()},
                "absolute_spread": spread,
                "relative_spread": relative,
                "best_depth": best,
                "verdict": verdict,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
