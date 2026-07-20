"""Capacity + stability probe for the recurrent operator.

Three questions that decide whether "warping the weights" buys anything:

1. SPECTRAL: is the recurrent window transform norm-preserving? If the
   operator's gain drifts from 1.0, repeated application explodes or
   collapses -- which is exactly why the engine needs RMSMatch, alpha
   interpolation, clip ratios and divergence guards. A norm-preserving
   (orthogonal/unitary) operator would be stable BY CONSTRUCTION.

2. BANDWIDTH: what is the effective rank of the M-slot workspace? If 4
   slots carry far less than 4 independent directions, the latent channel
   is the bottleneck and more slots (or richer slot structure) pays.

3. DEPTH HEADROOM: khop improved -33% from depth 1->8 and was still
   monotone at 8. Does it keep improving at 16/32, or saturate? If it keeps
   going, stability-limited depth is leaving measurable gains on the table.
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
from core.learning.recurrence_native_objective_v2 import (  # noqa: E402
    live_path_forward,
    live_path_loss,
)

MODEL = str(REPO / "models/Qwen2.5-1.5B-Instruct-4bit")


def effective_rank(matrix) -> float:
    """Participation ratio of singular values: how many directions are
    actually carrying signal (1.0 = fully redundant, N = fully used)."""
    singular = mx.linalg.svd(
        matrix.astype(mx.float32), compute_uv=False, stream=mx.cpu
    )
    values = mx.abs(singular)
    total = mx.sum(values)
    if float(total) < 1e-9:
        return 0.0
    probabilities = values / total
    entropy = -mx.sum(
        probabilities * mx.log(mx.maximum(probabilities, 1e-12))
    )
    return float(mx.exp(entropy))


def main() -> int:
    print(f"loading {MODEL}\n", flush=True)
    model, tokenizer = load(MODEL)
    tasks = curriculum.task_battery(["khop"], [8], 3, seed=4242)
    results: dict[str, object] = {}

    # ---- 2. BANDWIDTH: effective rank of the slot workspace -------------
    print("=== slot-workspace effective rank (bandwidth) ===")
    ranks = []
    for n_slots in (4, 8, 16):
        spec = RLCExecutionSpec(
            n_slots=n_slots,
            branch_roles=("constructive_solution",),
            recurrent_steps=4,
            exchange_interval=1,
        )
        task = tasks[0]
        prompt = list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": task.prompt}],
                add_generation_prompt=True,
                tokenize=True,
            )
        )
        answer = list(tokenizer.encode(task.answer, add_special_tokens=False))
        forward = live_path_forward(model, prompt, answer, spec=spec)
        state = forward.branch_states[0][0]  # (n_slots, hidden)
        rank = effective_rank(state)
        ranks.append({"n_slots": n_slots, "effective_rank": round(rank, 3)})
        print(
            f"  slots={n_slots:3d}  effective_rank={rank:6.3f}  "
            f"utilization={100*rank/n_slots:5.1f}%"
        )
    results["bandwidth"] = ranks

    # ---- 3. DEPTH HEADROOM: does khop keep improving past 8? ------------
    print("\n=== khop depth headroom (does the -33% keep going?) ===")
    spec = RLCExecutionSpec(
        n_slots=4,
        branch_roles=("constructive_solution",),
        recurrent_steps=32,
        exchange_interval=1,
    )
    depth_means = {}
    for depth in (1, 2, 4, 8, 16, 32):
        losses = []
        for task in tasks:
            prompt = list(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": task.prompt}],
                    add_generation_prompt=True,
                    tokenize=True,
                )
            )
            answer = list(
                tokenizer.encode(task.answer, add_special_tokens=False)
            )
            value = live_path_loss(
                model, prompt, answer, spec=spec.with_depth(depth)
            )
            mx.eval(value)
            losses.append(float(value))
        mean = sum(losses) / len(losses)
        depth_means[depth] = mean
        base = depth_means[1]
        print(
            f"  depth {depth:3d}: CE={mean:7.4f}  "
            f"delta_vs_d1={100*(mean-base)/base:+6.1f}%"
        )
    results["depth_headroom"] = {str(k): v for k, v in depth_means.items()}
    best_depth = min(depth_means, key=depth_means.get)
    saturated = best_depth <= 8
    print(f"\n  best depth = {best_depth}")
    print(
        "  VERDICT: "
        + (
            "saturates by depth 8 — deeper recurrence buys little"
            if saturated
            else f"still improving at depth {best_depth} — "
            "stability-limited depth is leaving gains unclaimed"
        )
    )
    results["best_depth"] = best_depth
    results["depth_saturated_by_8"] = saturated

    out = Path(__file__).with_name("capacity_probe_result.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
