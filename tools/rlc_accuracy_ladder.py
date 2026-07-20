#!/usr/bin/env python
"""Exact-match accuracy ladder for the Recursive Latent Cortex (CP212).

Every prior measurement in this program used cross-entropy. CE falls when a
model learns the ``FINAL_ANSWER: {...}`` FORMAT, which is not reasoning.
The only metric that supports a reasoning claim is whether the emitted
answer is CORRECT, and whether correctness rises with recurrent depth.

This tool measures exact-match accuracy across a depth ladder, for one or
more arms, on tasks the training generators never produced, and writes a
self-describing receipt. It never awards a claim: it produces the numbers
a preregistered decision rule consumes.

Arms:
  base        frozen checkpoint, no adapter
  adapter     with a recurrence-native adapter attached

Success shape the program is looking for (NOT asserted here):
  * accuracy rises with depth on families where depth helps;
  * accuracy does not fall on families where depth hurts (shallow selected);
  * held-out template/depth generalization, not just held-out instances.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

ACCURACY_SCHEMA = "aura.rlc_accuracy_ladder.v1"


def _load_tasks(families: list[str], task_depth: int, per_cell: int, seed: int):
    from core.learning import recurrence_curriculum as curriculum

    return curriculum.task_battery(families, [task_depth], per_cell, seed=seed)


def _render(tokenizer, task):
    prompt = list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": task.prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )
    )
    return prompt


def _decode_answer(model, tokenizer, prepared, state, max_tokens: int):
    """Greedy-decode an answer conditioned on the persisted slot state."""
    import mlx.core as mx

    from core.brain.llm.latent_cortex.answer_contract import (
        is_contract_complete,
    )
    from core.learning.recurrence_native_objective_v2 import _persist_and_score

    # Teacher-forcing the produced tokens back through the SAME persisted
    # context is what makes this a real decode. An earlier version re-ran
    # the model on the generated tokens alone, dropping prompt and slot
    # context entirely, which produced 0% at every depth -- a broken
    # harness, not a model result.
    produced: list[int] = []
    text = ""
    for _ in range(max_tokens):
        logits = _persist_and_score(
            model,
            prepared.prompt_embeddings,
            prepared.seeds[0],
            state,
            _extend_tail(model, prepared, produced),
            bridge_count=prepared.bridge_count,
            answer_count=max(1, len(produced) + 1),
            prelude_end=prepared.prelude_end,
            coda_start=prepared.coda_start,
        )
        token = int(mx.argmax(logits[0, -1]))
        produced.append(token)
        text = tokenizer.decode(produced)
        if is_contract_complete(text):
            break
        if tokenizer.eos_token_id is not None and token == tokenizer.eos_token_id:
            break
    return text


def _extend_tail(model, prepared, produced: list[int]):
    """Tail embeddings = bridge + tokens generated so far."""
    import mlx.core as mx

    if not produced:
        return prepared.tail_embeddings
    generated = model.model.embed_tokens(mx.array([produced]))
    bridge = prepared.tail_embeddings[:, : prepared.bridge_count, :]
    return mx.concatenate([bridge, generated], axis=1)


def _score(task, text: str) -> bool:
    try:
        return bool(task.score(text).correct)
    except Exception:  # noqa: BLE001 - malformed output is simply incorrect
        return False


def run_arm(
    model,
    tokenizer,
    tasks,
    depths: list[int],
    n_slots: int,
    max_tokens: int,
) -> dict:
    import mlx.core as mx

    from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
    from core.learning.recurrence_native_objective_v2 import (
        _advance_recurrent_states,
        _prepare_live_path,
    )

    spec = RLCExecutionSpec(
        n_slots=n_slots,
        branch_roles=("constructive_solution",),
        recurrent_steps=max(depths),
        exchange_interval=1,
    )
    by_depth: dict[int, list[bool]] = {depth: [] for depth in depths}
    by_family: dict[str, dict[int, list[bool]]] = {}
    for task in tasks:
        prompt = _render(tokenizer, task)
        answer_probe = list(
            tokenizer.encode(task.answer, add_special_tokens=False)
        )
        prepared = _prepare_live_path(
            model,
            prompt,
            answer_probe,
            spec=spec.with_depth(max(depths)),
            bridge_tokens=(),
        )
        states = list(prepared.states)
        for step in range(max(depths)):
            states = _advance_recurrent_states(
                model,
                prepared.prompts_at_window,
                states,
                prepared.anchors,
                spec.with_depth(max(depths)),
                step,
                prepared.prelude_end,
                prepared.coda_start,
            )
            depth = step + 1
            if depth not in by_depth:
                continue
            text = _decode_answer(
                model, tokenizer, prepared, states[0], max_tokens
            )
            correct = _score(task, text)
            by_depth[depth].append(correct)
            by_family.setdefault(task.family, {}).setdefault(depth, []).append(
                correct
            )
        mx.clear_cache()
    return {
        "by_depth": {
            str(depth): {
                "correct": sum(values),
                "n": len(values),
                "accuracy": (sum(values) / len(values)) if values else 0.0,
            }
            for depth, values in by_depth.items()
        },
        "by_family": {
            family: {
                str(depth): {
                    "correct": sum(values),
                    "n": len(values),
                    "accuracy": (sum(values) / len(values)) if values else 0.0,
                }
                for depth, values in per_depth.items()
            }
            for family, per_depth in by_family.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--families", default="khop,modular,register_trace")
    parser.add_argument("--task-depth", type=int, default=8)
    parser.add_argument("--per-cell", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=20260721)
    parser.add_argument("--depths", default="1,2,4,8")
    parser.add_argument("--n-slots", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from mlx_lm import load

    depths = [int(v) for v in args.depths.split(",") if v.strip()]
    families = [v.strip() for v in args.families.split(",") if v.strip()]
    started = time.time()
    print(f"loading {args.model}", flush=True)
    load_kwargs = {"adapter_path": args.adapter} if args.adapter else {}
    model, tokenizer = load(args.model, **load_kwargs)
    tasks = _load_tasks(families, args.task_depth, args.per_cell, args.eval_seed)
    print(
        f"{len(tasks)} eval tasks (seed {args.eval_seed}), depths {depths}",
        flush=True,
    )
    result = run_arm(
        model, tokenizer, tasks, depths, args.n_slots, args.max_tokens
    )
    print("\n=== exact-match accuracy by depth ===")
    for depth in depths:
        row = result["by_depth"][str(depth)]
        print(
            f"  depth {depth:2d}: {row['correct']:3d}/{row['n']:3d}"
            f" = {100*row['accuracy']:5.1f}%"
        )
    print("\n=== by family ===")
    for family, per_depth in sorted(result["by_family"].items()):
        cells = "  ".join(
            f"d{depth}={100*per_depth[str(depth)]['accuracy']:.0f}%"
            for depth in depths
            if str(depth) in per_depth
        )
        print(f"  {family:16s} {cells}")

    payload = {
        "schema": ACCURACY_SCHEMA,
        "model": args.model,
        "adapter": args.adapter,
        "families": families,
        "task_depth": args.task_depth,
        "per_cell": args.per_cell,
        "eval_seed": args.eval_seed,
        "depths": depths,
        "n_slots": args.n_slots,
        "max_tokens": args.max_tokens,
        "elapsed_s": round(time.time() - started, 3),
        "metric": "exact_match_correctness",
        "claims_awarded": [],
        "results": result,
    }
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
