#!/usr/bin/env python3
"""Verifier-driven RL on the resident cortex (CP233).

The training loop for CP229. Anima Rationis line 511 records the existence
proof: QwQ-32B reached DeepSeek-R1-comparable reasoning through RL over a
32B foundation with correctness verifiers for mathematics and execution
feedback for code -- the same parameter class as Aura's cortex.

Per step:

    1. sample K completions for one prompt at temperature
    2. grade each with a PROGRAM (never the model's own opinion)
    3. advantage_i = (r_i - mean r) / std r      -- the group is the baseline
    4. loss = -mean(advantage_i * logprob_i) + beta * KL(policy || reference)

The reference policy is this same model with the adapter scope disabled,
so the KL leash is measured against the true pre-RL behaviour rather than
a stale copy -- and it costs no extra memory, which matters on a host that
has already been taken down once by an unbounded run.

What this run refuses to do:

* **Report a loss curve as progress.** If every completion in a group earns
  the same grade, the advantages are all zero and the step taught nothing.
  Those groups are counted, and a run made mostly of them is declared to
  have no learning signal regardless of how tidy its loss looks.
* **Score itself on its training set.** Held-out tasks come from a
  separate seed with proven-disjoint prompts, and the verdict is the
  held-out number.
* **Claim a gain from format compliance.** Reward is correctness; format
  credit is capped, because formatting is far easier to learn than
  reasoning and a model that learns it looks like it is improving.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.learning.grpo import (  # noqa: E402
    GRPOConfig,
    GRPOTelemetry,
    group_advantages,
    grpo_loss,
    reward_from_verdict,
    sequence_logprob,
)
from core.learning.verifiable_tasks import (  # noqa: E402
    disjoint_split,
    scaling_report,
)
from core.runtime.mlx_memory_guard import mlx_memory_envelope  # noqa: E402

GRPO_TRAIN_SCHEMA = "aura.grpo_training.v1"


def _render(tokenizer, task) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": task.prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )


def sample_group(model, tokenizer, task, *, size, max_tokens, temperature, seed):
    """K completions for one prompt. Diversity is the mechanism."""
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    prompt = _render(tokenizer, task)
    completions: list[str] = []
    for index in range(size):
        mx.random.seed(seed * 1000 + index)
        pieces: list[str] = []
        for response in stream_generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature, top_p=0.95),
        ):
            pieces.append(response.text)
        completions.append("".join(pieces))
    return prompt, completions


def completion_logprob(model, tokenizer, prompt, completion, *, adapters_on):
    """Log-probability of a completion, with adapters on or off.

    Adapters off gives the reference policy for the KL term at zero extra
    memory -- a second resident copy of a 32B is exactly the kind of thing
    that took this host down.
    """
    import mlx.core as mx

    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_scope,
    )

    prompt_ids = tokenizer.encode(prompt)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    if not completion_ids:
        return None
    full = mx.array([prompt_ids + completion_ids])
    count = len(completion_ids)

    def forward():
        logits = model(full)
        start = full.shape[1] - count - 1
        return sequence_logprob(
            logits[:, start : start + count, :], mx.array([completion_ids])
        )

    if adapters_on:
        with recurrence_adapter_scope(start=None, stop=None):
            return forward()
    return forward()  # no scope => ScopedLoRALinear passes through


def evaluate_heldout(model, tokenizer, tasks, *, max_tokens, envelope):
    """Greedy held-out accuracy by depth -- the number that counts."""
    from mlx_lm import stream_generate

    results = []
    for task in tasks:
        pieces: list[str] = []
        for response in stream_generate(
            model, tokenizer, prompt=_render(tokenizer, task),
            max_tokens=max_tokens,
        ):
            pieces.append(response.text)
        verdict = task.grade("".join(pieces))
        results.append((task, bool(verdict["correct"])))
        if envelope is not None:
            envelope.reclaim(force=True)
    return scaling_report(results)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--domains", default="arithmetic_chain,program_trace,constraint_order")
    parser.add_argument("--depths", default="2,4,8")
    parser.add_argument("--train-per-cell", type=int, default=32)
    parser.add_argument("--holdout-per-cell", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--kl-coefficient", type=float, default=0.04)
    parser.add_argument("--format-credit", type=float, default=0.05)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-targets", default="o_proj,v_proj,q_proj")
    parser.add_argument("--lora-layers", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--max-minutes", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--memory-fraction", type=float, default=0.55)
    args = parser.parse_args()

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    config = GRPOConfig(
        group_size=args.group_size, kl_coefficient=args.kl_coefficient
    )
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    train_tasks, holdout = disjoint_split(
        domains=domains, depths=depths,
        train_per_cell=args.train_per_cell,
        holdout_per_cell=args.holdout_per_cell, seed=args.seed,
    )
    import random

    random.Random(args.seed).shuffle(train_tasks)
    print(
        f"[tasks] {len(train_tasks)} train / {len(holdout)} held-out "
        f"(disjoint prompts verified)",
        flush=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    deadline = started + args.max_minutes * 60.0

    from mlx_lm import load

    with mlx_memory_envelope(fraction=args.memory_fraction) as envelope:
        print(f"[envelope] {envelope.to_receipt()}", flush=True)
        model, tokenizer = load(args.model)
        model.freeze()

        from core.brain.llm.latent_cortex.recurrence_adapter import (
            ScopedLoRALinear,
            recurrence_adapter_scope,
        )

        total_layers = len(model.model.layers)
        targets = tuple(t.strip() for t in args.lora_targets.split(","))
        attached = 0
        for index in range(max(0, total_layers - args.lora_layers), total_layers):
            layer = model.model.layers[index]
            for parent_name in ("self_attn", "mlp"):
                parent = getattr(layer, parent_name, None)
                if parent is None:
                    continue
                for target in targets:
                    projection = getattr(parent, target, None)
                    if projection is not None and not isinstance(
                        projection, ScopedLoRALinear
                    ):
                        setattr(
                            parent, target,
                            ScopedLoRALinear.from_base(projection, r=args.lora_rank),
                        )
                        attached += 1
        if not attached:
            raise RuntimeError("no projections adapted; check --lora-targets")
        print(f"[wiring] {attached} projections adapted", flush=True)

        optimizer = optim.Adam(learning_rate=args.learning_rate)
        telemetry = GRPOTelemetry()
        history: list[dict] = []
        baseline_eval = None
        step = 0

        while step < args.max_steps and time.time() < deadline:
            task = train_tasks[step % len(train_tasks)]
            with recurrence_adapter_scope(start=None, stop=None):
                prompt, completions = sample_group(
                    model, tokenizer, task, size=config.group_size,
                    max_tokens=args.max_tokens, temperature=args.temperature,
                    seed=args.seed + step,
                )
            rewards = [
                reward_from_verdict(
                    task.grade(text), format_credit=args.format_credit
                )
                for text in completions
            ]
            advantage_report = group_advantages(
                rewards, clip=config.advantage_clip
            )
            telemetry.observe(advantage_report)
            step += 1

            def run_eval() -> None:
                """Evaluation must not depend on the step having trained.

                Placed on both paths deliberately: when it lived after the
                degenerate `continue`, a step that produced no signal also
                skipped its evaluation -- so a run could spend its entire
                budget and never measure itself. The smoke run did exactly
                that: 6 groups, 0 evals.
                """
                nonlocal baseline_eval
                if step % args.eval_every != 0:
                    return
                with recurrence_adapter_scope(start=None, stop=None):
                    report = evaluate_heldout(
                        model, tokenizer, holdout,
                        max_tokens=args.max_tokens, envelope=envelope,
                    )
                if baseline_eval is None:
                    baseline_eval = report
                report["step"] = step
                history.append(report)
                print(
                    f"[eval {step}] overall={report['overall']:.3f} "
                    f"by_depth={report['accuracy_by_depth']} "
                    f"falloff={report['depth_falloff']}",
                    flush=True,
                )

            if advantage_report["degenerate"]:
                # No preference to learn from. Skipping is honest; training
                # on zeros would produce a smooth curve over no signal.
                if step % 10 == 0:
                    print(
                        f"[step {step}] degenerate "
                        f"(mean_r={advantage_report['mean_reward']:.2f}) "
                        f"({(time.time()-started)/60:.1f}m)",
                        flush=True,
                    )
                run_eval()
                continue

            reference = [
                mx.stop_gradient(
                    completion_logprob(
                        model, tokenizer, prompt, text, adapters_on=False
                    )
                )
                for text in completions
            ]

            def loss_fn(_model):
                policy = [
                    completion_logprob(
                        _model, tokenizer, prompt, text, adapters_on=True
                    )
                    for text in completions
                ]
                loss, _ = grpo_loss(
                    policy, advantage_report["advantages"],
                    reference_logprobs=reference,
                    kl_coefficient=config.kl_coefficient,
                )
                return loss

            loss, grads = nn.value_and_grad(model, loss_fn)(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            envelope.reclaim(force=True)

            if step % 10 == 0:
                print(
                    f"[step {step}] loss={float(loss):.4f} "
                    f"mean_r={advantage_report['mean_reward']:.2f} "
                    f"({(time.time()-started)/60:.1f}m)",
                    flush=True,
                )

            run_eval()

        from mlx.utils import tree_flatten

        adapters = {
            name: value
            for name, value in tree_flatten(model.trainable_parameters())
            if "lora" in name
        }
        mx.save_safetensors(str(out_dir / "grpo_adapters.safetensors"), adapters)

    signal = telemetry.verdict(config)
    final = history[-1] if history else None
    receipt = {
        "schema": GRPO_TRAIN_SCHEMA,
        "model": args.model,
        "config": config.to_receipt(),
        "domains": domains,
        "depths": depths,
        "train_tasks": len(train_tasks),
        "holdout_tasks": len(holdout),
        "steps": step,
        "learning_signal": signal,
        "history": history,
        "final": final,
        "verdict": {
            # A run with no usable groups did not fail to improve -- it
            # never had anything to improve from, and those need opposite
            # fixes.
            "had_signal": bool(signal["learning_signal"]),
            "heldout_improved": bool(
                final and baseline_eval
                and final["overall"] > baseline_eval["overall"]
            ),
            "diagnosis": signal["diagnosis"],
        },
        "elapsed_minutes": round((time.time() - started) / 60.0, 2),
    }
    (out_dir / "grpo_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(f"[verdict] {receipt['verdict']}", flush=True)
    print(f"[receipt] {out_dir / 'grpo_receipt.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
