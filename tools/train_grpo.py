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


# Set by main() from --cot. Reasoning room is the fix the CP238 finding
# pointed at: the model failed program_trace at 0.05 because the terse
# FINAL_ANSWER format denied it chain-of-thought. This invites the
# token-level deliberation that actually makes models reason.
_COT_PREAMBLE = ""


def _render(tokenizer, task) -> str:
    content = task.prompt
    if _COT_PREAMBLE:
        content = _COT_PREAMBLE + "\n\n" + content
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
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
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--calibrate", action="store_true",
                        help="measure pass rates before training to skip dead cells")
    parser.add_argument("--calibrate-samples", type=int, default=2)
    parser.add_argument("--calibrate-group", type=int, default=4,
                        help="completions per calibration probe (cheaper than the train group)")
    parser.add_argument("--calibrate-tokens", type=int, default=160,
                        help="max tokens per calibration completion (shorter than training)")
    parser.add_argument("--calibrate-minutes", type=float, default=15.0,
                        help="wall-clock cap on the whole calibration phase")
    parser.add_argument("--cot", action="store_true",
                        help="invite step-by-step reasoning before the answer")
    parser.add_argument("--max-minutes", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--memory-fraction", type=float, default=0.55)
    args = parser.parse_args()

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    global _COT_PREAMBLE
    if args.cot:
        _COT_PREAMBLE = (
            "Work through this step by step, then end with your answer on "
            "its own line."
        )

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

        # Minimax curriculum (CP237): sample where the model is weakest but
        # not hopeless, so groups have reward variance instead of being
        # all-wrong. This is the cold-start fix -- skipping degenerate
        # groups after sampling them wastes the compute already spent.
        from core.learning.adaptive_curriculum import AdaptiveCurriculum
        from core.learning.durable_run import DurableRun

        by_cell: dict[tuple[str, int], list] = {}
        for task in train_tasks:
            by_cell.setdefault((task.domain, task.depth), []).append(task)
        curriculum = AdaptiveCurriculum.over(
            sorted({d for d, _ in by_cell}), sorted({p for _, p in by_cell})
        )
        sampler = random.Random(args.seed)


        # Durable resume (CP237): pick up from the last checkpoint after a
        # sleep, jetsam kill, or power loss -- the failure that killed the
        # CP227 gate mid-run.
        durable = DurableRun(out_dir / "checkpoints")
        resumed = durable.latest()
        if resumed is not None:
            step = resumed.step
            curriculum = AdaptiveCurriculum.from_state(resumed.payload["curriculum"])
            adapter_file = out_dir / "checkpoints" / resumed.payload["adapters"]
            if adapter_file.exists():
                model.load_weights(str(adapter_file), strict=False)
            print(f"[resume] continuing from step {step}", flush=True)

        def checkpoint_now() -> None:
            from mlx.utils import tree_flatten

            name = f"adapters_{step:08d}.safetensors"
            mx.save_safetensors(
                str(out_dir / "checkpoints" / name),
                {
                    k: v for k, v in tree_flatten(model.trainable_parameters())
                    if "lora" in k
                },
            )
            durable.save(step, {"curriculum": curriculum.state(), "adapters": name})

        # Warm-start calibration (CP237/238): the first CP238 run wasted 86%
        # of its steps on cells that were already mastered or impossible,
        # because it discovered difficulty online while paying for every
        # degenerate group. Measuring pass rates FIRST -- a few cheap
        # rollouts per cell, no backward pass -- lets the curriculum start on
        # the learnable band instead of finding it the expensive way. Opt-in
        # so a resumed run (which already has a learned map) skips it.
        if args.calibrate and resumed is None:
            # Cheap, observable, bounded probe. A pass-rate ESTIMATE needs a
            # handful of short completions, not the full training group at
            # full length -- the earlier config did 360 x 320-token
            # generations and took over an hour with no output, which made
            # "fail fast" not fast at all. Here: a small group, short
            # completions, per-cell progress, and a wall-clock cap.
            cal_group = min(config.group_size, args.calibrate_group)
            cal_tokens = min(args.max_tokens, args.calibrate_tokens)
            cal_deadline = time.time() + args.calibrate_minutes * 60.0
            cells_sorted = sorted(by_cell.keys())
            print(
                f"[calibrate] {len(cells_sorted)} cells x {cal_group} completions "
                f"x {cal_tokens} tokens, cap {args.calibrate_minutes}m",
                flush=True,
            )

            def _measure(family: str, difficulty: int) -> float:
                pool = by_cell.get((family, difficulty))
                if not pool or time.time() > cal_deadline:
                    return 0.5  # unmeasured -> optimistic (curriculum explores it)
                probe = pool[sampler.randrange(len(pool))]
                with recurrence_adapter_scope(start=None, stop=None):
                    _, comps = sample_group(
                        model, tokenizer, probe, size=cal_group,
                        max_tokens=cal_tokens, temperature=args.temperature,
                        seed=args.seed + hash((family, difficulty)) % 9973,
                    )
                rate = sum(
                    reward_from_verdict(probe.grade(c), format_credit=args.format_credit)
                    for c in comps
                ) / len(comps)
                print(
                    f"[calibrate] {family}@{difficulty} pass={rate:.2f} "
                    f"({(time.time()-started)/60:.1f}m)",
                    flush=True,
                )
                return rate

            from core.learning.adaptive_curriculum import warm_start_pass_rates

            curriculum = warm_start_pass_rates(
                sorted({d for d, _ in by_cell}),
                sorted({p for _, p in by_cell}),
                _measure, samples_per_cell=args.calibrate_samples,
            )
            calib = curriculum.report()
            print(f"[calibrate] {calib}", flush=True)
            # Fail fast on a known-bad config. If calibration finds no
            # learnable cell -- everything already mastered or impossible --
            # training would burn its whole budget on degenerate groups and
            # produce no signal, which is exactly how the first two runs were
            # wasted. Abort with the diagnosis instead of running for hours.
            if not calib.get("learnable") and not calib.get("unexplored"):
                verdict = {
                    "schema": GRPO_TRAIN_SCHEMA,
                    "aborted": "no_reachable_frontier",
                    "diagnosis": (
                        "all cells saturated (too easy) or hopeless (too hard); "
                        "widen or shift the difficulty band before training"
                    ),
                    "calibration": calib,
                }
                (out_dir / "grpo_receipt.json").write_text(json.dumps(verdict, indent=2))
                print(f"[abort] no reachable frontier -> {verdict['diagnosis']}", flush=True)
                return 0

        while step < args.max_steps and time.time() < deadline:
            cell = curriculum.sample(sampler)
            pool = by_cell.get(cell) or train_tasks
            task = pool[sampler.randrange(len(pool))]
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
            # Teach the curriculum what it just learned about this cell, so
            # difficulty tracks competence instead of being fixed.
            curriculum.observe(
                task.domain, task.depth, advantage_report["mean_reward"],
                degenerate=advantage_report["degenerate"],
            )
            step += 1
            if step % args.checkpoint_every == 0:
                checkpoint_now()

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
        checkpoint_now()  # a final resumable point at the true end
        curriculum_report = curriculum.report()
        print(f"[curriculum] {curriculum_report}", flush=True)

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
        "curriculum": curriculum_report,
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
