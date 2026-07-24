#!/usr/bin/env python3
"""Which lever actually rotates the recurrent increments? (pass-divergence rig)

CP226 reduced the intrinsic-recurrence obstacle to one number: on a 1.5B,
float32, cos(Δpass1, Δpass2) = 0.9994 — iterating one fixed window is a
damped fixed-point iteration, and successive increments of such an
iteration align with the dominant eigendirection of its local Jacobian
(power iteration). Extra depth then re-computes the previous step instead
of taking a new one. That is an ARCHITECTURE property; fifteen GRPO
campaigns on the fixed architecture could not have trained it away.

This rig measures, mechanically and cheaply, how far each candidate
intervention moves that geometry BEFORE anyone spends 32B training hours:

  * renormalize        — RMS re-anchoring (the CP227 stabilizer)
  * anchor_x           — re-inject the prelude state each re-entry
                         (Geiping-style input re-injection)
  * noise_x            — deterministic isotropic kick each re-entry
  * step_ops_x         — a DIFFERENT operator per pass (depth-conditioned
                         LoRA deltas, zero at pass 1, random at re-entries)
  * all_levers         — the combination

Reported per condition: consecutive-increment cosines (the CP226 number),
alignment of later increments with the first (does motion ever leave the
initial ray?), relative step magnitudes, and the fixed-point flag. Geometry
only — no accuracy claims; a rotated increment is necessary for depth to
compute anything new, not sufficient for it to compute something useful.
The training question stays with the SPARK-069 campaign.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MEASURE_SCHEMA = "aura.pass_divergence_measurement.v1"

PROMPTS = [
    "Track the register: x=3. x doubles, then loses 1, then doubles again. "
    "What is x? Work step by step.",
    "A is north of B. C is south of B. D is north of A. Order them from "
    "north to south, reasoning carefully.",
    "Compute ((7 mod 5) * 9 + 4) mod 6, showing each intermediate value.",
    "If every glarb is a fleem and no fleem is a snork, can a glarb be a "
    "snork? Explain the chain.",
    "Reverse the list [2, 9, 4, 7], then swap the middle two elements. "
    "What list results?",
    "k-hop: f(1)=4, f(4)=2, f(2)=8, f(8)=5. Starting at 1, apply f four "
    "times. Where do you land?",
]


def _flat32(state):
    import mlx.core as mx

    return mx.reshape(state.astype(mx.float32), (-1,))


def _cos(a, b) -> float:
    import mlx.core as mx

    denominator = mx.maximum(mx.linalg.norm(a) * mx.linalg.norm(b), 1e-9)
    return float(mx.sum(a * b) / denominator)


def _geometry(trajectory) -> dict:
    """Increment geometry in float32 — fp16 reductions overflow (CP226 bug)."""
    import mlx.core as mx

    states = [_flat32(state) for state in trajectory]
    increments = [after - before for before, after in zip(states, states[1:], strict=False)]
    consecutive = [
        round(_cos(first, second), 6)
        for first, second in zip(increments, increments[1:], strict=False)
    ]
    versus_first = [
        round(_cos(increments[0], later), 6) for later in increments[1:]
    ]
    magnitudes = [
        round(
            float(mx.linalg.norm(step))
            / max(float(mx.linalg.norm(states[0])), 1e-9),
            6,
        )
        for step in increments
    ]
    return {
        "consecutive_increment_cos": consecutive,
        "increment_vs_first_cos": versus_first,
        "relative_step_magnitudes": magnitudes,
    }


def _attach_step_operators(model, *, prelude_end, coda_start, depths, scale, seed):
    """ScopedLoRA + per-depth deltas: zero at pass 1, random at re-entries.

    Pass 1 keeps the exact base operator so its increment matches the
    baseline arm and any cosine change is attributable to the re-entry
    operators alone. Deterministic per seed so the receipt replays.
    """
    import mlx.core as mx

    from core.brain.llm.latent_cortex.recurrence_adapter import ScopedLoRALinear
    from core.learning.depth_conditioned_lora import wrap_depth_conditioned

    for index in range(prelude_end, len(model.model.layers)):
        if index >= coda_start:
            break
        attention = model.model.layers[index].self_attn
        for target in ("o_proj", "v_proj"):
            projection = getattr(attention, target, None)
            if projection is not None and not isinstance(projection, ScopedLoRALinear):
                setattr(
                    attention, target, ScopedLoRALinear.from_base(projection, r=8)
                )
    banks = wrap_depth_conditioned(model, depths=depths)
    for bank_index, (_name, bank) in enumerate(sorted(banks.items())):
        for depth in range(1, depths):
            key_a = mx.random.key(seed + bank_index * 1009 + depth * 13)
            key_b = mx.random.key(seed + bank_index * 1009 + depth * 13 + 7)
            bank.depth_a[depth] = (
                mx.random.normal(bank.scoped.lora_a.shape, key=key_a) * scale
            )
            bank.depth_b[depth] = (
                mx.random.normal(bank.scoped.lora_b.shape, key=key_b) * scale
            )
    mx.eval(model.parameters())
    return banks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="models/Qwen2.5-1.5B-Instruct-4bit"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--prelude-frac", type=float, default=0.25)
    parser.add_argument("--coda-frac", type=float, default=0.25)
    parser.add_argument("--max-minutes", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--memory-fraction", type=float, default=0.35)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_scope,
    )
    from core.learning.intrinsic_recurrence import (
        RecurrentDepthPlan,
        recurrent_hidden_states,
        trajectory_dynamics,
    )
    from core.runtime.mlx_memory_guard import mlx_memory_envelope
    from core.runtime.model_lane_control import standalone_model_lane

    deadline = time.monotonic() + args.max_minutes * 60.0
    started = time.time()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with standalone_model_lane(
        owner_id=f"pass-divergence-rig:{out_path.name}",
        model_path=args.model,
        purpose="evaluation",
        preemptible=False,
        metadata={"tool": "measure_pass_divergence", "operator_launched": True},
    ), mlx_memory_envelope(fraction=args.memory_fraction) as envelope:
        model, tokenizer = load(args.model)
        total_layers = len(model.model.layers)
        prelude_end = max(1, int(total_layers * args.prelude_frac))
        coda_start = min(total_layers - 1, total_layers - max(1, int(total_layers * args.coda_frac)))
        print(
            f"[model] {total_layers} layers; window [{prelude_end}:{coda_start}), "
            f"T={args.iterations}",
            flush=True,
        )

        def plan(**overrides) -> RecurrentDepthPlan:
            merged = {
                "prelude_end": prelude_end,
                "coda_start": coda_start,
                "iterations": args.iterations,
                "noise_seed": args.seed,
            }
            merged.update(overrides)
            return RecurrentDepthPlan(**merged)

        conditions: list[tuple[str, RecurrentDepthPlan, bool]] = [
            ("baseline", plan(), False),
            ("renormalize", plan(renormalize=True), False),
            ("anchor_0.1", plan(anchor_injection=0.1, renormalize=True), False),
            ("anchor_0.3", plan(anchor_injection=0.3, renormalize=True), False),
            ("noise_0.05", plan(interpass_noise=0.05, renormalize=True), False),
            ("noise_0.1", plan(interpass_noise=0.1, renormalize=True), False),
            ("step_ops_0.005", plan(), "small"),
            ("step_ops_0.02", plan(), "medium"),
            (
                "all_levers",
                plan(
                    anchor_injection=0.1,
                    interpass_noise=0.05,
                    renormalize=True,
                ),
                "medium",
            ),
        ]

        prompt_ids = []
        for prompt in PROMPTS:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt_ids.append(tokenizer.encode(rendered))

        # Step-operator banks are attached ONCE and gated by the adapter
        # scope: arms that do not enter the scope run the bare base model
        # (ScopedLoRALinear contract), so attachment cannot contaminate
        # them. Two delta scales share one bank set by rescaling.
        banks = _attach_step_operators(
            model,
            prelude_end=prelude_end,
            coda_start=coda_start,
            depths=args.iterations,
            scale=1.0,
            seed=args.seed,
        )
        print(f"[step-ops] {len(banks)} projections banked", flush=True)
        scale_by_name = {"small": 0.005, "medium": 0.02}

        results = []
        for name, condition_plan, ops_mode in conditions:
            if time.monotonic() > deadline:
                results.append({"condition": name, "skipped": "wall_clock"})
                continue
            for bank in banks.values():
                bank.delta_scale = scale_by_name.get(ops_mode, 0.0) if ops_mode else 0.0
            per_prompt = []
            condition_started = time.monotonic()
            for ids in prompt_ids:
                tokens = mx.array([ids])
                if ops_mode:
                    with recurrence_adapter_scope(start=None, stop=None):
                        _, trajectory = recurrent_hidden_states(
                            model, tokens, condition_plan
                        )
                else:
                    _, trajectory = recurrent_hidden_states(
                        model, tokens, condition_plan
                    )
                dynamics = trajectory_dynamics(trajectory)
                row = {"dynamics": dynamics}
                if dynamics.get("measurable") and not dynamics.get("diverged"):
                    row.update(_geometry(trajectory))
                per_prompt.append(row)
                envelope.reclaim(force=True)
            usable = [row for row in per_prompt if "consecutive_increment_cos" in row]
            aggregate = None
            if usable:
                pair_count = len(usable[0]["consecutive_increment_cos"])
                aggregate = {
                    "mean_consecutive_cos": [
                        round(
                            sum(row["consecutive_increment_cos"][i] for row in usable)
                            / len(usable),
                            6,
                        )
                        for i in range(pair_count)
                    ],
                    "mean_vs_first_cos": [
                        round(
                            sum(row["increment_vs_first_cos"][i] for row in usable)
                            / len(usable),
                            6,
                        )
                        for i in range(len(usable[0]["increment_vs_first_cos"]))
                    ],
                    "diverged": sum(
                        1 for row in per_prompt if row["dynamics"].get("diverged")
                    ),
                    "at_fixed_point": sum(
                        1
                        for row in per_prompt
                        if row["dynamics"].get("at_fixed_point")
                    ),
                }
            entry = {
                "condition": name,
                "plan": condition_plan.to_receipt(total_layers),
                "step_operator_scale": scale_by_name.get(ops_mode) if ops_mode else 0.0,
                "prompts": len(per_prompt),
                "aggregate": aggregate,
                "per_prompt": per_prompt,
                "seconds": round(time.monotonic() - condition_started, 1),
            }
            results.append(entry)
            summary = aggregate["mean_consecutive_cos"] if aggregate else "UNMEASURABLE"
            print(f"[{name}] cos={summary} ({entry['seconds']}s)", flush=True)

    receipt = {
        "schema": MEASURE_SCHEMA,
        "model": args.model,
        "iterations": args.iterations,
        "seed": args.seed,
        "prompts": PROMPTS,
        "results": results,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    out_path.write_text(json.dumps(receipt, indent=2))
    print(f"[receipt] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
