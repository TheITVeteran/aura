"""Does giving the recurrence a PHASE break the fixed point?

Measured: the recurrent operator is a contraction (residual 0.302 -> 0.026,
asymptoting) so it converges by step ~10 and stops computing. Root cause:
every step applies the identical operator, so no step can know which step
it is, and no staged algorithm is expressible.

Cheapest possible intervention: inject a per-step sinusoidal phase signal
into the slot states -- the same trick positional encoding uses to make
identical tokens behave differently. No new weights, no new parameters.

If phase is the missing ingredient we expect:
  * residual stops collapsing (the operator keeps computing), and
  * answer CE keeps improving past the depth-8 ceiling.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path("/Users/bryan/.aura/live-source")
sys.path.insert(0, str(REPO))

import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec  # noqa: E402
from core.learning import recurrence_curriculum as curriculum  # noqa: E402
import core.learning.recurrence_native_objective_v2 as objective  # noqa: E402

MODEL = str(REPO / "models/Qwen2.5-1.5B-Instruct-4bit")
DEPTHS = (1, 2, 4, 8, 16, 24)
PHASE_SCALE = 0.08  # relative to state RMS


def phase_vector(step: int, hidden: int) -> Any:
    """Fixed sinusoidal phase code for a recurrent step (no parameters)."""
    positions = mx.arange(hidden, dtype=mx.float32)
    frequency = mx.exp(
        -math.log(10000.0) * (2 * mx.floor(positions / 2)) / hidden
    )
    angle = float(step) * frequency
    return mx.where(positions % 2 == 0, mx.sin(angle), mx.cos(angle))


def install_phase(enabled: bool):
    """Wrap _window_pass so each recurrent step sees its own phase."""
    original = objective._window_pass
    counter = {"step": 0}
    captured: list = []

    def wrapped(model, prompt_at_window, slots, prelude_end, coda_start):
        if enabled:
            hidden = int(slots.shape[-1])
            rms = mx.sqrt(mx.mean(mx.square(slots)) + 1e-9)
            code = phase_vector(counter["step"], hidden)[None, None, :]
            slots = slots + PHASE_SCALE * rms * code
        counter["step"] += 1
        result = original(model, prompt_at_window, slots, prelude_end, coda_start)
        captured.append(result)
        return result

    objective._window_pass = wrapped
    return original, counter, captured


def run(model, tokenizer, task, depth: int, phase: bool):
    original, counter, captured = install_phase(phase)
    try:
        prompt = list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": task.prompt}],
                add_generation_prompt=True,
                tokenize=True,
            )
        )
        answer = list(tokenizer.encode(task.answer, add_special_tokens=False))
        spec = RLCExecutionSpec(
            n_slots=4,
            branch_roles=("constructive_solution",),
            recurrent_steps=max(DEPTHS),
            exchange_interval=1,
        )
        value = objective.live_path_loss(
            model, prompt, answer, spec=spec.with_depth(depth)
        )
        mx.eval(value)
        loss = float(value)
    finally:
        objective._window_pass = original
    residuals = []
    for index in range(1, len(captured)):
        delta = mx.linalg.norm(
            mx.reshape(captured[index] - captured[index - 1], (-1,))
        )
        scale = mx.maximum(
            mx.linalg.norm(mx.reshape(captured[index], (-1,))), 1e-9
        )
        residuals.append(float(delta / scale))
    return loss, residuals


def main() -> int:
    print(f"loading {MODEL}\n", flush=True)
    model, tokenizer = load(MODEL)
    tasks = curriculum.task_battery(["khop"], [8], 3, seed=4242)

    results = {}
    for phase in (False, True):
        label = "PHASE" if phase else "BASE "
        print(f"=== {label}: answer CE by depth ===")
        per_depth = {}
        for depth in DEPTHS:
            losses = [run(model, tokenizer, t, depth, phase)[0] for t in tasks]
            per_depth[depth] = sum(losses) / len(losses)
            base = per_depth[DEPTHS[0]]
            print(
                f"  depth {depth:3d}: CE={per_depth[depth]:7.4f}  "
                f"delta_vs_d1={100*(per_depth[depth]-base)/base:+6.1f}%"
            )
        results["phase" if phase else "base"] = {
            str(k): v for k, v in per_depth.items()
        }
        best = min(per_depth, key=per_depth.get)
        print(f"  best depth = {best}, CE = {per_depth[best]:.4f}\n")
        results[("phase" if phase else "base") + "_best_depth"] = best

    # Residual behaviour at max depth: does phase keep the operator working?
    _loss, base_res = run(model, tokenizer, tasks[0], max(DEPTHS), False)
    _loss, phase_res = run(model, tokenizer, tasks[0], max(DEPTHS), True)

    def late_over_early(values):
        early = sum(values[:3]) / max(1, len(values[:3]))
        late = sum(values[-3:]) / max(1, len(values[-3:]))
        return late / max(early, 1e-9)

    print("=== residual decay (lower = converged/stopped computing) ===")
    print(f"  base  late/early = {late_over_early(base_res):.4f}")
    print(f"  phase late/early = {late_over_early(phase_res):.4f}")
    results["base_residual_ratio"] = late_over_early(base_res)
    results["phase_residual_ratio"] = late_over_early(phase_res)

    base_best = results["base"][str(results["base_best_depth"])]
    phase_best = results["phase"][str(results["phase_best_depth"])]
    gain = 100 * (base_best - phase_best) / base_best
    print(f"\nbest-CE improvement from phase: {gain:+.2f}%")
    results["phase_gain_pct"] = gain
    out = Path(__file__).with_name("phase_probe_result.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
