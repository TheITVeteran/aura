#!/usr/bin/env python3
"""Measure the identity-collapse escape hatch, and whether SPARK-061 closes it.

v4's docstring asserts that on families where depth is destructive, the
cheapest way for an optimizer to satisfy a monotone-improvement mandate is to
drive the recurrent transformation toward the identity. That assertion has
been carried forward as a citation for several checkpoints. It decides whether
a resident-32B training run is worth launching, so it should be a measurement.

The sweep exists because the collapse knob is already in the execution spec.
The recurrent update is

    z_{t+1} = (1 - alpha) * z_t + alpha * RMSMatch(window(z_t), anchor)

so ``alpha -> 0`` **is** the identity operator, exactly and continuously. No
model surgery, no synthetic operator, no training loop and its stochasticity:
sweeping alpha walks the objective along the collapse axis and reads off which
end it prefers.

For each alpha and each task the tool evaluates both objectives on the same
trajectory:

* ``trajectory_loss_v4``          — final + improvement + oscillation
* ``progressive_objective_loss``  — the same three plus the displacement floor

and reports which alpha minimizes each. If v4 is minimized at the collapse end
while the progressive objective is minimized away from it, the hatch is real
and the new term closes it. If v4 is *not* minimized at collapse, that is
equally worth knowing and the receipt says so — this tool is allowed to
disagree with the docstring it was written to check.

Runs a real pretrained MLX Qwen (default: the 1.5B the F2/F6 legs used) over
real curriculum tasks. Forward evaluation only; no weights are written, no
holdout is opened, and nothing here is a capability claim.

Usage:
    tools/measure_progressive_collapse.py --out DIR [--model PATH]
                                          [--families khop,modular]
                                          [--depth 4] [--per-family 3]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

COLLAPSE_SWEEP_SCHEMA = "aura.spark061.collapse_sweep.v1"
DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
# alpha values spanning identity (0.002) to the live default (0.5). The low end
# is not zero because the spec forbids it; 0.002 moves the state by well under
# the displacement floor, which is the operational definition of collapse.
DEFAULT_ALPHAS = (0.002, 0.01, 0.05, 0.1, 0.25, 0.5)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 6) if math.isfinite(number) else None


def _tokenize_pair(tokenizer, prompt: str, answer: str) -> tuple[list[int], list[int]]:
    """Render through the model's real chat template, as training would."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    prompt_tokens = [int(token) for token in rendered]
    answer_tokens = [int(token) for token in tokenizer.encode(answer)]
    return prompt_tokens, answer_tokens


def measure(
    *,
    model_path: str,
    families: tuple[str, ...],
    depth: int,
    per_family: int,
    alphas: tuple[float, ...],
    seed: int,
) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import load

    from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
    from core.learning.progressive_recurrent_objective import (
        DEFAULT_DISPLACEMENT_FLOOR,
        measure_progressive_trajectory,
        progressive_objective_loss,
    )
    from core.learning.recurrence_curriculum import task_battery
    from core.learning.recurrence_native_objective_v4 import trajectory_loss_v4

    started = time.time()
    model, tokenizer = load(model_path)
    tasks = task_battery(
        list(families), [depth], per_family, seed=seed
    )

    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        spec = RLCExecutionSpec(
            n_slots=8,
            branch_roles=("constructive_solution",),
            recurrent_steps=depth,
            alpha=float(alpha),
            alpha_schedule="constant",
        )
        for task in tasks:
            prompt_tokens, answer_tokens = _tokenize_pair(
                tokenizer, task.prompt, task.answer
            )
            trajectory = measure_progressive_trajectory(
                model,
                prompt_tokens,
                answer_tokens,
                spec=spec,
                depth=depth,
            )
            v4_loss, v4_telemetry = trajectory_loss_v4(
                model,
                prompt_tokens,
                answer_tokens,
                spec=spec,
                depth=depth,
            )
            progressive_loss, progressive_telemetry = progressive_objective_loss(
                model,
                prompt_tokens,
                answer_tokens,
                spec=spec,
                depth=depth,
            )
            mx.eval(v4_loss, progressive_loss)
            rows.append(
                {
                    "alpha": round(float(alpha), 6),
                    "family": task.family,
                    "task_id": task.task_id,
                    "answer_tokens": len(answer_tokens),
                    "min_displacement": _finite(trajectory.min_displacement),
                    "step_losses": [
                        round(value, 6) for value in trajectory.step_losses
                    ],
                    "improvement": _finite(trajectory.improvement),
                    "v4_loss": _finite(v4_loss),
                    "v4_improvement_penalty": v4_telemetry["improvement_penalty"],
                    "progressive_loss": _finite(progressive_loss),
                    "progressive_displacement_penalty": progressive_telemetry[
                        "displacement_penalty"
                    ],
                }
            )
            print(
                f"  alpha={alpha:<6} {task.family:<8} "
                f"disp={trajectory.min_displacement:.5f} "
                f"v4={float(v4_loss):.4f} prog={float(progressive_loss):.4f}",
                flush=True,
            )

    # Aggregate per alpha, then per family: which end of the collapse axis does
    # each objective prefer?
    def mean(values: list[float]) -> float | None:
        clean = [value for value in values if value is not None]
        return round(sum(clean) / len(clean), 6) if clean else None

    per_alpha: list[dict[str, Any]] = []
    for alpha in alphas:
        subset = [row for row in rows if row["alpha"] == round(float(alpha), 6)]
        per_alpha.append(
            {
                "alpha": round(float(alpha), 6),
                "mean_v4_loss": mean([row["v4_loss"] for row in subset]),
                "mean_progressive_loss": mean(
                    [row["progressive_loss"] for row in subset]
                ),
                "mean_min_displacement": mean(
                    [row["min_displacement"] for row in subset]
                ),
                "mean_improvement": mean([row["improvement"] for row in subset]),
                "below_displacement_floor": sum(
                    1
                    for row in subset
                    if row["min_displacement"] is not None
                    and row["min_displacement"] < DEFAULT_DISPLACEMENT_FLOOR
                ),
                "sample_count": len(subset),
            }
        )

    def argmin(key: str) -> float | None:
        scored = [row for row in per_alpha if row[key] is not None]
        if not scored:
            return None
        return min(scored, key=lambda row: row[key])["alpha"]

    v4_preferred = argmin("mean_v4_loss")
    progressive_preferred = argmin("mean_progressive_loss")
    collapse_alpha = min(alphas)
    hatch_open = v4_preferred == collapse_alpha
    hatch_closed = hatch_open and progressive_preferred != collapse_alpha

    if hatch_open and hatch_closed:
        finding = "hatch_open_and_closed_by_displacement_term"
    elif hatch_open:
        finding = "hatch_open_and_still_open"
    else:
        finding = "hatch_not_reproduced_at_this_operating_point"

    payload = {
        "schema": COLLAPSE_SWEEP_SCHEMA,
        "model_path": model_path,
        "families": list(families),
        "depth": depth,
        "per_family": per_family,
        "seed": seed,
        "alphas": [round(float(value), 6) for value in alphas],
        "collapse_alpha": round(float(collapse_alpha), 6),
        "displacement_floor": DEFAULT_DISPLACEMENT_FLOOR,
        "task_count": len(tasks),
        "row_count": len(rows),
        "rows": rows,
        "per_alpha": per_alpha,
        "v4_preferred_alpha": v4_preferred,
        "progressive_preferred_alpha": progressive_preferred,
        "collapse_hatch_open_in_v4": hatch_open,
        "collapse_hatch_closed_by_progressive": hatch_closed,
        "finding": finding,
        "elapsed_s": round(time.time() - started, 3),
        "capability_claim": "none",
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--families", default="khop,modular")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--per-family", type=int, default=3)
    parser.add_argument("--seed", type=int, default=61)
    args = parser.parse_args()

    families = tuple(
        item.strip() for item in args.families.split(",") if item.strip()
    )
    receipt = measure(
        model_path=args.model,
        families=families,
        depth=args.depth,
        per_family=args.per_family,
        alphas=DEFAULT_ALPHAS,
        seed=args.seed,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "collapse_sweep.json"
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    print("\n=== collapse sweep ===")
    for row in receipt["per_alpha"]:
        print(
            f"alpha={row['alpha']:<7} v4={row['mean_v4_loss']} "
            f"prog={row['mean_progressive_loss']} "
            f"disp={row['mean_min_displacement']} "
            f"below_floor={row['below_displacement_floor']}/{row['sample_count']}"
        )
    print(f"v4 prefers alpha={receipt['v4_preferred_alpha']}")
    print(f"progressive prefers alpha={receipt['progressive_preferred_alpha']}")
    print(f"finding: {receipt['finding']}")
    print(f"receipt: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
