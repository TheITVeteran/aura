#!/usr/bin/env python3
"""SPARK-013 semantic leg: state causality on a real pretrained model.

Runs the full seven-arm state-causality experiment on a real pretrained
small model (default: the locally cached Qwen2.5-1.5B-Instruct 4-bit MLX
build), measuring task-appropriate loss through the live slot channel and
the structural byte-identities on real weights.  Never touches the
resident 32B.  Writes the sealed receipt (plus independent replay) under
artifacts/closeout/latent_cortex/spark013_state_causality/.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
DEFAULT_OUTPUT = "artifacts/closeout/latent_cortex/spark013_state_causality"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tasks", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--recurrent-steps", type=int, default=3)
    parser.add_argument("--n-slots", type=int, default=8)
    parser.add_argument("--decode-max-tokens", type=int, default=16)
    parser.add_argument("--episode-wall-clock-s", type=float, default=180.0)
    parser.add_argument("--deadline-minutes", type=float, default=45.0)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--pilot", action="store_true", help="run 1 task and exit")
    arguments = parser.parse_args()

    output_root = Path(arguments.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AURA_LOG_DIR", str(output_root / "logs"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from mlx_lm import load

    from core.brain.llm.latent_cortex.engine import LatentCortexEngine
    from core.brain.llm.latent_cortex.state_causality import (
        build_state_binding_tasks,
        engine_episode_runner,
        replay_state_causality_receipt,
        run_state_causality_experiment,
    )
    from core.brain.llm.latent_cortex.types import (
        BranchConfig,
        CortexConfig,
        LatentOptConfig,
        RecurrenceConfig,
        WorkspaceConfig,
    )
    from core.runtime.model_lane_control import acquire_standalone_model_lane

    print(f"loading {arguments.model} ...", flush=True)
    model_lane = acquire_standalone_model_lane(
        owner_id=f"state-causality:{output_root.name}",
        model_path=arguments.model,
        purpose="evaluation",
        preemptible=False,
        metadata={"tool": "run_state_causality_semantic", "operator_launched": True},
    )
    atexit.register(model_lane.release)
    model, tokenizer = load(arguments.model)

    def build_engine() -> LatentCortexEngine:
        return LatentCortexEngine(
            model,
            tokenizer,
            config=CortexConfig(
                workspace=WorkspaceConfig(n_slots=arguments.n_slots, seed=23),
                recurrence=RecurrenceConfig(
                    min_steps=arguments.recurrent_steps,
                    max_steps=arguments.recurrent_steps,
                    convergence_eps=1e-12,
                ),
                branches=BranchConfig(n_branches=1),
                latent_opt=LatentOptConfig(enabled=False),
                decode_max_tokens=arguments.decode_max_tokens,
            ),
        )

    runner = engine_episode_runner(
        build_engine,
        decode_max_tokens=arguments.decode_max_tokens,
        wall_clock_s=arguments.episode_wall_clock_s,
    )
    task_count = 1 if arguments.pilot else arguments.tasks
    tasks = build_state_binding_tasks(count=task_count, seed=arguments.seed)

    deadline = time.monotonic() + arguments.deadline_minutes * 60.0
    completed_episodes = 0

    def _bounded_runner(prompt: str, cognitive_context: list) -> dict:
        nonlocal completed_episodes
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"deadline reached after {completed_episodes} episodes"
            )
        outcome = runner(prompt, cognitive_context)
        completed_episodes += 1
        if completed_episodes % 7 == 0:
            print(
                f"  {completed_episodes} episodes "
                f"({time.monotonic() - start:.0f}s elapsed)",
                flush=True,
            )
        return outcome

    start = time.monotonic()
    receipt = run_state_causality_experiment(
        _bounded_runner,
        tasks,
        minimum_tasks=1 if arguments.pilot else 20,
        runner_identity={
            "model": arguments.model,
            "n_slots": arguments.n_slots,
            "recurrent_steps": arguments.recurrent_steps,
            "decode_max_tokens": arguments.decode_max_tokens,
            "seed": arguments.seed,
        },
    )
    replay = replay_state_causality_receipt(receipt)
    elapsed = time.monotonic() - start

    suffix = "pilot" if arguments.pilot else "full"
    receipt_path = output_root / f"state_causality_receipt_{suffix}.json"
    payload = json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    receipt_path.write_bytes(payload + b"\n")

    print(f"\nelapsed: {elapsed:.0f}s over {completed_episodes} episodes")
    print(f"receipt: {receipt_path}")
    print(f"receipt_sha256: {receipt['receipt_sha256']}")
    print(f"replay_verified: {replay['replayed']}")
    print("\nclaims:")
    for claim in receipt["claims"]:
        print(f"  {claim['tier']:10s} {claim['experiment']}")
        if claim["experiment"] == "expS_task_appropriate_loss":
            evidence = claim["evidence"]
            print(
                f"             intact={evidence['intact_rate']:.2f} "
                f"lesioned={evidence['lesioned_rate']:.2f} "
                f"sham={evidence['sham_rate']:.2f} "
                f"substitution_tracks={evidence['substitution_tracks_rate']:.2f} "
                f"readable={evidence['channel_readable']}"
            )
    model_lane.release()
    atexit.unregister(model_lane.release)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
