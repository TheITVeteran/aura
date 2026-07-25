#!/usr/bin/env python3
"""SPARK-070 dry run: execute every runnable falsification-matrix row.

Loads a real pretrained small model (never the resident 32B), builds the
lab-style solver closures, runs each runnable row of the falsification
matrix against the current engine, binds enforced rows to the threat-model
registry and blocked rows to their named SPARK blockers, and writes one
replayable matrix receipt.  The acceptance event stays the post-training
run on fresh held-out tasks against the SPARK-069 treatment; this tool
proves the harness and records the untrained-baseline dry-run evidence.
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
DEFAULT_OUTPUT = "artifacts/closeout/latent_cortex/spark070_falsification_matrix"
STATE_CAUSALITY_RECEIPT = (
    "artifacts/closeout/latent_cortex/spark013_state_causality/"
    "state_causality_receipt_full.json"
)


class MatrixDeadlineError(RuntimeError):
    pass


def _scrub_nonfinite(value):
    """Replace NaN/inf with None so canonical hashing stays well-defined."""

    import math

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _scrub_nonfinite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub_nonfinite(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--families", default="modular,boolean")
    parser.add_argument("--depths", default="2,3")
    parser.add_argument("--steps", default="1,2,4")
    parser.add_argument("--per-cell", type=int, default=4)
    parser.add_argument("--n-slots", type=int, default=8)
    parser.add_argument("--branches", type=int, default=2)
    parser.add_argument("--task-seed", type=int, default=20260724)
    parser.add_argument("--max-minutes", type=float, default=35.0)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--state-causality-receipt", default=STATE_CAUSALITY_RECEIPT
    )
    arguments = parser.parse_args()

    output_root = Path(arguments.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AURA_LOG_DIR", str(output_root / "logs"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from mlx_lm import load

    from core.brain.llm.latent_cortex.capability_canaries import (
        CapabilityCanaries,
        compare_canaries,
    )
    from core.brain.llm.latent_cortex.engine import LatentCortexEngine
    from core.brain.llm.latent_cortex.experiments import (
        run_depth_extrapolation,
        run_latent_opt_control,
        run_recurrence_sweep,
        run_role_lesion,
        run_slot_causality,
        task_battery,
    )
    from core.brain.llm.latent_cortex.falsification_matrix import (
        assemble_falsification_matrix_receipt,
        replay_falsification_matrix_receipt,
        run_transition_matrix,
        validate_falsification_matrix,
    )
    from core.brain.llm.latent_cortex.state_causality import (
        replay_state_causality_receipt,
    )
    from core.brain.llm.latent_cortex.threat_model import validate_threat_model
    from core.brain.llm.latent_cortex.types import (
        BranchConfig,
        ComputeBudget,
        CortexConfig,
        LatentOptConfig,
        RecurrenceConfig,
        WorkspaceConfig,
    )
    from core.runtime.model_lane_control import acquire_standalone_model_lane

    families = [item.strip() for item in arguments.families.split(",") if item.strip()]
    depths = [int(item) for item in arguments.depths.split(",")]
    steps = sorted({int(item) for item in arguments.steps.split(",")})
    deadline = time.monotonic() + arguments.max_minutes * 60.0
    start = time.monotonic()

    print(f"loading {arguments.model} ...", flush=True)
    model_lane = acquire_standalone_model_lane(
        owner_id=f"falsification-matrix:{output_root.name}",
        model_path=arguments.model,
        purpose="evaluation",
        preemptible=False,
        metadata={"tool": "run_falsification_matrix", "operator_launched": True},
    )
    atexit.register(model_lane.release)
    model, tokenizer = load(arguments.model)

    def make_engine(
        max_steps: int,
        *,
        latent_opt: str = "off",
        roles: tuple[str, ...] = (),
    ) -> LatentCortexEngine:
        branch_kwargs: dict = {"n_branches": arguments.branches}
        if roles:
            branch_kwargs["roles"] = roles
        return LatentCortexEngine(
            model,
            tokenizer,
            CortexConfig(
                workspace=WorkspaceConfig(n_slots=arguments.n_slots, seed=7),
                recurrence=RecurrenceConfig(
                    max_steps=max_steps, min_steps=max_steps, convergence_eps=1e-9
                ),
                branches=BranchConfig(**branch_kwargs),
                latent_opt=LatentOptConfig(
                    enabled=latent_opt != "off",
                    control_mode=latent_opt == "control",
                    steps=4,
                ),
                decode_max_tokens=64,
            ),
        )

    episode_count = 0

    def solve(
        task,
        n_steps: int,
        *,
        latent_opt: str = "off",
        ablate=None,
        roles: tuple[str, ...] = (),
    ) -> tuple[bool, int]:
        nonlocal episode_count
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise MatrixDeadlineError("matrix wall-clock bound reached")
        engine = make_engine(n_steps, latent_opt=latent_opt, roles=roles)
        budget = ComputeBudget(wall_clock_s=min(120.0, remaining))
        result = engine.reason(
            messages=[{"role": "user", "content": task.prompt}],
            budget=budget,
            ablate_slot=ablate,
            decode_max_tokens=64,
        )
        episode_count += 1
        if episode_count % 25 == 0:
            print(
                f"  {episode_count} episodes ({time.monotonic() - start:.0f}s)",
                flush=True,
            )
        return (bool(result.ok and task.verify(result.text)), budget.spent_layer_apps)

    battery = task_battery(families, depths, arguments.per_cell, seed=arguments.task_seed)
    by_family = {
        family: [task for task in battery if task.family == family]
        for family in families
    }
    print(
        f"battery: {len(battery)} tasks over {families} at depths {depths}",
        flush=True,
    )

    row_results: dict[str, dict] = {}

    print("row: recurrence_depth_curves", flush=True)
    row_results["recurrence_depth_curves"] = run_recurrence_sweep(
        lambda task, n: solve(task, n), battery, steps
    )

    print("row: wrong_right_transition_matrix", flush=True)
    row_results["wrong_right_transition_matrix"] = run_transition_matrix(
        lambda task, n: solve(task, n),
        battery,
        shallow_steps=steps[0],
        deep_steps=steps[-1],
    )

    print("row: structural_diversity_arms", flush=True)
    from core.brain.llm.latent_cortex.branches import BRANCH_ROLES

    def solve_role_arm(task, arm: str):
        k = max(2, arguments.branches)
        base_roles = tuple(
            BRANCH_ROLES[index % len(BRANCH_ROLES)] for index in range(k)
        )
        if arm in {"distinct_roles", "restored_roles"}:
            roles = base_roles
        elif arm == "lesioned_uniform_role":
            roles = (base_roles[0],) * k
        elif arm == "swapped_roles":
            roles = base_roles[1:] + base_roles[:1]
        else:
            raise ValueError(f"unknown role arm: {arm}")
        ok, cost = solve(task, steps[-1], roles=roles)
        return (ok, cost, None)

    row_results["structural_diversity_arms"] = run_role_lesion(
        solve_role_arm, by_family
    )

    print("row: latent_ablate_perturb_transplant", flush=True)
    row_results["latent_ablate_perturb_transplant"] = run_slot_causality(
        lambda task, slot: solve(task, steps[-1], ablate=slot)[0],
        battery,
        slot_indices=list(
            range(0, arguments.n_slots, max(1, arguments.n_slots // 4))
        ),
    )

    print("row: sham_noise_noop_controls", flush=True)
    row_results["sham_noise_noop_controls"] = run_latent_opt_control(
        lambda task, arm: solve(task, steps[-1], latent_opt=arm), by_family
    )

    print("row: compute_generalization", flush=True)
    extrapolation: dict[str, dict] = {}
    for family in families:
        extrapolation[family] = run_depth_extrapolation(
            lambda task, n: solve(task, n),
            family,
            depths + [depths[-1] * 2],
            steps,
            per_depth=max(2, arguments.per_cell // 2),
            seed=arguments.task_seed + 1,
        )
    row_results["compute_generalization"] = extrapolation

    print("row: lesions_restorations (bind existing replayed receipt)", flush=True)
    causality_path = Path(arguments.state_causality_receipt)
    if not causality_path.is_absolute():
        causality_path = REPO_ROOT / causality_path
    causality_receipt = json.loads(causality_path.read_bytes())
    replay_state_causality_receipt(causality_receipt)
    row_results["lesions_restorations"] = causality_receipt

    print("row: non_reasoning_regressions", flush=True)
    import mlx.core as mx

    canaries = CapabilityCanaries(
        tokenizer, vocab_size=int(model.model.embed_tokens.weight.shape[0])
    )

    def logits_fn(tokens: list[int]):
        return model(mx.array([tokens]))

    baseline_canaries = canaries.measure(logits_fn)
    row_results["non_reasoning_regressions"] = {
        "baseline_canaries": {
            name: round(value, 6) for name, value in baseline_canaries.items()
        },
        "self_comparison": compare_canaries(
            baseline_canaries, baseline_canaries, max_logprob_drop=0.5
        ),
        "note": (
            "dry run measures the untrained baseline battery; the "
            "acceptance run compares the SPARK-069 treatment against it"
        ),
    }

    row_results = {
        row_id: _scrub_nonfinite(payload)
        for row_id, payload in row_results.items()
    }
    threat_registry = validate_threat_model()
    matrix_registry = validate_falsification_matrix()
    receipt = assemble_falsification_matrix_receipt(
        row_results=row_results,
        runner_identity={
            "mode": "pre_training_dry_run",
            "model": arguments.model,
            "families": families,
            "depths": depths,
            "steps": steps,
            "per_cell": arguments.per_cell,
            "n_slots": arguments.n_slots,
            "branches": arguments.branches,
            "task_seed": arguments.task_seed,
            "episodes": episode_count,
            "elapsed_s": round(time.monotonic() - start, 1),
        },
        threat_model_registry_sha256=threat_registry["registry_sha256"],
    )
    replay = replay_falsification_matrix_receipt(
        receipt, row_payloads=row_results
    )

    receipt_path = output_root / "matrix_receipt_dryrun.json"
    payload_path = output_root / "matrix_row_payloads_dryrun.json"
    receipt_path.write_bytes(
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        + b"\n"
    )
    payload_path.write_bytes(
        json.dumps(
            row_results, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        + b"\n"
    )

    print(f"\nepisodes: {episode_count} in {time.monotonic() - start:.0f}s")
    print(f"matrix registry: {matrix_registry['registry_sha256']}")
    print(f"receipt: {receipt_path}")
    print(f"receipt_sha256: {receipt['receipt_sha256']}")
    print(f"replay_verified: {replay['replayed']}")
    print("\nrows:")
    for row in receipt["rows"]:
        tiers = ",".join(
            f"{item['experiment']}={item['tier']}" for item in row["claim_tiers"]
        )
        blockers = (
            " blockers=" + ",".join(str(b) for b in row["blockers"])
            if row["blockers"]
            else ""
        )
        print(f"  {row['status']:9s} {row['row_id']}{blockers} {tiers[:120]}")
    model_lane.release()
    atexit.unregister(model_lane.release)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
