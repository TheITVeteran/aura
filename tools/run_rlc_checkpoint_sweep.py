#!/usr/bin/env python
"""Find whether any retained training checkpoint beats an ordinary decode.

The cp796 run published 97 immutable optimizer generations and only its last
one was ever evaluated behaviorally. Its selection metric could not have
caught what went wrong: validation cross-entropy fell smoothly from 3.347 to
2.072 across the whole run while the model was learning to answer without
reasoning, because answer-span cross-entropy falls precisely as the answer
becomes more immediately predictable.

So the run is not one result. It is 97, of which one was measured.

This tool bisects them on the signal that actually separates a reasoning
checkpoint from a collapsed one -- whether the response has any work before
its answer -- and then scores the surviving candidate properly. Bisection is
what makes it affordable: locating the collapse costs about seven probes
instead of ninety-seven.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

CHECKPOINT_SWEEP_SCHEMA = "aura.rlc_checkpoint_sweep.v1"
ANSWER_MARKER = "FINAL_ANSWER:"


def _atomic_write(path: Path, payload: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def _answered_without_working(text: str) -> bool:
    index = text.find(ANSWER_MARKER)
    if index < 0:
        return False
    return not text[:index].strip()


def _generations(root: Path) -> list[Path]:
    return sorted(
        (p for p in root.glob("sequence-*") if (p / "adapter.safetensors").exists()),
        key=lambda p: p.name,
    )


def _status(out_dir: Path, **fields: Any) -> None:
    path = out_dir / "status.json"
    body: dict[str, Any] = {}
    if path.exists():
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            body = {}
    body.update(fields)
    body["heartbeat_unix"] = time.time()
    body["pid"] = os.getpid()
    _atomic_write(path, json.dumps(body, indent=1, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reference-arm", default="rlc_nodisp")
    parser.add_argument("--vanilla-correct", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--probe-tasks", type=int, default=4)
    parser.add_argument("--score-tasks", type=int, default=4)
    parser.add_argument("--n-slots", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--memory-fraction", type=float, default=0.40)
    args = parser.parse_args()

    import run_rlc_reconciliation_sweep as sweep

    from core.brain.llm.latent_cortex import frontier_tasks as ft

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.checkpoint_root)
    generations = _generations(root)
    if not generations:
        _atomic_write(
            out_dir / "verdict.json",
            json.dumps(
                {
                    "schema": CHECKPOINT_SWEEP_SCHEMA,
                    "beats_ordinary_decode": False,
                    "reason": "no_retained_generations",
                    "checkpoint_root": str(root),
                },
                indent=1,
            )
            + "\n",
        )
        print("no retained generations", file=sys.stderr)
        return 0

    arm = next((a for a in sweep.ARMS if a[0] == args.reference_arm), None)
    if arm is None or arm[1] is None:
        print(f"reference arm {args.reference_arm} is not a recurrent arm", file=sys.stderr)
        return 2
    _, steps, policy = arm
    config = sweep._build_config(steps, args.n_slots, policy, args.max_tokens)

    seeds = [args.seed + i for i in range(max(args.probe_tasks, args.score_tasks))]
    tasks = ft.generate_task_battery(seeds, difficulty=2)
    probe_tasks = tasks[: args.probe_tasks]

    print(f"{len(generations)} retained generations", flush=True)
    _status(out_dir, phase="loading_model", generations=len(generations))

    from mlx_lm import load

    from core.brain.llm.latent_cortex.resident_adapter_loader import (
        load_resident_adapter,
    )
    from core.runtime.mlx_memory_guard import mlx_memory_envelope
    from core.runtime.model_lane_control import standalone_model_lane

    manifest_path = root.parent / "recurrence_adapter_manifest.json"
    if not manifest_path.exists():
        found = list(root.parent.rglob("recurrence_adapter_manifest.json"))
        manifest_path = found[0] if found else manifest_path
    if not manifest_path.exists():
        _atomic_write(
            out_dir / "verdict.json",
            json.dumps(
                {
                    "schema": CHECKPOINT_SWEEP_SCHEMA,
                    "beats_ordinary_decode": False,
                    "reason": "no_adapter_manifest_for_retained_generations",
                },
                indent=1,
            )
            + "\n",
        )
        print("no adapter manifest; cannot attach retained generations", file=sys.stderr)
        return 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    results: dict[str, dict[str, Any]] = {}

    with standalone_model_lane(
        owner_id=f"rlc-checkpoint-sweep:{os.getpid()}",
        model_path=args.model,
        purpose="evaluation",
        preemptible=False,
        metadata={"tool": "run_rlc_checkpoint_sweep", "operator_launched": True},
    ), mlx_memory_envelope(fraction=args.memory_fraction) as envelope:
        def probe(generation: Path) -> dict[str, Any]:
            """Reload the base and attach exactly one generation."""
            model, tokenizer = load(args.model)
            load_resident_adapter(model, generation, manifest)
            answered_blind = 0
            correct = 0
            for task in probe_tasks:
                text, _receipt = sweep._run_rlc(
                    model, config, sweep._render_prompt(tokenizer, task), tokenizer
                )
                answered_blind += int(_answered_without_working(text))
                correct += int(ft.score_task(task, text).correct)
                envelope.reclaim(force=True)
            del model, tokenizer
            envelope.reclaim(force=True)
            return {"answered_blind": answered_blind, "correct": correct}

        # Bisect for the earliest generation that answers without working on
        # every probe task -- the point the checkpoint stopped reasoning.
        low, high = 0, len(generations) - 1
        collapse_index = None
        while low <= high:
            mid = (low + high) // 2
            name = generations[mid].name
            if name not in results:
                results[name] = probe(generations[mid])
                _status(
                    out_dir,
                    phase="bisecting",
                    probed=len(results),
                    last=name,
                    last_result=results[name],
                )
                print(f"  probe {name} -> {results[name]}", flush=True)
            if results[name]["answered_blind"] >= len(probe_tasks):
                collapse_index = mid
                high = mid - 1
            else:
                low = mid + 1

        candidate_index = (
            len(generations) - 1 if collapse_index is None else max(collapse_index - 1, 0)
        )
        candidate = generations[candidate_index]
        if candidate.name not in results:
            results[candidate.name] = probe(candidate)
        print(f"candidate {candidate.name} -> {results[candidate.name]}", flush=True)

    best_correct = results[candidate.name]["correct"]
    scaled_vanilla = round(args.vanilla_correct * len(probe_tasks) / 28.0, 3)
    verdict = {
        "schema": CHECKPOINT_SWEEP_SCHEMA,
        "checkpoint_root": str(root),
        "generations": len(generations),
        "probed": results,
        "collapse_generation": (
            None if collapse_index is None else generations[collapse_index].name
        ),
        "best_checkpoint": candidate.name,
        "best_correct": best_correct,
        "probe_tasks": len(probe_tasks),
        "reference_arm": args.reference_arm,
        "ordinary_decode_correct_scaled": scaled_vanilla,
        "beats_ordinary_decode": best_correct > scaled_vanilla,
        "claims": {
            "reasoning_gain_proven": False,
            "fusion_authorized": False,
        },
    }
    _atomic_write(out_dir / "verdict.json", json.dumps(verdict, indent=1, sort_keys=True) + "\n")
    _status(out_dir, phase="complete")
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
