#!/usr/bin/env python3
"""Chain: train, then run the exact evals we failed (CP238).

Bryan's directive: build everything, run it, then run the evals and tests we
failed, then go silent. This is the chaining. Two stages, each a
subprocess, with a stage manifest so the campaign resumes at STAGE
granularity if the machine dies between stages -- the GRPO trainer already
resumes at STEP granularity within stage 1 (CP237).

    stage 1  GRPO training               (verifier-driven RL, CP233)
    stage 2  accuracy gate               (the exact CP227 gate that failed)
    stage 3  integrated reasoning eval    (organs + recurrence, CP236)

Why stages as subprocesses rather than one process: a crash or OOM in the
eval must not destroy the training receipt, subprocess isolation frees the
32B between stages (in-process would hold two resident models = OOM), and a
completed stage must not re-run on resume.

INCOMPATIBLE WITH run_detached_step.py's sandbox: the supervisor blocks its
target from spawning child processes (PermissionError: Operation not
permitted on fork_exec) as a containment measure. So this campaign must run
UNSUPERVISED (plain caffeinate/nohup) -- or each stage must be supervised as
its own separate detached job. The CP238 launch supervises train_grpo
DIRECTLY (it spawns nothing) and chains the integrated eval as a second
supervised job on completion. Kept here for the unsupervised path and as
the stage definition of record.

The campaign never claims a gain. It runs the three measurements and writes
one combined receipt with every verdict, positive or negative. The point of
running the gate we already failed is to see whether GRPO -- which
optimizes CORRECTNESS, the thing CE-training missed -- converts where
intrinsic-recurrence CE-training did not.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CAMPAIGN_SCHEMA = "aura.reasoning_campaign.v1"
PYTHON = str(REPO / ".venv" / "bin" / "python")


def _run_stage(name: str, argv: list[str], log: Path) -> dict:
    """Run one stage, streaming its output to a durable log."""
    started = time.time()
    with log.open("a") as sink:
        sink.write(f"\n===== stage {name} @ {time.ctime()} =====\n")
        sink.flush()
        result = subprocess.run(
            argv, stdout=sink, stderr=subprocess.STDOUT, cwd=str(REPO)
        )
    return {
        "stage": name,
        "returncode": result.returncode,
        "ok": result.returncode == 0,
        "minutes": round((time.time() - started) / 60.0, 2),
    }


def _manifest(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (ValueError, OSError):
            pass
    return {"schema": CAMPAIGN_SCHEMA, "completed": [], "stages": {}}


def _save_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--domains", default="arithmetic_chain,program_trace,constraint_order")
    parser.add_argument("--depths", default="2,4,8")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--gate-depths", default="1,2,4")
    parser.add_argument("--gate-per-cell", type=int, default=16)
    parser.add_argument("--intrinsic-adapter",
                        default="artifacts/closeout/latent_cortex/cp227_intrinsic_training",
                        help="the CP227 recurrence adapter the depth gate is for")
    parser.add_argument("--train-minutes", type=float, default=480.0)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--memory-fraction", type=float, default=0.5)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "campaign_manifest.json"
    manifest = _manifest(manifest_path)
    log = out / "campaign.log"
    adapter_dir = out / "grpo"

    stages = [
        (
            "train_grpo",
            [
                PYTHON, "tools/train_grpo.py",
                "--model", args.model, "--out-dir", str(adapter_dir),
                "--domains", args.domains, "--depths", args.depths,
                "--max-steps", str(args.max_steps),
                "--group-size", str(args.group_size),
                "--max-minutes", str(args.train_minutes),
                "--seed", str(args.seed),
                "--memory-fraction", str(args.memory_fraction),
            ],
        ),
        (
            "integrated_eval",
            [
                PYTHON, "tools/eval_integrated_reasoning.py",
                "--model", args.model,
                "--adapter", str(adapter_dir),
                "--out", str(out / "integrated_eval.json"),
                "--per-cell", str(args.gate_per_cell),
                "--seed", str(args.seed + 404),
                "--memory-fraction", str(args.memory_fraction),
            ],
        ),
    ]

    for name, argv in stages:
        if name in manifest["completed"]:
            print(f"[skip] {name} already complete", flush=True)
            continue
        print(f"[stage] {name} starting", flush=True)
        result = _run_stage(name, argv, log)
        manifest["stages"][name] = result
        if result["ok"]:
            manifest["completed"].append(name)
        _save_manifest(manifest_path, manifest)
        print(f"[stage] {name} -> {result}", flush=True)
        if not result["ok"]:
            # A failed stage stops the chain: the downstream evals depend on
            # the trained adapter, and running them on a broken artifact
            # would produce a confident, meaningless number. Resumable: fix
            # and re-launch, completed stages are skipped.
            print(f"[halt] {name} failed rc={result['returncode']}", flush=True)
            manifest["final"] = {"halted_at": name}
            _save_manifest(manifest_path, manifest)
            return result["returncode"]

    manifest["final"] = {"all_stages_complete": True}
    _save_manifest(manifest_path, manifest)
    print(f"[campaign] complete -> {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
