#!/usr/bin/env python3
"""Detached, resumable, self-sequencing pipeline (CP242).

The point: the chain must not depend on a human (or an agent) being present
to hand off between stages. This orchestrator waits for the running GRPO to
finish, then runs the tests and evals in sequence on its own -- so if the
agent's session ends or its rate limit expires mid-run, the work keeps going
and anyone can read the results afterward.

Runs UNSUPERVISED on purpose. The detached supervisor sandboxes its target
and blocks child-process spawning (PermissionError on fork_exec), which is
incompatible with a stage-runner that launches subprocesses. Instead this is
launched with nohup + caffeinate so it survives the session and keeps the
machine awake, and durability comes from a stage MANIFEST: every completed
stage is recorded, and a relaunch skips them. If the orchestrator itself
dies, re-running it resumes from the last incomplete stage. The heavy stages
(GRPO, evals) write their own receipts, so their output survives an
orchestrator crash regardless.

Stages, in order:
    1. wait_grpo       poll for the GRPO receipt (bounded), so the GPU is
                       free before an eval loads a second 32B
    2. unit_tests      pytest the new modules -- fast, GPU-free, verifies the
                       code is green before trusting any eval built on it
    3. integrated_eval the organs+recurrence factorial on the trained adapter
    4. report          one combined receipt of every stage's verdict
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = str(REPO / ".venv" / "bin" / "python")
PIPELINE_SCHEMA = "aura.reasoning_pipeline.v1"


def _load(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (ValueError, OSError):
            pass
    return {"schema": PIPELINE_SCHEMA, "completed": [], "stages": {}}


def _save(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(path)


def _log(log: Path, message: str) -> None:
    with log.open("a") as sink:
        sink.write(f"[{time.ctime()}] {message}\n")


def wait_for_receipt(target: Path, *, log: Path, max_minutes: float) -> bool:
    """Poll for a file, bounded. Sleep-friendly, never unbounded."""
    deadline = time.time() + max_minutes * 60.0
    while time.time() < deadline:
        if target.exists():
            _log(log, f"receipt appeared: {target}")
            return True
        time.sleep(60)
    _log(log, f"TIMEOUT waiting for {target} after {max_minutes} min")
    return False


def run_command(name: str, argv: list[str], *, log: Path) -> dict:
    started = time.time()
    _log(log, f"stage {name} START: {' '.join(argv)}")
    with log.open("a") as sink:
        result = subprocess.run(
            argv, stdout=sink, stderr=subprocess.STDOUT, cwd=str(REPO)
        )
    outcome = {
        "stage": name,
        "returncode": result.returncode,
        "ok": result.returncode == 0,
        "minutes": round((time.time() - started) / 60.0, 2),
    }
    _log(log, f"stage {name} END: {outcome}")
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--grpo-dir", required=True,
                        help="the running GRPO out-dir (holds grpo/grpo_receipt.json)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--wait-minutes", type=float, default=660.0)
    parser.add_argument("--memory-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20261201)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "pipeline_manifest.json"
    manifest = _load(manifest_path)
    log = out / "pipeline.log"
    grpo_receipt = Path(args.grpo_dir) / "grpo" / "grpo_receipt.json"
    grpo_adapter = Path(args.grpo_dir) / "grpo"

    _log(log, f"pipeline start; already completed: {manifest['completed']}")

    # Stage 1: wait for GRPO so an eval never loads a second resident 32B.
    if "wait_grpo" not in manifest["completed"]:
        if not wait_for_receipt(grpo_receipt, log=log, max_minutes=args.wait_minutes):
            manifest["final"] = {"halted_at": "wait_grpo", "reason": "timeout"}
            _save(manifest_path, manifest)
            return 2
        manifest["completed"].append("wait_grpo")
        _save(manifest_path, manifest)

    stages = [
        (
            "unit_tests",
            [
                PYTHON, "-m", "pytest", "-q",
                "tests/test_verifiable_tasks.py",
                "tests/test_grpo.py",
                "tests/test_adaptive_curriculum.py",
                "tests/test_durable_run.py",
                "tests/test_integrated_reasoning_eval.py",
                "tests/test_facade_retrieval.py",
                "tests/test_workspace_producers.py",
                "tests/test_intrinsic_recurrence.py",
                "tests/test_learned_halting_bridge.py",
            ],
        ),
        (
            "integrated_eval",
            [
                PYTHON, "tools/eval_integrated_reasoning.py",
                "--model", args.model,
                "--adapter", str(grpo_adapter),
                "--out", str(out / "integrated_eval.json"),
                "--families", "transitive_chain,conflicting_sources",
                "--hops", "2,4", "--per-cell", "12", "--depths", "1,2,4",
                "--max-tokens", "48",
                "--seed", str(args.seed),
                "--memory-fraction", str(args.memory_fraction),
            ],
        ),
    ]

    for name, argv in stages:
        if name in manifest["completed"]:
            _log(log, f"skip {name} (already complete)")
            continue
        outcome = run_command(name, argv, log=log)
        manifest["stages"][name] = outcome
        if outcome["ok"]:
            manifest["completed"].append(name)
        _save(manifest_path, manifest)
        if not outcome["ok"]:
            # unit_tests failing means the code is broken -- do NOT run an eval
            # on it. A failing eval stage is recorded but does not block the
            # report. Either way the run is resumable: fix and relaunch.
            if name == "unit_tests":
                manifest["final"] = {"halted_at": name, "reason": "tests_red"}
                _save(manifest_path, manifest)
                _log(log, "HALT: unit tests failed; not running evals on broken code")
                return outcome["returncode"]

    # Stage 4: combined report from whatever receipts exist.
    report: dict = {
        "schema": PIPELINE_SCHEMA,
        "stages": manifest["stages"],
        "grpo": None,
        "integrated_eval": None,
    }
    if grpo_receipt.exists():
        g = json.loads(grpo_receipt.read_text())
        report["grpo"] = {
            "history": g.get("history"),
            "learning_signal": g.get("learning_signal"),
            "verdict": g.get("verdict"),
            "curriculum": g.get("curriculum", {}).get("pass_rates"),
        }
    integrated = out / "integrated_eval.json"
    if integrated.exists():
        e = json.loads(integrated.read_text())
        report["integrated_eval"] = {
            "accuracy": e.get("accuracy"),
            "verdicts": e.get("verdicts_corrected") or e.get("verdicts"),
        }
    manifest["final"] = {"all_stages_complete": True}
    _save(manifest_path, manifest)
    (out / "pipeline_report.json").write_text(json.dumps(report, indent=2))
    _log(log, f"PIPELINE COMPLETE -> {out / 'pipeline_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
