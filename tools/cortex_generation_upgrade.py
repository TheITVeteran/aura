#!/usr/bin/env python
"""Cortex generation upgrade — operator CLI for the governed base-model swap.

The pipeline evaluates a candidate generation (breadth + reasoning +
identity batteries), plans identity migration, stages the activation
pointer with a byte-exact rollback, and activates ONLY with an explicit
operator authorization plus a PASS verdict. Effect is at next boot; the
running mind is never hot-swapped.

    .venv/bin/python tools/cortex_generation_upgrade.py evaluate \
        --candidate ~/models/Qwen3-32B-4bit --out artifacts/current/cortex_upgrade
    .venv/bin/python tools/cortex_generation_upgrade.py plan
    .venv/bin/python tools/cortex_generation_upgrade.py stage \
        --candidate ~/models/Qwen3-32B-4bit --base Qwen3-32B --tag qwen3-gen
    .venv/bin/python tools/cortex_generation_upgrade.py activate \
        --authorized-by "bryan" --evaluation artifacts/current/cortex_upgrade/comparison.json
    .venv/bin/python tools/cortex_generation_upgrade.py rollback

MEMORY SAFETY: `evaluate` loads models and is guarded — it refuses when the
host cannot afford the candidate beside resident processes (live app or a
training run). Never force it past a refusal.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("AURA_LOG_DIR", str(Path.home() / ".aura" / "lab-logs"))


def _write(out_dir: Path, name: str, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    print(f"📄 {path}")
    return path


def _load_model(path: str):
    from mlx_lm import load
    from core.runtime.model_lane_control import standalone_model_lane

    with standalone_model_lane(
        owner_id=f"cortex-generation-upgrade:{Path(path).name}",
        model_path=path,
        purpose="evaluation",
        preemptible=False,
        metadata={"tool": "cortex_generation_upgrade", "operator_launched": True},
    ):
        return load(path)


def cmd_evaluate(args) -> int:
    from core.learning.cortex_generation_upgrade import (
        MemoryGuard,
        capability_battery,
        compare_batteries,
    )

    out_dir = Path(args.out)
    guard = MemoryGuard()
    admission = guard.admit(args.candidate)
    _write(out_dir, "admission.json", admission)
    if not admission["admitted"]:
        print(f"🚫 refused: {admission.get('refusal_reason')}")
        return 2

    receipts = {}
    for label, model_path in (("current", args.current), ("candidate", args.candidate)):
        if not model_path:
            continue
        print(f"▶ loading {label}: {model_path}", flush=True)
        model, tokenizer = _load_model(model_path)
        receipts[label] = capability_battery(model, tokenizer, label=label)
        _write(out_dir, f"battery_{label}.json", receipts[label])
        del model, tokenizer
        try:
            import mlx.core as mx

            mx.clear_cache()
        except (ImportError, AttributeError):
            pass
    if "current" in receipts and "candidate" in receipts:
        comparison = compare_batteries(receipts["current"], receipts["candidate"])
        _write(out_dir, "comparison.json", comparison)
        print(f"VERDICT: {comparison['verdict']} "
              f"(breadth {comparison['breadth_delta']:+.3f}, "
              f"reasoning {comparison['reasoning_delta']:+.3f})")
        return 0 if comparison["verdict"] == "PASS" else 1
    return 0


def cmd_plan(args) -> int:
    from core.learning.cortex_generation_upgrade import build_migration_plan

    plan = build_migration_plan()
    _write(Path(args.out), "migration_plan.json", plan)
    for step in plan["steps"]:
        marker = "•" if step["lane"] == "automatic" else "◦"
        print(f" {marker} {step['name']} [{step['lane']}] exists={step['exists']}")
    return 0


def cmd_stage(args) -> int:
    from core.learning.cortex_generation_upgrade import stage_upgrade

    evaluation = None
    if args.evaluation:
        evaluation = json.loads(Path(args.evaluation).read_text())
    receipt = stage_upgrade(
        candidate_model_path=args.candidate,
        base_model_path=args.base,
        tag=args.tag,
        evaluation=evaluation,
    )
    _write(Path(args.out), "staging.json", receipt)
    return 0


def cmd_activate(args) -> int:
    from core.learning.cortex_generation_upgrade import activate_upgrade

    evaluation = json.loads(Path(args.evaluation).read_text())
    receipt = activate_upgrade(
        authorized_by=args.authorized_by, evaluation=evaluation
    )
    _write(Path(args.out), "activation.json", receipt)
    print("⚠️  effective at NEXT BOOT — restart Aura to think on the new cortex")
    return 0


def cmd_rollback(args) -> int:
    from core.learning.cortex_generation_upgrade import rollback_upgrade

    receipt = rollback_upgrade()
    _write(Path(args.out), "rollback.json", receipt)
    return 0 if receipt["byte_exact"] else 1


def cmd_status(args) -> int:
    from core.brain.llm.model_registry import BASE_DIR
    from core.learning.cortex_generation_upgrade import (
        ROLLBACK_POINTER_NAME,
        STAGED_POINTER_NAME,
    )

    fused = Path(BASE_DIR) / "training" / "fused-model"
    status = {
        "active": json.loads((fused / "active.json").read_text())
        if (fused / "active.json").is_file()
        else None,
        "staged": json.loads((fused / STAGED_POINTER_NAME).read_text())
        if (fused / STAGED_POINTER_NAME).is_file()
        else None,
        "rollback_available": (fused / ROLLBACK_POINTER_NAME).is_file(),
        "checked_at": time.time(),
    }
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--candidate", required=True)
    evaluate.add_argument("--current", default="")
    evaluate.add_argument("--out", default="artifacts/current/cortex_upgrade")
    evaluate.set_defaults(func=cmd_evaluate)

    plan = sub.add_parser("plan")
    plan.add_argument("--out", default="artifacts/current/cortex_upgrade")
    plan.set_defaults(func=cmd_plan)

    stage = sub.add_parser("stage")
    stage.add_argument("--candidate", required=True)
    stage.add_argument("--base", required=True)
    stage.add_argument("--tag", required=True)
    stage.add_argument("--evaluation", default="")
    stage.add_argument("--out", default="artifacts/current/cortex_upgrade")
    stage.set_defaults(func=cmd_stage)

    activate = sub.add_parser("activate")
    activate.add_argument("--authorized-by", required=True)
    activate.add_argument("--evaluation", required=True)
    activate.add_argument("--out", default="artifacts/current/cortex_upgrade")
    activate.set_defaults(func=cmd_activate)

    rollback = sub.add_parser("rollback")
    rollback.add_argument("--out", default="artifacts/current/cortex_upgrade")
    rollback.set_defaults(func=cmd_rollback)

    status = sub.add_parser("status")
    status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
