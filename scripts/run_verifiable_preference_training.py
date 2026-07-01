#!/usr/bin/env python3
"""Check or run local DPO/ORPO training from Aura's verifier-derived pairs."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.preference_trainer import (  # noqa: E402
    PreferenceTrainingRequest,
    check_preference_trainer_available,
    run_verifiable_preference_training,
)


def _default_store() -> Path:
    try:
        from core.config import config

        return Path(config.paths.data_dir) / "verifiable_preferences.jsonl"
    except Exception:
        return Path.home() / ".aura" / "data" / "verifiable_preferences.jsonl"


def _default_model() -> Path:
    try:
        from core.brain.llm.model_registry import get_model_path

        return Path(get_model_path())
    except Exception:
        return Path("models/Aura-32B")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="only report trainer package availability")
    parser.add_argument("--dry-run", action="store_true", help="export splits and print command without training")
    parser.add_argument("--model", type=Path, default=_default_model())
    parser.add_argument("--store", type=Path, default=_default_store())
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["dpo", "orpo", "online_dpo"], default="dpo")
    parser.add_argument("--train-type", choices=["lora", "dora", "full"], default="lora")
    parser.add_argument("--min-rows", type=int, default=8)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--no-grad-checkpoint", action="store_true")
    parser.add_argument("--no-efficient-long-context", action="store_true")
    args = parser.parse_args(argv)

    if args.check:
        print(json.dumps(check_preference_trainer_available(), indent=2, sort_keys=True))
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_root = Path("training") / "verifiable-preferences" / stamp
    adapter_path = args.adapter_path or (run_root / "adapter")
    data_dir = args.data_dir or (run_root / "data")
    request = PreferenceTrainingRequest(
        model_path=args.model,
        store_path=args.store,
        adapter_path=adapter_path,
        data_dir=data_dir,
        train_mode=args.mode,
        train_type=args.train_type,
        min_rows=args.min_rows,
        limit=args.limit,
        iters=args.iters,
        batch_size=args.batch_size,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        timeout_seconds=args.timeout_seconds,
        grad_checkpoint=not args.no_grad_checkpoint,
        efficient_long_context=not args.no_efficient_long_context,
    )
    result = run_verifiable_preference_training(request, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
