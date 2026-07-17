#!/usr/bin/env python
"""Run the durable-learning consolidation train on real queue candidates.

scan queue → validate → build proposals → distill each into a durable
adapter → interference battery (natural probes) → optional held-out check →
activation trial → PROVEN rollback → receipts. The model is returned to its
exact pre-run state (this tool proves it); durable ACTIVATION on the live
instance goes through the service/adapter seam, not this operator tool.

MEMORY SAFETY: only point this at the 32B when the live instance is DOWN.

  caffeinate -dims .venv/bin/python tools/latent_consolidation_train.py \
      --model <mlx-model-dir> [--queue <dir>] [--out <report.json>] \
      [--max-minutes 20]
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="mlx model directory")
    parser.add_argument("--queue", default="", help="consolidation queue dir")
    parser.add_argument("--adapter-dir", default="", help="durable adapter output dir")
    parser.add_argument("--out", default="", help="report path")
    parser.add_argument("--max-minutes", type=float, default=20.0)
    args = parser.parse_args()

    from core.runtime.model_lane_control import standalone_model_lane

    with standalone_model_lane(
        owner_id=f"latent-consolidation-train:{os.getpid()}",
        model_path=args.model,
        purpose="benchmark",
        preemptible=False,
        metadata={"tool": "latent_consolidation_train", "operator_launched": True},
    ):
        return _run(args)


def _run(args: argparse.Namespace) -> int:
    from core.config import DATA_DIR
    from core.learning.latent_adapter_distillation import (
        rollback_adapter,
        run_consolidation_train,
    )
    from core.learning.latent_consolidation import build_proposals, scan_queue

    deadline = time.monotonic() + args.max_minutes * 60.0
    queue_dir = Path(args.queue) if args.queue else (
        Path(DATA_DIR) / "latent_cortex" / "consolidation_queue"
    )
    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else (
        Path(DATA_DIR) / "latent_cortex" / "durable_adapters"
    )

    records = scan_queue(queue_dir)
    proposals = build_proposals(records)
    print(
        f"queue: {len(records)} candidates "
        f"({sum(1 for r in records if r.valid)} valid) → {len(proposals)} proposal(s)",
        flush=True,
    )
    report: dict = {
        "schema": "aura.latent_consolidation_train_report.v1",
        "model": args.model,
        "queue": str(queue_dir),
        "started_at": time.time(),
        "candidates": [record.to_dict() for record in records],
        "trains": [],
    }
    if proposals:
        from mlx_lm import load

        model, tokenizer = load(args.model)
        for proposal in proposals:
            if time.monotonic() > deadline:
                report["deadline_exceeded"] = True
                break
            print(f"▶ train: domain={proposal['domain']} "
                  f"candidates={proposal['candidate_count']} …", flush=True)
            receipt = run_consolidation_train(
                proposal,
                model,
                adapter_dir=adapter_dir,
                tokenizer=tokenizer,
            )
            active = receipt.pop("active_adapter", None)
            if active is not None and active.active:
                # Operator-tool contract: prove activation AND prove rollback;
                # durable live activation belongs to the governed service seam.
                receipt["rollback"] = rollback_adapter(model, active)
            report["trains"].append(receipt)
            print(
                f"  activated={receipt['activated']} "
                f"battery={receipt.get('interference_battery', {}).get('verdict')} "
                f"rollback_proven={receipt.get('rollback', {}).get('rollback_proven')}",
                flush=True,
            )
    report["finished_at"] = time.time()
    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "artifacts" / "current" / f"latent_consolidation_train_{int(time.time())}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    print(f"📄 report → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
