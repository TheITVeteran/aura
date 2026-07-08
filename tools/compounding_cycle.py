#!/usr/bin/env python3
"""tools/compounding_cycle.py — run governed weight-compounding cycles.

Operator entrypoint for core/learning/weight_compounding.py. Each cycle trains
a LoRA on the CURRENT published model, gates it on a sealed held-out battery
(incumbent vs candidate, one model in memory at a time), promotes or refuses
honestly, and appends a generation record to the tamper-evident lineage
ledger. `--cycles N` chains cycles so generation N trains on generation N-1's
published artifact — that chain, verified by `--verdict`, is the compounding
evidence.

Examples:
  # status + verdict from the ledger (no training)
  python tools/compounding_cycle.py --status

  # one dry-run cycle: admission + harvest checks only
  python tools/compounding_cycle.py --dry-run

  # two real proof cycles on the small model into an isolated workspace
  python tools/compounding_cycle.py --cycles 2 \
      --model models/Qwen2.5-1.5B-Instruct-4bit \
      --work-root data/learning/compounding-proof \
      --fused-root data/learning/compounding-proof/fused \
      --operator

Anything consequential this tool does is recorded: cycle receipts under
<work-root>/runs/<generation>/, the hash-chained ledger at
<work-root>/lineage.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_loop(args: argparse.Namespace):
    from core.learning.weight_compounding import (
        CompoundingConfig,
        WeightCompoundingLoop,
        default_config,
    )

    overrides: dict = {"operator_run": bool(args.operator)}
    if args.model:
        overrides["model_override"] = str(Path(args.model).expanduser().resolve())
    if args.work_root:
        overrides["work_root"] = Path(args.work_root).expanduser().resolve()
    if args.fused_root:
        overrides["fused_root"] = Path(args.fused_root).expanduser().resolve()
    if args.sft_buffer:
        overrides["sft_buffer_path"] = Path(args.sft_buffer).expanduser().resolve()
    if args.dpo_store:
        overrides["dpo_store_path"] = Path(args.dpo_store).expanduser().resolve()
    if args.iters:
        overrides["iters"] = int(args.iters)
    if args.battery_size:
        overrides["battery_size"] = int(args.battery_size)
    if args.min_sft is not None:
        overrides["min_sft_examples"] = int(args.min_sft)
    if args.min_dpo is not None:
        overrides["min_dpo_pairs"] = int(args.min_dpo)
    if args.no_publish:
        overrides["publish"] = False

    try:
        config = default_config(**overrides)
    except ImportError:
        # standalone mode without the full runtime config
        if "work_root" not in overrides:
            raise SystemExit("--work-root is required when the runtime config is unavailable")
        overrides.setdefault("fused_root", overrides["work_root"] / "fused")
        config = CompoundingConfig(**overrides)

    return WeightCompoundingLoop(config)


def cmd_status(loop) -> int:
    stats = loop.stats()
    print(json.dumps(stats, indent=2, sort_keys=True))
    intact, problems = loop.verify_ledger()
    if not intact:
        print(f"LEDGER INTEGRITY FAILURE: {problems}", file=sys.stderr)
        return 1
    return 0


def cmd_dry_run(loop) -> int:
    from core.learning.heldout_battery import BatterySpec, generate_battery

    base, source = loop.resolve_base()
    print(f"base_model: {base} (source={source})")
    ok, reasons = loop.admission_check(base)
    print(f"admission: {'OK' if ok else 'BLOCKED'} {reasons or ''}")
    tasks = generate_battery(BatterySpec(seed=loop.config.battery_seed_base, size=8))
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            mode, _, counts = loop.harvest(Path(tmp), tasks)
        print(f"harvest: mode={mode} counts={counts}")
    except RuntimeError as exc:
        print(f"harvest: BLOCKED {exc}")
        return 1
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cycles", type=int, default=1, help="number of chained cycles")
    parser.add_argument("--model", default="", help="explicit base model (else active manifest)")
    parser.add_argument("--work-root", default="", help="workspace for runs + ledger")
    parser.add_argument("--fused-root", default="", help="where fused artifacts + active.json go")
    parser.add_argument("--sft-buffer", default="", help="override SFT experience buffer path")
    parser.add_argument("--dpo-store", default="", help="override DPO preference store path")
    parser.add_argument("--iters", type=int, default=0, help="training iterations per cycle")
    parser.add_argument("--battery-size", type=int, default=0, help="held-out battery size")
    parser.add_argument("--min-sft", type=int, default=None)
    parser.add_argument("--min-dpo", type=int, default=None)
    parser.add_argument("--operator", action="store_true",
                        help="operator run: allows models beyond the autonomous size cap")
    parser.add_argument("--no-publish", action="store_true",
                        help="train + gate + record but do not fuse/publish")
    parser.add_argument("--dry-run", action="store_true",
                        help="admission + harvest checks only, no training")
    parser.add_argument("--status", action="store_true",
                        help="print ledger stats + compounding verdict and exit")
    args = parser.parse_args()

    loop = build_loop(args)

    if args.status:
        return cmd_status(loop)
    if args.dry_run:
        return cmd_dry_run(loop)

    exit_code = 0
    for i in range(max(1, args.cycles)):
        print(f"=== cycle {i + 1}/{args.cycles} ===")
        receipt = loop.run_cycle()
        print(json.dumps(receipt.to_dict(), indent=2, sort_keys=True))
        if receipt.status in ("failed", "blocked"):
            exit_code = 1
            break

    verdict = loop.lineage_verdict()
    print("\n=== lineage verdict ===")
    print(json.dumps(verdict.to_dict(), indent=2, sort_keys=True))
    intact, problems = loop.verify_ledger()
    print(f"ledger_intact: {intact}{'' if intact else ' PROBLEMS: ' + str(problems)}")
    return exit_code if intact else 1


if __name__ == "__main__":
    sys.exit(main())
