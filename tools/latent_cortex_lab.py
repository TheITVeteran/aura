#!/usr/bin/env python
"""Latent Cortex Lab — run the falsification experiments on a real checkpoint.

Operator-launched, bounded, and honest by construction: every run prints the
checkpoint fingerprint, the graded claims, and writes the full JSON report.

Usage (run with the repo venv python; bound long runs with caffeinate):

  caffeinate -dims .venv/bin/python tools/latent_cortex_lab.py \\
      --model ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-4bit/snapshots/<hash> \\
      --experiments 1,2,3 --per-cell 8 --max-minutes 30

MEMORY SAFETY: never point this at a second 32B while the live instance is
up. The 1.5B/7B checkpoints are the offline lab vehicles; the resident 32B
runs episodes through the worker action instead.
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
    parser.add_argument("--experiments", default="1,2", help="comma list of 1..5")
    parser.add_argument("--per-cell", type=int, default=8)
    parser.add_argument("--depths", default="2,4,8")
    parser.add_argument("--steps", default="1,2,4,8")
    parser.add_argument("--families", default="khop,boolean,modular")
    parser.add_argument("--n-slots", type=int, default=16)
    parser.add_argument("--branches", type=int, default=2)
    parser.add_argument("--max-minutes", type=float, default=30.0)
    parser.add_argument("--out", default="")
    parser.add_argument("--record-foundry", action="store_true")
    args = parser.parse_args()

    from mlx_lm import load

    from core.brain.llm.latent_cortex.engine import LatentCortexEngine
    from core.brain.llm.latent_cortex.experiments import (
        record_claim_to_foundry,
        run_depth_extrapolation,
        run_latent_opt_control,
        run_recurrence_sweep,
        run_slot_causality,
        task_battery,
    )
    from core.brain.llm.latent_cortex.types import (
        BranchConfig,
        ComputeBudget,
        CortexConfig,
        LatentOptConfig,
        RecurrenceConfig,
        WorkspaceConfig,
    )

    deadline = time.monotonic() + args.max_minutes * 60.0
    model, tokenizer = load(args.model)
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    depths = [int(d) for d in args.depths.split(",")]
    steps = [int(s) for s in args.steps.split(",")]
    wanted = {e.strip() for e in args.experiments.split(",")}

    def make_engine(max_steps: int, *, latent_opt: str = "off", branches: int | None = None):
        return LatentCortexEngine(
            model,
            tokenizer,
            CortexConfig(
                workspace=WorkspaceConfig(n_slots=args.n_slots, seed=7),
                recurrence=RecurrenceConfig(
                    max_steps=max_steps, min_steps=max_steps, convergence_eps=0.0
                ),
                branches=BranchConfig(n_branches=branches or args.branches),
                latent_opt=LatentOptConfig(
                    enabled=latent_opt != "off",
                    control_mode=latent_opt == "control",
                    steps=4,
                ),
                decode_max_tokens=64,
            ),
            model_path=args.model,
        )

    def out_of_time() -> bool:
        if time.monotonic() > deadline:
            print("⏰ wall-clock bound reached — reporting what completed", flush=True)
            return True
        return False

    def solve(task, n_steps: int, *, latent_opt: str = "off", ablate=None) -> bool:
        engine = make_engine(n_steps, latent_opt=latent_opt)
        result = engine.reason(
            prompt=task.prompt,
            budget=ComputeBudget(wall_clock_s=120.0),
            ablate_slot=ablate,
            decode_max_tokens=64,
        )
        return result.ok and task.verify(result.text)

    report: dict = {
        "model": args.model,
        "started_at": time.time(),
        "settings": vars(args),
        "results": {},
    }

    battery = task_battery(families, depths, args.per_cell, seed=11)
    if "1" in wanted and not out_of_time():
        print(f"▶ Experiment 1: recurrence sweep over {len(battery)} tasks …", flush=True)
        report["results"]["exp1"] = run_recurrence_sweep(
            lambda t, s: solve(t, s), battery, steps
        )
        print("  claim:", report["results"]["exp1"]["claim"]["tier"], flush=True)
    if "2" in wanted and not out_of_time():
        report["results"]["exp2"] = {}
        for family in families:
            if out_of_time():
                break
            print(f"▶ Experiment 2: depth extrapolation on {family} …", flush=True)
            report["results"]["exp2"][family] = run_depth_extrapolation(
                lambda t, s: solve(t, s), family, depths, steps, per_depth=args.per_cell
            )
            print("  claim:", report["results"]["exp2"][family]["claim"]["tier"], flush=True)
    if "3" in wanted and not out_of_time():
        print("▶ Experiment 3: slot causality …", flush=True)
        report["results"]["exp3"] = run_slot_causality(
            lambda t, slot: solve(t, max(steps), ablate=slot),
            battery,
            slot_indices=list(range(0, args.n_slots, max(1, args.n_slots // 4))),
        )
        print("  claim:", report["results"]["exp3"]["claim"]["tier"], flush=True)
    if "5" in wanted and not out_of_time():
        print("▶ Experiment 5: latent opt vs random control …", flush=True)
        by_family = {f: [t for t in battery if t.family == f] for f in families}
        report["results"]["exp5"] = run_latent_opt_control(
            lambda t, arm: solve(t, max(steps), latent_opt=arm), by_family
        )
        print("  claim:", report["results"]["exp5"]["claim"]["tier"], flush=True)

    report["finished_at"] = time.time()
    out_path = Path(args.out) if args.out else REPO_ROOT / "data" / "latent_cortex" / (
        f"lab_report_{int(time.time())}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    print(f"📄 report → {out_path}")

    if args.record_foundry:
        for key, res in report["results"].items():
            claims = [res["claim"]] if "claim" in res else [
                v["claim"] for v in res.values() if isinstance(v, dict) and "claim" in v
            ]
            for claim in claims:
                record_claim_to_foundry(claim, domain="latent_lab")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
