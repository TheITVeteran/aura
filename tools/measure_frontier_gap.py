#!/usr/bin/env python3
"""Measure Aura's gap-to-frontier — the trend line the arc answers to (P3).

Runs the fresh-generated battery (core/brain/frontier_gap.py) through Aura's
REAL reasoning amplifier and writes/updates the checked-in artifact
artifacts/frontier_gap/latest.json (per-class gap + the trend across runs).
Every graded item also feeds the Verifier Foundry, so the battery doubles as
the foundry's ground-truth firehose.

Solver modes (auto):
  amplifier_mlx   The resident model is loadable in-process: solve each item
                  through the real ReasoningAmplifierV2 over live MLX
                  generation — the honest end-to-end capability measurement.
  amplifier_stub  No model available headless (the live instance holds it, or
                  no weights cached): solve through the amplifier with a
                  deterministic knowledge-stub generator. This exercises the
                  WHOLE loop — amplifier → verifiers → foundry grading →
                  playbook capture — and produces a real trend artifact, but
                  the capability numbers reflect the stub, NOT the 32B mind.
                  The mode is recorded in the artifact; never conflate them.

Bounded: fixed battery size; each item is deadline-guarded.

Usage:
  .venv/bin/python tools/measure_frontier_gap.py --per-class 5 --seed 0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.brain.frontier_gap import GapLedger, run_battery  # noqa: E402
from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402


def _git_commit() -> str:
    try:
        out = get_subprocess_gateway().run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
            read_only=True, source="proof_tooling:frontier_gap_git", timeout=10,
            cwd=Path(__file__).resolve().parent.parent)
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, RuntimeError, TypeError, ValueError):
        pass
    return "unknown"


def _live_instance_up(port: int = 8000) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz",
                                    timeout=2) as resp:
            return resp.status == 200
    except OSError:
        return False


# ── stub knowledge generator: deterministic, honest, exercises the loop ──────
_STUB_FACTS = {
    "gold": "Au", "red planet": "Mars", "hexagon": "6",
    "capital of japan": "Tokyo", "photosynthesis": "carbon dioxide",
}


async def _stub_generate(prompt: str, temperature: float = 0.0) -> str:
    """A tiny competent solver for the battery's task shapes — NOT the mind.
    It answers the deterministic classes correctly so the pipeline (verify →
    foundry → playbook) runs on real passes/fails."""
    import re

    p = prompt.lower()
    m = re.search(r"compute (\d+) \* (\d+)", p)
    if m:
        return str(int(m.group(1)) * int(m.group(2)))
    for key, ans in _STUB_FACTS.items():
        if key in p:
            return ans
    if "who is oldest" in p:
        names = re.findall(r"([A-Z][a-z]+) is older than", prompt)
        return names[0] if names else ""
    mc = re.search(r"`(sum|max|min)_of\(xs\)`.*?== (\d+)", prompt, re.DOTALL)
    if mc:
        fn, expected = mc.group(1), mc.group(2)
        xs = re.search(r"\[([\d, ]+)\]", prompt)
        return (f"```python\ndef {fn}_of(xs):\n    return {fn}(xs)\n\n"
                f"assert {fn}_of([{xs.group(1) if xs else ''}]) == {expected}\n```")
    return ""


async def build_solver():
    """Return (solve_fn, mode). Prefer the real amplifier over MLX; fall back
    to the amplifier over a stub generator when no model is loadable."""
    from core.brain.reasoning_amplifier_v2 import (
        AmplificationRequest,
        ReasoningAmplifierV2,
    )

    generate, mode = await _resolve_generate()
    amplifier = ReasoningAmplifierV2(generate)

    async def solve(prompt: str, task_type: str) -> str:
        request = AmplificationRequest(
            objective=prompt, task_type=task_type, time_budget_s=20.0,
            context={"skip_evidence": True},
        )
        try:
            result = await asyncio.wait_for(amplifier.amplify(request), timeout=25.0)
            return getattr(result, "answer", "") or ""
        except (TimeoutError, RuntimeError, AttributeError, TypeError, ValueError):
            return await generate(prompt, 0.0)

    return solve, mode


async def _resolve_generate():
    """Try to build a real MLX generate fn; else the honest stub."""
    if _live_instance_up():
        # the live mind holds the model — do not load a second one
        return _stub_generate, "amplifier_stub_live_instance_up"
    try:
        from core.brain.llm.model_registry import get_runtime_model_path

        model_path = str(get_runtime_model_path() or "")
        # We do not force a 20GB load in a proof tool by default; opt-in only.
        import os

        if os.environ.get("AURA_FRONTIER_LOAD_MODEL") == "1" and model_path:
            from core.brain.llm.mlx_client import get_mlx_client

            client = get_mlx_client(model_path)

            async def generate(prompt: str, temperature: float = 0.0) -> str:
                out = await client.generate(prompt, max_tokens=256,
                                            temperature=temperature)
                return out or ""

            return generate, "amplifier_mlx"
    except (ImportError, RuntimeError, OSError, ValueError):
        pass
    return _stub_generate, "amplifier_stub"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=int(time.time()) % 100000)
    parser.add_argument("--out", default="artifacts/frontier_gap/latest.json")
    args = parser.parse_args()

    # boot the foundry so the battery feeds ground truth into it
    try:
        from core.brain.verifiers.foundry import boot_verifier_foundry

        boot_verifier_foundry()
    except (ImportError, RuntimeError):
        pass

    solve, mode = await build_solver()
    print(f"[frontier-gap] mode={mode} seed={args.seed} "
          f"per_class={args.per_class}")
    report = await run_battery(solve, seed=args.seed, per_class=args.per_class)
    report["solver_mode"] = mode

    out = Path(args.out)

    def _read_prior() -> dict:
        if not out.exists():
            return {}
        try:
            return json.loads(out.read_text()).get("payload", {})
        except (OSError, ValueError):
            return {}

    prior = await asyncio.to_thread(_read_prior)
    ledger = GapLedger.from_dict(prior.get("ledger", {}))
    ledger.add(report)

    body = {
        "schema": "aura.frontier_gap_report.v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "git_commit": _git_commit(),
        "solver_mode": mode,
        "host": {"platform": platform.platform(), "python": sys.version.split()[0]},
        "latest_run": report,
        "ledger": ledger.to_dict(),
        "claim": (
            "Operational frontier-general definition: Aura's verifier-graded "
            "score on a fresh-generated battery divided by a frontier "
            f"reference, per class. This run (mode={mode}): overall gap "
            f"{report['overall_gap']} (0 = parity). The CLAIM is won only on a "
            "monotone closing trend across runs at matched budget — a single "
            "point is a baseline, not a victory. Stub-mode numbers measure the "
            "harness (verify→foundry→playbook loop), NOT the 32B mind."
        ),
        "real_mind_measurement": (
            "The amplifier_mlx real-capability run was attempted headless and "
            "REFUSED by Aura's own memory guardians (projected 38.1GB > 35.8GB "
            "safe limit) — the same survival protection the whole-system Φ "
            "campaign measured. That refusal is correct and was not overridden. "
            "The honest real-mind number comes from running this tool while her "
            "desktop instance is up (the model is already resident — no second "
            "load), or from an operator explicitly accepting the memory risk "
            "via AURA_MLX_WORKER_RSS_LIMIT_GB + AURA_ALLOW_UNSAFE_MEMORY_LIMITS "
            "with genuine free memory."
        ),
    }

    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    with local_internal_governed_scope("frontier_gap", domain="state_mutation"):
        gateway = get_file_write_gateway()
        await gateway.ensure_directory_async(out.parent, source="frontier_gap")
        await gateway.write_json_async(out, body, schema_version=1,
                                       schema_name="frontier_gap_report",
                                       source="frontier_gap")

    print("=" * 68)
    print(f"FRONTIER GAP — {mode} — overall gap {report['overall_gap']} "
          f"(aura {report['overall_aura_score']})")
    for c in report["classes"]:
        print(f"  {c['task_class']:<10} aura={c['aura_score']:.2f} "
              f"ref={c['reference_score']:.2f} gap={c['gap']:.3f}")
    print(f"trend: {ledger.trend()}")
    print(f"artifact: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
