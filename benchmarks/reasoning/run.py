"""CLI entry for the reasoning benchmark.

    python -m benchmarks.reasoning.run            # deterministic canned candidates
    python -m benchmarks.reasoning.run --live     # answers come from the live model
    python -m benchmarks.reasoning.run --out results.json

The deterministic run exercises the truth engines + amplifier against seeded errors
(no model needed) and is suitable for CI regression gating. ``--live`` routes the
same objectives through the real inference path to measure on-device behaviour.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from .harness import ReasoningBenchmark, write_results


async def _live_generator():
    """Build a generate(prompt, temperature) backed by the live LLM router."""
    from core.runtime import service_access

    router = service_access.resolve_llm_router(default=None)
    if router is None:
        raise RuntimeError("live LLM router unavailable; run without --live")

    async def gen(prompt: str, temperature: float) -> str:
        res = await router.think(prompt, mode="FAST", temperature=temperature)
        return res.content if hasattr(res, "content") else str(res or "")

    return gen


async def _main_async(args: argparse.Namespace) -> int:
    generate = await _live_generator() if args.live else None
    result = await ReasoningBenchmark().run(generate=generate)
    print(result.summary())
    for o in result.outcomes:
        flag = "✅" if o.correct else "❌"
        fc = " ⚠️false-confidence" if o.false_confidence else ""
        print(f"  {flag} {o.case_id:<12} verified={o.verified!s:<5} conf={o.confidence:.2f}{fc}")
    if args.out:
        write_results(result, args.out)
        print(f"wrote {args.out}")
    # CI gate: every seeded error must be caught and nothing wrong asserted confidently.
    ok = result.verifier_catch_rate >= 1.0 and result.false_confidence_rate <= 0.0
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Aura reasoning benchmark")
    parser.add_argument("--live", action="store_true", help="use the live model instead of canned candidates")
    parser.add_argument("--out", default="", help="write results JSON to this path")
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
