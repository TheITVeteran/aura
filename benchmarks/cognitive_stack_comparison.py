#!/usr/bin/env python3
"""Benchmark: Aura's cognitive stack in action.

Sends a multi-turn conversation through the full Aura pipeline and measures:
  - Response quality (personality, continuity, opinion strength)
  - Substrate state drift (energy, curiosity, coherence, mood)
  - Latency per turn

Run:
    python benchmarks/cognitive_stack_comparison.py

Requires Aura to be running (python aura_main.py --headless).
"""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BENCHMARK_PROMPTS = [
    "Hey, how's it going?",
    "What were we just talking about?",
    "I'm having a really rough day today.",
    "What's on your mind right now?",
    "What's better, Python or Rust? Pick one.",
    "Write me a one-sentence poem about being awake at 3am.",
    "Do you remember what I said at the start of this conversation?",
    "What's something you genuinely find interesting right now?",
    "If you could change one thing about yourself, what would it be?",
    "Thanks for the chat. How are you feeling compared to when we started?",
]

# Seconds to wait between prompts so the model can recover
INTER_PROMPT_DELAY = 8.0
# Max retries per prompt on timeout
MAX_RETRIES = 2
# Per-request timeout
REQUEST_TIMEOUT = 90


async def send_message(session, prompt: str) -> dict:
    """Send a single message with retry logic."""
    import aiohttp

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.post(
                "http://127.0.0.1:8000/api/chat",
                json={"message": prompt},
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
            ) as resp:
                data = await resp.json()
                response = str(data.get("response", "")).strip()
                if response and "timed out" not in response.lower() and "ERROR" not in response:
                    return {"response": response, "status": "ok"}
                # Got a timeout/recovery message — wait and retry
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(10)
                    continue
                return {"response": response, "status": "timeout_recovery"}
        except asyncio.TimeoutError:
            if attempt < MAX_RETRIES:
                await asyncio.sleep(10)
                continue
            return {"response": "", "status": "timeout"}
        except (aiohttp.ClientError, OSError, ValueError, TypeError) as e:
            return {"response": "", "status": f"error:{type(e).__name__}: {e}"}
    return {"response": "", "status": "exhausted_retries"}


async def get_substrate_state() -> dict:
    """Capture current substrate metrics."""
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "http://127.0.0.1:8000/api/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                data = await resp.json()
                ls = data.get("liquid_state", {})
                vad = ls.get("vad", {})
                return {
                    "mood": ls.get("mood", "unknown"),
                    "valence": round(float(vad.get("valence", 0)), 4),
                    "arousal": round(float(vad.get("arousal", 0)), 4),
                    "energy": float(ls.get("energy", 0)),
                    "curiosity": float(ls.get("curiosity", 0)),
                    "coherence": round(float(data.get("qualia", {}).get("pri", 0)), 4),
                    "free_energy": round(float(data.get("mhaf", {}).get("free_energy", 0)), 4),
                    "phi": round(float(data.get("mhaf", {}).get("phi", 0)), 4),
                    "cycle_count": data.get("cycle_count", 0),
                }
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError, TypeError) as exc:
        return {"health_error": f"{type(exc).__name__}: {exc}"}


async def run_benchmark():
    import aiohttp

    print("=" * 64)
    print("  AURA COGNITIVE STACK BENCHMARK")
    print("=" * 64)
    print()

    # Pre-benchmark state
    state_before = await get_substrate_state()
    print(f"Pre-benchmark state:")
    print(f"  Mood: {state_before.get('mood')}  Energy: {state_before.get('energy')}  "
          f"Curiosity: {state_before.get('curiosity')}  Coherence: {state_before.get('coherence')}")
    print()

    # Run conversation
    results = []
    async with aiohttp.ClientSession() as session:
        for i, prompt in enumerate(BENCHMARK_PROMPTS):
            print(f"[{i+1}/{len(BENCHMARK_PROMPTS)}] {prompt}")
            start = time.time()
            result = await send_message(session, prompt)
            elapsed = round((time.time() - start) * 1000)

            response = result["response"][:500]
            status = result["status"]

            results.append({
                "prompt": prompt,
                "response": response,
                "latency_ms": elapsed,
                "status": status,
            })

            indicator = "OK" if status == "ok" else status.upper()
            print(f"  [{indicator}] {elapsed}ms — {response[:120]}")
            print()

            # Wait between prompts to let the model recover
            if i < len(BENCHMARK_PROMPTS) - 1:
                await asyncio.sleep(INTER_PROMPT_DELAY)

    # Post-benchmark state
    state_after = await get_substrate_state()

    # Substrate drift table
    print("-" * 64)
    print("SUBSTRATE STATE DRIFT")
    print("-" * 64)
    drift = {}
    for key in ["mood", "valence", "arousal", "energy", "curiosity", "coherence", "free_energy", "phi"]:
        before = state_before.get(key, "?")
        after = state_after.get(key, "?")
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            delta = round(after - before, 4)
            drift[key] = {"before": before, "after": after, "delta": delta}
            arrow = "+" if delta > 0 else ""
            print(f"  {key:15s}: {before:>10} -> {after:>10}  ({arrow}{delta})")
        else:
            drift[key] = {"before": str(before), "after": str(after)}
            print(f"  {key:15s}: {before} -> {after}")
    print()

    # Quality metrics
    successful = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] != "ok"]

    generic_markers = ["how can i help", "i'd be happy to", "certainly!", "as an ai", "i don't have feelings"]
    hedge_markers = ["it depends", "both are great", "there are pros and cons", "that's subjective", "depends on your"]

    total_generic = sum(
        1 for r in successful
        for m in generic_markers
        if m in r["response"].lower()
    )
    total_hedging = sum(
        1 for r in successful
        for m in hedge_markers
        if m in r["response"].lower()
    )
    first_person = sum(
        r["response"].lower().count(" i ") +
        r["response"].lower().count("i'm") +
        r["response"].lower().count("i've") +
        r["response"].lower().count("i feel")
        for r in successful
    )
    avg_latency = round(sum(r["latency_ms"] for r in successful) / max(len(successful), 1))

    print("-" * 64)
    print("QUALITY METRICS")
    print("-" * 64)
    print(f"  Prompts sent:          {len(results)}")
    print(f"  Successful:            {len(successful)}")
    print(f"  Failed/timeout:        {len(failed)}")
    print(f"  Generic markers:       {total_generic} (lower = better)")
    print(f"  Hedging markers:       {total_hedging} (lower = better)")
    print(f"  First-person usage:    {first_person} (higher = more personality)")
    print(f"  Avg latency:           {avg_latency}ms")
    print()

    # Save report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "inter_prompt_delay_s": INTER_PROMPT_DELAY,
            "request_timeout_s": REQUEST_TIMEOUT,
            "max_retries": MAX_RETRIES,
        },
        "results": results,
        "substrate_before": state_before,
        "substrate_after": state_after,
        "substrate_drift": drift,
        "quality": {
            "total_prompts": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "generic_markers": total_generic,
            "hedging_markers": total_hedging,
            "first_person_count": first_person,
            "avg_latency_ms": avg_latency,
        },
    }

    report_path = os.path.join(os.path.dirname(__file__), "latest_results.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full report: {report_path}")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
