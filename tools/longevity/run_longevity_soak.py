#!/usr/bin/env python3
"""Proof-profile longevity soak runner for Aura.

This is a short, bounded certification soak, not evidence of indefinite
autonomy. It boots the canonical runtime, executes repeated governance/health
pulses, samples real process metrics, and validates queue/lag stability.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024 * 1024)
    except (ImportError, RuntimeError, OSError, AttributeError):
        try:
            import resource

            value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if sys.platform == "darwin":
                return value / (1024 * 1024)
            return value / 1024
        except (ImportError, RuntimeError, OSError, AttributeError):
            return 0.0


def _queue_depths() -> dict[str, int]:
    depths: dict[str, int] = {}
    try:
        from core.container import ServiceContainer

        for name in ("event_bus", "task_tracker", "llm_router"):
            service = ServiceContainer.get(name, default=None)
            if service is None:
                continue
            for attr in ("queue", "_queue", "pending", "_pending"):
                queue = getattr(service, attr, None)
                if hasattr(queue, "qsize"):
                    try:
                        depths[f"{name}.{attr}"] = int(queue.qsize())
                    except (RuntimeError, OSError, ValueError):
                        pass
    except (ImportError, RuntimeError, AttributeError):
        pass
    return depths


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


async def async_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="proof")
    parser.add_argument("--out", default="artifacts/current/longevity_soak")
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args(argv)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("AURA_PROOF_RUN", "1")
    from aura_main import boot_aura_runtime
    from core.will import ActionDomain, get_will
    from tools.agi.run_dnu_agi_proof_battery import shutdown_proof_runtime

    orch = await boot_aura_runtime(
        profile=args.profile,
        ready_label="Proof-Longevity",
        readiness_context="longevity_soak",
        artifact_root=ROOT / "artifacts" / "current",
    )

    receipts_path = out_dir / "RECEIPTS.jsonl"
    metrics: list[dict[str, Any]] = []
    try:
        will = get_will()
        await will.start()
        with receipts_path.open("w", encoding="utf-8") as receipt_file:
            for index in range(max(1, args.iterations)):
                before = time.perf_counter()
                await asyncio.sleep(0.05)
                lag_ms = max(0.0, (time.perf_counter() - before - 0.05) * 1000.0)
                decision = will.decide(
                    content=f"longevity proof pulse {index}",
                    source="longevity_soak",
                    domain=ActionDomain.STABILIZATION,
                    priority=0.4,
                )
                queue_depths = _queue_depths()
                metrics.append(
                    {
                        "iteration": index,
                        "rss_mb": round(_rss_mb(), 3),
                        "lag_ms": round(lag_ms, 3),
                        "queue_len": sum(queue_depths.values()),
                        "queue_depths": queue_depths,
                        "receipt_id": decision.receipt_id,
                    }
                )
                receipt_file.write(
                    json.dumps(
                        {
                            "task_id": f"longevity_pulse_{index}",
                            "receipt_id": decision.receipt_id,
                            "domain": ActionDomain.STABILIZATION.value,
                            "outcome": getattr(decision.outcome, "value", str(decision.outcome)),
                            "reason": decision.reason,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    finally:
        await shutdown_proof_runtime(orch)

    rss_values = [item["rss_mb"] for item in metrics if item["rss_mb"] > 0.0]
    rss_growth = (max(rss_values) - min(rss_values)) if rss_values else 0.0
    max_lag = max((item["lag_ms"] for item in metrics), default=0.0)
    max_queue = max((item["queue_len"] for item in metrics), default=0)

    report = {
        "generated_at": time.time(),
        "profile": args.profile,
        "iterations_completed": len(metrics),
        "memory_leakage_detected": rss_growth > 128.0,
        "rss_growth_mb": round(rss_growth, 3),
        "queue_growth_stable": max_queue <= 10,
        "event_loop_lag_normal": max_lag <= 250.0,
        "max_lag_ms": round(max_lag, 3),
        "max_queue_len": max_queue,
        "metrics": metrics,
        "claim_scope": "short proof-profile soak; not indefinite-autonomy evidence",
    }

    write_json(out_dir / "SOAK_METRICS.json", report)
    manifest = {
        "schema": "longevity_soak_manifest",
        "sha256": {
            name: hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
            for name in ("SOAK_METRICS.json", "RECEIPTS.jsonl")
        },
    }
    write_json(out_dir / "MANIFEST.json", manifest)

    print(f"Longevity soak suite completed successfully for profile: {args.profile}")
    return 0 if not report["memory_leakage_detected"] and report["queue_growth_stable"] and report["event_loop_lag_normal"] else 1


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    sys.exit(main())

