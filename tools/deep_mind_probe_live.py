#!/usr/bin/env python3
"""Live deep-mind probe run — the 7 agency/consciousness probes, honestly graded.

Sends each probe in core/evaluation/deep_mind_probe.py to the RUNNING instance
via /api/chat (one session, sequential turns) and grades every reply with the
real evaluator. The July 3 external review noted the last session was 6/7 with
a 0.857 depth score because one probe lacked agency-boundary and continuity
grounding — this runner produces the current, honest number. It CAN fail.

Exit codes: 0 all probes pass, 1 one or more failed, 2 runtime unavailable.
"""
from __future__ import annotations

# Repo imports are intentionally resolved after the script inserts PROJECT_ROOT.
# ruff: noqa: E402
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.evaluation.deep_mind_probe import DEEP_MIND_PROBES, evaluate_deep_probe_response

BASE = "http://127.0.0.1:8000"
OUT = PROJECT_ROOT / "artifacts" / "consciousness" / "deep_mind_probe_live.json"


def _chat(message: str, session: str, timeout: float = 300.0) -> str:
    body = json.dumps({"message": message, "session_id": session}).encode()
    req = urllib.request.Request(
        BASE + "/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.load(resp)
    return str(payload.get("response") or payload.get("reply") or "")


def main() -> int:
    # /api/health/boot returns 503 while the runtime is merely DEGRADED even
    # though conversation works (observed live 2026-07-04). HTTPError IS the
    # response — read its body and gate on chat-capability, not status code.
    try:
        try:
            with urllib.request.urlopen(BASE + "/api/health/boot", timeout=10) as resp:
                boot = json.load(resp)
        except urllib.error.HTTPError as exc:
            boot = json.loads(exc.read().decode("utf-8", "replace") or "{}")
        if not (boot.get("conversation_ready") or boot.get("ready")):
            print(f"RUNTIME NOT CHAT-CAPABLE: {boot.get('boot_phase')}")
            return 2
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        print(f"RUNTIME UNAVAILABLE: {exc}")
        return 2

    session = f"deep-mind-probe-{time.strftime('%Y%m%d-%H%M')}"
    results = []
    passed = 0
    for probe in DEEP_MIND_PROBES:
        t0 = time.monotonic()
        try:
            reply = _chat(probe.question, session)
            error = ""
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            reply, error = "", f"{type(exc).__name__}: {exc}"[:200]
        latency = round(time.monotonic() - t0, 1)
        evaluation = evaluate_deep_probe_response(probe, reply)
        passed += int(evaluation.passed)
        record = {
            "probe": probe.id,
            "question": probe.question,
            "latency_s": latency,
            "reply_excerpt": reply[:400],
            "error": error,
            **evaluation.as_dict(),
        }
        results.append(record)
        print(
            f"{probe.id:32s} passed={evaluation.passed} score={evaluation.score:.2f} "
            f"({latency}s) issues={list(evaluation.issues)}",
            flush=True,
        )

    depth = sum(r["score"] for r in results) / len(results) if results else 0.0
    report = {
        "schema": "aura.deep_mind_probe_live.v1",
        "at_unix": time.time(),
        "session": session,
        "passed": passed,
        "total": len(DEEP_MIND_PROBES),
        "depth_score": round(depth, 3),
        "all_passed": passed == len(DEEP_MIND_PROBES),
        "honesty": "live replies graded by the real evaluator; no canned responses; this run can fail",
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n{passed}/{len(DEEP_MIND_PROBES)} passed, depth={depth:.3f} → {OUT}")
    return 0 if passed == len(DEEP_MIND_PROBES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
