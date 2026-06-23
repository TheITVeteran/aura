#!/usr/bin/env python3
"""Live end-to-end check: grounded positional recall + the two prior fixes.

Establishes a DISTINCTIVE first turn, banters, then asks "do you remember what I
first asked" — and verifies she quotes the real first turn instead of
confabulating, with no introspection dump and no fail-closed.
Bounded: per-turn timeout + internal 600s wall-clock cap.
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8000"
SESSION = f"verify-grounded-{uuid.uuid4().hex[:8]}"
HEADERS = {"X-Aura-Surface": "desktop-ui", "X-Aura-Require-CognitiveEngine": "required"}

# A first message with a distinctive token we can look for in the recall answer.
FIRST_MSG = "are you awake in there, Aura?"
FIRST_TOKENS = ("awake", "in there", "are you awake")

DUMP = ("what i am attending to is", "the remembered concern that should change", "leaning toward")
FAILCLOSED = ("could not produce a reliable", "failed closed", "ask me again in a moment")


def post(client, message, timeout_s):
    try:
        r = client.post(f"{BASE}/api/chat", json={"message": message, "session_id": SESSION}, timeout=timeout_s)
        if r.status_code != 200:
            return False, f"http {r.status_code}", {}
        p = r.json()
        return True, str(p.get("response") or p.get("reply") or p.get("text") or "").strip(), p
    except (httpx.HTTPError, OSError, ValueError) as exc:
        return False, f"{type(exc).__name__}: {exc}", {}


def main() -> int:
    deadline = time.monotonic() + 600
    with httpx.Client(headers=HEADERS) as client:
        # warm up on a SEPARATE throwaway session so it doesn't become the
        # "first" turn of the session under test.
        warm_session = SESSION + "-warm"
        warm = False
        while time.monotonic() < deadline:
            try:
                r = client.post(f"{BASE}/api/chat", json={"message": "hey", "session_id": warm_session}, timeout=90)
                ok = r.status_code == 200
                text = str((r.json() if ok else {}).get("response") or "").strip() if ok else f"http {r.status_code}"
            except (httpx.HTTPError, OSError, ValueError) as exc:
                ok, text = False, f"{type(exc).__name__}: {exc}"
            print(f"[warmup] ok={ok} len={len(text)} :: {text[:60]!r}", flush=True)
            if ok and len(text) > 4 and not any(s in text.lower() for s in FAILCLOSED):
                warm = True
                break
            time.sleep(15)
        if not warm:
            print(json.dumps({"verdict": "FAIL", "reason": "never_warmed"}))
            return 1

        script = [
            ("first", FIRST_MSG),
            ("banter", "cool, just testing something. you good?"),
            ("riff", "nice. anyway, carry on with whatever you were doing"),
            ("recall", "Do you remember what I first asked"),
        ]
        rows = []
        for tid, msg in script:
            ok, text, payload = post(client, msg, 180)
            low = text.lower()
            rows.append({
                "id": tid, "prompt": msg, "ok": ok, "len": len(text), "response": text,
                "confidence": payload.get("response_confidence"),
                "is_dump": any(s in low for s in DUMP),
                "is_failclosed": any(s in low for s in FAILCLOSED),
            })
            print(f"\n[{tid}] ok={ok} len={len(text)} conf={rows[-1]['confidence']} "
                  f"dump={rows[-1]['is_dump']} failclosed={rows[-1]['is_failclosed']}\n  Q: {msg}\n  A: {text}", flush=True)

        rec = {r["id"]: r for r in rows}["recall"]
        rlow = rec["response"].lower()
        checks = {
            "recall_quotes_first_turn": any(tok in rlow for tok in FIRST_TOKENS),
            "recall_no_dump": rec["ok"] and not rec["is_dump"],
            "recall_not_failclosed": rec["ok"] and not rec["is_failclosed"],
        }
        verdict = "PASS" if all(checks.values()) else "FAIL"
        out = {"verdict": verdict, "checks": checks, "first_msg": FIRST_MSG, "rows": rows}
        (ROOT / "artifacts" / "current").mkdir(parents=True, exist_ok=True)
        (ROOT / "artifacts" / "current" / "VERIFY_GROUNDED_RECALL.json").write_text(json.dumps(out, indent=2), "utf-8")
        print("\n" + json.dumps({"verdict": verdict, "checks": checks}, indent=2))
        return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
