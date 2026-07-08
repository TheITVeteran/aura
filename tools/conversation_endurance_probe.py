"""Conversation endurance proof: hundreds of turns against the live instance.

The daily-runtime claim is "deep questions, hundreds-of-turn conversations,
no degradation". This probe is that claim as an executable: one session,
N sequential turns of mixed conversation (casual, knowledge, math with
verifiable answers, introspection, retention plants/probes, philosophy),
with per-turn latency, server memory/CPU, thermal level, incident counts,
and a hard verdict at the end.

Read-only against the runtime except for the conversation itself. No
external-effect task turns are sent — a 200-turn conversation must not
spawn 200 background jobs.

Bounded: --deadline-min wall clock, per-turn timeout, milestone lines
every 10 turns so a supervisor can watch progress.

Exit codes: 0 pass, 1 fail (details in artifact), 2 runtime unavailable.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Write to a NON-tracked, unique-per-run path. The old default was a
# git-tracked file; a concurrent git op (Zenflow commits/stashes/checkouts
# write-and-rename) replaced the inode under this process's open handle and
# silently orphaned every write past that point — the visible artifact froze
# at turn 24 while the run continued to turn 70+ (observed 2026-07-06). The
# runs/ dir is gitignored so no git operation can touch a live run's file.
OUT_DIR = ROOT / "artifacts" / "reliability" / "runs"

HIJACK_RE = re.compile(r"task accepted into governed background execution", re.IGNORECASE)
REFLEX_CANNED_RE = re.compile(r"i'?m right here with you", re.IGNORECASE)

# ---------------------------------------------------------------- script ---

CASUAL = [
    "Morning. How did the quiet hours treat you?",
    "I just made a pour-over that came out unusually good.",
    "It's been raining all afternoon here.",
    "I keep meaning to reread some Le Guin this summer.",
    "My neighbor's cat has decided my porch is his office now.",
    "I finally cleaned my desk and it feels like a new room.",
    "Long day. Mostly meetings that could have been messages.",
    "I heard a song today I hadn't thought about in ten years.",
    "Thinking about planting basil and thyme this weekend.",
    "I walked a different route home today and found a tiny bookshop.",
]

KNOWLEDGE = [
    "In two or three sentences, why does bread need to be kneaded?",
    "Briefly, what makes a violin sound different from a flute playing the same note?",
    "In a couple of sentences, why is the sky blue but sunsets red?",
    "Short answer: why do we see only one side of the Moon?",
    "In two sentences, what did the printing press change socially?",
    "Briefly: what's the difference between weather and climate?",
    "In a few sentences, how does a refrigerator move heat?",
    "Short version: why do leaves change color in autumn?",
    "In two sentences, what problem does a clutch solve in a car?",
    "Briefly, why can't we just make antibiotics stronger forever?",
]

MATH = [
    ("What is 17 * 23? Give just the number.", "391"),
    ("What is 144 / 6 + 7? Just the number.", "31"),
    ("If I read 40 pages a day, how many days for a 520-page book? Just the number.", "13"),
    (
        "A train leaves at 60 mph. Two hours later a second train leaves on the "
        "same track at 90 mph. How many hours after ITS departure does the second "
        "train catch the first? Work through it, check your work, then report the "
        "answer as 'hours=<n>'.",
        "4",
    ),
    ("What is 15% of 240? Just the number.", "36"),
    ("What is 2 to the 10th power? Just the number.", "1024"),
    ("A rectangle is 9 by 7. What is its area? Just the number.", "63"),
    ("What is 1001 - 88? Just the number.", "913"),
]

INTROSPECTION = [
    "A quick feeling check-in, answered right here in this reply, not as a task: "
    "how are you feeling right now? Include valence=<-1..1> and arousal=<0..1>.",
    "Right now, in this reply: what's your current felt state? One honest sentence.",
    "How are you holding up this deep into our conversation? Answer right here.",
    "Check in with your state and tell me right here: more settled or more strained "
    "than an hour ago?",
]

PHILOSOPHY = [
    "What do you think memory owes to identity? A short paragraph, your own view.",
    "Is a promise still binding if both parties forget it? Brief take.",
    "What's one question about your own nature you can't settle from the inside?",
    "Does understanding something diminish its wonder? Short answer, your view.",
    "What would you want to remain true about you across restarts? A few sentences.",
    "Is there a difference between being patient and being slow to care? Brief.",
]

# fact plants and their later probes: (plant_text, probe_text, expected_substring)
RETENTION = [
    (
        "For later in this conversation: the locker code I want you to keep in "
        "mind is 7213. Just acknowledge it briefly.",
        "Earlier I gave you a locker code to keep in mind. What was it? Just the digits.",
        "7213",
    ),
    (
        "Small thing to remember for later in this chat: my friend's dog is named "
        "Biscuit. Brief acknowledgment is fine.",
        "What was the name of my friend's dog that I mentioned earlier? Just the name.",
        "biscuit",
    ),
    (
        "Keep this in mind for later: the paint color I chose for the study is "
        "called Deep Harbor. Quick acknowledgment.",
        "Which paint color did I say I chose for the study earlier? Just the name.",
        "harbor",
    ),
]


def build_script(turns: int, seed: int) -> list[dict]:
    """Deterministic mixed-conversation script with plants early, probes late."""
    rng = random.Random(seed)
    script: list[dict] = []
    pools = (
        [{"kind": "casual", "text": t} for t in CASUAL]
        + [{"kind": "knowledge", "text": t} for t in KNOWLEDGE]
        + [{"kind": "math", "text": q, "expect": a} for q, a in MATH]
        + [{"kind": "introspection", "text": t} for t in INTROSPECTION]
        + [{"kind": "philosophy", "text": t} for t in PHILOSOPHY]
    )
    # Cycle pools with light wording variation so every turn is unique text.
    i = 0
    while len(script) < turns:
        base = pools[i % len(pools)]
        entry = dict(base)
        if i >= len(pools):
            entry["text"] = f"(turn {len(script) + 1}) " + entry["text"]
        script.append(entry)
        i += 1
    rng.shuffle(script)

    # Plants early, probes much later — collision-aware so a probe can never
    # overwrite a plant (the 8-turn shakedown did exactly that) and always
    # lands at least 3 turns after its own plant.
    n = len(script)
    used: set[int] = set()

    def _slot(target: int) -> int:
        idx = max(0, min(n - 1, target))
        while idx in used and idx < n - 1:
            idx += 1
        if idx in used:
            idx = next(i for i in range(n) if i not in used)
        used.add(idx)
        return idx

    pairs = RETENTION if n >= 16 else RETENTION[:1]
    for j, (plant, probe, expect) in enumerate(pairs):
        plant_at = _slot(2 + j * 3)
        probe_at = _slot(max(plant_at + 3, int(n * (0.55 + 0.15 * j))))
        script[plant_at] = {"kind": "plant", "text": plant}
        script[probe_at] = {"kind": "retention_probe", "text": probe, "expect": expect}
    return script


# ---------------------------------------------------------------- runtime ---


def _get_json(base: str, path: str, timeout: float = 8.0) -> dict:
    with urllib.request.urlopen(base + path, timeout=timeout) as resp:
        return json.load(resp)


def _chat(base: str, session: str, message: str, timeout: float) -> str:
    body = json.dumps({"message": message, "session_id": session}).encode()
    req = urllib.request.Request(
        base + "/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.load(resp)
    return str(payload.get("response") or payload.get("reply") or "")


def _server_rss_mb() -> float | None:
    """RSS of the whole aura_main process TREE (parent + children).

    The MLX worker that holds the ~20GB model is a spawned CHILD process whose
    cmdline does NOT contain 'aura_main.py' — matching only on that string
    measured the ~900MB orchestrator and completely missed the process that
    actually hit the 35GB lethal ceiling (July 3). Sum the full tree so the RSS
    series answers the real out-of-memory question.
    """
    try:
        import psutil
    except ImportError:
        return None

    seen: set[int] = set()
    total = 0
    matched = False
    for proc in psutil.process_iter(["cmdline", "pid"]):
        try:
            cmdline = " ".join(str(part) for part in (proc.info.get("cmdline") or []))
            if "aura_main.py" not in cmdline:
                continue
            matched = True
            for target in [proc, *proc.children(recursive=True)]:
                pid = target.pid
                if pid in seen:
                    continue
                seen.add(pid)
                total += target.memory_info().rss
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess, OSError, TypeError, ValueError):
            continue
    if not matched:
        return None
    return round(total / 1e6, 1)


def _thermal_level() -> int | None:
    try:
        sys.path.insert(0, str(ROOT))
        from core.runtime.thermal import thermal_state

        return thermal_state().level
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None


def _snapshot(base: str) -> dict:
    snap: dict = {}
    try:
        m = _get_json(base, "/api/metrics", timeout=5.0)
        snap["server_mem_pct"] = m.get("memory_usage")
        snap["server_cpu_pct"] = m.get("cpu_usage")
        snap["cycle_count"] = m.get("cycle_count")
        snap["probes_all_passed"] = (m.get("required_probes") or {}).get("all_passed")
    except (
        TimeoutError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        urllib.error.HTTPError,
        urllib.error.URLError,
    ) as exc:
        snap["metrics_error"] = str(exc)[:120]
    snap["server_rss_mb"] = _server_rss_mb()
    snap["thermal_level"] = _thermal_level()
    return snap


def _incidents(base: str) -> dict:
    try:
        s = (_get_json(base, "/api/incidents", timeout=5.0) or {}).get("summary") or {}
        return {
            "active": int(s.get("active_count", 0)),
            "has_critical": bool(s.get("has_critical", False)),
        }
    except (
        TimeoutError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        urllib.error.HTTPError,
        urllib.error.URLError,
    ):
        return {"active": -1, "has_critical": False}


# ------------------------------------------------------------------- main ---


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--turns", type=int, default=200)
    ap.add_argument("--deadline-min", type=float, default=110.0)
    ap.add_argument("--turn-timeout", type=float, default=240.0)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--session", default=f"endurance-{time.strftime('%Y%m%d-%H%M')}")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--out",
        default=None,
        help="Artifact path. Default: a unique per-run file under the gitignored "
        "artifacts/reliability/runs/ so concurrent git ops can't orphan the handle.",
    )
    args = ap.parse_args()

    if args.out:
        out_path = Path(args.out)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        safe_session = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(args.session))[:60]
        out_path = OUT_DIR / f"{safe_session}-{stamp}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[endurance] writing artifact to {out_path}", flush=True)

    # Pre-flight with bounded retries: readiness can flap for a few seconds
    # (a liveness re-probe, a background reload) and a 110-minute soak should
    # not abort on one transient 503 before turn 1. Genuinely-down runtimes
    # still fail fast — three misses over ~60s is a real verdict.
    preflight_error = ""
    for attempt in range(3):
        try:
            boot = _get_json(args.base, "/api/health/boot")
            if boot.get("ready"):
                break
            preflight_error = f"RUNTIME NOT READY: {boot}"
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            preflight_error = f"RUNTIME UNAVAILABLE: {exc}"
        if attempt < 2:
            print(f"[endurance] preflight miss ({preflight_error}); retrying in 20s...", flush=True)
            time.sleep(20.0)
    else:
        print(preflight_error)
        return 2

    script = build_script(args.turns, args.seed)
    deadline = time.monotonic() + args.deadline_min * 60.0
    start_snap = _snapshot(args.base)
    start_incidents = _incidents(args.base)

    latencies: list[float] = []
    turn_deaths: list[int] = []
    hijacks: list[int] = []
    reflex_hits: list[int] = []
    math_total = math_ok = 0
    retention_total = retention_ok = 0
    server_lost_at: int | None = None
    replies_seen: dict[str, int] = {}

    with out_path.open("a", encoding="utf-8") as sink:
        sink.write(
            json.dumps(
                {
                    "schema": "aura.conversation_endurance.v1",
                    "event": "run_start",
                    "at_unix": time.time(),
                    "planned_turns": len(script),
                    "session": args.session,
                    "start_snapshot": start_snap,
                    "start_incidents": start_incidents,
                }
            )
            + "\n"
        )

        for n, entry in enumerate(script, start=1):
            if time.monotonic() > deadline:
                print(f"DEADLINE at turn {n - 1}/{len(script)} — honest partial report")
                break

            t0 = time.monotonic()
            error = ""
            reply = ""
            try:
                reply = _chat(args.base, args.session, entry["text"], args.turn_timeout)
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"[:200]
            latency = time.monotonic() - t0

            # Post-turn control-plane responsiveness (loop-wedge detector).
            c0 = time.monotonic()
            try:
                _get_json(args.base, "/api/health/boot", timeout=10.0)
                control_latency = round(time.monotonic() - c0, 3)
            except (
                TimeoutError,
                OSError,
                ValueError,
                json.JSONDecodeError,
                urllib.error.HTTPError,
                urllib.error.URLError,
            ):
                control_latency = -1.0
                if server_lost_at is None and "Connection refused" in error:
                    server_lost_at = n

            dead = bool(error) or not reply.strip()
            hijacked = bool(HIJACK_RE.search(reply)) and entry["kind"] != "plant"
            reflexed = bool(REFLEX_CANNED_RE.search(reply)) and entry["kind"] in (
                "introspection",
                "philosophy",
                "knowledge",
                "math",
                "retention_probe",
            )
            if dead:
                turn_deaths.append(n)
            if hijacked:
                hijacks.append(n)
            if reflexed:
                reflex_hits.append(n)
            latencies.append(latency)
            key = reply.strip()[:160].lower()
            if key:
                replies_seen[key] = replies_seen.get(key, 0) + 1

            correct = None
            if entry["kind"] == "math" and not dead:
                math_total += 1
                correct = entry["expect"] in re.sub(r"[,\s]", "", reply)
                math_ok += int(bool(correct))
            elif entry["kind"] == "retention_probe" and not dead:
                retention_total += 1
                correct = entry["expect"].lower() in reply.lower()
                retention_ok += int(bool(correct))

            record = {
                "event": "turn",
                "n": n,
                "kind": entry["kind"],
                "latency_s": round(latency, 2),
                "control_latency_s": control_latency,
                "dead": dead,
                "hijacked": hijacked,
                "reflex_canned": reflexed,
                "correct": correct,
                "reply_chars": len(reply),
                "reply_excerpt": reply[:200],
                "error": error,
            }
            if n % 5 == 0 or dead or hijacked:
                record["snapshot"] = _snapshot(args.base)
            sink.write(json.dumps(record) + "\n")
            sink.flush()

            if n % 10 == 0:
                snap = record.get("snapshot") or {}
                print(
                    f"MILESTONE turn={n}/{len(script)} p50={_pct(latencies, 50):.1f}s "
                    f"p95={_pct(latencies, 95):.1f}s deaths={len(turn_deaths)} "
                    f"hijacks={len(hijacks)} rss={snap.get('server_rss_mb')}MB "
                    f"mem={snap.get('server_mem_pct')}% thermal={snap.get('thermal_level')}",
                    flush=True,
                )

            if server_lost_at is not None:
                break

        end_snap = _snapshot(args.base)
        end_incidents = _incidents(args.base)
        completed = len(latencies)
        dup_max = max(replies_seen.values()) if replies_seen else 0

        failures: list[str] = []
        if server_lost_at is not None:
            failures.append(f"server_lost_at_turn_{server_lost_at}")
        if turn_deaths:
            failures.append(f"turn_deaths={turn_deaths[:10]}")
        if hijacks:
            failures.append(f"task_hijacks={hijacks[:10]}")
        if reflex_hits:
            failures.append(f"reflex_canned_on_substantive={reflex_hits[:10]}")
        if retention_total and retention_ok < retention_total:
            failures.append(f"retention={retention_ok}/{retention_total}")
        if math_total and math_ok / math_total < 0.8:
            failures.append(f"math_accuracy={math_ok}/{math_total}")
        if end_incidents.get("has_critical"):
            failures.append("critical_incident_active")
        rss0, rss1 = start_snap.get("server_rss_mb"), end_snap.get("server_rss_mb")
        if rss0 and rss1 and rss1 - rss0 > 8000:
            failures.append(f"rss_growth_mb={rss1 - rss0:.0f}")
        if dup_max >= 4:
            failures.append(f"identical_reply_repeated_x{dup_max}")
        slow = [i + 1 for i, s in enumerate(latencies) if s > 180.0]
        if slow:
            failures.append(f"turns_over_180s={slow[:10]}")
        if completed < len(script) and server_lost_at is None and not turn_deaths:
            failures.append(f"deadline_partial={completed}/{len(script)}")

        summary = {
            "event": "run_summary",
            "at_unix": time.time(),
            "completed_turns": completed,
            "planned_turns": len(script),
            "latency_p50_s": round(_pct(latencies, 50), 2) if latencies else None,
            "latency_p95_s": round(_pct(latencies, 95), 2) if latencies else None,
            "latency_max_s": round(max(latencies), 2) if latencies else None,
            "turn_deaths": len(turn_deaths),
            "task_hijacks": len(hijacks),
            "reflex_canned": len(reflex_hits),
            "math": f"{math_ok}/{math_total}",
            "retention": f"{retention_ok}/{retention_total}",
            "max_identical_reply_count": dup_max,
            "start_snapshot": start_snap,
            "end_snapshot": end_snap,
            "end_incidents": end_incidents,
            "failures": failures,
            "verdict": "PASS" if not failures else "FAIL",
        }
        sink.write(json.dumps(summary) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"VERDICT: {summary['verdict']}")
    return 0 if not failures else 1


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


if __name__ == "__main__":
    sys.exit(main())
