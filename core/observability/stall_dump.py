"""core/observability/stall_dump.py — truth-preferring stall-dump attribution.

A stall dump is a snapshot of EVERY thread at the moment the event loop
froze. Most of those threads are innocent — parked in sleeps, queue gets,
selector waits — and a parser that grabs "the first project frame" (or the
last) routinely blames a bystander: the July triage run fingerprinted 19
stalls to ``flagship_doctor._monitor_loop`` whose culprit frame was
``time.sleep`` on a daemon thread, while the REAL stall was a synchronous
SQLite ``COUNT(*)`` running on the event loop inside the research pipeline.
Misattribution is worse than no attribution — it sends the next engineer
(or Aura's own narrator) chasing a ghost.

This module is the ONE parser both consumers share (``tools/crash_triage``
and the incident narrator), and it prefers truth in layers:

1. **Stamped loop thread** — new dumps mark the event-loop thread
   (``[EVENT LOOP]``, written by the stall watchdog, which learns the loop's
   thread id from its own heartbeat callback). If that thread is executing
   real work, its deepest project frame is the culprit. No guessing.
2. **Callback-execution heuristic** (old dumps) — a thread whose stack shows
   ``asyncio/events.py … in _run`` is the loop actively running a callback;
   same rule as above.
3. **GIL suspects** — if the loop thread itself looks parked (a starved loop
   often does), the culprit is busy NON-idle threads: pure-Python work on a
   background thread holds the GIL and starves the loop just as effectively.
   The most common deepest project frame among busy threads wins, so one
   anatomy yields one stable fingerprint.
4. Anything else is honestly ``unknown`` — never a sleeping thread.

Dependency-free (stdlib only) so the standalone triage tool can import it
without dragging the runtime along.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["StallCulprit", "parse_stall_dump", "parse_stall_dump_text"]

_STALL_HEADER = re.compile(r"STALL DETECTED:\s*(?P<seconds>[0-9.]+)s")
_LOOP_THREAD_HEADER = re.compile(r"^LOOP THREAD:\s*(?P<tid>\d+)\s*$", re.MULTILINE)
_THREAD_SECTION = re.compile(r"^Thread ID:\s*(?P<tid>\d+)(?P<marker>[^\n]*)$", re.MULTILINE)
_FRAME = re.compile(r'File "(?P<path>[^"]+)", line (?P<line>\d+), in (?P<fn>\S+)')

# Project frames: the only frames worth naming in a fingerprint.
_REPO_MARKERS = ("/core/", "/interface/", "/tools/", "aura_main")
# Wrapper/plumbing frames that appear in every thread and carry no signal.
_INFRA_FILES = (
    "runtime_hygiene.py",
    "aura_logging.py",
    "task_tracker.py",
    "concurrency.py",
)
# A thread whose DEEPEST frame is one of these is parked, not working:
# blocking waits release the GIL and cannot stall the loop.
_IDLE_FUNCTIONS = frozenset(
    {
        "wait",
        "acquire",
        "sleep",
        "select",
        "control",
        "get",
        "get_nowait",
        "dequeue",
        "_feed",
        "_worker",
        "join",
        "kevent",
        "poll",
        "epoll",
        "accept",
        "recv",
        "recv_into",
        "read",
        "readline",
        "readinto",
        "settimeout",
        "wait_for",
    }
)
# The loop executing a scheduled callback shows this exact frame.
_CALLBACK_FRAME = re.compile(r"asyncio/events\.py\", line \d+, in _run")
# The loop parked in its selector shows one of these as the deepest frame.
_SELECTOR_FILES = ("selectors.py",)
# Blocking C calls (sleep, lock waits, socket waits) leave NO Python frame —
# the deepest visible frame is their CALLER, whose function name says
# nothing. The executing source line still names the call: a thread whose
# final source line is one of these is parked, not working.
# Only C-LEVEL blocking calls belong here (they alone are frameless);
# Python-level waits (queue.get, threading.wait) already surface through
# the frame-name check, and generic names like ``.get(`` would misclassify
# busy threads executing innocent dict lookups.
_IDLE_CALL_SNIPPETS = (
    ".sleep(",
    "sleep(",
    ".acquire(",
    ".select(",
    ".control(",
    ".recv(",
    ".accept(",
    ".poll(",
    "kevent(",
)


@dataclass(frozen=True)
class StallCulprit:
    """Attribution verdict for one stall dump."""

    elapsed_s: float
    file_name: str  # e.g. "local_corpus.py" ("" when unknown)
    line: int  # 0 when unknown
    function: str  # e.g. "document_count" ("" when unknown)
    thread_kind: str  # "event_loop" | "gil_suspect" | "unknown"

    @property
    def known(self) -> bool:
        return bool(self.file_name)

    def fingerprint_frame(self) -> str:
        """Stable ``file.py:function`` form for incident-class fingerprints
        (line numbers churn across commits; identities must not)."""
        if not self.known:
            return "unknown_frame"
        return f"{self.file_name}:{self.function}"

    def described(self) -> str:
        """Human form with the line number, for narration."""
        if not self.known:
            return ""
        return f"{self.file_name}:{self.line} ({self.function})"


def parse_stall_dump(path: Path) -> StallCulprit:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return StallCulprit(0.0, "", 0, "", "unknown")
    return parse_stall_dump_text(text)


def parse_stall_dump_text(text: str) -> StallCulprit:
    elapsed = 0.0
    header = _STALL_HEADER.search(text)
    if header:
        elapsed = float(header.group("seconds"))

    sections = _split_thread_sections(text)
    if not sections:
        return StallCulprit(elapsed, "", 0, "", "unknown")

    # Layer 1 + 2: the loop thread, stamped or inferred, actively working.
    loop_section = _find_loop_section(text, sections)
    if loop_section is not None and _is_busy(loop_section):
        frame = _deepest_repo_frame(loop_section)
        if frame is not None:
            return StallCulprit(elapsed, *frame, "event_loop")

    # Layer 3: busy background threads holding the GIL. The watchdog's own
    # reporter thread is excluded — it is composing THIS dump, so it is
    # busy by definition and can never be the suspect.
    culprits: list[tuple[str, int, str]] = []
    for _tid, _marker, body in sections:
        if body is loop_section:
            continue
        if "stall_watchdog.py" in body:
            continue
        if not _is_busy(body):
            continue
        frame = _deepest_repo_frame(body)
        if frame is not None:
            culprits.append(frame)
    if culprits:
        # One anatomy → one stable fingerprint: most common wins, ties
        # break lexicographically.
        counts: dict[str, tuple[int, tuple[str, int, str]]] = {}
        for frame in culprits:
            key = f"{frame[0]}:{frame[2]}"
            count, _ = counts.get(key, (0, frame))
            counts[key] = (count + 1, frame)
        _key, (_count, best) = min(
            counts.items(), key=lambda item: (-item[1][0], item[0])
        )
        return StallCulprit(elapsed, *best, "gil_suspect")

    return StallCulprit(elapsed, "", 0, "", "unknown")


# ── internals ────────────────────────────────────────────────────────────


def _split_thread_sections(text: str) -> list[tuple[str, str, str]]:
    """Return (thread_id, marker, section_body) per thread."""
    matches = list(_THREAD_SECTION.finditer(text))
    sections: list[tuple[str, str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((match.group("tid"), match.group("marker"), text[start:end]))
    return sections


def _find_loop_section(text: str, sections: list[tuple[str, str, str]]) -> str | None:
    # Stamped marker (new dumps).
    for _tid, marker, body in sections:
        if "EVENT LOOP" in marker:
            return body
    # Stamped header (new dumps, marker-less fallback).
    header = _LOOP_THREAD_HEADER.search(text)
    if header:
        for tid, _marker, body in sections:
            if tid == header.group("tid"):
                return body
    # Heuristic (old dumps): the thread running an asyncio callback.
    for _tid, _marker, body in sections:
        if _CALLBACK_FRAME.search(body):
            return body
    return None


def _frames(body: str) -> list[tuple[str, int, str]]:
    return [
        (match.group("path"), int(match.group("line")), match.group("fn"))
        for match in _FRAME.finditer(body)
    ]


def _is_busy(body: str) -> bool:
    """A thread is busy when its DEEPEST frame is doing work — not parked in
    a blocking wait (which releases the GIL) or the loop's own selector."""
    matches = list(_FRAME.finditer(body))
    if not matches:
        return False
    last = matches[-1]
    path, fn = last.group("path"), last.group("fn")
    if fn in _IDLE_FUNCTIONS:
        return False
    if any(marker in path for marker in _SELECTOR_FILES):
        return False
    # Blocking C calls (time.sleep, lock.acquire, socket waits) have no
    # Python frame; the executing SOURCE LINE after the deepest frame is
    # the only witness that this thread is parked.
    trailing = body[last.end() :].strip().splitlines()
    if trailing:
        source_line = trailing[0].strip()
        if any(snippet in source_line for snippet in _IDLE_CALL_SNIPPETS):
            return False
    return True


def _deepest_repo_frame(body: str) -> tuple[str, int, str] | None:
    """The project frame nearest the top of the stack — the code that was
    actually executing, not the entry point it descended from."""
    for path, line, fn in reversed(_frames(body)):
        if "site-packages" in path:
            continue
        if not any(marker in path for marker in _REPO_MARKERS):
            continue
        name = Path(path).name
        if name in _INFRA_FILES:
            continue
        return (name, line, fn)
    return None
