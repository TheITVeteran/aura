#!/usr/bin/env python3
"""Where her time goes while she plays an archived game: the thread she thinks on, looked at every few milliseconds.

    python tools/where_her_time_goes.py --out /tmp/where --games 20 --minutes 3

Everything after ``--out DIR`` is handed to tools/measure_on_archived_games.py. A second thread reads the main
thread's stack every 4 ms and writes ``DIR/where_her_time_goes.txt`` when the run ends. It answers the questions
that have each found something to make faster:

- which of her functions were on the stack, and which was the innermost of hers (her own work, or a library she called);
- for the time a library was innermost, who of hers called it (2026-10-12: a tenth of a game went to file-system
  checks made on every call to the browser, and to reading her remembered worlds from disk again);
- when a module was being loaded mid-run, and who asked for it;
- every time the loop was kept busy for a fifth of a second or more, and by whom: while it is, no key is pressed
  and no picture looked at;
- where each of her tasks was waiting while the loop was idle.

Nothing here changes how she plays. The sampler takes the interpreter lock for a few microseconds each look.
"""
from __future__ import annotations

import asyncio
import collections
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: How often the main thread's stack is read, and how long the loop is busy at a stretch before it is listed.
EVERY_S = 0.004
KEPT_BUSY_S = 0.2
#: The folders whose code is hers.
_HERS = tuple(os.sep + part + os.sep for part in ("core", "tools", "skills", "interface", "llm", "executors", "security"))


def _hers(code: Any) -> str:
    """A function of hers as ``folder/file.py:function``; "" for anyone else's."""
    path = code.co_filename
    for mark in _HERS:
        if mark in path:
            return path[path.rindex(mark) + 1:] + ":" + code.co_name
    return ""


def _whoevers(code: Any) -> str:
    return code.co_filename.split("site-packages/")[-1].split("lib/python3.12/")[-1].split(os.sep)[-1] + ":" + code.co_name


class WhereHerTimeGoes:
    """The main thread's stack, read every few milliseconds from a thread of its own."""

    def __init__(self) -> None:
        self.looks = 0
        self.on_the_stack: collections.Counter[str] = collections.Counter()
        self.innermost_of_hers: collections.Counter[str] = collections.Counter()
        self.innermost: collections.Counter[str] = collections.Counter()
        self.a_library_for: collections.Counter[str] = collections.Counter()
        self.loading_for: collections.Counter[str] = collections.Counter()
        self.waiting_at: collections.Counter[str] = collections.Counter()
        self.kept_busy: list[tuple[float, list[tuple[str, int]]]] = []
        self._busy_since: float | None = None
        self._busy_with: collections.Counter[str] = collections.Counter()
        self._stop = threading.Event()
        self._main = threading.main_thread().ident
        self._thread = threading.Thread(target=self._run, daemon=True, name="where-her-time-goes")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            frame = sys._current_frames().get(self._main)
            if frame is not None:
                self._look(frame)
            time.sleep(EVERY_S)

    def _look(self, frame: Any) -> None:
        self.looks += 1
        leaf = _whoevers(frame.f_code)
        self.innermost[leaf] += 1
        seen: set[str] = set()
        first, loading, walked = "", False, frame
        while walked is not None:
            name = _hers(walked.f_code)
            if name:
                first = first or name
                seen.add(name)
            loading = loading or "importlib._bootstrap" in walked.f_code.co_filename
            walked = walked.f_back
        for name in seen:
            self.on_the_stack[name] += 1
        self.innermost_of_hers[first or "(none of hers: idle, or a library on its own)"] += 1
        if first and not _hers(frame.f_code):
            self.a_library_for[f"{leaf}  <-  {first}"] += 1
        if loading:
            self.loading_for[first or "(none of hers)"] += 1
        now = time.perf_counter()
        if leaf.endswith("selectors.py:select"):
            self._waits()
            if self._busy_since is not None and now - self._busy_since >= KEPT_BUSY_S:
                self.kept_busy.append((now - self._busy_since, self._busy_with.most_common(3)))
            self._busy_since, self._busy_with = None, collections.Counter()
        else:
            self._busy_since = now if self._busy_since is None else self._busy_since
            self._busy_with[first or leaf] += 1

    def _waits(self) -> None:
        """Where each of her tasks is waiting: the deepest coroutine of hers in each."""
        for task in list(getattr(asyncio.tasks, "_scheduled_tasks", ())):
            try:
                if task.done():
                    continue
                waiting, deepest = task.get_coro(), ""
                for _depth in range(80):
                    inside = getattr(waiting, "cr_frame", None) or getattr(waiting, "gi_frame", None)
                    name = _hers(inside.f_code) if inside is not None else ""
                    if name:
                        deepest = f"{name}:{inside.f_lineno}"
                    waiting = getattr(waiting, "cr_await", None) or getattr(waiting, "gi_yieldfrom", None)
                    if waiting is None:
                        break
                if deepest:
                    self.waiting_at[deepest] += 1
            except (RuntimeError, ValueError, AttributeError):      # a task that ended as it was looked at
                continue

    def said(self) -> str:
        total = max(1, self.looks)

        def shares(counted: collections.Counter[str], most: int) -> list[str]:
            return [f"  {100 * times / total:5.1f}%  {what}" for what, times in counted.most_common(most)]

        busy = sorted(self.kept_busy, key=lambda one: -one[0])
        return "\n".join([
            f"{self.looks} looks at the main thread, one every {EVERY_S * 1000:.0f} ms", "",
            "ON THE STACK (hers)", *shares(self.on_the_stack, 70), "",
            "INNERMOST OF HERS", *shares(self.innermost_of_hers, 50), "",
            "INNERMOST OF ALL", *shares(self.innermost, 30), "",
            "A LIBRARY, AND WHO OF HERS CALLED IT", *shares(self.a_library_for, 40), "",
            f"A MODULE BEING LOADED ({100 * sum(self.loading_for.values()) / total:.1f}% in all), AND WHO ASKED",
            *shares(self.loading_for, 20), "",
            f"THE LOOP KEPT BUSY {KEPT_BUSY_S} s OR MORE AT A STRETCH: {len(busy)} times, "
            f"{sum(seconds for seconds, _who in busy):.1f} s in all",
            *[f"  {seconds:5.2f} s  {who}" for seconds, who in busy[:30]], "",
            "WAITING WHILE THE LOOP IS IDLE (the deepest coroutine of hers in each task; tasks overlap)",
            *shares(self.waiting_at, 40), "",
        ])


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] != "--out":
        print(__doc__)
        return 2
    out = Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    from tools import measure_on_archived_games as harness

    watch = WhereHerTimeGoes()
    watch.start()
    try:
        asyncio.run(harness.main(argv[2:]))
    finally:
        watch.stop()
        (out / "where_her_time_goes.txt").write_text(watch.said())
        print(f"written: {out / 'where_her_time_goes.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
