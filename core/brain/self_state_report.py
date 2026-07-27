"""Her own instrument panel, in words, read at the moment she is asked.

Stopping the web search for "how much memory are you holding" is only half a
fix. The other half is the answer. Asked for one concrete thing that had
happened in her runtime in the last hour, she said:

    "I processed a user request to summarize a 45-page PDF on neuromorphic
     computing. It took about three minutes ..."

No such request existed. It is the same failure as inventing the weather: a
question about a present she had no channel to. Every number here is read from
a live source at call time — the process, the host, the service container, the
degradation ledger — and anything unavailable is omitted rather than guessed,
because a missing line is honest and a plausible one is not.

Bounded on purpose: a handful of lines, no subsystem sweeps, no health report
assembly. This runs on a foreground turn while the user waits.
"""
from __future__ import annotations

import os
import time

from core.runtime.errors import record_degradation

_RECOVERABLE = (RuntimeError, AttributeError, TypeError, ValueError, OSError, ImportError, KeyError)

SELF_STATE_HEADER = "## YOUR OWN INSTRUMENTS"


def _humanize(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{int(seconds)} seconds"
    if seconds < 5400:
        return f"{seconds / 60:.0f} minutes"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} hours"
    return f"{seconds / 86400:.1f} days"


def _uptime_line() -> str:
    try:
        from core.runtime.service_registry import get_runtime_service

        orch = get_runtime_service("orchestrator", default=None)
        for candidate in (
            getattr(orch, "start_time", None),
            getattr(getattr(orch, "status", None), "start_time", None),
        ):
            try:
                start = float(candidate or 0.0)
            except (TypeError, ValueError):
                continue
            if start > 0.0:
                elapsed = max(0.0, time.time() - start)
                started = time.strftime("%H:%M", time.localtime(start))
                return f"- Uptime: {_humanize(elapsed)} (this runtime started at {started})."
    except _RECOVERABLE as exc:
        record_degradation("self_state_report", exc, severity="info", action="omitted uptime line")
    return ""


def _memory_lines() -> list[str]:
    """RSS understates her badly on Apple Silicon; say both numbers."""
    lines: list[str] = []
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        rss_gb = proc.memory_info().rss / 1e9
        virt = psutil.virtual_memory()
        lines.append(
            f"- This process holds {rss_gb:.1f}GB resident; the host is at "
            f"{virt.percent:.0f}% of {virt.total / 1e9:.0f}GB with "
            f"{virt.available / 1e9:.1f}GB available."
        )
        lines.append(
            "- Your model's weights live in wired GPU memory and do NOT appear "
            "in that resident figure — the real total is larger."
        )
    except _RECOVERABLE as exc:
        record_degradation("self_state_report", exc, severity="info", action="omitted memory lines")
    return lines


def _model_line() -> str:
    try:
        from core.container import ServiceContainer

        client = ServiceContainer.peek("mlx_client", default=None)
        for attr in ("model_path", "model_name", "_model_path"):
            value = str(getattr(client, attr, "") or "").strip()
            if value:
                return f"- Cortex model actually loaded: {os.path.basename(value)}."
    except _RECOVERABLE as exc:
        record_degradation("self_state_report", exc, severity="info", action="omitted model line")
    return ""


def _degradation_line() -> str:
    """What has actually gone wrong lately, from the ledger that records it."""
    try:
        from core.runtime.errors import get_degradation_tracker

        status = get_degradation_tracker().status() or {}
        total = int(status.get("total_degradations") or 0)
        by_subsystem = status.get("counts_by_subsystem") or {}
    except _RECOVERABLE:
        return ""
    if not total:
        return "- No degradations recorded this session."
    try:
        ranked = sorted(
            ((name, sum(int(n) for n in sevs.values())) for name, sevs in by_subsystem.items()),
            key=lambda pair: -pair[1],
        )[:3]
        summary = ", ".join(f"{name} x{count}" for name, count in ranked)
        return f"- Degradations recorded this session: {total} total ({summary})."
    except _RECOVERABLE as exc:
        record_degradation(
            "self_state_report", exc, severity="info", action="omitted degradation line"
        )
        return ""


def runtime_self_report() -> str:
    """A short, true readout of her machine state right now.

    Returns "" when nothing could be read, so a caller never pastes an empty
    heading into the prompt and invites her to fill it in.
    """
    lines = [line for line in (_uptime_line(), _model_line()) if line]
    lines.extend(_memory_lines())
    degradations = _degradation_line()
    if degradations:
        lines.append(degradations)
    if not lines:
        return ""
    return "\n".join(
        [
            SELF_STATE_HEADER,
            "Read from your live runtime just now, for this question. These are "
            "your actual readings — quote them, and do not supplement them with "
            "numbers or events you cannot see here.",
            *lines,
        ]
    )


__all__ = ["SELF_STATE_HEADER", "runtime_self_report"]
