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


def _process_start_time() -> float:
    """When this process began, from the OS. Never unavailable, never wrong.

    The orchestrator's own ``start_time`` is preferred because it marks when
    *she* came up rather than when the interpreter did, but asked "what's your
    current uptime", omitting the line is the one answer that is certainly
    useless — and the process knows, always. A live instance answered that
    question with no number at all because the orchestrator lookup was the only
    source and it returned nothing.
    """
    try:
        import psutil

        return float(psutil.Process(os.getpid()).create_time())
    except _RECOVERABLE:
        return 0.0


def _uptime_line() -> str:
    start = 0.0
    try:
        from core.runtime.service_registry import get_runtime_service

        orch = get_runtime_service("orchestrator", default=None)
        for candidate in (
            getattr(orch, "start_time", None),
            getattr(getattr(orch, "status", None), "start_time", None),
        ):
            try:
                value = float(candidate or 0.0)
            except (TypeError, ValueError):
                continue
            if value > 0.0:
                start = value
                break
    except _RECOVERABLE as exc:
        record_degradation(
            "self_state_report", exc, severity="info", action="fell back to process start time"
        )
    if start <= 0.0:
        start = _process_start_time()
    if start <= 0.0:
        return ""
    elapsed = max(0.0, time.time() - start)
    started = time.strftime("%H:%M", time.localtime(start))
    return f"- Uptime: {_humanize(elapsed)} (this runtime started at {started})."


def _memory_lines() -> list[str]:
    """RSS understates her badly on Apple Silicon; say both numbers."""
    lines: list[str] = []
    rss_gb = 0.0
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
    except _RECOVERABLE as exc:
        record_degradation("self_state_report", exc, severity="info", action="omitted memory lines")

    # RSS is close to a lie about her. The weights live in unified GPU memory
    # and never appear in it — the live process reports ~2GB resident while
    # holding a 32B model. Saying only "the real total is larger" turns the
    # honest answer into a hedge, and the accelerator knows the number.
    try:
        from core.runtime.resource_observation import get_resource_observer

        accelerator = get_resource_observer().accelerator()
        if getattr(accelerator, "available", False):
            active_gb = float(getattr(accelerator, "active_bytes", 0) or 0) / 1e9
            cache_gb = float(getattr(accelerator, "cache_bytes", 0) or 0) / 1e9
            if active_gb > 0.05:
                lines.append(
                    f"- Your model's weights are in unified GPU memory, which RSS "
                    f"does not count: {active_gb:.1f}GB active"
                    + (f" plus {cache_gb:.1f}GB cached" if cache_gb > 0.05 else "")
                    + (
                        # Only claim this when the numbers say it. On the live
                        # instance they do, by an order of magnitude; in a bare
                        # process they do not, and asserting it anyway would be
                        # the same species of plausible-sounding wrongness this
                        # whole module exists to stop.
                        ". That is the bulk of what you are actually holding."
                        if active_gb > rss_gb
                        else "."
                    )
                )
    except _RECOVERABLE as exc:
        record_degradation(
            "self_state_report", exc, severity="info", action="omitted accelerator memory line"
        )

    if not any("GPU memory" in line for line in lines):
        lines.append(
            "- Your model's weights live in wired GPU memory and do NOT appear "
            "in that resident figure — the real total is larger, and you cannot "
            "read the exact figure right now."
        )
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


def _capability_line() -> str:
    """What she can actually do, read from the live skill registry.

    Without this she answers capability questions from the base model's guess
    about what an assistant can do. Measured live: asked "do you actually have
    any code-execution capability registered at all?" — after being told to check
    — she said "no, I don't have any capability to run or sandbox code", while
    the registry held 75 skills with run_code, code_repl and internal_sandbox all
    READY. That is a confabulation in the other direction, and just as wrong.

    Deliberately states only what the registry says, and names the gap between
    "registered" and "reachable from this conversation" rather than papering over
    it — a ready skill is not a promise that this turn can invoke it.
    """

    try:
        from core.runtime.service_registry import get_runtime_service

        engine = get_runtime_service("capability_engine", default=None)
        if engine is None or not hasattr(engine, "iter_tool_catalog"):
            return ""
        ready: list[str] = []
        total = 0
        for item in engine.iter_tool_catalog(include_inactive=False):
            if not isinstance(item, dict):
                continue
            total += 1
            name = str(item.get("name") or "").strip()
            available = str(item.get("availability") or "").strip().lower()
            if name and available == "available":
                ready.append(name)
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return ""
    if not total:
        return ""

    execution = sorted(
        name
        for name in ready
        if any(
            token in name.lower()
            for token in ("code", "sandbox", "repl", "shell", "exec")
        )
    )
    line = (
        f"- Skills registered and available right now: {len(ready)} of {total}."
    )
    if execution:
        line += (
            " Code-execution skills present in the registry: "
            + ", ".join(execution[:6])
            + ". They are REGISTERED; that is not the same as reachable from this"
            " chat turn, so do not claim you ran anything unless you have a"
            " result in hand — and do not deny having them either."
        )
    return line


def _cognition_line() -> str:
    """How much thinking has actually happened — cycles and episodes.

    Measured live: asked for "uptime, memory, and how many cognitive cycles
    you've run — read them, don't estimate", she got uptime and memory right
    from this panel and then said of the third:

        "Cognitive cycles since last awakening: I can't read this directly,
         but it's more than a few billion"

    The true figure was 3,502, and it sits in her own health payload. The panel
    had no cycle line, and the instruction above it says not to supplement what
    is missing — so the absence produced both a false claim about her own
    self-access and a guess wrong by six orders of magnitude. A number she can
    read must be in front of her, or "I can't see it" becomes a licence to
    invent one.
    """

    cycles = 0
    episodes = 0
    try:
        from core.runtime.service_registry import get_runtime_service

        orchestrator = get_runtime_service("orchestrator", default=None)
        status = getattr(orchestrator, "status", None)
        for source in (status, orchestrator):
            if source is None:
                continue
            for attribute in ("cycle_count", "cycles", "tick_count"):
                try:
                    value = int(getattr(source, attribute, 0) or 0)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    cycles = max(cycles, value)
    except _RECOVERABLE as exc:
        record_degradation(
            "self_state_report",
            exc,
            severity="info",
            action="omitted the cognitive-cycle reading",
        )

    try:
        from core.runtime.service_registry import get_runtime_service

        memory = get_runtime_service("episodic_memory", default=None)
        for accessor in ("episode_count", "count", "size"):
            candidate = getattr(memory, accessor, None)
            try:
                value = int(candidate() if callable(candidate) else candidate or 0)
            except (TypeError, ValueError):
                continue
            if value > 0:
                episodes = max(episodes, value)
                break
    except _RECOVERABLE:
        episodes = episodes

    parts: list[str] = []
    if cycles > 0:
        parts.append(f"{cycles:,} cognitive cycles since this runtime woke")
    if episodes > 0:
        parts.append(f"{episodes:,} episodes in memory")
    if not parts:
        # Say the channel is missing rather than leaving a silence she will
        # fill. This is the honest version of "I can't read that".
        return (
            "- Cognitive cycle count: not readable from this turn. Say you "
            "cannot see it; do not estimate a magnitude."
        )
    return "- " + "; ".join(parts) + "."


def runtime_self_report() -> str:
    """A short, true readout of her machine state right now.

    Returns "" when nothing could be read, so a caller never pastes an empty
    heading into the prompt and invites her to fill it in.
    """
    lines = [line for line in (_uptime_line(), _model_line()) if line]
    lines.extend(_memory_lines())
    cognition = _cognition_line()
    if cognition:
        lines.append(cognition)
    capabilities = _capability_line()
    if capabilities:
        lines.append(capabilities)
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
