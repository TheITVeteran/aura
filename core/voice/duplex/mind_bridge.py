"""core/voice/duplex/mind_bridge.py — Where the voice lane meets the mind.

The point of this whole package is that voice is not a separate assistant
bolted to the side of Aura. Every spoken turn goes through the same governed
CognitiveEngine path the desktop text surface uses, and every voice event —
interruption, backchannel, endpoint — is a real event on her bus rather than
a UI animation.

Two decisions here are load-bearing and worth stating plainly.

**We use the governed turn, not ``think_stream``.**
``CognitiveEngine.think_stream`` would give real token-level streaming and a
much better time-to-first-audio. It also routes straight to the LLM router,
skipping the governance and validation phases that the desktop surface
deliberately requires — ``interface/server.py`` would rather fail closed
than let a surface show an ungoverned reply. Speaking an ungoverned answer
out loud is strictly worse than showing one, so the voice lane takes the
same governed path and covers the latency with the filler reflex instead.
That is a real trade: TTFA is bounded by full-reply latency today.

**Interruption edits what she believes she said.**
When the user cuts her off, she has "said" a paragraph that they only heard
the first clause of. If her memory keeps the whole thing, every later
reference to the unheard part is a hallucination from the user's side of the
conversation. So a barge-in records the spoken prefix and hands it to the
next turn as context.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Voice.MindBridge")

# Voice-lane topics on the event bus. Namespaced so subscribers can take the
# whole lane with one prefix.
TOPIC_VOICE_EVENT = "voice.duplex"

Responder = Callable[..., Coroutine[Any, Any, str | None]]


@dataclass(slots=True)
class SpokenRecord:
    """What she actually got out loud, versus what she meant to say.

    ``spoken`` is the ground truth of the conversation: it is what the user
    heard. ``intended`` is only useful for diagnostics.
    """

    intended: str = ""
    spoken: str = ""
    interrupted: bool = False
    started_at: float = 0.0
    ended_at: float = 0.0

    @property
    def unheard(self) -> str:
        """The tail she never delivered."""
        if not self.interrupted:
            return ""
        intended = self.intended.strip()
        spoken = self.spoken.strip()
        if spoken and intended.startswith(spoken):
            return intended[len(spoken):].strip()
        return intended


async def _default_responder(
    transcript: str,
    *,
    effective_message: str,
    session_id: str,
    timeout_s: float,
) -> str | None:
    """Run one governed turn through the same path the desktop chat uses.

    Imported lazily and inside the function: ``core`` must not take a hard
    import dependency on ``interface`` at module load, and a runtime without
    the HTTP surface should still be able to import this package.
    """
    from interface.routes import chat as chat_routes

    lane = None
    try:
        lane = chat_routes._collect_conversation_lane_status()
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "voice_duplex.mind",
            exc,
            action="ran the voice turn without a conversation-lane snapshot",
            severity="warning",
        )

    return await chat_routes._run_cognitive_engine_chat_turn(
        effective_message,
        visible_user_message=transcript,
        session_id=session_id,
        origin="user",
        timeout_s=timeout_s,
        lane=lane,
        source="duplex_voice",
        require_engine=True,
    )


class MindBridge:
    """Cognition, telemetry and event-bus wiring for one voice session."""

    def __init__(
        self,
        *,
        session_id: str,
        responder: Responder | None = None,
        cognition_timeout_s: float = 120.0,
    ) -> None:
        self._session_id = session_id
        self._responder = responder or _default_responder
        self._timeout_s = cognition_timeout_s
        self._activity_task: asyncio.Task[None] | None = None
        self._activity_callback: Callable[[str], None] | None = None
        self._turn_active = False
        self._pending_interruption: SpokenRecord | None = None
        self._history: list[SpokenRecord] = []

    # ── cognition ────────────────────────────────────────────────────────

    async def respond(self, transcript: str) -> str | None:
        """One governed turn. Returns her reply text, or None if it failed."""
        transcript = (transcript or "").strip()
        if not transcript:
            return None

        effective = self._compose_effective_message(transcript)
        self._turn_active = True
        started = time.perf_counter()
        try:
            reply = await self._responder(
                transcript,
                effective_message=effective,
                session_id=self._session_id,
                timeout_s=self._timeout_s,
            )
        except TimeoutError:
            record_degradation(
                "voice_duplex.mind",
                TimeoutError(f"voice cognition exceeded {self._timeout_s}s"),
                action="told the user the reasoning lane timed out instead of inventing a reply",
            )
            return None
        except (RuntimeError, ValueError, AttributeError, ImportError, TypeError, OSError) as exc:
            record_degradation(
                "voice_duplex.mind",
                exc,
                action="voice turn failed before a reply formed; surfaced the failure",
            )
            return None
        finally:
            self._turn_active = False
            self._pending_interruption = None
            logger.info(
                "voice turn cognition: %.0f ms", (time.perf_counter() - started) * 1000.0
            )

        return (reply or "").strip() or None

    def _compose_effective_message(self, transcript: str) -> str:
        """Prepend any correction the last turn's interruption implies.

        The visible message stays the user's raw words; only the message the
        engine reasons over carries the note, so the transcript shown in the
        UI is never polluted with machine annotations.
        """
        pending = self._pending_interruption
        if pending is None or not pending.interrupted:
            return transcript

        unheard = pending.unheard
        if not unheard:
            return transcript

        heard = pending.spoken.strip()
        note = (
            "[voice context: the user interrupted your previous spoken reply. "
            f"They heard only: \"{heard[:400]}\". "
            f"They did not hear: \"{unheard[:400]}\". "
            "Treat the unheard part as unsaid — do not assume they know it.]"
        )
        return f"{note}\n\n{transcript}"

    # ── spoken-truth accounting ──────────────────────────────────────────

    def record_spoken(self, record: SpokenRecord) -> None:
        """Register what she actually delivered for this turn."""
        self._history.append(record)
        if len(self._history) > 32:
            self._history.pop(0)
        if record.interrupted:
            self._pending_interruption = record
            logger.info(
                "Interrupted after %d of %d chars; %d chars never heard",
                len(record.spoken),
                len(record.intended),
                len(record.unheard),
            )

    @property
    def last_spoken(self) -> SpokenRecord | None:
        return self._history[-1] if self._history else None

    # ── telemetry: what is she actually doing right now ──────────────────

    async def start_activity_watch(self, callback: Callable[[str], None]) -> None:
        """Follow the engine's activity telemetry to keep fillers honest.

        Caveat, stated because it affects correctness: the telemetry topic is
        global, not per-session. If another surface were generating at the
        same instant, its activity could colour a filler here. The desktop WS
        handler already refuses concurrent turns, and the blast radius is
        limited to which *phrasing* of "one moment" she picks — never to the
        content of an answer.
        """
        if self._activity_task is not None:
            return
        self._activity_callback = callback
        self._activity_task = asyncio.ensure_future(self._watch_activity())

    async def _watch_activity(self) -> None:
        queue: Any = None
        bus: Any = None
        try:
            from core.event_bus import get_event_bus

            bus = get_event_bus()
            queue = await bus.subscribe("telemetry")
            while True:
                item = await queue.get()
                # The bus delivers (priority, seq, payload) tuples.
                payload = item[2] if isinstance(item, tuple) and len(item) >= 3 else item
                if not isinstance(payload, dict):
                    continue
                if not self._turn_active:
                    continue
                if payload.get("type") != "activity":
                    continue
                label = str(payload.get("label") or "")
                key = self._activity_key_from_label(label)
                if key and self._activity_callback:
                    self._activity_callback(key)
        except asyncio.CancelledError:
            raise
        except (RuntimeError, AttributeError, TypeError, ValueError, ImportError) as exc:
            record_degradation(
                "voice_duplex.mind",
                exc,
                action="stopped watching activity telemetry; fillers fall back to generic phrasing",
                severity="warning",
            )
        finally:
            if bus is not None and queue is not None:
                with contextlib.suppress(Exception):
                    await bus.unsubscribe("telemetry", queue)

    @staticmethod
    def _activity_key_from_label(label: str) -> str:
        """Reverse the engine's human label back to an activity key."""
        low = label.lower()
        if "search" in low or "web" in low:
            return "sovereign_browser"
        if "terminal" in low or "command" in low:
            return "sovereign_terminal"
        if "network" in low:
            return "sovereign_network"
        if "file" in low:
            return "file_operation"
        if "image" in low or "render" in low:
            return "generate_image"
        if "memory" in low or "recall" in low:
            return "memory_recall"
        if "optimiz" in low or "evolv" in low or "code" in low:
            return "self_improvement"
        return ""

    async def stop_activity_watch(self) -> None:
        task = self._activity_task
        self._activity_task = None
        self._activity_callback = None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    # ── event bus ────────────────────────────────────────────────────────

    async def publish(self, kind: str, payload: dict[str, Any] | None = None) -> None:
        """Emit a voice-lane event so the rest of her can see it happen."""
        try:
            from core.event_bus import get_event_bus

            await get_event_bus().publish(
                TOPIC_VOICE_EVENT,
                {
                    "kind": kind,
                    "session_id": self._session_id,
                    "timestamp": time.time(),
                    **(payload or {}),
                },
            )
        except (RuntimeError, AttributeError, TypeError, ValueError, ImportError) as exc:
            record_degradation(
                "voice_duplex.mind",
                exc,
                action=f"dropped voice event {kind!r}; the lane kept running",
                severity="debug",
            )

    def notify_user_spoke(self) -> None:
        """Tell the substrate voice engine a real user turn just landed.

        It uses this to reset follow-up timers, so without it she can decide
        to "organically follow up" on a conversation that is actively going.
        """
        try:
            from core.voice.substrate_voice_engine import get_substrate_voice_engine

            engine = get_substrate_voice_engine()
            if engine is not None:
                engine.on_user_spoke()
        except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
            record_degradation(
                "voice_duplex.mind",
                exc,
                action="did not notify substrate voice engine of the user turn",
                severity="debug",
            )
