"""Companion mode: she keeps looking, so she already knows when you ask.

WHAT THIS IS FOR
────────────────
When the Aura window is closed she is still running, and the only way she can
answer "what do you think of this?" without a long pause is to have been
looking already. Latency is the whole point: a question about the screen
should be answered from something she saw thirty seconds ago, not from a
capture that starts when you finish typing.

That means a continuous observation loop, and a continuous observation loop
pointed at a person's screen is the most invasive thing in this codebase. So
the privacy rules are structural, not policy:

  INCOGNITO IS NEVER READ.       Not read-then-discarded, not read-then-
                                 redacted. The capture does not happen. A
                                 private window is the one place a person has
                                 explicitly said "not this", and honouring
                                 that after reading it is not honouring it.

  SUPPRESSION WINS.              The same suppression that stops her speaking
                                 unprompted stops her looking unprompted.
                                 Reading someone's screen is not a lesser act
                                 than talking to them.

  NOTHING IS RETAINED RAW.       Captures land in ObservationMemory, which
                                 ages them out, and reach reasoning through
                                 Observation.for_reasoning() — labelled, not
                                 as a blob. That path is already the one that
                                 stopped a screen dump being reproduced
                                 verbatim as an answer.

WHAT MAKES IT CHEAP
───────────────────
Re-reading an unchanged screen costs a capture and buys nothing. The loop
watches the frontmost app and window title first — a cheap query — and only
pays for a text capture when the CONTEXT changed. An idle machine costs
almost nothing; a person moving between windows costs one capture per move.

WHAT MAKES HER QUIET
────────────────────
Having something to say is not a reason to say it. The bubble stays empty
unless an utterance clears the Will, and the default is silence: a companion
that comments on everything is a companion you turn off.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock

AMBIENT_SCHEMA = "aura.perception.ambient_presence.v1"

#: Never observed. Matched against the frontmost window's title and the app
#: name, and a match means the capture is not taken at all.
#:
#: Deliberately broad, and deliberately biased toward refusing: a false
#: positive costs one skipped observation, a false negative reads something
#: the person explicitly marked private.
_PRIVATE_WINDOW_MARKERS: tuple[str, ...] = (
    "incognito",
    "private browsing",
    "private window",
    "inprivate",
    "guest",
    "1password",
    "bitwarden",
    "keychain access",
    "keeper",
    "lastpass",
    "dashlane",
    "authenticator",
    "banking",
    "password",
)

#: Apps whose whole purpose is holding secrets. Never read regardless of title.
_PRIVATE_APPS: frozenset[str] = frozenset(
    {
        "1password",
        "1password 7",
        "bitwarden",
        "keychain access",
        "keeper password manager",
        "lastpass",
        "dashlane",
        "gpg keychain",
        "secretive",
    }
)

_PRIVATE_RE = re.compile("|".join(re.escape(m) for m in _PRIVATE_WINDOW_MARKERS), re.I)

#: How long a context stays "the same thing" before a re-read is worth paying
#: for even when the title has not changed — a long document being scrolled is
#: the same window and different content.
_RESTALE_AFTER_S = 90.0

#: How recently the bubble must have polled for it to count as a surface that
#: can actually draw. Comfortably longer than its 4s idle cadence, short
#: enough that a launcher which quit is noticed before she claims to have
#: pointed at something on a screen nobody is drawing on.
_SURFACE_ALIVE_S = 15.0

#: A queued highlight that nobody collected is stale — the screen has moved on
#: and a rectangle drawn around where something USED to be is worse than no
#: rectangle. Roughly two idle polls.
_HIGHLIGHT_TTL_S = 10.0


class PresenceMode(StrEnum):
    """What surface she is present on right now."""

    #: The full desktop window is open. The bubble does not exist.
    WINDOW = "window"
    #: Window closed, runtime alive. Bubble only.
    BUBBLE = "bubble"
    #: The person hid her. She observes nothing and says nothing.
    HIDDEN = "hidden"


class SkipReason(StrEnum):
    """Why an observation tick did not observe. Every one is reportable."""

    NONE = "none"
    PRIVATE_WINDOW = "private_window"
    SUPPRESSED = "proactivity_suppressed"
    HIDDEN = "hidden_by_user"
    UNCHANGED = "context_unchanged"
    NO_PERMISSION = "no_permission"
    CAPTURE_FAILED = "capture_failed"


@dataclass(frozen=True)
class ScreenContext:
    """The cheap query: what is in front, without reading anything."""

    app: str = ""
    title: str = ""
    at: float = field(default_factory=time.time)

    @property
    def key(self) -> str:
        return f"{self.app.strip().lower()}|{self.title.strip().lower()}"

    @property
    def is_private(self) -> bool:
        """Is this a window the person has marked as not-for-observation?

        Checked BEFORE any capture. The app name and the window title are
        both consulted because a private tab shows in the title and a
        password manager shows in the app.
        """
        app = self.app.strip().lower()
        if app in _PRIVATE_APPS:
            return True
        haystack = f"{self.app} {self.title}"
        return bool(_PRIVATE_RE.search(haystack))


@dataclass
class TickResult:
    """What one ambient tick did, and why."""

    observed: bool
    skip_reason: SkipReason = SkipReason.NONE
    context: ScreenContext | None = None
    characters: int = 0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": AMBIENT_SCHEMA,
            "observed": self.observed,
            "skip_reason": self.skip_reason.value,
            "app": self.context.app if self.context else "",
            # The TITLE is deliberately absent from the receipt when the
            # window was private: a receipt naming "Chase Bank — Private" has
            # leaked the thing the skip existed to protect.
            "title": (
                ""
                if (self.context is None or self.context.is_private)
                else self.context.title[:120]
            ),
            "characters": self.characters,
            "detail": self.detail[:200],
        }


class AmbientPresence:
    """The companion-mode organ: keep looking, stay quiet, never read private.

    One instance per runtime. It owns the bubble's state and the observation
    cadence; it does not own the UI, which reads ``state()``.
    """

    def __init__(self) -> None:
        self._lock = checked_lock("ambient_presence.state")
        self._mode = PresenceMode.WINDOW
        self._last_context: ScreenContext | None = None
        self._last_observed_at = 0.0
        self._pending_utterance: str = ""
        self._utterance_at = 0.0
        self._bubble_position: tuple[float, float] = self._load_bubble_position()
        self._ticks = 0
        self._observations = 0
        self._skips: dict[str, int] = {}
        self._private_skips = 0
        self._running = False
        self._last_spoke_at = 0.0
        #: A rectangle waiting for the bubble to collect and draw, and when
        #: that surface was last known to be alive. Both live here rather
        #: than in the route because "can anything actually draw right now"
        #: must be answerable BEFORE she claims to have pointed at something.
        self._pending_highlight: dict[str, float] | None = None
        self._last_surface_poll_at = 0.0
        #: She does not speak twice in quick succession. Unsolicited comment
        #: is a budget, not a feature, and the budget is deliberately small.
        self._min_speech_gap_s = 180.0

    # ── mode ─────────────────────────────────────────────────────────────

    def set_mode(self, mode: PresenceMode | str) -> PresenceMode:
        resolved = mode if isinstance(mode, PresenceMode) else PresenceMode(str(mode))
        with self._lock:
            self._mode = resolved
            if resolved is PresenceMode.HIDDEN:
                # Hidden means hidden: the queued thought goes too, rather
                # than waiting to appear the moment she is unhidden.
                self._pending_utterance = ""
        return resolved

    @property
    def mode(self) -> PresenceMode:
        with self._lock:
            return self._mode

    def hide(self) -> None:
        self.set_mode(PresenceMode.HIDDEN)

    def show(self) -> None:
        self.set_mode(PresenceMode.BUBBLE)

    # ── the bubble ───────────────────────────────────────────────────────

    def offer_utterance(self, text: str) -> bool:
        """Queue something for the bubble, if she is allowed to speak.

        Returns whether it was accepted. The gate is the same one that
        governs unprompted speech everywhere else — companion mode does not
        get its own, quieter authority.
        """
        body = str(text or "").strip()
        if not body:
            return False
        if self.mode is PresenceMode.HIDDEN:
            return False
        if _proactivity_suppressed():
            return False
        with self._lock:
            self._pending_utterance = body[:600]
            self._utterance_at = time.time()
        return True

    def clear_utterance(self) -> None:
        """The person dismissed it. It does not come back."""
        with self._lock:
            self._pending_utterance = ""

    def move_bubble(self, x: float, y: float) -> tuple[float, float]:
        """Remember where she was parked. In memory; the disk write is async.

        Kept synchronous and trivial because it is called from a request
        handler on the event loop. Durability is ``persist_bubble_position``,
        which the route awaits — an fsync on this loop once froze the live
        runtime for twenty minutes.
        """
        with self._lock:
            self._bubble_position = (float(x), float(y))
            return self._bubble_position

    async def persist_bubble_position(self) -> bool:
        """Write the parked position so it survives a restart.

        Position was held in memory only and the launcher never read it back,
        so "she can move it, position persists" was true of the drag and false
        of the persistence: every restart put her back in the bottom-left
        corner, including the restarts a person did not choose.
        """
        with self._lock:
            x, y = self._bubble_position
        try:
            from core.config import DATA_DIR
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            target = Path(DATA_DIR) / "companion" / "bubble_position.json"
            gateway = get_file_write_gateway()
            with local_internal_governed_scope(
                "ambient_presence.persist_bubble_position",
                domain="file_write",
            ):
                # Both calls are the async variants: this runs on the event
                # loop that serves every request, and a sync fsync here is the
                # exact shape of the write that froze the live runtime.
                await gateway.ensure_directory_async(
                    target.parent, source="ambient_presence.bubble_position"
                )
                await gateway.write_json_async(
                    target,
                    {"x": x, "y": y, "saved_at": time.time()},
                    schema_version=1,
                    schema_name=AMBIENT_SCHEMA,
                    source="ambient_presence.bubble_position",
                )
            return True
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "ambient_presence",
                exc,
                severity="debug",
                action=(
                    "bubble position not persisted; she reappears in the "
                    "default corner after a restart"
                ),
            )
            return False

    def _load_bubble_position(self) -> tuple[float, float]:
        """Where she was parked last time, or the origin meaning "unset".

        (0, 0) is the sentinel the launcher reads as "use the default corner",
        so a missing or corrupt file degrades to first-run behaviour rather
        than to a bubble stranded off-screen.
        """
        try:
            from core.config import DATA_DIR

            target = Path(DATA_DIR) / "companion" / "bubble_position.json"
            if not target.is_file():
                return (0.0, 0.0)
            stored = json.loads(target.read_text(encoding="utf-8"))
            # The gateway writes a schema envelope — {"payload": …, "schema":
            # …} — so the coordinates live one level down. Both shapes are
            # accepted because reading only the envelope would silently return
            # the "never parked" sentinel for any file written another way,
            # and a position loader that quietly reports "unset" is exactly
            # the failure this whole change exists to remove.
            payload = stored.get("payload") if isinstance(stored, dict) else None
            if not isinstance(payload, dict):
                payload = stored if isinstance(stored, dict) else {}
            return (float(payload.get("x", 0.0)), float(payload.get("y", 0.0)))
        except (ImportError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return (0.0, 0.0)

    # ── pointing at things ───────────────────────────────────────────────

    def note_surface_poll(self, surface: str = "") -> None:
        """The bubble just polled, so something exists that can draw.

        Only the bubble counts. The restrained chat window reads the same
        state endpoint, and treating its one read on open as "a drawing
        surface is attached" would let her claim to have pointed at something
        when no overlay host was listening.
        """
        if str(surface).strip().lower() != "bubble":
            return
        with self._lock:
            self._last_surface_poll_at = time.time()

    def drawing_surface_attached(self) -> bool:
        """Is there a live host that could put a rectangle on the screen?

        Asked BEFORE she says she pointed at something. Without this the
        overlay path returns "drawn" into a void — the honest answer when
        nothing is listening is that she could not point, and then she
        describes the location in words instead.
        """
        with self._lock:
            last = self._last_surface_poll_at
        return bool(last) and (time.time() - last) < _SURFACE_ALIVE_S

    def request_highlight(
        self, x: float, y: float, width: float, height: float, seconds: float
    ) -> bool:
        """Queue a rectangle for the bubble to draw. False if nothing can.

        False is not a failure to be retried — it is the answer. The caller
        turns it into "I could not point at it, here is where it is in
        words", which is a true sentence, where a claimed highlight nobody
        drew is not.
        """
        if not self.drawing_surface_attached():
            return False
        if self.mode is PresenceMode.HIDDEN:
            # Hidden means she is not on this person's screen at all, and an
            # overlay is the most present thing she can put there.
            return False
        with self._lock:
            self._pending_highlight = {
                "x": float(x),
                "y": float(y),
                "width": float(width),
                "height": float(height),
                "seconds": float(seconds),
                "queued_at": time.time(),
            }
        return True

    def take_highlight(self) -> dict[str, float] | None:
        """Collect the queued rectangle, exactly once.

        Popped rather than read so a rectangle is never drawn twice, and
        dropped when stale: the screen has moved on, and a box around where
        something used to be points at the wrong thing with full confidence.
        """
        with self._lock:
            pending = self._pending_highlight
            self._pending_highlight = None
        if pending is None:
            return None
        if time.time() - float(pending.get("queued_at", 0.0)) > _HIGHLIGHT_TTL_S:
            return None
        return pending

    # ── the loop ─────────────────────────────────────────────────────────

    async def tick(self) -> TickResult:
        """One ambient observation cycle. Cheap when nothing changed."""
        with self._lock:
            self._ticks += 1
            mode = self._mode

        if mode is PresenceMode.HIDDEN:
            return self._skip(SkipReason.HIDDEN)
        if _proactivity_suppressed():
            # Not looking is part of not intruding. Reading someone's screen
            # unprompted is not a lesser act than speaking to them unprompted.
            return self._skip(SkipReason.SUPPRESSED)

        context = await self._current_context()
        if context is None:
            return self._skip(SkipReason.CAPTURE_FAILED, detail="no frontmost window")

        if context.is_private:
            with self._lock:
                self._private_skips += 1
            # No capture is attempted. This is the entire point.
            return self._skip(SkipReason.PRIVATE_WINDOW, context=context)

        with self._lock:
            unchanged = (
                self._last_context is not None
                and self._last_context.key == context.key
                and (time.time() - self._last_observed_at) < _RESTALE_AFTER_S
            )
        if unchanged:
            return self._skip(SkipReason.UNCHANGED, context=context)

        return await self._observe(context)

    async def _observe(self, context: ScreenContext) -> TickResult:
        try:
            from core.governance_context import local_internal_governed_scope
            from core.perception.observation_evidence import (
                Observation,
                ObservationKind,
                remember_observation,
            )
        except ImportError as exc:
            return self._skip(SkipReason.NO_PERMISSION, context=context, detail=str(exc))

        try:
            with local_internal_governed_scope(
                "ambient_presence.observe", domain="environment_action"
            ):
                text = await self._read_screen_text()
        except (RuntimeError, OSError, TypeError, ValueError) as exc:
            record_degradation(
                "ambient_presence",
                exc,
                severity="warning",
                action="ambient observation skipped; she will have to look when asked",
            )
            return self._skip(SkipReason.CAPTURE_FAILED, context=context, detail=str(exc))

        if not str(text or "").strip():
            return self._skip(SkipReason.CAPTURE_FAILED, context=context)

        remember_observation(
            Observation(
                kind=ObservationKind.SCREEN_TEXT,
                capture=str(text),
                # Ambient, so there is no request. The empty request is
                # honest: nothing was asked, and a later question will
                # supply its own shape via recall_for().
                request="",
                source=context.app or "screen",
            )
        )
        with self._lock:
            self._last_context = context
            self._last_observed_at = time.time()
            self._observations += 1
        return TickResult(
            observed=True, context=context, characters=len(str(text))
        )

    async def _current_context(self) -> ScreenContext | None:
        try:
            from core.capabilities.host_automation import get_host_automation

            receipt = await get_host_automation().get_window_title()
        except (ImportError, AttributeError, RuntimeError, OSError, TypeError) as exc:
            record_degradation(
                "ambient_presence", exc, severity="debug",
                action="ambient tick could not read the frontmost window",
            )
            return None
        if not getattr(receipt, "success", False):
            return None
        raw = str(getattr(receipt, "result", "") or "")
        app, _, title = raw.partition("|")
        return ScreenContext(app=app.strip(), title=title.strip())

    async def _read_screen_text(self) -> str:
        from core.capabilities.host_automation import get_host_automation

        receipt = await get_host_automation().get_screen_text(retain_screenshot=False)
        if not getattr(receipt, "success", False):
            return ""
        return str(getattr(receipt, "result", "") or "")

    def _skip(
        self,
        reason: SkipReason,
        *,
        context: ScreenContext | None = None,
        detail: str = "",
    ) -> TickResult:
        with self._lock:
            self._skips[reason.value] = self._skips.get(reason.value, 0) + 1
        return TickResult(
            observed=False, skip_reason=reason, context=context, detail=detail
        )

    # ── the driver ───────────────────────────────────────────────────────

    async def run(self, *, interval_s: float = 6.0) -> None:
        """Drive ticks until stopped. Bounded, back-off on failure.

        The cadence is not the observation rate: most ticks are a window-title
        query that skips. What this bounds is how quickly she notices you
        moved to a different window, which is the latency the whole organ
        exists to remove.
        """
        import asyncio

        self._running = True
        consecutive_failures = 0
        while self._running:
            try:
                result = await asyncio.wait_for(self.tick(), timeout=20.0)
                consecutive_failures = 0
                if result.observed:
                    await self._consider_speaking(result)
            except asyncio.CancelledError:
                raise
            except (TimeoutError, RuntimeError, OSError, TypeError, ValueError) as exc:
                consecutive_failures += 1
                record_degradation(
                    "ambient_presence",
                    exc,
                    severity="warning" if consecutive_failures > 3 else "debug",
                    action=(
                        "ambient tick failed; she will fall back to looking "
                        "when asked"
                    ),
                )
            # Back off on repeated failure rather than hammering a broken
            # capture path — an accessibility permission that was revoked
            # would otherwise spin at the full cadence forever.
            delay = min(
                interval_s * (2 ** min(consecutive_failures, 4)), 300.0
            )
            await asyncio.sleep(max(1.0, delay))

    def stop(self) -> None:
        self._running = False

    async def _consider_speaking(self, result: TickResult) -> None:
        """Does she have something worth saying about what she just saw?

        The default is silence, and it is enforced in three places rather
        than one, because a companion that comments on everything is a
        companion you turn off:

          * the judgment itself must return something;
          * ``offer_utterance`` re-checks suppression and hidden state;
          * a message already waiting is not replaced, so a person who has
            not read the last one does not get a stream.
        """
        with self._lock:
            if self._pending_utterance:
                return
            if time.time() - self._last_spoke_at < self._min_speech_gap_s:
                return
        try:
            from core.perception.ambient_utterance import consider_utterance

            thought = await consider_utterance(result)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "ambient_presence", exc, severity="debug",
                action="ambient tick observed but did not consider speaking",
            )
            return
        if thought and self.offer_utterance(thought):
            with self._lock:
                self._last_spoke_at = time.time()

    # ── what the UI reads ────────────────────────────────────────────────

    def state(self, *, surface: str = "") -> dict[str, Any]:
        """What the surface should render.

        ``surface`` names who is asking. Only the bubble collects a queued
        highlight, because only the bubble has a host that can draw one — and
        a rectangle handed to the wrong reader is a rectangle nobody draws.
        """
        self.note_surface_poll(surface)
        highlight = (
            self.take_highlight()
            if str(surface).strip().lower() == "bubble"
            else None
        )
        with self._lock:
            return {
                "highlight": highlight,
                "schema": AMBIENT_SCHEMA,
                "mode": self._mode.value,
                "has_utterance": bool(self._pending_utterance),
                "utterance": self._pending_utterance,
                "utterance_age_s": (
                    round(time.time() - self._utterance_at, 1)
                    if self._pending_utterance
                    else None
                ),
                "bubble_position": list(self._bubble_position),
                "ticks": self._ticks,
                "observations": self._observations,
                "skips": dict(self._skips),
                # Surfaced, because a person should be able to see that the
                # privacy rule is doing something rather than trusting that
                # it is.
                "private_windows_skipped": self._private_skips,
                "last_app": self._last_context.app if self._last_context else "",
                "running": self._running,
                "seconds_since_spoke": (
                    round(time.time() - self._last_spoke_at, 1)
                    if self._last_spoke_at
                    else None
                ),
                "seconds_since_observation": (
                    round(time.time() - self._last_observed_at, 1)
                    if self._last_observed_at
                    else None
                ),
            }

    def fresh_observation_for(
        self, question: str, *, max_age_s: float = 45.0
    ) -> Any:
        """A recent-enough observation to answer from, or None.

        The whole reason the loop exists. A question about the screen should
        be answered from what she saw a moment ago; a capture that starts
        when the question arrives is the latency this removes.

        ``max_age_s`` is the honesty bound. Screens change, and answering
        "what's on my screen" from a two-minute-old reading is answering a
        question about NOW with a fact about THEN. Past the bound this
        returns None and the caller captures — which is the pre-ambient
        behaviour, so falling back is always safe.
        """
        try:
            from core.perception.observation_evidence import (
                ObservationKind,
                get_observation_memory,
            )

            memory = get_observation_memory()
            age = memory.age_of_latest(ObservationKind.SCREEN_TEXT)
            if age is None or age > max(0.0, float(max_age_s)):
                return None
            observation = memory.latest(ObservationKind.SCREEN_TEXT)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "ambient_presence", exc, severity="debug",
                action="ambient fast path unavailable; the caller captures fresh",
            )
            return None
        if observation is None or observation.is_empty:
            return None
        # The stored observation was taken with no request. Re-frame it for
        # THIS question so the shape of the answer follows the shape of the
        # ask — describe, locate, or quote — exactly as a fresh capture would.
        try:
            import copy

            reframed = copy.copy(observation)
            reframed.request = str(question or "")
            return reframed
        except (TypeError, ValueError):
            return observation

    def recall_for(self, question: str) -> str:
        """What she already saw that bears on this question.

        The latency payoff: the answer comes from ObservationMemory rather
        than from a capture that starts now. Empty when she has nothing —
        which the caller must treat as "look now", not as "nothing is there".
        """
        try:
            from core.perception.observation_evidence import get_observation_memory

            return get_observation_memory().recall_for(question)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "ambient_presence", exc, severity="debug",
                action="ambient recall unavailable; the caller must capture fresh",
            )
            return ""


def _proactivity_suppressed() -> bool:
    """Is she currently barred from acting unprompted?

    Fails CLOSED: if the check cannot run, she does not observe. An
    unavailable permission system is not permission.

    The failure is RECORDED rather than swallowed. This function used to
    import the gate from a module that does not define it, so it returned
    True on every call and the ambient loop skipped every tick for its whole
    life — silently, because a bare ``return True`` here is indistinguishable
    from a quiet window that is legitimately in force. Fail-closed is the
    right default and a silent fail-closed is still a dead subsystem, so the
    two cases are now told apart in the record.
    """
    try:
        from core.brain.initiative_engine import proactivity_suppressed_now

        return bool(proactivity_suppressed_now())
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "ambient_presence",
            exc,
            severity="warning",
            action=(
                "ambient observation suppressed because the proactivity gate "
                "could not be reached — this is a wiring fault, not a quiet "
                "window; she will look only when asked"
            ),
        )
        return True


_PRESENCE: AmbientPresence | None = None
_PRESENCE_LOCK = checked_lock("ambient_presence.singleton")


def get_ambient_presence() -> AmbientPresence:
    global _PRESENCE
    if _PRESENCE is None:
        with _PRESENCE_LOCK:
            if _PRESENCE is None:
                _PRESENCE = AmbientPresence()
    return _PRESENCE


def is_private_context(app: str, title: str) -> bool:
    """Public so any other surface can apply the same rule."""
    return ScreenContext(app=app, title=title).is_private


__all__ = [
    "AMBIENT_SCHEMA",
    "AmbientPresence",
    "PresenceMode",
    "ScreenContext",
    "SkipReason",
    "TickResult",
    "get_ambient_presence",
    "is_private_context",
]
