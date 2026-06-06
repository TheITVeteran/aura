"""core/voice/wake_word.py — Always-Listening Wake Word Detection
=================================================================
Runs as a background thread, minimal CPU. On detection of "Hey Aura",
raises foreground priority, starts a command session, and submits a
high-priority impulse to InitiativeSynthesizer.

Uses the existing Whisper transcript stream from the audio service.
Pattern-matches for wake phrases in transcript chunks.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from enum import Enum
from typing import Any, Dict, Optional

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.runtime.task_ownership import create_tracked_task

logger = logging.getLogger("Aura.WakeWord")


class WakeState(str, Enum):
    IDLE = "idle"               # Passive listening for wake word
    LISTENING = "listening"     # Wake word detected, accumulating command
    PROCESSING = "processing"   # Command received, decomposing into task
    EXECUTING = "executing"     # Task graph running
    REPORTING = "reporting"     # Summarizing results


# Wake phrases (case-insensitive)
WAKE_PHRASES = [
    r"\bhey\s+aura\b",
    r"\bhi\s+aura\b",
    r"\bokay?\s+aura\b",
    r"\baura\b.*\blisten\b",
]
WAKE_PATTERN = re.compile("|".join(WAKE_PHRASES), re.IGNORECASE)

# Interrupt phrases
INTERRUPT_PHRASES = [
    r"\bstop\b",
    r"\bcancel\b",
    r"\bpause\b",
    r"\bwait\b",
    r"\babort\b",
    r"\bnever\s*mind\b",
]
INTERRUPT_PATTERN = re.compile("|".join(INTERRUPT_PHRASES), re.IGNORECASE)


class WakeWordDetector:
    """Always-listening wake word detection.

    Lifecycle:
        IDLE → (wake word) → LISTENING → (VAD silence) → PROCESSING
        → EXECUTING → REPORTING → IDLE

    Barge-in: user can say "stop"/"cancel" at any point to interrupt.
    """

    SILENCE_TIMEOUT_S = 1.5      # silence after this = end of command
    SESSION_TIMEOUT_S = 30.0     # max session length
    POLL_INTERVAL_S = 0.2        # check transcript this often
    MAX_MISSION_STEPS = 200

    def __init__(self) -> None:
        self.state = WakeState.IDLE
        self._task: Optional[asyncio.Task] = None
        self._session_start: float = 0.0
        self._last_speech: float = 0.0
        self._accumulated_transcript: str = ""
        self._last_processed_transcript: str = ""
        self._wake_count: int = 0
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        ServiceContainer.register_instance("wake_word", self, required=False)
        self._started = True
        self._task = create_tracked_task(
            self._detection_loop(),
            name="Aura.WakeWordDetector",
        )
        logger.info("WakeWordDetector ONLINE — listening for 'Hey Aura'")

    async def stop(self) -> None:
        self._started = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("WakeWordDetector OFFLINE (detected %d wake events)", self._wake_count)

    async def _detection_loop(self) -> None:
        """Main detection loop — reads from audio service transcript."""
        try:
            while self._started:
                try:
                    transcript = self._get_latest_transcript()

                    if self.state == WakeState.IDLE:
                        await self._check_wake_word(transcript)

                    elif self.state == WakeState.LISTENING:
                        await self._accumulate_command(transcript)

                    elif self.state in (WakeState.EXECUTING, WakeState.PROCESSING):
                        # Check for interrupts
                        if transcript and INTERRUPT_PATTERN.search(transcript):
                            await self._handle_interrupt()

                except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                    record_degradation("wake_word.loop", e)

                await asyncio.sleep(self.POLL_INTERVAL_S)

        except asyncio.CancelledError:
            raise

    def _get_latest_transcript(self) -> str:
        """Read the latest transcript from the audio service or WorldState."""
        try:
            ws = ServiceContainer.get("world_state", default=None)
            if ws and hasattr(ws, "last_voice_transcript"):
                transcript = ws.last_voice_transcript or ""
                if transcript != self._last_processed_transcript:
                    self._last_processed_transcript = transcript
                    return transcript
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("wake_word.world_state_transcript", exc)

        # Try audio service directly
        try:
            from pathlib import Path
            import json
            audio_path = Path(__file__).resolve().parent.parent.parent / "sensory_audio.json"
            if audio_path.exists() and (time.time() - audio_path.stat().st_mtime) < 10:
                data = json.loads(audio_path.read_text(encoding="utf-8"))
                transcript = str(data.get("transcript") or data.get("text") or "")
                if transcript != self._last_processed_transcript:
                    self._last_processed_transcript = transcript
                    return transcript
        except (OSError, ValueError, TypeError) as exc:
            record_degradation("wake_word.audio_transcript_file", exc)

        return ""

    async def _check_wake_word(self, transcript: str) -> None:
        """Check for wake word in transcript."""
        if not transcript:
            return

        if WAKE_PATTERN.search(transcript):
            self._wake_count += 1
            self.state = WakeState.LISTENING
            self._session_start = time.time()
            self._last_speech = time.time()
            self._accumulated_transcript = ""

            logger.info("🎤 Wake word detected! Starting command session #%d", self._wake_count)

            # Emit salient event
            try:
                ws = ServiceContainer.get("world_state", default=None)
                if ws:
                    ws.record_event(
                        "Wake word detected — command session started",
                        source="voice",
                        salience=0.9,
                        ttl=60,
                    )
            except (ImportError, AttributeError, RuntimeError) as exc:
                record_degradation("wake_word.world_state_event", exc)

    async def _accumulate_command(self, transcript: str) -> None:
        """Accumulate spoken command after wake word."""
        now = time.time()

        if transcript:
            # Remove wake phrase from beginning
            command = WAKE_PATTERN.sub("", transcript).strip()
            if command:
                self._accumulated_transcript = command
                self._last_speech = now

        # Check for end of command (silence timeout)
        silence_duration = now - self._last_speech
        session_duration = now - self._session_start

        if silence_duration > self.SILENCE_TIMEOUT_S and self._accumulated_transcript:
            # End of command — process it
            await self._process_command(self._accumulated_transcript)

        elif session_duration > self.SESSION_TIMEOUT_S:
            # Session timeout
            if self._accumulated_transcript:
                await self._process_command(self._accumulated_transcript)
            else:
                logger.info("Wake session timed out without command")
                self.state = WakeState.IDLE

    async def _process_command(self, command: str) -> None:
        """Process a spoken command into a mission."""
        self.state = WakeState.PROCESSING
        logger.info("🎤 Voice command received: '%s'", command[:100])

        try:
            # Submit as high-priority initiative impulse
            synthesizer = ServiceContainer.get("initiative_synthesizer", default=None)
            if synthesizer and hasattr(synthesizer, "submit_impulse"):
                synthesizer.submit_impulse({
                    "source": "voice_command",
                    "content": command,
                    "priority": 0.95,  # Voice commands are highest priority
                    "salience": 1.0,
                    "metadata": {
                        "session_id": self._wake_count,
                        "session_duration": time.time() - self._session_start,
                    },
                })

            # Also create a mission directly
            mission_state = ServiceContainer.get("mission_state", default=None)
            if mission_state:
                self.state = WakeState.EXECUTING
                mission = await mission_state.create_mission(
                    command, source="voice", priority=0.9,
                )
                # Execute the mission with a hard step cap so voice cannot loop forever.
                step_count = 0
                while mission.graph and not mission.graph.is_complete:
                    if step_count >= self.MAX_MISSION_STEPS:
                        raise RuntimeError("voice mission exceeded step cap")
                    step_count += 1
                    node = await mission_state.advance_mission(mission.mission_id)
                    if node is None:
                        break

                self.state = WakeState.REPORTING
                # Narrate result
                if mission.graph and mission.graph.is_successful:
                    logger.info("🎤 Voice mission completed successfully")
                else:
                    failure = mission.graph.get_failure_summary() if mission.graph else "Unknown"
                    logger.info("🎤 Voice mission had issues: %s", failure[:100])
            else:
                logger.info("MissionState not available for voice command processing")

        except (ImportError, AttributeError, RuntimeError, TypeError) as e:
            record_degradation("wake_word.process", e)
            logger.warning("Voice command processing failed: %s", e)

        # Return to idle
        self._accumulated_transcript = ""
        self.state = WakeState.IDLE

    async def _handle_interrupt(self) -> None:
        """Handle a spoken interrupt ("stop", "cancel", etc.)."""
        logger.info("🎤 Voice interrupt received — cancelling current action")
        self.state = WakeState.IDLE
        self._accumulated_transcript = ""

        # Try to cancel current mission
        try:
            mission_state = ServiceContainer.get("mission_state", default=None)
            if mission_state:
                active = mission_state.list_active_missions()
                for m in active:
                    if m.source == "voice":
                        from core.planning.mission_state import MissionStatus
                        mission_state.update_mission_status(m.mission_id, MissionStatus.CANCELLED)
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("wake_word.interrupt_cancel", exc)

    def get_status(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "wake_count": self._wake_count,
            "session_active": self.state != WakeState.IDLE,
            "accumulated": self._accumulated_transcript[:60] if self._accumulated_transcript else "",
        }


_instance: Optional[WakeWordDetector] = None


def get_wake_word_detector() -> WakeWordDetector:
    global _instance
    if _instance is None:
        _instance = WakeWordDetector()
    return _instance


__all__ = ["WakeWordDetector", "WakeState", "get_wake_word_detector"]
