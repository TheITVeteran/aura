"""core/perception/perceptual_pump.py — Continuous Perceptual Ingestion Loop
===========================================================================
The PRIMARY driver of Aura's substrate ODE and affect system.

This is NOT context for the LLM. This is the causal grounding loop.
Real-time perceptual input (screen, mic, system telemetry) perturbs the
continuous state vector at 10 Hz BEFORE the LLM ever runs.

The pump collects a PerceptualFrame every ~100ms from all modalities,
maps it into RuntimeBody fields, feeds it through the PhenomenalEngine
to produce an ExperienceState, and injects the result into the
ContinuousSubstrate ODE.

Architecture:
  sensors → PerceptualFrame → RuntimeBody → PhenomenalEngine.step()
           → ExperienceState → ContinuousSubstrate.inject_perceptual_frame()
           → WorldState.update_from_perceptual_frame()

Everything downstream (LLM prompts, initiative scoring, speech prosody)
reads from the substrate state summary — never from raw sensors.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.runtime.task_ownership import create_tracked_task

if TYPE_CHECKING:
    from core.phenomenal_substrate.types import RuntimeBody

logger = logging.getLogger("Aura.PerceptualPump")

_PUMP_RUNTIME_ERRORS = (
    ImportError,
    OSError,
    RuntimeError,
    AttributeError,
    TypeError,
    ValueError,
    asyncio.TimeoutError,
    # A hung osascript raises TimeoutExpired (a SubprocessError, not an
    # OSError) — it escaped every catch list and killed the whole pump
    # task during the 110GB incident. No sensor failure may end perception.
    subprocess.SubprocessError,
)

# After an AppleScript probe times out, skip subprocess-based screen
# probes for this long. A hung System Events under host load otherwise
# costs 2s of a worker thread every 500ms, forever.
_SCREEN_PROBE_BACKOFF_S = 10.0
_LAST_SCREEN_PROBE_TIMEOUT_AT: float = 0.0

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScreenState:
    """What is happening on the screen right now."""
    active_app: str = ""
    window_title: str = ""
    content_hash: str = ""          # hash of OCR text (change detection)
    content_snippet: str = ""       # first 200 chars of OCR text
    screen_changed: bool = False    # did the screen change since last frame?
    change_magnitude: float = 0.0   # 0-1, how much changed
    timestamp: float = field(default_factory=time.time)


@dataclass
class AudioState:
    """What is happening on the mic right now."""
    rms_energy: float = 0.0         # 0-1, ambient sound level
    voice_activity: bool = False    # is someone speaking?
    transcript_snippet: str = ""    # most recent speech (last ~5s, display-bounded)
    transcript_full: str = ""       # full utterance (command-fidelity, 4000-char bound)
    transcript_changed: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class SystemState:
    """System telemetry snapshot."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    thermal_pressure: float = 0.0   # 0-1
    battery_percent: float = 100.0
    battery_charging: bool = True
    disk_io_pressure: float = 0.0   # 0-1
    timestamp: float = field(default_factory=time.time)


@dataclass
class UserState:
    """Inferred user state."""
    idle_seconds: float = 0.0
    last_interaction_type: str = ""  # "text", "voice", "mouse", "none"
    message_count: int = 0
    estimated_mood: str = "unknown"
    presence: float = 0.5           # 0 = absent, 1 = actively engaged
    timestamp: float = field(default_factory=time.time)


@dataclass
class PerceptualFrame:
    """A single 100ms snapshot of all perceptual modalities.

    This is the fundamental unit of grounded perception.
    """
    frame_id: int = 0
    screen: ScreenState = field(default_factory=ScreenState)
    audio: AudioState = field(default_factory=AudioState)
    system: SystemState = field(default_factory=SystemState)
    user: UserState = field(default_factory=UserState)
    timestamp: float = field(default_factory=time.time)

    def novelty_score(self) -> float:
        """How novel is this frame compared to baseline?"""
        screen_novelty = self.screen.change_magnitude
        audio_novelty = 0.3 if self.audio.voice_activity else 0.0
        system_novelty = max(0.0, (self.system.cpu_percent - 50) / 50)
        return min(1.0, 0.5 * screen_novelty + 0.3 * audio_novelty + 0.2 * system_novelty)

    def threat_score(self) -> float:
        """Perceived threat level from this frame."""
        thermal = self.system.thermal_pressure
        memory = max(0.0, (self.system.memory_percent - 80) / 20)
        battery_low = max(0.0, (20 - self.system.battery_percent) / 20) if not self.system.battery_charging else 0.0
        return min(1.0, 0.4 * thermal + 0.3 * memory + 0.3 * battery_low)

    def social_signal(self) -> float:
        """Social engagement signal from this frame."""
        voice = 0.6 if self.audio.voice_activity else 0.0
        presence = self.user.presence
        recency = max(0.0, 1.0 - self.user.idle_seconds / 300.0)  # decays over 5 min
        return min(1.0, 0.4 * voice + 0.3 * presence + 0.3 * recency)


# ---------------------------------------------------------------------------
# Sensor collectors — each one reads from available hardware/services
# ---------------------------------------------------------------------------

def _collect_screen_state(prev_hash: str) -> ScreenState:
    """Collect screen state from available sources.

    Tries (in order): ScreenObserver JSON, AppleScript, fallback.
    """
    state = ScreenState()
    now = time.time()
    state.timestamp = now

    # 1. Get active app and window title via AppleScript (fast, no imports)
    global _LAST_SCREEN_PROBE_TIMEOUT_AT
    probe_allowed = (now - _LAST_SCREEN_PROBE_TIMEOUT_AT) >= _SCREEN_PROBE_BACKOFF_S
    try:
        if probe_allowed:
            from core.runtime.subprocess_gateway import get_subprocess_gateway
            gw = get_subprocess_gateway()

            # Active app name
            result = gw.run(
                ["osascript", "-e",
                 'tell application "System Events" to get name of first application process whose frontmost is true'],
                capture_output=True, timeout=2.0, read_only=True, source="perceptual_pump.screen.app",
            )
            if result.returncode == 0 and result.stdout:
                state.active_app = result.stdout.strip()

            # Window title
            if state.active_app:
                result = gw.run(
                    ["osascript", "-e",
                     f'tell application "System Events" to get name of front window of process "{state.active_app}"'],
                    capture_output=True, timeout=2.0, read_only=True, source="perceptual_pump.screen.title",
                )
                if result.returncode == 0 and result.stdout:
                    state.window_title = result.stdout.strip()[:200]
    except subprocess.SubprocessError as e:
        _LAST_SCREEN_PROBE_TIMEOUT_AT = time.time()
        record_degradation("perceptual_pump.screen", e)
        logger.debug(
            "Screen probe timed out; backing off AppleScript for %.0fs: %s",
            _SCREEN_PROBE_BACKOFF_S,
            e,
        )
    except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as e:
        record_degradation("perceptual_pump.screen", e)
        logger.debug("Screen state AppleScript probe failed: %s", e)

    # 2. Screen content from ScreenObserver JSON (if vision service is running)
    try:
        from pathlib import Path
        import json
        vision_path = Path(__file__).resolve().parent.parent.parent / "sensory_vision.json"
        if vision_path.exists() and (now - vision_path.stat().st_mtime) < 15:
            data = json.loads(vision_path.read_text(encoding="utf-8"))
            text = str(data.get("text") or data.get("ocr_text") or "")
            if text:
                state.content_snippet = text[:200]
                state.content_hash = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    except (OSError, ValueError, TypeError) as e:
        record_degradation("perceptual_pump.screen_ocr", e)

    # 3. Change detection
    if prev_hash and state.content_hash:
        state.screen_changed = state.content_hash != prev_hash
        state.change_magnitude = 1.0 if state.screen_changed else 0.0
    elif state.active_app:
        # Even without OCR, app switch counts as change
        state.screen_changed = True
        state.change_magnitude = 0.3

    return state


def _collect_audio_state() -> AudioState:
    """Collect audio state from available sources."""
    state = AudioState()
    state.timestamp = time.time()

    # Read from audio service JSON (if running)
    try:
        from pathlib import Path
        import json
        audio_path = Path(__file__).resolve().parent.parent.parent / "sensory_audio.json"
        if audio_path.exists() and (time.time() - audio_path.stat().st_mtime) < 10:
            data = json.loads(audio_path.read_text(encoding="utf-8"))
            state.rms_energy = min(1.0, max(0.0, float(data.get("rms", data.get("energy", 0.0)) or 0.0)))
            state.voice_activity = bool(data.get("vad") or data.get("speech_detected") or state.rms_energy > 0.15)
            transcript = str(data.get("transcript") or data.get("text") or "")
            if transcript.strip():
                state.transcript_snippet = transcript.strip()[-200:]
                # Command fidelity: the wake-word lane consumes the full
                # utterance; the 200-char display snippet destroyed long
                # spoken commands when it overwrote last_voice_transcript.
                state.transcript_full = transcript.strip()[:4000]
                state.transcript_changed = True
    except (OSError, ValueError, TypeError) as e:
        record_degradation("perceptual_pump.audio", e)

    return state


def _collect_system_state() -> SystemState:
    """Collect system telemetry."""
    state = SystemState()
    state.timestamp = time.time()

    try:
        import psutil
        state.cpu_percent = psutil.cpu_percent(interval=0)
        mem = psutil.virtual_memory()
        state.memory_percent = mem.percent
        battery = psutil.sensors_battery()
        if battery:
            state.battery_percent = battery.percent
            state.battery_charging = battery.power_plugged or False

        # Thermal
        try:
            temps = psutil.sensors_temperatures()
        except (AttributeError, OSError, RuntimeError, ValueError):
            temps = None
        if temps:
            max_temp = max(
                (t.current for sensors in temps.values() for t in sensors),
                default=0.0,
            )
            state.thermal_pressure = min(1.0, max(0.0, (max_temp - 60) / 40))
        else:
            # macOS fallback via SubstrateMonitor
            try:
                from core.resilience.substrate_monitor import SubstrateMonitor
                _level, pressure, _source = SubstrateMonitor().thermal()
                state.thermal_pressure = pressure
            except (ImportError, OSError, RuntimeError, TypeError, ValueError):
                state.thermal_pressure = 0.0

        # Disk I/O
        try:
            disk = psutil.disk_io_counters()
            if disk:
                # Rough heuristic: busy_time / elapsed. Not perfect but directional.
                state.disk_io_pressure = min(1.0, max(0.0,
                    getattr(disk, 'busy_time', 0) / max(1, state.timestamp) / 1000.0))
        except (AttributeError, RuntimeError):
            pass

    except ImportError:
        pass  # psutil not installed — degrade gracefully
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as e:
        record_degradation("perceptual_pump.system", e)

    return state


def _collect_user_state() -> UserState:
    """Collect inferred user state from WorldState."""
    state = UserState()
    state.timestamp = time.time()

    try:
        ws = ServiceContainer.get("world_state", default=None)
        if ws:
            state.idle_seconds = ws.user_idle_seconds
            state.message_count = ws.user_message_count
            state.estimated_mood = ws.estimated_user_mood

            # Presence: decays over 5 minutes of idle
            if state.idle_seconds < 30:
                state.presence = 1.0
            elif state.idle_seconds < 300:
                state.presence = max(0.1, 1.0 - state.idle_seconds / 300.0)
            else:
                state.presence = 0.1

            # Interaction type heuristic
            if state.idle_seconds < 5:
                state.last_interaction_type = "text"
            elif state.idle_seconds < 60:
                state.last_interaction_type = "recent"
            else:
                state.last_interaction_type = "none"
    except (ImportError, AttributeError, RuntimeError) as e:
        record_degradation("perceptual_pump.user", e)

    return state


# ---------------------------------------------------------------------------
# RuntimeBody mapping
# ---------------------------------------------------------------------------

def frame_to_runtime_body(frame: PerceptualFrame) -> "RuntimeBody":
    """Map a PerceptualFrame into RuntimeBody fields for the phenomenal engine.

    This is where physical-world perception becomes interoceptive state.
    """
    from core.phenomenal_substrate.types import RuntimeBody

    # Energy: inversely proportional to system load
    energy = max(0.1, 1.0 - 0.4 * (frame.system.cpu_percent / 100.0)
                 - 0.3 * (frame.system.memory_percent / 100.0)
                 - 0.3 * frame.system.thermal_pressure)

    # Continuity: stable when screen hasn't changed dramatically
    continuity = max(0.2, 1.0 - 0.6 * frame.screen.change_magnitude)

    # Agency: high when system is responsive and user is present
    agency = max(0.2, 0.4 + 0.3 * (1.0 - frame.system.cpu_percent / 100.0)
                 + 0.3 * frame.user.presence)

    # Safety: inversely proportional to threat score
    safety = max(0.1, 1.0 - frame.threat_score())

    # Social contact: driven by voice activity and user presence
    social = frame.social_signal()

    # Novelty: from screen changes and new audio
    novelty = frame.novelty_score()

    # Uncertainty: high when screen content is unfamiliar or changing rapidly
    uncertainty = min(0.9, 0.2 + 0.5 * frame.screen.change_magnitude
                      + 0.3 * (1.0 - frame.user.presence))

    # Compute pressure: direct from CPU
    compute_pressure = frame.system.cpu_percent / 100.0

    # Memory pressure: direct from RAM
    memory_pressure = frame.system.memory_percent / 100.0

    # Error pressure: thermal + low battery
    error_pressure = max(0.0,
        0.5 * frame.system.thermal_pressure
        + 0.5 * max(0.0, (20 - frame.system.battery_percent) / 20)
            if not frame.system.battery_charging else 0.0)

    return RuntimeBody(
        energy=min(1.0, max(0.0, energy)),
        continuity=min(1.0, max(0.0, continuity)),
        agency=min(1.0, max(0.0, agency)),
        safety=min(1.0, max(0.0, safety)),
        social_contact=min(1.0, max(0.0, social)),
        novelty=min(1.0, max(0.0, novelty)),
        uncertainty=min(1.0, max(0.0, uncertainty)),
        compute_pressure=min(1.0, max(0.0, compute_pressure)),
        memory_pressure=min(1.0, max(0.0, memory_pressure)),
        error_pressure=min(1.0, max(0.0, error_pressure)),
        screen_novelty=frame.screen.change_magnitude,
        audio_energy=frame.audio.rms_energy,
        voice_present=frame.audio.voice_activity,
        foreground_app_familiar=0.8 if frame.screen.active_app else 0.2,
    )


# ---------------------------------------------------------------------------
# The Pump
# ---------------------------------------------------------------------------

class PerceptualPump:
    """Continuous 10 Hz perceptual loop — the PRIMARY substrate driver.

    This replaces text-only context as the dominant input to Aura's
    inner state. Real-world changes perturb her continuous state vector
    at 10 Hz BEFORE the LLM sees anything.

    Architecture:
        sensors → PerceptualFrame → RuntimeBody
        → PhenomenalEngine.step() → ExperienceState
        → ContinuousSubstrate.inject_perceptual_frame()
        → WorldState.update_from_perceptual_frame()
    """

    PUMP_HZ = 10.0
    PUMP_DT = 1.0 / PUMP_HZ

    # Not every collector runs every tick (to save CPU)
    SCREEN_EVERY_N = 5      # every 500ms (AppleScript is slow)
    AUDIO_EVERY_N = 2       # every 200ms
    SYSTEM_EVERY_N = 10     # every 1s
    USER_EVERY_N = 5        # every 500ms

    # Substrate injection rate (can be lower than perception rate)
    SUBSTRATE_INJECT_EVERY_N = 2  # every 200ms

    def __init__(self) -> None:
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self._frame_count: int = 0
        self._last_screen_hash: str = ""
        self._latest_frame: Optional[PerceptualFrame] = None
        self._frame_history: Deque[PerceptualFrame] = deque(maxlen=100)

        # Cached sub-states (updated at different rates)
        self._screen: ScreenState = ScreenState()
        self._audio: AudioState = AudioState()
        self._system: SystemState = SystemState()
        self._user: UserState = UserState()

        # Stats
        self._frames_produced: int = 0
        self._substrate_injections: int = 0
        self._errors: int = 0
        self._started_at: float = 0.0

    async def start(self) -> None:
        """Start the perceptual pump."""
        if self.running:
            return
        self.running = True
        self._started_at = time.time()
        ServiceContainer.register_instance("perceptual_pump", self, required=False)
        self._task = create_tracked_task(
            self._pump_loop(),
            name="Aura.PerceptualPump",
        )
        logger.info(
            "👁️ PerceptualPump ONLINE — %.0f Hz perceptual grounding active "
            "(screen every %d ticks, audio every %d ticks, system every %d ticks)",
            self.PUMP_HZ, self.SCREEN_EVERY_N, self.AUDIO_EVERY_N, self.SYSTEM_EVERY_N,
        )

    async def stop(self) -> None:
        """Stop the perceptual pump."""
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(
            "👁️ PerceptualPump OFFLINE — produced %d frames, %d substrate injections, %d errors",
            self._frames_produced, self._substrate_injections, self._errors,
        )

    @property
    def latest_frame(self) -> Optional[PerceptualFrame]:
        """The most recent perceptual frame."""
        return self._latest_frame

    def get_status(self) -> Dict[str, Any]:
        """Pump status for dashboards."""
        return {
            "running": self.running,
            "frames_produced": self._frames_produced,
            "substrate_injections": self._substrate_injections,
            "errors": self._errors,
            "uptime_s": round(time.time() - self._started_at, 1) if self._started_at else 0,
            "latest_frame": {
                "active_app": self._screen.active_app,
                "window_title": self._screen.window_title[:60],
                "voice_activity": self._audio.voice_activity,
                "cpu": round(self._system.cpu_percent, 1),
                "user_presence": round(self._user.presence, 2),
            } if self._latest_frame else None,
            "pump_hz": self.PUMP_HZ,
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _pump_loop(self) -> None:
        """The main 10 Hz perceptual ingestion loop."""
        try:
            while self.running:
                tick_start = time.time()
                try:
                    await self._tick()
                except asyncio.CancelledError:
                    raise
                except _PUMP_RUNTIME_ERRORS as e:
                    self._errors += 1
                    record_degradation("perceptual_pump", e)
                    if self._errors % 50 == 1:
                        logger.warning("PerceptualPump tick error #%d: %s", self._errors, e)

                # Maintain target rate
                elapsed = time.time() - tick_start
                sleep_time = max(0.01, self.PUMP_DT - elapsed)
                await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            raise
        except _PUMP_RUNTIME_ERRORS as e:
            record_degradation("perceptual_pump", e)
            logger.error("PerceptualPump loop crashed: %s", e)
            self.running = False

    def _cognitive_load_throttle_active(self) -> bool:
        """Perception yields to cognition.

        While foreground inference owns the machine or memory is tight,
        the subprocess-spawning sensors (AppleScript probes) slow down to
        a crawl instead of competing for a starved host. Attention
        narrows under load; it does not flail.
        """
        try:
            gate = ServiceContainer.get("inference_gate", default=None)
            if gate and hasattr(gate, "get_conversation_status"):
                lane = gate.get_conversation_status() or {}
                if lane.get("foreground_owned") or int(lane.get("active_generations", 0) or 0) > 0:
                    return True
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
            pass
        return float(getattr(self._system, "memory_percent", 0.0) or 0.0) >= 85.0

    # While throttled, screen/user sensors run 10x slower (every 5s).
    THROTTLED_MULTIPLIER = 10

    async def _tick(self) -> None:
        """One pump tick: collect sensors, build frame, inject into substrate."""
        self._frame_count += 1
        n = self._frame_count

        throttled = self._cognitive_load_throttle_active()
        screen_every = self.SCREEN_EVERY_N * (self.THROTTLED_MULTIPLIER if throttled else 1)
        user_every = self.USER_EVERY_N * (self.THROTTLED_MULTIPLIER if throttled else 1)

        # Collect from each modality at its configured rate
        # These run in a thread to avoid blocking the event loop
        if n % screen_every == 0:
            self._screen = await asyncio.to_thread(
                _collect_screen_state, self._last_screen_hash
            )
            self._last_screen_hash = self._screen.content_hash or self._last_screen_hash

        if n % self.AUDIO_EVERY_N == 0:
            self._audio = await asyncio.to_thread(_collect_audio_state)

        if n % self.SYSTEM_EVERY_N == 0:
            self._system = await asyncio.to_thread(_collect_system_state)

        if n % user_every == 0:
            self._user = await asyncio.to_thread(_collect_user_state)

        # Build the frame
        frame = PerceptualFrame(
            frame_id=self._frames_produced,
            screen=self._screen,
            audio=self._audio,
            system=self._system,
            user=self._user,
            timestamp=time.time(),
        )
        self._latest_frame = frame
        self._frame_history.append(frame)
        self._frames_produced += 1

        # Inject into substrate at its configured rate
        if n % self.SUBSTRATE_INJECT_EVERY_N == 0:
            await self._inject_into_substrate(frame)

        # Update WorldState with perceptual data (every tick)
        self._update_world_state(frame)

    async def _inject_into_substrate(self, frame: PerceptualFrame) -> None:
        """Map frame → RuntimeBody → PhenomenalEngine → Substrate.

        This is the causal grounding path. The substrate ODE state is
        directly perturbed by physical-world perception.
        """
        try:
            body = frame_to_runtime_body(frame)

            # 1. Feed through PhenomenalEngine for affect computation
            engine = ServiceContainer.get("phenomenal_engine", default=None)
            if engine and hasattr(engine, "step"):
                from core.phenomenal_substrate.types import Event
                event = Event(
                    label=f"perceptual_frame_{frame.frame_id}",
                    source="perceptual_pump",
                    novelty=frame.novelty_score(),
                    threat=frame.threat_score(),
                    affiliation=frame.social_signal(),
                    control_gain=max(0.0, frame.user.presence - 0.5) * 0.3,
                    goal_delta=0.0,
                )
                experience = engine.step(body, event, recurrent_cycles=2)

                # 2. Inject the experience-derived affect into the substrate
                substrate = ServiceContainer.get("conscious_substrate", default=None)
                if substrate and hasattr(substrate, "inject_perceptual_frame"):
                    substrate.inject_perceptual_frame({
                        "source": "perceptual_pump",
                        "valence": experience.valence,
                        "arousal": experience.arousal,
                        "novelty": frame.novelty_score(),
                        "threat": frame.threat_score(),
                        "social": frame.social_signal(),
                        "cpu_percent": frame.system.cpu_percent,
                        "memory_percent": frame.system.memory_percent,
                        "thermal": frame.system.thermal_pressure,
                        "user_presence": frame.user.presence,
                        "screen_changed": frame.screen.screen_changed,
                        "voice_activity": frame.audio.voice_activity,
                        "active_app": frame.screen.active_app,
                        "timestamp": frame.timestamp,
                    })
                    # Adapt projections so readouts track real affect
                    if hasattr(substrate, "adapt_projections"):
                        substrate.adapt_projections({
                            "valence": experience.valence,
                            "arousal": experience.arousal,
                            "curiosity": experience.curiosity,
                        }, lr=0.005)
                    self._substrate_injections += 1
                elif substrate and hasattr(substrate, "inject_observation"):
                    # Fallback to raw observation injection
                    substrate.inject_observation({
                        "source": "perceptual_pump",
                        "confidence": 0.9,
                        "energy": frame.audio.rms_energy,
                        "summary": f"app={frame.screen.active_app} voice={frame.audio.voice_activity}",
                        "timestamp_unix": frame.timestamp,
                    })
                    self._substrate_injections += 1

            else:
                # No phenomenal engine — inject directly into substrate
                substrate = ServiceContainer.get("conscious_substrate", default=None)
                if substrate and hasattr(substrate, "inject_observation"):
                    substrate.inject_observation({
                        "source": "perceptual_pump",
                        "confidence": 0.85,
                        "energy": frame.audio.rms_energy,
                        "summary": f"screen={frame.screen.active_app} cpu={frame.system.cpu_percent:.0f}%",
                        "timestamp_unix": frame.timestamp,
                    })
                    self._substrate_injections += 1

        except (ImportError, AttributeError, RuntimeError) as e:
            self._errors += 1
            record_degradation("perceptual_pump.substrate", e)
            if self._errors % 100 == 1:
                logger.debug("Substrate injection failed: %s", e)

    def _update_world_state(self, frame: PerceptualFrame) -> None:
        """Push perceptual data into WorldState for downstream consumers."""
        try:
            ws = ServiceContainer.get("world_state", default=None)
            if not ws:
                return

            # Update direct fields
            if hasattr(ws, "active_foreground_app"):
                ws.active_foreground_app = frame.screen.active_app
            if hasattr(ws, "active_window_title"):
                ws.active_window_title = frame.screen.window_title
            if hasattr(ws, "screen_content_hash"):
                ws.screen_content_hash = frame.screen.content_hash
            if hasattr(ws, "ambient_audio_level"):
                ws.ambient_audio_level = frame.audio.rms_energy
            if hasattr(ws, "voice_activity_detected"):
                ws.voice_activity_detected = frame.audio.voice_activity
            if hasattr(ws, "last_voice_transcript") and frame.audio.transcript_snippet:
                # Full fidelity here: the wake-word command lane reads this
                # field, and the 200-char display snippet used to truncate
                # long spoken commands mid-utterance.
                ws.last_voice_transcript = (
                    frame.audio.transcript_full or frame.audio.transcript_snippet
                )

            # System telemetry — keep WorldState's own fields in sync
            ws.cpu_percent = frame.system.cpu_percent
            ws.memory_percent = frame.system.memory_percent
            ws.thermal_pressure = frame.system.thermal_pressure
            ws.battery_percent = frame.system.battery_percent
            ws.battery_charging = frame.system.battery_charging

            # Generate salient events for significant changes
            if frame.screen.screen_changed and frame.screen.active_app:
                ws.record_event(
                    f"App switched to {frame.screen.active_app}",
                    source="perception",
                    salience=0.4,
                    ttl=120,
                )

            if frame.audio.voice_activity and frame.audio.transcript_changed:
                ws.record_event(
                    f"Voice detected: {frame.audio.transcript_snippet[:80]}",
                    source="perception",
                    salience=0.7,
                    ttl=60,
                )

        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("perceptual_pump.world_state", e)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_pump_instance: Optional[PerceptualPump] = None


def get_perceptual_pump() -> PerceptualPump:
    """Get or create the global PerceptualPump."""
    global _pump_instance
    if _pump_instance is None:
        _pump_instance = PerceptualPump()
    return _pump_instance


__all__ = [
    "PerceptualPump",
    "PerceptualFrame",
    "ScreenState",
    "AudioState",
    "SystemState",
    "UserState",
    "frame_to_runtime_body",
    "get_perceptual_pump",
]
