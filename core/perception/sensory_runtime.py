"""Sensory runtime — eyes, ears, and voice that Aura uses like a person uses theirs.

One unified, low-latency interface to the real senses, usable on a whim:

  look()    a glance through the camera (cv2): is someone there, do I recognize them?
  listen()  a moment of listening (sounddevice + mlx-whisper): what was said, in whose voice?
  speak()   say it out loud (macOS `say`), immediately
  sense()   a combined glance + listen, the way you take in a room at once

Each sense is a provider behind a clean seam, so the real backend (cv2 / sounddevice /
mlx-whisper / say) activates when the library, hardware, and OS permission are present, and a
bounded unavailable result keeps the mind aware of disabled hardware instead of pretending
capture succeeded. Captured perception is routed to
the perception sentinel (recognition + threat → the immune system → the unified state), so what
Aura sees and hears is reasoned about, not just logged.

Camera/mic are TCC-gated by macOS; the first use prompts Bryan to grant access. Continuous
sensing stays behind the owner flag (AURA_SENTINEL_PERCEPTION); a single look()/listen() on
demand is always available once permission is granted.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger("Perception.Sensory")
_SENSORY_IMPORT_ERRORS = (ImportError, ModuleNotFoundError)
_SENSORY_RUNTIME_ERRORS = (
    AttributeError,
    FloatingPointError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


# ── results ─────────────────────────────────────────────────────────────────

@dataclass
class Sight:
    captured: bool
    person_present: bool = False
    descriptor: np.ndarray | None = None   # coarse appearance descriptor (pluggable → real embeddings)
    width: int = 0
    height: int = 0
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Sound:
    captured: bool
    transcript: str = ""
    voice_descriptor: np.ndarray | None = None
    duration_s: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)


# ── provider seams (real backends activate when present; stubs otherwise) ────

class CameraProvider:
    """cv2-backed single-frame grab + lightweight face presence. Fail-open."""

    def __init__(self) -> None:
        self._cv2 = None
        self._cascade = None

    def available(self) -> bool:
        try:
            import cv2  # noqa: F401
            return True
        except _SENSORY_IMPORT_ERRORS:
            return False

    def _load(self) -> bool:
        if self._cv2 is not None:
            return True
        try:
            import cv2
            self._cv2 = cv2
            try:
                self._cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
            except _provider_errors(cv2):
                self._cascade = None
            return True
        except _SENSORY_IMPORT_ERRORS + _SENSORY_RUNTIME_ERRORS as exc:
            logger.debug("cv2 unavailable: %s", exc)
            return False

    def capture(self, *, camera_index: int = 0) -> Sight:
        if not self._load():
            return Sight(captured=False, detail={"reason": "cv2_unavailable"})
        cv2 = self._cv2
        cap = None
        try:
            cap = cv2.VideoCapture(camera_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                return Sight(captured=False, detail={"reason": "no_frame_or_permission"})
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            person = False
            descriptor = None
            if self._cascade is not None:
                faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                if len(faces) > 0:
                    person = True
                    x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
                    crop = gray[y:y + fh, x:x + fw]
                    descriptor = _appearance_descriptor(cv2, crop)
            return Sight(captured=True, person_present=person, descriptor=descriptor,
                         width=int(w), height=int(h), detail={"faces": int(person)})
        except _provider_errors(cv2) as exc:
            return Sight(captured=False, detail={"reason": f"capture_error:{type(exc).__name__}"})
        finally:
            if cap is not None:
                try:
                    cap.release()
                except _provider_errors(cv2) as exc:
                    logger.debug("camera release failed: %s", exc)


class MicProvider:
    """sounddevice capture + mlx-whisper transcription. Fail-open."""

    def __init__(self, *, sample_rate: int = 16000) -> None:
        self._sr = sample_rate

    def available(self) -> bool:
        try:
            import sounddevice  # noqa: F401
            return True
        except _SENSORY_IMPORT_ERRORS:
            return False

    def capture(self, *, seconds: float = 3.0) -> Sound:
        try:
            import sounddevice as sd
        except _SENSORY_IMPORT_ERRORS as exc:
            return Sound(captured=False, detail={"reason": f"sounddevice_unavailable:{type(exc).__name__}"})
        try:
            audio = sd.rec(int(seconds * self._sr), samplerate=self._sr, channels=1, dtype="float32")
            sd.wait()
            audio = audio.reshape(-1)
            transcript = self._transcribe(audio)
            return Sound(captured=True, transcript=transcript,
                         voice_descriptor=_voice_descriptor(audio, self._sr),
                         duration_s=seconds, detail={"samples": int(audio.shape[0])})
        except _provider_errors(sd) as exc:
            return Sound(captured=False, detail={"reason": f"capture_error:{type(exc).__name__}"})

    def _transcribe(self, audio: np.ndarray) -> str:
        try:
            import mlx_whisper
            result = mlx_whisper.transcribe(
                audio, path_or_hf_repo="mlx-community/whisper-small.en-mlx",
            )
            return str(result.get("text", "")).strip() if isinstance(result, dict) else ""
        except _SENSORY_IMPORT_ERRORS + _SENSORY_RUNTIME_ERRORS as exc:
            logger.debug("transcription unavailable: %s", exc)
            return ""


class VoiceProvider:
    """macOS `say` — instant, precise text-to-speech. Fail-open."""

    def speak(self, text: str, *, voice: str | None = None, rate: int | None = None) -> bool:
        text = str(text or "").strip()
        if not text:
            return False
        try:
            from core.runtime.subprocess_gateway import get_subprocess_gateway
            argv = ["say"]
            if voice:
                argv += ["-v", voice]
            if rate:
                argv += ["-r", str(int(rate))]
            argv.append(text[:2000])
            proc = get_subprocess_gateway().run(argv, timeout=30.0, source="perception.sensory.voice")
            return getattr(proc, "returncode", 1) == 0
        except _SENSORY_IMPORT_ERRORS + _SENSORY_RUNTIME_ERRORS as exc:
            logger.debug("say unavailable: %s", exc)
            return False


# ── descriptors (coarse now; pluggable to real embeddings) ──────────────────

def _appearance_descriptor(cv2: Any, gray_crop: np.ndarray) -> np.ndarray | None:
    try:
        small = cv2.resize(gray_crop, (16, 16)).astype(np.float64).reshape(-1)
        n = float(np.linalg.norm(small))
        return small / n if n > 1e-9 else small
    except _provider_errors(cv2):
        return None


def _voice_descriptor(audio: np.ndarray, sr: int) -> np.ndarray | None:
    # Coarse spectral fingerprint (a real speaker-ID embedding plugs in here later).
    try:
        if audio.size < sr // 4:
            return None
        spec = np.abs(np.fft.rfft(audio[: sr * 2]))
        if spec.size < 32:
            return None
        binned = spec[:512].reshape(32, -1).mean(axis=1)
        n = float(np.linalg.norm(binned))
        return binned / n if n > 1e-9 else binned
    except _SENSORY_RUNTIME_ERRORS:
        return None


def _provider_errors(provider: Any | None = None) -> tuple[type[BaseException], ...]:
    extra: list[type[BaseException]] = []
    if provider is not None:
        for attr in ("error", "PortAudioError"):
            candidate = getattr(provider, attr, None)
            if isinstance(candidate, type) and issubclass(candidate, BaseException):
                extra.append(candidate)
    return (*_SENSORY_IMPORT_ERRORS, *_SENSORY_RUNTIME_ERRORS, *extra)


# ── the runtime ──────────────────────────────────────────────────────────────

class SensoryRuntime:
    """Unified, on-demand eyes/ears/voice, routed to the mind."""

    def __init__(
        self,
        *,
        camera: CameraProvider | None = None,
        mic: MicProvider | None = None,
        voice: VoiceProvider | None = None,
    ) -> None:
        self.camera = camera or CameraProvider()
        self.mic = mic or MicProvider()
        self.voice = voice or VoiceProvider()
        self._lock = threading.RLock()

    # eyes ----------------------------------------------------------------
    def look(self, *, route: bool = True) -> Sight:
        sight = self.camera.capture()
        if route and sight.captured and sight.person_present:
            self._route_face(sight)
        return sight

    # ears ----------------------------------------------------------------
    def listen(self, *, seconds: float = 3.0, route: bool = True) -> Sound:
        sound = self.mic.capture(seconds=seconds)
        if route and sound.captured and (sound.transcript or sound.voice_descriptor is not None):
            self._route_voice(sound)
        return sound

    # voice ---------------------------------------------------------------
    def speak(self, text: str, **kw: Any) -> bool:
        return self.voice.speak(text, **kw)

    # take in the room at once -------------------------------------------
    def sense(self, *, listen_seconds: float = 2.0) -> dict[str, Any]:
        sight = self.look()
        sound = self.listen(seconds=listen_seconds)
        return {"sight": sight, "sound": sound}

    def capabilities(self) -> dict[str, bool]:
        return {
            "eyes": self.camera.available(),
            "ears": self.mic.available(),
            "voice": True,  # `say` is effectively always present on macOS
        }

    # routing to the mind -------------------------------------------------
    def _route_face(self, sight: Sight) -> None:
        try:
            from core.perception.perception_sentinel import (
                Modality,
                Observation,
                get_perception_sentinel,
            )
            get_perception_sentinel().assess(Observation(
                Modality.FACE, descriptor=sight.descriptor, content="",
                context={"source": "camera", "size": [sight.width, sight.height]},
            ))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("face routing skipped: %s", exc)

    def _route_voice(self, sound: Sound) -> None:
        try:
            from core.perception.perception_sentinel import (
                Modality,
                Observation,
                get_perception_sentinel,
            )
            get_perception_sentinel().assess(Observation(
                Modality.VOICE, descriptor=sound.voice_descriptor, content=sound.transcript,
                context={"source": "mic", "duration_s": sound.duration_s},
            ))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("voice routing skipped: %s", exc)


_runtime: SensoryRuntime | None = None
_lock = threading.Lock()


def get_sensory_runtime() -> SensoryRuntime:
    global _runtime
    if _runtime is None:
        with _lock:
            if _runtime is None:
                _runtime = SensoryRuntime()
    return _runtime
