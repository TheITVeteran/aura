import asyncio
import logging
import os
import time
from collections import deque
from enum import StrEnum
from pathlib import Path
from typing import Any

from core.runtime.boot_safety import main_process_camera_policy
from core.runtime.errors import record_degradation
from core.runtime.shutdown_coordinator import is_shutdown_requested
from core.security.screen_capture_policy import evaluate_screen_capture_admission_async
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger(__name__)


class ScreenBackendState(StrEnum):
    """Truthful, privacy-safe state of the continuous screen capture lane."""

    UNINITIALIZED = "uninitialized"
    READY = "ready"
    PRIVACY_DEFERRED = "privacy_deferred"
    PERMISSION_DEFERRED = "permission_deferred"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    NO_MONITORS = "no_monitors"
    BACKEND_ERROR = "backend_error"


# When a screen frame was last captured, as a monotonic clock reading.
#
# The continuous vision feed grabs the screen every couple of seconds and is
# the reason she can describe what is on it. It is not a tool dispatch and
# files no receipt, so a reliability gate looking only for receipts concluded
# that an ACCURATE description of Bryan's screen was fabricated. A fresh
# frame is the evidence that perception really happened.
_LAST_SCREEN_FRAME_AT: float = 0.0


def _note_screen_frame() -> None:
    global _LAST_SCREEN_FRAME_AT
    _LAST_SCREEN_FRAME_AT = time.monotonic()


def screen_frame_age_seconds() -> float | None:
    """Seconds since the last screen capture, or None if there has been none."""
    if _LAST_SCREEN_FRAME_AT <= 0.0:
        return None
    return max(0.0, time.monotonic() - _LAST_SCREEN_FRAME_AT)


class ContinuousSensoryBuffer:
    """Maintains a rolling buffer of screen captures for real-time spatial awareness."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.sct = None
        from concurrent.futures import ThreadPoolExecutor
        self._vision_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="AuraVision")
        self._capture_lock = asyncio.Lock()
        self._mss_module = None
        self._screen_probe_cooldown_until = 0.0
        self._last_backend_fail_log = 0.0
        self._screen_permission_notice_at = 0.0
        self._screen_permission_notice_interval_s = 300.0
        self._screen_backend_state = ScreenBackendState.UNINITIALIZED
        self._screen_backend_reason = "not_probed"
        self._screen_retry_delay_s = 0.75
        try:
            import mss

            self._mss_module = mss
        except (ImportError, ModuleNotFoundError):
            logger.warning("👁️ [VISION] mss not found. Continuous Sensory Buffer will be disabled.")

        self.frame_buffer = deque(maxlen=6)
        self._capture_task = None
        self._is_active = False
        # The device handle is the camera authority's, never this
        # object's. Holding it here was how two subsystems ended up
        # opening the same camera with neither able to see the other.
        self._camera_lease: Any | None = None
        self._last_camera_denial: str | None = None

        from core.config import get_config

        requested_camera = get_config().features.camera_enabled
        if os.environ.get("AURA_FORCE_CAMERA") == "1":
            requested_camera = True

        self.camera_enabled, camera_reason = main_process_camera_policy(requested_camera)
        # `camera_enabled` intentionally continues to mean "safe to open in
        # this process" for compatibility and diagnostics. Capture can still
        # be enabled through the isolated sidecar when the macOS policy says no.
        sidecar_available = False
        if requested_camera and not self.camera_enabled:
            try:
                from core.perception.camera_authority import get_camera_authority

                sidecar_available = bool(
                    get_camera_authority().state().get("backend_available")
                )
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
                sidecar_available = False
        self.camera_capture_enabled = bool(
            requested_camera and (self.camera_enabled or sidecar_available)
        )
        if self.camera_enabled and os.environ.get("AURA_FORCE_CAMERA") == "1":
            logger.info("👁️ [VISION] Camera FORCED ON via AURA_FORCE_CAMERA=1.")
        elif requested_camera and not self.camera_enabled:
            logger.warning("👁️ [VISION] %s", camera_reason)
        elif not self.camera_enabled:
            logger.info(
                "👁️ [VISION] Camera disabled by default (Metal Conflict Safety). "
                "Use AURA_FORCE_CAMERA=1 plus "
                "AURA_ALLOW_UNSAFE_MAIN_PROCESS_CAMERA=1 to override."
            )

        self.monitor = None
        self._last_compute_budget = None

    @staticmethod
    def _compute_budget():
        """Dynamic cadence for this always-on but non-foreground sense."""
        from core.runtime.background_policy import constitutive_compute_budget

        return constitutive_compute_budget(
            "continuous_sensory_buffer",
            0.5,
            min_hz=0.1,
            foreground_hz=0.1,
            memory_high_hz=0.2,
            memory_critical_hz=0.1,
            compute_pressure_hz=0.1,
            failure_pressure_hz=0.1,
        )

    def start(self):
        """Starts the background rolling capture loop."""
        if is_shutdown_requested():
            logger.info("👁️ Continuous Sensory Buffer not started: runtime shutdown requested.")
            return
        from core.senses.vision_policy import vision_policy_reason

        vision_refusal = vision_policy_reason()
        if vision_refusal:
            # Named reason: "headless" and "operator_disabled" used to log the
            # same sentence, which made a missing screen indistinguishable from
            # a deliberate setting.
            logger.info(
                "👁️ Continuous Sensory Buffer not started (%s).", vision_refusal
            )
            return
        if not self._is_active:
            if self._mss_module is None and not self.camera_capture_enabled:
                logger.warning("👁️ Continuous Sensory Buffer not started: no capture backends are available.")
                return
            self._is_active = True
            try:
                self._capture_task = get_task_tracker().create_task(
                    self._capture_loop(),
                    name="ContinuousSensoryCapture",
                )
            except RuntimeError as exc:
                self._is_active = False
                record_degradation("continuous_vision", exc)
                logger.warning("👁️ Continuous Sensory Buffer not started: no running event loop.")
                return
            logger.info("👁️ Continuous Sensory Buffer Online.")

    def stop(self):
        """Stops the capture loop."""
        self._is_active = False
        if self._camera_lease is not None:
            try:
                from core.perception.camera_authority import get_camera_authority

                get_camera_authority().release(self._camera_lease)
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
                logger.debug("ContinuousSensoryBuffer: camera release skipped", exc_info=True)
            self._camera_lease = None
        if self._capture_task:
            self._capture_task.cancel()
            self._capture_task = None
            logger.info("👁️ Continuous Sensory Buffer Offline.")

    async def _screen_permission_active(self) -> bool:
        admission = await evaluate_screen_capture_admission_async()
        if not admission.allowed:
            self._set_screen_backend_state(
                ScreenBackendState.PRIVACY_DEFERRED,
                admission.reason.value,
            )
            now = time.monotonic()
            if (
                getattr(self, "_screen_permission_notice_at", 0.0) <= 0.0
                or (now - getattr(self, "_screen_permission_notice_at", 0.0))
                >= getattr(self, "_screen_permission_notice_interval_s", 300.0)
            ):
                logger.info("👁️ [VISION] Continuous screen buffer deferred: %s.", admission.public_error)
                self._screen_permission_notice_at = now
            return False
        try:
            from core.container import ServiceContainer
            from core.security.permission_guard import PermissionType

            guard = ServiceContainer.get("permission_guard", default=None)
            if not guard:
                granted = os.getenv("AURA_ASSUME_SCREEN_PERMISSION", "0") == "1"
                if not granted:
                    self._set_screen_backend_state(
                        ScreenBackendState.PERMISSION_DEFERRED,
                        "permission_guard_unavailable",
                    )
                return granted
            check = await guard.check_permission(PermissionType.SCREEN)
            granted = bool(check.get("granted", False))
            if not granted:
                self._set_screen_backend_state(
                    ScreenBackendState.PERMISSION_DEFERRED,
                    str(check.get("status") or "screen_permission_inactive"),
                )
                now = time.monotonic()
                if (
                    getattr(self, "_screen_permission_notice_at", 0.0) <= 0.0
                    or (now - getattr(self, "_screen_permission_notice_at", 0.0))
                    >= getattr(self, "_screen_permission_notice_interval_s", 300.0)
                ):
                    logger.info(
                        "👁️ [VISION] Continuous screen buffer deferred: screen permission is not active for this app identity."
                    )
                    self._screen_permission_notice_at = now
            else:
                self._screen_permission_notice_at = 0.0
                if self.sct is None or self.monitor is None:
                    self._set_screen_backend_state(
                        ScreenBackendState.UNINITIALIZED,
                        "permission_active",
                    )
            return granted
        except (
            ImportError,
            AttributeError,
            OSError,
            RuntimeError,
            TimeoutError,
            TypeError,
            ValueError,
        ) as exc:
            record_degradation('continuous_vision', exc)
            self._set_screen_backend_state(
                ScreenBackendState.PERMISSION_DEFERRED,
                "permission_probe_failed",
            )
            logger.debug("ContinuousSensoryBuffer permission probe failed: %s", exc)
            return False

    def _set_screen_backend_state(
        self,
        state: ScreenBackendState,
        reason: str,
    ) -> None:
        self._screen_backend_state = state
        self._screen_backend_reason = str(reason or "unknown")

    def _screen_retry_delay(self) -> float:
        state = getattr(
            self,
            "_screen_backend_state",
            ScreenBackendState.UNINITIALIZED,
        )
        reason = str(getattr(self, "_screen_backend_reason", "") or "")
        if state is ScreenBackendState.PRIVACY_DEFERRED:
            if reason in {"foreground_unknown", "browser_title_unknown"}:
                return 0.75
            if reason in {"private_foreground", "private_visible"}:
                return 2.0
            return 15.0
        if state is ScreenBackendState.PERMISSION_DEFERRED:
            return 2.0
        if state is ScreenBackendState.NO_MONITORS:
            return 30.0
        if state in {
            ScreenBackendState.BACKEND_UNAVAILABLE,
            ScreenBackendState.BACKEND_ERROR,
        }:
            return 15.0
        return 0.75

    def _schedule_screen_retry(self) -> float:
        delay = self._screen_retry_delay()
        self._screen_retry_delay_s = delay
        self._screen_probe_cooldown_until = time.monotonic() + delay
        return delay

    async def _close_screen_backend_candidate(self, candidate: Any) -> None:
        close = getattr(candidate, "close", None)
        if not callable(close):
            return
        try:
            executor = getattr(self, "_vision_executor", None)
            if executor is None:
                close()
            else:
                await asyncio.get_running_loop().run_in_executor(executor, close)
        except (OSError, RuntimeError, TypeError, ValueError):
            logger.debug("ContinuousSensoryBuffer candidate close failed", exc_info=True)

    async def _invalidate_screen_backend(self, reason: str) -> None:
        """Close a failed capture handle so the next tick performs a clean probe."""

        candidate = self.sct
        self.sct = None
        self.monitor = None
        if candidate is not None:
            await self._close_screen_backend_candidate(candidate)
        self._set_screen_backend_state(ScreenBackendState.BACKEND_ERROR, reason)
        self._schedule_screen_retry()

    async def _ensure_screen_backend(self) -> bool:
        if self.sct is not None and self.monitor is not None:
            self._set_screen_backend_state(ScreenBackendState.READY, "capture_ready")
            return True
        if self._mss_module is None:
            self._set_screen_backend_state(
                ScreenBackendState.BACKEND_UNAVAILABLE,
                "mss_unavailable",
            )
            return False
        if time.monotonic() < getattr(self, "_screen_probe_cooldown_until", 0.0):
            return False
        if not await self._screen_permission_active():
            self._schedule_screen_retry()
            return False
        sct = None
        try:
            sct = await asyncio.get_running_loop().run_in_executor(
                self._vision_executor,
                self._mss_module.mss,
            )
            monitors = list(getattr(sct, "monitors", None) or [])
            monitor = None
            # Try to find the first monitor with non-zero size
            for m in monitors:
                if m.get("width", 0) > 0 and m.get("height", 0) > 0:
                    # Skip monitor 0 (combined) if others are available
                    if m == monitors[0] and len(monitors) > 1:
                        continue
                    monitor = m
                    break

            if not monitor and monitors:
                monitor = monitors[0]

            if (
                monitor
                and monitor.get("width", 0) > 0
                and monitor.get("height", 0) > 0
            ):
                self.sct = sct
                self.monitor = monitor
                self._screen_probe_cooldown_until = 0.0
                self._screen_retry_delay_s = 0.75
                self._set_screen_backend_state(
                    ScreenBackendState.READY,
                    "capture_ready",
                )
                logger.info(
                    "👁️ [VISION] Continuous screen capture backend initialized: %sx%s.",
                    monitor.get("width", 0),
                    monitor.get("height", 0),
                )
                return True
            await self._close_screen_backend_candidate(sct)
            self._set_screen_backend_state(
                ScreenBackendState.NO_MONITORS,
                "monitor_enumeration_empty",
            )
            self._schedule_screen_retry()
            logger.info(
                "👁️ [VISION] Screen capture unavailable: monitor enumeration returned no valid displays."
            )
            return False
        except (OSError, ConnectionError, RuntimeError, TimeoutError) as exc:
            if sct is not None:
                await self._close_screen_backend_candidate(sct)
            record_degradation('continuous_vision', exc)
            self._set_screen_backend_state(
                ScreenBackendState.BACKEND_ERROR,
                type(exc).__name__,
            )
            self._schedule_screen_retry()
            logger.warning("👁️ [VISION] Continuous screen capture backend unavailable: %s", exc)
            return False

    async def _capture_loop(self):
        """Runs continuously in the background, updating Aura's visual working memory."""
        while self._is_active:
            budget = self._compute_budget()
            self._last_compute_budget = budget
            try:
                if self.sct is None or self.monitor is None:
                    await self._ensure_screen_backend()
                    if self.sct is None:
                        now = time.monotonic()
                        if now - getattr(self, "_last_backend_fail_log", 0) > 300.0:
                            logger.info(
                                "👁️ [VISION] Screen capture deferred: state=%s reason=%s.",
                                getattr(
                                    self,
                                    "_screen_backend_state",
                                    ScreenBackendState.UNINITIALIZED,
                                ).value,
                                getattr(self, "_screen_backend_reason", "unknown"),
                            )
                            self._last_backend_fail_log = now
                        if not self.camera_capture_enabled:
                            await asyncio.sleep(
                                max(
                                    0.25,
                                    float(
                                        getattr(
                                            self,
                                            "_screen_retry_delay_s",
                                            self._screen_retry_delay(),
                                        )
                                    ),
                                )
                            )
                            continue

                if self.sct and self.monitor:
                    async with self._capture_lock:
                        # The signed native bridge rechecks foreground privacy
                        # at the final boundary before the Python capture call.
                        admission = await evaluate_screen_capture_admission_async()
                        if not admission.allowed:
                            self._set_screen_backend_state(
                                ScreenBackendState.PRIVACY_DEFERRED,
                                admission.reason.value,
                            )
                            # A prior public frame must not masquerade as the
                            # current private screen.
                            self.frame_buffer.clear()
                            sct_img = None
                        else:
                            self._set_screen_backend_state(
                                ScreenBackendState.READY,
                                "capture_ready",
                            )
                            try:
                                sct_img = await asyncio.wait_for(
                                    asyncio.get_running_loop().run_in_executor(
                                        self._vision_executor, self.sct.grab, self.monitor
                                    ),
                                    timeout=10.0
                                )
                            except TimeoutError:
                                logger.error("👁️ [VISION] Screenshot capture timed out after 10s. Skipping frame.")
                                await self._invalidate_screen_backend("capture_timeout")
                                sct_img = None
                            except (OSError, ConnectionError, RuntimeError, TypeError, ValueError) as exc:
                                record_degradation(
                                    "continuous_vision",
                                    exc,
                                    severity="warning",
                                    action="discarded failed screen handle and scheduled a clean reprobe",
                                    enforce_failure_policy=False,
                                )
                                logger.warning(
                                    "👁️ [VISION] Screenshot capture failed; reopening the backend: %s",
                                    exc,
                                )
                                await self._invalidate_screen_backend(type(exc).__name__)
                                sct_img = None

                    if sct_img:
                        import mss.tools

                        # PNG compression is CPU-heavy on a full-resolution
                        # frame. Running it on the asyncio thread caused the
                        # same multi-second event-loop lag the health monitor
                        # correctly reported during active perception.
                        png_bytes = await asyncio.get_running_loop().run_in_executor(
                            self._vision_executor,
                            mss.tools.to_png,
                            sct_img.rgb,
                            sct_img.size,
                        )
                        self.frame_buffer.append(("image/png", png_bytes))
                        # A frame IS the evidence for "I can see your screen".
                        # Recorded so the reliability gate can tell a real
                        # observation from an invented one without needing a
                        # per-turn tool dispatch — this feed is continuous and
                        # never produces one.
                        _note_screen_frame()

                camera_admitted = (
                    self.camera_capture_enabled
                    and not budget.foreground_active
                    and budget.effective_hz > 0.100001
                )
                if not camera_admitted and self._camera_lease is not None:
                    from core.perception.camera_authority import get_camera_authority

                    await asyncio.to_thread(
                        get_camera_authority().release, self._camera_lease
                    )
                    self._camera_lease = None

                if camera_admitted:
                    # `self.camera_enabled` is the build-time feature flag.
                    # It is NOT the owner's settings toggle, so this loop
                    # used to keep filming after the camera was switched off
                    # in Aura's settings. The authority checks the owner's
                    # switch on every acquisition, holds the single device
                    # lease, and — because this is a continuous feed Aura
                    # runs on her own initiative rather than at the owner's
                    # request — asks whether autonomous observation is
                    # permitted at all.
                    from core.perception.camera_authority import (
                        CameraDenial,
                        get_camera_authority,
                    )

                    authority = get_camera_authority()
                    if self._camera_lease is None or not self._camera_lease.active:
                        acquired = await asyncio.to_thread(
                            authority.acquire,
                            "continuous_vision",
                            purpose="rolling visual context buffer",
                            autonomous=True,
                        )
                        if isinstance(acquired, CameraDenial):
                            if acquired.reason != self._last_camera_denial:
                                self._last_camera_denial = acquired.reason
                                logger.info(
                                    "👁️ [VISION] Camera not available: %s — %s",
                                    acquired.reason,
                                    acquired.detail,
                                )
                        else:
                            self._last_camera_denial = None
                            self._camera_lease = acquired

                    if self._camera_lease is not None:
                        frame = await asyncio.to_thread(authority.read, self._camera_lease)
                    else:
                        frame = None
                    if frame is None:
                        if self._camera_lease is not None:
                            # Either the frame failed or the lease was reclaimed.
                            # Drop it and re-acquire next tick rather than
                            # spinning on a handle that may already be closed.
                            await asyncio.to_thread(authority.release, self._camera_lease)
                            self._camera_lease = None
                    else:
                        jpeg_bytes = await asyncio.to_thread(
                            authority.jpeg_bytes,
                            self._camera_lease,
                            frame,
                        )
                        self.frame_buffer.append(("image/jpeg", jpeg_bytes))
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation('continuous_vision', e)
                logger.error("Sensory Buffer capture failed: %s", e)

            await asyncio.sleep(budget.interval_s)

    def get_visual_context_parts(self) -> list:
        """Retrieves the rolling visual buffer formatted for the Gemini API."""
        if not self.frame_buffer:
            return []

        return [
            {"mime_type": mime_type, "data": frame_bytes}
            for mime_type, frame_bytes in self.frame_buffer
        ]

    async def query_visual_context(self, prompt: str, brain: Any, mode: Any | None = None) -> str:
        """
        Sends the current frame buffer and the prompt to the brain for visual reasoning.

        Args:
            prompt: The specific question or directive for the visual context.
            brain: The CognitiveEngine (or GeminiAdapter) instance capable of multimodal logic.
        """
        if not self.frame_buffer:
            return "I don't have any visual frames in my buffer yet."

        parts = self.get_visual_context_parts()
        parts.insert(0, {"text": prompt})

        try:
            if hasattr(brain, "think"):
                from core.brain.types import ThinkingMode

                thought = await brain.think(prompt, mode=mode or ThinkingMode.FAST, parts=parts)
                return thought.content if hasattr(thought, "content") else str(thought)
            elif hasattr(brain, "call"):
                success, text, _ = await brain.call(prompt, parts=parts)
                return text if success else "I failed to process the visual data."
            else:
                return "My cognitive systems are not equipped for that visual request."
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('continuous_vision', e)
            logger.error("Visual reasoning failed: %s", e)
            return f"I had an error analyzing my vision: {e}"
