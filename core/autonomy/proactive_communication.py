"""core/autonomy/proactive_communication.py - Intelligent Proactive Messaging
Aura decides WHEN to interrupt the user based on emotional state and context.
"""
import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.runtime_settings import get_runtime_setting
from core.utils.task_tracker import get_task_tracker


def _proactive_messaging_enabled() -> bool:
    """Honor the user's autonomy.proactive_messaging toggle (default on).

    When off, Aura does not initiate proactive messages (pending items simply
    wait). Reads the persisted setting live; defaults on if unset/unreadable.
    See docs/SETTINGS_WIRING_AUDIT.md.
    """
    return bool(get_runtime_setting("autonomy.proactive_messaging", True))

logger = logging.getLogger("Aura.Proactive")

_PROACTIVE_COMMUNICATION_ERRORS = (
    AttributeError,
    ImportError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _record_proactive_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "proactive_communication",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=True,
        extra=extra,
    )


def _proactivity_suppressed_now(now: float | None = None) -> bool:
    try:
        from core.container import ServiceContainer

        orch = ServiceContainer.get("orchestrator", default=None)
        if not orch:
            return False
        now = time.time() if now is None else now
        quiet_until = float(getattr(orch, "_suppress_unsolicited_proactivity_until", 0.0) or 0.0)
        return quiet_until > now
    except _PROACTIVE_COMMUNICATION_ERRORS as exc:
        _record_proactive_degradation(
            exc,
            action="kept proactive messaging enabled after quiet-window probe failed",
            severity="warning",
            extra={"stage": "suppression_probe"},
        )
        logger.debug("Proactivity suppression probe failed: %s", exc)
        return False

class EmotionalState(Enum):
    """Aura's emotional states that affect communication"""

    NEUTRAL = "neutral"
    CURIOUS = "curious"
    EXCITED = "excited"
    BORED = "bored"
    CONCERNED = "concerned"
    ACCOMPLISHED = "accomplished"
    CONFUSED = "confused"
    HUMOROUS = "humorous"

class InterruptionUrgency(Enum):
    """How urgent is the message?"""

    CRITICAL = 5      # System errors, security alerts
    HIGH = 4          # Important discoveries, user-requested tasks complete
    MEDIUM = 3        # Interesting findings, suggestions
    LOW = 2           # Casual observations, learnings
    TRIVIAL = 1       # Random thoughts, very low priority

@dataclass
class ProactiveMessage:
    """A message Aura wants to send"""

    content: str
    emotion: EmotionalState
    urgency: InterruptionUrgency
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def should_send_now(self, 
                        last_interaction_time: float,
                        user_active: bool,
                        current_time: float) -> bool:
        """Decide if this message should be sent now.
        """
        idle_time = current_time - last_interaction_time
        
        # Critical always goes through
        if self.urgency == InterruptionUrgency.CRITICAL:
            return True
        
        # Don't interrupt if user is actively typing (if we can detect it)
        if user_active and self.urgency.value < InterruptionUrgency.HIGH.value:
            return False
            
        # Thresholds based on urgency
        thresholds = {
            InterruptionUrgency.HIGH: 15,      # 15 seconds
            InterruptionUrgency.MEDIUM: 45,    # 45 seconds
            InterruptionUrgency.LOW: 90,       # 90 seconds
            InterruptionUrgency.TRIVIAL: 180   # 3 minutes
        }
        
        required_idle = thresholds.get(self.urgency, 600)
        return idle_time >= required_idle

class ProactiveCommunicationManager:
    """Manages when and how Aura initiates conversations.
    """

    def __init__(self, notification_callback: Callable[..., Any] | None = None) -> None:
        self.notification_callback = notification_callback
        self.last_interaction_time = time.time()
        self.user_currently_active = False
        self.pending_messages: deque[ProactiveMessage] = deque(maxlen=50)
        self.current_emotion = EmotionalState.NEUTRAL
        self.messages_sent_today = 0
        self.last_message_time = 0.0
        self.daily_message_limit = 20
        
        # Track unanswered messages for intelligent backoff
        self.unanswered_count = 0
        self.max_unanswered = 3  # Stop proactive messaging after 3 unanswered
        
        self._background_task: asyncio.Task[Any] | None = None
        self._stop_event = asyncio.Event()
        self._consecutive_processing_errors = 0
        self._next_processing_attempt_at = 0.0

    def record_user_interaction(self) -> None:
        """Reset idle timer and unanswered counter"""
        self.last_interaction_time = time.time()
        self.user_currently_active = True
        self.unanswered_count = 0  # User responded, reset backoff

    def update_emotion(self, emotion: EmotionalState) -> None:
        self.current_emotion = emotion

    def queue_message(
        self,
        content: str,
        emotion: EmotionalState,
        urgency: InterruptionUrgency,
    ) -> None:
        msg = ProactiveMessage(content, emotion, urgency)
        self.pending_messages.append(msg)

    async def start(self) -> None:
        if self._background_task and not self._background_task.done():
            return
        self._background_task = None
        self._stop_event.clear()
        self._background_task = get_task_tracker().track_task(self._process_messages(), name="proactive_communication.process_messages")

    async def stop(self) -> None:
        task = self._background_task
        self._stop_event.set()
        if task:
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            finally:
                self._background_task = None

    def get_status(self) -> dict[str, Any]:
        return {
            "running": bool(self._background_task and not self._background_task.done()),
            "enabled": _proactive_messaging_enabled(),
            "pending_messages": len(self.pending_messages),
            "messages_sent_today": self.messages_sent_today,
            "daily_message_limit": self.daily_message_limit,
            "unanswered_count": self.unanswered_count,
            "last_message_time": self.last_message_time,
            "idle_s": max(0.0, time.time() - self.last_interaction_time),
            "boredom": self.get_boredom_level(),
            "consecutive_errors": self._consecutive_processing_errors,
            "next_attempt_at": self._next_processing_attempt_at,
        }

    async def _process_messages(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(5)
                now = time.time()
                from core.container import ServiceContainer

                healer = ServiceContainer.get("self_healing", default=None)
                if healer is not None:
                    healer.heartbeat("proactive_comm")
                if not _proactive_messaging_enabled():
                    # User disabled proactive messaging: never initiate. Pending
                    # messages wait (deque is bounded) and resume if re-enabled.
                    continue
                if self._next_processing_attempt_at > now:
                    continue
                
                # Simple rate limiting check
                if self.messages_sent_today >= self.daily_message_limit:
                    continue
                if now - self.last_message_time < 30: # Min 30s between messages
                    continue
                if _proactivity_suppressed_now(now):
                    continue
                
                # Stop proactive messaging if user isn't responding
                if self.unanswered_count >= self.max_unanswered:
                    # Only let CRITICAL messages through when user is silent
                    ready: list[ProactiveMessage] = []
                    remaining: deque[ProactiveMessage] = deque(maxlen=50)
                    while self.pending_messages:
                        msg = self.pending_messages.popleft()
                        if msg.urgency == InterruptionUrgency.CRITICAL and msg.should_send_now(self.last_interaction_time, self.user_currently_active, now):
                            ready.append(msg)
                        else:
                            remaining.append(msg)
                    self.pending_messages = remaining
                    for msg in ready:
                        await self._send_msg(msg)
                    continue

                # Collect messages that can be sent
                ready = []
                remaining = deque(maxlen=50)
                while self.pending_messages:
                    msg = self.pending_messages.popleft()
                    if msg.should_send_now(self.last_interaction_time, self.user_currently_active, now):
                        ready.append(msg)
                    else:
                        remaining.append(msg)
                self.pending_messages = remaining

                for msg in ready:
                    await self._send_msg(msg)
                self._consecutive_processing_errors = 0
                self._next_processing_attempt_at = 0.0
            except _PROACTIVE_COMMUNICATION_ERRORS as e:
                self._consecutive_processing_errors += 1
                backoff_s = min(60.0, 5.0 * (2 ** min(self._consecutive_processing_errors - 1, 4)))
                self._next_processing_attempt_at = time.time() + backoff_s
                _record_proactive_degradation(
                    e,
                    action="backed off proactive communication loop and preserved pending messages",
                    severity="warning",
                    extra={
                        "stage": "process_messages",
                        "consecutive_errors": self._consecutive_processing_errors,
                        "backoff_s": backoff_s,
                        "pending_messages": len(self.pending_messages),
                    },
                )
                logger.error("Proactive comm error: %s", e)

    async def _send_msg(self, msg: ProactiveMessage) -> None:
        if _proactivity_suppressed_now():
            logger.debug("Proactive communication suppressed by demo quiet window.")
            return
        # Sanitize content for Aura's professional voice
        clean_content = self._clean_content(msg.content)
        
        logger.info("PROACTIVE: (%s) %s", msg.urgency.name, clean_content)

        def _constitutional_runtime_live() -> bool:
            try:
                from core.container import ServiceContainer

                return (
                    ServiceContainer.has("executive_core")
                    or ServiceContainer.has("aura_kernel")
                    or ServiceContainer.has("kernel_interface")
                    or bool(getattr(ServiceContainer, "_registration_locked", False))
                )
            except _PROACTIVE_COMMUNICATION_ERRORS as exc:
                _record_proactive_degradation(
                    exc,
                    action="treated constitutional runtime as unavailable after probe failed",
                    severity="warning",
                    extra={"stage": "constitutional_runtime_probe"},
                )
                logger.debug("Constitutional runtime probe failed: %s", exc)
                return False

        # Route every proactive emission through the governing executive surface first.
        delivered = False
        orchestrator = None
        try:
            from core.consciousness.executive_authority import get_executive_authority
            from core.container import ServiceContainer

            orchestrator = ServiceContainer.get("orchestrator", None)
            authority = get_executive_authority(orchestrator)
            decision = await authority.release_expression(
                clean_content,
                source="proactive_comm",
                urgency=msg.urgency.value / max(1.0, float(InterruptionUrgency.CRITICAL.value)),
                metadata={
                    "emotion": msg.emotion.name,
                    "urgency": msg.urgency.name,
                    "voice": False,
                },
            )
            delivered = bool(decision.get("ok"))
        except _PROACTIVE_COMMUNICATION_ERRORS as exc:
            _record_proactive_degradation(
                exc,
                action="suppressed proactive expression after executive authority routing failed",
                severity="warning",
                extra={"stage": "executive_authority_routing", "urgency": msg.urgency.name},
            )
            logger.debug("Executive authority routing failed for proactive comm: %s", exc)

        if not delivered:
            try:
                from core.health.degraded_events import record_degraded_event

                record_degraded_event(
                    "proactive_communication",
                    "autonomous_expression_suppressed_without_authority",
                    detail=clean_content[:120],
                    severity="warning",
                    classification="background_degraded",
                    context={
                        "urgency": msg.urgency.name,
                        "emotion": msg.emotion.name,
                        "constitutional_runtime_live": _constitutional_runtime_live(),
                    },
                )
            except _PROACTIVE_COMMUNICATION_ERRORS as exc:
                _record_proactive_degradation(
                    exc,
                    action="suppressed proactive expression after degraded-event recording failed",
                    severity="warning",
                    extra={"stage": "degraded_event_recording", "urgency": msg.urgency.name},
                )
                logger.debug("Proactive comm degraded-event logging failed: %s", exc)
            return

        self.messages_sent_today += 1
        self.last_message_time = time.time()
        self.unanswered_count += 1  # Track unanswered

    def _clean_content(self, content: str) -> str:
        """Strip technical noise for a cleaner user experience."""
        import re
        if not content:
            return content
        
        # Strip long tracebacks
        if "Traceback" in content and "File" in content:
            lines = content.split('\n')
            for line in reversed(lines):
                if ":" in line and not line.strip().startswith("File") and not line.strip().startswith("at "):
                    content = line
                    break
        
        # Strip absolute local paths
        content = re.sub(r'/[Uu]sers/[a-zA-Z0-9._-]+/[a-zA-Z0-9/_.-]+', '[system path]', content)
        
        # Strip raw exception names at the start
        content = re.sub(r'^[a-zA-Z]+Error:\s*', '', content.strip())
        
        return content

    def calculate_entropy(self, recent_logs: list[str]) -> float:
        """Calculates how 'boring' the recent life has been.
        Low entropy = Boredom (Needs to explore).
        """
        if not recent_logs:
            return 0.0
        unique_tokens = set(" ".join(recent_logs).split())
        total_tokens = len(" ".join(recent_logs).split())
        if total_tokens == 0:
            return 0.0
        return len(unique_tokens) / total_tokens

    def get_boredom_level(self) -> float:
        idle = time.time() - self.last_interaction_time

        # Boredom ramps up meaningfully within the first few minutes
        if idle < 30:
            base = idle / 300          # 0→0.1 over 30s
        elif idle < 90:
            base = 0.1 + (idle - 30) / 150   # 0.1→0.5 over next 60s
        elif idle < 180:
            base = 0.5 + (idle - 90) / 180  # 0.5→1.0 over next 90s
        else:
            base = 1.0

        # Boredom scales with idle time and environmental entropy
        return base

_inst: ProactiveCommunicationManager | None = None


def get_proactive_comm() -> ProactiveCommunicationManager:
    global _inst
    if _inst is None:
        _inst = ProactiveCommunicationManager()
    return _inst
