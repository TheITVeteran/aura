"""Bounded sensory routing from physical channels into Aura's cognition.

Reality Reach owns physical declarations and metrology.  This module turns
those readings into a controllable exteroceptive stream without allowing a
fast sensor, a large installation, or a failing device to monopolize the event
loop or Aura's attention.  Raw device payloads remain with their adapters;
only bounded scalar claims and provenance cross the cognitive boundary.
"""

from __future__ import annotations

import asyncio
import math
import re
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from core.perception.multimodal_sync import (
    Calibration,
    Modality,
    PerceptualClaim,
    PerceptualEvent,
    PrivacyClass,
    PrivacyPolicy,
)
from core.reality_reach.contracts import ChannelDeclaration, ChannelKind
from core.reality_reach.live import (
    ChannelReading,
    ReadingStatus,
    RealityReachService,
)
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_SELECTOR = re.compile(r"^(?:\*|[a-z0-9][a-z0-9_.:-]{0,127}\*?)$")
_AVAILABLE = frozenset({ReadingStatus.AVAILABLE, ReadingStatus.SIMULATED})


def _finite(value: float, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, _finite(value, name="value")))


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


@dataclass(frozen=True, slots=True)
class ObservationSubscription:
    """One attention policy selected by Aura or a trusted runtime owner.

    ``selector`` is an exact channel id, a prefix ending in ``*``, or ``*``.
    The most specific active selector wins.  Expiring focus subscriptions let
    cognition temporarily inspect a device at higher cadence without changing
    the conservative background budget.
    """

    subscription_id: str
    selector: str = "*"
    max_rate_hz: float = 0.5
    min_delta: float = 0.0
    min_salience: float = 0.12
    enabled: bool = True
    expires_monotonic: float | None = None
    retain_for_memory: bool = False

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.subscription_id):
            raise ValueError("subscription_id must be a canonical identifier")
        if not _SELECTOR.fullmatch(self.selector):
            raise ValueError("selector must be an exact channel, prefix*, or *")
        rate = _finite(self.max_rate_hz, name="max_rate_hz")
        delta = _finite(self.min_delta, name="min_delta")
        salience = _clamp01(self.min_salience)
        if not 0.01 <= rate <= 20.0:
            raise ValueError("max_rate_hz must lie inside [0.01, 20]")
        if delta < 0.0:
            raise ValueError("min_delta must be non-negative")
        if self.expires_monotonic is not None:
            expires = _finite(self.expires_monotonic, name="expires_monotonic")
            if expires <= 0.0:
                raise ValueError("expires_monotonic must be positive")
            object.__setattr__(self, "expires_monotonic", expires)
        object.__setattr__(self, "max_rate_hz", rate)
        object.__setattr__(self, "min_delta", delta)
        object.__setattr__(self, "min_salience", salience)

    def matches(self, channel_id: str, *, now: float) -> bool:
        if not self.enabled:
            return False
        if self.expires_monotonic is not None and now >= self.expires_monotonic:
            return False
        if self.selector == "*":
            return True
        if self.selector.endswith("*"):
            return channel_id.startswith(self.selector[:-1])
        return channel_id == self.selector

    @property
    def specificity(self) -> tuple[int, int]:
        if self.selector == "*":
            return (0, 0)
        if self.selector.endswith("*"):
            return (1, len(self.selector))
        return (2, len(self.selector))

    def to_dict(self) -> dict[str, Any]:
        return {
            "subscription_id": self.subscription_id,
            "selector": self.selector,
            "max_rate_hz": self.max_rate_hz,
            "min_delta": self.min_delta,
            "min_salience": self.min_salience,
            "enabled": self.enabled,
            "expires_monotonic": self.expires_monotonic,
            "retain_for_memory": self.retain_for_memory,
        }


@dataclass(frozen=True, slots=True)
class RealityObservation:
    observation_id: str
    adapter_id: str
    declaration: ChannelDeclaration
    reading: ChannelReading
    salience: float
    received_at_ns: int
    received_monotonic_ns: int
    subscription_id: str

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.observation_id):
            raise ValueError("observation_id must be a canonical identifier")
        if not _IDENTIFIER.fullmatch(self.adapter_id):
            raise ValueError("adapter_id must be a canonical identifier")
        if self.declaration.kind != ChannelKind.SENSOR:
            raise ValueError("only sensor declarations can become observations")
        if self.reading.channel_id != self.declaration.channel_id:
            raise ValueError("reading and declaration channel identities differ")
        if self.reading.unit != self.declaration.unit:
            raise ValueError("reading and declaration units differ")
        if self.received_at_ns <= 0 or self.received_monotonic_ns <= 0:
            raise ValueError("observation receipt clocks must be positive")
        object.__setattr__(self, "salience", _clamp01(self.salience))

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "adapter_id": self.adapter_id,
            "channel_id": self.declaration.channel_id,
            "observable": self.declaration.observable,
            "unit": self.declaration.unit,
            "reading": self.reading.to_dict(),
            "salience": self.salience,
            "received_at_ns": self.received_at_ns,
            "received_monotonic_ns": self.received_monotonic_ns,
            "subscription_id": self.subscription_id,
        }


@dataclass(frozen=True, slots=True)
class ObservationReceipt:
    observation_id: str
    accepted: bool
    reason: str
    queue_depth: int
    salience: float
    evicted_observation_id: str = ""


@dataclass(slots=True)
class _ChannelState:
    status: ReadingStatus
    value: float | None
    accepted_monotonic: float
    reading_sha256: str


@dataclass(slots=True)
class _Sampler:
    adapter_id: str
    declarations: dict[str, ChannelDeclaration]
    callback: Callable[[], Awaitable[ChannelReading | tuple[ChannelReading, ...]]]
    sample_rate_hz: float
    next_due_monotonic: float = 0.0


class RealityObservationRouter:
    """Backpressured, salience-aware physical observation service."""

    def __init__(
        self,
        service: RealityReachService,
        *,
        queue_limit: int = 256,
        poll_interval_s: float = 2.0,
        sampler_timeout_s: float = 8.5,
        max_delivery_rate_hz: float = 20.0,
    ) -> None:
        if not isinstance(service, RealityReachService):
            raise TypeError("service must be a RealityReachService")
        if not 8 <= int(queue_limit) <= 8192:
            raise ValueError("queue_limit must lie inside [8, 8192]")
        self._service = service
        self._queue_limit = int(queue_limit)
        self._poll_interval_s = max(0.1, min(float(poll_interval_s), 60.0))
        self._sampler_timeout_s = max(0.1, min(float(sampler_timeout_s), 30.0))
        self._max_delivery_rate_hz = max(
            0.5,
            min(_finite(max_delivery_rate_hz, name="max_delivery_rate_hz"), 100.0),
        )
        self._queue: deque[RealityObservation] = deque()
        self._latest: dict[str, RealityObservation] = {}
        self._channel_state: dict[str, _ChannelState] = {}
        self._subscriptions: dict[str, ObservationSubscription] = {
            "reality.default": ObservationSubscription(
                subscription_id="reality.default",
                selector="*",
                max_rate_hz=0.5,
                min_delta=0.0,
                min_salience=0.12,
            )
        }
        self._samplers: dict[str, _Sampler] = {}
        self._lock = threading.RLock()
        self._wake = asyncio.Event()
        self._worker_task: asyncio.Task[Any] | None = None
        self._poll_task: asyncio.Task[Any] | None = None
        self._running = False
        self._sequence = 0
        self._accepted = 0
        self._delivered = 0
        self._deduplicated = 0
        self._rate_limited = 0
        self._below_salience = 0
        self._overflow_drops = 0
        self._coalesced = 0
        self._delivery_failures = 0
        self._sampler_failures = 0
        self._last_delivery_ns = 0

    def configure_subscription(self, subscription: ObservationSubscription) -> None:
        if not isinstance(subscription, ObservationSubscription):
            raise TypeError("subscription must be an ObservationSubscription")
        with self._lock:
            self._subscriptions[subscription.subscription_id] = subscription

    def remove_subscription(self, subscription_id: str) -> None:
        if subscription_id == "reality.default":
            raise ValueError("the bounded default subscription cannot be removed")
        with self._lock:
            if self._subscriptions.pop(subscription_id, None) is None:
                raise LookupError(f"unknown observation subscription: {subscription_id}")

    def focus(
        self,
        channel_or_prefix: str,
        *,
        duration_s: float = 30.0,
        max_rate_hz: float = 4.0,
        min_salience: float = 0.0,
    ) -> ObservationSubscription:
        selector = str(channel_or_prefix or "").strip().lower()
        duration = max(0.1, min(_finite(duration_s, name="duration_s"), 3600.0))
        digest = _digest({"selector": selector, "at": time.monotonic_ns()})
        subscription = ObservationSubscription(
            subscription_id=f"reality.focus.{digest.removeprefix('sha256:')[:24]}",
            selector=selector,
            max_rate_hz=max_rate_hz,
            min_salience=min_salience,
            expires_monotonic=time.monotonic() + duration,
        )
        self.configure_subscription(subscription)
        return subscription

    def pause(self) -> None:
        with self._lock:
            self._subscriptions = {
                key: ObservationSubscription(
                    **{**value.to_dict(), "enabled": False}
                )
                for key, value in self._subscriptions.items()
            }

    def resume(self) -> None:
        with self._lock:
            self._subscriptions = {
                key: ObservationSubscription(
                    **{**value.to_dict(), "enabled": True}
                )
                for key, value in self._subscriptions.items()
            }

    def register_sampler(self, adapter: Any) -> None:
        adapter_id = str(getattr(adapter, "adapter_id", "") or "")
        if not _IDENTIFIER.fullmatch(adapter_id):
            raise ValueError("sampled adapter requires a canonical adapter_id")
        callback = getattr(adapter, "refresh_readback", None)
        declarations_fn = getattr(adapter, "declarations", None)
        if not callable(callback) or not asyncio.iscoroutinefunction(callback):
            raise TypeError("sampled adapter requires async refresh_readback")
        if not callable(declarations_fn):
            raise TypeError("sampled adapter requires declarations")
        declarations = {
            item.channel_id: item
            for item in tuple(declarations_fn())
            if isinstance(item, ChannelDeclaration) and item.kind == ChannelKind.SENSOR
        }
        if not declarations:
            raise ValueError("sampled adapter declares no sensor channel")
        rate = max(
            0.01,
            min(20.0, max(item.sample_rate_hz for item in declarations.values())),
        )
        with self._lock:
            existing = self._samplers.get(adapter_id)
            if existing is not None and existing.callback != callback:
                raise ValueError(f"sampler already registered: {adapter_id}")
            self._samplers[adapter_id] = _Sampler(
                adapter_id=adapter_id,
                declarations=declarations,
                callback=callback,
                sample_rate_hz=rate,
            )

    def unregister_sampler(self, adapter_id: str) -> None:
        with self._lock:
            if self._samplers.pop(adapter_id, None) is None:
                raise LookupError(f"sampler is not registered: {adapter_id}")

    async def submit(
        self,
        declaration: ChannelDeclaration,
        reading: ChannelReading,
        *,
        adapter_id: str,
    ) -> ObservationReceipt:
        if declaration.kind != ChannelKind.SENSOR:
            raise ValueError("only sensor declarations can be submitted")
        if reading.channel_id != declaration.channel_id or reading.unit != declaration.unit:
            raise ValueError("reading differs from its declaration")
        now = time.monotonic()
        policy = self._policy_for(declaration.channel_id, now=now)
        if policy is None:
            return ObservationReceipt("", False, "not_subscribed", len(self._queue), 0.0)
        reading_sha256 = reading.sha256
        with self._lock:
            previous = self._channel_state.get(declaration.channel_id)
        if previous is not None and previous.reading_sha256 == reading_sha256:
            self._deduplicated += 1
            return ObservationReceipt("", False, "duplicate", len(self._queue), 0.0)
        salience = self._salience(declaration, reading, previous)
        if previous is not None:
            interval = 1.0 / policy.max_rate_hz
            if now - previous.accepted_monotonic < interval and previous.status == reading.status:
                self._rate_limited += 1
                return ObservationReceipt("", False, "rate_limited", len(self._queue), salience)
            if (
                previous.value is not None
                and reading.value is not None
                and abs(reading.value - previous.value) < policy.min_delta
                and previous.status == reading.status
            ):
                self._deduplicated += 1
                return ObservationReceipt("", False, "below_min_delta", len(self._queue), salience)
        if salience < policy.min_salience:
            self._below_salience += 1
            return ObservationReceipt("", False, "below_salience", len(self._queue), salience)
        received_at_ns = max(1, time.time_ns())
        received_monotonic_ns = max(1, time.monotonic_ns())
        self._sequence += 1
        digest = _digest(
            {
                "adapter_id": adapter_id,
                "channel_id": declaration.channel_id,
                "reading_sha256": reading_sha256,
                "sequence": self._sequence,
                "received_monotonic_ns": received_monotonic_ns,
            }
        )
        observation = RealityObservation(
            observation_id=f"reality.obs.{digest.removeprefix('sha256:')[:32]}",
            adapter_id=adapter_id,
            declaration=declaration,
            reading=reading,
            salience=salience,
            received_at_ns=received_at_ns,
            received_monotonic_ns=received_monotonic_ns,
            subscription_id=policy.subscription_id,
        )
        evicted = ""
        with self._lock:
            for pending_index, pending in enumerate(self._queue):
                if pending.declaration.channel_id != declaration.channel_id:
                    continue
                evicted = pending.observation_id
                del self._queue[pending_index]
                self._coalesced += 1
                break
            if len(self._queue) >= self._queue_limit:
                least_index, least = min(
                    enumerate(self._queue),
                    key=lambda item: (item[1].salience, item[1].received_monotonic_ns),
                )
                if least.salience >= observation.salience:
                    self._overflow_drops += 1
                    return ObservationReceipt(
                        observation.observation_id,
                        False,
                        "queue_full_lower_priority",
                        len(self._queue),
                        salience,
                    )
                evicted = least.observation_id
                del self._queue[least_index]
                self._overflow_drops += 1
            self._queue.append(observation)
            self._latest[declaration.channel_id] = observation
            self._channel_state[declaration.channel_id] = _ChannelState(
                status=reading.status,
                value=reading.value,
                accepted_monotonic=now,
                reading_sha256=reading_sha256,
            )
            depth = len(self._queue)
            self._accepted += 1
        self._wake.set()
        return ObservationReceipt(
            observation.observation_id,
            True,
            "accepted",
            depth,
            salience,
            evicted_observation_id=evicted,
        )

    async def poll_once(self) -> int:
        readings = await asyncio.to_thread(self._service.refresh)
        declarations = {
            item.channel_id: item
            for item in self._service.declarations()
            if item.kind == ChannelKind.SENSOR
        }
        ownership = self._service.adapter_channels()
        owner_by_channel = {
            channel_id: adapter_id
            for adapter_id, channel_ids in ownership.items()
            for channel_id in channel_ids
        }
        with self._lock:
            sampled_adapter_ids = set(self._samplers)
        accepted = 0
        for channel_id, declaration in declarations.items():
            if owner_by_channel.get(channel_id) in sampled_adapter_ids:
                # Async sampled adapters refresh below. Submitting the cached
                # service reading first would rate-limit their fresh readback.
                continue
            reading = readings.get(channel_id)
            if reading is None:
                continue
            receipt = await self.submit(
                declaration,
                reading,
                adapter_id=owner_by_channel.get(channel_id, "reality.unknown"),
            )
            accepted += int(receipt.accepted)
        accepted += await self._poll_samplers()
        return accepted

    async def _poll_samplers(self) -> int:
        now = time.monotonic()
        with self._lock:
            due = [
                sampler
                for sampler in self._samplers.values()
                if now >= sampler.next_due_monotonic
            ]
            for sampler in due:
                sampler.next_due_monotonic = now + (1.0 / sampler.sample_rate_hz)
        if not due:
            return 0
        semaphore = asyncio.Semaphore(8)

        async def _sample(sampler: _Sampler) -> int:
            async with semaphore:
                try:
                    result = await asyncio.wait_for(
                        sampler.callback(),
                        timeout=self._sampler_timeout_s,
                    )
                    readings = result if isinstance(result, tuple) else (result,)
                    accepted = 0
                    for reading in readings:
                        if not isinstance(reading, ChannelReading):
                            raise TypeError("sampler returned a non-reading")
                        declaration = sampler.declarations.get(reading.channel_id)
                        if declaration is None:
                            raise ValueError("sampler returned an undeclared channel")
                        receipt = await self.submit(
                            declaration,
                            reading,
                            adapter_id=sampler.adapter_id,
                        )
                        accepted += int(receipt.accepted)
                    return accepted
                except asyncio.CancelledError:
                    raise
                except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                    self._sampler_failures += 1
                    record_degradation(
                        "reality_observation_router.sampler",
                        exc,
                        action=f"retained other physical samplers after {sampler.adapter_id} failed",
                    )
                    return 0

        return sum(await asyncio.gather(*(_sample(item) for item in due)))

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker_task = get_task_tracker().create_task(
            self._worker_loop(),
            name="RealityObservationRouter",
        )
        self._poll_task = get_task_tracker().create_task(
            self._poll_loop(),
            name="RealityObservationPoll",
        )

    async def stop(self) -> None:
        self._running = False
        self._wake.set()
        for task in (self._poll_task, self._worker_task):
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._poll_task = None
        self._worker_task = None

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                record_degradation(
                    "reality_observation_router.poll",
                    exc,
                    action="continued bounded physical sensing after one inventory poll failed",
                )
            await asyncio.sleep(self._poll_interval_s)

    async def _worker_loop(self) -> None:
        next_delivery = time.monotonic()
        while self._running:
            await self._wake.wait()
            while self._running:
                with self._lock:
                    observation = self._queue.popleft() if self._queue else None
                    if not self._queue:
                        self._wake.clear()
                if observation is None:
                    break
                try:
                    now = time.monotonic()
                    if now < next_delivery:
                        await asyncio.sleep(next_delivery - now)
                    next_delivery = max(next_delivery, time.monotonic()) + (
                        1.0 / self._max_delivery_rate_hz
                    )
                    await self._deliver(observation)
                    self._delivered += 1
                    self._last_delivery_ns = max(1, time.time_ns())
                except asyncio.CancelledError:
                    raise
                except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                    self._delivery_failures += 1
                    record_degradation(
                        "reality_observation_router.delivery",
                        exc,
                        action="retained subsequent physical observations after one cognitive delivery failed",
                    )

    async def _deliver(self, observation: RealityObservation) -> None:
        from core.container import ServiceContainer

        synchronizer = ServiceContainer.get("multimodal_synchronizer", default=None)
        if synchronizer is not None and callable(getattr(synchronizer, "ingest", None)):
            reading = observation.reading
            claims: list[PerceptualClaim] = [
                PerceptualClaim(
                    key=f"{observation.declaration.channel_id}.status",
                    value=reading.status.value,
                    confidence=1.0,
                )
            ]
            if reading.value is not None:
                claims.append(
                    PerceptualClaim(
                        key=observation.declaration.channel_id,
                        value=reading.value,
                        confidence=self._confidence(observation),
                    )
                )
            calibration_id = (
                observation.declaration.calibration_id
                or observation.declaration.reference_id
                or f"{observation.adapter_id}.uncalibrated"
            )[:160]
            event = PerceptualEvent(
                event_id=observation.observation_id,
                modality=Modality.DEVICE,
                source=f"reality:{observation.adapter_id}"[:160],
                sequence=max(0, reading.sequence or self._sequence),
                observed_at=max(0.001, reading.captured_at_ns / 1_000_000_000),
                observed_monotonic_ns=observation.received_monotonic_ns,
                summary=(
                    f"{observation.declaration.observable} "
                    f"{reading.status.value}"
                    + (f" {reading.value:g} {reading.unit}" if reading.value is not None else "")
                )[:320],
                confidence=self._confidence(observation),
                claims=tuple(claims),
                calibration=Calibration(
                    calibration_id=calibration_id,
                    status=(
                        "valid"
                        if observation.declaration.coupling_validated
                        else "unknown"
                    ),
                    reliability=self._confidence(observation),
                ),
                provenance=(
                    observation.reading.sha256,
                    observation.declaration.sha256,
                    observation.adapter_id,
                ),
                privacy=PrivacyPolicy(
                    classification=PrivacyClass.PRIVATE,
                    retention="ephemeral",
                    consent_scope="reality_reach.sensor_summary",
                    redacted=True,
                    raw_retained=False,
                ),
                quality_flags=(
                    f"status:{reading.status.value}",
                    f"evidence:{observation.declaration.evidence_level.value}",
                ),
            )
            synchronizer.ingest(event)

        advanced = ServiceContainer.get("advanced_cognition", default=None)
        if advanced is None:
            from core.advanced_cognition.integration import (
                get_advanced_cognition_runtime,
            )

            advanced = get_advanced_cognition_runtime()
        observe_state = getattr(advanced, "observe_state", None)
        if callable(observe_state):
            await asyncio.to_thread(
                observe_state,
                "physical_environment",
                {
                    "adapter_id": observation.adapter_id,
                    "channel_id": observation.declaration.channel_id,
                    "observable": observation.declaration.observable,
                    "value": observation.reading.value,
                    "unit": observation.reading.unit,
                    "status": observation.reading.status.value,
                    "uncertainty": observation.reading.uncertainty,
                    "salience": observation.salience,
                    "evidence_level": observation.declaration.evidence_level.value,
                    "reality_layers": [
                        layer.value for layer in observation.declaration.reality_layers
                    ],
                    "observation_sha256": observation.reading.sha256,
                },
                source=f"reality:{observation.adapter_id}",
                confidence=self._confidence(observation),
            )

    def latest(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                channel_id: observation.to_dict()
                for channel_id, observation in self._latest.items()
            }

    def subscriptions(self) -> tuple[ObservationSubscription, ...]:
        now = time.monotonic()
        with self._lock:
            return tuple(
                sorted(
                    (
                        item
                        for item in self._subscriptions.values()
                        if item.expires_monotonic is None or item.expires_monotonic > now
                    ),
                    key=lambda item: item.subscription_id,
                )
            )

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "alive": self.is_alive(),
                "ready": self.is_ready(),
                "queue_depth": len(self._queue),
                "queue_limit": self._queue_limit,
                "latest_channels": len(self._latest),
                "subscriptions": len(self.subscriptions()),
                "samplers": len(self._samplers),
                "accepted": self._accepted,
                "delivered": self._delivered,
                "deduplicated": self._deduplicated,
                "rate_limited": self._rate_limited,
                "below_salience": self._below_salience,
                "overflow_drops": self._overflow_drops,
                "coalesced": self._coalesced,
                "delivery_failures": self._delivery_failures,
                "sampler_failures": self._sampler_failures,
                "last_delivery_ns": self._last_delivery_ns,
            }

    def get_status(self) -> dict[str, Any]:
        return self.status()

    def is_alive(self) -> bool:
        return bool(
            self._running
            and self._worker_task is not None
            and not self._worker_task.done()
            and self._poll_task is not None
            and not self._poll_task.done()
        )

    def is_ready(self) -> bool:
        return self.is_alive() and any(item.enabled for item in self.subscriptions())

    def _policy_for(
        self,
        channel_id: str,
        *,
        now: float,
    ) -> ObservationSubscription | None:
        with self._lock:
            matches = [
                item
                for item in self._subscriptions.values()
                if item.matches(channel_id, now=now)
            ]
        if not matches:
            return None
        return max(matches, key=lambda item: (item.specificity, item.max_rate_hz))

    @staticmethod
    def _confidence(observation: RealityObservation) -> float:
        reading = observation.reading
        if reading.status not in _AVAILABLE:
            return 0.0
        domain_width = max(
            1e-12,
            observation.declaration.domain.maximum
            - observation.declaration.domain.minimum,
        )
        uncertainty = max(0.0, float(reading.uncertainty or 0.0))
        uncertainty_penalty = min(0.75, uncertainty / domain_width)
        evidence = 0.4 + 0.08 * observation.declaration.evidence_level.rank
        if not observation.declaration.coupling_validated:
            evidence *= 0.7
        return _clamp01(evidence * (1.0 - uncertainty_penalty))

    @staticmethod
    def _salience(
        declaration: ChannelDeclaration,
        reading: ChannelReading,
        previous: _ChannelState | None,
    ) -> float:
        if previous is None:
            base = 0.42 if reading.status in _AVAILABLE else 0.65
        elif previous.status != reading.status:
            base = 0.92
        elif reading.value is None or previous.value is None:
            base = 0.2
        else:
            width = max(1e-12, declaration.domain.maximum - declaration.domain.minimum)
            resolution = max(0.0, declaration.resolution)
            meaningful = max(width * 0.01, resolution, 1e-12)
            normalized = abs(reading.value - previous.value) / meaningful
            base = min(0.85, 0.08 + 0.18 * normalized)
        tags = set(declaration.compliance_tags)
        if tags & {"safety_critical", "life_safety", "interlock", "alarm"}:
            base = max(base, 0.95)
        if reading.status in {
            ReadingStatus.DEGRADED,
            ReadingStatus.PERMISSION_DENIED,
            ReadingStatus.UNAVAILABLE,
        }:
            base = max(base, 0.75)
        return _clamp01(base)


__all__ = [
    "ObservationReceipt",
    "ObservationSubscription",
    "RealityObservation",
    "RealityObservationRouter",
]
