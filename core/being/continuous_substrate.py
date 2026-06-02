from __future__ import annotations

import hashlib
import math
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContinuousFieldPacket:
    tick: int
    timestamp: float
    monotonic_time: float
    state: tuple[float, ...]
    elapsed_s: float
    private_residue_hash: str


class ContinuousSelfField:
    """Small continuous-time self/world field.

    The field is intentionally dependency-light. It gives Aura a persistent
    temporal substrate that keeps evolving between foreground turns and can be
    sampled by Cortex, Will, tests, and proof artifacts.
    """

    def __init__(self, *, dim: int = 32) -> None:
        self.dim = max(8, int(dim))
        self._state = [0.0 for _ in range(self.dim)]
        self._lock = threading.RLock()
        self._last_step = time.monotonic()
        self._created = self._last_step
        self._tick = 0
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self, *, hz: float = 20.0) -> None:
        if self._running:
            return
        interval = 1.0 / max(1.0, min(100.0, float(hz)))
        self._running = True

        def _loop() -> None:
            while self._running:
                self.step()
                time.sleep(interval)

        self._thread = threading.Thread(target=_loop, name="AuraNowContinuousSelfField", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def step(
        self,
        telemetry: dict[str, float] | None = None,
        cognitive_projection: tuple[float, ...] | None = None,
        *,
        dt: float | None = None,
    ) -> ContinuousFieldPacket:
        now = time.monotonic()
        with self._lock:
            elapsed = max(0.001, float(dt) if dt is not None else now - self._last_step)
            self._last_step = now
            telemetry = telemetry or {}
            projection = cognitive_projection or ()
            drive = (
                float(telemetry.get("body_pressure", 0.0) or 0.0)
                + float(telemetry.get("prediction_error", 0.0) or 0.0)
                + float(telemetry.get("attention_salience", 0.0) or 0.0)
            ) / 3.0
            new_state: list[float] = []
            for idx, value in enumerate(self._state):
                neighbor = self._state[idx - 1] if idx else self._state[-1]
                projected = projection[idx % len(projection)] if projection else 0.0
                oscillator = math.sin((self._tick * 0.07) + idx) * 0.015
                dx = (-0.12 * value) + (0.03 * neighbor) + (0.08 * drive) + (0.04 * projected) + oscillator
                new_state.append(max(-1.0, min(1.0, value + dx * elapsed)))
            self._state = new_state
            self._tick += 1
            return self.read(elapsed_s=elapsed)

    def read(self, *, elapsed_s: float = 0.0) -> ContinuousFieldPacket:
        with self._lock:
            state = tuple(round(float(value), 6) for value in self._state)
            residue_payload = repr((self._tick, state[:8], round(time.monotonic() - self._created, 3))).encode("utf-8")
            return ContinuousFieldPacket(
                tick=self._tick,
                timestamp=time.time(),
                monotonic_time=time.monotonic(),
                state=state,
                elapsed_s=max(0.0, float(elapsed_s)),
                private_residue_hash=hashlib.sha256(residue_payload).hexdigest(),
            )

    def get_phenomenal_packet(self) -> dict[str, Any]:
        packet = self.read()
        norm = math.sqrt(sum(value * value for value in packet.state))
        return {
            "tick": packet.tick,
            "field_norm": round(norm, 4),
            "elapsed_since_start_s": round(packet.monotonic_time - self._created, 4),
            "private_residue_hash": packet.private_residue_hash,
        }
