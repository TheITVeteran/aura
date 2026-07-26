"""core/brain/ontology_genesis.py — Aura 3.0: Experimental Containment Mode
=======================================================================
Implements the "Hibernation Mode" protocol for high-compute autonomous
scientific discovery.

Refactored for ZENITH Protocol efficiency:
  - Background discovery loop is DISABLED by default.
  - Must be explicitly triggered by user or deep-research soul drive.
  - Mandatory resource_anxiety abort if system load exceeds thresholds.

**Honest status: the discovery step is NOT IMPLEMENTED.** CP126 c0a3c26e found
that the advertised "autonomous formation of cognitive laws" was a loop that
logged, slept sixty seconds, and reached a comment saying the real logic went
there. It produced no candidate, experiment, verifier result or artifact, and
never wrote to its own discovery log — while logging "Discovery loop active"
and reporting ``active: true``.

Rather than dress that up, this module now says so. ``DISCOVERY_IMPLEMENTED``
is False, ``start_discovery`` refuses with ``not_implemented``, and the status
surface carries ``implemented: false`` so nothing downstream can read an idle
timer as research in progress. The admission checks around it — volition,
authority, resource pressure, supervision — are real and are fixed here, so
the day a discovery step is written it lands behind a working gate.

CP126 c0a3c26e / 72c94940 / d52e6a29 / 96cb9483 / 478804d4.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from core.runtime.errors import record_degradation
from core.runtime.numeric_safety import validated_unit
from core.runtime.service_registry import get_runtime_service, register_runtime_service
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.OntologyGenesis")

#: Whether a real discovery step exists. Flipping this to True without writing
#: one re-creates CP126 c0a3c26e.
DISCOVERY_IMPLEMENTED = False

#: Single source of truth for the abort boundary. CP126 96cb9483: status
#: advertised 0.2 while start_discovery used 0.5 and the loop used 0.3 or 0.6,
#: so an operator could not infer the real boundary from the service status.
ANXIETY_THRESHOLD_ADMISSION = 0.5
ANXIETY_THRESHOLD_RUNNING = 0.3
ANXIETY_THRESHOLD_RUNNING_HIGH_VOLITION = 0.6
HIGH_VOLITION_LEVEL = 3

#: Resource pressure attributed when telemetry is unavailable. CP126 d52e6a29:
#: this was 0.0 — the SAFEST possible value — so a missing homeostasis service
#: admitted high-compute work instead of deferring it for missing telemetry.
UNKNOWN_ANXIETY = 1.0

_GENESIS_ERRORS = (
    AttributeError,
    ImportError,
    RuntimeError,
    TypeError,
    ValueError,
)


class OntologyGenesisEngine:
    """
    Manages autonomous discovery of new cognitive laws and heuristics.

    ZENITH Protocol:
      - Default state is inert.
      - Throttled by resource anxiety.

    The discovery step itself is not implemented; see the module docstring.
    """

    def __init__(self) -> None:
        self._active = False
        self._genesis_task: Optional[asyncio.Task] = None
        self._discovery_log: List[str] = []
        self._last_refusal = ""
        self._started_at = 0.0

    # -- resource telemetry ----------------------------------------------
    def resource_anxiety(self) -> tuple[float, bool]:
        """Current resource pressure and whether it was actually measured."""
        try:
            homeostasis = get_runtime_service("homeostasis", default=None)
            if homeostasis is not None and hasattr(homeostasis, "anxiety"):
                raw = homeostasis.anxiety
                if callable(raw):
                    raw = raw()
                scalar = validated_unit(raw, name="anxiety", cautious_high=True)
                if not scalar.fault:
                    return float(scalar), True
                logger.debug("Homeostasis anxiety unusable: %s", scalar.fault)
        except _GENESIS_ERRORS as exc:
            logger.debug("Homeostasis probe failed: %s", exc)
        # Unknown pressure is treated as maximum: high-compute work is deferred
        # for missing telemetry rather than admitted on a convenient default.
        return UNKNOWN_ANXIETY, False

    def _get_resource_anxiety(self) -> float:
        """Backwards-compatible scalar accessor."""
        return self.resource_anxiety()[0]

    # -- admission --------------------------------------------------------
    async def start_discovery(
        self, mode: str = "manual", *, capability_token: str = ""
    ) -> bool:
        """Trigger a discovery cycle.

        Returns False and records the reason whenever admission is refused.
        """
        self._last_refusal = ""

        # CP126 c0a3c26e: there is nothing to start. Refusing here is the
        # honest answer; the alternative is an idle timer that reports itself
        # as active research.
        if not DISCOVERY_IMPLEMENTED:
            self._last_refusal = "not_implemented"
            logger.info(
                "OntologyGenesis: no discovery step is implemented; refusing to "
                "report an active discovery loop."
            )
            return False

        volition = self._volition_level()

        # CP126 72c94940: `mode` is a caller-supplied string, and passing
        # "deep_research" skipped the volition check entirely — no principal,
        # standing authority, signed plan, scope or compute lease. The mode
        # alone no longer grants anything; it must be attested.
        authorized_mode = mode == "deep_research" and self._mode_authorized(
            mode, capability_token
        )
        if not authorized_mode and volition < 1:
            self._last_refusal = "insufficient_volition"
            logger.info(
                "OntologyGenesis: refused — volition %d < 1 and deep_research was "
                "not attested.", volition,
            )
            return False

        anxiety, measured = self.resource_anxiety()
        if not measured:
            self._last_refusal = "resource_telemetry_unavailable"
            logger.warning(
                "OntologyGenesis: refused — resource telemetry unavailable, so "
                "high-compute work is deferred rather than admitted."
            )
            return False
        if anxiety >= ANXIETY_THRESHOLD_ADMISSION:
            self._last_refusal = f"resource_pressure={anxiety:.2f}"
            logger.warning(
                "OntologyGenesis: Abort. Resource pressure too high (%.2f >= %.2f).",
                anxiety, ANXIETY_THRESHOLD_ADMISSION,
            )
            return False

        if self._active:
            # CP126 478804d4: a task that died left _active True forever, so
            # this reported an already-active service that was not running.
            if self._genesis_task is not None and self._genesis_task.done():
                logger.warning(
                    "OntologyGenesis: previous discovery task is finished; clearing "
                    "the stale active flag before restarting."
                )
                self._active = False
            else:
                return True

        self._active = True
        self._started_at = time.time()
        self._genesis_task = get_task_tracker().create_task(
            self._discovery_loop(volition), name="ontology_genesis.discovery"
        )
        self._genesis_task.add_done_callback(self._on_task_done)
        logger.info(
            "OntologyGenesis: Hibernation ended (Volition=%d). Discovery loop active.",
            volition,
        )
        return True

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Clear the active flag however the task ended (CP126 478804d4)."""
        self._active = False
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            record_degradation(
                "ontology_genesis",
                exc,
                action="cleared the active flag after the discovery task failed",
                severity="error",
            )
            logger.error("OntologyGenesis discovery task failed: %s", exc)

    @staticmethod
    def _volition_level() -> int:
        kernel = get_runtime_service("aura_kernel", default=None)
        try:
            return int(getattr(kernel, "volition_level", 0) or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _mode_authorized(mode: str, capability_token: str) -> bool:
        """Whether a deep_research request carries real authority."""
        token = str(capability_token or "").strip()
        if not token:
            return False
        try:
            from core.agency.capability_token import get_token_store

            get_token_store().validate(
                token, domain="self_modification", action="deep_research"
            )
        except (PermissionError, *_GENESIS_ERRORS) as exc:
            logger.info("OntologyGenesis: deep_research token rejected (%s)", exc)
            return False
        return True

    # -- lifecycle --------------------------------------------------------
    async def stop_discovery(self) -> None:
        self._active = False
        task, self._genesis_task = self._genesis_task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError as _exc:
                logger.debug("Suppressed asyncio.CancelledError: %s", _exc)
        logger.info("OntologyGenesis: Returning to hibernation.")

    def running_threshold(self, volition: int) -> float:
        return (
            ANXIETY_THRESHOLD_RUNNING_HIGH_VOLITION
            if volition >= HIGH_VOLITION_LEVEL
            else ANXIETY_THRESHOLD_RUNNING
        )

    async def _discovery_loop(self, volition: int = 0) -> None:
        """Main loop for high-compute cognitive law formation.

        Unreachable while DISCOVERY_IMPLEMENTED is False. Kept because the
        admission and abort logic around the (absent) discovery step is real
        and tested; the step itself is what has to be written.
        """
        while self._active:
            # CP126 478804d4: volition was captured once at start, so a
            # revocation mid-run never tightened the boundary.
            current_volition = self._volition_level()
            threshold = self.running_threshold(current_volition)
            anxiety, measured = self.resource_anxiety()
            if not measured or anxiety >= threshold:
                logger.warning(
                    "OntologyGenesis: Emergency hibernation (anxiety=%.2f "
                    "threshold=%.2f measured=%s).", anxiety, threshold, measured,
                )
                self._active = False
                break

            logger.debug("OntologyGenesis: Processing candidate heuristics...")
            await asyncio.sleep(60)
            # A discovery step belongs here. It must produce a candidate, run
            # an experiment, obtain a verifier result, and append an entry to
            # self._discovery_log — anything less is the CP126 c0a3c26e defect.

    # -- status -----------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        anxiety, measured = self.resource_anxiety()
        volition = self._volition_level()
        return {
            "active": self._active and DISCOVERY_IMPLEMENTED,
            # CP126 c0a3c26e: the honest headline.
            "implemented": DISCOVERY_IMPLEMENTED,
            "discoveries": len(self._discovery_log),
            # CP126 96cb9483: the thresholds actually in force, all of them.
            "anxiety_threshold_admission": ANXIETY_THRESHOLD_ADMISSION,
            "anxiety_threshold_running": self.running_threshold(volition),
            "current_anxiety": anxiety,
            "anxiety_measured": measured,
            "volition_level": volition,
            "last_refusal": self._last_refusal,
            "task_done": self._genesis_task.done() if self._genesis_task else None,
            "uptime_s": round(time.time() - self._started_at, 3) if self._started_at else 0.0,
        }


def register_ontology_genesis(orchestrator: Optional[Any] = None) -> OntologyGenesisEngine:
    """Legacy registration helper expected by boot code."""
    engine = OntologyGenesisEngine()
    register_runtime_service("ontology_genesis", engine)
    if orchestrator is not None:
        try:
            setattr(orchestrator, "ontology_genesis", engine)
        except _GENESIS_ERRORS as _exc:
            record_degradation('ontology_genesis', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
    return engine
