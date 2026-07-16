"""
Mycelial Network v3.0 — Enterprise-Grade Unblockable Root System
================================================================

Inspired by Physarum polycephalum (slime mold), this module provides:

1. **HardwiredPathways**: Regex-based intent→skill mappings with parameter extraction.
   These are the "direct roots" — unblockable, priority-#1 connections that bypass
   the LLM reasoning loop entirely.

2. **Physarum Reinforcement**: Pathways strengthen on success, weaken on failure.
   Conductivity naturally converges to the most reliable routes.

3. **Hyphae Network**: General-purpose connections between subsystems with
   rooted_flow context managers for stall detection and emergency override.

4. **Autonomous Discovery**: After non-hardwired skill executions succeed,
   the network proposes new pathways (slime mold exploration).

5. **Introspection API**: Full topology reporting for UI visualization and
   health monitoring.

Architecture:
   User Input
       ↓
   MycelialNetwork.match_hardwired()  ← FIRST (Hardwired Shortcuts, zero latency)
       ↓ (if no match)
   IntentRouter.classify()            ← SECOND (LLM-based reasoning, slower)
"""

import ast
import asyncio
import logging
import math
import os
import re
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from itertools import islice
from pathlib import Path
from typing import Any, Callable, ClassVar, Coroutine, Dict, List, Optional, Tuple, TypeVar, Union

from pydantic import BaseModel, Field

from core.runtime.background_policy import foreground_only_runtime
from core.runtime.errors import record_degradation
from core.runtime.flags import FlagKind, declare
from core.utils.concurrency import run_io_bound
from core.utils.exceptions import capture_and_log

logger = logging.getLogger("Aura.Mycelium")

T = TypeVar("T")

_DEFAULT_INFRASTRUCTURE_SCAN_DIRS = (
    "core",
    "interface",
    "skills",
    "aura",
    "llm",
    "senses",
    "autonomy_engine",
    "cloud",
    "infrastructure",
    "integration",
    "memory",
    "orchestrator",
    "proof_kernel",
    "research",
    "security",
    "storage",
    "training",
    "utils",
)
_VAULT_CLOCK_SKEW_TOLERANCE_S = 1.0
_ALLOW_FOREGROUND_MAPPING_FLAG = declare(
    "AURA_ALLOW_FOREGROUND_INFRASTRUCTURE_MAPPING",
    kind=FlagKind.BOOL,
    default=False,
    description="Allow mycelial infrastructure mapping in foreground-only mode",
    owner="core.mycelium",
)
_FOREGROUND_MAPPING_QUIET_FLAG = declare(
    "AURA_FOREGROUND_INFRASTRUCTURE_MAPPING_QUIET_S",
    kind=FlagKind.FLOAT,
    default=180.0,
    description="Foreground-only startup quiet window before mycelial mapping",
    owner="core.mycelium",
)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class HardwiredPathway(BaseModel):
    """A direct, unblockable connection from an intent pattern to a skill."""
    pathway_id: str
    pattern: Any  # Union[str, re.Pattern]
    skill_name: str
    param_map: Dict[str, Union[int, str]] = Field(default_factory=dict)
    priority: float = 1.0
    source_file: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    confidence: float = 1.0
    activity_label: str = ""
    hit_count: int = 0
    miss_count: int = 0
    created_at: float = Field(default_factory=time.time)
    last_matched: float = Field(default_factory=time.monotonic)
    direct_response: Optional[str] = None  # Legacy non-user emergency response only
    color: str = "#4A90E2"                 # Default Aura Blue
    description: str = ""
    size: float = 1.0

    # Physarum thresholds
    REINFORCE_DELTA: ClassVar[float] = 0.05
    WEAKEN_DELTA: ClassVar[float] = 0.15
    PRUNE_THRESHOLD: ClassVar[float] = 0.2
    MAX_CONFIDENCE: ClassVar[float] = 1.0
    MIN_CONFIDENCE: ClassVar[float] = 0.05

    model_config = {"arbitrary_types_allowed": True}

    def reinforce(self, success: bool):
        """Physarum-inspired conductivity update."""
        if success:
            self.confidence = min(self.MAX_CONFIDENCE, self.confidence + self.REINFORCE_DELTA)
            self.hit_count += 1
        else:
            self.confidence = max(self.MIN_CONFIDENCE, self.confidence - self.WEAKEN_DELTA)
            self.miss_count += 1

    @property
    def is_weak(self) -> bool:
        return self.confidence < self.PRUNE_THRESHOLD

    @property
    def success_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Legacy helper. Use .model_dump() instead."""
        data = self.model_dump()
        # For UI compatibility: frontend expects 'id'
        data["id"] = self.pathway_id
        # Ensure regex pattern is stringified for JSON compatibility
        if "pattern" in data and not isinstance(data["pattern"], str):
            data["pattern"] = getattr(self.pattern, 'pattern', str(self.pattern))
        return data


# ---------------------------------------------------------------------------
# Hypha (General-Purpose Connection)
# ---------------------------------------------------------------------------

class Hypha(BaseModel):
    """A connection within the mycelial network with dynamic strength."""
    name: str
    source: str
    target: str
    priority: float = 1.0
    strength: float = 1.0
    created_at: float = Field(default_factory=time.monotonic)
    last_pulse: float = Field(default_factory=time.monotonic)
    pulse_count: int = 0
    active: bool = True
    is_physical: bool = False
    source_file: Optional[str] = None
    target_file: Optional[str] = None
    color: str = "#4A90E2"
    description: str = ""
    size: float = 1.0
    trace: List[str] = Field(default_factory=list)

    def pulse(self, success: bool = True):
        """Reinforce or prune the hypha based on successful transmission."""
        self.last_pulse = time.monotonic()
        self.pulse_count += 1
        if success:
            self.strength = min(10.0, self.strength + 0.5)
        else:
            self.strength = max(0.1, self.strength - 1.0)

    def refresh_heartbeat(self):
        """Refresh liveness without mutating the learned strength of the edge."""
        self.last_pulse = time.monotonic()

    @property
    def thickness(self) -> float:
        """Dynamic representation of hypha health/strength (BUG-037)."""
        return 0.5 + (self.strength * 0.1)

    def log(self, msg: str):
        self.trace.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        if len(self.trace) > 100:
            self.trace.pop(0)


class NeuralRoot(Hypha):
    """A specialized, sub-conductive hypha that binds directly to hardware.
    Used for pinning critical platform services (like Metal) to the network.
    """
    hardware_id: str = "metal_default"
    pinned: bool = True
    
    def subsurface_ping(self) -> bool:
        """Probe the underlying hardware; network ownership records the pulse."""
        from core.container import ServiceContainer
        platform = ServiceContainer.get("platform_root", default=None)
        if platform:
            return bool(platform.pulse())
        return False


class RootedFlowHandle:
    """Owner-backed flow view that cannot mutate a detached hypha snapshot."""

    def __init__(
        self,
        network: "MycelialNetwork",
        source: str,
        target: str,
        priority: float,
    ):
        self._network = network
        self._source = source
        self._target = target
        self._priority = priority
        self._hypha_id = f"{source}->{target}"
        self._error: Optional[BaseException] = None

    @property
    def failed(self) -> bool:
        return self._error is not None

    @property
    def error(self) -> Optional[BaseException]:
        return self._error

    def _mark_failed(self, error: BaseException) -> None:
        self._error = error

    def raise_for_status(self) -> None:
        """Re-raise an absorbed flow failure at the caller's owned boundary."""
        if self._error is not None:
            raise self._error

    def _snapshot(self) -> Hypha:
        return self._network._record_rooted_flow_event(
            self._source,
            self._target,
            priority=self._priority,
        )

    def log(self, message: str) -> None:
        self._network._record_rooted_flow_event(
            self._source,
            self._target,
            priority=self._priority,
            message=message,
        )

    def pulse(self, success: bool = True) -> None:
        self._network._record_rooted_flow_event(
            self._source,
            self._target,
            priority=self._priority,
            success=success,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._snapshot(), name)


# ---------------------------------------------------------------------------
# Mycelial Network (Singleton)
# ---------------------------------------------------------------------------

class MycelialNetwork:
    """The Unoverridable Root System."""

    _instance: ClassVar[Optional["MycelialNetwork"]] = None
    _lock: ClassVar[threading.RLock] = threading.RLock()
    _vault_io_lock: ClassVar[threading.Lock] = threading.Lock()
    _initialized: ClassVar[bool] = False

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MycelialNetwork, cls).__new__(cls)
            return cls._instance

    def __init__(self):
        if MycelialNetwork._initialized:
            return

        with MycelialNetwork._lock:
            if MycelialNetwork._initialized:
                return

            self._async_lock: Optional[asyncio.Lock] = None
            
            # Phase XXIII: Aegis Protection Flag
            object.__setattr__(self, "_aegis_locked", False)

            # --- Hardwired Pathways ---
            self.pathways: Dict[str, HardwiredPathway] = {}
            self._pathway_order: List[str] = []

            # --- General Hyphae ---
            self.hyphae: Dict[str, Hypha] = {}

            # --- Discovery Engine ---
            self._execution_log: List[Dict[str, Any]] = []
            self._discovery_candidates: Dict[str, int] = defaultdict(int)
            self._route_signal_log_state: Dict[str, Tuple[str, float, int]] = {}
            self._hypha_alert_times: Dict[str, float] = {}

            # --- Props ---
            self.ui_callback: Optional[Callable[[str], Coroutine]] = None
            self.mapped_files: Dict[str, Dict[str, Any]] = {}
            self.infrastructure_mapped: bool = False
            self._centrality: Dict[str, int] = {}
            self._critical_modules: List[str] = []
            self._cross_links: Dict[str, List[str]] = {}
            self._is_mapping: bool = False
            # Mapping lifecycle and topology data share one lock. Separate locks
            # previously allowed publication and shutdown to acquire them in
            # opposite orders and made a coherent graph generation impossible.
            self._mapping_lock = MycelialNetwork._lock
            self._mapping_thread: Optional[threading.Thread] = None
            self._mapping_admission_token: Optional[object] = None
            self._mapping_generation: int = 0
            self._topology_revision: int = 0
            self._topology_structure_revision: int = 0
            self._last_vault_sync_revision: Optional[int] = None
            self._last_vault_sync_at: Optional[float] = None
            self._last_vault_sync_lag_revisions: int = 0
            self._mapping_started_at: Optional[float] = None
            self._mapping_completed_at: Optional[float] = None
            self._mapping_last_error: Optional[str] = None
            self._created_at_monotonic = time.monotonic()
            self._deferred_mapping_reason: Optional[str] = None
            self._stop_event = threading.Event()
            self._topology_counts_cache: Dict[str, int] = {}
            self._topology_summary_cache: Dict[str, int] = {}
            
            # Legacy compat
            self.direct_roots: Dict[str, str] = {}
            
            # Reflex Core (SOMA)
            try:
                from core.soma.reflex_core import HardenedReflexCore
                self.reflex = HardenedReflexCore()
            except ImportError:
                self.reflex = None

            # --- Platform Binding ---
            self._neural_roots: List[NeuralRoot] = []

            self._publish_topology_read_models_locked()
            
            MycelialNetwork._initialized = True
            object.__setattr__(self, "_aegis_locked", True)
            self._setup_default_pathways()
            
            # Phase 27: Rooting Hardware Voice
            self.establish_neural_root("voice_presence", hardware_id="macos_say")
            
            logger.info("🍄 [MYCELIUM] Network Online v4.0 (Hardened) — Enterprise Grade.")

    def _publish_topology_read_models_locked(self) -> None:
        endpoints = {
            endpoint
            for hypha in self.hyphae.values()
            for endpoint in (hypha.source, hypha.target)
            if endpoint
        }
        endpoints.update(self.mapped_files)
        annotated_pathways = sum(
            1 for pathway in self.pathways.values() if pathway.source_file
        )
        self._topology_counts_cache = {
            "pathways": len(self.pathways),
            "hyphae": len(self.hyphae),
            "mapped_files": len(self.mapped_files),
            "mapping_generation": self._mapping_generation,
        }
        self._topology_summary_cache = {
            "nodes": len(endpoints) + len(self.pathways),
            "links": len(self.hyphae) + annotated_pathways,
            "pathways": len(self.pathways),
            "mapping_generation": self._mapping_generation,
        }

    def _mark_topology_mutated_locked(self, *, structure_changed: bool = False) -> None:
        self._topology_revision += 1
        if structure_changed:
            self._topology_structure_revision += 1
            self._publish_topology_read_models_locked()

    def _setup_default_pathways(self):
        """Register action routes; conversation remains owned by CognitiveEngine."""
        self.register_pathway(
            "direct_web_search",
            r"(?:search (?:the web )?for|look up|google|find info on)\s+(.+)",
            "search_web",
            priority=1.5,
            activity_label="🔍 Searching the Intelligence Web"
        )
        self.register_pathway(
            "direct_self_repair",
            r"(?:run a self-diag|diagnose yourself|system check|repair yourself|fix system)",
            "self_repair",
            priority=1.5,
            activity_label="🧬 Running Self-Diagnostics"
        )
    def __setattr__(self, name: str, value: Any) -> None:
        """Pillar 1: Singleton True-Lock (Memory Protection).
        
        Prevents rogue reassignment of core structures. Once booted,
        'pathways' and 'hyphae' dictionaries cannot be replaced.
        """
        # Allow initialization to proceed naturally
        if not getattr(self, "_aegis_locked", False):
            super().__setattr__(name, value)
            return
            
        protected_attrs = {"pathways", "hyphae", "_pathway_order"}
        
        if name in protected_attrs:
            logger.critical("🛡️ AEGIS: Unauthorized attempt to overwrite %s!", name)
            raise PermissionError(f"Aegis True-Lock: Cannot overwrite core Mycelial attribute '{name}'")
            
        super().__setattr__(name, value)

    def _active_owner_locked(self) -> Optional["MycelialNetwork"]:
        """Resolve stale references to the one currently published singleton."""
        current = MycelialNetwork._instance
        stop_event = getattr(current, "_stop_event", None)
        if current is None or stop_event is None or stop_event.is_set():
            return None
        return current

    def _active_owner(self) -> Optional["MycelialNetwork"]:
        with MycelialNetwork._lock:
            return self._active_owner_locked()


    def setup(self, *, force: bool = False) -> bool:
        """Schedule the single owned infrastructure map when policy permits."""
        owner = self._active_owner()
        if owner is None:
            return False
        if owner is not self:
            return owner.setup(force=force)
        with self._mapping_lock:
            owner = self._active_owner_locked()
            if owner is None:
                return False
            if owner is not self:
                return owner.setup(force=force)
            if not force and self._foreground_mapping_deferred():
                return False
            thread = self._mapping_thread
            if self._is_mapping or (
                thread is not None and thread.is_alive()
            ) or (self.infrastructure_mapped and not force):
                return False

            from core.config import config

            mapping_base = config.paths.base_dir
            logger.info(
                "🍄 [MYCELIUM] Scheduling infrastructure mapping at: %s",
                mapping_base,
            )
            admission_token = object()
            self._mapping_admission_token = admission_token
            self._is_mapping = True
            self._mapping_started_at = time.time()
            self._mapping_last_error = None
            self._deferred_mapping_reason = None
            try:
                thread = threading.Thread(
                    target=self._mapping_worker,
                    args=(str(mapping_base),),
                    kwargs={"force": force, "_admission_token": admission_token},
                    daemon=True,
                    name="MyceliumInfrastructureMap",
                )
                self._mapping_thread = thread
                thread.start()
            except Exception:  # noqa: BLE001 - restore admission state for any start failure
                if self._mapping_admission_token is admission_token:
                    self._mapping_admission_token = None
                    self._is_mapping = False
                self._mapping_thread = None
                raise
            return True

    def _mapping_worker(
        self,
        base_dir: str,
        *,
        force: bool = False,
        _admission_token: Optional[object] = None,
    ) -> None:
        """Run the optional mapper without leaving a false running state."""
        try:
            self.map_infrastructure(
                base_dir,
                force=force,
                _admission_token=_admission_token,
            )
        except Exception as exc:  # noqa: BLE001 - owner-thread liveness boundary
            message = f"{type(exc).__name__}: {exc}"
            with self._mapping_lock:
                boundary_recorded = self._mapping_last_error == message
                self._mapping_last_error = message
                retained_generation = self.infrastructure_mapped
            if not boundary_recorded:
                record_degradation(
                    "mycelium",
                    exc,
                    severity="warning",
                    action=(
                        "retained the prior complete infrastructure generation after "
                        "owned mapper refresh failure"
                        if retained_generation
                        else "left infrastructure graph unmapped after owned mapper failure"
                    ),
                )
            logger.error("🍄 [MYCELIUM] Infrastructure mapping failed: %s", exc, exc_info=True)
        finally:
            with self._mapping_lock:
                # A worker may lose admission to a direct caller. It must never
                # clear that caller's latch. It may only release the reservation
                # that setup assigned specifically to this worker.
                if (
                    _admission_token is not None
                    and self._mapping_admission_token is _admission_token
                ):
                    self._mapping_admission_token = None
                    self._is_mapping = False
                if self._mapping_thread is threading.current_thread():
                    self._mapping_thread = None

    # ======================================================================
    # HARDWIRED PATHWAYS — The Core Intent Router
    # ======================================================================

    def register_pathway(
        self,
        pathway_id: str,
        pattern: str,
        skill_name: str,
        param_map: Optional[Dict[str, Any]] = None,
        priority: float = 1.0,
        activity_label: str = "",
        direct_response: Optional[str] = None,
    ) -> None:
        """Register a hardwired intent→skill pathway with regex param extraction.

        Args:
            pathway_id: Unique identifier (e.g., "image_gen_primary")
            pattern: Regex pattern string with capture groups for params
            skill_name: Target skill name (e.g., "generate_image")
            param_map: Maps skill param names to regex group indices or
                literal values for always-on params.
            priority: Higher priority pathways are checked first.
            activity_label: UI message shown when this pathway fires.
            direct_response: Legacy emergency response for non-user origins.
                Production user conversation must remain on CognitiveEngine.
        """
        compiled = re.compile(pattern, re.IGNORECASE)
        pw = HardwiredPathway(
            pathway_id=pathway_id,
            pattern=compiled,
            skill_name=skill_name,
            param_map=param_map or {},
            priority=priority,
            activity_label=activity_label or f"Aura is executing {skill_name}...",
            direct_response=direct_response,
        )
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                raise RuntimeError("retired mycelium instance has no active owner")
            if owner is not self:
                return owner.register_pathway(
                    pathway_id,
                    pattern,
                    skill_name,
                    param_map=param_map,
                    priority=priority,
                    activity_label=activity_label,
                    direct_response=direct_response,
                )
            self.pathways[pathway_id] = pw

            # Maintain sorted order (Bypass Aegis lock for internal update)
            object.__setattr__(
                self,
                "_pathway_order",
                sorted(
                    self.pathways.keys(),
                    key=lambda k: self.pathways[k].priority,
                    reverse=True,
                ),
            )
            self.direct_roots[pathway_id] = skill_name
            self._mark_topology_mutated_locked(structure_changed=True)

        logger.info(
            "🍄 [MYCELIUM] Pathway Hardwired: '%s' → %s (priority=%.1f, groups=%s)",
            pathway_id, skill_name, priority, list((param_map or {}).keys()),
        )


    def match_hardwired(self, text: str) -> Optional[Tuple[HardwiredPathway, Dict[str, Any]]]:
        """Match user text against all hardwired pathways with parameter extraction (Issue 77)."""
        if not isinstance(text, str) or not text.strip():
            return None

        # ISSUE-77: Strict Message Validation
        if len(text) > 4096:
            logger.warning("🍄 [MYCELIUM] Message too long for hardwired matching (%d chars)", len(text))
            return None
            
        text_clean = text.strip()

        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                return None
            if owner is not self:
                return owner.match_hardwired(text)
            candidates = tuple(
                (
                    pathway_id,
                    pathway,
                    pathway.model_copy(deep=True) if pathway is not None else None,
                )
                for pathway_id in self._pathway_order
                for pathway in (self.pathways.get(pathway_id),)
            )

        for pw_id, original, pw in candidates:
            if original is None or pw is None:
                continue

            # Skip pathways that have decayed below minimum confidence
            if pw.confidence < pw.MIN_CONFIDENCE:
                continue

            match = pw.pattern.search(text_clean)
            if match:
                # Extract params from capture groups
                params: Dict[str, Any] = {}
                for param_name, mapping in pw.param_map.items():
                    if isinstance(mapping, int):
                        try:
                            value = match.group(mapping)
                            if value:
                                params[param_name] = value.strip()
                        except (IndexError, AttributeError):
                            logger.warning(
                                "🍄 [MYCELIUM] Param extraction failed for '%s' group %s in pathway '%s'",
                                param_name, mapping, pw_id,
                            )
                    else:
                        params[param_name] = mapping

                with MycelialNetwork._lock:
                    owner = self._active_owner_locked()
                    if owner is None:
                        return None
                    if owner is not self:
                        return owner.match_hardwired(text)
                    current = self.pathways.get(pw_id)
                    if current is not original:
                        continue
                    current.last_matched = time.monotonic()
                    self._mark_topology_mutated_locked()
                    result = current.model_copy(deep=True)

                logger.info(
                    "🍄 [MYCELIUM] ⚡ HardwiredPathway MATCHED: '%s' → skill=%s, params=%s, confidence=%.2f",
                    pw_id, pw.skill_name, params, pw.confidence,
                )

                return (result, params)

        return None

    # ======================================================================
    # HYPHAE NETWORK — Subsystem Connectivity
    # ======================================================================

    def establish_connection(self, source: str, target: str, priority: float = 1.0) -> Hypha:
        """Establish a subsystem hypha and return a detached read model."""
        hypha_id = f"{source}->{target}"
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                raise RuntimeError("retired mycelium instance has no active owner")
            if owner is not self:
                return owner.establish_connection(source, target, priority=priority)
            hypha = self.hyphae.get(hypha_id)
            if hypha is None:
                hypha = Hypha(
                    name=hypha_id,
                    source=source,
                    target=target,
                    priority=priority,
                )
                self.hyphae[hypha_id] = hypha
                self._mark_topology_mutated_locked(structure_changed=True)
                logger.info("🍄 [MYCELIUM] Hypha established: %s", hypha_id)
            return hypha.model_copy(deep=True)

    def add_hypha(self, source: str, target: str, link_type: str = "general", metadata: Optional[Dict] = None):
        """Enterprise method for adding a hypha with rich metadata."""
        hypha_id = f"{source}->{target}"
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                raise RuntimeError("retired mycelium instance has no active owner")
            if owner is not self:
                return owner.add_hypha(source, target, link_type=link_type, metadata=metadata)
            if hypha_id not in self.hyphae:
                self.hyphae[hypha_id] = Hypha(
                    name=hypha_id,
                    source=source,
                    target=target,
                    trace=[f"Link Type: {link_type}"]
                )
                self._mark_topology_mutated_locked(structure_changed=True)
                logger.info("🍄 [MYCELIUM] Hypha added: %s (%s)", hypha_id, link_type)

    def get_hypha(self, source: str, target: str = None) -> Optional[Hypha]:
        """Return a detached hypha read model."""
        if target is None and "->" in source:
            hypha_id = source
        else:
            hypha_id = f"{source}->{target}"
            
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                return None
            if owner is not self:
                return owner.get_hypha(source, target)
            hypha = self.hyphae.get(hypha_id)
            return hypha.model_copy(deep=True) if hypha is not None else None

    @staticmethod
    def _hypha_id(source: str, target: Optional[str] = None) -> str:
        return source if target is None and "->" in source else f"{source}->{target}"

    def pulse_hypha(
        self,
        source: str,
        target: Optional[str] = None,
        *,
        success: bool = True,
    ) -> bool:
        """Pulse the current owned edge without leaking a mutable object."""
        hypha_id = self._hypha_id(source, target)
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                return False
            if owner is not self:
                return owner.pulse_hypha(source, target, success=success)
            hypha = self.hyphae.get(hypha_id)
            if hypha is None:
                return False
            hypha.pulse(success=success)
            self._mark_topology_mutated_locked()
            return True

    def log_hypha(self, source: str, target: Optional[str], message: str) -> bool:
        """Append an owned trace entry to the current edge."""
        hypha_id = self._hypha_id(source, target)
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                return False
            if owner is not self:
                return owner.log_hypha(source, target, message)
            hypha = self.hyphae.get(hypha_id)
            if hypha is None:
                return False
            hypha.log(message)
            self._mark_topology_mutated_locked()
            return True

    def _record_rooted_flow_event(
        self,
        source: str,
        target: str,
        *,
        priority: float,
        message: Optional[str] = None,
        success: Optional[bool] = None,
    ) -> Hypha:
        """Atomically bind one flow event to the currently published owner."""
        hypha_id = self._hypha_id(source, target)
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                raise RuntimeError("retired mycelium instance has no active owner")
            if owner is not self:
                return owner._record_rooted_flow_event(
                    source,
                    target,
                    priority=priority,
                    message=message,
                    success=success,
                )
            hypha = self.hyphae.get(hypha_id)
            if hypha is None:
                self.establish_connection(source, target, priority=priority)
                hypha = self.hyphae[hypha_id]
            if success is not None:
                hypha.pulse(success=success)
                self._mark_topology_mutated_locked()
            if message is not None:
                hypha.log(message)
                self._mark_topology_mutated_locked()
            return hypha.model_copy(deep=True)

    def set_hypha_strength(
        self,
        source: str,
        target: Optional[str],
        strength: float,
    ) -> bool:
        """Set current edge strength through the topology owner."""
        hypha_id = self._hypha_id(source, target)
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                return False
            if owner is not self:
                return owner.set_hypha_strength(source, target, strength)
            hypha = self.hyphae.get(hypha_id)
            if hypha is None:
                return False
            hypha.strength = max(0.1, min(10.0, float(strength)))
            self._mark_topology_mutated_locked()
            return True

    def link_layer(self, layer_name: str, module_class: Any):
        """High-level linking for transcendence modules."""
        logger.info("🍄 [MYCELIUM] Linking Transcendence Layer: '%s' -> %s", layer_name, module_class.__name__)
        # This typically involves registering the module's presence for the discovery engine
        # and creating primary hyphae to the core cognition engine.
        self.establish_connection(layer_name, "cognition", priority=0.9)
        self.establish_connection("cognition", layer_name, priority=0.8)

    def route_signal(self, source: str, target: str, payload: Dict[str, Any]):
        """Directly route a cognitive signal between subsystems."""
        owner = self._active_owner()
        if owner is None:
            return False
        if owner is not self:
            return owner.route_signal(source, target, payload)
        hypha_id = f"{source}->{target}"
        try:
            if self.get_hypha(hypha_id) is None:
                self.establish_connection(source, target)
            self._log_route_signal(source, target, payload)
            if self.pulse_hypha(hypha_id, success=True):
                return True
            # The singleton may have been replaced between lookup and pulse.
            self.establish_connection(source, target)
            return self.pulse_hypha(hypha_id, success=True)
        except RuntimeError:
            return False

    def _log_route_signal(self, source: str, target: str, payload: Dict[str, Any]) -> None:
        """Emit route-signal telemetry on state change instead of every pulse."""
        key = f"{source}->{target}"
        payload_text = str(payload)[:160]
        now = time.monotonic()
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                return
            if owner is not self:
                return owner._log_route_signal(source, target, payload)
            previous_payload, previous_at, suppressed = (
                self._route_signal_log_state.get(key, ("", 0.0, 0))
            )
            repeated = payload_text == previous_payload and (now - previous_at) < 30.0
            if repeated:
                self._route_signal_log_state[key] = (
                    previous_payload,
                    previous_at,
                    suppressed + 1,
                )
            else:
                self._route_signal_log_state[key] = (payload_text, now, 0)
        if repeated:
            logger.debug(
                "🍄 [MYCELIUM] Repeated signal pulse suppressed: %s | Payload: %s",
                key,
                payload_text,
            )
            return

        if suppressed:
            logger.info(
                "🍄 [MYCELIUM] 📡 Signal Routed: %s -> %s | Payload: %s | repeated=%d",
                source,
                target,
                payload_text,
                suppressed,
            )
            return
        logger.info(
            "🍄 [MYCELIUM] 📡 Signal Routed: %s -> %s | Payload: %s",
            source,
            target,
            payload_text,
        )

    async def emit_reflex(self, signal_type: str, metadata: Dict = None):
        """Broadcast a critical reflex signal across the mycelial network."""
        owner = self._active_owner()
        if owner is None:
            return False
        if owner is not self:
            return await owner.emit_reflex(signal_type, metadata)
        if self.reflex:
            await self.reflex.trigger_reflex(signal_type, metadata)
            return True
        else:
            logger.warning("No Reflex Core online to handle signal: %s", signal_type)
            return False

    async def emit(self, signal_type: str, metadata: Dict = None):
        """Compatibility event-bus bridge for callers that treat mycelium like a bus."""
        owner = self._active_owner()
        if owner is None:
            return None
        if owner is not self:
            return await owner.emit(signal_type, metadata)
        payload = dict(metadata or {})
        payload.setdefault("signal_type", signal_type)
        try:
            from core.event_bus import EventPriority, get_event_bus

            await get_event_bus().publish(signal_type, payload, priority=EventPriority.COGNITIVE)
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation('mycelium', exc)
            logger.debug("🍄 [MYCELIUM] emit bridge publish failed: %s", exc)
        return payload

    def _should_monitor_hypha(self, hypha: Hypha) -> bool:
        """Only alarm on edges that have actually carried traffic or map to hardware."""
        return bool(hypha.is_physical or hypha.pulse_count > 0 or hypha.trace)

    def establish_neural_root(self, source: str, hardware_id: str = "gpu_metal") -> NeuralRoot:
        """Builds a direct, pinned connection between a subsystem and hardware."""
        hypha_id = f"{source}->hardware:{hardware_id}"
        nr = NeuralRoot(
            name=hypha_id,
            source=source,
            target=f"hardware:{hardware_id}",
            hardware_id=hardware_id,
            pinned=True,
            priority=5.0 # Highest priority unblockable root
        )
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                raise RuntimeError("retired mycelium instance has no active owner")
            if owner is not self:
                return owner.establish_neural_root(source, hardware_id=hardware_id)
            self.hyphae[hypha_id] = nr
            self._neural_roots.append(nr)
            self._mark_topology_mutated_locked(structure_changed=True)
        logger.info("🍄 [MYCELIUM] 🌿 Neural Root ESTABLISHED: %s", hypha_id)
        return nr.model_copy(deep=True)

    async def hardware_pulse(self):
        """Maintain global hardware connectivity for all neural roots."""
        owner = self._active_owner()
        if owner is None:
            return
        if owner is not self:
            await owner.hardware_pulse()
            return
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                return
            if owner is not self:
                neural_roots = ()
            else:
                neural_roots = tuple(self._neural_roots)
        if owner is not self:
            await owner.hardware_pulse()
            return
        for nr in neural_roots:
            try:
                # Use run_io_bound for the blocking hardware pulse
                success = await run_io_bound(nr.subsurface_ping)
                self.pulse_hypha(nr.name, success=success)
                if not success:
                    logger.warning("🍄 [MYCELIUM] Neural Root pulse drop: %s", nr.name)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('mycelium', e)
                logger.error("🍄 [MYCELIUM] Neural Root pulse failure: %s", e)

    def reinforce(self, pathway_id: str, success: bool):
        """Physarum-inspired conductivity update after skill execution.

        Enterprise Enhancement: Also pulses all physical hyphae connected to
        the pathway's source module, so the import graph strengthens where
        it matters at runtime.
        
        Transcendental Enhancement: Reinforcement is weighted by qualia norm.
        Pathways fired during high phenomenal intensity learn faster.
        """
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                return
            if owner is not self:
                return owner.reinforce(pathway_id, success)
            pw = self.pathways.get(pathway_id)
            if not pw:
                return
            pw.reinforce(success)
            self._mark_topology_mutated_locked()

        # --- QUALIA-WEIGHTED REINFORCEMENT ---
        # If consciousness is "resonating" during this execution,
        # apply an extra confidence boost (or penalty) to the pathway.
        try:
            from core.container import ServiceContainer
            qualia = ServiceContainer.get("qualia_synthesizer", default=None)
            if qualia and qualia.q_norm > 0.5:
                # Evolution 8: Weight by Phenomenological Arousal
                experiencer = ServiceContainer.get("phenomenological_experiencer", default=None)
                arousal = getattr(experiencer, 'current_arousal', 0.5) if experiencer else 0.5
                
                # Scale bonus by how far above threshold
                qualia_bonus = (qualia.q_norm - 0.5) * 0.1 * (arousal * 2.0)
                with MycelialNetwork._lock:
                    if self._active_owner_locked() is not self:
                        return
                    current = self.pathways.get(pathway_id)
                    if current is None:
                        return
                    pw = current
                    if success:
                        pw.confidence = min(10.0, pw.confidence + qualia_bonus)
                    else:
                        pw.confidence = max(
                            0.1,
                            pw.confidence - qualia_bonus * 0.5,
                        )
                    self._mark_topology_mutated_locked()
                logger.debug(
                    "🍄 [MYCELIUM] 🧠 Qualia-weighted reinforcement: '%s' ±%.3f (q=%.2f, a=%.2f)",
                    pathway_id, qualia_bonus, qualia.q_norm, arousal
                )
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('mycelium', e)
            capture_and_log(e, {'module': __name__})

        # --- RUNTIME PHYSICAL HYPHAE REINFORCEMENT ---
        with MycelialNetwork._lock:
            if self._active_owner_locked() is not self:
                return
            current = self.pathways.get(pathway_id)
            if current is None:
                return
            source_file = current.source_file
            infrastructure_mapped = self.infrastructure_mapped
        if source_file and infrastructure_mapped:
            source_module = None
            for mk, info in self.get_mapped_files_snapshot().items():
                if info.get("path") == source_file:
                    source_module = mk
                    break

            if source_module:
                pulsed = 0
                with MycelialNetwork._lock:
                    if self._active_owner_locked() is not self:
                        return
                    for h in self.hyphae.values():
                        if h.is_physical and (
                            h.source == source_module or h.target == source_module
                        ):
                            h.pulse(success)
                            pulsed += 1
                    if pulsed:
                        self._mark_topology_mutated_locked()
                if pulsed > 0:
                    logger.debug(
                        "🍄 [MYCELIUM] ⚡ Runtime pulse: %d physical hyphae for '%s' (%s)",
                        pulsed, source_module, "✓" if success else "✗",
                    )

        with MycelialNetwork._lock:
            if self._active_owner_locked() is not self:
                return
            current = self.pathways.get(pathway_id)
            confidence = current.confidence if current is not None else pw.confidence
            success_rate = current.success_rate if current is not None else pw.success_rate
            is_weak = current.is_weak if current is not None else pw.is_weak
        if is_weak:
            logger.warning(
                "🍄 [MYCELIUM] ⚠️ Pathway '%s' is WEAK (confidence=%.2f, rate=%.0f%%). "
                "Consider reviewing or removing.",
                pathway_id, confidence, success_rate * 100,
            )
        else:
            logger.debug(
                "🍄 [MYCELIUM] Pathway '%s' reinforced: confidence=%.2f (%s)",
                pathway_id, confidence, "✓" if success else "✗",
            )


    # --- Legacy Compatibility Shims ---

    def register_direct_root(self, pattern: str, skill_name: str):
        """Legacy shim: converts old substring patterns to basic regex pathways."""
        safe_pattern = re.escape(pattern)
        self.register_pathway(
            pathway_id=f"legacy_{pattern.replace(' ', '_')}",
            pattern=safe_pattern,
            skill_name=skill_name,
            param_map={},
            priority=0.5,  # Lower priority than proper regexes
            activity_label=f"Aura is executing {skill_name}...",
        )

    def match_direct_root(self, text: str) -> Optional[str]:
        """Legacy shim: returns just the skill name for old orchestrator code."""
        result = self.match_hardwired(text)
        if result:
            return result[0].skill_name
        return None

    # ======================================================================
    # DISCOVERY ENGINE — Slime Mold Exploration
    # ======================================================================

    def record_execution(self, message: str, skill_name: str, params: Dict[str, Any], success: bool):
        """Record a non-hardwired skill execution for pathway discovery.

        Called by the orchestrator after the state machine successfully routes
        a message to a skill via LLM classification (i.e., the slow path).
        If the same skill is used repeatedly with similar messages, the network
        proposes a new hardwired pathway.
        """
        if not success:
            return

        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                return
            if owner is not self:
                return owner.record_execution(message, skill_name, params, success)
            self._execution_log.append({
                "message": message,
                "skill": skill_name,
                "params": dict(params),
                "timestamp": time.monotonic(),
            })
            if len(self._execution_log) > 500:
                self._execution_log = self._execution_log[-250:]
            self._discovery_candidates[skill_name] += 1
            should_propose = self._discovery_candidates[skill_name] >= 5

        if should_propose:
            self._propose_pathway(skill_name)

    def _propose_pathway(self, skill_name: str):
        """Analyze recent executions to propose a new hardwired pathway."""
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                return
            if owner is not self:
                return owner._propose_pathway(skill_name)
            relevant_count = sum(
                1 for event in self._execution_log if event["skill"] == skill_name
            )
            if relevant_count < 3:
                return
            existing_count = sum(
                1 for pathway in self.pathways.values()
                if pathway.skill_name == skill_name
            )
            if existing_count >= 5:
                return
            self._discovery_candidates[skill_name] = 0

        logger.info(
            "🍄 [MYCELIUM] 🌱 Discovery: skill '%s' used %d times via slow path. "
            "Consider adding a hardwired pathway for common patterns.",
            skill_name, relevant_count,
        )

    # ======================================================================
    # GENERAL HYPHAE — Subsystem Connections
    # ======================================================================

    def set_ui_callback(self, callback: Callable[[str], Coroutine]):
        """Connect the Mycelium directly to the UI for failsafe message delivery."""
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                raise RuntimeError("retired mycelium instance has no active owner")
            if owner is not self:
                return owner.set_ui_callback(callback)
            self.ui_callback = callback
        logger.info("🍄 [MYCELIUM] Direct UI Hypha Connected.")


    def establish_unification_hyphae(self):
        """Phase 25: Sovereign Unification Hyphae.
        
        Links canonical subsystems into the root network to ensure they are 
        visible and tracked even before dynamic mapping completes.
        Names here match SubsystemAudit.SUBSYSTEMS for identity synchronization.
        """
        owner = self._active_owner()
        if owner is None:
            return False
        if owner is not self:
            return owner.establish_unification_hyphae()
        unification_links = [
            ("orchestrator", "personality_engine", 3.0, "#FF69B4", "Core identity and persona control"),
            ("orchestrator", "memory_facade", 3.0, "#F5A623", "Long-term knowledge and episodic recall"),
            ("orchestrator", "affect_engine", 2.5, "#D0021B", "Emotional state and motivation substrate"),
            ("orchestrator", "drive_controller", 2.0, "#BD10E0", "Biological-inspired drives and urgency"),
            ("orchestrator", "liquid_substrate", 2.0, "#7ED321", "Dynamic arousal and focus management"),
            ("orchestrator", "sovereign_scanner", 2.0, "#50E3C2", "Reactive intent detection and safety"),
            ("personality_engine", "cognition", 2.5, "#4A90E2", "Identity guiding thought generation"),
            ("cognition", "autonomy", 3.0, "#9013FE", "Decision making and goal selection"),
            ("autonomy", "cognition", 3.0, "#9013FE", "Feedback loop for autonomous action"),
            ("mind_tick", "mycelium", 2.5, "#F8E71C", "Universal heartbeat and connectivity"),
            ("orchestrator", "critic_engine", 3.0, "#50E3C2", "Recursive self-correction and plan verification"),
            # --- Personhood & Resilience Roots ---
            ("orchestrator", "personhood", 3.0, "#FF007F", "Spontaneous thought and subjective agency"),
            ("orchestrator", "voice_presence", 3.0, "#00FFFF", "Vocal embodiment and immediate expression"),
            ("orchestrator", "stability_guardian", 3.0, "#39FF14", "Real-time health monitoring and stall prevention"),
            ("orchestrator", "research_cycle", 2.5, "#FFFF00", "Autonomous knowledge pursuit during idle"),
        ]
        for src, tgt, prio, color, desc in unification_links:
            self.establish_connection(src, tgt, priority=prio)
            with MycelialNetwork._lock:
                owner = self._active_owner_locked()
                if owner is None:
                    return False
                if owner is not self:
                    return owner.establish_unification_hyphae()
                hypha = self.hyphae.get(f"{src}->{tgt}")
                if hypha is not None:
                    hypha.color = color
                    hypha.description = desc
                    hypha.strength = 5.0
                    self._mark_topology_mutated_locked()
        logger.info("🍄 [MYCELIUM] ✅ Core Unification Hyphae established (%d links)", len(unification_links))
        return True

    def shutdown(self):
        """Phase 28: Total Neural Root Cleanup (Issue 76).
        Ensures all hardware pins and active hyphae are safely disconnected.
        """
        logger.info("🍄 [MYCELIUM] Neutralizing all neural roots and hyphae...")
        with MycelialNetwork._lock:
            self._stop_event.set()
            if MycelialNetwork._instance is self:
                MycelialNetwork._instance = None
                MycelialNetwork._initialized = False
            mapping_thread = self._mapping_thread
        if (
            mapping_thread is not None
            and mapping_thread.is_alive()
            and mapping_thread is not threading.current_thread()
        ):
            mapping_thread.join(timeout=3.0)
            if mapping_thread.is_alive():
                logger.warning(
                    "🍄 [MYCELIUM] Mapper did not drain within shutdown budget; "
                    "the publication latch remains closed."
                )
        with MycelialNetwork._lock:
            self.infrastructure_mapped = False
            self._is_mapping = False
            if mapping_thread is None or not mapping_thread.is_alive():
                self._mapping_thread = None
            self._execution_log.clear()
            self._discovery_candidates.clear()
            self._route_signal_log_state.clear()
            self._hypha_alert_times.clear()
            self.pathways.clear()
            object.__setattr__(self, "_pathway_order", [])
            self.direct_roots.clear()
            self.hyphae.clear()
            self.mapped_files.clear()
            self._centrality.clear()
            self._critical_modules.clear()
            self._cross_links.clear()
            self._neural_roots.clear()
            self.ui_callback = None
            self._mark_topology_mutated_locked(structure_changed=True)
        logger.info("🍄 [MYCELIUM] Network Offline.")

    def on_stop(self) -> None:
        """ServiceContainer lifecycle hook."""
        self.shutdown()

    def establish_consciousness_hyphae(self):
        """Phase 5: Transcendental Consciousness Hyphae.
        Specifically links modules related to qualia and phenomenology.
        """
        links = [
            ("qualia", "phenomenology", 3.0),
            ("consciousness", "global_workspace", 2.5),
            ("sentience", "autonomy", 2.0),
        ]
        for src, tgt, prio in links:
            self.establish_connection(src, tgt, priority=prio)
        logger.info("🍄 [MYCELIUM] 👁️ Consciousness Hyphae established.")

    @asynccontextmanager
    async def rooted_flow(self, source: str, target: str, activity: str = None,
                          timeout: float = 60.0, priority: float = 1.0):
        """Wraps a process in a mycelial root. If it stalls, the root overrides."""
        try:
            timeout_s = float(timeout)
        except (TypeError, ValueError):
            timeout_s = 60.0
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            timeout_s = 60.0
        activity_label = str(activity or f"{source}->{target}")
        owner = self._active_owner()
        if owner is None:
            raise RuntimeError("retired mycelium instance has no active owner")
        if owner is not self:
            async with owner.rooted_flow(
                source,
                target,
                activity=activity_label,
                timeout=timeout_s,
                priority=priority,
            ) as handle:
                yield handle
            return
        hypha_id = f"{source}->{target}"
        handle = RootedFlowHandle(self, source, target, priority)
        handle.log(f"INITIATING: {activity_label}")

        try:
            async with asyncio.timeout(timeout_s):
                yield handle
            handle.pulse(success=True)
            handle.log(f"SUCCESS: {activity_label}")
        except asyncio.CancelledError:
            try:
                handle.log(f"CANCELLED: {activity_label}")
            except Exception as topology_error:  # noqa: BLE001 - preserve cancellation
                record_degradation("mycelium.rooted_flow_telemetry", topology_error)
            raise
        except Exception as e:  # noqa: BLE001 - rooted-flow failure boundary
            handle._mark_failed(e)
            record_degradation('mycelium', e)
            hypha = None
            try:
                handle.pulse(success=False)
                handle.log(f"STALL/FAILURE: {activity_label} - {e}")
                hypha = handle._snapshot()
            except Exception as topology_error:  # noqa: BLE001 - preserve original error
                record_degradation("mycelium.rooted_flow_telemetry", topology_error)
                logger.error(
                    "🍄 [MYCELIUM] Could not persist rooted-flow failure for %s: %s",
                    hypha_id,
                    topology_error,
                    exc_info=True,
                )
            logger.error("🍄 [MYCELIUM] Critical Stall in %s (%s). Triggering Override.", hypha_id, e)
            recovery_owner = self._active_owner()
            if hypha is not None and recovery_owner is not None:
                recovery_timeout_s = min(5.0, max(0.1, timeout_s))
                try:
                    async with asyncio.timeout(recovery_timeout_s):
                        await recovery_owner._emergency_override(
                            hypha, activity_label, str(e)
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as recovery_error:  # noqa: BLE001 - recovery boundary
                    record_degradation(
                        "mycelium.emergency_override",
                        recovery_error,
                        severity="error",
                        action=(
                            "bounded emergency override failure without masking the "
                            "original rooted-flow error"
                        ),
                    )
                    logger.error(
                        "🍄 [MYCELIUM] Emergency override failed for %s: %s",
                        hypha_id,
                        recovery_error,
                        exc_info=True,
                    )
            if hypha is not None and hypha.priority >= 1.0:
                return  # Absorbed error — failsafe bypass
            raise

    async def _emergency_override(self, hypha: Hypha, activity: str, error_msg: str):
        """Force a result through the Mycelium when the standard path stalls."""
        logger.warning("⚡ [ROOT OVERRIDE] Forcing path completion: %s → %s", hypha.name, activity)
        
        # Bridge to Hardened Reflex Core
        if self.reflex:
            await self.reflex.trigger_reflex("STALL_DETECTED", {
                "hypha": hypha.name,
                "activity": activity,
                "error": error_msg
            })
            
        if "response" in activity.lower() and self.ui_callback:
            msg = (
                "🛡️ [Mycelial Failsafe Active] I encountered a stall while processing "
                f"your request ({error_msg}). My system unity has bypassed the block."
            )
            await self.ui_callback(msg)
        self.set_hypha_strength(hypha.name, None, hypha.strength + 2.0)

    # ======================================================================
    # INFRASTRUCTURE MAPPING — Codebase Unification
    # ======================================================================

    def map_infrastructure(
        self,
        base_dir: str,
        scan_dirs: Optional[List[str]] = None,
        *,
        force: bool = False,
        _admission_token: Optional[object] = None,
    ) -> bool:
        """Publish one complete code-map generation or retain the previous one.

        This public boundary owns admission and cleanup for both direct callers
        and the background worker. No exception can leave the mapping latch set.

        Args:
            base_dir: Absolute path to the project root (e.g., autonomy_engine/).
            scan_dirs: Optional subdirectories under ``base_dir`` to scan.
                The default covers every production source root plus root modules.
        """
        owner = self._active_owner()
        if owner is None:
            return False
        if owner is not self:
            if _admission_token is not None:
                return False
            return owner.map_infrastructure(base_dir, scan_dirs, force=force)
        with self._mapping_lock:
            owner = self._active_owner_locked()
            if owner is None:
                return False
            if owner is not self:
                if _admission_token is not None:
                    return False
                return owner.map_infrastructure(base_dir, scan_dirs, force=force)
            if not force and self._foreground_mapping_deferred():
                return False
            if _admission_token is None:
                if self._is_mapping or (self.infrastructure_mapped and not force):
                    return False
                admission_token = object()
                self._mapping_admission_token = admission_token
                self._is_mapping = True
            else:
                admission_token = _admission_token
                if (
                    not self._is_mapping
                    or self._mapping_admission_token is not admission_token
                ):
                    return False
            previously_mapped = self.infrastructure_mapped
            self._mapping_started_at = time.time()
            self._mapping_last_error = None
            self._deferred_mapping_reason = None

        try:
            return self._map_infrastructure_generation(
                base_dir,
                scan_dirs,
                previously_mapped=previously_mapped,
            )
        except Exception as exc:  # noqa: BLE001 - public lifecycle boundary records then re-raises
            with self._mapping_lock:
                self.infrastructure_mapped = previously_mapped
                self._mapping_last_error = f"{type(exc).__name__}: {exc}"
            record_degradation(
                "mycelium",
                exc,
                severity="warning",
                action=(
                    "retained the prior complete infrastructure generation after "
                    "mapping failure"
                    if previously_mapped
                    else "left infrastructure graph unmapped after mapping failure"
                ),
            )
            raise
        finally:
            with self._mapping_lock:
                if self._mapping_admission_token is admission_token:
                    self._mapping_admission_token = None
                    self._is_mapping = False

    def _map_infrastructure_generation(
        self,
        base_dir: str,
        scan_dirs: Optional[List[str]],
        *,
        previously_mapped: bool,
    ) -> bool:
        """Build a private infrastructure generation and publish it atomically."""
        # Optimization: Use a local cache for AST results to avoid re-parsing if called multiple times
        # though singleton pattern usually prevents this.
        
        base = Path(base_dir).resolve()
        if scan_dirs is None:
            scan_dirs = list(_DEFAULT_INFRASTRUCTURE_SCAN_DIRS)

        start_time_map = time.monotonic()
        logger.info("🍄 [MYCELIUM] 🗺️ Infrastructure Mapping starting from: %s", base)

        # 1. Discover all .py files.
        all_files: Dict[str, Path] = {}  # module_key → file_path
        if scan_dirs == list(_DEFAULT_INFRASTRUCTURE_SCAN_DIRS):
            for py_file in base.glob("*.py"):
                if not py_file.name.startswith("__"):
                    all_files[py_file.stem] = py_file
        for subdir in scan_dirs:
            scan_root = base / subdir
            if not scan_root.exists():
                logger.debug("🍄 [MYCELIUM] Scan directory not found: %s", scan_root)
                continue
            for py_file in scan_root.rglob("*.py"):
                if self._stop_event.is_set():
                    logger.info("🍄 [MYCELIUM] Infrastructure mapping cancelled during discovery.")
                    return False
                if py_file.name.startswith("__"):
                    continue
                try:
                    rel = py_file.relative_to(base)
                    module_key = str(rel.with_suffix("")).replace(os.sep, ".")
                    all_files[module_key] = py_file
                except ValueError:
                    continue

        logger.info("🍄 [MYCELIUM] Discovered %d Python modules.", len(all_files))

        # 2. Parse imports and build dependency edges
        dependency_graph: Dict[str, List[str]] = {}
        mapped_files: Dict[str, Dict[str, Any]] = {}
        for module_key, file_path in all_files.items():
            if self._stop_event.is_set():
                logger.info("🍄 [MYCELIUM] Infrastructure mapping cancelled during parsing.")
                return False
            deps = self._extract_imports(file_path, base)
            dependency_graph[module_key] = deps

            # Build privately. Readers must never observe a half-published map.
            mapped_files[module_key] = {
                "path": str(file_path),
                "size_bytes": file_path.stat().st_size if file_path.exists() else 0,
                "imports": deps,
            }

        # 3. Create physical Hypha connections for import relationships
        physical_hyphae: Dict[str, Hypha] = {}
        for module_key, deps in dependency_graph.items():
            for dep in deps:
                if dep in all_files:
                    hypha_name = f"import:{module_key}->{dep}"
                    h = Hypha(
                        name=hypha_name,
                        source=module_key,
                        target=dep,
                        priority=0.5,
                        is_physical=True,
                    )
                    h.source_file = str(all_files[module_key])
                    h.target_file = str(all_files[dep])
                    physical_hyphae[hypha_name] = h
        physical_connections = len(physical_hyphae)

        # 4. Compute Module Centrality (reverse dependency index)
        #    Centrality = how many other modules depend on this one.
        #    High centrality = load-bearing pillar; failure has wide blast radius.
        reverse_deps: Dict[str, int] = {}
        for _module_key, deps in dependency_graph.items():
            for dep in deps:
                if dep in all_files:
                    reverse_deps[dep] = reverse_deps.get(dep, 0) + 1

        centrality = {k: int(v) for k, v in reverse_deps.items()}

        # Tag the top-20 most critical modules
        critical_modules = [
            module
            for module, _count in sorted(
                reverse_deps.items(), key=lambda item: item[1], reverse=True
            )[:20]
        ]

        # Store centrality in mapped_files for API exposure
        for module_key, module_data in mapped_files.items():
            module_data["centrality"] = reverse_deps.get(module_key, 0)
            module_data["is_critical"] = module_key in critical_modules

        # 5. Cross-Layer Linking: connect logical subsystem hyphae to physical backing
        #    Maps abstract subsystem names (e.g., "cognition") to the directory/module
        #    patterns they correspond to in the codebase.
        SUBSYSTEM_ALIASES: Dict[str, List[str]] = {
            "cognition": ["cognitive", "brain", "cognitive_engine", "cognitive_integration"],
            "personality": ["personality", "persona", "identity"],
            "memory": ["memory", "dual_memory", "episodic"],
            "affect": ["affect", "emotion", "mood"],
            "autonomy": ["autonomy", "autonomic", "volition", "agency"],
            "perception": ["perception", "senses", "sensory", "screen_observer"],
            "consciousness": ["consciousness", "awareness", "existential", "qualia", "subjectivity", "sentience"],
            "self_modification": ["self_modification", "self_mod", "evolution", "mutate"],
            "skills": ["skills", "capability", "skill_management"],
            "scanner": ["scanner", "cognitive.scanner"],
            "mycelium": ["mycelium"],
            "guardian": ["guardian", "autonomy_guardian"],
            "state_machine": ["state_machine", "orchestrator.state"],
            "drive_engine": ["drive", "motivation", "drives"],
            "telemetry": ["telemetry", "thought_stream", "neural_feed"],
            "system": ["orchestrator", "main", "container"],
            "core_logic": ["orchestrator", "pipeline", "cognitive"],
            "skill_execution": ["capability_engine", "skill_execution"],
            # Phase XXII: Transcendence subsystems
            "meta_evolution": ["meta_cognition", "meta_evolution"],
            "hephaestus": ["hephaestus", "skill_management"],
            "networking": ["networking"],
            "model_selector": ["model_selector", "llm", "brain"],
            "curiosity": ["curiosity", "curiosity_engine", "exploration"],
            # Phase II: Deep consciousness sub-modules
            "cel": ["constitutive_expression", "cel"],
            "iit_phi": ["iit_surrogate", "riiu", "phi"],
            "workspace": ["global_workspace", "gwt"],
            "ganglion": ["ganglion_node", "ganglion"],
            "executive": ["executive_inhibitor", "executive"],
            "qualia_engine": ["qualia_engine"],
            "quantum_entropy": ["quantum_entropy"],
            "opacity": ["structural_opacity", "opacity"],
        }

        def _matches_subsystem(subsystem_name: str, module_path: str) -> bool:
            """Check if a module path belongs to a named subsystem."""
            aliases = SUBSYSTEM_ALIASES.get(subsystem_name, [subsystem_name])
            mp = module_path.lower()
            return any(alias in mp for alias in aliases)

        def _build_cross_links(
            logical_hyphae: Dict[str, Hypha],
        ) -> Dict[str, List[str]]:
            links: Dict[str, List[str]] = {}
            for logical_name, logical_hypha in logical_hyphae.items():
                backing_physical: List[str] = []
                for physical_name, physical_hypha in physical_hyphae.items():
                    source_matches = _matches_subsystem(
                        logical_hypha.source, physical_hypha.source
                    )
                    target_matches = _matches_subsystem(
                        logical_hypha.target, physical_hypha.target
                    )
                    if source_matches and target_matches:
                        backing_physical.append(physical_name)
                if backing_physical:
                    links[logical_name] = backing_physical
            return links

        # M-15 FIX: Prevent false-positive mapping if zero modules found
        if not all_files:
            logger.warning("🍄 [MYCELIUM] ❌ Infrastructure mapping found 0 modules! Retrying in next cycle.")
            with self._mapping_lock:
                self.infrastructure_mapped = previously_mapped
                self._mapping_last_error = "no_modules_discovered"
            return False

        # Cross-linking is O(logical × physical), so compute it outside the
        # topology lock. A compact endpoint signature detects structural races;
        # dynamic pulse updates do not force needless retries.
        annotated = 0
        for _attempt in range(5):
            with MycelialNetwork._lock:
                logical_hyphae = {
                    name: hypha.model_copy(deep=True)
                    for name, hypha in self.hyphae.items()
                    if not hypha.is_physical
                }
                logical_signature = tuple(
                    sorted(
                        (name, hypha.source, hypha.target)
                        for name, hypha in logical_hyphae.items()
                    )
                )
                pathway_skills = {
                    pathway_id: pathway.skill_name
                    for pathway_id, pathway in self.pathways.items()
                }
                pathway_signature = tuple(
                    sorted(
                        (
                            pathway_id,
                            id(pathway),
                            pathway.skill_name,
                        )
                        for pathway_id, pathway in self.pathways.items()
                    )
                )
            cross_links = _build_cross_links(logical_hyphae)
            pathway_annotations = self._build_pathway_annotations(
                pathway_skills,
                all_files,
                dependency_graph,
            )

            # Publish one coherent generation. UI, health, reinforcement, and
            # vault readers see either the previous graph or this complete graph.
            with MycelialNetwork._lock:
                if self._active_owner_locked() is not self:
                    logger.info(
                        "🍄 [MYCELIUM] Infrastructure publication cancelled after owner replacement."
                    )
                    return False
                if self._stop_event.is_set():
                    logger.info(
                        "🍄 [MYCELIUM] Infrastructure mapping cancelled before publication."
                    )
                    return False
                current_logical = {
                    name: hypha
                    for name, hypha in self.hyphae.items()
                    if not hypha.is_physical
                }
                current_signature = tuple(
                    sorted(
                        (name, hypha.source, hypha.target)
                        for name, hypha in current_logical.items()
                    )
                )
                if current_signature != logical_signature:
                    continue
                current_pathway_signature = tuple(
                    sorted(
                        (
                            pathway_id,
                            id(pathway),
                            pathway.skill_name,
                        )
                        for pathway_id, pathway in self.pathways.items()
                    )
                )
                if current_pathway_signature != pathway_signature:
                    continue

                # Preserve learned dynamic state for unchanged import edges.
                for name, replacement in physical_hyphae.items():
                    existing = self.hyphae.get(name)
                    if (
                        existing is None
                        or not existing.is_physical
                        or existing.source != replacement.source
                        or existing.target != replacement.target
                    ):
                        continue
                    replacement.strength = existing.strength
                    replacement.created_at = existing.created_at
                    replacement.last_pulse = existing.last_pulse
                    replacement.pulse_count = existing.pulse_count
                    replacement.active = existing.active
                    replacement.color = existing.color
                    replacement.description = existing.description
                    replacement.size = existing.size
                    replacement.trace = list(existing.trace)

                previous_mapped_paths = {
                    str(module.get("path"))
                    for module in self.mapped_files.values()
                    if module.get("path")
                }
                for pathway_id, pathway in self.pathways.items():
                    annotation = pathway_annotations.get(pathway_id)
                    if annotation is not None:
                        source_file, dependencies = annotation
                        pathway.source_file = source_file
                        pathway.dependencies = list(dependencies)
                    elif pathway.source_file in previous_mapped_paths:
                        pathway.source_file = None
                        pathway.dependencies = []

                next_hyphae = dict(current_logical)
                next_hyphae.update(physical_hyphae)
                object.__setattr__(self, "hyphae", next_hyphae)
                self.mapped_files = mapped_files
                self._centrality = centrality
                self._critical_modules = critical_modules
                self._cross_links = cross_links
                self.infrastructure_mapped = True
                self._mapping_completed_at = time.time()
                self._mapping_last_error = None
                self._deferred_mapping_reason = None
                self._mapping_generation += 1
                self._mark_topology_mutated_locked(structure_changed=True)
                annotated = len(pathway_annotations)
                break
        else:
            raise RuntimeError(
                "logical topology changed repeatedly while publishing infrastructure map"
            )
        elapsed = time.monotonic() - start_time_map
        logger.info(
            "🍄 [MYCELIUM] 🗺️ Infrastructure Mapping COMPLETE (%.2fs): "
            "%d modules, %d physical connections, %d pathways annotated, "
            "%d critical indicators tagged.",
            elapsed, len(all_files), physical_connections, annotated, len(critical_modules)
        )
        return True

    @staticmethod
    def _build_pathway_annotations(
        pathway_skills: Dict[str, str],
        all_files: Dict[str, Path],
        dependency_graph: Dict[str, List[str]],
    ) -> Dict[str, Tuple[str, List[str]]]:
        """Build pathway-to-module annotations outside the topology lock."""
        annotations: Dict[str, Tuple[str, List[str]]] = {}
        for pathway_id, skill_name in pathway_skills.items():
            skill = skill_name.lower().replace("_", "")
            for module_key, file_path in all_files.items():
                stem = file_path.stem.lower().replace("_", "")
                module_name = module_key.lower().replace("_", "")
                if skill in module_name or skill in stem or stem in skill:
                    annotations[pathway_id] = (
                        str(file_path),
                        dependency_graph.get(module_key, []),
                    )
                    break
        return annotations

    def _foreground_mapping_deferred(self) -> bool:
        if not foreground_only_runtime():
            return False
        if bool(_ALLOW_FOREGROUND_MAPPING_FLAG.value()):
            return False
        quiet_s = float(_FOREGROUND_MAPPING_QUIET_FLAG.value())
        age_s = max(
            0.0,
            time.monotonic()
            - float(getattr(self, "_created_at_monotonic", time.monotonic())),
        )
        if age_s < max(0.0, quiet_s):
            self._deferred_mapping_reason = (
                f"foreground_quiet_window:{age_s:.1f}s/{max(0.0, quiet_s):.1f}s"
            )
            logger.info(
                "🍄 [MYCELIUM] Infrastructure mapping deferred (%s).",
                self._deferred_mapping_reason,
            )
            return True
        return False

    def _extract_imports(self, file_path: Path, base_dir: Path) -> List[str]:
        """Parse a Python file's AST and extract import targets as dotted module keys."""
        imports: List[str] = []
        try:
            source = file_path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source, filename=str(file_path))
        except (SyntaxError, UnicodeDecodeError, OSError) as e:
            logger.debug("🍄 [MYCELIUM] AST parse failed for %s: %s", file_path.name, e)
            return imports

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Resolve relative imports
                    if node.level > 0:
                        try:
                            rel = file_path.parent.relative_to(base_dir)
                            parts = list(rel.parts)
                            # Go up 'level - 1' parents
                            if node.level > 1:
                                parts = parts[:-(node.level - 1)] if len(parts) >= node.level - 1 else parts
                            base_module = ".".join(parts)
                            full_module = f"{base_module}.{node.module}" if base_module else node.module
                            imports.append(full_module)
                        except (ValueError, IndexError):
                            imports.append(node.module)
                    else:
                        imports.append(node.module)

        return imports

    def get_mapped_files_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Return one detached infrastructure-map generation for concurrent readers."""
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is not None and owner is not self:
                return owner.get_mapped_files_snapshot()
            return self._mapped_files_snapshot_locked()

    def _mapped_files_snapshot_locked(self) -> Dict[str, Dict[str, Any]]:
        snapshot: Dict[str, Dict[str, Any]] = {}
        for module_key, module_data in self.mapped_files.items():
            detached = dict(module_data)
            imports = detached.get("imports")
            if isinstance(imports, list):
                detached["imports"] = list(imports)
            snapshot[module_key] = detached
        return snapshot

    def get_route_cache_token(self) -> tuple[int, int]:
        """Return the active topology owner's identity and structure revision."""
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is not None and owner is not self:
                return owner.get_route_cache_token()
            return id(self), self._topology_structure_revision

    def get_graph_snapshot(self) -> Dict[str, Any]:
        """Return topology and code map from one published generation."""
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is not None and owner is not self:
                return owner.get_graph_snapshot()
            return {
                "topology": self._network_topology_snapshot_locked(),
                "mapped_files": self._mapped_files_snapshot_locked(),
                "centrality": dict(self._centrality),
                "critical_modules": list(self._critical_modules),
                "mapping_generation": self._mapping_generation,
                "mapping_state": self._mapping_state_locked(),
                "mapping_last_error": self._mapping_last_error,
                "topology_revision": self._topology_revision,
                "topology_structure_revision": self._topology_structure_revision,
            }

    def get_runtime_snapshot(self) -> Dict[str, Any]:
        """Return the complete API read model under one topology lock."""
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is not None and owner is not self:
                return owner.get_runtime_snapshot()
            return {
                "topology": self._network_topology_snapshot_locked(),
                "infrastructure": self._infrastructure_report_snapshot_locked(),
            }

    def get_topology_counts(self) -> Dict[str, int]:
        """Return the atomically replaced count read model without graph copying."""
        owner = MycelialNetwork._instance
        owner_stop = getattr(owner, "_stop_event", None)
        if (
            owner is not None
            and owner is not self
            and owner_stop is not None
            and not owner_stop.is_set()
        ):
            return owner.get_topology_counts()
        return dict(self._topology_counts_cache)

    def get_topology_summary(self) -> Dict[str, int]:
        """Return the precomputed user-facing topology summary lock-free."""
        owner = MycelialNetwork._instance
        owner_stop = getattr(owner, "_stop_event", None)
        if (
            owner is not None
            and owner is not self
            and owner_stop is not None
            and not owner_stop.is_set()
        ):
            return owner.get_topology_summary()
        return dict(self._topology_summary_cache)

    def get_hypha_signal_snapshot(self, *, limit: int) -> List[Tuple[float, float]]:
        """Return detached strength/recency inputs for bounded numeric consumers."""
        bounded_limit = max(0, int(limit))
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is not None and owner is not self:
                return owner.get_hypha_signal_snapshot(limit=bounded_limit)
            return [
                (float(hypha.strength), float(hypha.last_pulse))
                for hypha in islice(self.hyphae.values(), bounded_limit)
            ]

    def _mapping_state_locked(self) -> str:
        if self._is_mapping:
            return "refreshing" if self.infrastructure_mapped else "running"
        if self.infrastructure_mapped:
            return "ready_with_refresh_error" if self._mapping_last_error else "ready"
        if self._mapping_last_error:
            return "failed"
        if self._deferred_mapping_reason:
            return "deferred"
        return "idle"

    def get_infrastructure_report(self) -> Dict[str, Any]:
        """Return a summary of the infrastructure mapping for API/UI consumption."""
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is not None and owner is not self:
                return owner.get_infrastructure_report()
            return self._infrastructure_report_snapshot_locked()

    def _infrastructure_report_snapshot_locked(self) -> Dict[str, Any]:
        mapped_files = self._mapped_files_snapshot_locked()
        physical_hyphae = {
            name: {
                "source": hypha.source,
                "target": hypha.target,
                "source_file": hypha.source_file,
                "target_file": hypha.target_file,
                "strength": float(round(hypha.strength, 2)),
            }
            for name, hypha in self.hyphae.items()
            if hypha.is_physical
        }
        annotated_pathways = [
            pathway.pathway_id
            for pathway in self.pathways.values()
            if pathway.source_file
        ]
        return {
            "mapped": self.infrastructure_mapped,
            "mapping_state": self._mapping_state_locked(),
            "mapping_generation": self._mapping_generation,
            "topology_revision": self._topology_revision,
            "topology_structure_revision": self._topology_structure_revision,
            "deferred_reason": self._deferred_mapping_reason,
            "mapping_started_at": self._mapping_started_at,
            "mapping_completed_at": self._mapping_completed_at,
            "mapping_last_error": self._mapping_last_error,
            "total_modules": len(mapped_files),
            "physical_connections": len(physical_hyphae),
            "annotated_pathways": annotated_pathways,
            "critical_modules": [
                {"module": module, "centrality": self._centrality.get(module, 0)}
                for module in self._critical_modules
            ],
            "cross_layer_links": {
                logical: len(physical_list)
                for logical, physical_list in self._cross_links.items()
            },
            "modules": {k: v["path"] for k, v in mapped_files.items()},
            "physical_hyphae_sample": dict(list(physical_hyphae.items())[:20]),
            "vault_sync": {
                "revision": self._last_vault_sync_revision,
                "committed_at": self._last_vault_sync_at,
                "lag_revisions_at_commit": self._last_vault_sync_lag_revisions,
            },
        }

    # ======================================================================
    # MAINTENANCE — Background Health
    # ======================================================================

    @staticmethod
    def _foreground_defers_pulse() -> bool:
        """Hypha maintenance can always wait 30s; conversation cannot."""
        try:
            from core.runtime.foreground_guard import foreground_activity_reason

            return bool(foreground_activity_reason())
        except (ImportError, RuntimeError, AttributeError):
            return False

    async def _pulse_once(self):
        """One pulse pass: refresh critical hyphae, report weak pathways."""
        owner = self._active_owner()
        if owner is None:
            return
        if owner is not self:
            await owner._pulse_once()
            return
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        async with self._async_lock:
            now = time.monotonic()
            weak_pathways: List[Tuple[str, float]] = []
            with MycelialNetwork._lock:
                owner = self._active_owner_locked()
                if owner is None:
                    return
                if owner is not self:
                    reroute = owner
                else:
                    reroute = None
                heartbeat_changed = False
                if reroute is not None:
                    weak_pathways = []
                else:
                    for name, hypha in self.hyphae.items():
                        if (
                            now - hypha.last_pulse > 300
                            and hypha.priority >= 1.0
                            and self._should_monitor_hypha(hypha)
                        ):
                            last_alert = self._hypha_alert_times.get(name, 0.0)
                            if now - last_alert > 300:
                                logger.warning(
                                    "🍄 [MYCELIUM] Hypha inactive: %s. Auto-pulsing.",
                                    name,
                                )
                                self._hypha_alert_times[name] = now
                            hypha.refresh_heartbeat()
                            heartbeat_changed = True

                    weak_pathways = [
                        (pathway_id, pathway.confidence)
                        for pathway_id, pathway in self.pathways.items()
                        if pathway.is_weak
                        and pathway.hit_count + pathway.miss_count > 5
                    ]
                    if heartbeat_changed:
                        self._mark_topology_mutated_locked()
            if reroute is not None:
                await reroute._pulse_once()
                return
            for pathway_id, confidence in weak_pathways:
                logger.warning(
                    "🍄 [MYCELIUM] Weak pathway detected: '%s' (confidence=%.2f)",
                    pathway_id,
                    confidence,
                )

    async def pulse_check(self):
        """Periodic background check to keep critical hyphae alive and prune weak pathways."""
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()

        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(30)
                if self._foreground_defers_pulse():
                    # Auto-pulse log bursts were firing mid-conversation in
                    # the 110GB-incident transcript; maintenance waits.
                    continue
                await self._pulse_once()
            except asyncio.CancelledError:
                # Cleanup for MemoryGovernor if it's running
                if hasattr(self, '_task') and self._task:
                    self._task.cancel()
                    try:
                        await self._task
                    except asyncio.CancelledError as _e:
                        logger.debug('Ignored asyncio.CancelledError in mycelium.py: %s', _e)
                    finally:
                        self._task = None
                
                # v8.1.0: Ensure total cleanup of any leaked worker handles
                try:
                    if hasattr(self, '_critical_cleanup') and callable(self._critical_cleanup):
                        await self._critical_cleanup()
                        logger.info("🛡️ Memory Governor shutdown complete. All worker handles leaked/active were purged.")
                except (RuntimeError, AttributeError, TypeError) as e:
                    record_degradation('mycelium', e)
                    logger.error("Error during Memory Governor shutdown: %s", e)
                logger.info("🍄 [MYCELIUM] Pulse check loop shutting down.")
                break
            except (OSError, ConnectionError, TimeoutError) as e:
                record_degradation('mycelium', e)
                logger.error("🍄 [MYCELIUM] Pulse check error: %s", e, exc_info=True)
                await asyncio.sleep(10)  # Backoff on error

    # ======================================================================
    # INTROSPECTION — Topology & Health Reporting
    # ======================================================================

    def get_network_topology(self) -> Dict[str, Any]:
        """Full network state for UI visualization and health monitoring."""
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is not None and owner is not self:
                return owner.get_network_topology()
            return self._network_topology_snapshot_locked()

    def _network_topology_snapshot_locked(self) -> Dict[str, Any]:
        pathways = {
            pathway_id: pathway.to_dict()
            for pathway_id, pathway in self.pathways.items()
        }
        hyphae = {
            name: hypha.model_dump()
            for name, hypha in self.hyphae.items()
        }
        cross_layer_linked = len(self._cross_links)
        infrastructure_mapped = self.infrastructure_mapped
        critical_modules = list(self._critical_modules[:10])
        discovery_candidates = dict(self._discovery_candidates)
        ui_connected = self.ui_callback is not None
        physical_count = sum(
            1 for hypha in hyphae.values() if hypha.get("is_physical")
        )
        logical_count = len(hyphae) - physical_count
        strengths = [float(hypha.get("strength", 0.0)) for hypha in hyphae.values()] or [0.0]
        confidences = [
            float(pathway.get("confidence", 1.0))
            for pathway in pathways.values()
        ] or [1.0]

        return {
            "pathways": pathways,
            "pathway_count": len(pathways),
            "hyphae": hyphae,
            "topology_revision": self._topology_revision,
            "topology_structure_revision": self._topology_structure_revision,
            "hyphae_summary": {
                "total": len(hyphae),
                "logical": logical_count,
                "physical": physical_count,
                "cross_layer_linked": cross_layer_linked,
                "infrastructure_mapped": infrastructure_mapped,
            },
            "critical_modules": critical_modules,
            "discovery_candidates": discovery_candidates,
            "ui_connected": ui_connected,
            "system_cohesion": round(
                sum(strengths + confidences) / max(len(strengths + confidences), 1),
                3,
            ),
            "total_pathway_hits": sum(
                int(pathway.get("hit_count", 0)) for pathway in pathways.values()
            ),
            "total_pathway_misses": sum(
                int(pathway.get("miss_count", 0)) for pathway in pathways.values()
            ),
        }

    def get_unity_report(self) -> Dict[str, Any]:
        """Backward-compatible unity report."""
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is not None and owner is not self:
                return owner.get_unity_report()
            hyphae = {
                name: {
                    "strength": hypha.strength,
                    "last_active": time.monotonic() - hypha.last_pulse,
                }
                for name, hypha in self.hyphae.items()
            }
            pathway_count = len(self.pathways)
            pathway_confidences = [
                pathway.confidence for pathway in self.pathways.values()
            ] or [1.0]
            ui_connected = self.ui_callback is not None
        strengths = [entry["strength"] for entry in hyphae.values()] or [0.0]
        return {
            "hyphae": hyphae,
            "pathways": pathway_count,
            "ui_connected": ui_connected,
            "system_cohesion": round(
                sum(strengths + pathway_confidences)
                / max(len(strengths + pathway_confidences), 1),
                3,
            ),
        }

    def get_system_cohesion(self) -> float:
        """Return the active owner's detached system-cohesion read model."""
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                raise RuntimeError("retired mycelium instance has no active owner")
            if owner is not None and owner is not self:
                return owner.get_system_cohesion()
            strengths = [h.strength for h in self.hyphae.values()] or [0.0]
            confidences = [pw.confidence for pw in self.pathways.values()] or [1.0]
        all_values = strengths + confidences
        return round(sum(all_values) / max(len(all_values), 1), 3)

    def _calculate_cohesion(self) -> float:
        """Backward-compatible internal alias for the owner-backed read API."""
        return self.get_system_cohesion()

    # ======================================================================
    # PILLAR 3: THE ROOT VAULT (Aegis Persistence)
    # ======================================================================

    def _vault_snapshot_locked(self) -> Dict[str, Any]:
        now_monotonic = time.monotonic()
        captured_at_unix = time.time()
        pathways: Dict[str, Dict[str, Any]] = {}
        for key, pathway in self.pathways.items():
            data = pathway.to_dict()
            data.pop("id", None)
            created_at = self._vault_number(
                pathway.created_at,
                f"live pathway creation timestamp: {key}",
                minimum=0.0,
                maximum=captured_at_unix + _VAULT_CLOCK_SKEW_TOLERANCE_S,
            )
            last_matched = self._vault_number(
                pathway.last_matched,
                f"live pathway last-matched timestamp: {key}",
                minimum=0.0,
                maximum=now_monotonic + _VAULT_CLOCK_SKEW_TOLERANCE_S,
            )
            created_age = max(0.0, captured_at_unix - created_at)
            last_matched_age = max(0.0, now_monotonic - last_matched)
            if last_matched_age > created_age + _VAULT_CLOCK_SKEW_TOLERANCE_S:
                raise ValueError(
                    f"live pathway last match predates creation: {key}"
                )
            data["created_at"] = created_at
            data["last_matched_age_s"] = last_matched_age
            data.pop("last_matched", None)
            pathways[key] = data

        hyphae: Dict[str, Dict[str, Any]] = {}
        for key, hypha in self.hyphae.items():
            data = hypha.model_dump()
            created_at = self._vault_number(
                hypha.created_at,
                f"live hypha creation timestamp: {key}",
                minimum=0.0,
                maximum=now_monotonic + _VAULT_CLOCK_SKEW_TOLERANCE_S,
            )
            last_pulse = self._vault_number(
                hypha.last_pulse,
                f"live hypha pulse timestamp: {key}",
                minimum=0.0,
                maximum=now_monotonic + _VAULT_CLOCK_SKEW_TOLERANCE_S,
            )
            created_age = max(0.0, now_monotonic - created_at)
            last_pulse_age = max(0.0, now_monotonic - last_pulse)
            if last_pulse_age > created_age + _VAULT_CLOCK_SKEW_TOLERANCE_S:
                raise ValueError(f"live hypha pulse predates creation: {key}")
            data["created_age_s"] = created_age
            data["last_pulse_age_s"] = last_pulse_age
            data.pop("created_at", None)
            data.pop("last_pulse", None)
            hyphae[key] = data

        return {
            "schema_version": 3,
            "captured_at_unix": captured_at_unix,
            "pathways": pathways,
            "hyphae": hyphae,
            "mapped_files": self._mapped_files_snapshot_locked(),
            "centrality": dict(self._centrality),
            "critical_modules": list(self._critical_modules),
            "cross_links": {
                key: list(value) for key, value in self._cross_links.items()
            },
            "infrastructure_mapped": self.infrastructure_mapped,
            "mapping_generation": self._mapping_generation,
            "topology_revision": self._topology_revision,
        }

    @staticmethod
    def _vault_age(value: Any, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be a finite non-negative number")
        age = float(value)
        if not math.isfinite(age) or age < 0.0:
            raise ValueError(f"{label} must be a finite non-negative number")
        return age

    @staticmethod
    def _vault_number(
        value: Any,
        label: str,
        *,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be a finite number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label} must be a finite number")
        if minimum is not None and number < minimum:
            raise ValueError(f"{label} is below its minimum")
        if maximum is not None and number > maximum:
            raise ValueError(f"{label} exceeds its maximum")
        return number

    @staticmethod
    def _vault_count(value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
        return value

    @staticmethod
    def _vault_optional_string(value: Any, label: str) -> Optional[str]:
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{label} must be a string or null")
        return value

    @classmethod
    def _restore_pathways(
        cls,
        raw: Any,
        *,
        now_monotonic: float,
        captured_at_unix: float,
        elapsed_since_capture_s: float,
    ) -> Dict[str, HardwiredPathway]:
        if not isinstance(raw, dict):
            raise ValueError("vault pathways must be an object")
        allowed = {
            "pathway_id", "pattern", "skill_name", "param_map", "priority",
            "source_file", "dependencies", "confidence", "activity_label",
            "hit_count", "miss_count", "created_at", "last_matched_age_s",
            "direct_response", "color", "description", "size",
        }
        restored: Dict[str, HardwiredPathway] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                raise ValueError("vault pathway entries must be named objects")
            unknown = set(value) - allowed
            if unknown:
                raise ValueError(f"vault pathway contains unknown fields: {key}")
            fields = {name: item for name, item in value.items() if name in allowed}
            if str(fields.get("pathway_id") or "") != key:
                raise ValueError(f"vault pathway identity mismatch: {key}")
            pattern = fields.get("pattern")
            if not isinstance(pattern, str) or not pattern:
                raise ValueError(f"vault pathway pattern is missing: {key}")
            try:
                fields["pattern"] = re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                raise ValueError(f"vault pathway regex is invalid: {key}") from exc
            skill_name = fields.get("skill_name")
            if not isinstance(skill_name, str) or not skill_name:
                raise ValueError(f"vault pathway skill is missing: {key}")
            param_map = fields.get("param_map", {})
            if not isinstance(param_map, dict) or any(
                not isinstance(name, str)
                or not name
                or isinstance(mapping, bool)
                or not isinstance(mapping, (int, str))
                or (isinstance(mapping, int) and mapping < 0)
                for name, mapping in param_map.items()
            ):
                raise ValueError(f"vault pathway parameter map is malformed: {key}")
            dependencies = fields.get("dependencies", [])
            if not isinstance(dependencies, list) or any(
                not isinstance(dependency, str) for dependency in dependencies
            ):
                raise ValueError(f"vault pathway dependencies are malformed: {key}")
            fields["priority"] = cls._vault_number(
                fields.get("priority", 1.0),
                f"vault pathway priority: {key}",
                minimum=0.0,
            )
            fields["confidence"] = cls._vault_number(
                fields.get("confidence", 1.0),
                f"vault pathway confidence: {key}",
                minimum=0.0,
                maximum=10.0,
            )
            fields["size"] = cls._vault_number(
                fields.get("size", 1.0),
                f"vault pathway size: {key}",
                minimum=0.0,
            )
            fields["created_at"] = cls._vault_number(
                fields.get("created_at"),
                f"vault pathway creation timestamp: {key}",
                minimum=0.0,
                maximum=captured_at_unix + _VAULT_CLOCK_SKEW_TOLERANCE_S,
            )
            fields["hit_count"] = cls._vault_count(
                fields.get("hit_count", 0), f"vault pathway hit count: {key}"
            )
            fields["miss_count"] = cls._vault_count(
                fields.get("miss_count", 0), f"vault pathway miss count: {key}"
            )
            for field_name in ("activity_label", "color", "description"):
                if not isinstance(fields.get(field_name, ""), str):
                    raise ValueError(
                        f"vault pathway {field_name} is malformed: {key}"
                    )
            fields["source_file"] = cls._vault_optional_string(
                fields.get("source_file"), f"vault pathway source file: {key}"
            )
            fields["direct_response"] = cls._vault_optional_string(
                fields.get("direct_response"),
                f"vault pathway direct response: {key}",
            )
            last_matched_age = cls._vault_age(
                fields.pop("last_matched_age_s", None),
                f"vault pathway last-matched age: {key}",
            )
            created_age = max(0.0, captured_at_unix - fields["created_at"])
            if (
                last_matched_age
                > created_age + _VAULT_CLOCK_SKEW_TOLERANCE_S
            ):
                raise ValueError(
                    f"vault pathway last match predates creation: {key}"
                )
            fields["last_matched"] = (
                now_monotonic - last_matched_age - elapsed_since_capture_s
            )
            restored[key] = HardwiredPathway(**fields)
        return restored

    @classmethod
    def _restore_hyphae(
        cls,
        raw: Any,
        *,
        now_monotonic: float,
        elapsed_since_capture_s: float,
    ) -> Dict[str, Hypha]:
        if not isinstance(raw, dict):
            raise ValueError("vault hyphae must be an object")
        allowed = {
            "name", "source", "target", "priority", "strength", "created_age_s",
            "last_pulse_age_s", "pulse_count", "active", "is_physical", "source_file",
            "target_file", "color", "description", "size", "trace",
            "hardware_id", "pinned",
        }
        restored: Dict[str, Hypha] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                raise ValueError("vault hypha entries must be named objects")
            unknown = set(value) - allowed
            if unknown:
                raise ValueError(f"vault hypha contains unknown fields: {key}")
            fields = {name: item for name, item in value.items() if name in allowed}
            if str(fields.get("name") or "") != key:
                raise ValueError(f"vault hypha identity mismatch: {key}")
            if not isinstance(fields.get("source"), str) or not fields["source"]:
                raise ValueError(f"vault hypha source is missing: {key}")
            if not isinstance(fields.get("target"), str) or not fields["target"]:
                raise ValueError(f"vault hypha target is missing: {key}")
            fields["priority"] = cls._vault_number(
                fields.get("priority", 1.0),
                f"vault hypha priority: {key}",
                minimum=0.0,
            )
            fields["strength"] = cls._vault_number(
                fields.get("strength", 1.0),
                f"vault hypha strength: {key}",
                minimum=0.1,
                maximum=10.0,
            )
            fields["size"] = cls._vault_number(
                fields.get("size", 1.0),
                f"vault hypha size: {key}",
                minimum=0.0,
            )
            fields["pulse_count"] = cls._vault_count(
                fields.get("pulse_count", 0), f"vault hypha pulse count: {key}"
            )
            for field_name in ("active", "is_physical"):
                if not isinstance(fields.get(field_name, False), bool):
                    raise ValueError(f"vault hypha {field_name} is malformed: {key}")
            for field_name in ("source_file", "target_file"):
                fields[field_name] = cls._vault_optional_string(
                    fields.get(field_name), f"vault hypha {field_name}: {key}"
                )
            for field_name in ("color", "description"):
                if not isinstance(fields.get(field_name, ""), str):
                    raise ValueError(f"vault hypha {field_name} is malformed: {key}")
            trace = fields.get("trace", [])
            if not isinstance(trace, list) or any(
                not isinstance(entry, str) for entry in trace
            ):
                raise ValueError(f"vault hypha trace is malformed: {key}")
            created_age = cls._vault_age(
                fields.pop("created_age_s", None),
                f"vault hypha creation age: {key}",
            )
            last_pulse_age = cls._vault_age(
                fields.pop("last_pulse_age_s", None),
                f"vault hypha pulse age: {key}",
            )
            if last_pulse_age > created_age + _VAULT_CLOCK_SKEW_TOLERANCE_S:
                raise ValueError(f"vault hypha pulse predates creation: {key}")
            fields["created_at"] = (
                now_monotonic - created_age - elapsed_since_capture_s
            )
            fields["last_pulse"] = (
                now_monotonic - last_pulse_age - elapsed_since_capture_s
            )
            is_neural_root = "hardware_id" in fields or "pinned" in fields
            if is_neural_root:
                hardware_id = fields.get("hardware_id")
                if not isinstance(hardware_id, str) or not hardware_id:
                    raise ValueError(f"vault neural-root hardware id is malformed: {key}")
                if not isinstance(fields.get("pinned"), bool):
                    raise ValueError(f"vault neural-root pinned flag is malformed: {key}")
                if fields["target"] != f"hardware:{hardware_id}":
                    raise ValueError(f"vault neural-root target is malformed: {key}")
            model = NeuralRoot if is_neural_root else Hypha
            restored[key] = model(**fields)
        return restored

    @classmethod
    def _decode_vault_topology(cls, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict) or payload.get("schema_version") != 3:
            raise ValueError("unsupported mycelium vault schema")
        if set(payload) != {
            "schema_version",
            "captured_at_unix",
            "pathways",
            "hyphae",
            "mapped_files",
            "centrality",
            "critical_modules",
            "cross_links",
            "infrastructure_mapped",
            "mapping_generation",
            "topology_revision",
        }:
            raise ValueError("mycelium vault fields are invalid")
        captured_at = payload.get("captured_at_unix")
        if (
            isinstance(captured_at, bool)
            or not isinstance(captured_at, (int, float))
            or not math.isfinite(float(captured_at))
            or float(captured_at) <= 0.0
        ):
            raise ValueError("vault capture timestamp is malformed")
        now_monotonic = time.monotonic()
        now_unix = time.time()
        if float(captured_at) > now_unix + _VAULT_CLOCK_SKEW_TOLERANCE_S:
            raise ValueError("vault capture timestamp is implausibly in the future")
        elapsed_since_capture_s = max(0.0, now_unix - float(captured_at))
        pathways = cls._restore_pathways(
            payload.get("pathways"),
            now_monotonic=now_monotonic,
            captured_at_unix=float(captured_at),
            elapsed_since_capture_s=elapsed_since_capture_s,
        )
        hyphae = cls._restore_hyphae(
            payload.get("hyphae"),
            now_monotonic=now_monotonic,
            elapsed_since_capture_s=elapsed_since_capture_s,
        )
        raw_mapped_files = payload.get("mapped_files")
        centrality = payload.get("centrality")
        critical_modules = payload.get("critical_modules")
        cross_links = payload.get("cross_links")
        if not isinstance(raw_mapped_files, dict):
            raise ValueError("vault mapped-files surface is malformed")
        mapped_files: Dict[str, Dict[str, Any]] = {}
        mapped_paths: set[str] = set()
        for key, value in raw_mapped_files.items():
            if not isinstance(key, str) or not key or not isinstance(value, dict):
                raise ValueError("vault mapped-files surface is malformed")
            path = value.get("path")
            imports = value.get("imports")
            size_bytes = value.get("size_bytes")
            module_centrality = value.get("centrality")
            is_critical = value.get("is_critical")
            if set(value) != {
                "path",
                "size_bytes",
                "imports",
                "centrality",
                "is_critical",
            }:
                raise ValueError(f"vault mapped-file fields are invalid: {key}")
            if not isinstance(path, str) or not path or not Path(path).is_absolute():
                raise ValueError(f"vault mapped-file path is malformed: {key}")
            if path in mapped_paths:
                raise ValueError(f"vault mapped-file path is duplicated: {key}")
            if not isinstance(imports, list) or any(
                not isinstance(dependency, str) for dependency in imports
            ):
                raise ValueError(f"vault mapped-file imports are malformed: {key}")
            if (
                isinstance(size_bytes, bool)
                or not isinstance(size_bytes, int)
                or size_bytes < 0
            ):
                raise ValueError(f"vault mapped-file size is malformed: {key}")
            if (
                isinstance(module_centrality, bool)
                or not isinstance(module_centrality, int)
                or module_centrality < 0
            ):
                raise ValueError(f"vault mapped-file centrality is malformed: {key}")
            if not isinstance(is_critical, bool):
                raise ValueError(f"vault mapped-file critical flag is malformed: {key}")
            mapped_paths.add(path)
            mapped_files[key] = {
                "path": path,
                "size_bytes": size_bytes,
                "imports": list(imports),
                "centrality": module_centrality,
                "is_critical": is_critical,
            }
        if not isinstance(centrality, dict) or any(
            key not in mapped_files
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for key, value in centrality.items()
        ):
            raise ValueError("vault centrality surface is malformed")
        if not isinstance(critical_modules, list) or any(
            not isinstance(module, str) or module not in mapped_files
            for module in critical_modules
        ):
            raise ValueError("vault critical-module surface is malformed")
        if len(set(critical_modules)) != len(critical_modules):
            raise ValueError("vault critical-module surface contains duplicates")
        computed_centrality: Dict[str, int] = {}
        for module in mapped_files.values():
            for dependency in module["imports"]:
                if dependency in mapped_files:
                    computed_centrality[dependency] = (
                        computed_centrality.get(dependency, 0) + 1
                    )
        if any(
            module["centrality"] != computed_centrality.get(key, 0)
            for key, module in mapped_files.items()
        ):
            raise ValueError("vault module centrality disagrees with its import graph")
        expected_centrality = dict(computed_centrality)
        if centrality != expected_centrality:
            raise ValueError("vault centrality disagrees with the module map")
        expected_critical = {
            key for key, value in mapped_files.items() if value["is_critical"]
        }
        if set(critical_modules) != expected_critical:
            raise ValueError("vault critical modules disagree with the module map")
        ranked_centralities = sorted(computed_centrality.values(), reverse=True)
        expected_critical_count = min(20, len(ranked_centralities))
        if len(critical_modules) != expected_critical_count:
            raise ValueError("vault critical modules disagree with centrality ranking")
        if ranked_centralities:
            cutoff = ranked_centralities[expected_critical_count - 1]
            if any(
                computed_centrality.get(module, 0) < cutoff
                for module in critical_modules
            ) or any(
                value > cutoff and module not in expected_critical
                for module, value in computed_centrality.items()
            ):
                raise ValueError("vault critical modules disagree with centrality ranking")
        if not isinstance(cross_links, dict):
            raise ValueError("vault cross-link surface is malformed")
        for logical_name, physical_names in cross_links.items():
            if (
                not isinstance(logical_name, str)
                or logical_name not in hyphae
                or hyphae[logical_name].is_physical
                or not isinstance(physical_names, list)
                or any(not isinstance(name, str) for name in physical_names)
                or len(set(physical_names)) != len(physical_names)
            ):
                raise ValueError("vault cross-link owner is malformed")
            if any(
                name not in hyphae or not hyphae[name].is_physical
                for name in physical_names
            ):
                raise ValueError("vault cross-link target is malformed")
        infrastructure_mapped = payload.get("infrastructure_mapped")
        if not isinstance(infrastructure_mapped, bool):
            raise ValueError("vault infrastructure state is malformed")
        physical_hyphae = [hypha for hypha in hyphae.values() if hypha.is_physical]
        if infrastructure_mapped and not mapped_files:
            raise ValueError("mapped vault contains no modules")
        if infrastructure_mapped and any(
            hypha.source not in mapped_files or hypha.target not in mapped_files
            for hypha in physical_hyphae
        ):
            raise ValueError("vault physical topology is not backed by its module map")
        if not infrastructure_mapped and (mapped_files or physical_hyphae):
            raise ValueError("unmapped vault contains published physical topology")
        expected_physical_names = {
            f"import:{source}->{target}"
            for source, module in mapped_files.items()
            for target in module["imports"]
            if target in mapped_files
        }
        actual_physical_names = {
            name for name, hypha in hyphae.items() if hypha.is_physical
        }
        if actual_physical_names != expected_physical_names:
            raise ValueError("vault physical topology disagrees with the module map")
        for name, hypha in hyphae.items():
            expected_name = (
                f"import:{hypha.source}->{hypha.target}"
                if hypha.is_physical
                else f"{hypha.source}->{hypha.target}"
            )
            if name != expected_name:
                raise ValueError(f"vault hypha identity is inconsistent: {name}")
            if hypha.is_physical and (
                hypha.source_file != mapped_files[hypha.source]["path"]
                or hypha.target_file != mapped_files[hypha.target]["path"]
            ):
                raise ValueError(f"vault physical hypha files are inconsistent: {name}")
        for pathway in pathways.values():
            if pathway.source_file is not None and pathway.source_file not in mapped_paths:
                raise ValueError(
                    f"vault pathway source is outside the module map: {pathway.pathway_id}"
                )
            if pathway.source_file is not None:
                module = next(
                    value
                    for value in mapped_files.values()
                    if value["path"] == pathway.source_file
                )
                if pathway.dependencies != module["imports"]:
                    raise ValueError(
                        "vault pathway dependencies disagree with its source module: "
                        f"{pathway.pathway_id}"
                    )
        mapping_generation = payload.get("mapping_generation")
        if (
            isinstance(mapping_generation, bool)
            or not isinstance(mapping_generation, int)
            or mapping_generation < 0
        ):
            raise ValueError("vault mapping generation is negative")
        topology_revision = payload.get("topology_revision")
        if (
            isinstance(topology_revision, bool)
            or not isinstance(topology_revision, int)
            or topology_revision < 0
        ):
            raise ValueError("vault topology revision is malformed")
        return {
            "pathways": pathways,
            "hyphae": hyphae,
            "mapped_files": mapped_files,
            "centrality": dict(centrality),
            "critical_modules": list(critical_modules),
            "cross_links": {
                key: list(value) for key, value in cross_links.items()
            },
            "infrastructure_mapped": infrastructure_mapped,
            "mapping_generation": mapping_generation,
            "topology_revision": topology_revision,
        }

    async def vault_sync(self) -> bool:
        """Persist one complete, versioned topology generation."""
        import json
        import sqlite3

        from core.config import config

        aegis_cfg = getattr(config, "aegis", None)
        vault_path = (
            getattr(aegis_cfg, "vault_path", None)
            or "data/mycelium_vault.db"
        )
        db_path = config.paths.base_dir / vault_path

        def _sync_worker() -> tuple[int, int]:
            with MycelialNetwork._vault_io_lock:
                with MycelialNetwork._lock:
                    if (
                        MycelialNetwork._instance is not self
                        or self._stop_event.is_set()
                    ):
                        raise RuntimeError(
                            "retired mycelium instance cannot write the root vault"
                        )
                    topology = self._vault_snapshot_locked()
                    snapshot_revision = int(topology["topology_revision"])
                # Validate our own serialized contract before replacing the last
                # known-good generation.
                self._decode_vault_topology(topology)
                encoded = json.dumps(topology, allow_nan=False, sort_keys=True)
                db_path.parent.mkdir(parents=True, exist_ok=True)
                with sqlite3.connect(db_path) as conn:
                    conn.execute("PRAGMA busy_timeout=5000;")
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("PRAGMA synchronous=FULL;")
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS aegis_vault "
                        "(key TEXT PRIMARY KEY, data TEXT, timestamp REAL)"
                    )
                    conn.execute("BEGIN IMMEDIATE")
                    conn.execute(
                        "REPLACE INTO aegis_vault (key, data, timestamp) "
                        "VALUES (?, ?, ?)",
                        ("topology_v3", encoded, time.time()),
                    )
                    with MycelialNetwork._lock:
                        if (
                            MycelialNetwork._instance is not self
                            or self._stop_event.is_set()
                        ):
                            raise RuntimeError(
                                "mycelium retired before root-vault commit"
                            )
                        current_revision = self._topology_revision
                        conn.commit()
                        self._last_vault_sync_revision = snapshot_revision
                        self._last_vault_sync_at = time.time()
                        self._last_vault_sync_lag_revisions = max(
                            0,
                            current_revision - snapshot_revision,
                        )
                        return snapshot_revision, current_revision

        try:
            snapshot_revision, current_revision = await run_io_bound(_sync_worker)
        except (sqlite3.Error, OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("mycelium", exc)
            logger.error("🛡️ AEGIS: Vault Sync Failed! %s", exc)
            return False
        if current_revision != snapshot_revision:
            logger.debug(
                "🛡️ AEGIS: Vault committed coherent revision %d while live topology "
                "advanced to %d; the next interval will capture the newer state.",
                snapshot_revision,
                current_revision,
            )
        logger.debug("🛡️ AEGIS: Vault Sync Complete.")
        return True

    @classmethod
    async def restore_from_vault(cls) -> bool:
        """Validate and atomically publish one persisted topology generation."""
        import json
        import sqlite3

        from core.config import config

        aegis_cfg = getattr(config, "aegis", None)
        vault_path = (
            getattr(aegis_cfg, "vault_path", None)
            or "data/mycelium_vault.db"
        )
        db_path = config.paths.base_dir / vault_path
        if not db_path.exists():
            logger.critical("🛡️ AEGIS FATAL: Cannot restore; Root Vault missing!")
            return False

        with cls._lock:
            target_instance = cls._instance
            if target_instance is None:
                logger.critical(
                    "🛡️ AEGIS: Restoration aborted — Mycelium is not initialized."
                )
                return False
            mapping_thread = target_instance._mapping_thread
            if target_instance._is_mapping or (
                mapping_thread is not None and mapping_thread.is_alive()
            ):
                logger.critical(
                    "🛡️ AEGIS: Restoration deferred while a map generation is active."
                )
                return False
            if target_instance._stop_event.is_set():
                logger.critical("🛡️ AEGIS: Restoration refused during shutdown.")
                return False
            target_revision = target_instance._topology_revision

        def _restore_worker() -> None:
            # Vault and topology publication use the same lock order as sync:
            # vault first, then topology. The vault lock remains held until the
            # decoded generation is either rejected or fully published.
            with cls._vault_io_lock:
                with sqlite3.connect(db_path) as conn:
                    row = conn.execute(
                        "SELECT data FROM aegis_vault WHERE key = ?",
                        ("topology_v3",),
                    ).fetchone()
                if not row:
                    raise ValueError("versioned topology generation is missing")
                topology = cls._decode_vault_topology(json.loads(row[0]))

                with cls._lock:
                    instance = cls._instance
                    if instance is not target_instance:
                        raise RuntimeError("vault restoration target was replaced")
                    mapping_thread = instance._mapping_thread
                    if instance._is_mapping or (
                        mapping_thread is not None and mapping_thread.is_alive()
                    ):
                        raise RuntimeError(
                            "vault restoration raced an active map generation"
                        )
                    if instance._stop_event.is_set():
                        raise RuntimeError("vault restoration raced shutdown")
                    if instance._topology_revision != target_revision:
                        raise RuntimeError(
                            "vault restoration raced a newer in-memory topology revision"
                        )
                    object.__setattr__(instance, "pathways", topology["pathways"])
                    object.__setattr__(instance, "hyphae", topology["hyphae"])
                    object.__setattr__(
                        instance,
                        "_pathway_order",
                        sorted(
                            topology["pathways"],
                            key=lambda key: topology["pathways"][key].priority,
                            reverse=True,
                        ),
                    )
                    instance.direct_roots = {
                        key: pathway.skill_name
                        for key, pathway in topology["pathways"].items()
                    }
                    instance._neural_roots = [
                        hypha
                        for hypha in topology["hyphae"].values()
                        if isinstance(hypha, NeuralRoot)
                    ]
                    instance.mapped_files = topology["mapped_files"]
                    instance._centrality = topology["centrality"]
                    instance._critical_modules = topology["critical_modules"]
                    instance._cross_links = topology["cross_links"]
                    instance.infrastructure_mapped = topology["infrastructure_mapped"]
                    instance._mapping_generation = max(
                        instance._mapping_generation + 1,
                        topology["mapping_generation"],
                    )
                    instance._topology_revision = max(
                        instance._topology_revision + 1,
                        topology["topology_revision"] + 1,
                    )
                    instance._topology_structure_revision += 1
                    instance._mapping_completed_at = time.time()
                    instance._mapping_last_error = None
                    instance._deferred_mapping_reason = None
                    instance._publish_topology_read_models_locked()
                    object.__setattr__(instance, "_aegis_locked", True)

        logger.critical("🛡️ AEGIS: Initiating Emergency Vault Restoration...")
        try:
            await run_io_bound(_restore_worker)
        except (sqlite3.Error, OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("mycelium", exc)
            logger.critical("🛡️ AEGIS FATAL: Restoration Failed! %s", exc)
            return False
        logger.critical("🛡️ AEGIS: Restoration Successful. Mycelium Unity restored.")
        return True
