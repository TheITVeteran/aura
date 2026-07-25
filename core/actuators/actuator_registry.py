"""core/actuators/actuator_registry.py
===================================
Open-Ended Actuators & Action Primitives.

Implements executable physical commands that modify the state of entities
in the PhysicsWorldModel. All operations are sandboxed and validated.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

# Bounds for synthesized world-model mutations. LLM-generated actuator output
# is untrusted, so both the fan-out (how many entities/fields it can touch)
# and individual values are capped to protect the live world model.
_MAX_SYNTH_ENTITIES = 256
_MAX_SYNTH_ATTRS_PER_ENTITY = 64
_MAX_SYNTH_STRING_LEN = 512
_MAX_SAFE_INT = 2**53  # exact-integer ceiling; beyond this floats lose precision
_SANDBOX_TIMEOUT_MIN_S = 0.5
_SANDBOX_TIMEOUT_MAX_S = 120.0

from core.runtime.task_ownership import (
    drain_owned_awaitable,
    runtime_shutdown_blocks_new_work,
)

logger = logging.getLogger("Aura.Actuators")


@dataclass
class ActuatorResult:
    """The result of executing an action primitive."""

    success: bool
    message: str
    updates: dict[str, Any]


def _finite_float(
    value: Any, *, minimum: float | None = None, maximum: float | None = None
) -> float | None:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(candidate):
        return None
    if minimum is not None and candidate < minimum:
        return None
    if maximum is not None and candidate > maximum:
        return None
    return candidate


class BaseActuator(ABC):
    """Abstract base class for all physical open-ended actuators."""

    synthesized: bool = False
    trust_score: float = 1.0
    # Fail SAFE: an actuator that forgets to declare its authority requirement
    # is governed by default. Low-risk, in-memory-only actuators opt OUT
    # explicitly with requires_authority = False.
    requires_authority: bool = True
    generation: int = 0
    source_code: str | None = None
    blocking_execution: bool = True

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this actuator."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable explanation of what this actuator does."""
        pass

    @abstractmethod
    def validate_params(self, params: dict[str, Any]) -> bool:
        """Validates that parameters satisfy all safety and physical constraints."""
        pass

    @abstractmethod
    def execute(self, params: dict[str, Any]) -> ActuatorResult:
        """Executes the action on the PhysicsWorldModel."""
        pass


class SandboxedSynthesizedActuator(BaseActuator):
    """Live wrapper for LLM-synthesized actuator code.

    The generated code never executes in Aura's main process. Execution happens
    through the validator sandbox, then this wrapper applies only bounded,
    finite update payloads to the live physics world.
    """

    synthesized: bool = True
    # Generated code that mutates the live world model is always governed.
    requires_authority: bool = True

    def __init__(
        self,
        *,
        name: str,
        description: str,
        source_code: str,
        trust_score: float = 0.3,
    ) -> None:
        self._name = str(name).strip() or "sandboxed_synthesized_actuator"
        self._description = str(description).strip() or "Sandboxed synthesized actuator"
        self.source_code = source_code
        self.trust_score = trust_score

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def validate_params(self, params: dict[str, Any]) -> bool:
        return isinstance(params, dict)

    def execute(self, params: dict[str, Any]) -> ActuatorResult:
        if not self.validate_params(params):
            return ActuatorResult(False, "Parameter validation failed", {})

        try:
            from core.actuators.actuator_validator import ActuatorCodeValidator

            sandbox_result = ActuatorCodeValidator.execute_sandboxed(self.source_code or "", params)
            if not sandbox_result.success:
                return ActuatorResult(False, sandbox_result.error or "Sandbox execution failed", {})
            updates = sandbox_result.details.get("updates", {})
            applied_updates = self._apply_bounded_updates(updates)
            # Delivery truth: if the sandbox proposed updates but none survived
            # validation, the world did NOT change — do not report success.
            requested = bool(isinstance(updates, dict) and updates)
            if requested and not applied_updates:
                return ActuatorResult(
                    False,
                    "Sandboxed actuator produced no valid updates; no world-model change applied",
                    {},
                )
            return ActuatorResult(
                True,
                str(sandbox_result.details.get("message") or "Sandboxed actuator executed"),
                applied_updates,
            )
        except (ImportError, AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            return ActuatorResult(False, f"Sandboxed actuator execution failed: {exc}", {})

    @staticmethod
    def _bounded_primitive(value: Any) -> Any:
        """Reject non-finite floats, oversized ints, and unbounded strings."""
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, int):
            return value if abs(value) <= _MAX_SAFE_INT else None
        if isinstance(value, str):
            return value[:_MAX_SYNTH_STRING_LEN]
        return None

    def _apply_bounded_updates(self, updates: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(updates, dict) or not updates:
            return {}

        from core.world.world_model import get_physics_world_model

        model = get_physics_world_model()
        applied: dict[str, Any] = {}
        entity_count = 0
        for entity_id, fields in updates.items():
            if entity_count >= _MAX_SYNTH_ENTITIES:
                logger.warning(
                    "Synthesized update fan-out exceeded %d entities; ignoring the rest.",
                    _MAX_SYNTH_ENTITIES,
                )
                break
            entity = model.get_entity(str(entity_id))
            if entity is None or not isinstance(fields, dict):
                continue
            entity_count += 1

            # Snapshot the fields this update may touch so a constraint failure
            # or an invalid later field rolls THIS entity back to its prior
            # state instead of leaving a half-applied mutation committed.
            snapshot_fields = ("capacity", "load", "flow_rate", "max_flow_rate", "latency", "coordinates")
            before = {f: getattr(entity, f, None) for f in snapshot_fields}
            before_attrs = dict(getattr(entity, "attributes", {}) or {})

            try:
                entity_updates: dict[str, Any] = {}
                for field in ("capacity", "load", "flow_rate", "max_flow_rate", "latency"):
                    if field in fields:
                        value = _finite_float(fields[field], minimum=0.0)
                        if value is not None:
                            setattr(entity, field, value)
                            entity_updates[field] = value

                if "coordinates" in fields:
                    coords = fields["coordinates"]
                    if isinstance(coords, (list, tuple)) and len(coords) == 2:
                        lat = _finite_float(coords[0], minimum=-90.0, maximum=90.0)
                        lon = _finite_float(coords[1], minimum=-180.0, maximum=180.0)
                        if lat is not None and lon is not None:
                            entity.coordinates = (lat, lon)
                            entity_updates["coordinates"] = entity.coordinates

                attrs = fields.get("attributes")
                if isinstance(attrs, dict):
                    safe_attrs: dict[str, Any] = {}
                    for key, value in attrs.items():
                        if len(safe_attrs) >= _MAX_SYNTH_ATTRS_PER_ENTITY:
                            break
                        if not isinstance(key, str):
                            continue
                        bounded = self._bounded_primitive(value)
                        if bounded is not None or value is None:
                            safe_attrs[key[:64]] = bounded
                    if safe_attrs:
                        entity.attributes.update(safe_attrs)
                        entity_updates["attributes"] = safe_attrs

                entity.enforce_constraints()
            except (AttributeError, TypeError, ValueError, KeyError) as exc:
                # Roll this entity back — partial mutations must not survive a
                # constraint or field error.
                for field, prior in before.items():
                    if prior is not None or hasattr(entity, field):
                        setattr(entity, field, prior)
                if hasattr(entity, "attributes"):
                    entity.attributes.clear()
                    entity.attributes.update(before_attrs)
                logger.warning(
                    "Synthesized update for entity %s rolled back after error: %s",
                    entity_id,
                    exc,
                )
                continue

            if entity_updates:
                applied[str(entity_id)] = entity_updates
        return applied


class RerouteVesselActuator(BaseActuator):
    """Actuator to adjust headings and speeds of maritime vessel edges."""

    # In-memory simulation only (mutates the physics world model, no external
    # effect) — explicitly opts out of the governed-by-default policy.
    requires_authority = False
    blocking_execution = False

    @property
    def name(self) -> str:
        return "reroute_vessel"

    @property
    def description(self) -> str:
        return "Adjusts heading (degrees) and speed (knots) of a target maritime vessel edge."

    def validate_params(self, params: dict[str, Any]) -> bool:
        vessel_id = params.get("vessel_id")
        heading = params.get("heading")
        speed = params.get("speed")

        if not vessel_id or heading is None or speed is None:
            return False

        heading_f = _finite_float(heading, minimum=0.0, maximum=360.0)
        speed_f = _finite_float(speed, minimum=0.0, maximum=40.0)
        if heading_f is None or speed_f is None:
            return False
        return True

    def execute(self, params: dict[str, Any]) -> ActuatorResult:
        if not self.validate_params(params):
            return ActuatorResult(False, "Parameter validation failed", {})

        try:
            from core.world.world_model import get_physics_world_model

            model = get_physics_world_model()
            vessel_id = str(params["vessel_id"])
            heading = _finite_float(params["heading"], minimum=0.0, maximum=360.0)
            speed = _finite_float(params["speed"], minimum=0.0, maximum=40.0)
            if heading is None or speed is None:
                return ActuatorResult(False, "Parameter validation failed", {})

            vessel = model.get_entity(vessel_id)
            if not vessel:
                return ActuatorResult(False, f"Vessel '{vessel_id}' not found", {})

            # Apply step update
            model.simulate(
                1.0,
                actions=[
                    {
                        "type": "reroute",
                        "entity_id": vessel_id,
                        "heading": heading,
                        "speed": speed,
                    }
                ],
            )

            logger.info(
                "Executed Actuator: reroute_vessel %s to heading=%s, speed=%s",
                vessel_id,
                heading,
                speed,
            )
            return ActuatorResult(
                success=True,
                message=f"Vessel '{vessel_id}' successfully rerouted.",
                updates={vessel_id: {"heading": heading, "speed": speed}},
            )

        except (ImportError, AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            return ActuatorResult(False, f"Actuator execution failed: {exc}", {})


class ReallocateFlowActuator(BaseActuator):
    """Actuator to transfer assets/cargo from one inventory node to another."""

    # In-memory simulation only — explicitly opts out of governed-by-default.
    requires_authority = False
    blocking_execution = False

    @property
    def name(self) -> str:
        return "reallocate_flow"

    @property
    def description(self) -> str:
        return (
            "Transfers inventory quantity (units) between two nodes to relieve bottleneck pressure."
        )

    def validate_params(self, params: dict[str, Any]) -> bool:
        source_id = params.get("source_id")
        target_id = params.get("target_id")
        amount = params.get("amount")

        if not source_id or not target_id or amount is None:
            return False

        amount_f = _finite_float(amount, minimum=1e-9)
        if amount_f is None:
            return False
        return True

    def execute(self, params: dict[str, Any]) -> ActuatorResult:
        if not self.validate_params(params):
            return ActuatorResult(False, "Parameter validation failed", {})

        try:
            from core.world.world_model import get_physics_world_model

            model = get_physics_world_model()
            source_id = str(params["source_id"])
            target_id = str(params["target_id"])
            amount = _finite_float(params["amount"], minimum=1e-9)
            if amount is None:
                return ActuatorResult(False, "Parameter validation failed", {})
            requested = amount

            source = model.get_entity(source_id)
            target = model.get_entity(target_id)

            if not source or not target:
                return ActuatorResult(False, "Source or target node not found", {})

            if source.load < amount:
                return ActuatorResult(
                    False,
                    f"Source '{source_id}' load {source.load} insufficient for transfer of {amount}",
                    {},
                )

            if target.load + amount > target.capacity:
                # Capacity constraint check
                transferable = target.capacity - target.load
                if transferable <= 0.0:
                    return ActuatorResult(
                        False, f"Target '{target_id}' at maximum capacity {target.capacity}", {}
                    )
                amount = transferable  # Clip transfer

            # Report what the world actually did, not what was asked of it.
            # The world clips a transfer against live capacity and constraints,
            # so the requested amount is a hope; the measured delta is the
            # fact. Claiming the requested figure made a no-op transfer read as
            # a success, which is how the same remedy can be re-issued forever
            # against a bottleneck it never relieved.
            source_load_before = float(source.load)
            target_load_before = float(target.load)

            model.simulate(
                1.0,
                actions=[
                    {
                        "type": "transfer",
                        "entity_id": source_id,
                        "target_id": target_id,
                        "amount": amount,
                    }
                ],
            )

            moved = source_load_before - float(source.load)
            received = float(target.load) - target_load_before
            if moved <= 0.0:
                return ActuatorResult(
                    False,
                    f"Transfer from '{source_id}' to '{target_id}' moved nothing "
                    f"(requested {amount}); the world did not accept it.",
                    {},
                )

            logger.info(
                "Executed Actuator: reallocate_flow transferred %s from %s to %s",
                moved,
                source_id,
                target_id,
            )
            message = (
                f"Flow of {moved} successfully reallocated from '{source_id}' "
                f"to '{target_id}'."
            )
            if moved + 1e-9 < requested:
                message += f" (clipped from the requested {requested})"
            return ActuatorResult(
                success=True,
                message=message,
                updates={
                    source_id: {"load": source.load},
                    target_id: {"load": target.load},
                    "_measured": {"moved": moved, "received": received, "requested": requested},
                },
            )

        except (ImportError, AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            return ActuatorResult(False, f"Actuator execution failed: {exc}", {})


class SandboxActuator(BaseActuator):
    """Actuator wrapper for the SandboxOperator, enabling dynamic code synthesis."""

    requires_authority = True

    def __init__(self) -> None:
        from core.actuators.sandbox_operator import SandboxOperator
        self.operator = SandboxOperator()

    @property
    def name(self) -> str:
        return "execute_in_sandbox"

    @property
    def description(self) -> str:
        return (
            "Executes raw Python code synthesized by Aura in a sandbox subprocess. "
            "Returns a dictionary with 'success', 'stdout', 'stderr', and 'exit_code'."
        )

    def validate_params(self, params: dict[str, Any]) -> bool:
        return isinstance(params, dict) and "code" in params and isinstance(params["code"], str)

    def execute(self, params: dict[str, Any]) -> ActuatorResult:
        if not params.get("_aura_authorized"):
            return ActuatorResult(False, "Sandbox execution requires ActuatorRegistry/AuthorityGateway authorization.", {})
        if not self.validate_params(params):
            return ActuatorResult(False, "Parameter validation failed: 'code' string parameter is required.", {})

        code = params["code"]
        # Bound the sandbox timeout: an unbounded or non-finite value could
        # hang the sandbox subprocess indefinitely.
        timeout_s = _finite_float(
            params.get("timeout_s", 10.0),
            minimum=_SANDBOX_TIMEOUT_MIN_S,
            maximum=_SANDBOX_TIMEOUT_MAX_S,
        )
        if timeout_s is None:
            timeout_s = 10.0

        res = self.operator.execute_synthesized_tool(code, timeout_s=timeout_s)
        
        msg = f"Execution completed with exit code {res['exit_code']}."
        if not res["success"]:
            msg = f"Execution failed (exit code {res['exit_code']}): {res['stderr']}"
            
        return ActuatorResult(
            success=res["success"],
            message=msg,
            updates={"sandbox_result": res}
        )


class ActuatorRegistry:
    """Registry of executable physical open-ended actuators."""

    def __init__(self) -> None:
        self.actuators: dict[str, BaseActuator] = {}
        # Guards registry mutations/lookups against concurrent registration,
        # deregistration, and execution-time lookups.
        self._lock = threading.RLock()
        # Names of required default actuators that failed to register at boot.
        self._missing_default_capabilities: list[str] = []
        self._register_default_actuators()

    def _register_default_actuators(self) -> None:
        self.register(RerouteVesselActuator())
        self.register(ReallocateFlowActuator())
        self.register(SandboxActuator())

        # Each required capability that fails to import is recorded as a
        # capability blocker so boot health can surface a degraded runtime,
        # instead of the failure only reaching the log.
        def _register_required(loader, capability: str) -> None:
            try:
                loader()
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                self._missing_default_capabilities.append(capability)
                logger.error("Failed to register required actuator '%s': %s", capability, exc)

        def _code() -> None:
            from core.actuators.code_execution_actuator import CodeExecutionActuator
            self.register(CodeExecutionActuator())

        def _web() -> None:
            from core.actuators.web_actuators import WebFetchActuator, WebSearchActuator
            self.register(WebSearchActuator())
            self.register(WebFetchActuator())

        def _git_pkg() -> None:
            from core.actuators.git_pkg_actuators import GitActuator, PackageInstallActuator
            self.register(GitActuator())
            self.register(PackageInstallActuator())

        def _process() -> None:
            from core.actuators.process_supervisor import ProcessSupervisorActuator
            self.register(ProcessSupervisorActuator())

        def _doc() -> None:
            from core.actuators.doc_ingest import DocumentIngestActuator
            self.register(DocumentIngestActuator())

        _register_required(_code, "code_execution")
        _register_required(_web, "web")
        _register_required(_git_pkg, "git_package")
        _register_required(_process, "process_supervisor")
        _register_required(_doc, "document_ingest")

    def missing_default_capabilities(self) -> list[str]:
        """Required default actuators that failed to register at boot."""
        with self._lock:
            return list(self._missing_default_capabilities)

    def health_blocker(self) -> str | None:
        """Return a capability-health blocker string, or None when complete."""
        missing = self.missing_default_capabilities()
        if not missing:
            return None
        return "actuator_capabilities_missing:" + ",".join(sorted(missing))

    def register(self, actuator: BaseActuator, *, allow_replace: bool = False) -> None:
        with self._lock:
            existing = self.actuators.get(actuator.name)
            if existing is not None and existing is not actuator and not allow_replace:
                # Refuse to silently hijack a canonical actuator name. A
                # deliberate swap must pass allow_replace=True.
                raise ValueError(
                    f"Actuator name '{actuator.name}' is already registered by "
                    f"{type(existing).__name__}; refusing silent replacement"
                )
            self.actuators[actuator.name] = actuator
        logger.info("Registered actuator: %s (%s)", actuator.name, actuator.description)

    def get_actuator(self, name: str) -> BaseActuator | None:
        with self._lock:
            return self.actuators.get(name)

    def register_synthesized(
        self, actuator: BaseActuator, source_code: str, trust_score: float = 0.3
    ) -> None:
        """Register a runtime-synthesized actuator with low trust and stored source code."""
        actuator.synthesized = True
        actuator.source_code = source_code
        # Non-finite trust must not slip past the execution-time thresholds
        # (NaN comparisons are always false), so clamp at registration.
        bounded_trust = _finite_float(trust_score, minimum=0.0, maximum=1.0)
        actuator.trust_score = bounded_trust if bounded_trust is not None else 0.0
        # Synthesized actuators may replace a prior generation of themselves.
        self.register(actuator, allow_replace=True)
        logger.info(
            "Registered synthesized actuator: %s (trust=%.2f)", actuator.name, actuator.trust_score
        )

    def deregister(self, name: str) -> None:
        """Remove an actuator from the registry (retirement)."""
        with self._lock:
            removed = self.actuators.pop(name, None)
        if removed is not None:
            logger.info("Deregistered actuator: %s", name)

    def get_synthesized_actuators(self) -> list[BaseActuator]:
        """List all runtime-synthesized actuators."""
        with self._lock:
            return [act for act in self.actuators.values() if getattr(act, "synthesized", False)]

    @staticmethod
    async def _authorize_actuator(
        name: str,
        params: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:
        from core.executive.authority_gateway import get_authority_gateway

        gateway = get_authority_gateway()
        # Bound caller-supplied priority to [0, 1] and reject non-finite
        # values — an unvalidated priority could otherwise skew the gateway's
        # admission ordering. is_critical stays a plain bool cast; the deeper
        # "authenticated principal" binding is a gateway-side contract change.
        priority = _finite_float(context.get("priority", 0.7), minimum=0.0, maximum=1.0)
        if priority is None:
            priority = 0.7
        return await gateway.authorize_tool_execution(
            name,
            params,
            source=str(context.get("source") or "actuator_registry"),
            priority=priority,
            is_critical=bool(context.get("is_critical", False)),
            context=dict(context or {}),
        )

    @staticmethod
    def _execute_actuator_body(
        actuator: BaseActuator,
        params: dict[str, Any],
        authority_decision: Any,
    ) -> ActuatorResult:
        if authority_decision is None:
            return actuator.execute(params)

        from core.governance_context import governed_scope_sync

        with governed_scope_sync(authority_decision):
            return actuator.execute(params)

    @staticmethod
    def _preflight_actuator(
        actuator: BaseActuator,
        name: str,
        params: dict[str, Any],
    ) -> ActuatorResult | None:
        if not getattr(actuator, "synthesized", False):
            return None
        # Non-finite trust must fail CLOSED. NaN < 0.2 is False, so a NaN
        # trust score would otherwise bypass both thresholds and execute.
        trust = _finite_float(getattr(actuator, "trust_score", None))
        if trust is None:
            return ActuatorResult(
                False,
                f"Actuator '{name}' has a non-finite trust score; refusing execution",
                {},
            )
        if trust < 0.2:
            return ActuatorResult(
                False,
                f"Actuator '{name}' has trust score too low ({trust:.2f}) to execute",
                {},
            )
        if trust < 0.5:
            logger.warning(
                "Executing low-trust synthesized actuator '%s' (trust=%.2f)",
                name,
                trust,
            )
            if not params:
                return ActuatorResult(
                    False,
                    "Low-trust actuator requires non-empty parameters",
                    {},
                )
        return None

    async def execute_action_async(
        self,
        name: str,
        params: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> ActuatorResult:
        """Execute one actuator without blocking the runtime owner loop."""
        actuator = self.get_actuator(name)
        if not actuator:
            return ActuatorResult(False, f"Actuator '{name}' not found", {})

        exec_params = dict(params or {})
        authority_decision = None
        capability_token_id = None
        result: ActuatorResult | None = None
        cancellation: asyncio.CancelledError | None = None

        # Reject deterministic local preflight failures before acquiring an
        # authority lease. Once a lease is acquired, every path below closes it.
        preflight = self._preflight_actuator(actuator, name, exec_params)
        if preflight is not None:
            return preflight
        if runtime_shutdown_blocks_new_work(
            f"actuator:{name}",
            resource_kind="actuator_execution",
        ):
            return ActuatorResult(
                False,
                f"Actuator '{name}' refused during runtime shutdown",
                {},
            )

        if getattr(actuator, "requires_authority", False):
            try:
                authority_decision = await self._authorize_actuator(
                    name,
                    exec_params,
                    dict(context or {}),
                )
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                return ActuatorResult(
                    False,
                    f"AuthorityGateway unavailable for actuator '{name}': {type(exc).__name__}: {exc}",
                    {},
                )
            if not getattr(authority_decision, "approved", False):
                return ActuatorResult(
                    False,
                    f"Actuator '{name}' refused by AuthorityGateway: {getattr(authority_decision, 'reason', '')}",
                    {},
                )
            capability_token_id = getattr(authority_decision, "capability_token_id", None)

        try:
            if authority_decision is not None:
                try:
                    from core.executive.authority_gateway import get_authority_gateway

                    if not get_authority_gateway().verify_tool_access(name, capability_token_id):
                        return ActuatorResult(
                            False, f"Capability token rejected for actuator '{name}'", {}
                        )
                except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    return ActuatorResult(
                        False,
                        f"Capability token verification failed for actuator '{name}': {type(exc).__name__}: {exc}",
                        {},
                    )
                exec_params["_aura_authorized"] = True
                exec_params["_capability_token_id"] = capability_token_id

            if runtime_shutdown_blocks_new_work(
                f"actuator:{name}",
                resource_kind="actuator_execution",
            ):
                result = ActuatorResult(
                    False,
                    f"Actuator '{name}' refused because runtime shutdown began during authorization",
                    {},
                )
                return result

            if getattr(actuator, "blocking_execution", True):
                drained = await drain_owned_awaitable(
                    asyncio.to_thread(
                        self._execute_actuator_body,
                        actuator,
                        exec_params,
                        authority_decision,
                    ),
                    name=f"actuator:{name}",
                    owner="core.actuators.actuator_registry",
                    allow_during_shutdown=True,
                )
                cancellation = drained.cancellation
                result = drained.task.result()
                if cancellation is not None:
                    # Python threads cannot be cancelled safely. Observe the
                    # bounded actuator to completion so authority closure and
                    # outcome receipts describe what really happened, then
                    # propagate cancellation to the caller.
                    logger.info(
                        "Actuator '%s' completed after caller cancellation; "
                        "closing authority before propagating cancellation",
                        name,
                    )
            else:
                result = self._execute_actuator_body(
                    actuator,
                    exec_params,
                    authority_decision,
                )
            if not isinstance(result, ActuatorResult):
                result = ActuatorResult(
                    False,
                    f"Actuator '{name}' returned invalid result type {type(result).__name__}",
                    {},
                )
        except asyncio.CancelledError:
            raise
        except (
            ImportError,
            OSError,
            RuntimeError,
            AttributeError,
            LookupError,
            TypeError,
            ValueError,
        ) as exc:
            # Full detail goes to the log; the returned message carries only the
            # error CLASS. Raw exception text can leak internal paths, provider
            # details, source snippets, or secrets to the caller surface.
            logger.exception("Actuator '%s' raised during execution", name)
            result = ActuatorResult(
                False,
                f"Actuator '{name}' failed ({type(exc).__name__}); see logs for detail",
                {},
            )
        finally:
            if authority_decision is not None:
                success = bool(result and result.success)
                update_keys = sorted(result.updates.keys()) if result and isinstance(result.updates, dict) else []
                receipt = {
                    "success": success,
                    "actuator": name,
                    "actuator_generation": int(getattr(actuator, "generation", 0) or 0),
                    "synthesized": bool(getattr(actuator, "synthesized", False)),
                    "update_key_count": len(update_keys),
                    "update_keys": update_keys[:32],
                    "message": (result.message[:240] if result else ""),
                    "cancelled": cancellation is not None,
                }
                try:
                    from core.executive.authority_gateway import get_authority_gateway

                    get_authority_gateway().finalize_tool_execution(
                        executive_intent_id=getattr(
                            authority_decision, "executive_intent_id", None
                        ),
                        capability_token_id=capability_token_id,
                        standing_authority_token=getattr(
                            authority_decision, "standing_authority_token", None
                        ),
                        success=success,
                        result=receipt,
                    )
                except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    # A successful external effect whose authority closure could
                    # not be recorded is NOT a clean success — surface the
                    # uncertain audit state instead of returning bare success.
                    logger.error("Failed to finalize actuator authority receipt for %s: %s", name, exc)
                    if result is not None and result.success:
                        result = ActuatorResult(
                            False,
                            f"Actuator '{name}' effect applied but authority closure failed; audit incomplete",
                            result.updates,
                        )

        if cancellation is not None:
            raise cancellation
        return result or ActuatorResult(False, f"Actuator '{name}' produced no result", {})

    def execute_action(
        self,
        name: str,
        params: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> ActuatorResult:
        """Synchronous bridge for callers that do not own an event loop.

        Async runtime code must await :meth:`execute_action_async`; blocking an
        active owner loop here would invalidate both latency and thread-bound
        standing-authority guarantees.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.execute_action_async(name, params, context=context)
            )
        raise RuntimeError(
            "execute_action cannot run on an active event loop; "
            "await execute_action_async instead"
        )


# Singleton Pattern
_instance: ActuatorRegistry | None = None
_instance_lock = threading.Lock()


def get_actuator_registry() -> ActuatorRegistry:
    global _instance
    # Double-checked locking: concurrent first access must not construct two
    # registries with divergent synthesized actuators / execution histories.
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = ActuatorRegistry()
    return _instance
