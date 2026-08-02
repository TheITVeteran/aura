"""Durable governed transaction coordinator for physical Reality Reach effects."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import stat
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from core.config import DATA_DIR
from core.governance.will import ActionDomain
from core.reality_reach.actuation import (
    ActuationCommand,
    ActuationLease,
    ActuationReceipt,
    ActuationState,
    ActuatorCapability,
    EffectReceipt,
    PreparedActuation,
    RealityAdapter,
    RollbackReceipt,
)
from core.reality_reach.live import RealityReachService
from core.runtime.action_executor import ActionExecutor
from core.runtime.atomic_writer import (
    atomic_write_bytes,
    ensure_private_directory,
    interprocess_file_lock,
)
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.skill_contract import ActionExpectation

TRANSACTION_SCHEMA = "aura.reality-reach-actuation-transaction.v1"
_TERMINAL = frozenset(
    {
        ActuationState.EFFECT_VERIFIED,
        ActuationState.CANCELLED,
        ActuationState.SAFE_STATE,
        ActuationState.COMPENSATED,
        ActuationState.ROLLED_BACK,
        ActuationState.TIMED_OUT,
        ActuationState.INDETERMINATE,
        ActuationState.MANUALLY_RECONCILED,
        ActuationState.FAILED,
    }
)
_NO_AUTOMATIC_REPLAY = frozenset(
    {
        ActuationState.DISPATCHED,
        ActuationState.EXECUTED,
        ActuationState.INDETERMINATE,
    }
)
Executor = Callable[..., Awaitable[dict[str, Any]]]
_MAX_TRANSACTION_BYTES = 1024 * 1024
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_DIGEST = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


class RealityActuationError(RuntimeError):
    """Stable fail-closed Reality Reach transaction error."""


def _sha256(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


def _record_path(root: Path, idempotency_key: str) -> Path:
    digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
    return root / f"{digest}.json"


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _private_root(path: Path) -> Path:
    requested = path.expanduser().absolute()
    if requested.is_symlink():
        raise RealityActuationError("reality_actuation_root_symlink_refused")
    root = Path(ensure_private_directory(requested))
    metadata = root.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise RealityActuationError("reality_actuation_root_custody_invalid")
    return root


def _read_record_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RealityActuationError("reality_actuation_transaction_open_failed") from exc
    try:
        metadata = os.fstat(fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or metadata.st_size > _MAX_TRANSACTION_BYTES
        ):
            raise RealityActuationError("reality_actuation_transaction_custody_invalid")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(fd, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
        if remaining or any(
            getattr(after, field) != getattr(metadata, field)
            for field in (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_uid",
                "st_nlink",
                "st_size",
                "st_mtime_ns",
            )
        ):
            raise RealityActuationError("reality_actuation_transaction_changed_during_read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _validate_record(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema",
        "transaction_sha256",
        "command_id",
        "command_sha256",
        "idempotency_key",
        "adapter_id",
        "channel_id",
        "state",
        "revision",
        "lease_sha256",
        "preparation_sha256",
        "actuation_receipt_sha256",
        "effect_receipt_sha256",
        "rollback_receipt_sha256",
        "authority_receipt_id",
        "created_at_ns",
        "updated_at_ns",
        "manual_reconciliation_required",
        "last_error",
    }
    if set(value) != required or value.get("schema") != TRANSACTION_SCHEMA:
        raise RealityActuationError("reality_actuation_transaction_schema_invalid")
    body = dict(value)
    digest = body.pop("transaction_sha256", None)
    if digest != _sha256(body):
        raise RealityActuationError("reality_actuation_transaction_digest_invalid")
    try:
        state = ActuationState(str(value["state"]))
    except ValueError as exc:
        raise RealityActuationError("reality_actuation_transaction_state_invalid") from exc
    if (
        not isinstance(value.get("revision"), int)
        or isinstance(value.get("revision"), bool)
        or int(value["revision"]) < 0
        or not isinstance(value.get("manual_reconciliation_required"), bool)
        or any(
            not isinstance(value.get(name), str)
            or not _IDENTIFIER.fullmatch(str(value[name]))
            for name in (
                "command_id",
                "idempotency_key",
                "adapter_id",
                "channel_id",
            )
        )
        or any(
            not isinstance(value.get(name), str)
            or (bool(value[name]) and not _DIGEST.fullmatch(str(value[name])))
            for name in (
                "lease_sha256",
                "preparation_sha256",
                "actuation_receipt_sha256",
                "effect_receipt_sha256",
                "rollback_receipt_sha256",
            )
        )
        or not isinstance(value.get("command_sha256"), str)
        or not _DIGEST.fullmatch(str(value["command_sha256"]))
        or not isinstance(value.get("authority_receipt_id"), str)
        or (
            bool(value["authority_receipt_id"])
            and not _IDENTIFIER.fullmatch(str(value["authority_receipt_id"]))
        )
        or any(
            not isinstance(value.get(name), int)
            or isinstance(value.get(name), bool)
            or int(value[name]) <= 0
            for name in ("created_at_ns", "updated_at_ns")
        )
        or not isinstance(value.get("last_error"), str)
        or len(str(value["last_error"])) > 500
    ):
        raise RealityActuationError("reality_actuation_transaction_fields_invalid")
    required_by_state: dict[ActuationState, tuple[str, ...]] = {
        ActuationState.ADMITTED: (
            "lease_sha256",
            "preparation_sha256",
            "authority_receipt_id",
        ),
        ActuationState.DISPATCHED: (
            "lease_sha256",
            "preparation_sha256",
            "authority_receipt_id",
        ),
        ActuationState.EXECUTED: ("actuation_receipt_sha256",),
        ActuationState.EFFECT_VERIFIED: (
            "actuation_receipt_sha256",
            "effect_receipt_sha256",
        ),
        ActuationState.MANUALLY_RECONCILED: (
            "actuation_receipt_sha256",
            "effect_receipt_sha256",
            "authority_receipt_id",
        ),
        ActuationState.COMPENSATED: ("rollback_receipt_sha256",),
        ActuationState.ROLLED_BACK: ("rollback_receipt_sha256",),
        ActuationState.SAFE_STATE: ("rollback_receipt_sha256",),
    }
    if any(not value.get(name) for name in required_by_state.get(state, ())):
        raise RealityActuationError("reality_actuation_transaction_lineage_invalid")
    result = dict(value)
    result["state"] = state.value
    return result


class RealityActuationCoordinator:
    """Runs one idempotent physical effect through ActionExecutor and readback."""

    def __init__(
        self,
        service: RealityReachService,
        *,
        root: Path | None = None,
        executor: Executor = ActionExecutor.execute,
        wall_clock_ns: Callable[[], int] = time.time_ns,
        monotonic_clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._service = service
        self._root = _private_root(
            root or (Path(DATA_DIR) / "reality_reach" / "transactions")
        )
        self._lock_path = self._root / ".transactions.lock"
        self._executor = executor
        self._wall_clock_ns = wall_clock_ns
        self._monotonic_clock_ns = monotonic_clock_ns

    def _load(self, command: ActuationCommand) -> dict[str, Any] | None:
        path = _record_path(self._root, command.idempotency_key)
        with interprocess_file_lock(self._lock_path):
            if path.is_symlink():
                raise RealityActuationError("reality_actuation_transaction_symlink_refused")
            if not path.exists():
                return None
            if not path.is_file():
                raise RealityActuationError("reality_actuation_transaction_type_invalid")
            try:
                raw = _read_record_bytes(path)
                value = json.loads(raw.decode("utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise RealityActuationError(
                    "reality_actuation_transaction_unreadable"
                ) from exc
            if not isinstance(value, Mapping):
                raise RealityActuationError("reality_actuation_transaction_not_mapping")
            record = _validate_record(value)
            if (
                record["command_id"] != command.command_id
                or record["command_sha256"] != command.sha256
                or record["idempotency_key"] != command.idempotency_key
            ):
                raise RealityActuationError("reality_actuation_idempotency_collision")
            return record

    def is_alive(self) -> bool:
        try:
            return self._root.is_dir() and not self._root.is_symlink()
        except OSError:
            return False

    def is_ready(self) -> bool:
        return bool(self._service.executable_actuator_channels())

    def status(self) -> dict[str, Any]:
        return {
            "alive": self.is_alive(),
            "ready": self.is_ready(),
            "executable_actuator_channels": list(
                self._service.executable_actuator_channels()
            ),
            "transaction_schema": TRANSACTION_SCHEMA,
        }

    def _create(self, command: ActuationCommand) -> dict[str, Any]:
        path = _record_path(self._root, command.idempotency_key)
        with interprocess_file_lock(self._lock_path):
            if path.is_symlink():
                raise RealityActuationError("reality_actuation_transaction_symlink_refused")
            if path.exists():
                existing = self._load(command)
                if existing is None:
                    raise RealityActuationError("reality_actuation_create_race")
                return existing
            now_ns = int(self._wall_clock_ns())
            body: dict[str, Any] = {
                "schema": TRANSACTION_SCHEMA,
                "command_id": command.command_id,
                "command_sha256": command.sha256,
                "idempotency_key": command.idempotency_key,
                "adapter_id": command.adapter_id,
                "channel_id": command.channel_id,
                "state": ActuationState.PLANNED.value,
                "revision": 0,
                "lease_sha256": "",
                "preparation_sha256": "",
                "actuation_receipt_sha256": "",
                "effect_receipt_sha256": "",
                "rollback_receipt_sha256": "",
                "authority_receipt_id": "",
                "created_at_ns": now_ns,
                "updated_at_ns": now_ns,
                "manual_reconciliation_required": False,
                "last_error": "",
            }
            record = {**body, "transaction_sha256": _sha256(body)}
            atomic_write_bytes(path, _canonical_bytes(record), mode=0o600)
            return _validate_record(record)

    def _transition(
        self,
        command: ActuationCommand,
        *,
        expected: set[ActuationState],
        state: ActuationState,
        updates: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = _record_path(self._root, command.idempotency_key)
        with interprocess_file_lock(self._lock_path):
            record = self._load(command)
            if record is None:
                raise RealityActuationError("reality_actuation_transaction_missing")
            current = ActuationState(record["state"])
            if current not in expected:
                if current == state:
                    return record
                raise RealityActuationError(
                    f"reality_actuation_transition_invalid:{current.value}->{state.value}"
                )
            permitted_updates = {
                "lease_sha256",
                "preparation_sha256",
                "actuation_receipt_sha256",
                "effect_receipt_sha256",
                "rollback_receipt_sha256",
                "authority_receipt_id",
                "manual_reconciliation_required",
                "last_error",
            }
            update_fields = dict(updates or {})
            if not set(update_fields).issubset(permitted_updates):
                raise RealityActuationError("reality_actuation_transition_fields_invalid")
            body = {
                key: value
                for key, value in record.items()
                if key != "transaction_sha256"
            }
            body.update(update_fields)
            body["state"] = state.value
            body["revision"] = int(record["revision"]) + 1
            body["updated_at_ns"] = int(self._wall_clock_ns())
            updated = {**body, "transaction_sha256": _sha256(body)}
            atomic_write_bytes(path, _canonical_bytes(updated), mode=0o600)
            return _validate_record(updated)

    async def execute(self, command: ActuationCommand) -> dict[str, Any]:
        if not isinstance(command, ActuationCommand):
            raise TypeError("command must be an ActuationCommand")
        inventory_sha256 = str(self._service.status()["registry_sha256"])
        if inventory_sha256 != command.inventory_sha256:
            raise RealityActuationError("reality_actuation_inventory_drift")
        adapter = self._service.actuator_adapter(command.channel_id)
        capability = self._service.actuator_capability(command.channel_id)
        if adapter is None or capability is None:
            raise RealityActuationError("reality_actuation_channel_not_executable")
        if (
            adapter.adapter_id != command.adapter_id
            or capability.adapter_id != command.adapter_id
            or not capability.magnitude_domain.contains(command.magnitude)
        ):
            raise RealityActuationError("reality_actuation_capability_mismatch")
        now_ns = int(self._wall_clock_ns())
        if now_ns >= command.deadline_ns:
            raise RealityActuationError("reality_actuation_command_expired")

        existing = await asyncio.to_thread(self._create, command)
        existing_state = ActuationState(existing["state"])
        if existing_state in _TERMINAL or existing_state in _NO_AUTOMATIC_REPLAY:
            return self._replay(existing)

        execution: dict[str, Any] = {}

        async def effect_handler(context: Mapping[str, Any]) -> Mapping[str, Any]:
            return await self._dispatch_adapter(
                command,
                adapter=adapter,
                capability=capability,
                context=context,
                execution=execution,
            )

        async def effect_verifier(_context: Mapping[str, Any]) -> Mapping[str, Any]:
            return await self._verify_adapter_effect(
                command,
                adapter=adapter,
                capability=capability,
                execution=execution,
            )

        remaining_s = max(0.1, (command.deadline_ns - now_ns) / 1_000_000_000)
        result = await self._executor(
            domain=ActionDomain.ENVIRONMENT_ACTION,
            action_name=f"reality_reach.{command.channel_id}",
            params={
                "command_id": command.command_id,
                "command_sha256": command.sha256,
                "channel_id": command.channel_id,
                "idempotency_key": command.idempotency_key,
            },
            source="reality_reach",
            rollback_target=(
                capability.compensation_action
                or ("adapter.rollback" if capability.supports_rollback else None)
            ),
            expectation=ActionExpectation(
                objective=f"produce and independently observe {command.observable}",
                acceptance_criteria=["effect_verified", *command.expected_effects],
                required_evidence=["reality_reach.effect_receipt"],
                rollback_hint=(
                    capability.compensation_action
                    or "use adapter safe_state and rollback receipts"
                ),
                allow_partial=False,
            ),
            effect_handler=effect_handler,
            effect_verifier=effect_verifier,
            execution_timeout_s=min(remaining_s, capability.watchdog_timeout_s),
            verification_timeout_s=min(remaining_s, capability.watchdog_timeout_s),
            action_id=command.command_id,
        )
        record = await asyncio.to_thread(self._load, command)
        if record is None:
            raise RealityActuationError("reality_actuation_transaction_lost")
        return {**dict(result), "reality_reach_transaction": record}

    async def reconcile(
        self,
        command: ActuationCommand,
        effect: EffectReceipt,
        *,
        authority_receipt_id: str,
    ) -> dict[str, Any]:
        """Close an executed-but-unverified crash using independent readback."""

        if not isinstance(effect, EffectReceipt) or not effect.independently_observed:
            raise RealityActuationError("reality_actuation_reconciliation_evidence_invalid")
        if not isinstance(authority_receipt_id, str) or not _IDENTIFIER.fullmatch(
            authority_receipt_id
        ):
            raise RealityActuationError("reality_actuation_reconciliation_authority_invalid")
        capability = self._service.actuator_capability(command.channel_id)
        if capability is None:
            raise RealityActuationError("reality_actuation_channel_not_executable")
        record = await asyncio.to_thread(self._load, command)
        if record is None:
            raise RealityActuationError("reality_actuation_transaction_missing")
        state = ActuationState(record["state"])
        if state not in {ActuationState.EXECUTED, ActuationState.INDETERMINATE}:
            raise RealityActuationError("reality_actuation_reconciliation_state_invalid")
        if (
            effect.command_sha256 != command.sha256
            or effect.state != ActuationState.EFFECT_VERIFIED
            or effect.observation_channel_id not in capability.observation_channels
            or not record["actuation_receipt_sha256"]
            or effect.actuation_receipt_sha256 != record["actuation_receipt_sha256"]
        ):
            raise RealityActuationError("reality_actuation_reconciliation_identity_invalid")
        reconciled = await asyncio.to_thread(
            self._transition,
            command,
            expected={state},
            state=ActuationState.MANUALLY_RECONCILED,
            updates={
                "effect_receipt_sha256": effect.sha256,
                "authority_receipt_id": authority_receipt_id,
                "manual_reconciliation_required": False,
                "last_error": "",
            },
        )
        return self._replay(reconciled)

    async def _dispatch_adapter(
        self,
        command: ActuationCommand,
        *,
        adapter: RealityAdapter,
        capability: ActuatorCapability,
        context: Mapping[str, Any],
        execution: dict[str, Any],
    ) -> Mapping[str, Any]:
        wall_now = int(self._wall_clock_ns())
        monotonic_now = int(self._monotonic_clock_ns())
        duration_ns = max(
            1,
            min(
                command.deadline_ns - wall_now,
                int(capability.watchdog_timeout_s * 1_000_000_000),
            ),
        )
        lease = ActuationLease(
            lease_id=f"lease.{command.sha256.removeprefix('sha256:')[:32]}",
            command_sha256=command.sha256,
            adapter_id=command.adapter_id,
            session_id=str(self._service.status()["session_id"]),
            authority_receipt_id=str(context.get("will_receipt_id") or ""),
            issued_at_ns=wall_now,
            expires_at_ns=wall_now + duration_ns,
            issued_monotonic_ns=monotonic_now,
            expires_monotonic_ns=monotonic_now + duration_ns,
        )
        timeout_s = capability.watchdog_timeout_s
        try:
            prepared = await asyncio.wait_for(
                adapter.prepare(command, lease),
                timeout=timeout_s,
            )
            self._validate_prepared(command, lease, capability, prepared)
            execution.update({"lease": lease, "prepared": prepared})
            await asyncio.to_thread(
                self._transition,
                command,
                expected={ActuationState.PLANNED, ActuationState.PREPARED},
                state=ActuationState.ADMITTED,
                updates={
                    "lease_sha256": lease.sha256,
                    "preparation_sha256": prepared.sha256,
                    "authority_receipt_id": lease.authority_receipt_id,
                },
            )
            await asyncio.to_thread(
                self._transition,
                command,
                expected={ActuationState.ADMITTED},
                state=ActuationState.DISPATCHED,
            )
            actuation = await asyncio.wait_for(
                adapter.actuate(command, lease, prepared),
                timeout=timeout_s,
            )
            self._validate_actuation(command, prepared, actuation)
            state = ActuationState.EXECUTED if actuation.executed else actuation.state
            await asyncio.to_thread(
                self._transition,
                command,
                expected={ActuationState.DISPATCHED},
                state=state,
                updates={"actuation_receipt_sha256": actuation.sha256},
            )
            execution["actuation"] = actuation
            return {
                "ok": actuation.executed,
                "transport_succeeded": actuation.transport_completed,
                "executed": actuation.executed,
                "actuation_receipt": actuation.to_dict(),
                "actuation_receipt_sha256": actuation.sha256,
            }
        except asyncio.CancelledError:
            await self._recover(
                command,
                adapter,
                execution,
                cancelled=True,
                timeout_s=timeout_s,
            )
            raise
        except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            await self._recover(
                command,
                adapter,
                execution,
                error=exc,
                timeout_s=timeout_s,
            )
            raise

    async def _verify_adapter_effect(
        self,
        command: ActuationCommand,
        *,
        adapter: RealityAdapter,
        capability: ActuatorCapability,
        execution: dict[str, Any],
    ) -> Mapping[str, Any]:
        actuation = execution.get("actuation")
        if not isinstance(actuation, ActuationReceipt) or not actuation.executed:
            return {"effect_verified": False, "reason": "actuation_not_executed"}
        try:
            effect = await asyncio.wait_for(
                adapter.verify_effect(command, actuation),
                timeout=capability.watchdog_timeout_s,
            )
            self._validate_effect(command, actuation, capability, effect)
            execution["effect"] = effect
            if effect.state == ActuationState.EFFECT_VERIFIED:
                await asyncio.to_thread(
                    self._transition,
                    command,
                    expected={ActuationState.EXECUTED},
                    state=ActuationState.EFFECT_VERIFIED,
                    updates={"effect_receipt_sha256": effect.sha256},
                )
                return {
                    "effect_verified": True,
                    "reality_reach.effect_receipt": effect.to_dict(),
                    "effect_receipt_sha256": effect.sha256,
                }
            await self._recover(
                command,
                adapter,
                execution,
                error=RealityActuationError("effect_not_verified"),
                timeout_s=capability.watchdog_timeout_s,
            )
            return {
                "effect_verified": False,
                "reason": "independent_effect_not_verified",
                "effect_receipt_sha256": effect.sha256,
            }
        except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            await self._recover(
                command,
                adapter,
                execution,
                error=exc,
                timeout_s=capability.watchdog_timeout_s,
            )
            return {
                "effect_verified": False,
                "reason": f"effect_verification_failed:{type(exc).__name__}",
            }

    async def _recover(
        self,
        command: ActuationCommand,
        adapter: RealityAdapter,
        execution: dict[str, Any],
        *,
        cancelled: bool = False,
        error: BaseException | None = None,
        timeout_s: float = 10.0,
    ) -> None:
        recovery_reason = str(
            error or ("cancelled" if cancelled else "recovery_required")
        )[:500]
        record = await asyncio.to_thread(self._load, command)
        if record is None:
            return
        state = ActuationState(record["state"])
        prepared = execution.get("prepared")
        actuation = execution.get("actuation")
        try:
            if state in {ActuationState.PLANNED, ActuationState.ADMITTED}:
                cancellation_receipt = await asyncio.wait_for(
                    adapter.cancel(
                        command,
                        prepared if isinstance(prepared, PreparedActuation) else None,
                    ),
                    timeout=timeout_s,
                )
                self._validate_actuation(
                    command,
                    prepared if isinstance(prepared, PreparedActuation) else None,
                    cancellation_receipt,
                    allow_without_preparation=True,
                )
                if cancellation_receipt.state != ActuationState.CANCELLED:
                    raise RealityActuationError("cancel_receipt_state_invalid")
                await asyncio.to_thread(
                    self._transition,
                    command,
                    expected={state},
                    state=ActuationState.CANCELLED,
                    updates={
                        "actuation_receipt_sha256": cancellation_receipt.sha256,
                        "last_error": recovery_reason,
                    },
                )
                return
            if state in {ActuationState.DISPATCHED, ActuationState.EXECUTED}:
                if isinstance(actuation, ActuationReceipt):
                    recovery_receipt = await asyncio.wait_for(
                        adapter.rollback(command, actuation),
                        timeout=timeout_s,
                    )
                else:
                    recovery_receipt = await asyncio.wait_for(
                        adapter.safe_state(command, None),
                        timeout=timeout_s,
                    )
                self._validate_rollback(command, actuation, recovery_receipt)
                await asyncio.to_thread(
                    self._transition,
                    command,
                    expected={state},
                    state=recovery_receipt.state,
                    updates={
                        "rollback_receipt_sha256": recovery_receipt.sha256,
                        "manual_reconciliation_required": not recovery_receipt.independently_observed,
                        "last_error": recovery_reason,
                    },
                )
        except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as recovery_exc:
            latest = await asyncio.to_thread(self._load, command)
            if latest is None:
                return
            latest_state = ActuationState(latest["state"])
            if latest_state not in _TERMINAL:
                await asyncio.to_thread(
                    self._transition,
                    command,
                    expected={latest_state},
                    state=ActuationState.INDETERMINATE,
                    updates={
                        "manual_reconciliation_required": True,
                        "last_error": (
                            f"{type(recovery_exc).__name__}:{recovery_exc}"
                        )[:500],
                    },
                )

    @staticmethod
    def _validate_prepared(
        command: ActuationCommand,
        lease: ActuationLease,
        capability: ActuatorCapability,
        prepared: PreparedActuation,
    ) -> None:
        if not isinstance(prepared, PreparedActuation) or (
            prepared.command_sha256 != command.sha256
            or prepared.lease_sha256 != lease.sha256
            or prepared.adapter_id != command.adapter_id
            or prepared.capability_sha256 != capability.sha256
        ):
            raise RealityActuationError("reality_actuation_preparation_identity_invalid")

    @staticmethod
    def _validate_actuation(
        command: ActuationCommand,
        prepared: PreparedActuation | None,
        receipt: ActuationReceipt,
        *,
        allow_without_preparation: bool = False,
    ) -> None:
        if not isinstance(receipt, ActuationReceipt):
            raise RealityActuationError("reality_actuation_receipt_type_invalid")
        expected_preparation = prepared.sha256 if prepared is not None else ""
        if (
            receipt.command_sha256 != command.sha256
            or receipt.adapter_id != command.adapter_id
            or (
                not allow_without_preparation
                and receipt.preparation_sha256 != expected_preparation
            )
        ):
            raise RealityActuationError("reality_actuation_receipt_identity_invalid")

    @staticmethod
    def _validate_effect(
        command: ActuationCommand,
        actuation: ActuationReceipt,
        capability: ActuatorCapability,
        effect: EffectReceipt,
    ) -> None:
        if not isinstance(effect, EffectReceipt) or (
            effect.command_sha256 != command.sha256
            or effect.actuation_receipt_sha256 != actuation.sha256
            or effect.observation_channel_id not in capability.observation_channels
        ):
            raise RealityActuationError("reality_actuation_effect_identity_invalid")

    @staticmethod
    def _validate_rollback(
        command: ActuationCommand,
        actuation: Any,
        receipt: RollbackReceipt,
    ) -> None:
        expected = actuation.sha256 if isinstance(actuation, ActuationReceipt) else ""
        if not isinstance(receipt, RollbackReceipt) or (
            receipt.command_sha256 != command.sha256
            or receipt.adapter_id != command.adapter_id
            or (expected and receipt.actuation_receipt_sha256 != expected)
        ):
            raise RealityActuationError("reality_actuation_rollback_identity_invalid")

    @staticmethod
    def _replay(record: Mapping[str, Any]) -> dict[str, Any]:
        state = ActuationState(str(record["state"]))
        effect_verified = state in {
            ActuationState.EFFECT_VERIFIED,
            ActuationState.MANUALLY_RECONCILED,
        }
        return {
            "ok": effect_verified,
            "effect_verified": effect_verified,
            "transport_succeeded": state
            in {
                ActuationState.EXECUTED,
                ActuationState.EFFECT_VERIFIED,
                ActuationState.MANUALLY_RECONCILED,
            },
            "retry_safe": False,
            "manual_reconciliation_required": bool(
                record["manual_reconciliation_required"]
            )
            or state in _NO_AUTOMATIC_REPLAY,
            "reality_reach_transaction": dict(record),
            "replayed": True,
        }


_COORDINATOR: RealityActuationCoordinator | None = None
_COORDINATOR_LOCK = threading.Lock()


def get_reality_actuation_coordinator(
    service: RealityReachService | None = None,
) -> RealityActuationCoordinator:
    global _COORDINATOR
    if _COORDINATOR is None:
        with _COORDINATOR_LOCK:
            if _COORDINATOR is None:
                if service is None:
                    from core.reality_reach.live import get_reality_reach_service

                    service = get_reality_reach_service()
                _COORDINATOR = RealityActuationCoordinator(service)
    return _COORDINATOR


__all__ = [
    "RealityActuationCoordinator",
    "RealityActuationError",
    "TRANSACTION_SCHEMA",
    "get_reality_actuation_coordinator",
]
