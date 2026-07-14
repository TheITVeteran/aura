"""infrastructure/rollback.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Atomic rollback controller for reliability-grade deployment safety.

Checkpoints current state before any deployment and provides atomic
swap between current and previous versions with verification.

Integrates with core/resilience/stem_cell.py for deep state snapshots.

Usage:
    from infrastructure.rollback import RollbackController

    controller = RollbackController()
    controller.register_state_applier(apply_runtime_state)  # restores on rollback
    controller.checkpoint("pre-deploy-v2.1", state_collector=collect_runtime_state)
    try:
        deploy_new_version()
        controller.verify()
    except Exception:
        controller.rollback()  # fails closed if state can't actually be restored
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.governance_context import local_internal_governed_scope
from core.runtime.file_write_gateway import get_file_write_gateway

logger = logging.getLogger("Infra.Rollback")

_STATE_SCHEMA = "aura.rollback.checkpoint_state.v1"


# Guarded-callable failure envelope: user-supplied callables (checks, guards,
# actions, hooks, channels) may raise anything. House discipline forbids broad
# `except Exception`; this tuple names the realistic failure universe
# explicitly. Exotic escapes — custom Exception subtypes outside these bases,
# SystemExit, KeyboardInterrupt — propagate loudly by design.
_GUARDED_CALLABLE_ERRORS = (
    RuntimeError, AttributeError, TypeError, ValueError,
    LookupError, ArithmeticError, OSError, ImportError,
)


@dataclass
class Checkpoint:
    """A state checkpoint for rollback."""
    name: str
    timestamp: float = field(default_factory=time.time)
    state_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    verified: bool = False
    state_required: bool = False
    state_sha256: str = ""
    persistence_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "state_path": self.state_path,
            "verified": self.verified,
            "state_required": self.state_required,
            "state_sha256": self.state_sha256,
            "persistence_error": self.persistence_error,
            "metadata": self.metadata,
        }


class RollbackController:
    """Atomic rollback controller with checkpoint management.

    Thread-safe. Maintains a stack of checkpoints for nested rollback support.
    """

    def __init__(
        self,
        checkpoint_dir: Path | None = None,
        max_checkpoints: int = 10,
    ) -> None:
        self._checkpoint_dir = checkpoint_dir or Path.home() / ".aura" / "checkpoints"
        with local_internal_governed_scope(
            "rollback.ensure_checkpoint_directory",
            domain="file_write",
        ):
            get_file_write_gateway().ensure_directory(
                self._checkpoint_dir,
                source="infrastructure.rollback.init",
            )
        self._lock = threading.Lock()
        self._checkpoints: list[Checkpoint] = []
        self._max_checkpoints = max(1, int(max_checkpoints))
        self._verify_hooks: list[Callable[[], bool]] = []
        self._state_applier: Callable[[dict[str, Any]], bool] | None = None

    def register_verify_hook(self, hook: Callable[[], bool]) -> None:
        """Register a verification hook run after rollback."""
        with self._lock:
            self._verify_hooks.append(hook)

    def register_state_applier(self, applier: Callable[[dict[str, Any]], bool]) -> None:
        """Register the callable that restores collected state on rollback.

        Rollback of a checkpoint that persisted state is only meaningful if
        someone can apply that state; without an applier such a rollback
        fails closed instead of pretending it restored anything.
        """
        with self._lock:
            self._state_applier = applier

    def checkpoint(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
        state_collector: Callable[[], dict[str, Any]] | None = None,
    ) -> Checkpoint:
        """Create a state checkpoint.

        If state_collector is provided, its output is persisted to disk.
        """
        cp = Checkpoint(
            name=str(name),
            metadata=dict(metadata or {}),
            state_required=state_collector is not None,
        )

        if state_collector is not None:
            try:
                state = state_collector()
                if not isinstance(state, dict):
                    raise TypeError("checkpoint state collector must return a dictionary")
                normalized_state = json.loads(json.dumps(state, default=str))
                state_digest = self._state_digest(normalized_state)
                state_path = self._checkpoint_dir / self._state_filename(cp.name)
                envelope = {
                    "schema": _STATE_SCHEMA,
                    "checkpoint_name": cp.name,
                    "checkpoint_timestamp": cp.timestamp,
                    "state_sha256": state_digest,
                    "state": normalized_state,
                }
                with local_internal_governed_scope(
                    "rollback.persist_checkpoint",
                    domain="file_write",
                ):
                    get_file_write_gateway().write_text(
                        state_path,
                        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n",
                        source="infrastructure.rollback.checkpoint",
                    )
                persisted = self._load_checkpoint_state(
                    state_path,
                    checkpoint_name=cp.name,
                    expected_sha256=state_digest,
                )
                if persisted != normalized_state:
                    raise RuntimeError("checkpoint state verification mismatch")
                cp.state_path = str(state_path)
                cp.state_sha256 = state_digest
                cp.verified = True
            except _GUARDED_CALLABLE_ERRORS as exc:
                cp.persistence_error = f"{type(exc).__name__}:{exc}"
                logger.error(
                    "Failed to capture or persist state for checkpoint '%s': %s",
                    name,
                    exc,
                )
        else:
            cp.verified = True

        old: list[Checkpoint] = []
        with self._lock:
            self._checkpoints.append(cp)
            if len(self._checkpoints) > self._max_checkpoints:
                old = self._checkpoints[:-self._max_checkpoints]
                self._checkpoints = self._checkpoints[-self._max_checkpoints:]
        for old_cp in old:
            self._cleanup_checkpoint(old_cp)

        logger.info("CHECKPOINT created: %s (state=%s)", name,
                     "persisted" if cp.state_path else "none")
        return cp

    def rollback(self, checkpoint_name: str | None = None) -> bool:
        """Rollback to the named checkpoint (or most recent if not specified).

        Returns True only if the checkpoint's persisted state was actually
        restored (or the checkpoint carried no state to restore). A rollback
        that merely locates a state file is not a rollback — restoration goes
        through the registered state applier and fails closed without one.
        """
        with self._lock:
            applier = self._state_applier
            if not self._checkpoints:
                logger.error("ROLLBACK FAILED: no checkpoints available")
                return False

            if checkpoint_name:
                target = None
                for cp in reversed(self._checkpoints):
                    if cp.name == checkpoint_name:
                        target = cp
                        break
                if target is None:
                    logger.error("ROLLBACK FAILED: checkpoint '%s' not found",
                                  checkpoint_name)
                    return False
            else:
                target = self._checkpoints[-1]

        logger.warning("ROLLBACK to checkpoint '%s' (created: %s)",
                        target.name, time.strftime("%Y-%m-%d %H:%M:%S",
                                                   time.localtime(target.timestamp)))

        if target.state_required and (
            not target.verified or not target.state_path or target.persistence_error
        ):
            logger.error(
                "ROLLBACK FAILED: checkpoint '%s' required state but persistence "
                "was not verified (%s)",
                target.name,
                target.persistence_error or "missing verified state",
            )
            return False

        # Restore persisted state through the applier
        if target.state_required:
            try:
                state_path = self._owned_state_path(target.state_path)
            except (OSError, TypeError, ValueError) as exc:
                logger.error("ROLLBACK FAILED: invalid state path: %s", exc)
                return False
            if not state_path.exists():
                logger.error("ROLLBACK FAILED: state file missing: %s",
                             target.state_path)
                return False
            try:
                state = self._load_checkpoint_state(
                    state_path,
                    checkpoint_name=target.name,
                    expected_sha256=target.state_sha256,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.error("ROLLBACK FAILED: could not load state: %s", exc)
                return False
            if applier is None:
                logger.error(
                    "ROLLBACK FAILED: checkpoint '%s' has persisted state but "
                    "no state applier is registered — refusing to report a "
                    "restore that did not happen", target.name,
                )
                return False
            try:
                applied = bool(applier(state))
            except _GUARDED_CALLABLE_ERRORS as exc:
                logger.error("ROLLBACK FAILED: state applier raised: %s", exc)
                return False
            if not applied:
                logger.error("ROLLBACK FAILED: state applier rejected state for '%s'",
                             target.name)
                return False
            logger.info("ROLLBACK: State restored from %s (%d keys)",
                         target.state_path, len(state))

        return True

    def verify(self) -> bool:
        """Run all verification hooks to confirm system health post-rollback.

        Returns True if all hooks pass.
        """
        with self._lock:
            hooks = list(self._verify_hooks)

        all_passed = True
        for hook in hooks:
            try:
                if not hook():
                    logger.error("VERIFY: Hook %s failed", hook.__name__)
                    all_passed = False
            except _GUARDED_CALLABLE_ERRORS as exc:
                logger.error("VERIFY: Hook %s raised: %s", hook.__name__, exc)
                all_passed = False

        if all_passed:
            logger.info("VERIFY: All %d hooks passed", len(hooks))
        else:
            logger.error("VERIFY: Some hooks failed")

        return all_passed

    def latest_checkpoint(self) -> Checkpoint | None:
        with self._lock:
            return self._checkpoints[-1] if self._checkpoints else None

    def list_checkpoints(self) -> list[Checkpoint]:
        with self._lock:
            return list(self._checkpoints)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "checkpoint_count": len(self._checkpoints),
                "checkpoints": [cp.to_dict() for cp in self._checkpoints[-5:]],
                "verify_hooks": len(self._verify_hooks),
                "state_applier_registered": self._state_applier is not None,
                "checkpoint_dir": str(self._checkpoint_dir),
            }

    def _cleanup_checkpoint(self, cp: Checkpoint) -> None:
        """Remove old checkpoint files."""
        if cp.state_path:
            try:
                state_path = self._owned_state_path(cp.state_path)
                with local_internal_governed_scope(
                    "rollback.cleanup_checkpoint",
                    domain="file_write",
                ):
                    get_file_write_gateway().delete_file(
                        state_path,
                        source="infrastructure.rollback.cleanup",
                    )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug("Checkpoint cleanup skipped for %s: %s", cp.state_path, exc)

    @staticmethod
    def _state_digest(state: dict[str, Any]) -> str:
        canonical = json.dumps(
            state,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _owned_state_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if path.is_symlink():
            raise ValueError("checkpoint state path cannot be a symlink")
        if path.parent.resolve() != self._checkpoint_dir.expanduser().resolve():
            raise ValueError("checkpoint state path escaped its owned directory")
        return path

    @staticmethod
    def _state_filename(name: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-")[:64]
        if not slug:
            slug = "checkpoint"
        name_digest = hashlib.sha256(name.encode("utf-8", "replace")).hexdigest()[:10]
        return f"{slug}-{name_digest}-{time.time_ns()}-{uuid.uuid4().hex[:8]}.json"

    @classmethod
    def _load_checkpoint_state(
        cls,
        path: Path,
        *,
        checkpoint_name: str,
        expected_sha256: str = "",
    ) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("checkpoint state must be a JSON object")

        schema = payload.get("schema")
        if schema == _STATE_SCHEMA:
            if payload.get("checkpoint_name") != checkpoint_name:
                raise ValueError("checkpoint state belongs to a different checkpoint")
            state = payload.get("state")
            if not isinstance(state, dict):
                raise ValueError("checkpoint envelope state must be an object")
            recorded_sha256 = str(payload.get("state_sha256", "") or "")
            actual_sha256 = cls._state_digest(state)
            if not recorded_sha256 or recorded_sha256 != actual_sha256:
                raise ValueError("checkpoint state digest mismatch")
            if expected_sha256 and expected_sha256 != actual_sha256:
                raise ValueError("checkpoint state no longer matches its receipt")
            return state
        if isinstance(schema, str) and schema.startswith("aura.rollback.checkpoint_state"):
            raise ValueError(f"unsupported checkpoint state schema: {schema}")
        if expected_sha256:
            raise ValueError("verified checkpoint was replaced by legacy unbound state")
        return payload
