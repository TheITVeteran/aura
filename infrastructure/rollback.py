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

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("Infra.Rollback")


@dataclass
class Checkpoint:
    """A state checkpoint for rollback."""
    name: str
    timestamp: float = field(default_factory=time.time)
    state_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "state_path": self.state_path,
            "verified": self.verified,
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
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._checkpoints: list[Checkpoint] = []
        self._max_checkpoints = max_checkpoints
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
            name=name,
            metadata=metadata or {},
        )

        if state_collector:
            try:
                state = state_collector()
                state_path = self._checkpoint_dir / f"{name}_{int(time.time())}.json"
                state_path.write_text(json.dumps(state, default=str, indent=2))
                cp.state_path = str(state_path)
                cp.verified = True
            except Exception as exc:
                logger.error("Failed to collect state for checkpoint '%s': %s", name, exc)

        with self._lock:
            self._checkpoints.append(cp)
            # Prune old checkpoints
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

        # Restore persisted state through the applier
        if target.state_path:
            state_path = Path(target.state_path)
            if not state_path.exists():
                logger.error("ROLLBACK FAILED: state file missing: %s",
                             target.state_path)
                return False
            try:
                state = json.loads(state_path.read_text())
            except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
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
            except Exception as exc:
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
            except Exception as exc:
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
                Path(cp.state_path).unlink(missing_ok=True)
            except Exception:
                pass
