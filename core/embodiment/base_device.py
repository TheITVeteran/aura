from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from core.runtime.errors import record_degradation

if TYPE_CHECKING:
    from core.embodiment.reality_adapter import HardwareRealityManifest

logger = logging.getLogger("Embodiment.BaseDevice")

class BaseHardwareDevice(ABC):
    """
    Abstract Base Class for all physical hardware components that Aura can embody.
    Provides a standardized interface and threading locks to prevent concurrent hardware collisions.
    """
    def __init__(self, device_id: str, device_name: str, device_type: str) -> None:
        self.device_id = device_id
        self.device_name = device_name
        self.device_type = device_type
        self.is_connected = False
        
        # A lock to ensure hardware commands are serialized
        self.hardware_lock = asyncio.Lock()

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the hardware device."""
        pass  # no-op: intentional

    @abstractmethod
    async def disconnect(self) -> bool:
        """Gracefully disconnect from the hardware device."""
        pass  # no-op: intentional

    @abstractmethod
    async def get_status(self) -> dict[str, Any]:
        """Query the device for its current state and telemetry."""
        pass  # no-op: intentional

    @abstractmethod
    async def execute_command(self, command: str, **kwargs: Any) -> dict[str, Any]:
        """
        Execute a hardware-specific command.
        Should ideally be wrapped by `safe_execute` for thread safety.
        """
        pass  # no-op: intentional

    def reality_manifest(self) -> HardwareRealityManifest | None:
        """Return an explicit physical capability contract, if this device has one.

        A device is inventory-only until its implementation deliberately opts in
        with a complete manifest. Hardware authority is never inferred from an
        ``execute_command`` method or a device-type string.
        """

        return None

    async def check_interlocks(
        self,
        command: str,
        parameters: Mapping[str, Any],
        status: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Perform the device-local, last-moment interlock check.

        Manifest-bearing devices must override this method. Refusal by default
        prevents a generic driver from becoming physically executable merely by
        publishing metadata.
        """

        return {
            "ok": False,
            "reason": "device_interlock_contract_not_implemented",
        }

    async def safe_execute(
        self,
        command: str,
        *,
        timeout_s: float = 10.0,
        expected_interlock_sha256: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Thread-safe execution wrapper. Most hardware interfaces (like Serial or distinct IoT sockets)
        can crash or corrupt if hit with concurrent commands.
        """
        if not self.is_connected:
            return {"ok": False, "error": f"Device {self.device_name} is not connected."}

        bounded_timeout = max(0.05, min(float(timeout_s), 120.0))
        async with self.hardware_lock:
            try:
                status = await asyncio.wait_for(
                    self.get_status(),
                    timeout=min(bounded_timeout, 10.0),
                )
                if not isinstance(status, Mapping):
                    return {"ok": False, "error": "device_status_not_mapping"}
                interlock = await asyncio.wait_for(
                    self.check_interlocks(command, kwargs, status),
                    timeout=min(bounded_timeout, 5.0),
                )
                if not isinstance(interlock, Mapping) or interlock.get("ok") is not True:
                    reason = "device_interlock_refused"
                    if isinstance(interlock, Mapping):
                        reason = str(interlock.get("reason") or reason)[:200]
                    return {"ok": False, "error": reason, "interlock_refused": True}
                observed_digest = str(interlock.get("interlock_sha256") or "")
                if expected_interlock_sha256 and observed_digest != expected_interlock_sha256:
                    return {
                        "ok": False,
                        "error": "device_interlock_state_changed_before_dispatch",
                        "interlock_refused": True,
                    }
                # Add a timeout to prevent deadlocks on hardware hangs
                return await asyncio.wait_for(
                    self.execute_command(command, **kwargs),
                    timeout=bounded_timeout,
                )
            except TimeoutError:
                logger.error("Timeout executing command '%s' on device %s", command, self.device_id)
                # Attempt to recover connection state on timeout
                self.is_connected = False
                return {"ok": False, "error": "Hardware command timed out. Connection forced closed."}
            except asyncio.CancelledError:
                raise
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('base_device', e)
                logger.error("Execution failed on device %s: %s", self.device_id, e, exc_info=True)
                return {"ok": False, "error": str(e)}

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata for the cognitive orchestrator."""
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "device_type": self.device_type,
            "is_connected": self.is_connected
        }
