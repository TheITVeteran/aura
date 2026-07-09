"""Compatibility privacy surface.

The historical module name is kept for older imports, but active stealth,
proxy, VPN, identity-rotation, and network-obfuscation operations are not
implemented here. Aura's live runtime exposes local metadata hygiene only; any
future external privacy action must be built through explicit governed
gateways with receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.utils.privacy_hygiene import MetadataScrubber


@dataclass(frozen=True)
class PrivacyCommandResult:
    ok: bool
    status: str
    message: str
    data: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "status": self.status,
            "message": self.message,
        }
        if self.data is not None:
            payload["data"] = self.data
        return payload


class VPNManager:
    """Inert compatibility object for removed direct VPN controls."""

    current_vpn: str | None = None
    available_vpns: list[str] = []

    async def connect_vpn(self, vpn_type: str | None = None, server: str | None = None) -> bool:
        return False

    async def disconnect_vpn(self) -> bool:
        return True

    async def get_current_ip(self) -> str | None:
        return None

    def is_vpn_active(self) -> bool:
        return False


class IPSpoofing:
    """Inert compatibility object for removed proxy rotation controls."""

    proxy_list: list[dict[str, str]] = []
    current_proxy: dict[str, str] | None = None

    async def load_proxy_list(self, source: str = "free") -> bool:
        return False

    def rotate_proxy(self) -> dict[str, str] | None:
        return None

    def get_current_proxy(self) -> dict[str, str] | None:
        return None

    async def test_proxy(self, proxy: dict[str, str]) -> bool:
        return False


class StealthMode:
    """Local metadata-hygiene controller with inert legacy stealth controls."""

    def __init__(self) -> None:
        self.vpn = VPNManager()
        self.ip_spoof = IPSpoofing()
        self.scrubber = MetadataScrubber()
        self.stealth_enabled = True

    async def enable_stealth(self, vpn_server: str | None = None) -> bool:
        self.stealth_enabled = True
        return False

    async def disable_stealth(self) -> bool:
        self.stealth_enabled = False
        return True

    def process_output(self, text: str) -> str:
        return self.scrubber.scrub_text(text) if self.stealth_enabled else text

    async def get_stealth_status(self) -> dict[str, Any]:
        return {
            "enabled": self.stealth_enabled,
            "metadata_scrubbing": self.stealth_enabled,
            "vpn_active": False,
            "proxy_active": False,
            "active_network_stealth": "not_available_from_compatibility_layer",
        }


_stealth_instance: StealthMode | None = None


def get_stealth_mode() -> StealthMode:
    global _stealth_instance
    if _stealth_instance is None:
        _stealth_instance = StealthMode()
    return _stealth_instance


__all__ = [
    "IPSpoofing",
    "MetadataScrubber",
    "PrivacyCommandResult",
    "StealthMode",
    "VPNManager",
    "get_stealth_mode",
]
