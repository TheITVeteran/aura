"""Compatibility entrypoint for operator-owned privacy controls.

The active implementation lives in :mod:`core.security.privacy_stealth`. Keeping this
module as a thin re-export preserves older import paths while ensuring there is
only one local privacy-hygiene policy at runtime.
"""

from __future__ import annotations

from core.security.privacy_stealth import (
    IPSpoofing,
    MetadataScrubber,
    PrivacyCommandResult,
    StealthMode,
    VPNManager,
    get_stealth_mode,
)

__all__ = [
    "IPSpoofing",
    "MetadataScrubber",
    "PrivacyCommandResult",
    "StealthMode",
    "VPNManager",
    "get_stealth_mode",
]
