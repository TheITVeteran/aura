"""core/agency_bus.py -- Compatibility Facade for AgencyBus

All actual implementation has been consolidated under the agency subsystem:
core/agency/agency_bus.py

This module re-exports all elements to ensure complete backward-compatibility.
"""
from __future__ import annotations

from core.agency.agency_bus import AgencyBus

__all__ = ["AgencyBus"]
