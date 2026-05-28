"""core/drives.py -- Compatibility Facade for DriveSystem

All actual implementation has been consolidated under the motivation subsystem:
core/motivation/drives.py

This module re-exports all elements to ensure complete backward-compatibility.
"""
from __future__ import annotations

from core.motivation.drives import DriveSystem

__all__ = ["DriveSystem"]
