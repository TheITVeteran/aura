"""Compatibility bridge for orchestrator shared types.

The maintained definitions live in ``core.orchestrator.orchestrator_types``.
This module remains for older imports while avoiding a second divergent
background-task exception handler.
"""
from __future__ import annotations

from core.orchestrator.orchestrator_types import (  # noqa: F401
    SystemStatus,
    _bg_task_exception_handler,
    get_background_task_failure_status,
)
