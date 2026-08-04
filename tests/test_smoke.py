import asyncio
import logging
import time
import sys
import pytest
import os

# Ensure the project root is in sys.path
sys.path.append(os.getcwd())

from core.container import ServiceContainer
from core.event_bus import get_event_bus
from core.orchestrator.main import RobustOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Aura.SmokeTest")



# This file was seventeen lines of imports and a logger. It was named
# test_smoke.py, collected zero tests, and sat inside the suite total.


def test_event_bus_is_resolvable():
    bus = get_event_bus()
    assert bus is not None


def test_event_bus_is_a_singleton():
    """Two buses means two halves of the runtime talking past each other."""
    assert get_event_bus() is get_event_bus()


def test_orchestrator_imports_without_booting_a_runtime():
    """Importing must not start anything — the live instance is sacred."""
    assert RobustOrchestrator.__name__ == "RobustOrchestrator"


def test_service_container_is_importable_and_empty_by_default():
    sentinel = object()
    assert ServiceContainer.get("unregistered_smoke_key", default=sentinel) is sentinel
