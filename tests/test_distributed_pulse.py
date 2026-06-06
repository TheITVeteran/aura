################################################################################

"""tests/test_distributed_pulse.py
Unit test for PulseManager distributed discovery logic.
"""
from types import SimpleNamespace
import unittest

from core.senses.pulse_manager import PulseManager

class TestPulseDiscovery(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.orchestrator = SimpleNamespace(
            loop=None,  # Will be set in async context
            peers={},
        )
        self.pulse_manager = PulseManager(self.orchestrator)
        self.pulse_manager.running = True
