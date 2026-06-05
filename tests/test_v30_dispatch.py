"""tests/test_v30_dispatch.py
───────────────────────────
Verifies that spontaneous autonomous messages reach the primary target.
"""

import asyncio
import logging
import unittest

from core.utils.output_gate import AutonomousOutputGate

logging.basicConfig(level=logging.INFO)


class StatusProbe:
    running = True


class OutputGateHarness:
    AI_ROLE = "assistant"

    def __init__(self):
        self.reply_queue = asyncio.Queue()
        self.status = StatusProbe()
        self.conversation_history = []
        self.output_gate = None


class TestV30Dispatch(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.orchestrator = OutputGateHarness()
        self.gate = AutonomousOutputGate(self.orchestrator)
        self.orchestrator.output_gate = self.gate
        
    async def test_spontaneous_bypass(self):
        """Spontaneous autonomous messages should reach primary target."""
        message = "Hello from the Void."
        metadata = {
            "autonomous": True,
            "spontaneous": True,
        }
        
        await self.gate.emit(message, origin="test", target="primary", metadata=metadata)
        
        self.assertFalse(self.orchestrator.reply_queue.empty())
        item = await self.orchestrator.reply_queue.get()
        self.assertEqual(item, message)
        self.assertEqual(self.orchestrator.conversation_history[-1]["content"], message)

    async def test_normal_autonomous_redirection(self):
        """Normal autonomous messages (non-spontaneous) should still be redirected to secondary."""
        message = "Technical log that should be hidden."
        metadata = {
            "autonomous": True,
            "spontaneous": False
        }
        
        while not self.orchestrator.reply_queue.empty():
            self.orchestrator.reply_queue.get_nowait()
        
        await self.gate.emit(message, origin="test", target="primary", metadata=metadata)
        
        self.assertTrue(self.orchestrator.reply_queue.empty())
        secondary_payload = await self.gate.secondary_queue.get()
        self.assertEqual(secondary_payload["content"], message)
        self.assertEqual(secondary_payload["origin"], "test")
        self.assertTrue(secondary_payload["metadata"]["autonomous"])

if __name__ == "__main__":
    unittest.main()
