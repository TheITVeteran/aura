from __future__ import annotations

import asyncio

from core.brain.inference_gate import InferenceGate
from core.utils.deadlines import get_deadline


class HangingClient:
    def __init__(self) -> None:
        self.abort_reasons: list[str] = []

    async def generate_text_async(self, **kwargs):
        await asyncio.sleep(5)
        return "late text"

    def force_abort_active_generation(self, reason: str = "hard_generation_deadline") -> bool:
        self.abort_reasons.append(reason)
        return True


def test_generate_with_client_aborts_when_client_ignores_deadline() -> None:
    async def run() -> None:
        client = HangingClient()
        gate = InferenceGate()

        text = await gate._generate_with_client(
            client,
            "say hello",
            "You are Aura.",
            [],
            get_deadline(0.05),
            "Cortex",
            foreground_request=True,
            origin="user",
        )

        assert text is None
        assert client.abort_reasons
        assert client.abort_reasons[0].startswith("inference_gate_generation_timeout:Cortex:")

    asyncio.run(run())
