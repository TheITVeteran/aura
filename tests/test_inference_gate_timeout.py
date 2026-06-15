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


def test_foreground_retry_schedule_only_retries_fast_failures() -> None:
    assert InferenceGate._foreground_retry_schedule(
        primary_attempt_elapsed=10.0,
        primary_timeout=150.0,
    ) == (2.0,)
    assert InferenceGate._foreground_retry_schedule(
        primary_attempt_elapsed=61.0,
        primary_timeout=150.0,
    ) == ()


def test_think_preserves_desktop_cognitive_engine_contract() -> None:
    async def run() -> None:
        gate = InferenceGate()
        captured: dict[str, object] = {}

        async def fake_generate(prompt, context=None, timeout=None):
            captured["prompt"] = prompt
            captured["context"] = dict(context or {})
            captured["timeout"] = timeout
            return "ready"

        gate.generate = fake_generate  # type: ignore[method-assign]
        gate._post_inference_update = lambda _text: None  # type: ignore[method-assign]

        result = await gate.think(
            "hello",
            origin="desktop_ui",
            cognitive_engine_required=True,
            desktop_cognitive_engine_required=True,
        )

        assert result == "ready"
        assert captured["context"]["cognitive_engine_required"] is True
        assert captured["context"]["desktop_cognitive_engine_required"] is True

    asyncio.run(run())
