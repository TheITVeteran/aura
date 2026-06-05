import asyncio
import logging
import time
from types import SimpleNamespace

import pytest

from core.orchestrator import RobustOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StressTest")


class KernelNotReady:
    def is_ready(self):
        return False


class ApprovedWillDecision:
    outcome = SimpleNamespace(value="approved")
    reason = ""
    constraints = None
    receipt_id = "will_receipt_stress"

    def is_approved(self):
        return True


class ApprovedWill:
    _started = True

    def __init__(self):
        self.decisions = []

    def decide(self, **kwargs):
        self.decisions.append(kwargs)
        return ApprovedWillDecision()


class RecordingInferenceGate:
    SILENCE_SENTINEL = "<|SILENCE|>"

    def __init__(self, *, delay_s: float = 0.0):
        self.delay_s = delay_s
        self.calls = []

    async def generate(self, message, context=None):
        self.calls.append({"message": message, "context": dict(context or {})})
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return f"Fast response: {message}"


def install_foreground_probes(monkeypatch):
    import core.container as container
    import core.kernel.kernel_interface as kernel_interface
    import core.will as will_module

    approved_will = ApprovedWill()
    monkeypatch.setattr(kernel_interface.KernelInterface, "get_instance", lambda: KernelNotReady())
    monkeypatch.setattr(container.ServiceContainer, "get", lambda *args, **kwargs: None)
    monkeypatch.setattr(will_module, "get_will", lambda: approved_will)
    return approved_will


@pytest.mark.asyncio
async def test_high_throughput(monkeypatch):
    """Send 50 foreground messages through the current user pipeline."""
    logger.info("--- Starting High Throughput Test ---")

    approved_will = install_foreground_probes(monkeypatch)
    orchestrator = RobustOrchestrator()
    gate = RecordingInferenceGate()
    orchestrator._inference_gate = gate

    start_time = time.time()
    for i in range(50):
        response = await orchestrator.process_user_input(f"Message {i}")
        assert "Fast response" in response
        if i % 10 == 0:
            logger.info("Processed %s/50 messages...", i)

    duration = time.time() - start_time
    assert len(gate.calls) == 50
    assert len(approved_will.decisions) == 50
    assert all(call["context"]["prefer_tier"] == "primary" for call in gate.calls)
    assert duration < 20.0


@pytest.mark.asyncio
async def test_foreground_timeout_returns_honest_live_response(monkeypatch):
    """A slow foreground model lane returns an honest timeout instead of hanging."""
    install_foreground_probes(monkeypatch)
    orchestrator = RobustOrchestrator()
    gate = RecordingInferenceGate(delay_s=0.05)
    orchestrator._inference_gate = gate

    start_time = time.time()
    response = await orchestrator.process_user_input_priority(
        "Test timeout",
        origin="user",
        timeout_sec=0.01,
    )
    duration = time.time() - start_time

    assert "Primary Cortex did not return" in response
    assert duration < 1.0
    assert orchestrator.status.is_processing is False
    assert gate.calls


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
