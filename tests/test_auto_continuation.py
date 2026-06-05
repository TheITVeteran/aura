import asyncio
from types import SimpleNamespace

import pytest

from core.orchestrator.mixins.message_handling import MessageHandlingMixin


class ProcessingStatus:
    is_processing = False


class RecordingInferenceGate:
    SILENCE_SENTINEL = "<|SILENCE|>"

    def __init__(self):
        self.responses = []
        self.calls = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def generate(self, message, context=None):
        self.calls.append({"message": message, "context": dict(context or {})})
        if not self.responses:
            raise AssertionError("generation response queue exhausted")
        return self.responses.pop(0)


class ContinuationOrchestrator(MessageHandlingMixin):
    def __init__(self):
        self.status = ProcessingStatus()
        self._inference_gate = RecordingInferenceGate()
        self.conversation_history = []

        self._lock = asyncio.Lock()
        self._last_emitted_fingerprint = ""
        self.gate_ready_contexts = []
        self.quiet_window_extensions = []
        self.telemetry_events = []
        self._last_user_interaction_time = 0.0
        
    def _is_user_facing_origin(self, origin):
        return origin == "user"
        
    async def _ensure_inference_gate_ready(self, context=None):
        self.gate_ready_contexts.append(context)
        return True
        
    def _get_fingerprint(self, text):
        return text
        
    def _extend_foreground_quiet_window(self, amt):
        self.quiet_window_extensions.append(amt)
        
    def _publish_telemetry(self, data):
        self.telemetry_events.append(data)
        
    def _record_message_in_history(self, message, role):
        self.conversation_history.append({"role": role, "content": message})


class KernelNotReady:
    def is_ready(self):
        return False


class ApprovedWillDecision:
    outcome = SimpleNamespace(value="approved")
    reason = ""
    constraints = None
    receipt_id = "will_receipt_approved"

    def is_approved(self):
        return True


class ApprovedWill:
    _started = True

    def __init__(self):
        self.decisions = []

    def decide(self, **kwargs):
        self.decisions.append(kwargs)
        return ApprovedWillDecision()


@pytest.mark.asyncio
async def test_auto_continuation_triggers(monkeypatch):
    orchestrator = ContinuationOrchestrator()

    import core.kernel.kernel_interface as ki
    import core.container as container
    import core.will as will

    approved_will = ApprovedWill()
    monkeypatch.setattr(ki.KernelInterface, "get_instance", lambda: KernelNotReady())
    monkeypatch.setattr(container.ServiceContainer, "get", lambda *args, **kwargs: None)
    monkeypatch.setattr(will, "get_will", lambda: approved_will)

    long_first_part = "This is the first part of a very long sentence that just cuts off. " * 5 + "and then it cuts off"
    long_second_part = " right here. And this is the end."
    orchestrator._inference_gate.responses = [
        long_first_part,
        long_second_part
    ]
    
    response = await orchestrator._process_user_input_core("Tell me a story.", origin="user")
    
    assert orchestrator._inference_gate.call_count == 2
    assert response == long_first_part + long_second_part
    assert approved_will.decisions[0]["domain"].value == "response"
    assert orchestrator.telemetry_events
    assert orchestrator._inference_gate.calls[1]["message"].startswith("[SYSTEM:")
    assert orchestrator._inference_gate.calls[1]["context"]["prefer_tier"] == "primary"
