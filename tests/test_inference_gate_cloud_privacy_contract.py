from __future__ import annotations

from core.brain.inference_gate import InferenceGate
from core.runtime.errors import get_degradation_tracker


class BrokenScrubber:
    def __init__(self) -> None:
        self.called = False

    def __call__(self, _text: str) -> str:
        self.called = True
        raise RuntimeError("scrubber offline")


def test_cloud_payload_scrub_failure_blocks_cloud_fallback():
    tracker = get_degradation_tracker()
    tracker.reset()
    gate = InferenceGate.__new__(InferenceGate)
    scrubber = BrokenScrubber()

    scrubbed = gate._scrub_cloud_payload(
        "System contains private context",
        "User prompt contains private context",
        scrubber=scrubber,
    )

    assert scrubbed is None
    assert scrubber.called is True
    recent = tracker.recent(subsystem="inference_gate", limit=1)
    assert recent
    assert recent[0].severity == "critical"
    assert "blocked cloud fallback" in recent[0].action


def test_cloud_payload_scrubber_returns_sanitized_pair():
    gate = InferenceGate.__new__(InferenceGate)

    def scrubber(text: str) -> str:
        return text.replace("Bryan", "[REDACTED]")

    assert gate._scrub_cloud_payload(
        "System mentions Bryan",
        "Prompt mentions Bryan",
        scrubber=scrubber,
    ) == ("System mentions [REDACTED]", "Prompt mentions [REDACTED]")
