from __future__ import annotations

from unittest.mock import patch

from core.will import ActionDomain, UnifiedWill, WillOutcome


def _neutral_will() -> UnifiedWill:
    will = UnifiedWill()
    will._started = True
    return will


def test_low_unity_blocks_external_tool_action():
    will = _neutral_will()
    with patch.object(will, "_consult_substrate", return_value=(0.8, 0.0, "receipt")):
        with patch.object(will, "_read_affect_valence", return_value=0.0):
            with patch.object(
                will,
                "_read_unity_context",
                return_value={
                    "level": "fragmented",
                    "unity_score": 0.34,
                    "fragmentation_score": 0.66,
                    "safe_to_act": False,
                    "safe_to_self_report": True,
                    "memory_commit_mode": "qualified",
                    "ownership_confidence": 0.9,
                },
            ):
                decision = will.decide(
                    content="publish this update to github",
                    source="tool_runner",
                    domain=ActionDomain.TOOL_EXECUTION,
                    context={"external_action": True},
                )

    assert decision.outcome == WillOutcome.REFUSE
    assert decision.unity_level == "fragmented"


def test_low_unity_blocks_all_consequential_external_surfaces():
    will = _neutral_will()
    unity_context = {
        "level": "fragmented",
        "unity_score": 0.2,
        "fragmentation_score": 0.8,
        "safe_to_act": False,
        "safe_to_self_report": True,
        "memory_commit_mode": "qualified",
        "ownership_confidence": 0.9,
        "causal_closure_score": 0.8,
    }

    with patch.object(will, "_consult_substrate", return_value=(0.8, 0.0, "receipt")):
        with patch.object(will, "_read_affect_valence", return_value=0.0):
            with patch.object(will, "_read_unity_context", return_value=unity_context):
                for domain in (
                    ActionDomain.EXTERNAL_ACTION,
                    ActionDomain.NETWORK_CALL,
                    ActionDomain.CLOUD_CALL,
                    ActionDomain.CLOUD_FALLBACK,
                    ActionDomain.FILE_WRITE,
                    ActionDomain.CI_CD,
                    ActionDomain.SELF_MODIFICATION,
                    ActionDomain.SEMANTIC_WEIGHT_UPDATE,
                    ActionDomain.BELIEF_UPDATE,
                ):
                    decision = will.decide(
                        content=f"attempt {domain.value}",
                        source="unity_negative_test",
                        domain=domain,
                    )
                    assert decision.outcome == WillOutcome.REFUSE
                    assert "unity_block" in decision.reason


def test_low_unity_allows_stabilization():
    will = _neutral_will()
    with patch.object(will, "_consult_substrate", return_value=(0.8, 0.0, "receipt")):
        with patch.object(will, "_read_affect_valence", return_value=0.0):
            with patch.object(
                will,
                "_read_unity_context",
                return_value={
                    "level": "fragmented",
                    "unity_score": 0.3,
                    "fragmentation_score": 0.7,
                    "safe_to_act": False,
                    "memory_commit_mode": "qualified",
                    "ownership_confidence": 0.9,
                },
            ):
                decision = will.decide(
                    content="recenter and stabilize",
                    source="homeostasis",
                    domain=ActionDomain.STABILIZATION,
                )

    assert decision.is_approved()
    assert decision.outcome in {WillOutcome.PROCEED, WillOutcome.CONSTRAIN}


def test_low_unity_defers_memory_write_when_drafts_are_unstable():
    will = _neutral_will()
    with patch.object(will, "_consult_substrate", return_value=(0.8, 0.0, "receipt")):
        with patch.object(will, "_read_affect_valence", return_value=0.0):
            with patch.object(
                will,
                "_read_unity_context",
                return_value={
                    "level": "fragmented",
                    "unity_score": 0.29,
                    "fragmentation_score": 0.71,
                    "safe_to_act": False,
                    "memory_commit_mode": "defer",
                    "ownership_confidence": 0.9,
                },
            ):
                decision = will.decide(
                    content="store this as a settled memory",
                    source="memory_facade",
                    domain=ActionDomain.MEMORY_WRITE,
                )

    assert decision.outcome == WillOutcome.DEFER
    assert "unity_memory_defer" in decision.reason
