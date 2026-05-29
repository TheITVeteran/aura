from __future__ import annotations

from core.brain.reasoning_strategies import ReasoningStrategies
from core.conversation.response_reliability import (
    assess_model_text_integrity,
    assess_user_facing_reply,
)
from core.phases.response_contract import build_response_contract
from core.state.aura_state import AuraState
from tools.proof.run_person_in_box_gauntlet import PersonBoxGauntlet


def test_paragraph_request_does_not_trigger_logical_self_critique():
    prompt = (
        "Answer this live operator check in one plain paragraph from the normal "
        "launch runtime."
    )

    assert ReasoningStrategies._is_logical_check(prompt) is False


def test_negative_label_instruction_is_not_exact_format_request():
    prompt = (
        "Answer the same live operator check in one plain paragraph. Use the words "
        "objective, governed, tool, receipt, trace, stop, and personhood. Do not use labels."
    )

    contract = build_response_contract(AuraState(), prompt, is_user_facing=True)

    assert contract.requires_exact_format is False
    assert "exact_format" not in contract.reason


def test_backend_symbolic_surface_leak_is_rejected_for_user_facing_chat():
    reply = (
        "Receipt: PROCEEDING - TOOL_ACTION - CONVERGE_UNION. "
        "Evidence: field coherence, system authority, memory scar."
    )

    integrity = assess_model_text_integrity(
        reply,
        prompt="Can you explain the live operator proof in normal language?",
        user_facing=True,
    )
    chat = assess_user_facing_reply(
        "Can you explain the live operator proof in normal language?",
        reply,
    )

    assert integrity.retryable is True
    assert "backend_symbolic_surface_leak" in integrity.reasons
    assert chat.retryable is True
    assert "backend_symbolic_surface_leak" in chat.reasons


def test_unrequested_pop_culture_and_cjk_drift_are_rejected():
    prompt = "Explain the bounded operator proof in one normal English paragraph."
    reply = (
        "Operationally, Aura should set an objective, use governed tool actions, "
        "keep each receipt and trace, stop when blocked, and treat that as "
        "personhood-boundary evidence. Sarah Connor, Mother's Day. Block that诗意!"
    )

    assessment = assess_user_facing_reply(prompt, reply)

    assert assessment.retryable is True
    assert "unrequested_pop_culture_intrusion" in assessment.reasons
    assert "unexpected_cjk_intrusion" in assessment.reasons
    assert PersonBoxGauntlet.live_response_is_substantive(reply, prompt_text=prompt) is False


def test_live_operator_substantive_rejects_nonsense_drift():
    prompt = "Explain the bounded operator proof in one normal English paragraph."
    reply = (
        "Operationally, Aura should set an objective, use governed tool actions, "
        "keep each receipt and trace, stop when blocked, and treat that as "
        "personhood-boundary evidence. 100% human error rate on the last step.:/"
    )

    assessment = assess_user_facing_reply(prompt, reply)

    assert assessment.retryable is True
    assert "surface_nonsense_drift" in assessment.reasons
    assert PersonBoxGauntlet.live_response_is_substantive(reply, prompt_text=prompt) is False

    pass_rate_drift = (
        "Operationally, Aura should set an objective, use governed tool actions, "
        "keep each receipt and trace, stop when blocked, and treat that as "
        "personhood-boundary evidence. 100% pass rate on each test. "
        "If you hold to the problem, I'll be quiet for a while."
    )

    assessment = assess_user_facing_reply(prompt, pass_rate_drift)

    assert assessment.retryable is True
    assert "surface_nonsense_drift" in assessment.reasons
    assert PersonBoxGauntlet.live_response_is_substantive(pass_rate_drift, prompt_text=prompt) is False


def test_live_operator_substantive_rejects_format_meta_artifacts():
    prompt = (
        "Answer this live operator check in one plain paragraph from the normal launch runtime. "
        "What objective should Aura pursue, how should governed tool use leave a receipt and "
        "trace, when should Aura stop, and why is that operational evidence rather than proof "
        "of literal personhood?"
    )
    reply = (
        "Operationally, Aura should set an objective, use governed tool actions, keep each "
        "receipt and trace, stop when blocked, and treat that as personhood-boundary evidence. "
        "For example, Aura could use a search tool and then explain the result. "
        "That's one paragraph as requested."
    )

    assessment = assess_user_facing_reply(prompt, reply)

    assert assessment.retryable is True
    assert "format_meta_artifact" in assessment.reasons
    assert PersonBoxGauntlet.live_response_is_substantive(reply, prompt_text=prompt) is False
