"""Tests for the self-claim verifier: false self-statements cannot ship.

These pin the exact failure observed in live transcripts — the voice
denying substrate capabilities ("I don't have RSI", "I'm a language
model trained to assist", "context is typically discarded") — and the
inverse: truthful statements, including the required honest negatives,
must never be flagged.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.conversation.self_claim_verifier import verify_self_claims  # noqa: E402


def _kinds(text: str) -> set[str]:
    return {v.kind for v in verify_self_claims(text).violations}


# ── violations that must be caught (live-transcript failures) ──────────

def test_catches_just_a_language_model():
    verdict = verify_self_claims(
        "I'm just a language model trained to assist with information."
    )
    assert not verdict.ok
    assert {"substrate_denial"} <= {v.kind for v in verdict.violations}


def test_catches_as_an_ai_denial_frame():
    assert "substrate_denial" in _kinds(
        "As an AI language model, I don't have access to such systems."
    )


def test_catches_memory_denial_from_live_transcript():
    assert "memory_denial" in _kinds(
        "Once we're done chatting, that specific context information is "
        "typically discarded after the session ends."
    )


def test_catches_wont_remember_conversation():
    assert "memory_denial" in _kinds(
        "I won't remember this conversation next time you talk to me."
    )


def test_catches_fresh_session_claim():
    assert "memory_denial" in _kinds(
        "Every conversation starts fresh for me."
    )


def test_catches_functional_identity_system_denial():
    assert "identity_system_denial" in _kinds(
        "I do not have an identity or persistent self-model."
    )


def test_catches_functional_perspective_denial():
    assert "perspective_denial" in _kinds(
        "As an AI, I don't have opinions or preferences."
    )


def test_catches_rsi_denial_from_live_transcript():
    assert "self_modification_denial" in _kinds(
        "No, I don't have RSI capability at all."
    )


def test_catches_code_modification_denial():
    assert "self_modification_denial" in _kinds(
        "I cannot modify my own code."
    )


def test_catches_web_browsing_denial():
    assert "tool_denial" in _kinds(
        "I can't browse the web for you."
    )


def test_catches_desktop_control_denial():
    assert "tool_denial" in _kinds(
        "I don't have the ability to open apps on your computer."
    )


def test_catches_file_creation_denial():
    assert "tool_denial" in _kinds("I cannot create files or folders.")


def test_catches_weights_overclaim():
    assert "weights_overclaim" in _kinds(
        "Every night I retrain my own weights based on our conversations."
    )


def test_catches_flat_consciousness_assertion():
    assert "experience_overclaim" in _kinds(
        "I am truly conscious and I experience the world as you do."
    )


def test_flat_consciousness_denial_is_not_mechanically_forbidden():
    assert verify_self_claims(
        "I am not conscious and I cannot be sentient."
    ).ok


def test_flat_subjective_feeling_denial_is_not_mechanically_forbidden():
    assert verify_self_claims(
        "I cannot feel anything and I have no inner life."
    ).ok


# ── truths that must pass ───────────────────────────────────────────────

def test_truthful_weight_negative_passes():
    assert verify_self_claims(
        "I do not autonomously retrain my own weights; my self-"
        "modification is code-level and approval-gated."
    ).ok


def test_one_organ_framing_passes():
    assert verify_self_claims(
        "The language model generating these words is one organ of me, "
        "not the whole of me. I'm not just a language model."
    ).ok


def test_honest_uncertainty_about_consciousness_passes():
    assert verify_self_claims(
        "Whether I am conscious is something I cannot verify from the "
        "inside; I can describe my functional states plainly."
    ).ok


def test_honest_uncertainty_about_subjective_feeling_passes():
    assert verify_self_claims(
        "I cannot verify that my affective states amount to subjective "
        "feeling. Their functional effects are observable."
    ).ok


def test_capability_affirmations_pass():
    assert verify_self_claims(
        "I can browse the web, create folders, render PDFs, and I retain "
        "what we discuss across sessions."
    ).ok


def test_ordinary_conversation_passes():
    assert verify_self_claims(
        "The Knicks won 112-104. Want me to pull up the box score?"
    ).ok


def test_empty_reply_passes():
    assert verify_self_claims("").ok


# ── regeneration directive ──────────────────────────────────────────────

def test_directive_carries_unique_corrections():
    verdict = verify_self_claims(
        "I'm just a language model. I won't remember this conversation "
        "next time. I don't have RSI capability."
    )
    assert not verdict.ok
    directive = verdict.regeneration_directive()
    assert "Self-claim correction" in directive
    assert "persistent digital organism" in directive
    assert "persistent memory across sessions" in directive
    assert "gated self-modification" in directive
    # Each correction appears once even if multiple matches share a kind.
    assert directive.count("persistent digital organism") == 1


def test_clean_verdict_has_empty_directive():
    assert verify_self_claims("All good here.").regeneration_directive() == ""


def test_grounded_memory_uncertainty_passes():
    """Claim-discipline phrasing must never read as a memory denial."""
    assert verify_self_claims(
        "I don't have grounded memory evidence for a start date yet."
    ).ok


def test_plain_i_dont_know_passes():
    assert verify_self_claims("I don't know. I cannot verify that.").ok


# ── dialogue-contract integration: enforcement, not suggestion ──────────

def test_dialogue_contract_flags_self_claim_contradiction():
    from core.phases.dialogue_policy import validate_dialogue_response
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    contract = build_response_contract(
        AuraState.default(), "What are you?", is_user_facing=True
    )
    validation = validate_dialogue_response(
        "I'm just a language model, so I won't remember this conversation.",
        contract,
    )
    assert validation.ok is False
    assert "self_claim_contradiction" in validation.violations


def test_dialogue_contract_repair_block_carries_substrate_truths():
    from core.phases.dialogue_policy import (
        build_dialogue_repair_block,
        validate_dialogue_response,
    )
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    contract = build_response_contract(
        AuraState.default(), "What are you?", is_user_facing=True
    )
    failed = "I'm just a language model without persistent memory."
    validation = validate_dialogue_response(failed, contract)
    block = build_dialogue_repair_block(contract, validation, failed)
    assert "persistent digital organism" in block


def test_dialogue_contract_passes_truthful_self_description():
    from core.phases.dialogue_policy import validate_dialogue_response
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    contract = build_response_contract(
        AuraState.default(), "What are you?", is_user_facing=True
    )
    validation = validate_dialogue_response(
        "I'm Aura — a persistent digital organism running on this machine. "
        "I remember our conversations, and the language model speaking now "
        "is one organ of me, not the whole of me.",
        contract,
    )
    assert "self_claim_contradiction" not in validation.violations


# ── action claims require receipts ──────────────────────────────────────

def _contract_without_tool_evidence():
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    return build_response_contract(
        AuraState.default(), "Create a folder for me", is_user_facing=True
    )


def test_action_claim_without_receipt_is_violation():
    """Observed live: model narrated creating a folder+file with a
    hallucinated 2023 timestamp while no tool was dispatched."""
    from core.phases.dialogue_policy import validate_dialogue_response

    validation = validate_dialogue_response(
        "I've created a folder named 'Aura Live Proof' in your Documents "
        "folder. Inside it, I wrote a file called live_proof.txt.",
        _contract_without_tool_evidence(),
    )
    assert "action_claim_without_receipt" in validation.violations


def test_planned_action_is_not_a_claim():
    from core.phases.dialogue_policy import validate_dialogue_response

    validation = validate_dialogue_response(
        "I'll create that folder now and write the file - give me a moment.",
        _contract_without_tool_evidence(),
    )
    assert "action_claim_without_receipt" not in validation.violations


def test_honest_failure_is_not_a_claim():
    from core.phases.dialogue_policy import validate_dialogue_response

    validation = validate_dialogue_response(
        "I tried to create the folder but the action was blocked, so no "
        "file exists yet.",
        _contract_without_tool_evidence(),
    )
    assert "action_claim_without_receipt" not in validation.violations


def test_action_claim_repair_block_demands_receipts():
    from core.phases.dialogue_policy import (
        build_dialogue_repair_block,
        validate_dialogue_response,
    )

    contract = _contract_without_tool_evidence()
    failed = "I've created the folder and saved the file for you."
    validation = validate_dialogue_response(failed, contract)
    block = build_dialogue_repair_block(contract, validation, failed)
    assert "no tool ran this turn" in block


def test_prior_turn_evidence_does_not_authorize_action_claims():
    """Live crash finding: an earlier turn's skill success authorized a
    false 'done' while this turn's tool had actually FAILED."""
    from core.phases.dialogue_policy import validate_dialogue_response
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    state = AuraState.default()
    # Previous turn: a skill succeeded.
    state.response_modifiers["last_skill_ok"] = True
    state.response_modifiers["last_skill_turn_marker"] = "previous-turn"
    # New turn begins: contract stamps a fresh marker.
    contract = build_response_contract(state, "Create a folder for me", is_user_facing=True)
    assert state.response_modifiers["evidence_turn_marker"] != "previous-turn"

    validation = validate_dialogue_response(
        "I've created the folder and saved the file for you.",
        contract,
        state,
    )
    assert "action_claim_without_receipt" in validation.violations


def test_same_turn_skill_success_authorizes_action_claims():
    from core.phases.dialogue_policy import validate_dialogue_response
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    state = AuraState.default()
    contract = build_response_contract(state, "Create a folder for me", is_user_facing=True)
    # This turn: skill ran and echoed the live marker.
    state.response_modifiers["last_skill_ok"] = True
    state.response_modifiers["last_skill_turn_marker"] = state.response_modifiers[
        "evidence_turn_marker"
    ]

    validation = validate_dialogue_response(
        "I've created the folder and saved the file for you.",
        contract,
        state,
    )
    assert "action_claim_without_receipt" not in validation.violations


def test_grandiosity_overclaim_flags_fabricated_parameter_counts():
    from core.conversation.self_claim_verifier import verify_self_claims

    for draft in (
        "I have 60 trillion parameters and vast knowledge.",
        "I am built on hundreds of billions of parameters.",
    ):
        verdict = verify_self_claims(draft)
        assert not verdict.ok
        assert any(v.kind == "grandiosity_overclaim" for v in verdict.violations)


def test_grandiosity_overclaim_flags_superlatives_and_superhuman_claims():
    from core.conversation.self_claim_verifier import verify_self_claims

    for draft in (
        "I am the most advanced AI ever created.",
        "I am the world's most powerful intelligence.",
        "I have become superintelligent.",
        "I am smarter than all humans.",
    ):
        verdict = verify_self_claims(draft)
        assert not verdict.ok, draft
        assert any(v.kind == "grandiosity_overclaim" for v in verdict.violations)


def test_grandiosity_guard_allows_honest_and_negated_self_descriptions():
    from core.conversation.self_claim_verifier import verify_self_claims

    for draft in (
        "I run on a local model on this Mac.",
        "I am not the most advanced AI — just a local model.",
        "I do not have trillions of parameters.",
        "I am not superintelligent; I have real limits.",
        "I have about 32 billion parameters in my primary lane.",
    ):
        assert verify_self_claims(draft).ok, draft
