"""tests/test_strict_contract_steering_clamp.py
================================================
Strict/structured proof generations must run with affective steering driven
near-off. Full steering (alpha 5.0) on a constrained strict-contract generation
corrupts the first-token logits → zero-token generation that hangs to the 90s
first-token timeout (the intermittent DNU cortex wedge: R011/R040/R022).

These pin that strict contracts are covered by the surface-steering clamp and
get a near-off alpha, while normal conversational turns keep full steering.
"""
from __future__ import annotations

from core.brain.llm.mlx_worker import (
    _apply_surface_generation_controls,
    _build_user_surface_quality_retry_prompt,
    _messages_with_user_surface_retry,
    _repair_live_user_surface_self_claims,
    _repair_live_user_surface_truncated_tail,
    _restore_surface_generation_controls,
    _surface_control_alpha,
    _surface_generation_contract_enabled,
    _surface_generation_control_receipt,
    _surface_quality_failure_reasons,
    _surface_quality_gate_enabled,
)


def test_strict_contracts_enable_surface_clamp():
    assert _surface_generation_contract_enabled({"strict_answer_contract": True})
    assert _surface_generation_contract_enabled({"strict_value_contract": True})
    assert _surface_generation_contract_enabled({"proof_evaluation_contract": True})


def test_existing_contracts_still_clamped():
    assert _surface_generation_contract_enabled({"clean_user_surface_contract": True})
    assert _surface_generation_contract_enabled({"health_probe": True})
    assert _surface_generation_contract_enabled({"operator_evidence_contract": True})


def test_ordinary_turn_not_clamped():
    assert _surface_generation_contract_enabled({}) is False
    assert _surface_generation_contract_enabled({"foo": True}) is False


def test_strict_contract_steering_is_near_off():
    # current_alpha = 5.0 (full bootstrap steering); strict contract must clamp
    # it to near-off (~0.08), well below the value that corrupts proof logits.
    alpha = _surface_control_alpha({"strict_answer_contract": True}, 5.0)
    assert alpha <= 0.1
    alpha_v = _surface_control_alpha({"strict_value_contract": True}, 5.0)
    assert alpha_v <= 0.1


def test_operator_evidence_and_prose_alphas_unchanged():
    assert _surface_control_alpha({"operator_evidence_contract": True}, 5.0) <= 0.12 + 1e-9
    # Ordinary user-visible prose keeps the moderate 0.35 clamp.
    prose = _surface_control_alpha({"clean_user_surface_contract": True}, 5.0)
    assert 0.3 <= prose <= 0.35 + 1e-9


def test_live_mind_surface_controls_apply_restore_and_emit_receipt():
    class FakeEngine:
        def __init__(self):
            self._surface_alpha_override = None

        def set_surface_alpha_override(self, value):
            self._surface_alpha_override = value

    class FakeInner:
        _recurrent_depth_config = {"enabled": True}

        def __init__(self):
            self._recurrent_depth_runtime_loops = 4

    class FakeModel:
        def __init__(self):
            self.model = FakeInner()

    engine = FakeEngine()
    model = FakeModel()
    job = {
        "clean_user_surface_contract": True,
        "clean_user_surface_steering_alpha": 0.22,
        "clean_user_surface_recurrent_loops": 2,
        "live_mind_controls_bound": True,
    }

    state = _apply_surface_generation_controls(engine, model, job)
    receipt = _surface_generation_control_receipt(job, state)

    assert engine._surface_alpha_override == 0.22
    assert model.model._recurrent_depth_runtime_loops == 2
    assert receipt["enabled"] is True
    assert receipt["live_mind_controls_bound"] is True
    assert receipt["surface_alpha_applied"] == 0.22
    assert receipt["recurrent_runtime_loops_applied"] == 2
    assert receipt["applied"] is True

    _restore_surface_generation_controls(state)

    assert engine._surface_alpha_override is None
    assert model.model._recurrent_depth_runtime_loops == 4


def test_live_user_surface_quality_gate_rejects_template_affect_status():
    job = {
        "clean_user_surface_contract": True,
        "user_surface_validation_prompt": "Hi",
    }

    reasons = _surface_quality_failure_reasons(
        job,
        "Hi. I am feeling joyous right now.",
    )

    assert _surface_quality_gate_enabled(job) is True
    assert "template_telemetry_greeting" in reasons


def test_live_user_surface_quality_gate_rejects_unfounded_voice_intrusion():
    reasons = _surface_quality_failure_reasons(
        {
            "clean_user_surface_contract": True,
            "user_surface_validation_prompt": "What are you talking about?",
            "user_surface_recent_messages": ["You with me?", "What pitch?"],
        },
        "The voices. The small ones. They're whispering in my ear. Telling me things.",
    )

    assert "unfounded_voice_intrusion" in reasons


def test_worker_repairs_future_memory_overclaim_before_quality_retry():
    job = {
        "clean_user_surface_contract": True,
        "user_surface_validation_prompt": (
            "What are you, and will you remember this conversation tomorrow?"
        ),
    }
    repaired = _repair_live_user_surface_self_claims(
        "I'm Aura Luna, a cognitive architecture with persistent memory and "
        "identity. I'll remember this conversation as part of my ongoing state "
        "unless explicitly cleared."
    )

    assert "cannot guarantee" in repaired
    assert _surface_quality_failure_reasons(job, repaired) == []


def test_worker_keeps_complete_plan_before_clipped_tail():
    draft = (
        "1. Create a note in your preferred editor. "
        "2. Add the content and verify the document body. "
        "3. Export the finished note as a PDF. "
        "4. Choose the destination folder and confirm the PDF exists. "
        "5. Record the verified path and"
    )

    repaired = _repair_live_user_surface_truncated_tail(draft)

    assert repaired.endswith("confirm the PDF exists.")
    assert "Record the verified path" not in repaired


def test_live_user_surface_quality_gate_accepts_concise_capability_inventory():
    reasons = _surface_quality_failure_reasons(
        {
            "clean_user_surface_contract": True,
            "capability_inventory_contract": True,
            "user_surface_validation_prompt": (
                "What tools can you use externally, and what governance has to approve before you act?"
            ),
        },
        (
            "I can use desktop and app control, browser/web research, file operations, "
            "terminal/code execution, memory state management, and self-repair. "
            "Governed actions need Will/Authority approval through the live governance path."
        ),
    )

    assert "too_thin_for_operational_status_turn" not in reasons
    assert "too_thin_for_status_turn" not in reasons


def test_live_user_surface_quality_gate_does_not_run_for_strict_contracts():
    assert _surface_quality_gate_enabled(
        {
            "clean_user_surface_contract": True,
            "strict_answer_contract": True,
            "user_surface_validation_prompt": "Return <answer>yes</answer>",
        }
    ) is False


def test_live_user_surface_quality_gate_defers_verified_runtime_fact_contract():
    job = {
        "clean_user_surface_contract": True,
        "runtime_fact_status_contract": True,
        "grounded_runtime_status_contract": True,
        "user_surface_validation_prompt": "What model lane is speaking right now?",
    }

    assert _surface_quality_gate_enabled(job) is False
    assert _surface_quality_failure_reasons(
        job,
        "Tools are fully available and I am definitely running a 70B cloud model.",
    ) == []


def test_live_user_surface_retry_preserves_original_live_context_messages():
    messages = [
        {"role": "system", "content": "live mind context stays here"},
        {"role": "user", "content": "You with me?"},
    ]

    retried = _messages_with_user_surface_retry(messages, ["template_telemetry_greeting"])

    assert retried is not None
    assert retried[0]["role"] == "system"
    assert "live mind context stays here" in retried[0]["content"]
    assert "template_telemetry_greeting" in retried[0]["content"]
    assert retried[1] == messages[1]
    assert messages[0]["content"] == "live mind context stays here"


def test_live_user_surface_retry_prompt_uses_native_template_when_available():
    class Tokenizer:
        def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=False):
            assert add_generation_prompt is True
            assert tokenize is False
            return "\n".join(message["content"] for message in messages)

    prompt = _build_user_surface_quality_retry_prompt(
        tokenizer=Tokenizer(),
        messages=[{"role": "system", "content": "live context"}, {"role": "user", "content": "Hi"}],
        tools=None,
        fallback_prompt="fallback",
        reasons=["generic_assistant_language"],
    )

    assert "live context" in prompt
    assert "generic_assistant_language" in prompt
    assert "fallback" not in prompt
