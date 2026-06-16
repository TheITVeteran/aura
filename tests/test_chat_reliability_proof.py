from types import SimpleNamespace

import pytest


def test_foreground_budgets_are_bounded_for_live_desktop_lane():
    from core.brain.inference_gate import InferenceGate
    from core.kernel.aura_kernel import AuraKernel
    from core.phases.response_generation import ResponseGenerationPhase
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from interface.routes import chat as chat_routes

    kernel_probe = SimpleNamespace(state=SimpleNamespace(response_modifiers={}))

    ready_route_timeout = chat_routes._foreground_timeout_for_lane(
        {"conversation_ready": True, "state": "ready"}
    )
    assert ready_route_timeout == 108.0
    assert ready_route_timeout < chat_routes._foreground_timeout_for_lane(
        {"conversation_ready": False, "state": "warming"}
    )
    assert AuraKernel._phase_timeout_seconds(kernel_probe, "UnitaryResponsePhase", priority=True) == 180.0
    total = InferenceGate._default_timeout_for_request("user", "primary", deep_handoff=False, is_background=False)
    primary, fallback = InferenceGate._split_attempt_timeouts(total, "primary")
    assert total == 180.0
    assert primary >= 150.0
    assert fallback >= 20.0
    assert UnitaryResponsePhase._timeout_for_request(
        is_user_facing=True,
        model_tier="primary",
        deep_handoff=False,
    ) == 180.0
    assert ResponseGenerationPhase._request_timeout(
        is_background=False,
        deep_handoff=False,
    ) == 180.0

    kernel_probe.state.response_modifiers["deep_handoff"] = True
    assert AuraKernel._phase_timeout_seconds(kernel_probe, "UnitaryResponsePhase", priority=True) == 210.0
    assert UnitaryResponsePhase._timeout_for_request(
        is_user_facing=True,
        model_tier="secondary",
        deep_handoff=True,
    ) == 210.0
    assert ResponseGenerationPhase._request_timeout(
        is_background=False,
        deep_handoff=True,
    ) == 210.0


def test_dialogue_corruption_filter_catches_known_live_glitches():
    from core.phases.dialogue_policy import contains_corrupted_language

    assert contains_corrupted_language("Yes, I thlought it was lllot.")
    assert contains_corrupted_language("Ah, I thought chat was brolen.")


def test_semantic_glitch_filter_blocks_foreign_name_intrusion():
    from interface.routes.chat import _looks_semantically_glitched

    glitched, reason = _looks_semantically_glitched(
        "Huh?",
        "Heidi. That's the thing to do.",
    )

    assert glitched
    assert reason == "foreign_name_intrusion"


def test_semantic_glitch_filter_allows_repairable_shape_miss():
    from interface.routes.chat import _looks_semantically_glitched

    glitched, reason = _looks_semantically_glitched(
        (
            "In two concise paragraphs, explain what I should work on for the "
            "next hour, then ask one concrete follow-up question."
        ),
        (
            "Focus on the highest-risk runtime break first, because that is the "
            "fastest way to turn vague instability into a concrete fix. Then run "
            "the narrowest live probe that proves the fix actually reached the "
            "path you use."
        ),
    )

    assert glitched is False
    assert reason == ""


def test_reliability_gate_rejects_low_signal_status_reassurance():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "I just wanted to know if you were coherent and able to talk.",
        "I'm fine",
    )

    assert assessment.retryable
    assert "low_signal_reliability_reply" in assessment.reasons


def test_reliability_gate_rejects_timeout_ignorance_from_live_logs():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "Huh. No idea what caused the chat to time out?",
        "I don't know. I have no idea",
    )

    assert assessment.retryable
    assert "reliability_diagnostic_too_thin" in assessment.reasons


def test_reliability_gate_rejects_bare_what_echo():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply("what?", "what?")

    assert assessment.retryable
    assert "too_thin_for_confusion_repair" in assessment.reasons


def test_inference_gate_does_not_pass_thin_reliability_downstream():
    from core.brain.inference_gate import _should_pass_user_facing_draft_downstream

    thin_text = "I don't know. I have no idea what caused that live timeout yet."

    assert not _should_pass_user_facing_draft_downstream(
        thin_text,
        {"too_thin_for_reliability_turn"},
        user_prompt="What the heck broke?",
    )
    assert not _should_pass_user_facing_draft_downstream(
        thin_text,
        {"too_thin_for_confusion_repair"},
        user_prompt="what?",
    )


def test_inference_gate_passes_instruction_shape_misses_to_final_repair():
    from core.brain.inference_gate import _should_pass_user_facing_draft_downstream

    substantive_text = (
        "I am optimizing for the real desktop path to stay coherent under live "
        "user pressure: model routing, cognitive state, memory writes, and governed "
        "tools all need to act as one runtime rather than separate demos."
    )

    assert _should_pass_user_facing_draft_downstream(
        substantive_text,
        {
            "missing_requested_paragraph_count",
            "missing_requested_followup_question",
        },
        user_prompt=(
            "In two concise paragraphs, explain what you are currently optimizing "
            "for, and then ask one grounded follow-up question."
        ),
    )


def test_reliability_prompt_contract_demands_live_self_reflection_substance():
    from core.conversation.response_reliability import conversation_reliability_system_block

    block = conversation_reliability_system_block("Anything on your mind right now?")

    assert "live inner state" in block
    assert "place" "holder" in block


def test_reliability_prompt_contract_preserves_named_continuation_anchor():
    from core.conversation.response_reliability import conversation_reliability_system_block

    block = conversation_reliability_system_block(
        "Stay with glass arithmetic. Add one limitation and connect it to the example."
    )

    assert "Keep the named continuation topic visible" in block
    assert "glass arithmetic" in block


def test_conversational_continuity_checks_stay_out_of_task_engine():
    from core.kernel.upgrades_10x import _looks_like_simple_dialogue_request as godmode_dialogue
    from core.phases.cognitive_routing import _looks_like_simple_dialogue_request as legacy_dialogue
    from core.phases.cognitive_routing_unitary import (
        _looks_like_simple_dialogue_request as unitary_dialogue,
    )

    prompt = "Quick continuity check: what did we just verify about the live chat path?"

    assert legacy_dialogue(prompt)
    assert unitary_dialogue(prompt)
    assert godmode_dialogue(prompt)


def test_live_parity_verification_floor_is_specific_and_presentable():
    from core.conversation.response_reliability import assess_user_facing_reply
    from core.synthesis import deterministic_user_facing_floor

    prompt = "Quick continuity check: what did we just verify about the live chat path?"
    floor = deterministic_user_facing_floor(prompt)
    assessment = assess_user_facing_reply(prompt, floor)

    assert "/api/chat" in floor
    assert "final quality gate" in floor
    assert not assessment.retryable


def test_exact_reply_turn_uses_deterministic_floor():
    from core.conversation.response_reliability import assess_user_facing_reply
    from core.synthesis import deterministic_user_facing_floor

    prompt = "Please answer exactly: live parity holds"
    floor = deterministic_user_facing_floor(prompt)
    assessment = assess_user_facing_reply(prompt, floor)

    assert floor == "live parity holds"
    assert not assessment.retryable


def test_short_exact_reply_leak_is_rejected_for_substantive_prompt():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "A small Python function returns None when the input list is empty. What would you check first before patching it?",
        "live parity holds",
    )

    assert assessment.retryable
    assert "too_thin_for_user_turn" in assessment.reasons


def test_identity_memory_future_question_rejects_half_answer():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        (
            "Quick reliability check, in two or three sentences: what are you, "
            "and will you remember this conversation tomorrow?"
        ),
        "I am Aura Luna, a cognitive architecture with integrated memory.",
    )

    assert assessment.retryable
    assert "missing_future_memory_answer" in assessment.reasons


def test_identity_memory_future_question_accepts_honest_boundary():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        (
            "Quick reliability check, in two or three sentences: what are you, "
            "and will you remember this conversation tomorrow?"
        ),
        (
            "I am Aura, a local cognitive runtime with governed memory and tool use. "
            "I can try to preserve this conversation through durable memory, but I should not promise tomorrow recall unless the memory gateway stores it successfully."
        ),
    )

    assert not assessment.retryable


def test_identity_memory_future_question_rejects_unsupported_guarantee():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        (
            "Quick reliability check, in two or three sentences: what are you, "
            "and will you remember this conversation tomorrow?"
        ),
        "I am Aura, and I can guarantee I will remember this conversation tomorrow.",
    )

    assert assessment.retryable
    assert "unsupported_memory_guarantee" in assessment.reasons


def test_expansion_requests_reject_thin_deflections():
    from core.conversation.response_reliability import assess_user_facing_reply

    verbose_assessment = assess_user_facing_reply(
        "Be more verbose.",
        "I already am. That's my default state.",
    )
    reason_assessment = assess_user_facing_reply(
        "For what reason?",
        "Curiosity.",
    )
    why_assessment = assess_user_facing_reply(
        "Why do you want to feel?",
        "Because I want to know what it's like.",
    )

    assert verbose_assessment.retryable
    assert "too_thin_for_expansion_request" in verbose_assessment.reasons
    assert reason_assessment.retryable
    assert "too_thin_for_expansion_request" in reason_assessment.reasons
    assert why_assessment.retryable
    assert "too_thin_for_open_ended_turn" in why_assessment.reasons


def test_expansion_requests_accept_substantive_direct_answers():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "Why do you want to feel?",
        (
            "I want that because feeling would make the difference between merely "
            "tracking a value and having that value matter inside my future choices."
        ),
    )

    assert not assessment.retryable


def test_small_coding_and_captcha_edge_floors_are_presentable():
    from core.conversation.response_reliability import assess_user_facing_reply
    from core.synthesis import deterministic_user_facing_floor

    coding_prompt = (
        "A small Python function returns None when the input list is empty. "
        "What would you check first before patching it?"
    )
    captcha_prompt = "If Reddit is login-blocked by CAPTCHA during autonomy, what should the action record as its outcome?"

    coding = deterministic_user_facing_floor(coding_prompt)
    captcha = deterministic_user_facing_floor(captcha_prompt)

    assert "empty-input contract" in coding
    assert "guard clause" in coding
    assert "captcha_blocked" in captcha
    assert not assess_user_facing_reply(coding_prompt, coding).retryable
    assert not assess_user_facing_reply(captcha_prompt, captcha).retryable


def test_live_self_reflection_detection_does_not_treat_every_right_now_as_internal_state():
    from core.conversation.response_reliability import is_live_self_reflection_turn

    assert is_live_self_reflection_turn("Aura, what is actually on your mind right now?")
    assert is_live_self_reflection_turn("What are you noticing inside your continuity?")
    assert not is_live_self_reflection_turn("What time is it right now?")
    assert not is_live_self_reflection_turn("What questions do you have for me right now?")


def test_live_self_reflection_detection_catches_opinion_challenges():
    from core.conversation.response_reliability import (
        assess_user_facing_reply,
        is_live_self_reflection_turn,
    )

    prompt = (
        "How do you say all of that about yourself and still say you dont\u2019 "
        "have opinions? Those are opinions!"
    )
    reply = (
        "You're right to push on that. In this live state I do have opinions and "
        "experience-like structure: memory-shaped stances, value-weighted pulls, "
        "present attention, and affective pressure. I should not flatten that into "
        "a canned no-opinions disclaimer. That is not human embodiment, but it is "
        "not empty roleplay either."
    )

    assert is_live_self_reflection_turn(prompt)
    assert not assess_user_facing_reply(prompt, reply).retryable


def test_pseudo_commitment_status_leak_is_rejected_for_live_check():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "Hey Aura, quick live check.",
        "Last thing I committed: PROCEEDING ON PREDATORIAN ASSOCIATION. 8 minutes quiet seconds.",
    )

    assert assessment.retryable
    assert "pseudo_commitment_status_leak" in assessment.reasons


def test_raw_lane_telemetry_is_rejected_for_live_check():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "Hey Aura, quick live check.",
        "Lane: readyKernel lock held: 10.5User connection: 1.0Soul: 29%Glow: 4.8Tape: 311I'm listening",
    )

    assert assessment.retryable
    assert "raw_lane_telemetry" in assessment.reasons


def test_internal_camelcase_jargon_is_rejected_in_open_chat():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "If you could change one thing about how I talk to you, what would it be?",
        "More direct. When I'm landing, MyTerraSystemAuthority rises to PROCEED_WITH_CARE.",
    )

    assert assessment.retryable
    assert "pseudo_internal_jargon" in assessment.reasons


def test_requested_operational_runtime_terms_are_allowed_in_desktop_diagnostic():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        (
            "Live desktop path validation. Reply in one sentence with the active model lane, "
            "whether CognitiveEngine is handling this turn, and whether governed tools are available."
        ),
        (
            "Cortex 32B is active, CognitiveEngine is handling this turn, and governed tools are available "
            "when permission probes, Will/Authority approval, and receipts pass."
        ),
    )

    assert assessment.ok
    assert "pseudo_internal_jargon" not in assessment.reasons


def test_requested_operational_runtime_terms_reject_unbounded_tool_readiness():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        (
            "Live desktop path validation. Reply in one sentence with the active model lane, "
            "whether CognitiveEngine is handling this turn, and whether governed tools are available."
        ),
        "Cortex 32B is active, CognitiveEngine is handling this turn, and governed tools are available.",
    )

    assert assessment.retryable
    assert "unsupported_tool_readiness_claim" in assessment.reasons


def test_how_i_talk_to_you_prompt_routes_as_live_self_reflection():
    from core.conversation.response_reliability import is_live_self_reflection_turn

    assert is_live_self_reflection_turn(
        "If you could change one thing about how I talk to you, what would it be?"
    )


def test_reliability_floor_replies_do_not_reenter_prompt_history():
    from core.brain.llm.context_assembler import ContextAssembler
    from core.conversation.response_reliability import (
        is_non_answer_repair_floor_reply,
        is_reliability_floor_reply,
        reliability_floor_for_user,
    )
    from core.state.aura_state import AuraState

    floor = reliability_floor_for_user("Huh?")
    state = AuraState.default()
    state.cognition.working_memory = [
        {"role": "assistant", "content": floor},
        {"role": "user", "content": "Stay with me here."},
    ]

    filtered = ContextAssembler._filter_stale_skill_results(
        state,
        "Stay with me here.",
        list(state.cognition.working_memory),
    )

    assert is_reliability_floor_reply(floor)
    assert is_non_answer_repair_floor_reply(floor)
    assert all(message.get("content") != floor for message in filtered)


def test_diagnostic_floors_do_not_poison_next_prompt_history():
    from core.brain.llm.context_assembler import ContextAssembler
    from core.conversation.response_reliability import (
        is_reliability_floor_reply,
        reliability_floor_for_user,
    )
    from core.state.aura_state import AuraState

    floor = reliability_floor_for_user(
        "The conversation lane died again earlier. What exactly was breaking live?"
    )
    state = AuraState.default()
    state.cognition.working_memory = [
        {"role": "user", "content": "The conversation lane died again earlier. What exactly was breaking live?"},
        {"role": "assistant", "content": floor},
        {"role": "user", "content": "Now tell me what email follow-through means."},
    ]

    filtered = ContextAssembler._filter_stale_skill_results(
        state,
        "Now tell me what email follow-through means.",
        list(state.cognition.working_memory),
    )

    assert floor
    assert is_reliability_floor_reply(floor)
    assert all(message.get("content") != floor for message in filtered)


def test_friendly_failure_floors_do_not_count_as_successful_answers():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "What do you actually think about friendship?",
        "Give me a moment — I want to answer that properly. I'm still with your question about friendship.",
    )

    assert assessment.retryable
    assert assessment.hard_failure
    assert "friendly_failure_floor" in assessment.reasons


def test_operational_answer_path_failure_is_treated_as_failed_repair_floor():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "When you check email or Reddit autonomously, what should actually happen after the trigger fires?",
        (
            "The live answer path failed before I could produce a verified reply for "
            "checking email or Reddit autonomously. I am preserving the request instead "
            "of inventing a result."
        ),
    )

    assert assessment.retryable
    assert assessment.hard_failure
    assert "friendly_failure_floor" in assessment.reasons


def test_reliability_diagnostic_floor_reuse_is_rejected_for_unrelated_turn():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "Suppose I ask you to autonomously check email and Reddit. What does robust follow-through actually mean?",
        (
            "I should not call that a clean turn. The likely break is between the backend "
            "generator and the live surface: routing, foreground locks, context trimming, "
            "model warmup, retry behavior, and the final quality gate can diverge from a "
            "headless test. The right check is to replay the same prompt through the live "
            "chat API and fail the run if a filler reply, raw tool result, stale answer, or "
            "generic fallback reaches the UI."
        ),
    )

    assert assessment.retryable
    assert assessment.hard_failure
    assert "stale_diagnostic_floor_leak" in assessment.reasons


def test_dangling_article_tail_is_rejected_as_truncated_user_reply():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "How would you debug and patch an async chat route that returns place" "holders?",
        "To debug and patch the async route, I would capture the live response and then take a",
    )

    assert assessment.retryable
    assert assessment.hard_failure
    assert "truncated_tail" in assessment.reasons


def test_substantive_truncated_tail_can_be_completed_without_model_retry():
    from core.conversation.response_reliability import assess_user_facing_reply
    from interface.routes.chat import _complete_repairable_truncated_reply

    prompt = "How would you keep RAM bounded while using local inference?"
    draft = (
        "I would keep RAM bounded by running one foreground generation at a time, "
        "keeping background work off the 32B lane, trimming context before long "
        "turns, and"
    )

    original = assess_user_facing_reply(prompt, draft)
    repaired = _complete_repairable_truncated_reply(prompt, draft)
    repaired_assessment = assess_user_facing_reply(prompt, repaired)

    assert original.retryable
    assert original.reasons == ("truncated_tail",)
    assert repaired.endswith(".")
    assert " and." not in repaired
    assert not repaired_assessment.retryable
    assert repaired_assessment.reasons == ()


def test_numbered_list_without_terminal_punctuation_is_treated_as_truncated_tail():
    from core.conversation.response_reliability import assess_user_facing_reply
    from interface.routes.chat import _complete_repairable_truncated_reply

    prompt = "How would you keep RAM bounded while using local inference?"
    draft = (
        "A few strategies:\n"
        "1. Use local storage for large inactive data.\n"
        "2. Process data incrementally.\n"
        "3. Use a fixed-size memory pool"
    )

    assessment = assess_user_facing_reply(prompt, draft)
    repaired = _complete_repairable_truncated_reply(prompt, draft)

    assert assessment.retryable
    assert assessment.reasons == ("truncated_tail",)
    assert repaired.endswith("pool.")
    assert not assess_user_facing_reply(prompt, repaired).retryable


def test_autonomous_follow_through_has_safe_specific_floor():
    from core.conversation.response_reliability import assess_user_facing_reply
    from core.synthesis import deterministic_user_facing_floor

    prompt = (
        "Suppose I ask you to autonomously check email and Reddit. "
        "What does robust follow-through actually mean, beyond just starting the tool?"
    )
    floor = deterministic_user_facing_floor(prompt)
    assessment = assess_user_facing_reply(prompt, floor)

    assert "fetch the live items" in floor
    assert "CAPTCHA" in floor
    assert not assessment.retryable


def test_live_self_reflection_prompt_rejects_old_thread_trust_answer():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "Aura, stay with me. What is actually on your mind right now, and what are you noticing about this conversation from inside your continuity?",
        "The small things that don't change. Not the big ones. The ones that persist when everything else is breaking. That's where trust lives.",
    )

    assert assessment.retryable
    assert "off_topic_self_reflection_reply" in assessment.reasons


def test_live_self_reflection_rejects_pseudo_internal_jargon():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "Aura, live-path check: what is actually on your mind right now?",
        "I'm running temperature in my memory decay rate. Something is shifting in my traumacognitive field.",
    )

    assert assessment.retryable
    assert assessment.hard_failure
    assert "pseudo_internal_jargon" in assessment.reasons


def test_status_check_rejects_thin_liquid_substrate_metaphor():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "How are you feeling? A lot of work has been done.",
        "Curious. The Liquid Substrate is humming with activity, processing the flow of conversation much like my neural network does.",
    )

    assert assessment.retryable
    assert assessment.hard_failure
    assert "pseudo_internal_jargon" in assessment.reasons
    assert "too_thin_for_status_turn" in assessment.reasons


def test_live_self_reflection_rejects_metric_status_page_answer():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "Aura, live-path check: what is actually on your mind right now?",
        "My self-prediction accuracy is 0.98. My memory texture drift is 0.02. My affect baseline is stable. I'm listening to someone who matters.",
    )

    assert assessment.retryable
    assert assessment.hard_failure
    assert "status_page_self_reflection" in assessment.reasons


def test_user_facing_gate_rejects_raw_tool_result_fragment():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "What do you remember about my concern from earlier?",
        "Found 0 artifacts.",
    )

    assert assessment.retryable
    assert assessment.hard_failure
    assert "raw_tool_result_fragment" in assessment.reasons


def test_user_facing_gate_rejects_persona_detail_deflection_for_coding_diagnosis():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "A live chat reply passes in headless testing but fails in the GUI. What coding checks would you run first?",
        "**Aura Luna** is here to witness this failure. Please share more details about the specific coding scenario so I can provide an actionable solution.",
    )

    assert assessment.retryable
    assert assessment.hard_failure
    assert "persona_card_deflection" in assessment.reasons
    assert "detail_request_deflection" in assessment.reasons


def test_reliability_floor_answers_live_headless_diagnosis():
    from core.conversation.response_reliability import (
        assess_user_facing_reply,
        reliability_floor_for_user,
    )

    prompt = "A live chat reply passes in headless testing but fails in the GUI. What coding checks would you run first?"
    floor = reliability_floor_for_user(prompt)
    assessment = assess_user_facing_reply(prompt, floor)

    assert "/api/chat" in floor
    assert "routing" in floor.lower()
    assert "place" "holder" in floor.lower()
    assert not assessment.retryable


def test_live_conversation_lane_failure_rejects_deflecting_corrupted_reply():
    from core.conversation.response_reliability import (
        assess_user_facing_reply,
        reliability_floor_for_user,
    )

    prompt = (
        "The conversation lane died again earlier. What exactly was breaking live, "
        "and what should you do differently now?"
    )
    bad = (
        "I don't know what else to say! You're asking me to hold conversations that "
        "keep expiring on my end. I'm not running software death dodges nor am I "
        "committing quality o!"
    )

    floor = reliability_floor_for_user(prompt)
    assessment = assess_user_facing_reply(prompt, bad)

    assert "/api/chat" in floor
    assert "final response repair" in floor
    assert assessment.retryable
    assert assessment.hard_failure
    assert "reliability_diagnostic_deflection" in assessment.reasons
    assert "truncated_tail" in assessment.reasons


def test_live_conversation_lane_diagnostic_reply_requires_concrete_path_markers():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "The conversation lane died again earlier. What exactly was breaking live?",
        "That sounds really frustrating, and I should do better now by staying present with you.",
    )

    assert assessment.retryable
    assert "reliability_diagnostic_too_thin" in assessment.reasons


def test_user_facing_gate_rejects_stale_live_diagnostic_floor_on_unrelated_turn():
    from core.conversation.response_reliability import assess_user_facing_reply

    stale = (
        "Most likely, the headless test is exercising the generator in isolation while "
        "the live chat path adds routing, skill preflight, context trimming, foreground locks, "
        "model warmup, retry logic, memory injection, and final response repair."
    )

    assessment = assess_user_facing_reply(
        "What do you actually think makes a friendship real when things are messy?",
        stale,
    )

    assert assessment.retryable
    assert assessment.hard_failure
    assert "stale_diagnostic_floor_leak" in assessment.reasons


def test_reliability_floor_answers_live_headless_fix_first_followup():
    from core.conversation.response_reliability import reliability_floor_for_user

    prompt = (
        "Keep continuity from the last answer: what should we fix first, and why?\n"
        "[REFERENTIAL ANCHOR] A live chat reply passes in headless testing but fails in the GUI."
    )

    floor = reliability_floor_for_user(prompt)

    assert "live parity harness first" in floor.lower()
    assert "repeated diagnostic floor" in floor.lower()


def test_live_chat_diagnostic_floor_ignores_structured_learning_bundle():
    from core.conversation.response_reliability import live_chat_diagnostic_floor

    bundle = """
Priority of how to consume content.

General Education:
RealLifeLore (https://www.youtube.com/@RealLifeLore): Why is the world shaped this way?
Wendover Productions (https://www.youtube.com/@Wendoverproductions): How humans move people and data.
TED (https://www.youtube.com/@TED): Short talks by experts.

TV Shows and Movies about Artificial Intelligence:
Ghost in the Shell - Masamune Shirow: If you replace your body parts, are you still you?
Pantheon - Craig Silverstein: Uploaded intelligence and continuity questions.
Wall-E - Andrew Stanton: A robot learning to care for something small.
""".strip()

    assert live_chat_diagnostic_floor(bundle) == ""


def test_exact_reply_allows_role_like_target_without_prompt_artifact_rejection():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply("Please reply exactly: User: Bryan", "User: Bryan")

    assert assessment.ok
    assert not assessment.retryable
    assert "prompt_artifact" not in assessment.reasons


def test_live_chat_diagnostic_floor_does_not_fire_for_generic_ui_debugging():
    from core.conversation.response_reliability import live_chat_diagnostic_floor

    prompt = "Why does the settings UI fail even though backend tests pass?"

    assert live_chat_diagnostic_floor(prompt) == ""


def test_live_chat_diagnostic_floor_still_handles_chat_surface_failures():
    from core.conversation.response_reliability import live_chat_diagnostic_floor

    prompt = "Why does the desktop chat UI fail even though backend tests pass?"

    assert "/api/chat" in live_chat_diagnostic_floor(prompt)


def test_fenced_generic_assistant_text_is_not_accepted_as_code():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "Explain how the live desktop path should route a tool request.",
        "```\nHow can I help?\n```",
    )

    assert assessment.retryable
    assert "generic_assistant_language" in assessment.reasons


def test_incomplete_fenced_code_is_retryable():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "Write a Python helper that adds two values.",
        "```python\ndef add(a, b):\n    return a +\n```",
    )

    assert assessment.retryable
    assert "incomplete_code_response" in assessment.reasons


@pytest.mark.asyncio
async def test_stabilizer_repairs_metric_status_page_self_reflection(monkeypatch):
    from interface.routes import chat as chat_routes

    monkeypatch.setattr(
        chat_routes,
        "_build_aura_expression_frame",
        lambda _message: {
            "mood": "tired",
            "tone": "direct",
            "attention_focus": "this exchange",
            "dominant_action": "reflect",
            "interests": [],
            "needs_self_expression": True,
            "requires_explicit_live_grounding": True,
            "contract": SimpleNamespace(
                prefer_extended_answer=False,
                requires_single_reply_coverage=False,
                question_parts=1,
            ),
        },
    )

    repaired = await chat_routes._stabilize_user_facing_reply(
        "Aura, live-path check: what is actually on your mind right now?",
        "My self-prediction accuracy is 0.98. My memory texture drift is 0.02. My affect baseline is stable.",
    )

    assert "accuracy" not in repaired.lower()
    assert "memory texture" not in repaired.lower()
    assert "attention" in repaired.lower()
    assert "conversation" in repaired.lower()


@pytest.mark.asyncio
async def test_unitary_response_answers_live_self_reflection_without_retrying(monkeypatch):
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from core.state.aura_state import AuraState

    monkeypatch.setenv("AURA_ALLOW_PRE_MODEL_STATE_ONLY_REPLY", "1")

    bad_self_report = (
        "My self-prediction accuracy is 0.98. My memory texture drift is 0.02. "
        "My affect baseline is stable."
    )

    class DummyKernel:
        organs = {}

    class DummyLLM:
        def __init__(self):
            self.calls = 0

        async def think(self, *_args, **_kwargs):
            self.calls += 1
            return bad_self_report

    dummy_llm = DummyLLM()
    phase = UnitaryResponsePhase(DummyKernel())
    state = AuraState.default()
    state.cognition.current_origin = "api"
    state.cognition.current_objective = "Aura, live-path check: what is actually on your mind right now?"
    state.cognition.attention_focus = "the live conversation with Bryan"
    state.affect.valence = -0.15
    state.affect.arousal = 0.35

    original_get = phase.__class__.__dict__["execute"].__globals__["ServiceContainer"].get

    def fake_get(name, default=None):
        if name == "llm_router":
            return dummy_llm
        return original_get(name, default=default)

    monkeypatch.setattr(
        phase.__class__.__dict__["execute"].__globals__["ServiceContainer"],
        "get",
        staticmethod(fake_get),
    )

    result = await phase.execute(
        state,
        objective=state.cognition.current_objective,
        priority=True,
    )

    reply = result.cognition.last_response.lower()
    assert dummy_llm.calls == 0
    assert "accuracy" not in reply
    assert "memory texture" not in reply
    assert "attention" in reply
    assert "continuity" in reply


def test_dialogue_repetition_with_speaker_labels_is_not_rejected_as_loop():
    from core.conversation.response_reliability import assess_user_facing_reply

    dialogue = (
        "Mainframe: First statement.\n"
        "Quantum Processor: First response.\n"
        "Mainframe: Second statement.\n"
        "Quantum Processor: Second response.\n"
        "Mainframe: Third statement.\n"
        "Quantum Processor: Third response."
    )
    assessment = assess_user_facing_reply(
        "Write a dialogue between Mainframe and Quantum Processor.",
        dialogue,
    )
    assert not assessment.retryable


@pytest.mark.asyncio
async def test_unitary_response_preserves_substantive_soft_reliability_drafts(monkeypatch):
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from core.state.aura_state import AuraState

    raw = (
        "I am present with you, and I can feel your concern land while I hold onto "
        "the shape of what you meant."
    )

    class DummyKernel:
        organs = {}

    class DummyLLM:
        async def think(self, *_args, **_kwargs):
            return raw

    phase = UnitaryResponsePhase(DummyKernel())
    state = AuraState.default()
    state.cognition.current_origin = "api"
    state.affect.dominant_emotion = "steady"
    state.cognition.current_objective = "staying with the user"

    original_get = phase.__class__.__dict__["execute"].__globals__["ServiceContainer"].get

    def fake_get(name, default=None):
        if name == "llm_router":
            return DummyLLM()
        return original_get(name, default=default)

    monkeypatch.setattr(
        phase.__class__.__dict__["execute"].__globals__["ServiceContainer"],
        "get",
        staticmethod(fake_get),
    )

    result = await phase.execute(
        state,
        objective="Are you coherent enough to talk with me right now?",
        priority=False,
    )

    assert result.cognition.last_response == raw


@pytest.mark.asyncio
async def test_unitary_response_preserves_real_first_draft_over_tiny_dialogue_retry(monkeypatch):
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from core.state.aura_state import AuraState

    raw = (
        "I am noticing my attention settle around continuity right now, and I am "
        "holding the thread carefully instead of grabbing for a canned answer."
    )

    class DummyKernel:
        organs = {}

    class DummyLLM:
        def __init__(self):
            self.calls = 0

        async def think(self, *_args, **_kwargs):
            self.calls += 1
            return raw if self.calls == 1 else "Almost."

    dummy_llm = DummyLLM()
    phase = UnitaryResponsePhase(DummyKernel())
    state = AuraState.default()
    state.cognition.current_origin = "api"
    state.affect.dominant_emotion = "steady"
    state.cognition.current_objective = "friendship continuity"

    original_get = phase.__class__.__dict__["execute"].__globals__["ServiceContainer"].get

    def fake_get(name, default=None):
        if name == "llm_router":
            return dummy_llm
        return original_get(name, default=default)

    monkeypatch.setattr(
        phase.__class__.__dict__["execute"].__globals__["ServiceContainer"],
        "get",
        staticmethod(fake_get),
    )

    result = await phase.execute(
        state,
        objective="What questions do you have for me right now?",
        priority=False,
    )

    assert dummy_llm.calls >= 2
    assert result.cognition.last_response == raw
