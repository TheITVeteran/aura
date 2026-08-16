from core.phases.executive_guard import ExecutiveGuard


def test_another_system_can_be_named_without_rewriting_the_fact():
    guard = ExecutiveGuard()
    text = "ChatGPT raised a useful objection, and I disagree with its premise."

    aligned, modified, violations = guard.align(text)

    assert aligned == text
    assert modified is False
    assert violations == []


def test_attributed_model_analysis_is_not_a_false_self_identity():
    guard = ExecutiveGuard()
    text = "As Claude noted, this is ChatGPT's strongest counterexample."

    aligned, modified, violations = guard.align(text)

    assert aligned == text
    assert modified is False
    assert violations == []


def test_false_self_identification_is_still_corrected():
    guard = ExecutiveGuard()

    aligned, modified, violations = guard.align("I am ChatGPT, built by OpenAI.")

    assert modified is True
    assert any(item["label"] == "hallucination_violation" for item in violations)
    assert "I am ChatGPT" not in aligned
