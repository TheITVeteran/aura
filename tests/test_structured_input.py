from core.runtime.structured_input import analyze_prompt_shape


def test_compound_compare_choose_explain_prompt_is_multipart() -> None:
    shape = analyze_prompt_shape(
        "Why can duplicate generation corrupt answer quality? Compare an early "
        "single-owner design with late deduplication, then choose the stronger "
        "architecture and explain how to verify it under cancellation faults."
    )

    assert shape.question_parts >= 2
    assert shape.connector_parts >= 1
    assert shape.prefers_extended_answer is True
    assert shape.requires_single_reply_coverage is True


def test_single_explanation_request_remains_single_part() -> None:
    shape = analyze_prompt_shape("Explain why checksums matter for artifact integrity.")

    assert shape.question_parts == 1
    assert shape.prefers_extended_answer is False
    assert shape.requires_single_reply_coverage is False
