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


def test_coordinated_imperatives_without_question_mark_are_multipart() -> None:
    shape = analyze_prompt_shape(
        "Compare optimistic and pessimistic locking for a hot task queue, choose "
        "which one you would use in a single-host async runtime, explain why, and "
        "verify your choice with one concrete failure scenario."
    )

    assert shape.question_parts == 4
    assert shape.imperative_parts == 4
    assert shape.prefers_extended_answer is True
    assert shape.requires_single_reply_coverage is True
    assert shape.to_dict()["imperative_parts"] == 4


def test_coordinated_nouns_do_not_create_fake_imperative_parts() -> None:
    shape = analyze_prompt_shape("Compare optimistic and pessimistic locking.")

    assert shape.question_parts == 1
    assert shape.imperative_parts == 1


def test_inline_parenthesized_obligations_are_first_class_request_parts() -> None:
    shape = analyze_prompt_shape(
        "Explain Dijkstra's algorithm in one complete response. Include: "
        "(1) the core invariant, (2) numbered pseudocode, (3) a worked example "
        "with five weighted edges, (4) binary-heap and array complexity, and "
        "(5) a negative-weight failure and the correct alternative."
    )

    assert shape.numbered_parts == 5
    assert shape.question_parts == 6
    assert shape.prefers_extended_answer is True
    assert shape.requires_single_reply_coverage is True
    assert shape.question_segments == (
        "Explain Dijkstra's algorithm in one complete response.",
        "the core invariant",
        "numbered pseudocode",
        "a worked example with five weighted edges",
        "binary-heap and array complexity, and",
        "a negative-weight failure and the correct alternative",
    )


def test_parenthesized_numbers_without_a_contiguous_list_remain_one_part() -> None:
    for prompt in (
        "Explain why f(2) is larger than f(1).",
        "Summarize finding (3) from the supplied report.",
        "Compare option (1) with option (3).",
    ):
        shape = analyze_prompt_shape(prompt)
        assert shape.numbered_parts == 0
        assert shape.question_parts == 1
