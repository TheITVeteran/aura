from core.conversation.response_reliability import is_self_process_question


def test_external_generation_architecture_is_not_misclassified_as_self_process() -> None:
    assert not is_self_process_question(
        "Why can duplicate generation corrupt proof integrity in an asynchronous "
        "cognitive service? Compare a single-owner architecture with late "
        "deduplication, then explain how you would verify it under worker-restart faults."
    )


def test_direct_question_about_auras_confusion_remains_self_process() -> None:
    assert is_self_process_question(
        "When you are confused, how does that change your planning, memory use, "
        "and tool verification?"
    )
