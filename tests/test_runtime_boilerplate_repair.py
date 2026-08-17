from __future__ import annotations

import pytest

from core.conversation.response_reliability import (
    _model_text_integrity_reasons,
    repair_runtime_boilerplate,
)

pytestmark = pytest.mark.unit


def test_runtime_status_sentence_does_not_condemn_the_answer_around_it():
    answer = (
        "1. Set dist[source] = 0 and every other distance to infinity.\n"
        "2. Extract the unsettled vertex with minimum tentative distance.\n"
        "3. Relax each outgoing edge and update the heap.\n\n"
        "With a binary heap this costs O((V + E) log V). "
        "My deeper processing is taking longer; ask me again. "
        "Negative edges require Bellman-Ford instead."
    )

    assert "runtime_boilerplate" in _model_text_integrity_reasons(
        answer,
        user_facing=True,
    )

    repaired = repair_runtime_boilerplate(answer)

    assert repaired == (
        "1. Set dist[source] = 0 and every other distance to infinity.\n"
        "2. Extract the unsettled vertex with minimum tentative distance.\n"
        "3. Relax each outgoing edge and update the heap.\n\n"
        "With a binary heap this costs O((V + E) log V). "
        "Negative edges require Bellman-Ford instead."
    )
    assert "runtime_boilerplate" not in _model_text_integrity_reasons(
        repaired,
        user_facing=True,
    )


def test_runtime_sentence_repair_preserves_unrelated_markdown_exactly():
    answer = (
        "**Invariant**\n\n"
        "```text\n"
        "while heap:\n"
        "    u = extract_min(heap)\n"
        "```\n\n"
        "I lost the response lane; send it again.\n\n"
        "**Complexity:** O((V + E) log V)."
    )

    repaired = repair_runtime_boilerplate(answer)

    assert "```text\nwhile heap:\n    u = extract_min(heap)\n```" in repaired
    assert "lost the response lane" not in repaired
    assert repaired.endswith("**Complexity:** O((V + E) log V).")


@pytest.mark.parametrize(
    "reply",
    (
        "Ask me again.",
        "Something went wrong with my external coordination.",
    ),
)
def test_pure_runtime_boilerplate_repairs_to_empty(reply: str):
    assert repair_runtime_boilerplate(reply) == ""
