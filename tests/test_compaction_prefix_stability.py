"""The compaction cut must land in the same place every turn.

`limit` drifts with the surrounding budget, so the head boundary drifted with
it and the omitted-middle marker landed on a different character each turn —
mid-word, as " obser" then "se". Everything behind the marker therefore looked
like new text to the KV prefix trie.

Measured live on the user surface: reuse stuck at 16% (286 of 1802 tokens),
with divergence beginning exactly at the omission marker, so roughly 1,500
tokens were re-prefilled on every single turn. Prefill is the latency.
"""

from __future__ import annotations

from core.brain.llm.latent_cortex.context_compaction import _OMISSION_MARKER, _fit_ends

SCAFFOLD = "You are Aura Luna. " + "stable scaffold that does not change between turns. " * 120


def _head(text: str, limit: int) -> str:
    return _fit_ends(text, limit).split(_OMISSION_MARKER)[0]


def test_the_head_is_identical_across_drifting_budgets():
    heads = {_head(SCAFFOLD, limit) for limit in (2388, 2395, 2397, 2400, 2402, 2410)}
    assert len(heads) == 1, (
        f"the cut moved {len(heads)} different ways under budget drift; "
        "everything behind it re-prefills"
    )


def test_the_cut_lands_on_a_word_boundary():
    head = _head(SCAFFOLD, 2400)
    assert head, "the head must not be empty"
    assert head.endswith(" ") or head[-1].isalnum(), (
        f"the cut landed mid-word: {head[-20:]!r}"
    )


def test_the_output_still_respects_its_limit():
    for limit in (600, 1200, 2400, 4096):
        out = _fit_ends(SCAFFOLD, limit)
        assert len(out) <= limit, f"compaction overran its budget at {limit}"


def test_short_text_is_returned_untouched():
    assert _fit_ends("short", 2400) == "short"


def test_a_tiny_limit_still_degrades_safely():
    out = _fit_ends(SCAFFOLD, 10)
    assert len(out) <= 10


def test_changing_the_text_still_changes_the_cut():
    """Stability must not mean ignoring real content changes."""
    other = "Different scaffold entirely. " + "other content here. " * 120
    assert _head(SCAFFOLD, 2400) != _head(other, 2400)
