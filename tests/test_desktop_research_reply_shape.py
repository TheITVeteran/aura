"""The reply that carries research must not cut a clause or cite twice.

Live 2026-07-30 00:33. The Orca demo worked — folder, PDF with a real text
layer, three real sources — and the message Bryan read said:

    "... In my view, the reliable path is to treat the articles as evidence to
     compare, not as a single conclusion to repeat: where the sources converge
     I can summarize confidently, and where they differ I should Sources: Map
     shows where orcas are attacking and sinking boats ..."

Two defects in one sentence. `synthesis[:1200]` cut at a character count, mid
clause, and then a "Sources:" block was appended unconditionally onto the stump
— restating all three sources the synthesis had already listed directly above.
"""
from __future__ import annotations

from interface.routes.chat import _desktop_task_research_response

SOURCES = [
    {"title": "USA TODAY orcas", "url": "https://usatoday.com/a"},
    {"title": "Oceana Spotlight", "url": "https://oceana.org/b"},
    {"title": "National Geographic", "url": "https://nationalgeographic.com/c"},
]

CITED_SYNTHESIS = (
    "Orcas are apex predators. In my view the evidence is mixed.\n\n"
    "Sources opened or consulted:\n"
    "1. USA TODAY orcas — https://usatoday.com/a\n"
    "2. Oceana Spotlight — https://oceana.org/b\n"
    "3. National Geographic — https://nationalgeographic.com/c"
)


def _reply(synthesis: str, sources=SOURCES) -> str:
    return _desktop_task_research_response(
        {"ok": True, "research": {"query": "orcas", "synthesis": synthesis, "sources": sources}},
        completed=4,
        requested=4,
    )


def test_sources_already_in_the_synthesis_are_not_restated() -> None:
    reply = _reply(CITED_SYNTHESIS)
    assert " Sources: " not in reply
    for source in SOURCES:
        assert reply.count(source["url"]) == 1, f"{source['url']} cited twice"


def test_sources_missing_from_the_synthesis_are_still_added() -> None:
    """Suppressing the duplicate must not suppress the only citation."""
    reply = _reply("Orcas are apex predators.")
    assert " Sources: " in reply
    for source in SOURCES:
        assert source["url"] in reply


def test_a_partially_cited_synthesis_still_gets_the_sentence() -> None:
    partial = "Orcas are apex predators. See https://usatoday.com/a for the map."
    reply = _reply(partial)
    assert " Sources: " in reply
    assert "https://oceana.org/b" in reply


def test_a_long_synthesis_is_clipped_at_a_sentence() -> None:
    long_synthesis = ("Orcas are apex predators. " * 60) + "and where they differ I should"
    reply = _reply(long_synthesis)
    assert "where they differ I should" not in reply
    body = reply.split(" Sources: ")[0]
    assert body.rstrip().endswith((".", "…")), body[-80:]


def test_the_step_count_still_survives_a_clip() -> None:
    reply = _reply("Orcas are apex predators. " * 200)
    assert "Completed 4/4 governed desktop steps." in reply


def test_sources_after_the_clip_boundary_are_not_lost() -> None:
    synthesis = ("Orcas are apex predators. " * 60) + CITED_SYNTHESIS
    reply = _reply(synthesis)
    for source in SOURCES:
        assert source["url"] in reply


def test_no_sources_at_all_says_so() -> None:
    reply = _reply("Orcas are apex predators.", sources=[])
    assert "No source URL was available" in reply
