"""She may be curious. She may not take the lane the person is waiting on.

Live 2026-07-28, on a fresh boot with nobody typing:

    [ZENITH] Prompt plan: ... origin=deep_research_synth
    Routing to Cortex (timeout=103s, user_facing=True)
    conversation_ready = False, reason = active_generation_in_flight

The 32B had finished warming at 15.5s. What kept the desktop unusable was her
own curiosity loop: deep_research escalates its synthesis to
foreground_request=True on the reasoning that it "IS the deliverable the
person asked for" — which is true when a person asked, and false when
curiosity started it. Nobody was waiting on that research. Bryan was waiting
to talk.

Autonomous research keeps running; it just stops claiming foreground
priority. An empty background result then means "not now", which is the
correct answer for work nobody is waiting on.
"""

import pytest

from core.skills.deep_research import ResearchState

pytestmark = pytest.mark.unit


def test_research_is_autonomous_unless_someone_asked():
    """The default matters: every existing caller inherits it."""
    assert ResearchState(original_question="orcas").requested_by_user is False


def test_a_person_can_still_request_research():
    assert (
        ResearchState(original_question="orcas", requested_by_user=True).requested_by_user
        is True
    )


def test_only_a_requested_synthesis_retries_on_the_foreground_lane():
    import inspect

    from core.skills import deep_research

    source = inspect.getsource(deep_research.synthesize_answer)
    assert "if not state.requested_by_user:" in source
    assert "leaving" in source and "foreground lane alone" in source


def test_the_skill_derives_who_asked_from_its_context():
    import inspect

    from core.skills import web_search

    source = inspect.getsource(web_search)
    assert "_requested_by_user" in source
    assert "requested_by_user=_requested_by_user" in source
