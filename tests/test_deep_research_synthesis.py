"""Deep research synthesis must not be discarded for being deprioritised."""
import asyncio, types, pytest
from core.skills import deep_research as dr


class _Brain:
    def __init__(self, empty_first=True):
        self.calls = []
        self.empty_first = empty_first

    async def generate(self, prompt, options=None, **kwargs):
        self.calls.append(kwargs)
        if self.empty_first and len(self.calls) == 1:
            return {"response": "", "status": "queued_behind_foreground"}
        return {"response": "A real synthesis across the sources."}


def _state(requested_by_user: bool = True):
    s = types.SimpleNamespace()
    s.requested_by_user = requested_by_user
    s.original_question = "orcas"
    s.all_sources = [{"url": f"https://x{i}.org/a"} for i in range(5)]
    s.running_summary = "summary"
    s.search_results = [types.SimpleNamespace(query="orcas", content="Orcas are apex predators.")]
    s.sources_gathered = s.all_sources
    s.synthesis_status = ""
    s.synthesis_detail = ""
    s.final_answer = ""
    s.search_queries = ["orcas"]
    s.loop_count = 1
    return s


def test_empty_background_synthesis_is_retried_as_foreground():
    """When a PERSON asked, the sources are worth a foreground retry."""
    brain = _Brain(empty_first=True)
    out = asyncio.run(dr.synthesize_answer(_state(requested_by_user=True), brain))
    assert out.synthesis_status == "ok", "sources were discarded instead of retried"
    assert len(brain.calls) == 2, "the retry must actually happen"
    assert brain.calls[1].get("foreground_request") is True


def test_autonomous_synthesis_never_takes_the_foreground_lane():
    """Curiosity may research; it may not take the lane a person waits on.

    Live 2026-07-28 an autonomous synthesis escalated itself to
    foreground_request=True on a fresh boot and held the cortex while
    conversation_ready stayed False — the desktop was unusable because she was
    reading about something nobody had asked for. An empty background result
    is the correct answer for work nobody is waiting on.
    """
    brain = _Brain(empty_first=True)
    asyncio.run(dr.synthesize_answer(_state(requested_by_user=False), brain))
    assert len(brain.calls) == 1, "autonomous research must not retry on the foreground lane"
    assert not brain.calls[0].get("foreground_request")


def test_a_first_pass_success_does_not_retry():
    brain = _Brain(empty_first=False)
    out = asyncio.run(dr.synthesize_answer(_state(), brain))
    assert out.synthesis_status == "ok"
    assert len(brain.calls) == 1, "a working synthesis must not pay for a second call"


def test_the_brain_adapter_honours_a_foreground_request():
    """The adapter accepted **kwargs and threw them away.

    It hardcoded is_background=True, so a synthesis the person was waiting for
    was admitted as background work, queued behind foreground headroom, and came
    back instantly empty — and the deep-research foreground retry could not take
    effect, because its request never left this method. Measured live twice in
    one turn, including the retry.
    """
    from core.skills.web_search import _DeepResearchBrainAdapter

    seen = {}

    class _Engine:
        async def generate(self, prompt, **kwargs):
            seen.update(kwargs)
            return {"response": "ok"}

    adapter = _DeepResearchBrainAdapter(_Engine())

    asyncio.run(adapter.generate("p", foreground_request=True))
    assert seen.get("is_background") is False, "a foreground request stayed background"
    assert seen.get("foreground_request") is True

    seen.clear()
    asyncio.run(adapter.generate("p"))
    assert seen.get("is_background") is True, "ordinary research must stay background"
    assert seen.get("foreground_request") is False
