"""Language center: the two public APIs disagreed, and the filter deleted truth."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.brain.language_center import LanguageCenter, strip_meta_commentary

pytestmark = pytest.mark.unit


def _thought(stance="neutral", tone="plain"):
    return SimpleNamespace(stance=stance, tone=tone, model_tier="fast",
                           llm_briefing="", to_system_prompt=lambda: "")


# ── identity integrity ─────────────────────────────────────────────────────


@pytest.mark.parametrize("disclosure", [
    "I don't have feelings about that, but here is what I found.",
    "I do not have opinions on which of these you should pick.",
    "I can't have experience of the room, so I'm going on the logs.",
    "I cannot have consciousness of what happened while I was down.",
])
def test_truthful_self_disclosure_is_not_deleted(disclosure):
    """language_center carried a dead _META_RE list whose patterns included one
    matching "I don't have feelings/opinions/consciousness". It never ran, but a
    test certified it as identity enforcement — so the trap was one wiring
    change away from deleting truthful self-disclosure and leaving the user with
    the opposite impression. The list is gone; this pins the behaviour."""
    out = strip_meta_commentary(disclosure)

    assert out.strip(), "truthful self-disclosure must survive"
    assert "have" in out


def test_leaked_internal_structure_is_still_stripped():
    """The scrubber's real job: remove leaked metadata headers, not
    self-disclosure."""
    cleaned = strip_meta_commentary(
        "INTERNAL STATE: agitated\nHere is the real answer."
    )

    assert "INTERNAL STATE:" not in cleaned
    assert "Here is the real answer." in cleaned


# ── the fallback must not claim health it never checked ────────────────────


def test_fallback_does_not_assert_cognitive_core_health():
    """It opened 'The cognitive core is active' while every backend was down
    and no readiness evidence had been consulted — the originating failure may
    BE a core failure, so the claim could be false exactly when emitted."""
    center = LanguageCenter()

    for stance, tone in (("curious", "plain"), ("neutral", "direct"),
                         ("neutral", "plain")):
        text = center._fallback_response(_thought(stance, tone), "hi")
        assert "cognitive core is active" not in text.lower()
        assert "core is active" not in text.lower()


# ── the two public APIs must behave the same ───────────────────────────────


def _collect(agen):
    async def _run():
        return [c async for c in agen]
    return asyncio.run(_run())


def test_streaming_applies_the_same_meta_scrub_as_express(monkeypatch):
    """express_stream yielded RAW router chunks, so whichever endpoint a caller
    used decided whether the output was governed at all."""
    center = LanguageCenter()
    center._router = object()
    center._fallback_mode = False

    async def _fake_stream(prompt, thought, *, origin="user"):
        yield "INTERNAL STATE: agitated\n"
        yield "Here is the actual answer.\n"

    monkeypatch.setattr(center, "_dispatch_stream", _fake_stream)

    out = "".join(_collect(center.express_stream(_thought(), "hi")))

    assert "INTERNAL STATE:" not in out, "streaming must scrub like express()"
    assert "Here is the actual answer." in out


def test_streaming_falls_back_when_everything_is_filtered_away(monkeypatch):
    """If the filters remove the entire response the user has received no
    answer, which is exactly the case the fallback exists for."""
    center = LanguageCenter()
    center._router = object()
    center._fallback_mode = False

    async def _only_leaked_metadata(prompt, thought, *, origin="user"):
        yield "INTERNAL STATE: agitated\n"

    monkeypatch.setattr(center, "_dispatch_stream", _only_leaked_metadata)

    out = "".join(_collect(center.express_stream(_thought(), "hi")))

    assert "language center failed" in out


def test_streaming_holds_partial_lines_until_complete(monkeypatch):
    """These filters are anchored, so a fragment cannot be judged — a partial
    line is held rather than half-matched."""
    center = LanguageCenter()
    center._router = object()
    center._fallback_mode = False

    async def _split(prompt, thought, *, origin="user"):
        for piece in ["INTERNAL ", "STATE: agitated\n", "Real content here."]:
            yield piece

    monkeypatch.setattr(center, "_dispatch_stream", _split)

    out = "".join(_collect(center.express_stream(_thought(), "hi")))

    assert "INTERNAL STATE:" not in out
    assert "Real content here." in out
