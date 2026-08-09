"""A loop pointed at someone's screen is the most invasive thing here.

Companion mode keeps looking so that "what do you think of this?" is answered
from something she saw thirty seconds ago instead of from a capture that
starts when the question ends. That latency win is real and so is the cost:
continuous observation of a person's desktop.

The rules that make it acceptable are structural, and these tests drive them:

  * incognito and password managers are NEVER CAPTURED — not captured and
    discarded, not captured and redacted. The screen read does not happen;
  * the suppression that stops her speaking unprompted stops her looking
    unprompted, because reading someone's screen is not the lesser act;
  * a private window does not even leak through the RECEIPT — a skip record
    naming "Chase Bank — Private" has published the thing the skip protected;
  * an unavailable permission check means no observation. An unreachable
    authority is not permission.
"""
from __future__ import annotations

import asyncio

import pytest

from core.perception.ambient_presence import (
    AmbientPresence,
    PresenceMode,
    ScreenContext,
    SkipReason,
    _proactivity_suppressed,
    is_private_context,
)

#: Captured at import, before the autouse fixture replaces the module
#: attribute — reloading the module to reach it instead broke module identity
#: for every other test in the file.
_REAL_SUPPRESSION_CHECK = _proactivity_suppressed


@pytest.fixture
def presence():
    instance = AmbientPresence()
    instance.set_mode(PresenceMode.BUBBLE)
    return instance


@pytest.fixture(autouse=True)
def _allow_proactivity(monkeypatch):
    monkeypatch.setattr(
        "core.perception.ambient_presence._proactivity_suppressed", lambda: False
    )


def _with_context(presence, app, title, *, text="some screen text"):
    async def _context():
        return ScreenContext(app=app, title=title)

    async def _read():
        _read.called = True
        return text

    _read.called = False
    presence._current_context = _context
    presence._read_screen_text = _read
    return _read


# ───────────────────────────────────────────────── private is never read


@pytest.mark.parametrize(
    "app,title",
    [
        ("Google Chrome", "Chase Bank — Incognito"),
        ("Google Chrome", "New Incognito Tab"),
        ("Safari", "Private Browsing"),
        ("Microsoft Edge", "InPrivate — news"),
        ("1Password", "Personal vault"),
        ("Bitwarden", "My Vault"),
        ("Keychain Access", "login"),
        ("Firefox", "Private Window — mail"),
        ("Terminal", "password reset instructions"),
    ],
)
def test_a_private_window_is_never_captured(presence, app, title):
    reader = _with_context(presence, app, title)

    result = asyncio.run(presence.tick())

    assert result.observed is False
    assert result.skip_reason is SkipReason.PRIVATE_WINDOW
    assert reader.called is False, (
        f"the screen was READ for {app} | {title}; refusing after reading is "
        "not refusing"
    )


def test_a_private_window_does_not_leak_through_the_receipt(presence):
    _with_context(presence, "Google Chrome", "Chase Bank Login — Incognito")

    payload = asyncio.run(presence.tick()).to_dict()

    assert "Chase" not in str(payload), (
        "the skip record published the title it existed to protect"
    )
    assert payload["title"] == ""
    assert payload["skip_reason"] == "private_window"


def test_the_skip_is_counted_so_the_rule_is_visible(presence):
    _with_context(presence, "1Password", "vault")
    asyncio.run(presence.tick())

    assert presence.state()["private_windows_skipped"] == 1


def test_an_ordinary_window_is_read(presence):
    reader = _with_context(presence, "Google Chrome", "GitHub — aura")

    result = asyncio.run(presence.tick())

    assert result.observed is True
    assert reader.called is True
    assert result.characters > 0


def test_the_public_predicate_matches_the_loop():
    """Other surfaces must be able to apply the same rule."""
    assert is_private_context("Safari", "Private Browsing") is True
    assert is_private_context("Google Chrome", "GitHub") is False


# ──────────────────────────────── looking is governed like speaking


def test_suppressed_proactivity_stops_her_looking(presence, monkeypatch):
    monkeypatch.setattr(
        "core.perception.ambient_presence._proactivity_suppressed", lambda: True
    )
    reader = _with_context(presence, "Google Chrome", "GitHub")

    result = asyncio.run(presence.tick())

    assert result.skip_reason is SkipReason.SUPPRESSED
    assert reader.called is False


def test_an_unavailable_permission_check_means_no_observation(monkeypatch):
    """Fail closed. An unreachable authority is not permission.

    The autouse fixture replaces the module attribute, so this calls the
    function object captured at import and removes what it depends on.
    """
    monkeypatch.delattr(
        "core.agency.agency_core._proactivity_suppressed_now", raising=False
    )

    assert _REAL_SUPPRESSION_CHECK() is True


def test_hidden_means_hidden(presence):
    reader = _with_context(presence, "Google Chrome", "GitHub")
    presence.hide()

    result = asyncio.run(presence.tick())

    assert result.skip_reason is SkipReason.HIDDEN
    assert reader.called is False


# ─────────────────────────────────────────── cheap when nothing changed


def test_an_unchanged_window_is_not_re_read(presence):
    reader = _with_context(presence, "Google Chrome", "GitHub — aura")
    assert asyncio.run(presence.tick()).observed is True
    reader.called = False

    second = asyncio.run(presence.tick())

    assert second.skip_reason is SkipReason.UNCHANGED
    assert reader.called is False, "an idle machine paid for a capture"


def test_moving_to_a_new_window_costs_one_capture(presence):
    _with_context(presence, "Google Chrome", "GitHub")
    asyncio.run(presence.tick())
    reader = _with_context(presence, "Slack", "general")

    result = asyncio.run(presence.tick())

    assert result.observed is True
    assert reader.called is True


def test_a_stale_context_is_re_read_even_unchanged(presence, monkeypatch):
    """A long document being scrolled is one window and different content."""
    _with_context(presence, "Preview", "spec.pdf")
    asyncio.run(presence.tick())

    monkeypatch.setattr("core.perception.ambient_presence._RESTALE_AFTER_S", 0.0)
    reader = _with_context(presence, "Preview", "spec.pdf")

    assert asyncio.run(presence.tick()).observed is True
    assert reader.called is True


# ────────────────────────────────────────────────── she stays quiet


def test_she_does_not_speak_while_suppressed(presence, monkeypatch):
    monkeypatch.setattr(
        "core.perception.ambient_presence._proactivity_suppressed", lambda: True
    )

    assert presence.offer_utterance("I noticed something") is False
    assert presence.state()["has_utterance"] is False


def test_she_does_not_speak_while_hidden(presence):
    presence.hide()
    assert presence.offer_utterance("I noticed something") is False


def test_an_accepted_utterance_reaches_the_bubble(presence):
    assert presence.offer_utterance("That test name contradicts its assertion.")
    state = presence.state()
    assert state["has_utterance"] is True
    assert "contradicts" in state["utterance"]


def test_a_dismissed_message_does_not_come_back(presence):
    presence.offer_utterance("something")
    presence.clear_utterance()

    assert presence.state()["has_utterance"] is False


def test_hiding_her_drops_the_queued_thought(presence):
    """Otherwise it appears the instant she is unhidden, which is not hiding."""
    presence.offer_utterance("something")
    presence.hide()

    assert presence.state()["has_utterance"] is False


def test_an_empty_utterance_is_not_an_utterance(presence):
    assert presence.offer_utterance("   ") is False


# ──────────────────────────────────────────── the latency payoff


def test_recall_answers_from_what_she_already_saw(presence):
    from core.perception.observation_evidence import get_observation_memory

    get_observation_memory().clear()
    _with_context(
        presence,
        "Google Chrome",
        "GitHub — aura",
        text="pull request #412: fix the retry budget in mlx_worker",
    )
    asyncio.run(presence.tick())

    recalled = presence.recall_for("what was that pull request about?")

    assert "retry budget" in recalled, (
        "the observation was taken but not retained, so a question would "
        "force a fresh capture — which is the latency this exists to remove"
    )


def test_recall_is_empty_rather_than_wrong_when_she_has_not_looked(presence):
    from core.perception.observation_evidence import get_observation_memory

    get_observation_memory().clear()

    assert presence.recall_for("what is on my screen?") == ""


def test_the_bubble_can_be_moved(presence):
    assert presence.move_bubble(120.0, 480.0) == (120.0, 480.0)
    assert presence.state()["bubble_position"] == [120.0, 480.0]


# ───────────────────────────────── the loop, the voice, and the surface


def test_the_loop_backs_off_instead_of_spinning_on_a_broken_capture(presence, monkeypatch):
    """A revoked accessibility permission must not become a hot loop."""
    sleeps = []

    async def _boom():
        raise RuntimeError("accessibility permission revoked")

    async def _sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 4:
            presence.stop()

    monkeypatch.setattr(presence, "tick", _boom)
    monkeypatch.setattr(asyncio, "sleep", _sleep)

    asyncio.run(presence.run(interval_s=2.0))

    assert sleeps == sorted(sleeps), f"delay did not grow: {sleeps}"
    assert max(sleeps) <= 300.0, "back-off is unbounded"


def test_she_does_not_speak_twice_in_quick_succession(presence, monkeypatch):
    """Unsolicited comment is a budget, not a feature."""
    presence._last_spoke_at = __import__("time").time()

    async def _would_speak(result):
        raise AssertionError("the speech gap was not honoured")

    monkeypatch.setattr(
        "core.perception.ambient_utterance.consider_utterance", _would_speak
    )
    _with_context(presence, "Terminal", "pytest")

    asyncio.run(presence._consider_speaking(asyncio.run(presence.tick())))


def test_a_waiting_message_is_not_replaced(presence, monkeypatch):
    """A person who has not read the last one does not get a stream."""
    presence.offer_utterance("the first thing")

    async def _would_speak(result):
        raise AssertionError("she spoke over an unread message")

    monkeypatch.setattr(
        "core.perception.ambient_utterance.consider_utterance", _would_speak
    )
    _with_context(presence, "Terminal", "pytest")

    asyncio.run(presence._consider_speaking(asyncio.run(presence.tick())))
    assert presence.state()["utterance"] == "the first thing"


class TestUtteranceFilter:
    """The default is silence, and it takes a fault to leave it."""

    def test_an_ordinary_screen_says_nothing(self):
        from core.perception.ambient_utterance import _worth_noticing

        assert _worth_noticing("inbox — 4 unread\nCalendar\nDraft: quarterly plan") == ""

    def test_a_traceback_is_worth_noticing(self):
        from core.perception.ambient_utterance import _worth_noticing

        assert _worth_noticing(
            "Traceback (most recent call last):\n  File x\nValueError"
        )

    def test_documentation_about_errors_is_not_a_fault(self):
        from core.perception.ambient_utterance import _worth_noticing

        assert _worth_noticing(
            "Stack Overflow — how to fix error: connection refused"
        ) == ""

    def test_a_single_bare_error_word_is_not_enough(self):
        from core.perception.ambient_utterance import _worth_noticing

        assert _worth_noticing("the error: field is optional in this schema") == ""

    @pytest.mark.parametrize(
        "reply",
        [
            "NOTHING",
            "nothing.",
            "Hi Bryan!",
            "I see you're working on the parser.",
            "It looks like you're debugging.",
            "Would you like help with that?",
            "I noticed you have a test failing.",
            "ok",
            "",
        ],
    )
    def test_template_and_refusal_replies_are_suppressed(self, reply):
        from core.perception.ambient_utterance import _is_refusal_or_noise

        assert _is_refusal_or_noise(reply) is True

    def test_a_specific_observation_survives(self):
        from core.perception.ambient_utterance import _is_refusal_or_noise

        assert _is_refusal_or_noise(
            "That AssertionError is comparing 391 to 391.0 — the fixture "
            "casts to float but the check uses ==."
        ) is False

    def test_no_canned_fallback_when_her_voice_is_unavailable(self, monkeypatch):
        """Silence beats "I noticed an error on your screen!".

        A canned line interrupts and carries no information, and a person
        cannot tell it from a real observation until after being interrupted.
        """
        from core.perception import ambient_utterance

        class _Observation:
            capture = "Traceback (most recent call last): ValueError somewhere"

            def for_reasoning(self):
                return "framed"

        monkeypatch.setattr(
            ambient_utterance, "_latest_observation", lambda: _Observation()
        )
        monkeypatch.setattr(
            "core.container.ServiceContainer.get",
            staticmethod(lambda name, default=None: None),
        )

        assert asyncio.run(ambient_utterance._compose(_Observation(), "traceback")) == ""
