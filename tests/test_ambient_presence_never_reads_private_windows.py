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


def test_the_gate_this_loop_asks_actually_exists():
    """The gate must be REACHABLE, not merely fail-closed when it is not.

    This is the test that was missing while the loop was dead. The original
    version deleted the symbol with ``raising=False`` and asserted the check
    returned True — which it did, but it did so because the import target had
    never existed, not because the deletion worked. A fail-closed assertion
    passes identically whether the gate is present or absent, so on its own
    it proves nothing about wiring.

    Importing the real name here is the half that has teeth: it fails if the
    gate moves or is renamed, which is exactly how this broke.
    """
    from core.brain.initiative_engine import proactivity_suppressed_now

    assert callable(proactivity_suppressed_now)


def test_the_gate_is_not_stuck_shut(monkeypatch):
    """A gate that always says "suppressed" is a dead subsystem, not a safe one.

    The whole organ skipped every tick for its entire life because the gate
    answered True unconditionally. Fail-closed is right; fail-closed-ALWAYS is
    an outage that looks like a policy. With an orchestrator that permits
    speech, the answer must be False and she must actually observe.
    """
    from core.container import ServiceContainer

    class _PermittingOrchestrator:
        _suppress_unsolicited_proactivity_until = 0.0

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: (
                _PermittingOrchestrator() if name == "orchestrator" else default
            )
        ),
    )

    assert _REAL_SUPPRESSION_CHECK() is False


def test_an_unavailable_permission_check_means_no_observation(monkeypatch):
    """Fail closed. An unreachable authority is not permission.

    Paired with the two tests above so that "returns True" is only accepted
    as correct once the gate has been shown to exist and to be capable of
    returning False.
    """
    from core.container import ServiceContainer

    def _explode(name, default=None):
        raise RuntimeError("container unavailable")

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(_explode))

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


class TestTheLatencyPayoff:
    """A question about the screen answered from what she already saw."""

    def test_a_recent_observation_answers_without_a_new_capture(self, presence):
        from core.perception.observation_evidence import get_observation_memory

        get_observation_memory().clear()
        _with_context(
            presence, "Google Chrome", "GitHub", text="PR #412 fix the retry budget"
        )
        asyncio.run(presence.tick())

        observation = presence.fresh_observation_for("what's on my screen?")

        assert observation is not None
        assert "retry budget" in observation.capture

    def test_a_stale_observation_is_refused_so_the_caller_captures(self, presence):
        """Answering about NOW with a fact about THEN is answering wrong."""
        from core.perception.observation_evidence import get_observation_memory

        get_observation_memory().clear()
        _with_context(presence, "Google Chrome", "GitHub", text="something")
        asyncio.run(presence.tick())

        assert presence.fresh_observation_for("what's on screen?", max_age_s=0.0) is None

    def test_a_public_observation_is_not_reused_after_switching_to_private(
        self, presence
    ):
        """Private is not read, and an older public screen is not called current."""
        from core.perception.observation_evidence import get_observation_memory

        get_observation_memory().clear()
        _with_context(
            presence,
            "Google Chrome",
            "GitHub",
            text="public pull request content",
        )
        assert asyncio.run(presence.tick()).observed is True

        private_reader = _with_context(
            presence,
            "Google Chrome",
            "Bank account - Incognito",
            text="must never be captured",
        )
        private_tick = asyncio.run(presence.tick())

        assert private_tick.skip_reason is SkipReason.PRIVATE_WINDOW
        assert private_reader.called is False
        assert presence.fresh_observation_for("what is on my screen?") is None
        state = presence.state()
        assert state["foreground_private"] is True
        assert "Bank" not in str(state)

    def test_hiding_invalidates_current_screen_fast_recall(self, presence):
        from core.perception.observation_evidence import get_observation_memory

        get_observation_memory().clear()
        _with_context(presence, "Terminal", "zsh", text="current terminal")
        assert asyncio.run(presence.tick()).observed is True

        presence.hide()

        assert presence.fresh_observation_for("what is on my screen?") is None

    def test_unknown_foreground_invalidates_current_screen_fast_recall(self, presence):
        from core.perception.observation_evidence import get_observation_memory

        get_observation_memory().clear()
        _with_context(presence, "Terminal", "zsh", text="current terminal")
        assert asyncio.run(presence.tick()).observed is True

        async def _unknown():
            return None

        presence._current_context = _unknown
        assert asyncio.run(presence.tick()).skip_reason is SkipReason.CAPTURE_FAILED
        assert presence.fresh_observation_for("what is on my screen?") is None

    def test_the_reframed_observation_follows_the_new_question(self, presence):
        """Ambient captures carry no request; the question supplies the shape."""
        from core.perception.observation_evidence import get_observation_memory

        get_observation_memory().clear()
        _with_context(
            presence, "Terminal", "zsh", text="line one\nline two\nline three"
        )
        asyncio.run(presence.tick())

        describe = presence.fresh_observation_for("what do you see?")
        quote = presence.fresh_observation_for("read me the exact text word for word")

        assert describe.for_reasoning() != quote.for_reasoning(), (
            "the same frame was produced for 'describe it' and 'quote it'"
        )

    def test_reframing_does_not_mutate_the_stored_observation(self, presence):
        from core.perception.observation_evidence import (
            ObservationKind,
            get_observation_memory,
        )

        get_observation_memory().clear()
        _with_context(presence, "Terminal", "zsh", text="content")
        asyncio.run(presence.tick())

        presence.fresh_observation_for("a question")
        stored = get_observation_memory().latest(ObservationKind.SCREEN_TEXT)

        assert stored.request == "", (
            "the stored ambient observation was rewritten by a reader, so the "
            "next question would inherit the previous question's shape"
        )

    def test_no_observation_returns_none_rather_than_an_empty_one(self, presence):
        from core.perception.observation_evidence import get_observation_memory

        get_observation_memory().clear()

        assert presence.fresh_observation_for("anything") is None


class TestThePrivacyRuleHoldsForEveryCaller:
    """A privacy rule enforced in one caller is a rule with a hole in it."""

    def test_the_skill_itself_refuses_a_private_foreground(self, monkeypatch):
        from core.skills.computer_use import ComputerUseSkill

        monkeypatch.setattr(
            "core.senses.screen_context.frontmost_window_hint",
            lambda: ("Google Chrome", "Chase Bank — Incognito"),
        )
        captured = {"called": False}

        def _never(self):
            captured["called"] = True
            return "SCREEN CONTENTS"

        monkeypatch.setattr(ComputerUseSkill, "_read_screen_text_macos", _never)

        result = ComputerUseSkill.__new__(ComputerUseSkill).read_screen_text()

        assert "refused" in result
        assert captured["called"] is False, (
            "an explicit screen read captured a private window; the rule only "
            "held in the ambient loop"
        )

    def test_the_refusal_does_not_name_the_private_window(self, monkeypatch):
        from core.skills.computer_use import ComputerUseSkill

        monkeypatch.setattr(
            "core.senses.screen_context.frontmost_window_hint",
            lambda: ("Google Chrome", "Chase Bank — Incognito"),
        )

        result = ComputerUseSkill.__new__(ComputerUseSkill).read_screen_text()

        assert "Chase" not in result

    def test_an_ordinary_foreground_is_still_read(self, monkeypatch):
        from core.skills.computer_use import ComputerUseSkill

        monkeypatch.setattr(
            "core.senses.screen_context.frontmost_window_hint",
            lambda: ("Terminal", "zsh"),
        )
        monkeypatch.setattr(
            ComputerUseSkill, "_read_screen_text_macos", lambda self: "SCREEN"
        )

        assert ComputerUseSkill.__new__(ComputerUseSkill).read_screen_text() == "SCREEN"

    def test_an_unknown_foreground_does_not_block_an_explicit_request(
        self, monkeypatch
    ):
        """Empty means "cannot tell", and the caller decides.

        An explicit request proceeds — the person asked. The ambient loop
        refuses. Both are right for their context, and screen_context does
        not make that policy.
        """
        from core.skills.computer_use import ComputerUseSkill

        monkeypatch.setattr(
            "core.senses.screen_context.frontmost_window_hint", lambda: ("", "")
        )
        monkeypatch.setattr(
            ComputerUseSkill, "_read_screen_text_macos", lambda self: "SCREEN"
        )

        assert ComputerUseSkill.__new__(ComputerUseSkill).read_screen_text() == "SCREEN"


class TestTheFastPathIsCausalInTheAnswerLane:
    """Retaining observations changes nothing unless the answer path reads them."""

    def _skill(self):
        from core.skills.desktop_task import DesktopTaskSkill

        return DesktopTaskSkill.__new__(DesktopTaskSkill)

    def _params(self, objective, steps=()):
        from core.skills.desktop_task import DesktopTaskParams

        return DesktopTaskParams(objective=objective, steps=list(steps))

    def test_a_screen_question_is_answered_from_a_recent_observation(
        self, presence, monkeypatch
    ):
        from core.perception.observation_evidence import get_observation_memory

        get_observation_memory().clear()
        _with_context(presence, "Terminal", "zsh", text="pytest: 3 failed, 12 passed")
        asyncio.run(presence.tick())
        monkeypatch.setattr(
            "core.perception.ambient_presence.get_ambient_presence", lambda: presence
        )

        answer = self._skill()._ambient_answer(
            "what's on my screen?", self._params("what's on my screen?")
        )

        assert answer is not None
        assert answer["captured_now"] is False
        assert "3 failed" in answer["observation"]
        assert answer["observation_age_s"] is not None, (
            "an answer from a moment ago must carry its age, or it reads as a "
            "reading of this instant"
        )

    def test_a_request_with_an_action_still_captures(self, presence, monkeypatch):
        """The screen is about to change; a cached reading is the wrong one."""
        from core.perception.observation_evidence import get_observation_memory

        get_observation_memory().clear()
        _with_context(presence, "Terminal", "zsh", text="anything")
        asyncio.run(presence.tick())
        monkeypatch.setattr(
            "core.perception.ambient_presence.get_ambient_presence", lambda: presence
        )

        objective = "click the run button and tell me what's on my screen"
        assert self._skill()._ambient_answer(objective, self._params(objective)) is None

    def test_an_explicit_plan_is_never_short_circuited(self, presence, monkeypatch):
        from core.perception.observation_evidence import get_observation_memory
        from core.skills.desktop_task import DesktopTaskStep

        get_observation_memory().clear()
        _with_context(presence, "Terminal", "zsh", text="anything")
        asyncio.run(presence.tick())
        monkeypatch.setattr(
            "core.perception.ambient_presence.get_ambient_presence", lambda: presence
        )

        params = self._params(
            "what's on my screen?",
            steps=[
                DesktopTaskStep(
                    action="read_screen_text", target="", reason="r", expect="e"
                )
            ],
        )
        assert self._skill()._ambient_answer("what's on my screen?", params) is None

    def test_a_non_observation_objective_is_untouched(self, presence, monkeypatch):
        monkeypatch.setattr(
            "core.perception.ambient_presence.get_ambient_presence", lambda: presence
        )
        objective = "open Safari and go to github.com"
        assert self._skill()._ambient_answer(objective, self._params(objective)) is None

    def test_no_recent_observation_falls_through_to_a_real_capture(
        self, presence, monkeypatch
    ):
        """None is the pre-ambient behaviour, so the fast path is always safe."""
        from core.perception.observation_evidence import get_observation_memory

        get_observation_memory().clear()
        monkeypatch.setattr(
            "core.perception.ambient_presence.get_ambient_presence", lambda: presence
        )

        assert (
            self._skill()._ambient_answer(
                "what's on my screen?", self._params("what's on my screen?")
            )
            is None
        )

    def test_a_broken_ambient_organ_falls_through_rather_than_failing(self, monkeypatch):
        def _boom():
            raise RuntimeError("ambient organ gone")

        monkeypatch.setattr(
            "core.perception.ambient_presence.get_ambient_presence", _boom
        )

        assert (
            self._skill()._ambient_answer(
                "what's on my screen?", self._params("what's on my screen?")
            )
            is None
        )
