"""A picture of a rock, and of anything else.

Bryan, recording the demo: "Tried to switch it up and ask for a rock to be my
background but it crapped out because it couldn't define what a rock was?
Shouldn't that feature be super general? It managed to look up a rock but
couldn't figure out what to do after it got to images."

Reproduced exactly. The fetch did one thing: ask Wikipedia's summary endpoint
for the literal topic, capitalised.

    "rock"    -> no image available for topic 'rock'
                 ("Rock" is a disambiguation page, which has no thumbnail)
    "a rock"  -> topic lookup failed: HTTP Error 404
                 (there is no article called "A_rock")

She had found the image search perfectly well and then had nowhere to go,
because the one endpoint she could use demanded an exact article title she did
not have.

Then, once the chain existed, seven topics in a burst returned "no image
available for topic X (HTTP Error 429)" — a rate limit reported to the person
as though the thing they asked for did not exist.
"""

from __future__ import annotations

import inspect

import pytest


@pytest.fixture()
def ComputerUseSkill():  # noqa: N802 - reads as the class it provides
    """Imported inside the test, not at collection.

    Importing core.skills.computer_use at module scope initialised enough of
    the runtime that tests/test_constitutional_core.py started failing when
    this file happened to run before it — the classic pass-alone /
    fail-together shape. The rest of the suite imports inside its tests for
    exactly this reason.
    """
    from core.skills.computer_use import ComputerUseSkill as _Skill

    return _Skill


class TestTheTopicIsResolvedNotDemanded:
    def test_an_article_is_stripped_from_the_topic(self, ComputerUseSkill):
        """"a rock" is a request for a picture of a rock."""
        candidates = ComputerUseSkill._image_topic_candidates(
            "a rock", _DeadGateway(), {}
        )
        assert "Rock" in candidates

    def test_a_phrasing_wrapper_is_stripped(self, ComputerUseSkill):
        candidates = ComputerUseSkill._image_topic_candidates(
            "a picture of a traffic cone", _DeadGateway(), {}
        )
        assert any(item.lower() == "traffic cone" for item in candidates), candidates

    def test_the_literal_topic_is_tried_first(self, ComputerUseSkill):
        """It is free when it works, which is most of the time."""
        candidates = ComputerUseSkill._image_topic_candidates(
            "orca", _DeadGateway(), {}
        )
        assert candidates[0] == "Orca"

    def test_an_empty_topic_yields_nothing(self, ComputerUseSkill):
        assert ComputerUseSkill._image_topic_candidates("", _DeadGateway(), {}) == []

    def test_a_disambiguation_page_is_not_a_picture(self, ComputerUseSkill):
        source = inspect.getsource(ComputerUseSkill._fetch_topic_image)
        assert 'endswith("disambiguation")' in source, (
            "a disambiguation page is a list of other pages and must not end "
            "the search"
        )

    def test_commons_is_the_last_resort(self, ComputerUseSkill):
        """The only step that is an image search rather than a lookup."""
        source = inspect.getsource(ComputerUseSkill._fetch_topic_image)
        assert "_commons_image_candidate" in source


class TestThrottlingIsNotAbsence:
    def test_a_429_is_retried_before_giving_up(self, ComputerUseSkill):
        source = inspect.getsource(ComputerUseSkill._polite_media_request)
        assert "429" in source
        assert "time.sleep" in source, "a rate limit is an instruction to wait"

    def test_the_error_says_throttled_rather_than_missing(self, ComputerUseSkill):
        source = inspect.getsource(ComputerUseSkill._fetch_topic_image)
        assert "rate limiting, not a missing" in source
        assert '"rate_limited": throttled' in source


class TestNothingIsHardcodedPerTopic:
    def test_no_topic_names_appear_in_the_fetch_path(self, ComputerUseSkill):
        """Bryan: "she should be able to do this with literally an image. Not
        hardcoded ones." The chain derives every topic the same way."""
        import ast
        import textwrap

        for method in (
            ComputerUseSkill._fetch_topic_image,
            ComputerUseSkill._image_topic_candidates,
            ComputerUseSkill._commons_image_candidate,
        ):
            source = inspect.getsource(method)
            # Executable code only: prose may name examples (that is what the
            # docstrings are for); a branch on a topic name is the defect.
            tree = ast.parse(textwrap.dedent(source))
            literals = {
                node.value.lower()
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
            docstrings = {
                ast.get_docstring(node) or ""
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            literals -= {text.lower() for text in docstrings if text}
            for topic in ("rock", "orca", "grizzly", "sunset", "tardigrade"):
                assert topic not in literals, (method.__name__, topic)


class TestTheFrameGovernor:
    """Bryan: "Aura lags a bit when I [screen record] and it makes it look
    kinda ugly." """

    def _js(self) -> str:
        import pathlib

        return pathlib.Path("interface/static/aura.js").read_text(encoding="utf-8")

    def _css(self) -> str:
        import pathlib

        return pathlib.Path("interface/static/aura.css").read_text(encoding="utf-8")

    def test_it_measures_the_symptom_not_the_cause(self):
        """A recorder, a slow machine and a busy generation all look the same
        from inside the page, and all three deserve the same response."""
        js = self._js()
        assert "auraFrameGovernor" in js
        assert "requestAnimationFrame" in js
        assert "perf-lean" in js

    def test_it_has_hysteresis(self):
        """Shedding on one bad frame would flicker the background, which looks
        worse than the lag."""
        js = self._js()
        assert "BAD_SAMPLES_TO_SHED" in js
        assert "GOOD_SAMPLES_TO_RESTORE" in js

    def test_a_hidden_tab_is_not_rescued(self):
        assert "document.hidden" in self._js()

    def test_lean_mode_sheds_the_expensive_work(self):
        css = self._css()
        assert "body.perf-lean .aurora-layer" in css
        assert "backdrop-filter: none !important" in css

    def test_reduced_motion_is_honoured_unconditionally(self):
        """The ambient blur must stop for anyone who asked the OS for less
        motion, whether or not the governor has fired."""
        css = self._css()
        blocks = css.split("@media (prefers-reduced-motion: reduce)")
        assert any(
            ".aurora-layer" in block[:500] and "animation: none" in block[:500]
            for block in blocks[1:]
        ), "no reduced-motion block covers the aurora layers"


class _DeadGateway:
    """A gateway that cannot answer, so only local derivation is exercised."""

    def request(self, *args, **kwargs):
        raise RuntimeError("offline")
