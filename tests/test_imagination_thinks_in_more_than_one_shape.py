"""Imagination must not break every idea down the same way.

LIVE FEEDBACK, 2026-07-25. Bryan, on the NOVEL THOUGHTS panel: "it always
breaks things down in the same way 'what if - is not the object' and the
others. maybe that should be more varied or left for aura to decide in real
time?"

He was right, and the cause was structural rather than stylistic. Novel
thoughts were two f-strings with the top two keywords slotted in:

    f"What if {focus} is not the object, but the lens for seeing {secondary}?"
    f"The useful novelty may be the smallest testable form of {focus}..."

So every frame Aura had ever produced was the same two cognitive moves in
different nouns. Rewording them would have changed nothing — the *thinking*
was identical. What varies now is the move itself, selected against her
measured internal state, with recently-used shapes suppressed.

A second defect surfaced while testing the first: the nouns were wrong.
Keywords were taken in raw text order, so the leading content word won — and
in English that is usually the verb. "how should I design the deployment
pipeline" imagined about "should"; "search the web and verify the 76ers
roster" imagined about "search"; and "76ers" could not be a keyword at all,
because the word pattern demanded a leading letter. Structurally sound
thoughts pointed at the wrong subject, which reads to a person as the whole
feature being generic.
"""
from __future__ import annotations

import pytest

from core.brain.imagination import (
    _COUNTERFACTUAL_MOVES,
    _NOVEL_MOVES,
    ImaginationEngine,
    _extract_keywords,
)


def _engine():
    return ImaginationEngine()


class TestTheSubjectIsTheSubject:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("how should I design the deployment pipeline", "deployment"),
            ("search the web and verify the 76ers roster", "76ers"),
            ("what if consciousness is just compression", "consciousness"),
            ("invent a new way to teach fractions to kids", "fractions"),
        ],
    )
    def test_the_leading_keyword_is_what_the_request_is_about(self, text, expected):
        assert _extract_keywords(text)[0] == expected

    def test_digit_initial_topics_are_visible_at_all(self):
        """The old pattern required a leading letter; "76ers" was invisible."""
        assert "76ers" in _extract_keywords("the 76ers roster")

    def test_a_bare_number_is_not_a_topic(self):
        assert "2026" not in _extract_keywords("ship it in 2026 with the scheduler")

    def test_proper_nouns_outrank_common_words(self):
        assert _extract_keywords("tell LeBron about the roster")[0] == "lebron"

    def test_weak_verbs_are_demoted_not_deleted(self):
        """"search" is a real subject when someone asks about search."""
        keywords = _extract_keywords("how should I design the deployment pipeline")
        assert "should" in keywords
        assert keywords.index("should") > keywords.index("deployment")

    def test_a_request_about_a_weak_word_still_finds_it(self):
        assert "search" in _extract_keywords("make search faster")


class TestOneStateProducesManyShapes:
    def test_a_single_frame_uses_distinct_moves(self):
        frame = _engine().imagine("how should I design the deployment pipeline")
        assert len(set(frame.novel_thoughts)) == len(frame.novel_thoughts)

    def test_consecutive_frames_do_not_repeat_the_same_shapes(self):
        """The literal complaint: the same breakdown, every time.

        Two frames on the same subject must not be the same four thoughts.
        """
        engine = _engine()
        first = engine.imagine("how should I design the deployment pipeline")
        second = engine.imagine("how should I design the deployment pipeline")
        assert set(first.novel_thoughts) != set(second.novel_thoughts)

    def test_no_single_opening_dominates_a_long_run(self):
        """Across many subjects, no one template may own the output.

        The old code scored 1.0 here for "What if ... is not the object".
        """
        engine = _engine()
        subjects = [
            "the deployment pipeline",
            "consciousness and compression",
            "teaching fractions to kids",
            "the 76ers roster",
            "why the tests keep failing",
            "a quieter notification design",
            "memory consolidation during sleep",
            "the cost of a rewrite",
        ]
        openings = [
            thought.split()[0] + " " + thought.split()[1]
            for subject in subjects
            for thought in engine.imagine(subject).novel_thoughts
        ]
        most_common = max(openings.count(o) for o in set(openings))
        assert most_common / len(openings) < 0.34, "one shape dominates the output"

    def test_the_old_template_is_no_longer_mandatory(self):
        engine = _engine()
        frames = [
            engine.imagine(s)
            for s in ("the deployment pipeline", "why tests fail", "quiet design")
        ]
        assert any(
            not any("is not the object" in t for t in f.novel_thoughts) for f in frames
        )


class TestTheStateChoosesTheThinking:
    def _thoughts(self, text):
        return " ".join(_engine().imagine(text).novel_thoughts).lower()

    def test_a_creative_request_reaches_for_creative_moves(self):
        text = self._thoughts("invent an original design combining two ideas")
        assert any(
            marker in text
            for marker in ("opposing pressure", "made of something else", "already solved the shape")
        )

    def test_a_verification_request_reaches_for_evidence_moves(self):
        text = self._thoughts("verify with real tools whether the roster claim is true")
        assert any(
            marker in text
            for marker in ("should i be seeing", "smallest testable", "stops being true")
        )

    def test_the_same_subject_thinks_differently_under_different_affect(self):
        """Interoception is causal here, not decorative."""
        from types import SimpleNamespace

        def state(**emotions):
            return SimpleNamespace(affect=SimpleNamespace(emotions=emotions, curiosity=0.0))

        curious = _engine().imagine("the deployment pipeline", state=state(curiosity=0.95))
        strained = _engine().imagine("the deployment pipeline", state=state(frustration=0.95))
        assert set(curious.novel_thoughts) != set(strained.novel_thoughts)


class TestDeterminismSurvivesVariety:
    def test_a_fresh_engine_reproduces_the_same_frame(self):
        """Varied is not random. Same input, same state, same output — or
        imagination stops being debuggable."""
        assert (
            _engine().imagine("the deployment pipeline").novel_thoughts
            == _engine().imagine("the deployment pipeline").novel_thoughts
        )

    def test_probes_and_thoughts_do_not_read_as_a_matched_pair(self):
        frame = _engine().imagine("the deployment pipeline")
        assert not (set(frame.novel_thoughts) & set(frame.counterfactuals))


class TestAuraMayWriteHerOwnThoughts:
    """The other half of the ask: "or left for aura to decide in real time".

    ``imagine`` is synchronous and side-effect free by contract, so it cannot
    call the model itself. A caller that already has model access passes
    authored lines in, and they win.
    """

    def test_authored_thoughts_take_precedence(self):
        frame = _engine().imagine(
            "the deployment pipeline",
            context={"authored_novel_thoughts": ["I keep wanting to rebuild it."]},
        )
        assert frame.novel_thoughts == ["I keep wanting to rebuild it."]

    def test_authored_probes_take_precedence(self):
        frame = _engine().imagine(
            "the deployment pipeline",
            context={"authored_counterfactuals": ["What if nobody deploys?"]},
        )
        assert frame.counterfactuals == ["What if nobody deploys?"]

    def test_the_registry_is_the_floor_not_the_fallback(self):
        """Empty or unusable authored input must not empty the frame."""
        for bad in ([], "", None, ["   "], 5):
            frame = _engine().imagine(
                "the deployment pipeline",
                context={"authored_novel_thoughts": bad},
            )
            assert frame.novel_thoughts, bad


class TestTheRegistryIsCoherent:
    def test_move_ids_are_unique(self):
        for moves in (_NOVEL_MOVES, _COUNTERFACTUAL_MOVES):
            ids = [m.move_id for m in moves]
            assert len(ids) == len(set(ids))

    def test_there_are_enough_shapes_to_avoid_a_house_style(self):
        assert len(_NOVEL_MOVES) >= 12
        assert len(_COUNTERFACTUAL_MOVES) >= 8

    def test_every_move_renders_a_sentence(self):
        for moves in (_NOVEL_MOVES, _COUNTERFACTUAL_MOVES):
            for move in moves:
                rendered = move.render("alpha", "beta", {})
                assert rendered.strip() and rendered.strip()[-1] in ".?"

    def test_no_move_leaks_filler_when_there_is_one_keyword(self):
        """Single-noun input must not produce "its constraint" prose — the
        moves that need two subjects simply do not fire."""
        frame = _engine().imagine("compression")
        assert not any("its constraint" in t for t in frame.novel_thoughts)
        assert not any("its constraint" in t for t in frame.counterfactuals)

    def test_a_broken_move_cannot_silence_imagination(self, monkeypatch):
        import core.brain.imagination as mod

        def explode(*_args, **_kwargs):
            raise RuntimeError("bad move")

        broken = mod._ThoughtMove("broken", explode, lambda _g: 99.0)
        monkeypatch.setattr(mod, "_NOVEL_MOVES", (broken, *_NOVEL_MOVES))
        frame = _engine().imagine("the deployment pipeline")
        assert frame.novel_thoughts
        assert all("bad move" not in t for t in frame.novel_thoughts)
