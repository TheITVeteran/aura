"""Aura's private thoughts must not enter the conversation as her own turns.

LIVE DEFECT, 2026-07-25. A user said "Just checking in" and got back an
unprompted monologue about ghosts — "leftover bits from other people's
experiences" — followed by the model inventing "<dispatch a somatic probe>"
as if it were speech. The next user message was "Ghosts? what're you talking
about".

Nothing was hallucinated out of nowhere. personhood_engine._emit_thought
writes spontaneous thoughts into ``cognition.working_memory`` as
``role="assistant"`` with ``origin="spontaneous"``. The context assembler
filtered background material by origin, and its prefix list contained
``"spontaneous:"`` — with a colon. ``"spontaneous".startswith("spontaneous:")``
is False, so every spontaneous thought reached the conversational prompt as
one of Aura's own prior TURNS. The model read them as shared context and
continued that voice. ``somatic_noise`` and ``baseline_continuity``, both
global-workspace winners, arrived the same way.

One character of vocabulary drift between the writer and the filter.

The real lesson is that a hand-maintained string list silently diverged from
the code that writes those strings, and the symptom was incoherent speech
rather than a failing test. These tests pin both directions so the next
divergence fails here instead of in front of the user.
"""
from __future__ import annotations

import inspect
import re

import pytest

from core.brain.llm.context_assembler import ContextAssembler


def _classifier():
    """The assembler's own background vocabulary, read from its source."""
    source = inspect.getsource(ContextAssembler)
    sources_block = re.search(r"background_sources = \{(.*?)\n        \}", source, re.S)
    prefixes_block = re.search(r"background_prefixes = \((.*?)\)", source, re.S)
    assert sources_block and prefixes_block, "background vocabulary not found"
    sources = {
        line.strip().strip('",')
        for line in sources_block.group(1).split("\n")
        if line.strip().startswith('"')
    }
    prefixes = tuple(
        line.strip().strip('",')
        for line in prefixes_block.group(1).split("\n")
        if line.strip().startswith('"')
    )

    def is_background(origin: str) -> bool:
        value = str(origin or "").strip().lower()
        return value in sources or any(value.startswith(p) for p in prefixes)

    return is_background


# Origins that represent Aura thinking, not Aura speaking to someone.
INTERNAL_ORIGINS = (
    "spontaneous",
    "spontaneous:idle",
    "somatic_noise",
    "baseline_continuity",
    "drive_growth",
    "drive_social",
    "mind_tick",
    "mind_tick_fallback",
    "autonomous_thought",
    "autonomous_volition",
    "subconscious_dream",
    "reflection_impulse",
    "impulse",
    "intention_loop",
    "memory_consolidation",
    "cognitive_background",
    "proactive_presence",
    "agency_core",
)

# Origins that ARE real dialogue and must survive filtering, or the model
# loses the conversation it is supposed to be having.
DIALOGUE_ORIGINS = (
    "user",
    "desktop_quick_user",
    "voice",
    "admin",
    "api",
)


class TestInternalThoughtsAreFilteredFromTheConversation:
    @pytest.mark.parametrize("origin", INTERNAL_ORIGINS)
    def test_internal_origins_do_not_reach_the_prompt(self, origin):
        assert _classifier()(origin) is True, origin

    def test_the_exact_origin_the_writer_emits_is_covered(self):
        """The regression, stated as the writer states it.

        personhood_engine writes the bare string "spontaneous". If this ever
        fails again, Aura will start answering greetings with her own inner
        monologue.
        """
        from core.autonomy import personhood_engine

        source = inspect.getsource(personhood_engine.PersonhoodEngine._emit_thought)
        emitted = re.search(r'"origin":\s*"([^"]+)"', source)
        assert emitted, "could not find the origin this writer emits"
        assert _classifier()(emitted.group(1)) is True, emitted.group(1)

    def test_spontaneous_thoughts_are_written_as_assistant_turns(self):
        """Documents WHY filtering matters: they are indistinguishable from
        speech once they are in working memory."""
        from core.autonomy import personhood_engine

        source = inspect.getsource(personhood_engine.PersonhoodEngine._emit_thought)
        assert '"role": "assistant"' in source
        assert "working_memory.append" in source


class TestRealDialogueSurvives:
    @pytest.mark.parametrize("origin", DIALOGUE_ORIGINS)
    def test_dialogue_origins_are_never_filtered(self, origin):
        """Over-filtering is the opposite failure and just as bad: it erases
        the conversation the reply is supposed to be about."""
        assert _classifier()(origin) is False, origin

    def test_an_unknown_origin_is_treated_as_dialogue(self):
        """Unknown origins default to visible. That is the right default for
        a conversation surface — but it is exactly why the vocabulary must
        be kept in step with the writers, since a NEW internal origin will
        leak until it is listed here."""
        assert _classifier()("some_new_surface") is False


class TestTheVocabularyCannotSilentlyDrift:
    def test_every_working_memory_writer_origin_is_classified(self):
        """The actual defect was drift between writer and filter.

        Any literal origin written into working_memory anywhere in core/
        must be a string this classifier has an opinion about — either
        internal (filtered) or dialogue (kept). A new internal origin that
        nobody adds here fails this test instead of reaching the user.
        """
        import pathlib

        root = pathlib.Path(inspect.getfile(ContextAssembler)).resolve()
        core_dir = root.parent.parent.parent  # core/
        known = set(INTERNAL_ORIGINS) | set(DIALOGUE_ORIGINS)
        unclassified: list[str] = []

        for path in core_dir.rglob("*.py"):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for block in re.finditer(
                r"working_memory\.append\((.{0,400}?)\)", text, re.S,
            ):
                origin = re.search(r'"origin":\s*"([a-z0-9_:.]+)"', block.group(1))
                if origin and origin.group(1) not in known:
                    unclassified.append(f"{path.name}: {origin.group(1)}")

        assert not unclassified, (
            "working-memory origins not classified as internal or dialogue: "
            + "; ".join(sorted(set(unclassified)))
        )


class TestTheLiveStateResolverReturnsAState:
    """LIVE DEFECT, 2026-07-25. ``_resolve_live_aura_state`` returned
    whatever was registered under "aura_state" without checking its type. On
    a live boot that was a StateRepository, so every caller hit ``.cognition``
    on it:

        AttributeError: 'StateRepository' object has no attribute 'cognition'

    That crashed the required-search contract, which is why "search for an
    article on how LeBron James will fit in with the 76ers" produced no
    search at all — the capability did not decline, it raised.
    """

    def _state(self):
        from core.state.aura_state import AuraState

        return AuraState.default()

    def test_a_real_state_passes_through(self):
        from interface.routes.chat import _unwrap_state

        state = self._state()
        assert _unwrap_state(state) is state

    def test_a_repository_is_unwrapped_to_its_state(self):
        from interface.routes.chat import _unwrap_state

        state = self._state()

        class _Repo:
            _current = state

        assert _unwrap_state(_Repo()) is state

    def test_a_repository_is_not_mistaken_for_a_state(self):
        from interface.routes.chat import _looks_like_aura_state

        class _Repo:
            _current = None

        assert _looks_like_aura_state(_Repo()) is False

    def test_an_unusable_object_resolves_to_none(self):
        from interface.routes.chat import _unwrap_state

        assert _unwrap_state(object()) is None
        assert _unwrap_state(None) is None

    def test_the_resolver_validates_every_source(self):
        import inspect

        from interface.routes import chat as mod

        source = inspect.getsource(mod._resolve_live_aura_state)
        # No raw registry value may be returned unchecked.
        assert 'ServiceContainer.get("aura_state", default=None)' in source
        assert "_unwrap_state(ServiceContainer.get" in source
        assert 'getattr(repo, "_current", None) if repo is not None else None' not in source


class TestBodyPressureActuallyReads:
    """LIVE DEFECT, 2026-07-25. ``total_pressure`` is a @property, and the
    caller invoked it as ``total_pressure()`` — so every call raised
    ``TypeError: 'float' object is not callable`` and fell into the handler,
    which answered 0.0. On this scale 0.0 means MAXIMUM headroom, so the
    latent cortex ran undamped and never once saw a real reading.

    It surfaced only because the handler was changed to record itself.
    """

    def test_total_pressure_is_a_property_not_a_method(self):
        from core.being.aura_now import BodyState

        assert isinstance(
            inspect.getattr_static(BodyState, "total_pressure"), property,
        )

    def test_the_caller_reads_it_as_a_property(self):
        from core.brain import latent_cortex_service as mod

        source = inspect.getsource(mod.LatentCortexService._body_pressure)
        assert ".total_pressure)" in source
        assert ".total_pressure()" not in source

    def test_a_real_reading_comes_back(self):
        from core.being.aura_now import BodyState

        value = float(BodyState.from_aura_state(None).total_pressure)
        assert 0.0 <= value <= 1.0
