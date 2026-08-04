"""A random projection over 32 words could be served as her answer.

``SubstrateTokenGenerator`` is documented as "a learned-readout style head over
the live substrate state". ``_ensure_readout`` is::

    rng = np.random.default_rng(self.seed + state_dim * 31 + self._vocab_size)
    self._readout = rng.standard_normal((vocab, state_dim)) / sqrt(state_dim)

— seeded, never trained — projecting onto ``PROTO_TOKENS``, 32 words. What
comes out is

    "Substrate path: world action hold grounded choose loop result repair."

which is a deterministic fingerprint of substrate state and useful as one. It
is not language.

MEASURED 2026-08-04: this was reachable as a LIVE USER-FACING REPLY.
``AURA_SUBSTRATE_PRIMARY_USER`` defaults to "1", the threshold is 0.34, and a
short prompt whose hashed vector aligns with the live state reaches a
prediction error of 0.157 — comfortably under. The router returned
``result.text`` on any turn where ``used_substrate`` was true.

So the two questions are now separate: ``used_substrate`` says the path ran,
``is_user_presentable`` says whether its output may be shown to a person. The
proto vocabulary answers no to the second, forever, until a trained head over
the model vocabulary exists.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.brain.llm.substrate_token_generator import (
    PROTO_TOKENS,
    SubstrateTokenGenerator,
)


class _FakeSubstrate:
    def __init__(self, x):
        self.x = np.asarray(x, dtype=np.float32)


def _aligned_generator(prompt: str) -> SubstrateTokenGenerator:
    """A substrate whose state aligns with ``prompt`` — the reachable case."""
    probe = SubstrateTokenGenerator(_FakeSubstrate(np.zeros(64, dtype=np.float32)))
    direction = probe._prompt_vector(prompt, dim=64)
    return SubstrateTokenGenerator(_FakeSubstrate(direction * 3.0))


def test_the_readout_head_is_untrained():
    """Named for what it is: nothing fits this matrix to anything."""
    generator = SubstrateTokenGenerator(_FakeSubstrate(np.zeros(64, dtype=np.float32)))
    first = generator._ensure_readout(64)
    second = SubstrateTokenGenerator(_FakeSubstrate(np.zeros(64))). _ensure_readout(64)
    assert np.allclose(first, second), "a seeded projection, identical every time"
    assert first.shape[1] == 64


def test_the_vocabulary_is_thirty_two_words():
    assert len(PROTO_TOKENS) == 32


def test_the_path_still_fires_and_that_is_the_point():
    """The defect was not that it runs; it is that its text was served."""
    generator = _aligned_generator("go")
    result = generator.generate("go")
    assert result.used_substrate is True
    assert result.prediction_error < result.threshold
    assert result.text.startswith("Substrate path:")


def test_but_its_text_is_not_presentable():
    generator = _aligned_generator("go")
    result = generator.generate("go")
    assert result.vocabulary == "proto"
    assert result.is_user_presentable is False, (
        f"{result.text!r} was cleared for a person to read"
    )


def test_a_deferred_generation_is_not_presentable_either():
    generator = SubstrateTokenGenerator(
        _FakeSubstrate(np.random.default_rng(0).standard_normal(64) * 0.3)
    )
    result = generator.generate("what is the capital of france")
    assert result.used_substrate is False
    assert result.is_user_presentable is False


class TestTheRouterDefers:
    @pytest.mark.asyncio
    async def test_a_user_turn_does_not_receive_proto_tokens(self, monkeypatch):
        from core.brain.llm import llm_router

        generator = _aligned_generator("go")
        result = generator.generate("go")
        assert result.used_substrate and result.text.strip()

        router = llm_router.IntelligentLLMRouter.__new__(llm_router.IntelligentLLMRouter)
        router.stats = {}
        router.last_tier = ""
        router.last_user_tier = ""

        monkeypatch.setattr(
            llm_router.IntelligentLLMRouter, "_substrate_primary_enabled", staticmethod(lambda: True)
        )
        monkeypatch.setattr(
            llm_router.IntelligentLLMRouter,
            "_substrate_user_facing_enabled",
            staticmethod(lambda: True),
        )

        from core.container import ServiceContainer

        monkeypatch.setattr(
            ServiceContainer,
            "get",
            staticmethod(
                lambda name, default=None: (
                    _FakeSubstrate(np.zeros(64))
                    if name == "continuous_substrate"
                    else default
                )
            ),
        )
        monkeypatch.setattr(
            "core.brain.llm.substrate_token_generator.get_substrate_token_generator",
            lambda _substrate: generator,
        )

        served = await router._try_substrate_primary("go", {}, is_background=False)
        assert served is None, f"a person was handed {served!r}"

    @pytest.mark.asyncio
    async def test_background_work_still_gets_the_fingerprint(self, monkeypatch):
        """Background and evaluation want exactly this deterministic readout."""
        from core.brain.llm import llm_router

        generator = _aligned_generator("go")
        router = llm_router.IntelligentLLMRouter.__new__(llm_router.IntelligentLLMRouter)
        router.stats = {}
        router.last_tier = ""
        router.last_user_tier = ""

        monkeypatch.setattr(
            llm_router.IntelligentLLMRouter, "_substrate_primary_enabled", staticmethod(lambda: True)
        )

        from core.container import ServiceContainer

        monkeypatch.setattr(
            ServiceContainer,
            "get",
            staticmethod(
                lambda name, default=None: (
                    _FakeSubstrate(np.zeros(64))
                    if name == "continuous_substrate"
                    else default
                )
            ),
        )
        monkeypatch.setattr(
            "core.brain.llm.substrate_token_generator.get_substrate_token_generator",
            lambda _substrate: generator,
        )

        served = await router._try_substrate_primary("go", {}, is_background=True)
        assert served is not None
        assert served.startswith("Substrate path:")
