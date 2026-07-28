"""Enhancement failures may cost the enhancement, never the answer.

CP126, two findings in core/brain/conversational_amplifier.py.

    "Classifier import failure admits actions and verifiable tasks. If the
     action/reasoning classifier cannot import, the function falls through
     to True for any substantive user-origin text. This fail-open path can
     apply creative rewrites to imperative or high-stakes verifiable
     responses that the module says are excluded."

It was ``except ImportError: pass`` followed by ``return True``. The one
component that knows which turns are EXCLUDED could vanish, and the answer
became "amplify everything".

    "Fail-open claim excludes several unhandled failure points. Numeric
     coercion, select_best, feature extraction, and result serialization are
     outside protected blocks... These errors abort the caller instead of
     returning the draft."

The docstring said fail-open and the body implemented it in patches — every
model call was wrapped, ``int(n)``, ``float(time_budget_s)``, select_best
and extract_features were not. A bad budget or a raising taste model lost
the whole turn instead of returning the plain draft.

The two need opposite treatments, which is the point. Eligibility fails
CLOSED — without the classifier there is no way to know an imperative is not
being rewritten, and declining costs only a plainer reply. Amplification
fails OPEN at the boundary — the worst case is the draft the caller already
had.
"""
from __future__ import annotations

import pytest

from core.brain.conversational_amplifier import (
    ConversationResult,
    amplify_conversation,
    is_conversationally_amplifiable,
)

SUBSTANTIVE = "what do you think about the deployment plan"


async def _generate(_prompt: str, _temp: float) -> str:
    return "an alternative phrasing of the reply"


class TestEligibilityFailsClosed:
    def test_a_missing_classifier_declines_amplification(self, monkeypatch):
        """The defect: import failure admitted everything."""
        import builtins

        real_import = builtins.__import__

        def _no_classifier(name, *args, **kwargs):
            if name == "core.brain.reasoning_amplifier_v2":
                raise ImportError("classifier gone")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_classifier)
        assert is_conversationally_amplifiable(SUBSTANTIVE, "user") is False

    def test_a_raising_classifier_declines_amplification(self, monkeypatch):
        import core.brain.reasoning_amplifier_v2 as classifier

        def _boom(*_args, **_kwargs):
            raise RuntimeError("classifier exploded")

        monkeypatch.setattr(classifier, "is_action_request", _boom)
        assert is_conversationally_amplifiable(SUBSTANTIVE, "user") is False

    def test_a_healthy_classifier_still_admits_conversation(self):
        """Declining everything would disable the feature silently."""
        assert is_conversationally_amplifiable(SUBSTANTIVE, "user") is True

    def test_non_user_origins_are_still_excluded(self):
        assert is_conversationally_amplifiable(SUBSTANTIVE, "autonomous") is False

    def test_short_turns_are_still_excluded(self):
        assert is_conversationally_amplifiable("hi", "user") is False


class TestAmplificationFailsOpen:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"time_budget_s": "not-a-number"},   # numeric coercion
            {"n": None},                          # numeric coercion
            {"n": float("nan")},
            {"time_budget_s": None},
        ],
    )
    async def test_bad_arguments_return_the_draft(self, kwargs):
        result = await amplify_conversation(
            "my draft answer", generate=_generate, user_message=SUBSTANTIVE, **kwargs,
        )
        assert result.answer == "my draft answer"

    @pytest.mark.asyncio
    async def test_a_raising_generator_returns_the_draft(self):
        async def _broken(_prompt, _temp):
            raise OSError("connection reset")

        result = await amplify_conversation(
            "my draft answer", generate=_broken, user_message=SUBSTANTIVE,
        )
        assert result.answer == "my draft answer"

    @pytest.mark.asyncio
    async def test_a_raising_taste_model_returns_the_draft(self, monkeypatch):
        """select_best was outside every protected block."""
        import core.brain.conversational_amplifier as mod

        def _boom(*_args, **_kwargs):
            raise RuntimeError("taste model exploded")

        monkeypatch.setattr(mod, "select_best", _boom)
        result = await amplify_conversation(
            "my draft answer", generate=_generate, user_message=SUBSTANTIVE,
        )
        assert result.answer == "my draft answer"

    @pytest.mark.asyncio
    async def test_raising_feature_extraction_returns_the_draft(self, monkeypatch):
        import core.brain.conversational_amplifier as mod

        def _boom(*_args, **_kwargs):
            raise ValueError("features exploded")

        monkeypatch.setattr(mod, "extract_features", _boom)
        result = await amplify_conversation(
            "my draft answer", generate=_generate, user_message=SUBSTANTIVE,
        )
        assert result.answer == "my draft answer"

    @pytest.mark.asyncio
    async def test_it_always_returns_a_result_object(self):
        result = await amplify_conversation(
            "", generate=_generate, user_message=SUBSTANTIVE, n="bad",
        )
        assert isinstance(result, ConversationResult)


class TestCancellationIsNotAbsorbed:
    @pytest.mark.asyncio
    async def test_cancellation_propagates(self):
        """Cancellation is the caller's decision, not a failure to swallow —
        absorbing it would turn a shutdown into a hung turn."""
        import asyncio

        async def _cancelled(_prompt, _temp):
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await amplify_conversation(
                "draft", generate=_cancelled, user_message=SUBSTANTIVE, revise=False,
            )


class TestTheHappyPathStillAmplifies:
    @pytest.mark.asyncio
    async def test_a_healthy_run_produces_candidates(self):
        result = await amplify_conversation(
            "my draft answer",
            generate=_generate,
            user_message=SUBSTANTIVE,
            n=2,
            time_budget_s=5.0,
            revise=False,
        )
        assert result.n_candidates >= 2
