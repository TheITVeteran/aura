"""CP126: absent governance must restrain, not expand, authority.

* ``237ae4c2`` — failure to inspect active LoRA processes recorded a
  degradation and CONTINUED feeding self-training data, so governance
  infrastructure being absent or broken expanded training authority.
* ``e0e64fb2`` — the escape ladder documents four rungs but defaulted to a
  three-attempt cap, so the last resort could never run.
"""
from __future__ import annotations

import inspect

import pytest


class TestSelfTrainingFailsClosed:
    def test_an_unobservable_governor_blocks_the_feed(self):
        """Not knowing whether a LoRA run is active is not permission to
        start another; it is precisely the state in which to wait."""
        from core.brain import reasoning_self_improvement as rsi

        source = inspect.getsource(rsi.ReasoningSelfImprovement.maybe_improve)
        block = source.split("reasoning_self_improvement_governor", 1)[1]
        assert "blocked_governor_unavailable" in block
        # The refusal must RETURN, not fall through into the feed.
        assert block.index("return {") < block.index("fed_fn")

    def test_the_refusal_is_recorded_with_its_reason(self):
        from core.brain import reasoning_self_improvement as rsi

        source = inspect.getsource(rsi.ReasoningSelfImprovement.maybe_improve)
        assert "refused to feed self-training while LoRA activity was unobservable" in source

    def test_verifier_admission_still_fails_closed(self):
        """The other half of the finding, verified still correct."""
        from core.brain import reasoning_self_improvement as rsi

        source = inspect.getsource(rsi.ReasoningSelfImprovement._domain_admitted)
        # Both the absent-Foundry and the exception path return False.
        assert source.count("return False") >= 2
        assert "return True" not in source

    def test_an_active_lora_run_still_blocks(self):
        from core.brain import reasoning_self_improvement as rsi

        source = inspect.getsource(rsi.ReasoningSelfImprovement.maybe_improve)
        assert "blocked_existing_training" in source


class TestEveryDocumentedRungIsReachable:
    def test_the_default_cap_covers_the_whole_ladder(self):
        from core.brain.llm.latent_cortex.escape import ESCAPE_RUNGS, EscapeConfig

        assert EscapeConfig().max_attempts >= len(ESCAPE_RUNGS)

    def test_the_last_resort_rung_exists(self):
        from core.brain.llm.latent_cortex.escape import ESCAPE_RUNGS

        assert ESCAPE_RUNGS[-1] == "matched_perturbation"

    def test_the_cap_is_derived_from_the_rungs_not_hardcoded(self):
        """A hardcoded cap drifts away from the ladder it is meant to cover."""
        from core.brain.llm.latent_cortex import escape

        source = inspect.getsource(escape)
        assert "max_attempts: int = len(ESCAPE_RUNGS)" in source

    def test_an_explicit_truncation_is_still_allowed(self):
        from core.brain.llm.latent_cortex.escape import EscapeConfig

        assert EscapeConfig(max_attempts=2).max_attempts == 2
