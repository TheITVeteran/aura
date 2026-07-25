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


class TestSelfCodeMutationHasRealGates:
    """``1cdbdb14`` — a caller-supplied example list was the only gate before
    mutating Aura's own source. Examples prove a function returns the right
    values for cases someone thought of; they say nothing about whether the
    file still imports or whether the change smuggled in a new capability."""

    ORIGINAL = "def add(a, b):\n    return a - b\n"
    CANDIDATE = "def add(a, b):\n    return a + b\n"
    FILE = "def add(a, b):\n    return a - b\n\n\ndef other():\n    return 1\n"

    def _blockers(self, candidate=None, original=None, checks=None):
        from core.capabilities.self_code_improver import _promotion_blockers

        return _promotion_blockers(
            original_src=original or self.ORIGINAL,
            candidate_src=candidate or self.CANDIDATE,
            file_before=self.FILE,
            func_name="add",
            checks=checks if checks is not None else [{"args": [], "expected": 0}] * 3,
        )

    def test_a_sound_candidate_passes(self):
        assert self._blockers() == []

    def test_a_token_example_list_cannot_mutate_source(self):
        blockers = self._blockers(checks=[{"args": [], "expected": 0}])
        assert any("insufficient_evidence" in b for b in blockers)

    def test_a_candidate_that_does_not_parse_is_blocked(self):
        blockers = self._blockers(candidate="def add(a, b):\n    return a +\n")
        assert any("candidate_does_not_parse" in b for b in blockers)

    def test_a_replacement_that_breaks_the_file_is_blocked(self):
        """A function can parse alone and still land badly in the file."""
        from core.capabilities.self_code_improver import _promotion_blockers

        blockers = _promotion_blockers(
            original_src=self.ORIGINAL,
            candidate_src=self.CANDIDATE,
            file_before="def add(a, b):\n    return a - b\n\ndef broken(:\n",
            func_name="add",
            checks=[{"args": [], "expected": 0}] * 3,
        )
        assert any("does_not_compile" in b or "not_extractable" in b for b in blockers)

    def test_a_newly_introduced_dangerous_capability_is_blocked(self):
        candidate = "def add(a, b):\n    import subprocess\n    return a + b\n"
        blockers = self._blockers(candidate=candidate)
        assert any("introduces_dangerous_capability" in b for b in blockers)
        assert any("subprocess" in b for b in blockers)

    def test_a_pre_existing_capability_is_not_relitigated(self):
        """The function already had it; a bug fix is not the place to argue."""
        candidate = "def add(a, b):\n    import subprocess\n    return a + b\n"
        original = "def add(a, b):\n    import subprocess\n    return a - b\n"
        assert self._blockers(candidate=candidate, original=original) == []

    def test_the_enactment_path_consults_the_gates(self):
        import inspect

        from core.capabilities import self_code_improver as sci

        source = inspect.getsource(sci.improve_function)
        assert "_promotion_blockers(" in source
        assert "promotion_blocked" in source
        # The refusal must precede the write.
        assert source.index("_promotion_blockers(") < source.index("_replace_function(src")


class TestRollbackReportsWhatItAchieved:
    """``8f695a21`` — rollback replaced only the named function and compared
    STRIPPED function text, while the docstring promised a byte-for-byte
    pre-image and the ledger's file_sha_before went unused."""

    def _source(self):
        import inspect

        from core.capabilities import self_code_improver as sci

        return inspect.getsource(sci.rollback_enactment)

    def test_the_whole_file_pre_image_is_checked(self):
        source = self._source()
        assert 'record.get("file_sha_before")' in source

    def test_exact_and_equivalent_are_distinguished(self):
        source = self._source()
        assert "function_pre_image_exact" in source
        assert "function_pre_image_equivalent" in source

    def test_a_function_only_rollback_says_so(self):
        source = self._source()
        assert "rolled_back_function_only" in source
        assert "rolled_back_exact" in source

    def test_residual_drift_is_surfaced(self):
        source = self._source()
        assert '"residual_drift"' in source
