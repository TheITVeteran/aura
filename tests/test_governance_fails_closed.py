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


class TestCacheKeysCarryTheirContext:
    """``2e31dd71`` — the key was task_type plus the objective, so the same
    words asked under a different interpreter, model or verifier hit the SAME
    entry, replaying a stale or cross-context answer as verified truth."""

    def test_the_same_question_in_a_different_context_is_a_different_key(self, monkeypatch):
        from core.brain import reasoning_solved_cache as rsc

        monkeypatch.setenv("AURA_ACTIVE_MODEL_ID", "model-a")
        key_a = rsc._problem_key("what is 6 times 7", "math")
        monkeypatch.setenv("AURA_ACTIVE_MODEL_ID", "model-b")
        key_b = rsc._problem_key("what is 6 times 7", "math")
        assert key_a != key_b

    def test_the_same_context_is_stable(self, monkeypatch):
        from core.brain import reasoning_solved_cache as rsc

        monkeypatch.setenv("AURA_ACTIVE_MODEL_ID", "model-a")
        assert rsc._problem_key("q", "math") == rsc._problem_key("q", "math")

    def test_the_fingerprint_includes_the_interpreter(self):
        import sys

        from core.brain import reasoning_solved_cache as rsc

        source = inspect.getsource(rsc._context_fingerprint)
        assert "sys.version_info" in source
        assert isinstance(rsc._context_fingerprint(), str)
        assert sys.version_info.major  # the input actually exists

    def test_the_key_schema_can_invalidate_everything(self):
        from core.brain import reasoning_solved_cache as rsc

        assert rsc._CACHE_KEY_SCHEMA >= 2

    def test_different_task_types_do_not_collide(self):
        from core.brain import reasoning_solved_cache as rsc

        assert rsc._problem_key("q", "math") != rsc._problem_key("q", "code")


class TestLedgerVerificationDoesNotBlockTheLoop:
    """``750942aa`` — flush_ledger polls with time.sleep and verify_ledger
    parses the entire event file synchronously; on the event loop that is an
    unbounded stall, the failure mode that once froze this runtime."""

    def test_async_variants_exist(self):
        from core.brain.verifiers.foundry import VerifierFoundry

        assert callable(getattr(VerifierFoundry, "flush_ledger_async", None))
        assert callable(getattr(VerifierFoundry, "verify_ledger_async", None))

    def test_the_async_flush_yields_the_loop(self):
        from core.brain.verifiers.foundry import VerifierFoundry

        source = inspect.getsource(VerifierFoundry.flush_ledger_async)
        # Body only: the docstring necessarily names time.sleep while
        # explaining what it replaced.
        body = source.split('"""', 2)[-1]
        assert "await asyncio.sleep" in body
        # The CALL is what would block; the word may appear in a comment
        # contrasting the two.
        assert "time.sleep(" not in body

    def test_the_async_verify_offloads_the_parse(self):
        from core.brain.verifiers.foundry import VerifierFoundry

        source = inspect.getsource(VerifierFoundry.verify_ledger_async)
        assert "asyncio.to_thread" in source
        assert "await self.flush_ledger_async()" in source

    def test_the_parse_is_separable_from_the_flush(self):
        """The thread-offloadable half must not itself flush."""
        from core.brain.verifiers.foundry import VerifierFoundry

        source = inspect.getsource(VerifierFoundry._verify_ledger_locked)
        assert "flush_ledger" not in source

    def test_the_sync_path_still_works(self):
        from core.brain.verifiers.foundry import VerifierFoundry

        source = inspect.getsource(VerifierFoundry.verify_ledger)
        assert "self.flush_ledger()" in source
        assert "self._verify_ledger_locked()" in source


class TestVerifierReceiptsAreAttestationsNotCopies:
    """``e404e00c`` — every receipt field was checked by comparing it to the
    corresponding trial field. That proves the two objects hold the same
    bytes — duplicate storage — and nothing about whether execution or
    scoring happened. A producer could emit the trial twice and pass."""

    def _trial(self):
        return {
            "trial_id": "t1",
            "task_payload_sha256": "a" * 64,
            "scorer_config_sha256": "b" * 64,
            "treatment_output_sha256": "c" * 64,
            "control_output_sha256": "d" * 64,
            "treatment_success": True,
            "control_success": False,
            "verifier_blinded": True,
            "evaluation_started_at": 1.0,
        }

    def _receipt(self, trial, **overrides):
        from core.brain.llm.latent_cortex.frontier_artifacts import (
            TRIAL_VERIFIER_RECEIPT_SCHEMA,
        )

        receipt = {k: v for k, v in trial.items() if k != "evaluation_started_at"}
        receipt.update(
            {
                "schema": TRIAL_VERIFIER_RECEIPT_SCHEMA,
                "scorer_implementation_sha256": "e" * 64,
                "verified_at": 2.0,
                "worker_identity_sha256": "9" * 64,
                "executable_sha256": "8" * 64,
            }
        )
        receipt.update(overrides)
        return receipt

    def _validate(self, receipt, trial):
        from core.brain.llm.latent_cortex.frontier_artifacts import (
            _validate_trial_verifier_receipt,
        )

        return _validate_trial_verifier_receipt(receipt, trial)

    def test_a_pure_mirror_of_the_trial_is_refused(self):
        trial = self._trial()
        mirror = self._receipt(trial)
        del mirror["worker_identity_sha256"]
        del mirror["executable_sha256"]
        with pytest.raises(Exception):
            self._validate(mirror, trial)

    def test_a_receipt_with_execution_provenance_is_accepted(self):
        trial = self._trial()
        assert self._validate(self._receipt(trial), trial) is None

    @pytest.mark.parametrize(
        "field", ["worker_identity_sha256", "executable_sha256"],
    )
    def test_malformed_provenance_is_refused(self, field):
        trial = self._trial()
        with pytest.raises(Exception):
            self._validate(self._receipt(trial, **{field: "not-a-hash"}), trial)

    def test_provenance_mirrored_from_the_trial_is_refused(self):
        """If the trial ever gains these fields, copying them is not
        evidence either — the receipt must stay independent."""
        trial = self._trial()
        trial["worker_identity_sha256"] = "9" * 64
        with pytest.raises(Exception):
            self._validate(self._receipt(trial), trial)

    def test_the_bundle_requires_one_worker_and_one_executable(self):
        from core.brain.llm.latent_cortex import frontier_artifacts

        source = inspect.getsource(frontier_artifacts)
        assert "trial_verifier_worker_identity_inconsistent" in source
        assert "trial_verifier_executable_inconsistent" in source

    def test_the_attestation_limit_is_stated_not_implied(self):
        """Binding is verified; signatures are not. A consistently
        fabricated bundle still passes, and saying so is the point."""
        from core.brain.llm.latent_cortex import frontier_artifacts

        source = inspect.getsource(frontier_artifacts)
        assert "binding_verified_unsigned" in source
        assert "HONEST LIMIT" in source
