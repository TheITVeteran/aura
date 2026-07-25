"""CP126 reasoning-amplifier cluster: the cache was a hole through every gate.

Five findings sharing one shape — work that the uncached path verifies, the
cached path serves without verifying:

* ``236526e0`` — the lookup sat ahead of mode selection, risk handling,
  evidence gathering, verification and calibration, so any nonempty entry
  came back as verified with agreement 1.0, including for a PROOF or
  high-stakes request carrying new required evidence.
* ``d45893c2`` — verification ran against ``answer``; calibration then
  rewrote it, and the REWRITE was cached and used as training data under the
  original verifier's pass.
* ``bbb356e0`` — only Phi-resolved sample counts were clamped; an explicit
  sample_budget went through unchanged.
* ``93e56508`` — the code regex ran before the repository regex, so
  source-DEPENDENT questions were cached as source-independent ones.
* ``8b2be13e`` — a vacuous pass (ok without checked) suppressed escalation.
"""
from __future__ import annotations

import inspect

import pytest

from core.brain.reasoning_amplifier_v2 import (
    ReasoningMode,
    _MAX_SAMPLES,
    _admit_sample_budget,
    _cache_hit_is_insufficient,
    classify_task_type,
)


class _Cached:
    def __init__(self, mode="normal", verifiers_run=("arith",), required_evidence=()):
        self.answer = "42"
        self.confidence = 0.9
        self.mode = mode
        self.verifiers_run = list(verifiers_run)
        self.required_evidence = list(required_evidence)


class _Request:
    def __init__(self, mode=None, risk_level="normal", required_evidence=()):
        self.mode = mode
        self.risk_level = risk_level
        self.required_evidence = list(required_evidence)


class _Problem:
    task_type = "math"


class TestACacheHitCannotOutrankTheQuestion:
    def test_an_ordinary_hit_is_served(self):
        assert _cache_hit_is_insufficient(
            _Cached(), request=_Request(), problem=_Problem(),
        ) == ""

    def test_proof_refuses_a_non_proof_derivation(self):
        """PROOF is defined by refusing to answer unless a verifier cleared
        it; a cached NORMAL answer has not met that bar."""
        reason = _cache_hit_is_insufficient(
            _Cached(mode="normal"),
            request=_Request(mode=ReasoningMode.PROOF.value),
            problem=_Problem(),
        )
        assert reason == "proof_requires_proof_grade_derivation"

    def test_proof_accepts_a_proof_derivation(self):
        assert _cache_hit_is_insufficient(
            _Cached(mode="proof"),
            request=_Request(mode=ReasoningMode.PROOF.value),
            problem=_Problem(),
        ) == ""

    def test_a_weaker_cached_mode_cannot_serve_a_stronger_request(self):
        reason = _cache_hit_is_insufficient(
            _Cached(mode="fast"),
            request=_Request(mode=ReasoningMode.DEEP.value),
            problem=_Problem(),
        )
        assert reason == "cached_mode_weaker_than_requested"

    def test_high_risk_refuses_a_shallow_derivation(self):
        """A high-stakes request is exactly the one that must not be
        answered from a derivation made when nothing was at stake."""
        reason = _cache_hit_is_insufficient(
            _Cached(mode="normal"),
            request=_Request(risk_level="high"),
            problem=_Problem(),
        )
        assert reason == "high_risk_requires_deep_derivation"

    def test_high_risk_accepts_a_deep_derivation(self):
        assert _cache_hit_is_insufficient(
            _Cached(mode="deep"),
            request=_Request(risk_level="high"),
            problem=_Problem(),
        ) == ""

    def test_new_required_evidence_misses(self):
        reason = _cache_hit_is_insufficient(
            _Cached(required_evidence=["docs"]),
            request=_Request(required_evidence=["docs", "benchmark"]),
            problem=_Problem(),
        )
        assert reason.startswith("required_evidence_not_covered")
        assert "benchmark" in reason

    def test_covered_evidence_is_served(self):
        assert _cache_hit_is_insufficient(
            _Cached(required_evidence=["docs", "benchmark"]),
            request=_Request(required_evidence=["docs"]),
            problem=_Problem(),
        ) == ""

    def test_a_missing_verifier_label_does_not_disable_the_cache(self):
        """Presence in the cache already proves a verifier cleared it.

        ReasoningSolvedCache.put refuses anything with verified=False, so
        verifiers_run is provenance, not proof-of-existence. An earlier
        version of this gate treated a missing label as missing
        verification and silently disabled the cache for legitimate
        entries.
        """
        assert _cache_hit_is_insufficient(
            _Cached(verifiers_run=()), request=_Request(), problem=_Problem(),
        ) == ""

    def test_a_stale_entry_without_the_evidence_field_fails_closed(self):
        """Entries written before required_evidence existed cover nothing."""

        class _Old:
            answer = "42"
            confidence = 0.9
            mode = "normal"
            verifiers_run = ["arith"]

        reason = _cache_hit_is_insufficient(
            _Old(), request=_Request(required_evidence=["docs"]), problem=_Problem(),
        )
        assert reason.startswith("required_evidence_not_covered")


class TestTheCacheReceiptDoesNotFabricateAgreement:
    def test_agreement_is_not_one_on_a_cache_hit(self):
        from core.brain import reasoning_amplifier_v2 as mod

        source = inspect.getsource(mod)
        block = source.split("solved-cache HIT", 1)[1][:900]
        # Agreement measures consensus ACROSS candidates; a cache hit ran none.
        assert "agreement=0.0" in block
        assert "agreement=1.0" not in block

    def test_the_status_says_it_came_from_cache(self):
        from core.brain import reasoning_amplifier_v2 as mod

        source = inspect.getsource(mod)
        assert 'epistemic_status="verified_cached"' in source


class TestOnlyVerifiedTextBecomesDurableTruth:
    def test_the_cache_stores_the_verified_text_not_the_rewrite(self):
        from core.brain import reasoning_amplifier_v2 as mod

        source = inspect.getsource(mod)
        assert "_durable_answer = answer" in source
        assert "answer=_durable_answer," in source

    def test_all_three_durable_sinks_take_the_verified_text(self):
        """Solved cache, self-improvement trace, procedural memory.

        The finding named all three. The CALLER still receives the
        calibrated text — that hedge is a presentation concern — it simply
        does not get to become durable truth or training data on the
        strength of a verdict it never faced.
        """
        from core.brain import reasoning_amplifier_v2 as mod

        source = inspect.getsource(mod)
        assert source.count("answer=_durable_answer,") == 3
        # The single remaining use is the return to the caller.
        assert source.count("answer=calibrated_answer,") == 1
        # The LAST return is the main path; the first is the cache-hit path,
        # which returns the cached text.
        returned = source.rsplit("return AmplifiedAnswer(", 1)[1][:200]
        assert "answer=calibrated_answer," in returned

    def test_required_evidence_is_persisted_with_the_entry(self):
        from core.brain import reasoning_amplifier_v2 as mod
        from core.brain.reasoning_solved_cache import SolvedEntry

        assert "required_evidence" in SolvedEntry.__dataclass_fields__
        assert "required_evidence=list(request.required_evidence or [])" in (
            inspect.getsource(mod)
        )

    def test_the_entry_round_trips_its_evidence(self):
        from core.brain.reasoning_solved_cache import SolvedEntry

        entry = SolvedEntry(
            answer="a", confidence=1.0, mode="deep", task_type="math",
            verifiers_run=["arith"], required_evidence=["docs"],
        )
        restored = SolvedEntry.from_dict(entry.to_dict())
        assert restored.required_evidence == ["docs"]


class TestSampleBudgetIsAlwaysAdmitted:
    @pytest.mark.parametrize("mode", list(ReasoningMode))
    def test_none_takes_the_mode_default_within_the_cap(self, mode):
        assert 1 <= _admit_sample_budget(None, mode) <= _MAX_SAMPLES

    def test_a_hostile_budget_is_clamped(self):
        assert _admit_sample_budget(10**9, ReasoningMode.NORMAL) == _MAX_SAMPLES

    def test_zero_and_negative_take_the_default(self):
        default = _admit_sample_budget(None, ReasoningMode.NORMAL)
        assert _admit_sample_budget(0, ReasoningMode.NORMAL) == default
        assert _admit_sample_budget(-5, ReasoningMode.NORMAL) == default

    def test_malformed_values_take_the_default(self):
        default = _admit_sample_budget(None, ReasoningMode.NORMAL)
        for value in ("x", None, True, [], {}):
            assert _admit_sample_budget(value, ReasoningMode.NORMAL) == default

    def test_a_usable_request_is_honoured(self):
        assert _admit_sample_budget(2, ReasoningMode.NORMAL) == 2

    def test_every_resolution_site_goes_through_the_gate(self):
        from core.brain import reasoning_amplifier_v2 as mod

        source = inspect.getsource(mod)
        assert "request.sample_budget or _MODE_BUDGET[mode]" not in source


class TestRepositoryQuestionsAreNotCachedAsCode:
    @pytest.mark.parametrize(
        "question",
        [
            "where is the retry logic implemented in this codebase",
            "which file has the bug in the import handler",
            "how does the module architecture work",
        ],
    )
    def test_source_dependent_questions_classify_as_repo(self, question):
        assert classify_task_type(question) == "repo_audit"

    @pytest.mark.parametrize(
        "question",
        ["write a function to reverse a list", "fix this traceback"],
    )
    def test_genuine_code_tasks_still_classify_as_code(self, question):
        assert classify_task_type(question) == "code"

    def test_repo_is_checked_before_code(self):
        from core.brain import reasoning_amplifier_v2 as mod

        source = inspect.getsource(mod.classify_task_type)
        # Compare the executable body, not the docstring, which necessarily
        # names the old ordering while explaining it.
        body = source.split('"""', 2)[-1]
        assert body.index("_REPO_HINT") < body.index("_CODE_HINT")


class TestAVacuousPassDoesNotStopTheSearch:
    def test_escalation_requires_both_ok_and_checked(self):
        """An unevaluated answer is not evidence that a stronger tier is
        unnecessary — it is the case with the least information."""
        from core.brain import reasoning_amplifier_v2 as mod

        source = inspect.getsource(mod)
        assert "not (verifier_ok and verifier_checked)" in source
        assert "            not verifier_ok\n" not in source
