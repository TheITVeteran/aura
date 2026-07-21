"""The conductor that makes the loop autonomous (CP243).

The loop wasn't autonomous because the agent was the orchestrator. This
runs the loop itself, for any query, calling organ seams -- and its own
health check reports whether it actually WORKS (verifies) versus merely
runs.
"""
from __future__ import annotations

import pytest

from core.learning.cognitive_loop import CognitiveLoop, loop_health
from core.learning.workspace_producers import RetrievalProducer, WorkspaceComposer


class _Retrieval:
    def __init__(self, passages):
        self._p = passages

    def retrieve(self, query, *, limit):
        return self._p[:limit]


class _Deliberator:
    """Answers by echoing the first retrieved fact if given material."""
    def __init__(self, answer_from_material=True, fixed=None):
        self.afm = answer_from_material
        self.fixed = fixed
        self.calls = 0

    def deliberate(self, query, material):
        self.calls += 1
        if self.fixed is not None:
            return self.fixed
        return material[0] if (self.afm and material) else ""


class _Verifier:
    def __init__(self, gold):
        self.gold = gold

    def check(self, query, candidate):
        return {"correct": self.gold.lower() in str(candidate).lower()}


# ── The loop runs end to end on real seams ──────────────────────────────


def test_full_loop_acquires_deliberates_verifies_and_learns():
    composer = WorkspaceComposer(producers=[RetrievalProducer(_Retrieval(["the answer is 42"]))])
    learned = []
    loop = CognitiveLoop(
        composer=composer,
        deliberator=_Deliberator(),
        verifier=_Verifier("42"),
        learner=lambda q, a, m: learned.append((q, a)) or True,
    )
    result = loop.run("what is the answer")
    assert result.verified is True
    assert result.learned is True
    assert learned == [("what is the answer", "the answer is 42")]
    names = [s.name for s in result.stages]
    assert names[0] == "identify_gap" and "acquire" in names and "verify" in names


def test_the_same_loop_handles_any_query_no_pipeline_redesign():
    """One conductor, different domains -- the point of the whole thing."""
    for gold, fact in (("42", "the answer is 42"), ("blue", "the sky is blue")):
        loop = CognitiveLoop(
            composer=WorkspaceComposer(producers=[RetrievalProducer(_Retrieval([fact]))]),
            deliberator=_Deliberator(),
            verifier=_Verifier(gold),
        )
        assert loop.run("q").verified is True


# ── Self-correction is real and legible ─────────────────────────────────


def test_verification_failure_triggers_bounded_retries():
    composer = WorkspaceComposer(producers=[RetrievalProducer(_Retrieval(["wrong"]))])
    delib = _Deliberator(fixed="wrong")
    loop = CognitiveLoop(composer=composer, deliberator=delib,
                         verifier=_Verifier("right"), max_attempts=3)
    result = loop.run("q")
    assert result.verified is False
    assert result.attempts == 3            # exhausted the budget
    assert delib.calls == 3                # actually retried
    # retries are legible, not collapsed
    verify_stages = [s for s in result.stages if s.name == "verify"]
    assert len(verify_stages) == 3
    assert {s.detail["attempt"] for s in verify_stages} == {1, 2, 3}


def test_verifier_feedback_causally_changes_next_deliberation():
    class CorrectingDeliberator:
        def __init__(self):
            self.material_seen = []

        def deliberate(self, query, material):
            self.material_seen.append(list(material))
            if any("Use 42" in item for item in material):
                return "the answer is 42"
            return "the answer is 41"

    class FeedbackVerifier:
        def check(self, query, candidate):
            if "42" in candidate:
                return {"correct": True}
            return {"correct": False, "feedback": "Use 42, not 41."}

    deliberator = CorrectingDeliberator()
    result = CognitiveLoop(
        deliberator=deliberator,
        verifier=FeedbackVerifier(),
        max_attempts=2,
    ).run("q")

    assert result.verified is True
    assert result.attempts == 2
    assert deliberator.material_seen[0] == []
    assert "Use 42, not 41." in deliberator.material_seen[1][-1]
    assert "the answer is 41" in deliberator.material_seen[1][-1]


def test_first_attempt_success_does_not_retry():
    loop = CognitiveLoop(
        composer=WorkspaceComposer(producers=[RetrievalProducer(_Retrieval(["42"]))]),
        deliberator=_Deliberator(), verifier=_Verifier("42"), max_attempts=3,
    )
    result = loop.run("q")
    assert result.verified is True
    assert result.attempts == 1


# ── Honest degradation: never fabricate, never assume correct ───────────


def test_missing_verifier_means_unverified_never_assumed_correct():
    loop = CognitiveLoop(
        composer=WorkspaceComposer(producers=[RetrievalProducer(_Retrieval(["x"]))]),
        deliberator=_Deliberator(),
    )
    result = loop.run("q")
    assert result.verified is False
    assert result.learned is False
    assert [s for s in result.stages if s.name == "verify"][0].status == "unavailable"


def test_missing_verifier_disables_unjudgeable_retries():
    deliberator = _Deliberator(fixed="a hypothesis")
    loop = CognitiveLoop(
        deliberator=deliberator,
        verifier=None,
        max_attempts=4,
    )

    result = loop.run("q")

    assert result.attempts == 1
    assert deliberator.calls == 1
    retry = [stage for stage in result.stages if stage.name == "retry_control"]
    assert retry[0].status == "skipped"
    assert retry[0].detail == {
        "reason": "verifier_unavailable",
        "configured_attempts": 4,
        "effective_attempts": 1,
    }


def test_empty_answer_skips_verifier_and_is_reported_as_failure():
    class CountingVerifier:
        calls = 0

        def check(self, query, candidate):
            self.calls += 1
            return {"correct": True}

    verifier = CountingVerifier()
    result = CognitiveLoop(
        deliberator=_Deliberator(fixed=""),
        verifier=verifier,
        max_attempts=1,
    ).run("q")

    assert verifier.calls == 0
    deliberate = [stage for stage in result.stages if stage.name == "deliberate"][0]
    verify = [stage for stage in result.stages if stage.name == "verify"][0]
    assert deliberate.status == "failed"
    assert deliberate.detail["error"] == "empty_answer"
    assert verify.status == "skipped"
    assert verify.detail["reason"] == "no_answer"


def test_malformed_verifier_result_fails_honestly():
    class MalformedVerifier:
        def check(self, query, candidate):
            return "looks good"

    result = CognitiveLoop(
        deliberator=_Deliberator(fixed="candidate"),
        verifier=MalformedVerifier(),
        max_attempts=1,
    ).run("q")

    verify = [stage for stage in result.stages if stage.name == "verify"][0]
    assert result.verified is False
    assert verify.status == "failed"
    assert verify.detail["error"] == "invalid_verifier_result"


def test_unverified_answer_is_never_retained():
    """Retaining unverified answers is how a system trains on its mistakes."""
    retained = []
    loop = CognitiveLoop(
        composer=WorkspaceComposer(producers=[RetrievalProducer(_Retrieval(["wrong"]))]),
        deliberator=_Deliberator(fixed="wrong"),
        verifier=_Verifier("right"),
        learner=lambda q, a, m: retained.append(a) or True,
    )
    result = loop.run("q")
    assert result.learned is False
    assert retained == []


def test_missing_composer_degrades_without_fabricating():
    loop = CognitiveLoop(composer=None, deliberator=_Deliberator(fixed="guess"),
                         verifier=_Verifier("guess"))
    result = loop.run("q")
    # deliberates with no material rather than inventing facts
    acquire = [s for s in result.stages if s.name == "acquire"][0]
    assert acquire.status == "unavailable"


def test_gap_detector_can_skip_acquisition():
    class KnowsEverything:
        def has_gap(self, query):
            return False

    composer = WorkspaceComposer(producers=[RetrievalProducer(_Retrieval(["fact"]))])
    loop = CognitiveLoop(composer=composer, deliberator=_Deliberator(fixed="known"),
                         verifier=_Verifier("known"), gap_detector=KnowsEverything())
    result = loop.run("q")
    assert [s for s in result.stages if s.name == "acquire"][0].status == "skipped"
    assert result.verified is True


# ── The loop's own health: working, or just running? ────────────────────


def test_loop_health_distinguishes_working_from_running():
    good = [
        CognitiveLoop(
            composer=WorkspaceComposer(producers=[RetrievalProducer(_Retrieval(["42"]))]),
            deliberator=_Deliberator(), verifier=_Verifier("42"),
        ).run("q") for _ in range(3)
    ]
    health = loop_health(good)
    assert health["working"] is True
    assert health["verified_rate"] == 1.0

    bad = [
        CognitiveLoop(
            composer=WorkspaceComposer(producers=[RetrievalProducer(_Retrieval(["no"]))]),
            deliberator=_Deliberator(fixed="no"), verifier=_Verifier("yes"),
        ).run("q") for _ in range(3)
    ]
    assert loop_health(bad)["working"] is False


def test_health_reports_self_correction_rescues():
    class SecondTimeLucky:
        def __init__(self):
            self.n = 0
        def deliberate(self, query, material):
            self.n += 1
            return "right" if self.n >= 2 else "wrong"

    loop = CognitiveLoop(
        composer=WorkspaceComposer(producers=[RetrievalProducer(_Retrieval(["x"]))]),
        deliberator=SecondTimeLucky(), verifier=_Verifier("right"), max_attempts=3,
    )
    result = loop.run("q")
    assert result.verified is True and result.attempts == 2
    assert loop_health([result])["self_correction_rescues"] == 1


def test_invalid_configuration_is_refused():
    with pytest.raises(ValueError, match="cannot think"):
        CognitiveLoop(deliberator=None)
    with pytest.raises(ValueError, match="max_attempts"):
        CognitiveLoop(deliberator=_Deliberator(), max_attempts=0)
    with pytest.raises(ValueError, match="non-empty"):
        CognitiveLoop(deliberator=_Deliberator()).run("")
    with pytest.raises(ValueError, match="no loop results"):
        loop_health([])
