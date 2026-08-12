"""The absence of a check, reported as a passed check.

This is the defect class this codebase keeps rediscovering, and these four
were live instances of it in the most load-bearing places available:

  * `IdentityGuard.validate_output` returned ``(True, "OK", 1.0)`` when the
    real persona gate could not be imported. The check that decides whether
    live model output still sounds like her reported a PERFECT pass when it
    was not running at all, and the caller had no way to tell that verdict
    apart from a real one. The old code called this "non-critical".
  * `proof_obligations` reached `ProofStatus.PROVED` on `verifier.ok`, which
    is documented as True when NOTHING was checked. The strongest claim the
    system can make rested on the weakest possible evidence.
  * `IndependentValidationLoop` scored a task with no hidden checker as
    PASSED with 1.0, so every unscoreable task raised the pass rate — the
    benchmark read best exactly where it measured least.
  * `pii_scrubber` returned an empty name list when it could not read
    `biography_private.json`, and scrubbing continued. The owner's and his
    family's real names would go to a cloud provider unredacted, and the
    only signal was a list that looks identical to "this user named nobody".

Each test states the failure it prevents, because the fix in each case is
one line away from being reverted by someone making the code "simpler".
"""
from __future__ import annotations

import pytest


# ────────────────────────────── the identity gate fails closed


def test_an_unreachable_identity_gate_is_not_a_pass(monkeypatch):
    """Losing the guard must not read as passing it."""
    from core.agency.identity_guard import IdentityGuard

    import builtins

    real_import = builtins.__import__

    def _no_persona_gate(name, *args, **kwargs):
        if name == "core.identity.identity_guard":
            raise ImportError("persona gate unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_persona_gate)

    ok, reason, score = IdentityGuard().validate_output("anything at all")

    assert ok is False, "the identity gate reported a pass while unavailable"
    assert score == 0.0, "an unavailable gate scored the output perfect"
    assert "unavailable" in reason


def test_the_caller_rolls_back_on_an_unavailable_identity_gate():
    """`IdentityReflectionPhase` treats `not ok` as a cognitive rollback.

    That is the correct response to "the identity check cannot run" and the
    wrong response to nothing at all, which is what it used to receive.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "core" / "agency" / "identity_guard.py"
    ).read_text("utf-8")

    assert 'return True, "OK", 1.0' not in source, (
        "the identity gate returns a perfect pass on failure again"
    )


# ────────────────────────────── PROVED requires a check that ran


def test_a_verification_result_distinguishes_unchecked_from_passed():
    from core.brain.verifiers.base import VerificationResult

    unchecked = VerificationResult(domain="d", ok=True, checked=False)
    passed = VerificationResult(domain="d", ok=True, checked=True)
    failed = VerificationResult(domain="d", ok=False, checked=True)

    assert unchecked.verdict == "UNCHECKED"
    assert passed.verdict == "PASSED"
    assert failed.verdict == "FAILED"

    # The property a GATE should read. `ok` is for ranking.
    assert unchecked.conclusively_ok is False
    assert passed.conclusively_ok is True
    assert failed.conclusively_ok is False


def test_ok_is_still_true_when_nothing_was_checked():
    """The documented contract stays intact.

    The fix must not change what `ok` means — several rankers depend on
    "no provable failure was found". It adds the question a gate needs.
    """
    from core.brain.verifiers.base import VerificationResult

    assert VerificationResult(domain="d", ok=True, checked=False).ok is True


def test_the_verdict_is_carried_on_the_receipt():
    from core.brain.verifiers.base import VerificationResult

    payload = VerificationResult(domain="d", ok=True, checked=False).to_dict()

    assert payload["verdict"] == "UNCHECKED"


def test_proof_status_does_not_reach_proved_without_a_real_check():
    """The strongest claim in the system, on the weakest evidence."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "core"
        / "learning"
        / "proof_obligations.py"
    ).read_text("utf-8")

    assert "ProofStatus.PROVED if verifier.ok and" not in source, (
        "PROVED is decided by `verifier.ok` again, which is True when no "
        "verifier checked anything"
    )
    assert "conclusively_ok" in source


# ────────────────────────────── an unscored task is not a passed task


def test_a_task_without_a_checker_is_not_recorded_as_passed():
    from core.advanced_cognition.validation import (
        BenchmarkTask,
        IndependentValidationLoop,
    )

    loop = IndependentValidationLoop()
    task = BenchmarkTask(task_id="t1", domain="d", prompt={"q": 1})

    result = loop.evaluate(task, runner=lambda _p: "anything")

    assert result.checked is False
    assert result.passed is False, (
        "a task with no hidden checker was recorded as passed, so every "
        "unscoreable task raised the pass rate"
    )
    assert result.score == 0.0


def test_a_task_with_a_checker_is_scored_normally():
    """The fix must not make real checks unscoreable."""
    from core.advanced_cognition.validation import (
        BenchmarkTask,
        IndependentValidationLoop,
    )

    loop = IndependentValidationLoop()
    task = BenchmarkTask(
        task_id="t2", domain="d", prompt={}, hidden_checker=lambda out: out == "right"
    )

    good = loop.evaluate(task, runner=lambda _p: "right")
    bad = loop.evaluate(task, runner=lambda _p: "wrong")

    assert (good.checked, good.passed, good.score) == (True, True, 1.0)
    assert (bad.checked, bad.passed, bad.score) == (True, False, 0.0)


def test_the_pass_rate_is_taken_over_checkable_tasks_only():
    """An unscoreable task must not move a rate it gave no evidence to."""
    from core.advanced_cognition.validation import (
        BenchmarkTask,
        IndependentValidationLoop,
    )

    loop = IndependentValidationLoop()
    loop.evaluate(
        BenchmarkTask(task_id="a", domain="d", prompt={}, hidden_checker=lambda o: True),
        runner=lambda _p: "x",
    )
    loop.evaluate(BenchmarkTask(task_id="b", domain="d", prompt={}), runner=lambda _p: "x")

    summary = loop.summary()

    assert summary["pass_rate"] == 1.0, (
        "the unchecked task dragged the pass rate of the checked one"
    )
    assert summary["unchecked_tasks"] == 1


# ────────────────────── redaction that cannot run must not report success


def test_an_unreadable_private_name_list_refuses_rather_than_returning_empty(
    monkeypatch,
):
    """The names this list exists to remove are exactly the ones no regex
    can catch. Continuing without it ships them."""
    from core.brain import pii_scrubber

    monkeypatch.setattr(pii_scrubber, "_cached_names", None)

    def _explode() -> list[str]:
        raise pii_scrubber.PrivateNamesUnavailable("cannot read")

    monkeypatch.setattr(pii_scrubber, "_get_private_names", _explode)

    with pytest.raises(pii_scrubber.PrivateNamesUnavailable):
        pii_scrubber.scrub_pii_for_cloud("a message mentioning someone")


def test_the_failure_has_its_own_exception_type():
    """A generic error would let a broad `except` upstream treat "redaction
    is unavailable" the same as "there was nothing to redact"."""
    from core.brain.pii_scrubber import PrivateNamesUnavailable

    assert issubclass(PrivateNamesUnavailable, RuntimeError)
    assert PrivateNamesUnavailable is not RuntimeError


def test_a_working_scrub_is_unaffected(monkeypatch):
    from core.brain import pii_scrubber

    monkeypatch.setattr(pii_scrubber, "_get_private_names", lambda: ["Alexandra"])

    out = pii_scrubber.scrub_pii_for_cloud("Alexandra asked about the weather")

    assert "Alexandra" not in out
    assert "the user" in out


def test_a_failed_load_is_not_cached_as_an_empty_list():
    """A transient read error must not disable redaction for the life of
    the process."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "core" / "brain" / "pii_scrubber.py"
    ).read_text("utf-8")

    body = source[source.index("def _load_private_names") :]
    body = body[: body.index("_cached_names: Optional")]

    assert "return []" in body, "the success path still returns a list"
    assert "raise PrivateNamesUnavailable" in body, (
        "a failed private-name load silently returns an empty redaction set "
        "again, and scrubbing continues on it"
    )


# ─────────────────── a rejected pool is not the same as an unchecked one


@pytest.mark.asyncio
async def test_a_unanimously_rejected_pool_is_not_shipped_as_merely_unverified():
    """`pool = valid if valid else cands` collapsed two different facts.

    "Nothing passed because nothing was checkable" and "every candidate was
    PROVABLY WRONG" both fell back to the full pool, and the confidence was
    then computed as though the answer were merely unverified. Agreement
    among wrong answers is agreement about being wrong.
    """
    from core.brain.reasoning_amplifier import VerifierOutcome, amplify

    async def _reject_everything(_text: str):
        return VerifierOutcome.FAIL, ["provable arithmetic error"]

    result = await amplify(
        ["the answer is 4", "the answer is 4", "the answer is 4"],
        verify=_reject_everything,
    )

    assert result.valid_n == 0
    assert result.confidence <= 0.30, (
        f"a unanimously rejected pool returned confidence {result.confidence}, "
        "which is what an ordinary unverified answer scores"
    )


@pytest.mark.asyncio
async def test_an_unverifiable_pool_keeps_its_ordinary_confidence():
    """The fix must not punish "could not check", which establishes nothing
    and is not evidence against the candidates."""
    from core.brain.reasoning_amplifier import VerifierOutcome, amplify

    async def _cannot_check(_text: str):
        return VerifierOutcome.UNKNOWN, []

    result = await amplify(
        ["the answer is 4", "the answer is 4"], verify=_cannot_check
    )

    assert result.confidence > 0.30


@pytest.mark.asyncio
async def test_an_unchecked_candidate_is_preferred_over_a_rejected_one():
    """With no PASS available, the pool should be the NOT-REJECTED set
    rather than everything — a provable error is grounds to exclude."""
    from core.brain.reasoning_amplifier import VerifierOutcome, amplify

    async def _reject_the_wrong_one(text: str):
        if "9" in text:
            return VerifierOutcome.FAIL, ["provable error"]
        return VerifierOutcome.UNKNOWN, []

    result = await amplify(
        ["the answer is 9", "the answer is 9", "the answer is 4"],
        verify=_reject_the_wrong_one,
    )

    assert "4" in result.answer, (
        "the rejected majority won because failed candidates stayed in the "
        "pool alongside the unchecked one"
    )
