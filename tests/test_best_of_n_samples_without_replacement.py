"""Best-of-8 was best-of-2, and the receipt did not say so.

``deliberate_best_of`` runs independent blind passes. Blindness buys
something real — a pass that sees a prior candidate can rationalise toward
it instead of solving — but it also makes the passes i.i.d., and i.i.d.
sampling from a peaked model spends most of its budget re-deriving the same
answer.

The change under test is narrow and it matters that it stays narrow: an
answer the verifier REFUTED, with a CHECKED verdict, is excluded from later
passes. Nothing else is ever shown. A refuted candidate cannot be
rationalised toward — it is removed, not offered — so the blindness survives
exactly where it was buying something, and the coverage improves where it
was costing something.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from core.brain.reasoning_revision_gate import deliberate_best_of


@dataclass
class _Verdict:
    ok: bool
    checked: bool = True
    score: float = 0.0
    issues: tuple = ()


def _run(**kwargs):
    return asyncio.run(deliberate_best_of(**kwargs))


# ─────────────────────────────────────────────── the duplicate work is real


def test_a_peaked_model_wastes_its_budget_and_the_receipt_says_so():
    """Eight passes, two distinct answers. This is the disease."""

    async def solve(index, conditioning=""):
        return "the mode answer" if index < 6 else "something else"

    async def verify(answer):
        return _Verdict(ok=False)

    result = _run(solve=solve, verify=verify, max_passes=8, exclude_refuted=False)

    assert result.passes == 8
    assert result.distinct_answers == 2
    assert result.to_dict()["duplicate_passes"] == 6, (
        "six of eight passes re-derived an answer already examined, and "
        "nothing in the receipt said so"
    )


def test_excluding_refuted_answers_raises_distinct_coverage():
    """The same model, the same eight passes, more of the space examined."""
    pool = ["wrong-1", "wrong-2", "wrong-3", "wrong-4", "right"]

    def _make_solver():
        def solve(index, conditioning=""):
            ruled_out = {
                line[len("- not ") :].strip()
                for line in conditioning.splitlines()
                if line.startswith("- not ")
            }
            for candidate in pool:
                if candidate not in ruled_out:
                    return candidate
            return pool[-1]

        return solve

    async def verify(answer):
        return _Verdict(ok=answer == "right")

    without = _run(
        solve=_make_solver(), verify=verify, max_passes=5, exclude_refuted=False
    )
    with_exclusion = _run(
        solve=_make_solver(), verify=verify, max_passes=5, exclude_refuted=True
    )

    assert without.distinct_answers == 1, (
        "without exclusion the solver has no reason to move off its mode"
    )
    assert with_exclusion.distinct_answers == 5
    assert with_exclusion.answer == "right"
    assert without.answer != "right"


def test_the_exclusion_count_is_on_the_receipt():
    def solve(index, conditioning=""):
        return f"candidate-{index}"

    async def verify(answer):
        return _Verdict(ok=False)

    result = _run(solve=solve, verify=verify, max_passes=4)

    assert result.exclusions == 4
    assert result.to_dict()["exclusions"] == 4


# ──────────────────────────────────────── only a CHECKED refutation excludes


def test_an_unchecked_failure_does_not_exclude():
    """Excluding what was never checked removes an answer for not looking."""

    def solve(index, conditioning=""):
        return f"candidate-{index}"

    async def verify(answer):
        return _Verdict(ok=False, checked=False)

    result = _run(solve=solve, verify=verify, max_passes=4)

    assert result.exclusions == 0


def test_a_passing_answer_is_never_excluded():
    def solve(index, conditioning=""):
        return f"candidate-{index}"

    async def verify(answer):
        return _Verdict(ok=True)

    result = _run(solve=solve, verify=verify, max_passes=3)

    assert result.exclusions == 0


def test_an_unrefuted_candidate_is_never_shown_to_a_later_pass():
    """The blindness the design depends on, preserved.

    Rationalisation is the hazard of seeing a PLAUSIBLE prior answer. Only
    refutations reach the conditioning block, so there is nothing there to
    agree with.
    """
    blocks = []

    def solve(index, conditioning=""):
        blocks.append(conditioning)
        return f"candidate-{index}"

    async def verify(answer):
        # Undecided: not ok, not checked.
        return _Verdict(ok=False, checked=False)

    _run(solve=solve, verify=verify, max_passes=4)

    assert all(block == "" for block in blocks), (
        f"an unrefuted candidate leaked into a later pass: {blocks}"
    )


def test_only_the_refuted_text_appears_in_the_block():
    blocks = []

    def solve(index, conditioning=""):
        blocks.append(conditioning)
        return "good" if index == 0 else "bad"

    async def verify(answer):
        return _Verdict(ok=answer == "good", checked=True)

    _run(solve=solve, verify=verify, max_passes=3)

    later = [block for block in blocks if block]
    assert later, "nothing was ever excluded"
    for block in later:
        ruled_out = {
            line[len("- not ") :].strip()
            for line in block.splitlines()
            if line.startswith("- not ")
        }
        assert "good" not in ruled_out, (
            "an ACCEPTED answer was shown to a later pass"
        )


# ───────────────────────────────────────────── compatibility and safety


def test_a_one_argument_solver_still_works():
    """Existing callers pass solve(index). They must keep working."""
    calls = []

    async def solve(index):
        calls.append(index)
        return f"answer-{index}"

    async def verify(answer):
        return _Verdict(ok=False)

    result = _run(solve=solve, verify=verify, max_passes=3)

    assert calls == [0, 1, 2]
    assert result.passes == 3
    assert result.distinct_answers == 3


def test_a_typeerror_inside_a_two_argument_solver_is_not_misread_as_arity():
    """Detected from the signature, not by calling and catching.

    Catching TypeError to infer arity would misread a TypeError raised
    INSIDE the solver as "it only takes one argument", and every later pass
    would silently drop the exclusions while the receipt still claimed they
    were applied.
    """
    seen = []

    def solve(index, conditioning=""):
        seen.append(conditioning)
        if index == 0:
            raise TypeError("something inside the solver is broken")
        return f"answer-{index}"

    async def verify(answer):
        return _Verdict(ok=False)

    try:
        _run(solve=solve, verify=verify, max_passes=2)
    except TypeError:
        pass

    assert seen, "the two-argument solver was never called with a block"


def test_exclusion_can_be_turned_off_and_then_changes_nothing():
    def solve(index, conditioning=""):
        assert conditioning == ""
        return f"answer-{index}"

    async def verify(answer):
        return _Verdict(ok=False)

    result = _run(solve=solve, verify=verify, max_passes=3, exclude_refuted=False)

    assert result.exclusions == 0
    assert result.passes == 3


def test_the_monotonic_invariant_survives_exclusion():
    """The gate's own promise: more passes can only help.

    Exclusion changes which candidates are examined; it must not change
    whether a worse-evidenced answer can be adopted.
    """
    answers = ["weak", "strong", "weak-again"]
    scores = {"weak": 0.2, "strong": 0.9, "weak-again": 0.1}

    def solve(index, conditioning=""):
        return answers[index]

    async def verify(answer):
        return _Verdict(ok=scores[answer] > 0.5, checked=True, score=scores[answer])

    result = _run(solve=solve, verify=verify, max_passes=3)

    assert result.answer == "strong"


# ───────────── rejection, not prompting — the measured form of the policy


def test_a_repeated_refuted_answer_is_redrawn_not_reverified():
    """The trade the measured gain comes from.

    Prompt-conditioned exclusion was measured and LOST (46.9% vs 48.1%
    i.i.d.): describing the excluded answers perturbs the distribution and
    anchors on the values being excluded. Rejection sampling won (59.4%,
    p=0.044) on FEWER verifier calls, because a repeat of an already-refuted
    answer is discarded before the verifier is paid.
    """
    verify_calls = []

    answers = ["wrong", "wrong", "wrong", "right"]

    def solve(index, conditioning=""):
        return answers[min(index, len(answers) - 1)]

    async def verify(answer):
        verify_calls.append(answer)
        return _Verdict(ok=answer == "right", checked=True)

    result = _run(solve=solve, verify=verify, max_passes=4)

    assert result.answer == "right"
    assert verify_calls.count("wrong") == 1, (
        f"the same refuted answer was verified {verify_calls.count('wrong')} "
        "times; each repeat should be redrawn before the verifier is paid"
    )
    assert result.rejected_redraws >= 1


def test_rejection_is_bounded_when_the_model_will_not_move():
    """A model that always repeats must not spin forever."""

    def solve(index, conditioning=""):
        return "always the same"

    async def verify(answer):
        return _Verdict(ok=False, checked=True)

    result = _run(solve=solve, verify=verify, max_passes=4)

    # The first draw IS verified and becomes best-seen — the gate's existing
    # monotonic promise, unchanged: a refuted answer still beats no answer.
    # What must be bounded is the redrawing, not the outcome.
    assert result.answer == "always the same"
    assert result.rejected_redraws <= 4 * 3
    assert result.distinct_answers == 1


def test_the_receipt_reports_the_verifier_calls_saved():
    def solve(index, conditioning=""):
        return "repeat" if index < 3 else "new"

    async def verify(answer):
        return _Verdict(ok=False, checked=True)

    result = _run(solve=solve, verify=verify, max_passes=4)

    payload = result.to_dict()
    assert payload["rejected_redraws"] >= 1
    assert payload["distinct_answers"] < result.passes + payload["rejected_redraws"]
