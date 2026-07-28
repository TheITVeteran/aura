"""An imperative after a preamble is still an imperative.

The implicit-permission check was anchored with ``^``, so ANY preamble defeated
it — which is exactly how people write. Measured live on the desktop surface:

    "Hey Aura, it's Bryan. Hold onto the codeword LANTERN for later. First real
     task: run a Python snippet that prints os.getpid() and os.cpu_count(),
     then give me the two actual numbers it returned."

``run_code``, ``code_repl`` and ``internal_sandbox`` were all READY and
available, and nothing dispatched: no permission was detected, ``should_execute``
stayed False, and the turn never reached an executor. The user got an honest
"the action didn't go through" for a request the sandbox could have answered.

The codebase had already learned this lesson once for the search-query
extractor — "every pattern below is anchored with .match(), so a preamble
defeats all of them".
"""

from __future__ import annotations

import pytest

from core.phases.action_intent import detect_action_intent

LIVE_FAILURE = (
    "Hey Aura, it's Bryan. Hold onto the codeword LANTERN for later. First real "
    "task: run a Python snippet that prints os.getpid() and os.cpu_count(), then "
    "give me the two actual numbers it returned."
)


@pytest.mark.parametrize(
    "message",
    [
        LIVE_FAILURE,
        "Run a python snippet that prints the pid.",
        "Quick one before we continue — open Notes and type the anchor number.",
        "First think about it, then execute the snippet and report the output.",
        "Two things. Please run the calculation and show the result.",
    ],
)
def test_a_clause_initial_imperative_grants_permission(message: str) -> None:
    intent = detect_action_intent(message)
    assert intent.has_action_request is True
    assert intent.has_permission_grant is True, (
        "a preamble must not defeat the imperative that follows it"
    )
    assert intent.should_execute is True


@pytest.mark.parametrize(
    "message",
    [
        # Negation is stripped upstream and must stay unexecuted.
        "Don't run anything, just tell me what you would do.",
        # "run" as a noun is not an imperative.
        "There was a run of failures in the test suite; what caused it?",
        # A hypothetical is not a request.
        "If you were to open a file, how would you verify it?",
    ],
)
def test_non_requests_still_do_not_execute(message: str) -> None:
    assert detect_action_intent(message).should_execute is False


@pytest.mark.parametrize(
    "message",
    [
        "I have a question about your process, show me your reasoning on that.",
        "Two things to discuss; make it clear how you decided.",
        "Let's talk about latency, show me how you think about it.",
    ],
)
def test_clause_widening_does_not_execute_on_conversational_verbs(message: str) -> None:
    """The clause-level verb set deliberately excludes "show" and "make".

    Reaching a conversational verb across a clause boundary must not read as an
    execution request; otherwise every "…, show me how you decided" would try to
    run something.
    """
    intent = detect_action_intent(message)
    assert intent.has_permission_grant is False
    assert intent.should_execute is False
