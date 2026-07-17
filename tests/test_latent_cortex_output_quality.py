"""Contract tests: the resident latent answer product-quality gate.

`evaluate_latent_output` is a fail-closed gate on user-visible text from
live 32B episodes. These tests prove the gate (a) accepts genuinely complete
answers, (b) rejects each named defect class for the right reason, and
(c) binds its verdict to the exact text and objective by hash so a receipt
cannot be replayed against different content.
"""
from __future__ import annotations

import hashlib

from core.brain.llm.latent_cortex.output_quality import (
    OUTPUT_QUALITY_SCHEMA,
    evaluate_latent_output,
)

GOOD_ANSWER = (
    "The supervisor design is stronger because it isolates faults per lane. "
    "First, each worker owns a single model and restarts independently, so a "
    "wedged lane never blocks its siblings. "
    "Second, timeouts cancel only the affected episode, so other lanes continue "
    "serving while the failed one recovers. "
    "Finally, receipts prove cleanup after every restart, which prevents silent "
    "state corruption from surviving into the next request."
)
OBJECTIVE = (
    "Explain why the supervisor design is stronger under timeout and restart faults."
)


def _grade(text, *, generated=120, termination="eos", objective=OBJECTIVE):
    return evaluate_latent_output(
        text, generated_tokens=generated, termination=termination, objective=objective
    )


def test_complete_answer_passes_with_full_receipt():
    receipt = _grade(GOOD_ANSWER)
    assert receipt["passed"] is True
    assert receipt["reasons"] == []
    assert receipt["schema"] == OUTPUT_QUALITY_SCHEMA
    assert receipt["terminal_complete"] is True
    assert receipt["word_count"] > 30
    assert "explain" in receipt["requested_facets"]
    assert "explain" in receipt["satisfied_facets"]


def test_empty_and_whitespace_outputs_fail():
    assert "empty_output" in _grade("")["reasons"]
    assert "empty_output" in _grade("   \n\n  ")["reasons"]
    assert "invalid_generated_token_count" in _grade(GOOD_ANSWER, generated=0)["reasons"]


def test_terminal_fragment_rejected():
    fragment = GOOD_ANSWER.rstrip(".") + " and the next point is that the"
    receipt = _grade(fragment)
    assert receipt["terminal_complete"] is False
    assert "terminal_fragment" in receipt["reasons"]


def test_repetitive_language_rejected():
    loop = "the plan is the plan is " * 30
    receipt = _grade(loop + ".", generated=200)
    assert "repetitive_language" in receipt["reasons"]


def test_repeated_lines_rejected():
    text = "\n".join(["The system restarts cleanly each time."] * 8)
    receipt = _grade(text)
    assert "repeated_lines" in receipt["reasons"]


def test_low_lexical_yield_rejected_for_long_generation():
    # 200 generated tokens that render to almost no words = decode babble.
    receipt = _grade("ok fine.", generated=200)
    assert "low_lexical_yield" in receipt["reasons"] or (
        "insufficient_lexical_content" in receipt["reasons"]
    )


def test_compound_request_demands_development():
    objective = "Compare the two designs and recommend which one we should adopt."
    thin = "Both are fine."
    receipt = _grade(thin, objective=objective)
    assert receipt["compound_request"] is True
    assert "missing_requested_facets" in receipt["reasons"]
    developed = (
        "The event-driven design is stronger, whereas the polling design wastes "
        "cycles checking queues that are usually empty. "
        "I recommend the event-driven approach because it scales with idle lanes "
        "and keeps latency flat as the number of designs under load grows. "
        "By contrast, polling degrades under load because every idle check still "
        "costs a full scheduling pass. Therefore we should adopt the event-driven "
        "design for this system."
    )
    good = _grade(developed, objective=objective)
    assert good["passed"] is True, good["reasons"]


def test_listed_subjects_must_be_covered():
    objective = (
        "Verify the recovery behavior under cancellation, timeout, and restart faults."
    )
    partial = (
        "Cancellation is handled by the owner token, which aborts the episode cleanly "
        "and proves cleanup with a receipt. This satisfies the verification request."
    )
    receipt = _grade(partial, objective=objective)
    assert len(receipt["listed_subjects"]) >= 2
    assert "listed_subjects_uncovered" in receipt["reasons"]
    complete = (
        "Cancellation aborts the episode and proves cleanup. A timeout trips the "
        "deadline guard and the worker cancels the stage. A restart replays the "
        "boot contract and verifies the invariant before serving again."
    )
    good = _grade(complete, objective=objective)
    assert good["listed_subject_coverage"] >= 0.6
    assert "listed_subjects_uncovered" not in good["reasons"]


def test_structured_list_answers_accepted_with_lower_word_floor():
    objective = "List the steps to verify the release."
    listy = (
        "These are the steps to verify the release:\n"
        "1. Build the app from the exact audited head.\n"
        "2. Sign the build with the stable release identity.\n"
        "3. Run the compound verification turn against the installed app.\n"
        "4. Keep the receipt that proves the release steps completed.\n"
    )
    receipt = _grade(listy, objective=objective)
    assert receipt["structured_output"] is True
    assert receipt["passed"] is True, receipt["reasons"]


def test_token_limit_termination_raises_word_floor():
    # 128+ generated tokens truncated at the limit with sparse words = suspect.
    sparse = "The final answer is forty two."
    receipt = _grade(sparse, generated=160, termination="token_limit")
    assert "insufficient_lexical_content" in receipt["reasons"]


def test_unbalanced_code_fence_is_terminal_fragment():
    text = "Here is the fix:\n```python\nprint('hi')\n"
    receipt = _grade(text)
    assert receipt["terminal_complete"] is False
    balanced = "Here is the fix:\n```python\nprint('hi')\n```"
    assert _grade(balanced)["terminal_complete"] is True


def test_verdict_is_hash_bound_to_text_and_objective():
    receipt = _grade(GOOD_ANSWER)
    assert receipt["text_sha256"] == hashlib.sha256(GOOD_ANSWER.encode()).hexdigest()
    assert receipt["objective_sha256"] == hashlib.sha256(OBJECTIVE.encode()).hexdigest()
    other = _grade(GOOD_ANSWER + " More.", objective=OBJECTIVE)
    assert other["text_sha256"] != receipt["text_sha256"]


def test_non_string_inputs_fail_closed_not_crash():
    receipt = evaluate_latent_output(
        None, generated_tokens="12", termination=7, objective=None
    )
    assert receipt["passed"] is False
    assert "empty_output" in receipt["reasons"]
    assert "invalid_generated_token_count" in receipt["reasons"]


def test_phrase_loop_gate_is_length_aware():
    """Shared reliability gate: a genuine loop is caught at any length, but a
    long technical answer naming its subject a few times is not a loop —
    the absolute 3-repeat rule rejected a correct live deep-reasoning
    answer (CP111 evidence)."""
    from core.conversation.response_reliability import _phrase_loop_reason

    # A real degeneration loop: one phrase dozens of times.
    loop = "the single owner design wins. " * 30
    assert _phrase_loop_reason("q", loop) == "repetitive_phrase_loop"

    # A 450+ word technical answer mentioning its key phrase four times.
    filler_topics = [
        "request admission and queue depth accounting under sustained load",
        "ledger compaction cadence and the audit trail it must preserve",
        "backpressure propagation between the ingest tier and the planner",
        "snapshot cadence for durable state and the recovery point objective",
        "capability lease renewal and the revocation path for stale owners",
        "telemetry sampling budgets and the alert thresholds they feed",
        "cold boot ordering across dependent subsystems and their probes",
        "schema migration staging with reversible cutover checkpoints",
    ]
    sentence_frames = [
        "First, {t} deserves a measured look before any budget lands.",
        "A second concern is {t}, which shapes how failures surface.",
        "Third, engineers must model {t} against realistic traffic.",
        "Another axis involves {t} and its interaction with rollout pace.",
        "Equally important, {t} constrains what operators can promise.",
        "Beyond that, {t} decides whether incidents stay recoverable.",
        "Planning must also cover {t} across every deployment ring.",
        "Finally, {t} rounds out the review with hard evidence.",
    ]
    body_sentences = [
        frame.format(t=topic)
        for frame, topic in zip(sentence_frames, filler_topics)
    ]
    long_answer = (
        "The single-owner design assigns each request to exactly one worker, "
        "preventing duplicate generation from racing the proof ledger. "
        + " ".join(body_sentences)
        + " By contrast, late deduplication merges concurrent attempts after "
        "the fact, so the single-owner design also simplifies verification. "
        "Cancellation revokes the owner token; timeouts trip the deadline "
        "guard; restarts replay the boot contract before serving traffic. "
        "Therefore the single-owner design is the stronger choice, and the "
        "single-owner design should be verified with fault injection."
    )
    from core.conversation.response_reliability import _phrase_loop_reason as plr

    words = len(long_answer.split())
    assert words > 220, words
    assert plr("compare designs", long_answer) == "", (
        "topical repetition in a long answer must not read as a loop"
    )
