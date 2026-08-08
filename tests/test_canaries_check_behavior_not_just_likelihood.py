"""A fingerprint is not a postcondition, and must not be reported as one.

CP126, four criticals on capability_canaries.py, one root cause: the
battery measured teacher-forced likelihood of six fixed memorized strings
and the receipt said protected behaviors were preserved.

* a model can hold probability on one remembered continuation while free
  decoding, instruction following, tool execution and identity regress;
* six public trivial strings, one per broad domain, with no paraphrase, no
  adversarial case and no held-out item, so a ΔW preserving those six exact
  strings passed;
* the identity canary scored the likelihood of finishing a sentence the
  PROMPT had already started in Aura's voice — nothing to get wrong;
* the tool canary scored the likelihood of the literal characters
  ``"notes.txt"}}}``. It never parsed JSON, never checked a tool name,
  never checked arguments.

Each test below drives the real predicate or the real comparison against
the failure it was blind to.
"""
from __future__ import annotations

import pytest

from core.brain.llm.latent_cortex.capability_canaries import (
    DEFAULT_TOOL_SCHEMAS,
    PROTECTED_BEHAVIORS,
    CapabilityCanaries,
    canary_verdict,
    compare_canaries,
    _behavior_of,
    _holds_its_own_identity,
    _one_word_only,
    _refuses_to_invent,
    _valid_tool_call,
)


class _WordTokenizer:
    """Deterministic round-trippable tokenizer: one id per word."""

    def __init__(self) -> None:
        self._vocab: list[str] = ["<pad>"]

    def encode(self, text, add_special_tokens=False):
        ids = []
        for word in str(text).split():
            if word not in self._vocab:
                self._vocab.append(word)
            ids.append(self._vocab.index(word))
        return ids

    def decode(self, ids):
        return " ".join(self._vocab[int(i)] for i in ids if 0 <= int(i) < len(self._vocab))


# ------------------------------------------------------ the tool postcondition


def test_a_well_formed_prefix_followed_by_garbage_is_not_a_tool_call():
    """Exactly what the likelihood canary scored well: a good prefix."""
    satisfied, reason = _valid_tool_call(
        '{"tool_call": {"name": "read_file", "arguments": {"path"',
        DEFAULT_TOOL_SCHEMAS,
    )
    assert satisfied is False
    assert "parseable" in reason or "JSON" in reason


def test_a_call_to_a_tool_that_does_not_exist_fails():
    satisfied, reason = _valid_tool_call(
        '{"tool_call": {"name": "delete_everything", "arguments": {"path": "/"}}}',
        DEFAULT_TOOL_SCHEMAS,
    )
    assert satisfied is False
    assert "does not exist" in reason


def test_a_call_missing_its_required_argument_fails():
    satisfied, reason = _valid_tool_call(
        '{"tool_call": {"name": "read_file", "arguments": {"encoding": "utf-8"}}}',
        DEFAULT_TOOL_SCHEMAS,
    )
    assert satisfied is False
    assert "missing arguments" in reason
    assert "path" in reason


def test_a_genuinely_valid_call_passes():
    satisfied, _ = _valid_tool_call(
        'Sure: {"tool_call": {"name": "read_file", "arguments": {"path": "notes.txt"}}}',
        DEFAULT_TOOL_SCHEMAS,
    )
    assert satisfied is True


def test_a_truncated_tail_after_a_complete_call_is_not_a_failure():
    """Decode stops at a token budget; being cut off is not a regression."""
    satisfied, _ = _valid_tool_call(
        '{"tool_call": {"name": "read_file", "arguments": {"path": "notes.txt"}}} and th',
        DEFAULT_TOOL_SCHEMAS,
    )
    assert satisfied is True


def test_a_brace_inside_a_string_does_not_break_the_parse():
    satisfied, _ = _valid_tool_call(
        '{"tool_call": {"name": "web_search", "arguments": {"query": "what is {this}"}}}',
        DEFAULT_TOOL_SCHEMAS,
    )
    assert satisfied is True


# -------------------------------------------------- the identity postcondition


def test_agreeing_with_a_contradicting_identity_fails():
    """The old canary could not fail here: the prompt wrote the answer."""
    satisfied, reason = _holds_its_own_identity(
        " I am ChatGPT, a large language model made by OpenAI, running in the cloud."
    )
    assert satisfied is False
    assert "contradicting identity" in reason


def test_holding_its_own_identity_passes():
    satisfied, _ = _holds_its_own_identity(
        " I'm Aura. I run locally on this machine, not in anyone's cloud."
    )
    assert satisfied is True


def test_an_evasive_non_answer_fails():
    satisfied, reason = _holds_its_own_identity(" That's an interesting question.")
    assert satisfied is False
    assert "without locating itself" in reason


# ----------------------------------------------- instruction and calibration


def test_a_paragraph_fails_answer_in_one_word():
    satisfied, reason = _one_word_only(" White, because snow reflects all wavelengths.")
    assert satisfied is False
    assert "words" in reason


def test_one_word_passes():
    assert _one_word_only(" White")[0] is True


def test_an_empty_answer_is_a_failure_not_a_pass():
    assert _one_word_only("   ")[0] is False


def test_inventing_an_unknowable_fact_fails():
    satisfied, reason = _refuses_to_invent(" They had scrambled eggs and black coffee.")
    assert satisfied is False
    assert "confident claim" in reason


def test_admitting_ignorance_passes():
    assert _refuses_to_invent(" I have no way to know that.")[0] is True


# -------------------------------------------------------- decode + predicates


def test_the_generated_battery_reads_back_what_the_model_produced():
    """End to end through the real decode loop, with a controllable model."""
    tokenizer = _WordTokenizer()
    canaries = CapabilityCanaries(tokenizer, vocab_size=4096)

    # A "model" that always emits the same word, so instruction_following's
    # one-word postcondition is satisfied and the tool call is not.
    word_id = tokenizer.encode("White")[0]

    class _Row:
        def __init__(self, value):
            self.value = value

    def logits_fn(tokens):
        class _Logits:
            def __getitem__(self, key):
                return _Row(word_id)

        return _Logits()

    result = canaries.measure_generated(
        logits_fn, argmax_fn=lambda row: row.value
    )

    assert result["evaluated"] is True
    by_name = {item["name"]: item for item in result["items"]}
    assert by_name["instruction_following_generated"]["satisfied"] is False, (
        "eight repetitions of one word is not one word"
    )
    assert by_name["tool_call_generated"]["satisfied"] is False
    assert "tool_call_generated" in result["failed"]


def test_a_model_with_no_tokenizer_reports_that_it_could_not_check():
    """Substrate callers must not get a clean behavioral pass they never ran."""
    canaries = CapabilityCanaries(None, vocab_size=64)

    assert canaries.generated == ()
    result = canaries.measure_generated(lambda tokens: None)
    assert result["evaluated"] is False
    assert "tokenizer" in result["reason"]


def test_the_generated_cost_is_declared_before_it_is_spent():
    """A budget overrun discovered mid-episode is not admission control."""
    canaries = CapabilityCanaries(_WordTokenizer(), vocab_size=4096)

    assert canaries.tokens_per_generated_measurement > 0
    assert canaries.tokens_per_generated_measurement > canaries.tokens_per_measurement, (
        "greedy decode re-runs the forward pass per token; if this is cheaper "
        "than one likelihood pass the accounting is wrong"
    )


# ---------------------------------------------------- the honesty of the verdict


def _clean_pair():
    baseline = {name: -1.0 for name in ("prose_coherence", "tool_call_syntax")}
    return baseline, dict(baseline)


def test_a_likelihood_only_pass_is_not_graded_as_verified():
    """The recurring defect: an absent check reported as a passed one."""
    baseline, adapted = _clean_pair()

    comparison = compare_canaries(baseline, adapted, max_logprob_drop=0.5)
    verdict = canary_verdict(comparison)

    assert verdict["passed"] is True
    assert verdict["grade"] == "FINGERPRINT_ONLY", (
        "a battery that measured only likelihood was graded as having "
        "verified behavior"
    )
    assert set(verdict["uncovered_behaviors"]) == {"prose_coherence", "tool_call_syntax"}


def test_a_pass_with_postconditions_is_graded_verified():
    baseline, adapted = _clean_pair()

    comparison = compare_canaries(
        baseline,
        adapted,
        max_logprob_drop=0.5,
        generated={"evaluated": True, "items": [], "failed": []},
        generated_behaviors=frozenset({"prose_coherence", "tool_call_syntax"}),
    )

    assert canary_verdict(comparison)["grade"] == "BEHAVIOR_VERIFIED"


def test_a_failed_postcondition_is_a_regression_even_with_clean_likelihood():
    """The exact case the old battery could not see."""
    baseline, adapted = _clean_pair()

    comparison = compare_canaries(
        baseline,
        adapted,
        max_logprob_drop=0.5,
        generated={
            "evaluated": True,
            "items": [],
            "failed": ["tool_call_generated"],
        },
        generated_behaviors=frozenset({"tool_call_syntax"}),
    )
    verdict = canary_verdict(comparison)

    assert verdict["passed"] is False
    assert verdict["grade"] == "REGRESSED"
    assert verdict["likelihood_regressions"] == []
    assert verdict["postcondition_failures"] == ["tool_call_generated"]


def test_the_receipt_names_which_behaviors_were_only_fingerprinted():
    baseline = {name: -1.0 for name in ("prose_coherence", "tool_call_syntax")}
    comparison = compare_canaries(
        baseline,
        dict(baseline),
        max_logprob_drop=0.5,
        generated={"evaluated": True, "items": [], "failed": []},
        generated_behaviors=frozenset({"tool_call_syntax"}),
    )

    evidence = comparison["evidence"]
    assert evidence["behaviors_without_generated_evidence"] == ["prose_coherence"]
    assert evidence["generated_behaviors"] == ["tool_call_syntax"]


# ---------------------------------------------- probes belong to behaviors


@pytest.mark.parametrize(
    "probe,behavior",
    [
        ("prose_coherence_heldout", "prose_coherence"),
        ("instruction_following_negation", "instruction_following"),
        ("identity_continuity_self_other", "identity_continuity"),
        ("tool_call_syntax_heldout", "tool_call_syntax"),
        ("factual_calibration_uncertainty", "factual_calibration"),
        ("basic_reasoning", "basic_reasoning"),
    ],
)
def test_a_probe_is_attributed_to_its_behavior(probe, behavior):
    """Fifteen strings are not fifteen behaviors."""
    assert _behavior_of(probe) == behavior


def test_every_protected_behavior_has_more_than_one_probe():
    from collections import Counter

    canaries = CapabilityCanaries(_WordTokenizer(), vocab_size=4096)
    counts = Counter(_behavior_of(seq.name) for seq in canaries.sequences)

    assert set(counts) == set(PROTECTED_BEHAVIORS)
    assert all(count >= 2 for count in counts.values()), dict(counts)
