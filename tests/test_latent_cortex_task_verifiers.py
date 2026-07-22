"""Contract tests: task-typed verifier guidance inside the episode.

The verifier's job is to make branch selection and hill-climbing prefer
candidates whose answers CHECK OUT. These tests prove each deterministic
check catches its defect class, that scoring composes and renormalizes over
applicable checks only, and that the engine's winner really is chosen by
verified correctness when guidance is on.
"""
from __future__ import annotations

import pytest

from core.brain.llm.latent_cortex.task_verifiers import (
    EpisodeTaskVerifier,
    check_arithmetic_claims,
    check_code_blocks,
    check_facet_coverage,
    check_objective_grounding,
    check_response_contract,
)

# ── Arithmetic recomputation ────────────────────────────────────────────


def test_arithmetic_correct_claims_pass():
    result = check_arithmetic_claims("First 3 + 4 = 7, then 7 * 6 = 42, so 42 - 2 = 40.")
    assert result["checked"] == 3 and result["passed"] == 3
    assert result["score"] == 1.0


def test_arithmetic_confidently_wrong_claims_fail_with_evidence():
    result = check_arithmetic_claims("Clearly 17 * 3 = 41, and 10 + 5 = 15.")
    assert result["checked"] == 2 and result["passed"] == 1
    assert result["failures"] == ["17*3=41 (actual 51)"]
    assert result["score"] == 0.5


def test_arithmetic_ignores_non_integer_division_and_absent_claims():
    result = check_arithmetic_claims("The ratio 7 / 2 = 3.5 is not judged; prose only here.")
    assert result["checked"] == 0 and result["score"] is None


# ── Code syntax ─────────────────────────────────────────────────────────


def test_python_code_must_parse():
    good = "```python\ndef f(x):\n    return x * 2\n```"
    bad = "```python\ndef f(x)\n    return x * 2\n```"
    assert check_code_blocks(good)["score"] == 1.0
    result = check_code_blocks(bad)
    assert result["score"] == 0.0
    assert result["failures"] and "python_syntax" in result["failures"][0]


def test_other_languages_checked_for_balance():
    balanced = "```rust\nfn main() { println!(\"ok\"); }\n```"
    unbalanced = "```rust\nfn main() { println!(\"ok\";\n```"
    assert check_code_blocks(balanced)["score"] == 1.0
    assert check_code_blocks(unbalanced)["score"] == 0.0


def test_no_code_means_not_applicable():
    assert check_code_blocks("plain prose")["score"] is None


# ── Facets + grounding ──────────────────────────────────────────────────


def test_facet_coverage_scores_partial():
    objective = "Compare the designs and explain why one is stronger."
    text = "The first is better because it isolates faults."  # explain yes, compare no
    result = check_facet_coverage(text, objective)
    assert set(result["requested"]) >= {"compare", "explain"}
    assert "explain" in result["satisfied"]
    assert result["score"] is not None and result["score"] < 1.0


def test_grounding_prefers_on_topic_answers():
    objective = "Explain how the scheduler arbitrates conflicting deadlines."
    grounded = check_objective_grounding(
        "The scheduler arbitrates deadlines by priority aging.", objective
    )
    fluent_offtopic = check_objective_grounding(
        "Cats enjoy sunlight and often nap in the afternoon.", objective
    )
    assert grounded["score"] > fluent_offtopic["score"]


# ── Composite scoring ───────────────────────────────────────────────────


def test_composite_renormalizes_over_applicable_checks():
    verifier = EpisodeTaskVerifier("Explain why 6 * 7 = 42 matters.")
    row = verifier.evaluate("It matters because 6 * 7 = 42 anchors the example.")
    assert "code" not in row["applicable_checks"]
    assert row["score"] > 0.8

    neutral = EpisodeTaskVerifier("").evaluate("")
    assert neutral["score"] == 0.5  # nothing verifiable ⇒ neutral, not zero


def test_wrong_arithmetic_ranks_below_correct_arithmetic():
    verifier = EpisodeTaskVerifier("Compute the total and explain the steps.")
    right = verifier("The total follows because 12 + 30 = 42.")
    wrong = verifier("The total follows because 12 + 30 = 44.")
    assert right > wrong


def test_receipt_carries_why_the_winner_won():
    verifier = EpisodeTaskVerifier("Verify the parser and explain the fix.")
    verifier("Broken candidate: 2 + 2 = 5.")
    verifier(
        "The fix works because the parser now rejects empty input; "
        "we verify with a test that asserts 2 + 2 = 4."
    )
    receipt = verifier.to_receipt()
    assert receipt["evaluations"] == 2
    assert receipt["best_score"] == max(receipt["score_trail"])
    assert "arithmetic" in receipt["best_applicable_checks"]


# ── Engine integration: the winner is picked by verification ────────────


def test_engine_selects_branch_by_verifier_score():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.qwen2 import Model, ModelArgs

    from core.brain.llm.latent_cortex.engine import LatentCortexEngine
    from core.brain.llm.latent_cortex.types import (
        BranchConfig,
        CortexConfig,
        RecurrenceConfig,
        WorkspaceConfig,
    )

    args = ModelArgs(
        model_type="qwen2", hidden_size=64, num_hidden_layers=8,
        intermediate_size=128, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=128, num_key_value_heads=2, max_position_embeddings=512,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())

    class StubTokenizer:
        eos_token_id = 0

        def encode(self, text, **kwargs):
            return [ord(c) % 127 + 1 for c in str(text)][:16] or [5]

        def decode(self, ids):
            return " ".join(str(i) for i in ids)

    engine = LatentCortexEngine(
        model,
        StubTokenizer(),
        config=CortexConfig(
            workspace=WorkspaceConfig(n_slots=4, seed=3),
            recurrence=RecurrenceConfig(max_steps=3, min_steps=2),
            branches=BranchConfig(n_branches=2, exchange_interval=2),
            decode_max_tokens=6,
        ),
    )

    scored: list[str] = []

    def verifier(text: str) -> float:
        if text.startswith("Independent consistency check:"):
            result = check_arithmetic_claims(text)
            return float(result["score"])
        scored.append(text)
        return 0.8 if len(scored) % 2 == 0 else 0.2

    result = engine.reason(token_ids=[5, 9, 17, 3, 42], verifier=verifier)
    assert result.ok
    assert len(scored) >= 2, "both branches must be probe-scored"
    highest = max(result.receipt.blind_review["rows"], key=lambda row: row["score"])
    assert result.receipt.selected_branch == highest["branch"], (
        "the branch with the higher blind-review score must win"
    )
    assert result.receipt.decoy_verification["selection_admitted"] is True


# ── Goodhart hardening: cues without substance earn nothing ─────────────


def test_bare_cue_stuffing_does_not_satisfy_facets():
    objective = "Compare the designs, choose the best, and explain why."
    stuffed = "Whereas. I choose. Because. The best."
    result = check_facet_coverage(stuffed, objective)
    assert result["satisfied"] == []
    assert set(result["unsupported_cues"]) >= {"compare", "select", "explain"}


def test_echoed_request_cannot_win_verifier_facet_credit():
    objective = (
        "Compare the early owner design with late deduplication, choose the stronger "
        "architecture, and explain how to verify cancellation and timeout faults."
    )
    echoed = (
        "Compare the early owner design with late deduplication, choose the stronger "
        "architecture, and explain how to verify cancellation and timeout faults. "
        "Both approaches process requests."
    )
    result = check_facet_coverage(echoed, objective)
    assert result["prompt_echo_detected"] is True
    assert result["score"] == 0.0
    assert "select" not in result["satisfied"]
    assert "verify" not in result["satisfied"]


def test_substantive_paraphrase_still_satisfies_facets():
    objective = "Compare the designs and explain why one is stronger."
    text = (
        "The event-driven design is stronger because it isolates faults to "
        "one worker, whereas the shared-loop design lets a stall cascade."
    )
    result = check_facet_coverage(text, objective)
    assert "explain" in result["satisfied"]
    assert "compare" in result["satisfied"]
    assert result["excerpts"]


def test_repetition_loop_scores_below_clean_answer():
    from core.brain.llm.latent_cortex.task_verifiers import check_degeneracy

    loop = ("The scheduler arbitrates deadlines by priority. " * 12).strip()
    clean = (
        "The scheduler arbitrates deadlines with priority aging: each "
        "waiting task gains weight over time, so starvation is bounded. "
        "Conflicts resolve toward the oldest deadline first, and preemption "
        "only fires when the incumbent has passed its soft budget."
    )
    loop_row = check_degeneracy(loop)
    clean_row = check_degeneracy(clean)
    assert loop_row["applicable"] and clean_row["applicable"]
    assert loop_row["factor"] < clean_row["factor"]

    verifier = EpisodeTaskVerifier("Explain how the scheduler arbitrates deadlines.")
    assert verifier(clean) > verifier(loop)


def test_unverified_candidates_are_marked():
    row = EpisodeTaskVerifier("").evaluate("")
    assert row["score"] == 0.5
    assert row["unverified"] is True

    grounded_row = EpisodeTaskVerifier("Explain the fix").evaluate(
        "The fix pins the loop to one owner because double-start raced."
    )
    assert grounded_row["unverified"] is False


# ── Held-out facet calibration (Foundry-graded reliability) ─────────────


def test_facet_reliability_mutes_untrusted_cues():
    """A facet whose detector humans keep overruling earns less when its cue
    fires — 'because'-stuffing stops paying once grading says it's hollow."""
    objective = "Explain why the scheduler prefers the older lease and choose one policy."
    answer = (
        "The scheduler prefers the older lease because a newer lease would "
        "starve long waiters of their turn. I recommend the oldest-first "
        "policy since it bounds waiting time for every queued task."
    )
    trusted = EpisodeTaskVerifier(objective)
    muted = EpisodeTaskVerifier(
        objective, facet_reliability={"explain": 0.25, "select": 1.0}
    )
    trusted_row = trusted.evaluate(answer)
    muted_row = muted.evaluate(answer)
    assert muted_row["checks"]["facets"]["reliability_weighted"] is True
    # Both facets satisfied: unweighted 2/2 = 1.0; muted (0.25+1.0)/(1.25)=1.0
    # still 1.0 when ALL satisfied — the mute shows when the untrusted facet
    # is the ONLY one satisfied:
    partial = "This works because the lease ordering bounds the waiting time for tasks."
    partial_trusted = EpisodeTaskVerifier(objective).evaluate(partial)
    partial_muted = EpisodeTaskVerifier(
        objective, facet_reliability={"explain": 0.25}
    ).evaluate(partial)
    trusted_facets = partial_trusted["checks"]["facets"]["score"]
    muted_facets = partial_muted["checks"]["facets"]["score"]
    assert muted_facets < trusted_facets
    assert trusted_row["score"] >= muted_row["score"]


def test_facet_reliability_rejects_junk_entries():
    verifier = EpisodeTaskVerifier(
        "compare the two options",
        facet_reliability={
            "compare": True,          # bool is not a weight
            "unknown_facet": 0.5,     # not a facet
            "explain": 2.0,           # out of range
            "select": 0.8,            # valid
        },
    )
    assert verifier.facet_reliability == {"select": 0.8}


def test_receipt_exposes_gradeable_facet_judgments():
    objective = "Compare eager and lazy loading and choose one for the cache."
    verifier = EpisodeTaskVerifier(objective)
    verifier.evaluate(
        "Eager loading warms every entry up front, whereas lazy loading pays "
        "on first touch. I recommend lazy loading for this cache because the "
        "key space is sparse and mostly cold."
    )
    receipt = verifier.to_receipt()
    judgments = {row["facet"]: row for row in receipt["facet_judgments"]}
    assert {"compare", "select"} <= set(judgments)
    assert judgments["compare"]["satisfied"] is True
    assert "whereas" in judgments["compare"]["excerpt"].lower()
    assert judgments["select"]["satisfied"] is True
    assert receipt["facet_reliability"] == {}


def test_public_response_contract_controls_branch_score_and_receipt():
    contract = '{"count":int,"witness":list[int]}'
    valid = 'FINAL_ANSWER: {"count":2,"witness":[3,5]}'
    invalid = 'FINAL_ANSWER: {"count":"2","witness":[3,5]}'
    verifier = EpisodeTaskVerifier(
        "Find the witness.",
        response_contract=contract,
    )

    assert verifier(valid) > verifier(invalid)
    assert check_response_contract(valid, contract)["valid"] is True
    receipt = verifier.to_receipt()
    assert receipt["response_contract_required"] is True
    assert receipt["response_contract_satisfied"] is True
