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
        scored.append(text)
        # Prefer whichever probe this deterministic stub sees SECOND —
        # proving selection follows the score, not branch order.
        return float(len(scored))

    result = engine.reason(token_ids=[5, 9, 17, 3, 42], verifier=verifier)
    assert result.ok
    assert len(scored) >= 2, "both branches must be probe-scored"
    assert result.receipt.selected_branch == 1, (
        "the branch with the higher verified score must win"
    )
