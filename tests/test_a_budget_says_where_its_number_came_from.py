"""The prompt budget's chars-per-token ratio, and what it admits about itself.

Every budget in the assembler is a character count converted from a token
window by one number. It was `* 4`, annotated "Rough estimation" — the English
prose average, applied to prompts made of code, JSON receipts and file paths,
which run nearer two to three characters per token. A prompt built to fit could
be half again over the real window, and overflow is not the symmetric failure:
the backend drops from the head, and the head is the identity lock and the
structural constraint block.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.brain.llm import token_budget_evidence as tbe

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clean_ratio():
    tbe.reset_for_test()
    yield
    tbe.reset_for_test()


def test_an_unobserved_ratio_says_it_is_assumed():
    ratio = tbe.chars_per_token()

    assert ratio.source is tbe.RatioSource.ASSUMED
    assert ratio.measured is False
    assert ratio.observations == 0


def test_the_assumption_is_below_the_prose_average():
    """Under-filling wastes context; over-filling deletes the constraints.
    Those costs are not comparable, so the guess leans to the survivable one."""
    assert tbe.ASSUMED_CHARS_PER_TOKEN < 4.0


def test_enough_observations_make_it_measured():
    for _ in range(tbe.MIN_OBSERVATIONS):
        assert tbe.observe_prompt_tokenization(300, 100) is True

    ratio = tbe.chars_per_token()
    assert ratio.source is tbe.RatioSource.MEASURED
    assert ratio.measured is True
    assert ratio.ratio == pytest.approx(3.0)
    assert ratio.observations == tbe.MIN_OBSERVATIONS


def test_one_prompt_is_not_a_measurement():
    tbe.observe_prompt_tokenization(300, 100)

    assert tbe.chars_per_token().source is tbe.RatioSource.ASSUMED


def test_an_impossible_ratio_is_refused_not_averaged():
    """A ratio outside the plausible band means the caller paired a string with
    a token count from a different string. Averaging that in corrupts every
    later budget, so it is discarded and recorded."""
    assert tbe.observe_prompt_tokenization(100_000, 3) is False
    assert tbe.observe_prompt_tokenization(3, 100_000) is False
    assert tbe.chars_per_token().observations == 0


def test_zero_and_negative_counts_are_refused():
    assert tbe.observe_prompt_tokenization(0, 10) is False
    assert tbe.observe_prompt_tokenization(10, 0) is False
    assert tbe.observe_prompt_tokenization(-5, 10) is False


def test_the_assembler_no_longer_multiplies_by_four():
    source = (ROOT / "core" / "brain" / "llm" / "context_assembler.py").read_text("utf-8")
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "char_limit" not in targets:
            continue
        rendered = ast.dump(node.value)
        assert "tokens_to_chars" in rendered, ast.unparse(node)


def test_the_worker_reports_what_it_already_tokenized():
    """Nothing in the evidence module tokenizes. It is fed by the process that
    encoded the prompt anyway — the assembler must not load a tokenizer in the
    process that serves conversation."""
    worker = (ROOT / "core" / "brain" / "llm" / "mlx_worker.py").read_text("utf-8")
    module = (ROOT / "core" / "brain" / "llm" / "token_budget_evidence.py").read_text("utf-8")

    assert "observe_prompt_tokenization" in worker
    assert "tokenizer" not in module.split('"""', 2)[2], "the evidence module tokenizes"
