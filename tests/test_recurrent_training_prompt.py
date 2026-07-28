"""Contract tests for recurrent training prompt rendering."""

from __future__ import annotations

from types import SimpleNamespace

from core.learning.recurrent_training_prompt import (
    RECURRENT_TRAINING_COT_PREAMBLE,
    answer_contract_instruction,
    render_recurrent_training_prompt,
    render_recurrent_training_prompt_text,
)


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"add_generation_prompt": True, "tokenize": False}
        return f"<user>{messages[0]['content']}</user><assistant>"

    def encode(self, text):
        return list(text.encode("ascii"))


def test_prompt_renderer_binds_cot_answer_keys_and_exact_tokens() -> None:
    task = SimpleNamespace(
        prompt="Compute the value.",
        expected={"result": 7, "trace": [1, 7]},
    )

    rendered, tokens = render_recurrent_training_prompt(
        _Tokenizer(),
        task,
        include_chain_of_thought=True,
    )

    assert rendered.startswith(f"<user>{RECURRENT_TRAINING_COT_PREAMBLE}\n\n")
    assert "Use exactly these JSON keys: result, trace." in rendered
    assert tokens == tuple(rendered.encode("ascii"))


def test_answer_contract_does_not_expose_values() -> None:
    task = SimpleNamespace(prompt="Compute.", expected={"secret": 81273})

    instruction = answer_contract_instruction(task)

    assert "secret" in instruction
    assert "81273" not in instruction


def test_text_only_renderer_does_not_require_encode() -> None:
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **_kwargs: messages[0]["content"]
    )
    task = SimpleNamespace(prompt="Compute.", expected={"result": 1})

    rendered = render_recurrent_training_prompt_text(
        tokenizer,
        task,
        include_chain_of_thought=False,
    )

    assert rendered.endswith("\n\nCompute.")
