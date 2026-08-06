#!/usr/bin/env python3
"""A real local model behind every arm of the capability ablation.

Operationally: this loads one MLX model once and answers every condition with
it, under identical decode settings and an identical token budget, so the only
thing that differs between arms is what they were allowed to see.

One model, loaded once, is not an implementation convenience — it is the
control. Two loads can differ in quantisation, in cache state, in sampler seed,
and any of those would be an uncontrolled variable sitting exactly where the
experiment's answer is supposed to come from. The retracted agi_live bundle
failed on a coarser version of this: different token budgets, and a solver on
one side only.

Arms differ ONLY in the prompt they are handed:

    stateless      the final question, alone
    long_context   the full transcript, verbatim, in the context window
    full           the same information routed through Aura's memory assembly

The third is the one that matters, and it is deliberately not "the same string
with a different label" — it goes through the real retrieval path, so a result
here is about the machinery rather than about string formatting.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_SYSTEM = (
    "You are answering a question from a conversation. Reply with the answer "
    "only — no preamble, no explanation. If the conversation does not contain "
    "the answer, reply exactly: I do not know."
)


class MemoryPathUnavailable(RuntimeError):
    """Raised rather than falling back to the transcript.

    The first version of this function caught ImportError and returned
    "\\n".join(history) — the long_context prompt with different whitespace. It
    named a function that does not exist (`score_and_rank`), so every run took
    the fallback, and the `full_architecture` arm silently BECAME its own
    control. It then measured a −0.125 delta against itself and reported it as
    a result about Aura's memory.

    That is the exact failure this runner exists to prevent, committed by the
    runner. A missing memory path must stop the experiment, because an
    experiment whose treatment arm has quietly turned into the control is worse
    than no experiment: it produces a number, and numbers get cited.
    """


def _assembled_by_memory(history: list[str], question: str, *, budget_turns: int) -> str:
    """Route the history through Aura's real retrieval, within the SAME budget.

    `budget_turns` is not a detail — it is the control. An earlier version
    passed `top_k=len(memories)`, so this arm received all 41 turns while
    long_context received its 12-turn window, and the +1.000 it produced was a
    context-size difference wearing an architecture label. Exactly the defect
    that voided the agi_live bundle, committed by the tool built to prevent it.

    With the budget equalised, the arms differ only in HOW they spend it:
    long_context takes the most recent N turns, retrieval takes the N most
    relevant. That is a real architectural question and this can answer it.
    """
    from core.memory.rag import retrieve_memories

    memories = [
        {"content": text, "id": f"turn-{index}", "metadata": {"turn": index}}
        for index, text in enumerate(history)
    ]
    ranked = retrieve_memories(question, memories, top_k=max(1, budget_turns))
    if not ranked:
        # A retrieval that returns nothing is a real answer about the
        # retrieval, not a reason to hand the model the transcript instead. The
        # arm keeps what Aura would actually have had: nothing.
        return ""
    return "\n".join(str(item.get("content", "")) for item in ranked)


def make_mlx_responder(
    *, model_id: str, max_output_tokens: int, budget_turns: int
) -> Callable[[str, Any, int, list[str]], str]:
    """Load once, answer every arm with the same weights and the same budget."""
    from mlx_lm import generate, load

    model, tokenizer = load(model_id)

    def respond(condition: str, task: Any, _turn: int, history: list[str]) -> str:
        question = task.turns[-1]
        if condition == "stateless":
            body = question
        elif condition == "full_architecture":
            body = (
                f"{_assembled_by_memory(list(history), question, budget_turns=budget_turns)}"
                f"\n\n{question}"
            )
        else:
            body = "\n".join([*history, question])

        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": body},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Identical for every arm: same max_tokens, same (greedy) sampler.
        return str(
            generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_output_tokens,
                verbose=False,
            )
        ).strip()

    return respond


__all__ = ["make_mlx_responder"]
