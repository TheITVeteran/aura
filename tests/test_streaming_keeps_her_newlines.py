"""Every newline she generated was deleted in flight.

Live 2026-07-28. Asked for three fruits, one per line::

    AppleBananaOrange

Asked to echo a Python block verbatim::

    ```pythonimport randomdef f(x): return x + 1```

``_truncate_role_continuation`` runs on the ENTIRE accumulated buffer after
every token, and it ended in ``.strip()``. So the moment the model emitted a
newline, the buffer ended with one, the strip removed it, and the next token
landed flush against the previous word. Apple → "Apple\\n" → stripped →
"Apple" → "AppleBanana".

Mid-stream, trailing whitespace is not trailing. It is the next line
beginning.

This is why the 2048 reconstruction failed five times with "invalid syntax at
line 1" on code the model had written correctly — in Python whitespace *is*
the syntax — and it silently flattened every paragraph and list she has ever
produced.
"""
from __future__ import annotations

import ast

import pytest

from core.brain.llm.mlx_worker import _truncate_role_continuation


def _stream(tokens: list[str]) -> str:
    """Feed tokens the way the generation loop does."""
    buffer = ""
    for token in tokens:
        buffer += token
        buffer, _ = _truncate_role_continuation(buffer)
    return buffer


def test_the_live_failure_three_fruits_one_per_line() -> None:
    assert _stream(["Apple", "\n", "Banana", "\n", "Orange"]) == "Apple\nBanana\nOrange"


def test_generated_python_survives_and_parses() -> None:
    code = _stream(
        ["import random", "\n", "\n", "def f(x):", "\n", "    return x + 1", "\n"]
    )
    ast.parse(code)
    assert "\n    return x + 1" in code, "indentation is the syntax"


def test_blank_lines_between_paragraphs_survive() -> None:
    assert _stream(["First.", "\n", "\n", "Second."]) == "First.\n\nSecond."


@pytest.mark.parametrize(
    "tokens",
    [
        ["a", "\n"],
        ["a", "\n", "\n"],
        ["a", "\n", " ", "b"],
        ["  ", "a", "\n", "b"],
    ],
)
def test_no_token_boundary_eats_a_newline(tokens: list[str]) -> None:
    assert _stream(tokens).count("\n") == "".join(tokens).count("\n")


# ── What the strip was there for, kept ────────────────────────────────────

def test_a_finished_reply_is_still_trimmed() -> None:
    trimmed, _ = _truncate_role_continuation("  Hello there.  \n\n", final=True)
    assert trimmed == "Hello there."


def test_role_continuation_is_still_clipped() -> None:
    """The function's actual job must not regress."""
    text = "The answer is four.\nUser: and what about five?"
    clipped, hit = _truncate_role_continuation(text, final=True)
    assert hit
    assert "and what about five" not in clipped
    assert "The answer is four." in clipped


def test_clipping_still_fires_mid_stream() -> None:
    """The flag is the contract: the generation loop breaks on it.

    The function clips the buffer and reports the hit; it is the caller that
    stops pulling tokens. Feeding it more afterwards is not something the real
    loop does.
    """
    buffer = ""
    hits = []
    for token in ["The answer is four.", "\n", "User:"]:
        buffer += token
        buffer, hit = _truncate_role_continuation(buffer)
        hits.append(hit)
    assert hits == [False, False, True], "the hit must fire on the role token"
    assert buffer == "The answer is four."
