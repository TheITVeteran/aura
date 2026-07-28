"""A generation that ran out of tokens threw away the whole implementation.

Live 2026-07-28, the first 2048 build that got past the covenant and actually
reached synthesis::

    I didn't get 2048 rebuilt, and I'm not going to say I did. 0/14 held-out
    positions reproduced. RuntimeError: LLM returned no valid Python source
    (invalid syntax); model said: "`pythonimport randomdef move(case): board =
    case['board'] direction = case['direction'] score = 0 ..."

Two defects in one line.

The extractor's fence pattern requires an opening ``` AND a closing one. A
generation that hits its token ceiling mid-block has only the opening, so the
pattern matched nothing, the raw text — fence marker and all — went to
``ast.parse``, and it reported invalid syntax on line 1. Running out of tokens
is an ordinary event; losing an otherwise usable implementation to it is not.

And the failure preview collapsed whitespace, which is why the quoted output
above reads as "`pythonimport randomdef move(...)". That is the diagnosis
destroying itself: when the complaint is a *syntax* error, the line structure
is the evidence, and a reader cannot tell a truncated fence from a model that
genuinely emitted no newlines.
"""
from __future__ import annotations

import ast

import pytest

from core.brain.llm.code_generator import extract_python_code

COMPLETE = "```python\nimport random\n\n\ndef move(case):\n    return case\n```"
TRUNCATED = "```python\nimport random\n\n\ndef move(case):\n    board = case['board']\n    score = 0"
BARE = "import random\n\n\ndef move(case):\n    return case"
PROSE_THEN_TRUNCATED = (
    "Here is the implementation:\n\n```python\nimport random\n\n\ndef move(case):\n    return case"
)


@pytest.mark.parametrize(
    ("label", "raw"),
    [
        ("complete fence", COMPLETE),
        ("truncated fence — the live failure", TRUNCATED),
        ("no fence at all", BARE),
        ("prose, then a truncated fence", PROSE_THEN_TRUNCATED),
    ],
)
def test_every_shape_yields_parseable_python(label: str, raw: str) -> None:
    code = extract_python_code(raw)
    assert code, label
    ast.parse(code)  # raises if the extractor handed back the fence marker
    assert not code.lstrip().startswith("```"), label


def test_the_fence_marker_never_survives_into_the_code() -> None:
    assert "```" not in extract_python_code(TRUNCATED)


def test_a_complete_fence_is_still_preferred_over_the_tail() -> None:
    """Two blocks: the closed one is the real answer, not everything after."""
    raw = "```python\nimport random\n```\n\nand some trailing prose"
    code = extract_python_code(raw)
    assert code.strip() == "import random"
    assert "trailing prose" not in code


def test_the_longest_complete_block_still_wins() -> None:
    raw = (
        "```python\nx = 1\n```\n"
        "```python\nimport random\n\n\ndef move(case):\n    return case\n```"
    )
    code = extract_python_code(raw)
    assert "def move" in code


def test_nothing_usable_still_returns_nothing() -> None:
    assert extract_python_code("") == ""
    assert extract_python_code("   \n  ") == ""


# ── The preview has to stay readable ──────────────────────────────────────

def test_the_failure_preview_keeps_its_line_structure() -> None:
    """Collapsed whitespace is how the diagnosis destroyed itself."""
    import inspect

    from core.brain.llm import code_generator

    source = inspect.getsource(code_generator.LLMCodeGenerator.generate_async)
    assert "' '.join(code.split())" not in source, "the preview must not collapse newlines"
    assert 'splitlines()[:8]' in source
    assert "exc.lineno" in source, "a syntax error should say which line"
