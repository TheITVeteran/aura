"""The phrase-pin classifier, held to known-good cases.

``tools/lint_phrase_pinned_tests.py`` decides which test assertions are debt.
It counted 454 and the true number was 280 — it was folding in two things that
are not pinned production wording:

* **an assert's own message.** ``assert guard < reg, "shutdown guard must
  precede service registration"`` pins nothing. The old regex took the first
  literal on the line, which for that shape is the explanation shown when the
  test fails — the test's own prose.
* **an unclosed call site.** ``_self_health_answer_or_empty(`` and
  ``getUserMedia({`` are structural assertions that a call exists. They can
  never parse, because they are half an expression, so the parse-based
  discriminator rejected them and they were counted as sentences.

A gate that miscounts by 62% is a gate whose number nobody can act on, and
loosening it without a test is how it drifts back. These are the cases that
define it — including the two REAL pins the tool's own docstring names as the
reason it exists, which must still be caught.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOL = REPO_ROOT / "tools" / "lint_phrase_pinned_tests.py"


def _load():
    spec = importlib.util.spec_from_file_location("lint_phrase_pins", _TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Every case is a source read followed by one assertion.
#:
#: Written plainly. The scanner would otherwise count the examples below as
#: four real pins — it looks for this call name on a line and reads the
#: assertions under it, which is exactly what a fixture file full of example
#: assertions looks like. Rather than contort these into something the scanner
#: cannot see, the tool skips this file by name (``_SELF_TEST``). An
#: instrument should not measure its own calibration bench.
_READ = "source = inspect.getsource(core_thing)\n"

_CAUGHT = [
    pytest.param('assert "There are 3 .py files" in source', id="a_sentence"),
    pytest.param('assert "just a language model" in source', id="a_self_description"),
    pytest.param('assert "I am here now" in source, "wording changed"', id="prose_with_a_message"),
    pytest.param('assert "what is the capital of Peru" in source', id="a_prompt"),
]

_NOT_CAUGHT = [
    pytest.param(
        'assert guard < reg, "shutdown guard must precede service registration"',
        id="the_asserts_own_message",
    ),
    pytest.param('assert "_self_health_answer_or_empty(" in source', id="an_unclosed_call"),
    pytest.param('assert "getUserMedia({" in source', id="an_unclosed_js_call"),
    pytest.param('assert "replace(" in source', id="a_bare_call_name"),
    pytest.param(
        'assert "from core.container import ServiceContainer" in source',
        id="a_layering_invariant",
    ),
    pytest.param('assert "reverse=True" in source', id="a_keyword_argument"),
    pytest.param('assert "def _drain_phi_residual_ring" in source', id="a_function_exists"),
]


@pytest.mark.parametrize("assertion", _CAUGHT)
def test_real_wording_is_still_counted(assertion: str) -> None:
    """The tool must not be loosened into uselessness."""
    module = _load()
    assert module._phrase_pins(_READ + assertion) == 1, assertion


@pytest.mark.parametrize("assertion", _NOT_CAUGHT)
def test_structural_assertions_are_not_debt(assertion: str) -> None:
    """The honest use the tool's own docstring says it relies on heavily."""
    module = _load()
    assert module._phrase_pins(_READ + assertion) == 0, assertion


def test_a_fixture_read_is_never_counted() -> None:
    module = _load()
    source = 'body = (tmp_path / "out.txt").read_text()\nassert "hello there" in body'
    assert module._phrase_pins(source) == 0


def test_a_read_that_names_no_production_package_is_ignored() -> None:
    module = _load()
    source = 'body = something.read_text()\nassert "hello there" in body'
    assert module._phrase_pins(source) == 0


def test_the_window_is_bounded() -> None:
    """An assertion far below a source read is not fed by it."""
    module = _load()
    far = _READ + "\n" * 40 + 'assert "There are 3 .py files" in source'
    assert module._phrase_pins(far) == 0
