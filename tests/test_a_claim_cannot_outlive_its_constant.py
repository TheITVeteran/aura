"""A claim citing a constant is checked against the constant.

CLAIMS_MATRIX.md is what an external reviewer reasons from. Row 4a stated, as
current fact, that `DEFAULT_ALPHA` is 5.0 and `_INJECTION_ALPHA_CEILING` clips
it to 3.0, and concluded that affect steering ships below the magnitude at
which it changes a token. c7dcc548a had already replaced both nine days
earlier — alpha became a fraction of the residual stream, 0.2 with a 0.6
ceiling, measured on two models — which is the closure condition row 4a itself
named.

A reviewer read the matrix, reasoned correctly from what it said, and reached a
conclusion that had stopped being true. `make evidence-integrity` refuses a
claim whose evidence was retracted; nothing refused a claim whose evidence is a
number in a file that has since changed, because a record that understates
looks like caution.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tools.verify_claim_constants import (
    collect_constants,
    read_assertions,
    verify,
)

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "CLAIMS_MATRIX.md"
SCANNED = ("core", "interface", "tools", "training")


def test_the_live_matrix_cites_no_stale_constant():
    findings = verify(MATRIX, SCANNED)

    assert findings == [], "\n".join(f.render() for f in findings)


def test_the_matrix_actually_asserts_something():
    """A gate that checks nothing passes for the wrong reason."""
    assertions = read_assertions(MATRIX)

    assert len(assertions) >= 5, f"only {len(assertions)} constants are checked"


def test_the_steering_constants_are_among_them():
    """These are the ones that went stale, so they are the ones that must be
    covered from now on."""
    names = {a.name for a in read_assertions(MATRIX)}

    assert "DEFAULT_ALPHA" in names
    assert "_INJECTION_ALPHA_CEILING" in names


def test_a_stale_value_is_caught(tmp_path):
    matrix = tmp_path / "CLAIMS_MATRIX.md"
    matrix.write_text("| 4a | shipped `DEFAULT_ALPHA` is 5.0, which is too small |\n")

    findings = verify(matrix, SCANNED)

    assert len(findings) == 1
    assert findings[0].kind == "mismatch"
    assert findings[0].name == "DEFAULT_ALPHA"
    assert findings[0].asserted == 5.0
    assert any("0.2" in actual for actual in findings[0].actual)


def test_the_current_value_passes(tmp_path):
    matrix = tmp_path / "CLAIMS_MATRIX.md"
    matrix.write_text("| 4a | shipped `DEFAULT_ALPHA` is 0.2 |\n")

    assert verify(matrix, SCANNED) == []


def test_a_deleted_constant_is_caught(tmp_path):
    """A claim can go stale by naming something that no longer exists at all."""
    matrix = tmp_path / "CLAIMS_MATRIX.md"
    matrix.write_text("| 9 | `_RETIRED_KNOB_THAT_NEVER_EXISTED` is 4.0 |\n")

    findings = verify(matrix, SCANNED)

    assert len(findings) == 1
    assert findings[0].kind == "missing"


@pytest.mark.parametrize(
    "line",
    [
        "measured at magnitude 150 against 2000 permutations",
        "n=160 at p=0.0005, paired 95% CI [1.000, 1.000]",
        "the responder scored 0.500 and the gate 0.800",
        "`tools/affect_causality_ablation.py --alpha 3` reproduces it",
    ],
)
def test_prose_containing_numbers_is_not_read_as_an_assertion(line, tmp_path):
    """A gate that cries wolf gets turned off. An assertion needs an explicit
    copula between a backticked identifier and the number."""
    matrix = tmp_path / "CLAIMS_MATRIX.md"
    matrix.write_text(line + "\n")

    assert read_assertions(matrix) == []


def test_a_historical_statement_is_not_an_assertion_about_now(tmp_path):
    """"`X` was 2.0" is a record of what changed, not a claim about the
    present, and reading it as one would make correcting a claim impossible."""
    matrix = tmp_path / "CLAIMS_MATRIX.md"
    matrix.write_text("| 4a | `_SYNC_STALE_AFTER_S` was 2.0 and a generation exceeds it |\n")

    assert read_assertions(matrix) == []


def test_a_lowercase_prose_word_is_not_a_constant(tmp_path):
    matrix = tmp_path / "CLAIMS_MATRIX.md"
    matrix.write_text("| 4 | the `responder` is 1 of three arms |\n")

    assert read_assertions(matrix) == []


def test_a_name_defined_in_several_modules_is_satisfied_by_any(tmp_path):
    """The matrix names a constant, not a module; demanding a unique
    definition would fail on names that are legitimately reused."""
    constants = collect_constants(SCANNED)
    reused = [name for name, defs in constants.items() if len(defs) > 1]

    assert reused, "no reused constant names, so this property is untested"
    name = reused[0]
    value = constants[name][0][1]
    matrix = tmp_path / "CLAIMS_MATRIX.md"
    matrix.write_text(f"| x | `{name}` is {value:g} |\n")

    assert verify(matrix, SCANNED) == []


def test_negative_and_integer_constants_resolve():
    constants = collect_constants(SCANNED)

    assert constants, "no constants collected at all"
    assert all(isinstance(v, float) for defs in constants.values() for _, v in defs)


def test_the_gate_is_wired_into_the_makefile():
    makefile = (ROOT / "Makefile").read_text("utf-8")

    assert "claim-constants:" in makefile
    assert "tools/verify_claim_constants.py" in makefile
