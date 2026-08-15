"""Numbers the prompt states about Aura's own condition.

Affect, felt thought, meta-qualia, divergence and relational trust were
formatted straight into f-strings. A None or a string raised TypeError inside
the format and the enclosing except swallowed it, so a whole block left the
prompt with nobody saying which. A NaN printed as "nan", so "valence=nan"
became a sentence about how she feels. An out-of-range value printed as-is, so
"Valence: +7.00" claimed a state that does not exist on a [-1, 1] scale.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.brain.llm.context_assembler import ContextAssembler

ROOT = Path(__file__).resolve().parents[1]
_n = ContextAssembler._self_state_number


@pytest.mark.parametrize("value", [None, "", "warm", object(), [], {}])
def test_a_non_number_is_unmeasured_not_an_exception(value):
    assert _n(value, low=0.0, high=1.0) == "unmeasured"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_reading_is_unmeasured(value):
    assert _n(value, low=0.0, high=1.0) == "unmeasured"


def test_an_ordinary_reading_renders_normally():
    assert _n(0.37, low=0.0, high=1.0) == "0.37"
    assert _n(-0.42, low=-1.0, high=1.0, signed=True) == "-0.42"
    assert _n(0.42, low=-1.0, high=1.0, signed=True) == "+0.42"


def test_an_impossible_reading_is_clamped_into_its_range():
    assert _n(7.0, low=-1.0, high=1.0, signed=True) == "+1.00"
    assert _n(-9.0, low=0.0, high=1.0) == "0.00"


def test_clamping_is_recorded():
    from core.runtime.errors import recent_degradations

    before = len(
        recent_degradations(limit=500, subsystem_prefixes=("context_assembler.self_state_range",))
    )
    _n(42.0, low=0.0, high=1.0)
    after = recent_degradations(
        limit=500, subsystem_prefixes=("context_assembler.self_state_range",)
    )

    assert len(after) > before


def test_an_integer_reading_is_accepted():
    assert _n(1, low=0.0, high=1.0) == "1.00"
    assert _n(True, low=0.0, high=1.0) == "1.00"


def test_no_self_state_number_is_formatted_raw_any_more():
    """The renders that state her own condition all go through the contract."""
    source = (ROOT / "core" / "brain" / "llm" / "context_assembler.py").read_text("utf-8")

    raw = re.findall(r"\{(?:affect|felt|mq|div_val|trust)[^{}]*:[+]?\.\d+f\}", source)
    assert not raw, f"raw self-state formats remain: {raw}"
