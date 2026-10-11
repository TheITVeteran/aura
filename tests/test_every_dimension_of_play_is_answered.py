"""Every value of every dimension of a place to play is answered by a way she has or a function that exists, and the
values with no answer are the ones the design map says have none (core/agency/dimensions_of_play.py,
docs/design-docs/THE_DIMENSIONS_OF_A_PLACE_TO_PLAY.md)."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from core.agency.dimensions_of_play import DIMENSIONS
from core.agency.ways_of_playing import WAYS

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
EVERY_VALUE = [value for values in DIMENSIONS.values() for value in values]


@pytest.mark.parametrize("value", EVERY_VALUE, ids=[f"{v.dimension}: {v.name}" for v in EVERY_VALUE])
def test_each_answer_is_a_way_she_has_or_a_function_that_exists(value):
    ways = {way.name for way in WAYS}
    for answer in value.done_by:
        if ":" not in answer:
            assert answer in ways, f"{value.dimension}: {value.name} names no way she has: {answer}"
            continue
        module_name, name = answer.split(":")
        assert hasattr(importlib.import_module(module_name), name), answer


def test_the_values_with_no_answer_are_the_ones_the_map_lists():
    table = (ROOT / "docs/design-docs/THE_DIMENSIONS_OF_A_PLACE_TO_PLAY.md").read_text()
    section = table.split("## The values with no answer yet")[1]
    listed = {line.split("|")[2].strip() for line in section.splitlines() if line.startswith("| ") and line[2].isdigit()}
    unanswered = {f"{value.dimension}: {value.name}" for value in EVERY_VALUE if not value.done_by}
    assert listed == unanswered, (listed - unanswered, unanswered - listed)


def test_each_value_names_what_it_asks_and_how_it_is_known():
    for value in EVERY_VALUE:
        assert value.asks, value
        assert value.dimension in DIMENSIONS
