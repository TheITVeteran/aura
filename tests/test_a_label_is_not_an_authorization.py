"""The nucleus manager's identity floor, its routing authority, and its
provider contract.

`_format_prompt` chose a caller's system prompt INSTEAD of the grounding
anchor, and `_apply_anchor` reinjected the anchor only after a token counter
shared across every caller crossed a threshold — so an ordinary request
carrying any system prompt could omit Aura's identity and evidence constraints
entirely, for a duration decided by unrelated traffic.

`_select_model_type` routed on a plain origin string taken from kwargs, so any
caller could claim `health_monitor` or `agency_core` and be routed with
constitutive semantics.

And `generate_stream`, `generate_text` and `generate_json` all accepted a
`model` parameter — the provider interface advertises that control — and all
three dropped it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.brain.llm.nucleus_manager import (
    _CONSTITUTIVE_ORIGINS,
    _NUCLEUS_MODEL_TYPES,
    NucleusManager,
    _with_requested_lane,
)
from core.governance_context import local_internal_governed_scope

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def nucleus():
    manager = NucleusManager.__new__(NucleusManager)
    manager._anchor_text = "ANCHOR: stay evidence-bounded."
    manager._strip_chatml_tokens = staticmethod(lambda t: str(t or ""))
    return manager


# ── the identity floor ────────────────────────────────────────────────────


def test_a_caller_system_prompt_does_not_displace_the_anchor(nucleus):
    formatted = nucleus._format_prompt("hi", system_prompt="You are a pirate.")

    assert "ANCHOR: stay evidence-bounded." in formatted
    assert "You are a pirate." in formatted


def test_the_anchor_comes_first(nucleus):
    formatted = nucleus._format_prompt("hi", system_prompt="You are a pirate.")

    assert formatted.index("ANCHOR:") < formatted.index("You are a pirate.")


def test_no_system_prompt_still_gets_the_anchor(nucleus):
    assert "ANCHOR:" in nucleus._format_prompt("hi")


def test_the_anchor_is_not_an_else_branch():
    """It was `system_prompt if system_prompt else self._anchor_text`."""
    source = (ROOT / "core" / "brain" / "llm" / "nucleus_manager.py").read_text("utf-8")

    assert "s_msg = system_prompt if system_prompt else self._anchor_text" not in source


# ── routing authority ─────────────────────────────────────────────────────


def test_a_constitutive_origin_from_an_ordinary_caller_is_not_honoured(nucleus):
    origin = sorted(_CONSTITUTIVE_ORIGINS)[0]

    assert nucleus._select_model_type(origin) == "cortex"


def test_the_same_origin_inside_a_governed_scope_routes_constitutively(nucleus):
    origin = sorted(_CONSTITUTIVE_ORIGINS)[0]

    with local_internal_governed_scope(
        "llm.nucleus.constitutive", receipt_prefix="nucleus-test"
    ):
        assert nucleus._select_model_type(origin) == "brainstem"


def test_an_ordinary_origin_routes_to_cortex_either_way(nucleus):
    assert nucleus._select_model_type("web_ui") == "cortex"
    with local_internal_governed_scope(
        "llm.nucleus.constitutive", receipt_prefix="nucleus-test"
    ):
        assert nucleus._select_model_type("web_ui") == "cortex"


# ── the requested model ───────────────────────────────────────────────────


@pytest.mark.parametrize("lane", sorted(_NUCLEUS_MODEL_TYPES))
def test_a_known_lane_is_carried_into_selection(lane):
    assert _with_requested_lane({}, lane)["requested_model_type"] == lane


def test_no_request_leaves_selection_alone():
    assert _with_requested_lane({"origin": "x"}, None) == {"origin": "x"}
    assert _with_requested_lane({}, "  ") == {}


def test_an_unknown_model_is_refused_not_ignored():
    """"I could not give you that" and "I gave you that" must not look the
    same."""
    carried = _with_requested_lane({}, "gpt-4o")

    assert "requested_model_type" not in carried
    assert carried["requested_model_unavailable"] == "gpt-4o"


def test_the_provider_methods_pass_the_model_through():
    source = (ROOT / "core" / "brain" / "llm" / "nucleus_manager.py").read_text("utf-8")

    assert source.count("_with_requested_lane(") >= 3
    assert "self.generate_text(f\"{prompt}{schema_hint}\", system_prompt, model=model)" in source


# ── lane substitution ─────────────────────────────────────────────────────


def test_a_fresh_manager_reports_no_substitution():
    manager = NucleusManager.__new__(NucleusManager)
    manager._last_served_lane = ""
    manager._last_lane_substituted = False

    assert manager.last_lane_receipt() == {"served_by": "", "substituted": False}


def test_a_substitution_is_readable():
    manager = NucleusManager.__new__(NucleusManager)
    manager._last_served_lane = "brainstem"
    manager._last_lane_substituted = True

    receipt = manager.last_lane_receipt()
    assert receipt["served_by"] == "brainstem"
    assert receipt["substituted"] is True


# ── the sync wrapper ──────────────────────────────────────────────────────


def test_a_failure_after_generation_does_not_run_the_request_again():
    """The outer handler covered run_until_complete as well as loop discovery,
    so a failure after generation began fell into asyncio.run and executed the
    whole request a second time."""
    calls: list[str] = []

    class _Manager(NucleusManager):
        def __init__(self):
            pass

        async def generate_text_async(self, prompt, system_prompt=None, **kwargs):
            calls.append(prompt)
            raise RuntimeError("failed after generation began")

    with pytest.raises(RuntimeError):
        _Manager().generate_text("hello")

    assert calls == ["hello"], f"the request ran {len(calls)} times"
