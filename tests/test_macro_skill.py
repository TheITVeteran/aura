"""A learned macro has to be callable, not just describable."""

from __future__ import annotations

import pytest

from core.agency.macro_skill import (
    MACRO_PREFIX,
    MacroSkill,
    derive_effect_scope,
    macro_tool_name,
)

SCOPES = {
    "clock": "status",
    "read_file": "read_only",
    "compute": "pure_compute",
    "write_file": "read_write_artifacts",
    "browser_action": "external_io",
    "auto_refactor": "privileged_mutation",
}


def _scope_for(tool: str) -> str:
    return SCOPES.get(tool, "")


# ── the blast radius is derived from the steps ───────────────────────────


def test_a_read_only_macro_keeps_a_read_only_scope():
    assert derive_effect_scope(["clock", "read_file"], _scope_for) == "read_only"


def test_the_widest_step_decides():
    assert derive_effect_scope(["read_file", "write_file"], _scope_for) == "read_write_artifacts"
    assert derive_effect_scope(["read_file", "browser_action"], _scope_for) == "external_io"


def test_step_order_does_not_change_the_scope():
    a = derive_effect_scope(["browser_action", "read_file"], _scope_for)
    b = derive_effect_scope(["read_file", "browser_action"], _scope_for)
    assert a == b == "external_io"


def test_an_unresolvable_step_is_treated_as_the_widest():
    """A tool nobody can classify is not thereby harmless."""
    assert derive_effect_scope(["read_file", "mystery"], _scope_for) == "privileged_mutation"


def test_a_resolver_that_raises_is_treated_as_unknown():
    def broken(_tool):
        raise RuntimeError("catalog unavailable")

    assert derive_effect_scope(["read_file"], broken) == "privileged_mutation"


def test_an_empty_macro_claims_nothing():
    assert derive_effect_scope([], _scope_for) == "status"


def test_a_privileged_step_dominates_everything():
    scope = derive_effect_scope(["clock", "auto_refactor", "read_file"], _scope_for)
    assert scope == "privileged_mutation"


# ── naming ───────────────────────────────────────────────────────────────


def test_a_macro_cannot_take_a_shipped_skill_name():
    assert macro_tool_name("clock").startswith(MACRO_PREFIX)
    assert macro_tool_name("clock") != "clock"


def test_a_name_is_made_a_legal_identifier():
    import re

    for raw in ("tidy downloads", "deploy/site", "3d-render", "a" * 200):
        assert re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", macro_tool_name(raw))


def test_an_unusable_name_is_refused():
    with pytest.raises(ValueError, match="no usable name"):
        macro_tool_name("   ")


# ── the skill itself ─────────────────────────────────────────────────────


def _skill(**kwargs) -> MacroSkill:
    defaults = {
        "macro_name": "tidy_downloads",
        "description": "Sort downloads into dated folders",
        "parameters": ["folder"],
        "effect_scope": "read_write_artifacts",
    }
    return MacroSkill(**{**defaults, **kwargs})


def test_the_skill_carries_the_catalog_metadata_registration_requires():
    skill = _skill()
    assert skill.name == "macro_tidy_downloads"
    assert skill.effect_scope == "read_write_artifacts"
    assert skill.description
    assert skill.macro_name == "tidy_downloads"


def test_an_invalid_scope_is_refused_at_construction():
    with pytest.raises(ValueError, match="unknown effect scope"):
        _skill(effect_scope="whatever")


@pytest.mark.asyncio
async def test_a_missing_argument_names_the_argument():
    """Not "the library is unavailable", which sends the caller to the wrong place."""
    result = await _skill().execute({}, {})
    assert not result["ok"]
    assert "folder" in result["error"]


@pytest.mark.asyncio
async def test_a_missing_library_is_reported_rather_than_raised():
    result = await _skill(parameters=[]).execute({}, {})
    assert not result["ok"]
    assert "library" in result["error"]


@pytest.mark.asyncio
async def test_a_macro_executes_through_the_library(monkeypatch):
    calls = []

    class Library:
        async def execute_skill(self, name, kwargs):
            calls.append((name, kwargs))
            return ["step one", "step two"]

    from core.container import ServiceContainer

    ServiceContainer.register_instance("skill_library", Library())
    try:
        result = await _skill().execute({"folder": "~/Downloads"}, {})
    finally:
        ServiceContainer.register_instance("skill_library", None)

    assert result["ok"] and result["steps"] == 2
    assert calls == [("tidy_downloads", {"folder": "~/Downloads"})]


@pytest.mark.asyncio
async def test_a_failing_macro_returns_a_result_rather_than_raising(monkeypatch):
    """A tool caller branches on ok; an exception mid-turn is not that."""

    class Library:
        async def execute_skill(self, name, kwargs):
            raise RuntimeError("step 2 failed")

    from core.container import ServiceContainer

    ServiceContainer.register_instance("skill_library", Library())
    try:
        result = await _skill().execute({"folder": "x"}, {})
    finally:
        ServiceContainer.register_instance("skill_library", None)

    assert not result["ok"]
    assert "step 2 failed" in result["error"]
