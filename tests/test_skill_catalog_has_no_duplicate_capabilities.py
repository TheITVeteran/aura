"""Three names, one skill, and a chooser asked to pick between them.

``EnhancedWebSearchSkill`` registers as ``web_search``. ``WebSearchSkill``
subclasses it and registers as ``search_web``. ``FreeSearchSkill`` subclasses
it and registers as ``free_search``, its own docstring calling it a
"Compatibility wrapper for legacy 'free_search' skill". The implementations
were already deduplicated by inheritance — and the CATALOG was not, so all
three appeared as separate options.

Measured live, 2026-08-10: asked "what's the weather where I am? and if you
can't actually get it, tell me that instead of guessing", she answered "I
don't have a window, camera, thermometer or weather feed" while five search
skills were READY and available. Three of those five were one skill wearing
three names.

The names stay registered and callable. Stored plans, beliefs and learned
policies reference skills by name, and breaking those to tidy a list would
trade a cosmetic problem for a real one. They are hidden from the catalog, not
removed from the registry.
"""
from __future__ import annotations

import pytest

from core.capability_engine import _SKILL_ALIASES


class _Meta:
    def __init__(self, enabled=True):
        self.enabled = enabled


class _Engine:
    """Minimal stand-in exercising the catalog's alias suppression."""

    def __init__(self, names):
        from core.capability_engine import CapabilityEngine

        self.skills = {name: _Meta() for name in names}
        self.active_skills = set(names)
        self.quarantined_skills = {}
        self._suppressed_aliases = CapabilityEngine._suppressed_aliases.__get__(self)
        self.iter_tool_catalog = CapabilityEngine.iter_tool_catalog.__get__(self)

    def _catalog_item_for_skill(self, name, meta):
        return {"name": name, "available": True, "active": True}

    def _catalog_item_for_quarantine(self, item):  # pragma: no cover
        return item


def _names(engine):
    return sorted(item["name"] for item in engine.iter_tool_catalog())


def test_one_capability_appears_once():
    engine = _Engine(["web_search", "search_web", "free_search"])

    assert _names(engine) == ["web_search"]


def test_the_alias_is_still_registered_and_callable():
    """Hidden from the menu is not removed from the machine."""
    engine = _Engine(["web_search", "search_web", "free_search"])

    engine.iter_tool_catalog()

    assert "search_web" in engine.skills
    assert "free_search" in engine.skills


def test_a_missing_canonical_falls_back_to_showing_the_alias():
    """Losing the capability entirely is worse than showing a duplicate."""
    engine = _Engine(["search_web", "free_search"])

    listed = _names(engine)

    assert "search_web" in listed
    assert "free_search" in listed


@pytest.mark.parametrize(
    "distinct",
    [
        "grounded_search",
        "local_reference_search",
        "sovereign_browser",
        "web_interlocutor",
    ],
)
def test_genuinely_different_search_skills_are_not_collapsed(distinct):
    """Similar names are not the same capability.

    grounded_search carries Google citation grounding, local_reference_search
    answers from an offline corpus when the network is gone, sovereign_browser
    interacts with pages, and web_interlocutor talks to another agent. Folding
    these into web_search would delete real capability.
    """
    engine = _Engine(["web_search", "search_web", "free_search", distinct])

    assert distinct in _names(engine)


def test_aliases_only_ever_point_at_a_real_canonical():
    """An alias whose target does not exist would hide a working skill."""
    for alias, canonical in _SKILL_ALIASES.items():
        assert alias != canonical, alias


def test_the_alias_map_matches_the_class_hierarchy():
    """The claim "these are the same skill" must be true of the code.

    If someone later gives search_web its own behaviour, this fails rather
    than silently hiding a skill that has become distinct.
    """
    from core.skills.free_search import FreeSearchSkill
    from core.skills.web_search import EnhancedWebSearchSkill
    from core.skills.web_search_skill import WebSearchSkill

    import inspect

    for cls in (WebSearchSkill, FreeSearchSkill):
        assert issubclass(cls, EnhancedWebSearchSkill)
        # Same behaviour, not merely a shared ancestor. An alias is allowed to
        # rename itself and to normalise its inputs — search_web wraps params
        # in a pydantic model — but it must DELEGATE the actual search. If it
        # ever grows its own implementation it is a distinct capability and
        # hiding it from the catalog would be deleting a skill.
        own_execute = vars(cls).get("execute")
        if own_execute is None:
            continue
        body = inspect.getsource(own_execute)
        assert "super().execute" in body, (
            f"{cls.__name__}.execute does not delegate; it is no longer an alias "
            f"and must be removed from _SKILL_ALIASES"
        )
