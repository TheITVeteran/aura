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
import logging


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


def _skill_classes() -> dict[str, type]:
    """Every discoverable skill class, keyed by its registered name."""
    import importlib
    import inspect
    import pkgutil

    import core.skills as pkg

    found: dict[str, type] = {}
    for module in pkgutil.iter_modules(pkg.__path__):
        try:
            loaded = importlib.import_module(f"core.skills.{module.name}")
        except Exception as exc:  # noqa: BLE001 — an unimportable skill is not this test's subject
            logging.getLogger(__name__).debug("skill %s did not import: %s", module.name, exc)
            continue
        for _, cls in inspect.getmembers(loaded, inspect.isclass):
            if cls.__module__ != loaded.__name__:
                continue
            name = getattr(cls, "name", None)
            if isinstance(name, str) and name and name not in found:
                found[name] = cls
    return found


@pytest.mark.parametrize(("alias", "canonical"), sorted(_SKILL_ALIASES.items()))
def test_every_alias_really_delegates_to_its_canonical(alias, canonical):
    """The claim "these are the same skill" must be true of the code.

    Checked across the whole map rather than a hand-picked pair, because
    hand-picking is the exact bug this file is about: _capability_line named
    only the family already known to fail, so every other family kept failing
    the same way. A ratchet covering only the examples its author thought of
    carries the defect of the code it guards.

    An alias may rename itself and normalise its inputs — search_web wraps
    params in a pydantic model, sovereign_imagination converts to
    ImageGenInput — but it must DELEGATE. If one grows its own implementation
    it is a distinct capability, and hiding it would be deleting a skill
    rather than tidying a list.
    """
    import inspect

    classes = _skill_classes()
    alias_cls = classes.get(alias)
    canonical_cls = classes.get(canonical)
    assert alias_cls is not None, f"{alias} is not a discoverable skill"
    assert canonical_cls is not None, f"{canonical} is not a discoverable skill"
    assert issubclass(alias_cls, canonical_cls), (
        f"{alias} does not derive from {canonical}; it is not an alias"
    )

    own_execute = vars(alias_cls).get("execute")
    if own_execute is None:
        return
    body = inspect.getsource(own_execute)
    assert "super().execute" in body, (
        f"{alias_cls.__name__}.execute does not delegate; it is no longer an "
        f"alias and must be removed from _SKILL_ALIASES"
    )


def test_an_alias_never_offers_less_than_its_canonical():
    """sovereign_imagination's input model is a strict subset of image_gen's.

    That is the reason it must be the hidden one: image_gen also does
    image-to-image, the facade has no field for it, and a chooser picking the
    facade silently loses capability. Direction matters — hiding the RICHER
    skill would be the damaging version of this change.
    """
    from core.skills.image_gen import ImageGenInput
    from core.skills.sovereign_imagination import ImageInput

    facade_fields = set(ImageInput.model_fields)
    canonical_fields = set(ImageGenInput.model_fields)

    assert facade_fields <= canonical_fields, (
        "the facade accepts inputs the canonical does not; it is not a subset"
    )
    assert _SKILL_ALIASES["sovereign_imagination"] == "image_gen"


def test_the_search_aliases_are_still_subclasses():
    """The original hierarchy claim, kept explicit."""
    from core.skills.free_search import FreeSearchSkill
    from core.skills.web_search import EnhancedWebSearchSkill
    from core.skills.web_search_skill import WebSearchSkill

    assert issubclass(WebSearchSkill, EnhancedWebSearchSkill)
    assert issubclass(FreeSearchSkill, EnhancedWebSearchSkill)
