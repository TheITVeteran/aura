"""Reloading a base class disabled every skill, and reported success.

MEASURED live 2026-08-10. Every desktop task came back:

    CRITICAL SERVICE FAILURE: Subsystem 'capability_engine' failed with
    failure policy 'fail-closed'. Original error: RuntimeError: TypeError:
    implementation does not satisfy canonical BaseSkill

All 84 skill classes conform statically — checked, all pass. What failed was
`issubclass(skill_class, BaseSkill)` against a BaseSkill that was no longer the
same object. `importlib.reload` builds NEW class objects and rebinds only the
reloaded module's namespace, so every class that had already subclassed the old
base keeps pointing at it and identity checks against the new one fail until
restart.

The `skills` scope reloads `core.skills.*`, which contains
`core.skills.base_skill`, so pressing the update button disabled the whole
capability engine — and the reloader reported ok=true, failed=[].

The same shape produced `PicklingError: Can't pickle <class HierarchicalPhi>:
it's not the same object as core.consciousness.hierarchical_phi.HierarchicalPhi`
from the consciousness scope in the same session.
"""

from __future__ import annotations

from core.ops.hot_reload import HotReloader


def test_a_base_class_with_live_subclasses_is_refused():
    import core.skills.base_skill  # noqa: F401
    import core.skills.desktop_task  # noqa: F401

    reason = HotReloader()._inheritance_anchor_reason("core.skills.base_skill")
    assert reason, "reloading BaseSkill orphans every skill that subclasses it"
    assert "BaseSkill" in reason
    assert "subclassed by" in reason


def test_a_leaf_module_is_still_reloadable():
    """The guard must not turn into a blanket refusal."""
    import core.conversation.screen_reading_claim  # noqa: F401

    assert HotReloader()._inheritance_anchor_reason(
        "core.conversation.screen_reading_claim"
    ) == ""


def test_reload_file_refuses_and_says_so():
    import core.skills.base_skill  # noqa: F401
    import core.skills.desktop_task  # noqa: F401

    result = HotReloader().reload_file("core/skills/base_skill.py")
    payload = result.to_dict()

    assert "core.skills.base_skill" not in payload["reloaded"]
    assert "core.skills.base_skill" in payload["skipped"]
    assert payload["orphan_risks"], "a refusal the caller cannot see is a silent failure"
    assert payload["orphan_risks"][0]["module"] == "core.skills.base_skill"


def test_the_skills_scope_no_longer_orphans_the_catalog():
    import core.skills.base_skill  # noqa: F401
    import core.skills.desktop_task  # noqa: F401

    result = HotReloader().reload_scope("skills")
    payload = result.to_dict()
    assert "core.skills.base_skill" not in payload["reloaded"]
    assert any(
        risk["module"] == "core.skills.base_skill" for risk in payload["orphan_risks"]
    )


def test_the_catalog_still_conforms_after_a_scope_reload():
    """The point of the guard: identity survives the reload."""
    import inspect

    import core.skills.desktop_task
    from core.skills.base_skill import BaseSkill

    HotReloader().reload_scope("skills")

    from core.skills.base_skill import BaseSkill as ReloadedBase

    assert BaseSkill is ReloadedBase, "the base class object must not be replaced"
    skill = core.skills.desktop_task.DesktopTaskSkill
    assert inspect.isclass(skill) and issubclass(skill, ReloadedBase), (
        "this is the exact check the capability engine fail-closed on"
    )
