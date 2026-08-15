"""A capability the system advertises must be one it can perform.

Skills are loaded dynamically by name from the catalog, so no import edge
points at them. The reachability sweep behind 053b0a8ab proved its case with an
import sweep — the correct instrument for ordinary modules and a blind one for
this directory — and retired fourteen skill modules whose only callers name them
as strings. Nine of them stayed on the declared surface and in the routing
tables afterwards: `catalog_policy` still assigned them an effect scope, the
capability engine still promoted them, and a hardwired pathway still matched
user text to them. One, `manifest_to_device`, has a recorded live execution
saving a file to the Desktop.

Nothing failed. The audits check that every discovered module reaches the live
registry, which is the direction that had no defect in it; the reverse
direction, that every advertised name has a module behind it, was unchecked.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = ROOT / "core" / "skills"


def _declared_skill_names() -> set[str]:
    """Every ``name = "..."`` on a class under core/skills."""
    names: set[str] = set()
    for path in sorted(SKILLS_DIR.glob("*.py")):
        try:
            tree = ast.parse(path.read_text("utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if not isinstance(item, ast.Assign):
                    continue
                for target in item.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id == "name"
                        and isinstance(item.value, ast.Constant)
                        and isinstance(item.value.value, str)
                    ):
                        names.add(item.value.value)
    return names


# Names that appear in the tables but are dispatched by something other than a
# skill module. Each entry states its real dispatcher, so a name that arrives
# here without one is a new orphan and fails.
_NON_SKILL_DISPATCH = {
    "browser_action": "core/planning/planner.py tool name",
    "delegate_shard": "core/skills/world_forge.py action",
    "spawn_agent": "core/skills/world_forge.py action",
    "spawn_agents_parallel": "core/skills/world_forge.py action",
    "knowledge_base": "belief_ops action",
    "network_discovery": "network_ops action",
    "network_ops": "reach_gateway action",
    "network_recon": "sec_ops action",
    "shell": "sovereign_terminal keyword",
    "system_ops": "os_manipulation action",
}


def test_every_policy_scope_names_a_real_skill():
    from core.skills.catalog_policy import SKILL_EFFECT_SCOPES

    declared = _declared_skill_names()
    orphans = sorted(
        name
        for name in SKILL_EFFECT_SCOPES
        if name not in declared and name not in _NON_SKILL_DISPATCH
    )
    assert not orphans, (
        "catalog_policy assigns an effect scope to skills that do not exist — "
        f"the surface advertises what it cannot run: {orphans}"
    )


def test_every_hardwired_pathway_reaches_a_real_skill():
    """A pathway is the highest-priority route there is: it matches user text
    directly. One pointing at a missing skill is a sentence the runtime cannot
    answer, at the priority reserved for the things it answers first."""
    source = (ROOT / "core" / "orchestrator" / "initializers" / "pathways.py").read_text()
    declared = _declared_skill_names()

    routed = set(re.findall(r'skill_name\s*=\s*"([a-zA-Z0-9_]+)"', source))
    orphans = sorted(
        name for name in routed if name not in declared and name not in _NON_SKILL_DISPATCH
    )
    assert not orphans, f"pathways route user text to missing skills: {orphans}"


def test_capability_promotion_only_promotes_real_skills():
    source = (ROOT / "core" / "capability_engine.py").read_text()
    declared = _declared_skill_names()

    promoted = set(re.findall(r'_promote\(\s*"([a-zA-Z0-9_]+)"', source))
    orphans = sorted(
        name for name in promoted if name not in declared and name not in _NON_SKILL_DISPATCH
    )
    assert not orphans, f"capability engine promotes missing skills: {orphans}"


@pytest.mark.parametrize(
    "name",
    [
        "build_app",
        "cognitive_trainer",
        "evolution_status",
        "grounded_search",
        "improve_own_code",
        "ManageAbilities",
        "manifest_to_device",
        "memory_sync",
        "plan_mode",
    ],
)
def test_the_dynamically_loaded_skills_survived_the_reachability_sweep(name):
    """Named one by one so a future sweep has to argue with each of them."""
    assert name in _declared_skill_names(), (
        f"{name} is advertised by catalog_policy and has no implementation; "
        "an import sweep cannot see a skill, so it cannot retire one"
    )
