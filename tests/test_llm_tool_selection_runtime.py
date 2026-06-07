from types import SimpleNamespace

from core.capability_engine import CapabilityEngine


def test_select_tool_definitions_is_bounded_and_relevant():
    engine = CapabilityEngine.__new__(CapabilityEngine)
    engine.SKILL_ALIASES = CapabilityEngine.SKILL_ALIASES
    engine.skills = {
        "web_search": SimpleNamespace(metabolic_cost=1),
        "clock": SimpleNamespace(metabolic_cost=1),
        "memory_ops": SimpleNamespace(metabolic_cost=1),
        "computer_use": SimpleNamespace(metabolic_cost=2),
        "os_manipulation": SimpleNamespace(metabolic_cost=2),
        "self_evolution": SimpleNamespace(metabolic_cost=3),
    }
    engine.detect_intent = lambda message: ["web_search", "clock", "memory_ops", "computer_use"]
    engine.active_skills = set(engine.skills)

    selected = CapabilityEngine.select_tool_definitions(
        engine,
        objective="Find the latest Bitcoin price and timestamp it for memory.",
        max_tools=3,
    )
    selected_names = [item["function"]["name"] for item in selected]

    assert len(selected_names) == 3
    assert "web_search" in selected_names
    assert "clock" in selected_names
    assert "memory_ops" in selected_names
    assert "self_evolution" not in selected_names


def test_select_tool_definitions_does_not_materialize_full_tool_catalog():
    engine = CapabilityEngine.__new__(CapabilityEngine)
    engine.SKILL_ALIASES = CapabilityEngine.SKILL_ALIASES
    engine.skills = {
        f"bulk_tool_{idx}": SimpleNamespace(
            metabolic_cost=1,
            enabled=True,
            is_core_personality=False,
            description=f"Bulk tool {idx}",
            schema_def={"type": "object", "properties": {"value": {"type": "string"}}},
        )
        for idx in range(500)
    }
    engine.skills["web_search"] = SimpleNamespace(
        metabolic_cost=1,
        enabled=True,
        is_core_personality=False,
        description="Search the web",
        schema_def={"type": "object", "properties": {"query": {"type": "string"}}},
    )
    engine.skills["clock"] = SimpleNamespace(
        metabolic_cost=1,
        enabled=True,
        is_core_personality=False,
        description="Read the clock",
        schema_def={"type": "object", "properties": {}},
    )
    engine.skills["memory_ops"] = SimpleNamespace(
        metabolic_cost=1,
        enabled=True,
        is_core_personality=False,
        description="Use memory",
        schema_def={"type": "object", "properties": {}},
    )
    engine.active_skills = set(engine.skills)
    engine.detect_intent = lambda message: ["web_search", "clock", "memory_ops"]

    full_catalog_calls = 0

    def _full_catalog_should_not_run():
        nonlocal full_catalog_calls
        full_catalog_calls += 1
        return []

    engine.get_tool_definitions = _full_catalog_should_not_run

    selected = CapabilityEngine.select_tool_definitions(
        engine,
        objective="Find the latest Bitcoin price and timestamp it for memory.",
        max_tools=3,
    )

    assert full_catalog_calls == 0
    assert [item["function"]["name"] for item in selected] == [
        "web_search",
        "clock",
        "memory_ops",
    ]
