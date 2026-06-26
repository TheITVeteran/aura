"""Deep mind support services are first-class canonical services.

The modules can exist and pass isolated unit tests while still being invisible to the
live runtime if they are only reachable through ad hoc singleton imports. This locks
the operational surfaces into the ServiceContainer provider map.
"""
from __future__ import annotations


def test_deep_mind_services_resolve_from_canonical_providers(tmp_path):
    from core.container import ServiceContainer
    from core.providers.consciousness_provider import register_consciousness_services
    from core.providers.sensory_provider import register_sensory_services

    import core.cognition.outcome_ledger as outcome_ledger
    import core.cognition.scientific_engine as scientific_engine

    ServiceContainer.clear()
    try:
        outcome_ledger._ledger = outcome_ledger.OutcomeLedger(
            db_path=str(tmp_path / "outcome_ledger.db")
        )
        scientific_engine._engine = scientific_engine.ScientificEngine(
            db_path=str(tmp_path / "scientific_engine.db")
        )

        register_consciousness_services(ServiceContainer)
        register_sensory_services(ServiceContainer)

        expected = {
            "global_workspace",
            "nociception",
            "affect_grounding",
            "drive_integration",
            "outcome_ledger",
            "scientific_engine",
            "unified_world_model",
            "screen_perception",
            "perceptual_pump",
            "general_terminal_parser",
            "terminal_parser",
        }
        missing = {name for name in expected if not ServiceContainer.has(name)}
        assert missing == set()

        assert ServiceContainer.get("global_workspace").__class__.__name__ == "GlobalWorkspace"
        assert ServiceContainer.get("nociception").__class__.__name__ == "NociceptionEngine"
        assert ServiceContainer.get("affect_grounding").__class__.__name__ == "AffectGroundingEngine"
        assert ServiceContainer.get("drive_integration").__class__.__name__ == "DriveIntegrationEngine"
        assert ServiceContainer.get("outcome_ledger") is outcome_ledger.get_outcome_ledger()
        assert ServiceContainer.get("scientific_engine") is scientific_engine.get_scientific_engine()
        assert ServiceContainer.get("unified_world_model").__class__.__name__ == "UnifiedWorldModel"
        assert ServiceContainer.get("screen_perception").__class__.__name__ == "ScreenPerception"
        assert ServiceContainer.get("perceptual_pump").__class__.__name__ == "PerceptualPump"
        assert ServiceContainer.get("terminal_parser").__class__.__name__ == "GeneralTerminalParser"
    finally:
        ServiceContainer.clear()
        outcome_ledger._ledger = None
        scientific_engine._engine = None
