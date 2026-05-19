from core.container import ServiceContainer
from core.orchestrator import RobustOrchestrator


def test_setup_registers_output_gate_for_runtime_health_contract():
    ServiceContainer.clear()
    try:
        orchestrator = RobustOrchestrator()
        orchestrator.setup()

        assert ServiceContainer.get("output_gate", default=None) is orchestrator.output_gate
    finally:
        ServiceContainer.clear()
