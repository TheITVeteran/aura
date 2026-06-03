from __future__ import annotations

from types import SimpleNamespace

import pytest

from integration.aura_master_integration import (
    IntegrationManager,
    IntegrationStatus,
    IntegrationStep,
)


def test_integration_step_executes_real_module_function(tmp_path) -> None:
    module_path = tmp_path / "real_integration_step.py"
    module_path.write_text(
        "def apply(orchestrator):\n"
        "    return {'source': orchestrator.source}\n",
        encoding="utf-8",
    )

    manager = IntegrationManager()
    result = manager.execute_step(
        IntegrationStep(
            name="real_step",
            module_path=str(module_path),
            function_name="apply",
            timeout_seconds=1,
        ),
        SimpleNamespace(source="unit"),
    )

    assert result.status is IntegrationStatus.COMPLETED
    assert result.metadata["result"] == {"source": "unit"}


def test_integration_step_metadata_is_json_safe(tmp_path) -> None:
    module_path = tmp_path / "nonserial_integration_step.py"
    module_path.write_text(
        "def apply(orchestrator):\n"
        "    return object()\n",
        encoding="utf-8",
    )

    manager = IntegrationManager()
    result = manager.execute_step(
        IntegrationStep(
            name="nonserial_step",
            module_path=str(module_path),
            function_name="apply",
            timeout_seconds=1,
        ),
        SimpleNamespace(source="unit"),
    )

    assert result.status is IntegrationStatus.COMPLETED
    assert isinstance(result.metadata["result"], str)


def test_integration_step_missing_module_fails_explicitly() -> None:
    manager = IntegrationManager()

    result = manager.execute_step(
        IntegrationStep(
            name="missing_step",
            module_path="missing.integration.module",
            function_name="apply",
            timeout_seconds=1,
        ),
        SimpleNamespace(source="unit"),
    )

    assert result.status is IntegrationStatus.FAILED
    assert "missing" in (result.error or "").lower()


def test_integration_step_preserves_invariant_failure(tmp_path) -> None:
    module_path = tmp_path / "broken_integration_step.py"
    module_path.write_text(
        "def apply(orchestrator):\n"
        "    assert False, 'integration invariant broken'\n",
        encoding="utf-8",
    )

    manager = IntegrationManager()

    with pytest.raises(AssertionError, match="integration invariant broken"):
        manager.execute_step(
            IntegrationStep(
                name="broken_step",
                module_path=str(module_path),
                function_name="apply",
                timeout_seconds=1,
            ),
            SimpleNamespace(source="unit"),
        )
