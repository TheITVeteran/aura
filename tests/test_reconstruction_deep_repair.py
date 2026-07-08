import json

import pytest

from core.brain.llm.code_generator import LLMCodeGenerator, extract_python_code
from core.container import ServiceContainer
from core.runtime.self_healing import SelfHealing
from core.service_names import ServiceNames
from core.self_improvement.interface_contract import LabResult, PromotionVerdict
from core.self_improvement.reimplementation_lab import register_reimplementation_lab
from core.self_modification.code_repair import AutonomousCodeRepair


def test_extract_python_code_prefers_fenced_source():
    text = "Here is the module:\n```python\n\ndef answer():\n    return 42\n```\nDone."
    assert extract_python_code(text) == "def answer():\n    return 42"


@pytest.mark.asyncio
async def test_llm_code_generator_uses_router_and_validates_python():
    class Router:
        def __init__(self):
            self.kwargs = {}

        async def think(self, prompt, **kwargs):
            self.kwargs = kwargs
            assert "Target module: core/example.py" in prompt
            return "```python\ndef run():\n    return 'ok'\n```"

    router = Router()
    generator = LLMCodeGenerator(router=router)

    code = await generator.generate_async(
        "# Module Reimplementation Task",
        {"module_path": "core/example.py", "attempt": 2, "stub_code": "def run(): ..."},
    )

    assert "def run" in code
    assert router.kwargs["origin"] == "reimplementation_lab"
    assert router.kwargs["is_background"] is True
    assert router.kwargs["system_prompt"]


@pytest.mark.asyncio
async def test_llm_code_generator_honors_request_generation_budget():
    class Router:
        def __init__(self):
            self.kwargs = {}

        async def think(self, prompt, **kwargs):
            self.kwargs = kwargs
            return "```python\ndef run():\n    return 'ok'\n```"

    router = Router()
    generator = LLMCodeGenerator(router=router, max_tokens=8192, temperature=0.8)

    await generator.generate_async(
        "# Module Reimplementation Task",
        {
            "module_path": "core/example.py",
            "max_tokens": 256,
            "temperature": 0.05,
            "prefer_tier": "coder",
        },
    )

    assert router.kwargs["max_tokens"] == 256
    assert router.kwargs["temperature"] == 0.05
    assert router.kwargs["prefer_tier"] == "coder"


@pytest.mark.asyncio
async def test_self_healing_request_deep_repair_uses_registered_lab(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_ENABLE_DEEP_REPAIR", "1")

    class Lab:
        async def run_reconstruction(self, module_path, max_attempts=None, metadata=None):
            assert module_path == "core/example.py"
            assert max_attempts == 1
            assert metadata["reason"] == "patch_repair_failed"
            return LabResult(
                success=False,
                module_path=module_path,
                verdict=PromotionVerdict.REJECT,
                attempts=1,
            )

    ServiceContainer.clear()
    ServiceContainer.register_instance("reimplementation_lab", Lab(), required=False)
    healer = SelfHealing()

    record = await healer.request_deep_repair(
        "core/example.py",
        reason="patch_repair_failed",
        metadata={"stage": "validation"},
        max_attempts=1,
    )

    assert record["result"] == "deep_repair_rejected"
    assert record["lab_result"]["module_path"] == "core/example.py"


def test_reimplementation_lab_registers_self_repair_alias(tmp_path, monkeypatch):
    from core.self_improvement import reimplementation_lab as lab_module

    monkeypatch.setattr(lab_module, "_INSTANCE", None)
    ServiceContainer.clear()

    lab = register_reimplementation_lab(project_root=str(tmp_path))

    assert ServiceContainer.get(ServiceNames.REIMPLEMENTATION_LAB) is lab
    assert ServiceContainer.get(ServiceNames.PROGRAM_DNA_RECONSTRUCTION, default=None) is None
    ServiceContainer.clear()


@pytest.mark.asyncio
async def test_self_healing_deep_repair_bootstraps_program_dna_lab(monkeypatch):
    monkeypatch.setenv("AURA_ENABLE_DEEP_REPAIR", "1")

    class Lab:
        async def run_reconstruction(self, module_path, max_attempts=None, metadata=None):
            assert module_path == "core/bootstrap.py"
            assert max_attempts == 2
            assert metadata["trigger"] == "self_healing"
            assert metadata["reason"] == "watchdog_restart_exhausted"
            return LabResult(
                success=True,
                module_path=module_path,
                verdict=PromotionVerdict.PROMOTE,
                attempts=1,
            )

    ServiceContainer.clear()
    monkeypatch.setattr(
        "core.self_improvement.reimplementation_lab.register_reimplementation_lab",
        lambda: Lab(),
    )
    healer = SelfHealing()

    record = await healer.request_deep_repair(
        "core/bootstrap.py",
        reason="watchdog_restart_exhausted",
        metadata={"stage": "restart_exhausted"},
        max_attempts=2,
    )

    assert record["result"] == "deep_repair_succeeded"
    assert record["lab_result"]["success"] is True
    ServiceContainer.clear()


@pytest.mark.asyncio
async def test_code_repair_fallback_calls_self_healing_deep_repair(monkeypatch):
    monkeypatch.setenv("AURA_ENABLE_DEEP_REPAIR", "1")

    class Lab:
        async def run_reconstruction(self, module_path, max_attempts=None, metadata=None):
            assert metadata["trigger"] == "patch_repair_failed"
            assert metadata["stage"] == "fix_generation"
            return LabResult(
                success=True,
                module_path=module_path,
                verdict=PromotionVerdict.PROMOTE,
                attempts=1,
            )

    ServiceContainer.clear()
    ServiceContainer.register_instance("reimplementation_lab", Lab(), required=False)
    repair = object.__new__(AutonomousCodeRepair)

    record = await repair._deep_repair_after_patch_failure(
        "core/example.py",
        10,
        {"summary": "patch failed"},
        stage="fix_generation",
    )

    assert record["result"] == "deep_repair_succeeded"
    assert record["lab_result"]["success"] is True


@pytest.mark.asyncio
async def test_self_healing_deep_repair_honors_explicit_disable(monkeypatch):
    monkeypatch.setenv("AURA_ENABLE_DEEP_REPAIR", "0")

    class Lab:
        def __init__(self):
            self.reconstruction_attempts = 0

        async def run_reconstruction(self, *_args, **_kwargs):
            self.reconstruction_attempts += 1
            raise AssertionError("deep repair must not run after explicit disable")

    ServiceContainer.clear()
    lab = Lab()
    ServiceContainer.register_instance("reimplementation_lab", lab, required=False)
    healer = SelfHealing()

    record = await healer.request_deep_repair(
        "core/example.py",
        reason="patch_repair_failed",
        metadata={"stage": "validation"},
        max_attempts=1,
    )

    assert record["result"] == "deep_repair_disabled"
    assert lab.reconstruction_attempts == 0


def test_self_healing_schedule_deep_repair_respects_foreground_only_runtime(monkeypatch):
    monkeypatch.setenv("AURA_ENABLE_DEEP_REPAIR", "1")
    monkeypatch.setenv("AURA_FOREGROUND_ONLY", "1")
    healer = SelfHealing()

    record = healer.schedule_deep_repair(
        "core/example.py",
        reason="watchdog_restart_exhausted",
    )

    assert record["result"] == "foreground_only_runtime"


@pytest.mark.asyncio
async def test_self_healing_performs_real_stop_start_lifecycle_restart(monkeypatch):
    class Restartable:
        def __init__(self):
            self.stops = 0
            self.starts = 0

        async def stop(self):
            self.stops += 1

        async def start(self):
            self.starts += 1

    ServiceContainer.clear()
    service = Restartable()
    ServiceContainer.register_instance("restartable", service, required=False)
    healer = SelfHealing()
    healer.watch("restartable", expected_interval_s=1.0, container_key="restartable")
    records = []

    async def append(record):
        records.append(record)

    monkeypatch.setattr(healer, "_append_record_async", append)
    await healer._heal(healer._watches["restartable"], 10.0)

    assert service.stops == 1
    assert service.starts == 1
    assert records[-1]["result"] == "restarted"
    ServiceContainer.clear()


@pytest.mark.asyncio
async def test_self_healing_never_claims_restart_without_restart_interface(monkeypatch):
    monkeypatch.setenv("AURA_ENABLE_DEEP_REPAIR", "0")
    ServiceContainer.clear()
    ServiceContainer.register_instance("static_service", object(), required=False)
    healer = SelfHealing()
    healer.watch("static", expected_interval_s=1.0, container_key="static_service")
    records = []

    async def append(record):
        records.append(record)

    monkeypatch.setattr(healer, "_append_record_async", append)
    await healer._heal(healer._watches["static"], 10.0)

    assert records[-1]["result"] == "deep_repair_disabled"
    ServiceContainer.clear()


def test_self_healing_ledger_append_uses_internal_governance(monkeypatch, tmp_path):
    from core.runtime import self_healing as self_healing_module

    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "strict")
    monkeypatch.setattr(self_healing_module, "_LEDGER", tmp_path / "events.jsonl")

    SelfHealing()._append_record({"name": "strict_heal", "result": "ok"})

    payload = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8"))
    assert payload["name"] == "strict_heal"
    assert payload["result"] == "ok"
