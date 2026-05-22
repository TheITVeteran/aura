from __future__ import annotations

import pytest


class _LiveLLM:
    pass


class _MockLLM:
    pass


class _Organ:
    def __init__(self, instance):
        self._instance = instance

    def get_instance(self):
        return self._instance


class _Kernel:
    def __init__(self, organs=None, boot_result=None):
        self.organs = organs or {}
        self.boot_result = boot_result or {"ready": True}
        self.stopped = False

    async def boot(self):
        return self.boot_result

    async def stop(self):
        self.stopped = True


class _FailingKernel(_Kernel):
    async def boot(self):
        message = "boot subsystem unavailable"
        raise RuntimeError(message)


def _noop_registrar(_container):
    return None


def _container():
    return object()


def _config():
    return object()


def _vault():
    return object()


@pytest.mark.asyncio
async def test_final_kernel_diagnostic_accepts_live_llm():
    from tools.diagnostics.final_kernel_diagnostic import run_final_kernel_diagnostic

    kernel = _Kernel(organs={"llm": _Organ(_LiveLLM())})

    result = await run_final_kernel_diagnostic(
        service_container_factory=_container,
        cognitive_service_registrar=_noop_registrar,
        config_factory=_config,
        vault_factory=_vault,
        kernel_factory=lambda _config, _vault: kernel,
    )

    assert result.ok is True
    assert result.status == "kernel_boot_live_llm_verified"
    assert result.details["llm_instance_class"] == "_LiveLLM"
    assert kernel.stopped is True


@pytest.mark.asyncio
async def test_final_kernel_diagnostic_rejects_fallback_llm():
    from tools.diagnostics.final_kernel_diagnostic import run_final_kernel_diagnostic

    kernel = _Kernel(organs={"llm": _Organ(_MockLLM())})

    result = await run_final_kernel_diagnostic(
        service_container_factory=_container,
        cognitive_service_registrar=_noop_registrar,
        config_factory=_config,
        vault_factory=_vault,
        kernel_factory=lambda _config, _vault: kernel,
    )

    assert result.ok is False
    assert result.status == "fallback_llm_resolved"
    assert "MockLLM" in result.error
    assert kernel.stopped is True


@pytest.mark.asyncio
async def test_final_kernel_diagnostic_records_boot_failure(monkeypatch):
    import tools.diagnostics.final_kernel_diagnostic as diagnostic

    records = []
    kernel = _FailingKernel()

    monkeypatch.setattr(
        diagnostic,
        "record_degradation",
        lambda *args, **kwargs: records.append((args, kwargs)),
    )

    result = await diagnostic.run_final_kernel_diagnostic(
        service_container_factory=_container,
        cognitive_service_registrar=_noop_registrar,
        config_factory=_config,
        vault_factory=_vault,
        kernel_factory=lambda _config, _vault: kernel,
    )

    assert result.ok is False
    assert result.status == "diagnostic_failed_closed"
    assert records
    assert "failed closed" in records[0][1]["action"]
    assert kernel.stopped is True
