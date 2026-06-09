from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.brain.llm import local_llm_setup
from core.brain.llm.local_llm_setup import OllamaManager


class _GatewayDouble:
    """Mirrors SubprocessGateway.run's contract (returncode/stdout/stderr)."""

    def __init__(self, listing="other-model"):
        self.calls = []
        self._listing = listing

    def run(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        if list(argv) == ["ollama", "list"]:
            return SimpleNamespace(returncode=0, stdout=self._listing, stderr="")
        return SimpleNamespace(returncode=0, stdout="ollama version 1.0", stderr="")


def test_ensure_installed_uses_bounded_version_check(monkeypatch):
    gateway = _GatewayDouble()
    monkeypatch.setattr(local_llm_setup.shutil, "which", lambda _name: "/usr/local/bin/ollama")
    monkeypatch.setattr(local_llm_setup, "get_subprocess_gateway", lambda: gateway)

    assert OllamaManager().ensure_installed() is True
    argv, kwargs = gateway.calls[0]
    assert argv == ["ollama", "--version"]
    assert kwargs["timeout"] == local_llm_setup._VERSION_TIMEOUT_S
    assert kwargs["capture_output"] is True
    assert kwargs["offline_tooling"] is True
    assert kwargs["source"] == "maintenance_tooling:local_llm_setup"


def test_ensure_model_uses_bounded_list_and_pull(monkeypatch):
    gateway = _GatewayDouble(listing="other-model")
    monkeypatch.setattr(local_llm_setup, "get_subprocess_gateway", lambda: gateway)

    manager = OllamaManager(model_name="aura-test")
    assert manager.ensure_model() is True

    list_argv, list_kwargs = gateway.calls[0]
    assert list_argv == ["ollama", "list"]
    assert list_kwargs["timeout"] == local_llm_setup._LIST_TIMEOUT_S
    assert "text" not in list_kwargs, "gateway sets text=True itself; passing it crashes"

    pull_argv, pull_kwargs = gateway.calls[1]
    assert pull_argv == ["ollama", "pull", "aura-test"]
    assert pull_kwargs["timeout"] == local_llm_setup._PULL_TIMEOUT_S


@pytest.mark.asyncio
async def test_start_cleans_up_process_when_readiness_never_arrives(monkeypatch):
    class FakeProcess:
        returncode = None

        def __init__(self):
            self.terminated = False
            self.killed = False

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def kill(self):
            self.killed = True
            self.returncode = -9

        async def wait(self):
            return self.returncode

    process = FakeProcess()

    async def _spawn(*_args, **_kwargs):
        return process

    monkeypatch.setattr(local_llm_setup.asyncio, "create_subprocess_exec", _spawn)
    monkeypatch.setattr(local_llm_setup.asyncio, "sleep", lambda *_args, **_kwargs: _noop())

    manager = OllamaManager(model_name="aura-test")
    checks = {"count": 0}

    async def _not_running():
        checks["count"] += 1
        return False

    manager.is_running = _not_running

    assert await manager.start() is False
    assert process.terminated is True
    assert process.killed is False
    assert checks["count"] == local_llm_setup._SERVE_READY_ATTEMPTS + 1


@pytest.mark.asyncio
async def test_start_cleans_up_process_after_spawn_failure(monkeypatch):
    async def _raise(*_args, **_kwargs):
        reason = "spawn failed"
        raise OSError(reason)

    monkeypatch.setattr(local_llm_setup.asyncio, "create_subprocess_exec", _raise)

    manager = OllamaManager(model_name="aura-test")
    manager.is_running = _false

    assert await manager.start() is False


async def _noop():
    return None


async def _false():
    return False
