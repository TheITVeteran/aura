import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException


class RequestProbe:
    def __init__(self, path="/api/chat", host="203.0.113.10", headers=None):
        self.url = SimpleNamespace(path=path)
        self.client = SimpleNamespace(host=host)
        self.headers = headers or {}


class ContainerProbe:
    def __init__(self):
        self.removed = False
        self.killed = False

    def wait(self, timeout):
        self.wait_timeout = timeout
        return {"StatusCode": 0}

    def logs(self):
        return b"ok"

    def kill(self):
        self.killed = True

    def remove(self, force=False):
        self.removed = True
        self.remove_force = force


class ContainerRunnerProbe:
    def __init__(self):
        self.calls = []
        self.last_container = None

    def run(self, **kwargs):
        self.calls.append(kwargs)
        self.last_container = ContainerProbe()
        return self.last_container


class DockerClientProbe:
    def __init__(self):
        self.containers = ContainerRunnerProbe()


class DockerModuleProbe:
    def __init__(self):
        self.client = DockerClientProbe()
        self.from_env_calls = 0

    def from_env(self):
        self.from_env_calls += 1
        return self.client


def test_config_firewall():
    """Verify that AuraConfig mirrors the explicit owner-autonomous posture."""
    from core.config import config

    assert os.environ.get("AURA_SECURITY_PROFILE") == "owner_autonomous"
    assert os.environ.get("AURA_INTERNAL_ONLY") == "0"
    assert os.environ.get("AURA_ALLOW_NETWORK_ACCESS") == "1"
    assert config.security.security_profile == "owner_autonomous"
    assert config.security.internal_only_mode is False
    assert config.security.allow_network_access is True


def test_server_auth_logic(monkeypatch):
    """Verify extracted interface auth fails closed and accepts valid bearer tokens."""
    from interface import auth

    monkeypatch.setattr(auth.config.security, "internal_only_mode", False, raising=False)
    monkeypatch.setattr(auth.config, "api_token", None, raising=False)
    with pytest.raises(HTTPException) as exc:
        auth.validate_runtime_security_request(RequestProbe())
    assert exc.value.status_code == 503

    monkeypatch.setattr(auth.config, "api_token", "secret_key", raising=False)
    auth.validate_runtime_security_request(
        RequestProbe(headers={"Authorization": "Bearer secret_key"})
    )
    with pytest.raises(HTTPException) as exc:
        auth.validate_runtime_security_request(
            RequestProbe(headers={"Authorization": "Bearer wrong"})
        )
    assert exc.value.status_code == 401


def test_local_llm_logic():
    """Verify the local brain constructs the configured model handle."""
    from core.brain.local_llm import LocalBrain

    brain = LocalBrain(model_name="test-model")
    assert brain.model == "test-model"


def test_sandbox_isolation_config(monkeypatch):
    """Verify that SecureDockerSandbox forces network and resource limits."""
    docker_probe = DockerModuleProbe()
    monkeypatch.setitem(sys.modules, "docker", docker_probe)
    sys.modules.pop("core.skills.secure_sandbox", None)
    secure_sandbox = importlib.import_module("core.skills.secure_sandbox")
    monkeypatch.setattr(secure_sandbox, "docker", docker_probe)

    sandbox = secure_sandbox.SecureDockerSandbox()
    result = sandbox.execute_code("print('hello')", "/tmp")

    assert result == {"ok": True, "exit_code": 0, "output": "ok"}
    assert docker_probe.from_env_calls == 1

    call = docker_probe.client.containers.calls[-1]
    assert call["network_disabled"] is True
    assert call["mem_limit"] == "1g"
    assert call["nano_cpus"] == 2_000_000_000
    assert call["detach"] is True
    assert call["remove"] is False
    assert docker_probe.client.containers.last_container.removed is True
    assert docker_probe.client.containers.last_container.remove_force is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__))]))
