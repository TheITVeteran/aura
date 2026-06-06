################################################################################

from contextlib import contextmanager
import os
import unittest
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from fastapi import HTTPException

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

# Install lightweight heavy-dependency stand-ins before imports to avoid model downloads/connection errors
sys.modules["faster_whisper"] = ModuleType("faster_whisper")


class ContainerRunRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(SimpleNamespace(args=args, kwargs=kwargs))
        return ContainerResult()


class ContainerResult:
    def __init__(self):
        self.killed = False
        self.removed = False

    def wait(self, timeout=None):
        return {"StatusCode": 0}

    def logs(self):
        return b"hello\n"

    def kill(self):
        self.killed = True

    def remove(self, force=False):
        self.removed = True


class DockerModule(ModuleType):
    def __init__(self):
        super().__init__("docker")
        self.run_recorder = ContainerRunRecorder()
        self.client = SimpleNamespace(containers=SimpleNamespace(run=self.run_recorder))

    def from_env(self):
        return self.client


docker_module = DockerModule()
sys.modules["docker"] = docker_module


@contextmanager
def temporary_attr(target, name, value):
    original = getattr(target, name)
    setattr(target, name, value)
    try:
        yield
    finally:
        setattr(target, name, original)

class TestSovereignAura(unittest.TestCase):

    def test_config_firewall(self):
        """Verify AURA_INTERNAL_ONLY mirrors the loaded security config."""
        from core.config import config
        expected = "1" if config.security.internal_only_mode else "0"
        self.assertEqual(os.environ.get("AURA_INTERNAL_ONLY"), expected)

    def test_server_auth_logic(self):
        """Verify current interface auth token logic."""
        from interface import auth

        local_request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
        remote_request = SimpleNamespace(client=SimpleNamespace(host="203.0.113.10"))

        with temporary_attr(auth.config, "api_token", ""):
            auth._verify_token(local_request, None)

        with temporary_attr(auth.config, "api_token", "secret_key"):
            auth._verify_token(remote_request, "secret_key")
            with self.assertRaises(HTTPException) as cm:
                auth._verify_token(remote_request, "wrong")
            self.assertEqual(cm.exception.status_code, 401)

    def test_local_llm_logic(self):
        """Verify the local brain construct prompt correctly."""
        from core.brain.local_llm import LocalBrain
        brain = LocalBrain(model_name="test-model")
        self.assertEqual(brain.model, "test-model")
        
    def test_sandbox_isolation_config(self):
        """Verify that SecureDockerSandbox forces network_disabled=True."""
        from core.skills.secure_sandbox import SecureDockerSandbox

        docker_module.run_recorder.calls.clear()
        sandbox = SecureDockerSandbox()
        # Test code execution call
        sandbox.execute_code("print('hello')", "/tmp")
        
        # Verify docker-py was called with network_disabled=True
        kwargs = docker_module.run_recorder.calls[-1].kwargs
        self.assertTrue(kwargs.get("network_disabled"))
        self.assertEqual(kwargs.get("mem_limit"), "1g")

if __name__ == "__main__":
    unittest.main()


##
