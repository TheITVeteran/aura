################################################################################

import os
import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path
from types import SimpleNamespace
from fastapi import HTTPException

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

# Mock out heavy dependencies before imports to avoid model downloads/connection errors
mock_whisper = MagicMock()
mock_docker = MagicMock()
sys.modules["faster_whisper"] = mock_whisper
sys.modules["docker"] = mock_docker

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

        with patch.object(auth.config, "api_token", ""):
            auth._verify_token(local_request, None)

        with patch.object(auth.config, "api_token", "secret_key"):
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
        
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        
        sandbox = SecureDockerSandbox()
        # Test code execution call
        sandbox.execute_code("print('hello')", "/tmp")
        
        # Verify docker-py was called with network_disabled=True
        args, kwargs = mock_client.containers.run.call_args
        self.assertTrue(kwargs.get("network_disabled"))
        self.assertEqual(kwargs.get("mem_limit"), "1g")

if __name__ == "__main__":
    unittest.main()


##
