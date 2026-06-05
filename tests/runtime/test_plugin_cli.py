"""tests/runtime/test_plugin_cli.py — Unit tests for plugin approval CLI.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from core.runtime.operator_cli import run_command


def test_plugin_cli_lifecycle(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp_dir:
        p_dir = Path(tmp_dir)
        allowlist_json = p_dir / "allowlist.json"
        
        from core.security.plugin_allowlist import PluginAllowlist
        orig_init = PluginAllowlist.__init__
        
        def patched_init(self, path=None, **kwargs):
            orig_init(self, path=allowlist_json, **kwargs)
            
        monkeypatch.setattr(PluginAllowlist, "__init__", patched_init)

        p_dir = Path(tmp_dir)
        plugin_file = p_dir / "test_plugin.py"
        plugin_file.write_text("print('hello plugin')", encoding="utf-8")
        
        # 1. Test approve
        res = run_command(["plugin", "approve", str(plugin_file), "--reason", "Test CLI Approval"])
        assert res["ok"] is True
        assert "sha256" in res
        
        # 2. Test list
        res_list = run_command(["plugin", "list"])
        assert res_list["ok"] is True
        assert len(res_list["entries"]) > 0
        
        # 3. Test verify
        res_verify = run_command(["plugin", "verify", str(plugin_file)])
        assert res_verify["ok"] is True
        
        # 4. Test verify after editing (should fail)
        plugin_file.write_text("print('hello plugin edited')", encoding="utf-8")
        res_verify_drift = run_command(["plugin", "verify", str(plugin_file)])
        assert res_verify_drift["ok"] is False
        assert res_verify_drift["error"] == "hash_not_in_allowlist"
        
        # Approve again
        run_command(["plugin", "approve", str(plugin_file)])
        assert run_command(["plugin", "verify", str(plugin_file)])["ok"] is True
        
        # 5. Test revoke
        res_revoke = run_command(["plugin", "revoke", str(plugin_file)])
        assert res_revoke["ok"] is True
        
        # Verify should now fail with hash_revoked
        res_verify_revoked = run_command(["plugin", "verify", str(plugin_file)])
        assert res_verify_revoked["ok"] is False
        assert res_verify_revoked["error"] == "hash_revoked"
