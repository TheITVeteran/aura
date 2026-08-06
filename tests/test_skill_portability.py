from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_clean_wheel_discovers_the_complete_catalog_without_rust(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["AURA_LOG_DIR"] = str(tmp_path / "logs")
    completed = subprocess.run(
        [sys.executable, "tools/closeout/audit_skill_portability.py"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        check=False,
        text=True,
        timeout=300,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    result = json.loads(completed.stdout)
    assert result["schema"] == "aura.skill_portability_audit.v1"
    assert result["ok"] is True
    assert result["failures"] == []
    assert result["source"]["accepted_count"] == 76
    assert result["clean_install"]["accepted_count"] == 76
    assert result["clean_install"]["native_extension_available"] is False
    assert result["clean_install"]["backend"] == "python"
    assert result["clean_install"]["parity_status"] == "unavailable"
    assert result["build"]["native_member_count"] == 0
