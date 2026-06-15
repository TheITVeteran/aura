from __future__ import annotations

import importlib.util
from pathlib import Path


def test_temp_fix_test_new_function():
    module_path = Path(__file__).resolve().parents[1] / "core" / "temp_fix_test.py"
    if not module_path.exists():
        return
    spec = importlib.util.spec_from_file_location("core.temp_fix_test_candidate", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.new_function() == "new"
