from __future__ import annotations

import sys

import pytest

from core.utils.safe_import import async_safe_import, is_missing, safe_import


def test_optional_safe_import_returns_fail_closed_missing_sentinel():
    module_name = "aura_missing_optional_dependency_for_test"

    module = safe_import(module_name, optional=True)

    assert is_missing(module)
    assert bool(module) is False
    assert module.__name__ == module_name
    with pytest.raises(ModuleNotFoundError, match=module_name):
        _ = module.some_attribute


def test_optional_safe_import_reraises_broken_transitive_dependency(tmp_path, monkeypatch):
    module_path = tmp_path / "broken_optional.py"
    module_path.write_text(
        "import missing_transitive_dependency_for_safe_import\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("broken_optional", None)

    with pytest.raises(ModuleNotFoundError, match="missing_transitive_dependency_for_safe_import"):
        safe_import("broken_optional", optional=True)


@pytest.mark.asyncio
async def test_async_safe_import_preserves_missing_sentinel_contract():
    module = await async_safe_import("aura_missing_async_optional_dependency_for_test", optional=True)

    assert is_missing(module)
    with pytest.raises(ModuleNotFoundError):
        module.call()
