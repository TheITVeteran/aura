from __future__ import annotations

import asyncio

import pytest


@pytest.mark.asyncio
async def test_task_ownership_fallback_is_allowed_in_strict_runtime(monkeypatch):
    from core.runtime import strict_task_owner
    from core.runtime import task_ownership

    strict_task_owner.reset_violations()
    loop = asyncio.get_running_loop()
    strict_task_owner.install_strict_task_owner(loop)
    monkeypatch.setenv("AURA_STRICT_RUNTIME", "1")
    monkeypatch.setattr(task_ownership, "_get_tracker", lambda: None)

    async def _owned_fallback():
        await asyncio.sleep(0)
        return "ok"

    try:
        task = task_ownership.create_tracked_task(
            _owned_fallback(),
            name="task_ownership.fallback.strict",
        )
        assert await task == "ok"
        assert strict_task_owner.violations() == []
    finally:
        strict_task_owner.restore_strict_task_owner(loop)
        strict_task_owner.reset_violations()


@pytest.mark.asyncio
async def test_task_ownership_fallback_does_not_let_children_inherit_skip(monkeypatch):
    from core.runtime import strict_task_owner
    from core.runtime import task_ownership

    strict_task_owner.reset_violations()
    loop = asyncio.get_running_loop()
    strict_task_owner.install_strict_task_owner(loop)
    monkeypatch.setenv("AURA_STRICT_RUNTIME", "1")
    monkeypatch.setattr(task_ownership, "_get_tracker", lambda: None)

    async def _owned_parent():
        async def _unowned_child():
            await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match="AURA_STRICT_RUNTIME"):
            asyncio.create_task(_unowned_child())

    try:
        await task_ownership.create_tracked_task(
            _owned_parent(),
            name="task_ownership.fallback.parent",
        )
        violations = strict_task_owner.violations()
        assert len(violations) == 1
        assert "_unowned_child" in violations[0]["coro"]
    finally:
        strict_task_owner.restore_strict_task_owner(loop)
        strict_task_owner.reset_violations()
