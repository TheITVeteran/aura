from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.kernel.shadow_kernel import ShadowExecutionPhase


class _State:
    version = 1
    mood = "neutral"
    vitality = 100.0

    async def derive_async(self, _label):
        return self


def _phase() -> ShadowExecutionPhase:
    return ShadowExecutionPhase(SimpleNamespace(state=_State()))


@pytest.mark.asyncio
async def test_shadow_kernel_allows_pure_validation_code():
    phase = _phase()

    ok = await phase._validate_mutation(
        "def helper(value):\n    return value + 1\n",
        "def validate(state):\n    return helper(1) == 2, 'ok'\n",
    )

    assert ok is True


@pytest.mark.asyncio
async def test_shadow_kernel_blocks_forbidden_imports_before_exec():
    phase = _phase()

    ok = await phase._validate_mutation(
        "import os\n",
        "def validate(state):\n    return True, 'should not run'\n",
    )

    assert ok is False


@pytest.mark.asyncio
async def test_shadow_kernel_blocks_dynamic_exec_and_file_access():
    phase = _phase()

    assert await phase._validate_mutation(
        "def validate(state):\n    exec('x = 1')\n    return True, 'bad'\n",
        "",
    ) is False
    assert await phase._validate_mutation(
        "def validate(state):\n    open('/tmp/aura-shadow-escape', 'w')\n    return True, 'bad'\n",
        "",
    ) is False
