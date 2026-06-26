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
async def test_shadow_kernel_emits_structured_validation_receipt():
    phase = _phase()

    receipt = await phase.evaluate_mutation_safely(
        "def helper(value):\n    return value + 1\n",
        "def validate(state):\n    return helper(1) == 2, {'score': 0.75, 'note': 'ok'}\n",
    )

    assert receipt.success is True
    assert receipt.behavioral_ok is True
    assert receipt.structural_ok is True
    assert receipt.validator_info["score"] == 0.75
    assert receipt.failure_reason == ""


@pytest.mark.asyncio
async def test_shadow_kernel_receipt_records_behavioral_failure():
    phase = _phase()

    receipt = await phase.evaluate_mutation_safely(
        "def validate(state):\n    return False, 'bad behavior'\n",
        "",
    )

    assert receipt.success is False
    assert receipt.behavioral_ok is False
    assert receipt.structural_ok is False
    assert "bad behavior" in receipt.failure_reason


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
        "def validate(state):\n    open('aura-shadow-escape', 'w')\n    return True, 'bad'\n",
        "",
    ) is False
