"""Role-conditioned recurrent operator contracts."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from core.brain.llm.latent_cortex.recurrence_adapter import (  # noqa: E402
    ScopedLoRALinear,
    recurrence_adapter_scope,
)
from core.learning.role_conditioned_lora import (  # noqa: E402
    RoleConditionedLoRA,
    current_branch_index,
    recurrent_branch_index,
)


def _projection() -> tuple[ScopedLoRALinear, RoleConditionedLoRA]:
    mx.random.seed(73)
    scoped = ScopedLoRALinear.from_base(
        nn.Linear(4, 3, bias=False),
        r=2,
        scale=1.0,
    )
    scoped.lora_a = mx.ones_like(scoped.lora_a) * 0.1
    scoped.lora_b = mx.ones_like(scoped.lora_b) * 0.1
    bank = RoleConditionedLoRA(scoped, branches=2)
    scoped.role_bank = bank
    return scoped, bank


def test_attached_bank_refuses_missing_branch_context():
    scoped, _bank = _projection()
    with recurrence_adapter_scope(), pytest.raises(
        RuntimeError,
        match="branch_context_missing",
    ):
        scoped(mx.ones((1, 2, 4)))


def test_zero_delta_is_exact_shared_parity_then_roles_can_diverge():
    scoped, bank = _projection()
    value = mx.ones((1, 2, 4))
    with recurrent_branch_index(0), recurrence_adapter_scope():
        branch_zero = scoped(value)
    with recurrent_branch_index(1), recurrence_adapter_scope():
        branch_one = scoped(value)
    mx.eval(branch_zero, branch_one)
    assert bool(mx.all(branch_zero == branch_one))

    bank.role_a[1] = mx.ones_like(bank.role_a[1]) * 0.2
    assert bank.differentiation()[1] > 0.0
    with recurrent_branch_index(0), recurrence_adapter_scope():
        branch_zero = scoped(value)
    with recurrent_branch_index(1), recurrence_adapter_scope():
        branch_one = scoped(value)
    mx.eval(branch_zero, branch_one)
    assert not bool(mx.all(branch_zero == branch_one))


def test_nested_branch_context_restores_exactly():
    assert current_branch_index() is None
    with recurrent_branch_index(0):
        assert current_branch_index() == 0
        with recurrent_branch_index(1):
            assert current_branch_index() == 1
        assert current_branch_index() == 0
    assert current_branch_index() is None


def test_role_bank_rejects_vacuous_single_branch_topology():
    scoped = ScopedLoRALinear.from_base(
        nn.Linear(4, 3, bias=False),
        r=2,
        scale=1.0,
    )
    with pytest.raises(ValueError, match=r"\[2, 32\]"):
        RoleConditionedLoRA(scoped, branches=1)
