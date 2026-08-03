"""Role-conditioned recurrent operators for branch-local specialization.

Virtual-width branches start from different role seeds, but a shared operator
can still contract them into the same attractor. This module gives each
execution branch a small, zero-delta operator bank while retaining a shared
base LoRA. The current branch is task-local and must be published explicitly;
an attached bank never silently guesses branch zero.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Final

ROLE_CONDITIONED_SCHEMA: Final = "aura.role_conditioned_lora.v1"

_CURRENT_BRANCH: ContextVar[int | None] = ContextVar(
    "aura_recurrent_branch_index",
    default=None,
)


@contextmanager
def recurrent_branch_index(branch_index: int) -> Iterator[None]:
    """Publish one branch identity for recurrent and persisted slot passes."""

    if type(branch_index) is not int or branch_index < 0:
        raise ValueError("recurrent branch index must be a non-negative integer")
    token = _CURRENT_BRANCH.set(branch_index)
    try:
        yield
    finally:
        _CURRENT_BRANCH.reset(token)


def current_branch_index() -> int | None:
    return _CURRENT_BRANCH.get()


class RoleConditionedLoRA:
    """Shared effective factors plus trainable branch-local factor deltas."""

    def __init__(self, scoped: Any, *, branches: int, delta_scale: float = 1.0):
        if type(branches) is not int or not 2 <= branches <= 32:
            raise ValueError("role-conditioned branch count must be inside [2, 32]")
        if (
            isinstance(delta_scale, bool)
            or not isinstance(delta_scale, (int, float))
            or not 0.0 <= float(delta_scale) <= 10.0
        ):
            raise ValueError("role-conditioned delta scale must be inside [0, 10]")
        if any(
            hasattr(scoped, name)
            for name in (
                "role_a",
                "role_b",
                "role_conditioned_branches",
            )
        ):
            raise ValueError("scoped adapter already has a role-conditioned bank")
        import mlx.core as mx

        self.scoped = scoped
        self.branches = branches
        self.delta_scale = float(delta_scale)
        # Additive deltas preserve the shared operator exactly at attachment.
        # They remain trainable because the opposite shared factor is nonzero.
        scoped.role_a = [mx.zeros_like(scoped.lora_a) for _ in range(branches)]
        scoped.role_b = [mx.zeros_like(scoped.lora_b) for _ in range(branches)]
        scoped.role_conditioned_branches = branches
        scoped.role_delta_scale = self.delta_scale
        self.role_a = scoped.role_a
        self.role_b = scoped.role_b

    def factors_for(
        self,
        shared_a: Any,
        shared_b: Any,
        branch_index: int | None,
    ) -> tuple[Any, Any]:
        if branch_index is None:
            raise RuntimeError("role_conditioned_adapter_branch_context_missing")
        if type(branch_index) is not int or not 0 <= branch_index < self.branches:
            raise ValueError("role-conditioned branch index is outside the bank")
        return (
            shared_a + self.delta_scale * self.role_a[branch_index],
            shared_b + self.delta_scale * self.role_b[branch_index],
        )

    def differentiation(self) -> list[float]:
        """Effective branch operator delta relative to the shared operator."""

        import mlx.core as mx

        shared_operator = self.scoped.lora_a @ self.scoped.lora_b
        scale = mx.maximum(
            mx.linalg.norm(mx.reshape(shared_operator, (-1,))),
            1e-9,
        )
        magnitudes = []
        for index in range(self.branches):
            effective = (
                self.scoped.lora_a + self.delta_scale * self.role_a[index]
            ) @ (
                self.scoped.lora_b + self.delta_scale * self.role_b[index]
            )
            delta = effective - shared_operator
            magnitudes.append(
                mx.linalg.norm(mx.reshape(delta, (-1,))) / scale
            )
        mx.eval(magnitudes)
        return [round(float(value), 6) for value in magnitudes]

    def to_receipt(self) -> dict[str, Any]:
        import mlx.core as mx

        return {
            "schema": ROLE_CONDITIONED_SCHEMA,
            "branches": self.branches,
            "delta_scale": self.delta_scale,
            "differentiation": self.differentiation(),
            "identity_at_init": all(
                bool(
                    mx.all(self.role_a[index] == 0)
                    and mx.all(self.role_b[index] == 0)
                )
                for index in range(self.branches)
            ),
        }


def wrap_role_conditioned(
    model: Any,
    *,
    branches: int,
    delta_scale: float = 1.0,
) -> dict[str, RoleConditionedLoRA]:
    """Attach a branch-local delta bank to every recurrent scoped projection."""

    from core.brain.llm.latent_cortex.recurrence_adapter import ScopedLoRALinear

    wrapped: dict[str, RoleConditionedLoRA] = {}
    layers = getattr(getattr(model, "model", None), "layers", None) or []
    for layer_index, layer in enumerate(layers):
        for parent_name in ("self_attn", "mlp"):
            parent = getattr(layer, parent_name, None)
            if parent is None:
                continue
            for target in (
                "o_proj",
                "v_proj",
                "q_proj",
                "k_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ):
                scoped = getattr(parent, target, None)
                if not isinstance(scoped, ScopedLoRALinear):
                    continue
                site = f"model.layers.{layer_index}.{parent_name}.{target}"
                wrapped[site] = RoleConditionedLoRA(
                    scoped,
                    branches=branches,
                    delta_scale=delta_scale,
                )
                scoped.role_bank = wrapped[site]
    if not wrapped:
        raise ValueError("no scoped recurrent adapters found for role conditioning")
    return wrapped


__all__ = [
    "ROLE_CONDITIONED_SCHEMA",
    "RoleConditionedLoRA",
    "current_branch_index",
    "recurrent_branch_index",
    "wrap_role_conditioned",
]
