"""LoRA weights that are active only during latent-slot computation.

A recurrence-native adapter is not a personality adapter. It must not alter
prompt prefill, ordinary generation, the lexical answer decoder, or any other
resident-model caller. ``ScopedLoRALinear`` therefore returns the wrapped base
projection unless an explicit, task-local activation scope is open.

The optional position span supports the differentiable training view of the
live cache path: prompt, slots, and teacher-forced answer tokens can share one
causal sequence while the learned delta is applied only to the slot positions.
Live RLC calls contain slots only and use the full-span scope.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator

from mlx_lm.tuner.lora import LoRALinear


@dataclass
class RecurrenceAdapterActivation:
    """Mutable receipt for one lexical or latent execution boundary."""

    start: int | None = None
    stop: int | None = None
    calls: int = 0
    adapted_positions: int = 0
    observed_positions: int = 0

    def to_dict(self) -> dict[str, int | None]:
        return {
            "start": self.start,
            "stop": self.stop,
            "calls": self.calls,
            "adapted_positions": self.adapted_positions,
            "observed_positions": self.observed_positions,
        }


_ACTIVE_SCOPE: ContextVar[RecurrenceAdapterActivation | None] = ContextVar(
    "aura_recurrence_adapter_scope",
    default=None,
)


def current_recurrence_adapter_scope() -> RecurrenceAdapterActivation | None:
    """Return the current task-local activation, if one is open."""

    return _ACTIVE_SCOPE.get()


@contextmanager
def recurrence_adapter_scope(
    *,
    start: int | None = None,
    stop: int | None = None,
) -> Iterator[RecurrenceAdapterActivation]:
    """Activate recurrent LoRA deltas for all or a slice of sequence positions.

    ``start`` and ``stop`` follow normal non-negative slice semantics over the
    sequence axis. Both omitted means every position. Nested scopes restore the
    parent exactly, and ``ContextVar`` keeps concurrent requests isolated.
    """

    if (start is None) != (stop is None):
        raise ValueError("recurrence adapter start and stop must be supplied together")
    if start is not None and (
        type(start) is not int
        or type(stop) is not int
        or start < 0
        or stop <= start
    ):
        raise ValueError("recurrence adapter span must be a non-empty positive slice")
    activation = RecurrenceAdapterActivation(start=start, stop=stop)
    token = _ACTIVE_SCOPE.set(activation)
    try:
        yield activation
    finally:
        _ACTIVE_SCOPE.reset(token)


class ScopedLoRALinear(LoRALinear):  # type: ignore[misc]
    """A load-compatible LoRA projection gated by ``recurrence_adapter_scope``."""

    @classmethod
    def from_base(
        cls,
        linear: Any,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ) -> ScopedLoRALinear:
        """Wrap ``linear`` without the base class factory erasing our subtype."""

        from core.brain.llm.latent_cortex.fast_weights import _linear_dims

        output_dims, input_dims = _linear_dims(linear)
        scoped = cls(
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            dropout=dropout,
            scale=scale,
        )
        scoped.linear = linear
        return scoped

    def __call__(self, x: Any) -> Any:
        activation = _ACTIVE_SCOPE.get()
        y = self.linear(x)
        if activation is None:
            return y

        sequence_length = int(x.shape[-2])
        activation.calls += 1
        activation.observed_positions += sequence_length
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        if activation.start is None or activation.stop is None:
            activation.adapted_positions += sequence_length
            return y + (self.scale * z).astype(x.dtype)

        start = int(activation.start)
        stop = int(activation.stop)
        if stop > sequence_length:
            raise ValueError(
                "recurrence adapter span exceeds sequence length: "
                f"span=[{start}:{stop}) sequence={sequence_length}"
            )
        import mlx.core as mx

        positions = mx.arange(sequence_length)
        mask = ((positions >= start) & (positions < stop)).astype(x.dtype)
        shape = (1,) * max(0, x.ndim - 2) + (sequence_length, 1)
        activation.adapted_positions += stop - start
        return y + (self.scale * z * mx.reshape(mask, shape)).astype(x.dtype)


__all__ = [
    "RecurrenceAdapterActivation",
    "ScopedLoRALinear",
    "current_recurrence_adapter_scope",
    "recurrence_adapter_scope",
]
