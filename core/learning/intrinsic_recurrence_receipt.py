"""Build a penultimate-execution receipt out of an actual forward pass.

`penultimate_execution_receipt` refuses incoherent latent claims. This is what
produces a coherent one, by running the recurrence and digesting what really
happened rather than describing what was supposed to happen.

Three things here are load-bearing and easy to get subtly wrong:

- **The decode state is the window's output, not the model's.** The stack is
  prelude → window x T → coda → norm → head. `recurrent_hidden_states` already
  returns the per-iteration trajectory alongside the final hidden, and the
  state the *coda* consumed is `trajectory[-1]`. Digesting the final hidden
  instead would make "decode consumed the recurrent state" trivially true for
  any forward pass, including one where the window's work was discarded — the
  check would pass by construction, which is no check.

- **Digests are taken over canonical bytes.** States are cast to float32 and
  hashed as contiguous little-endian buffers with their shape, so the same
  computation digests identically across runs and a different shape can never
  collide with a different state.

- **Activation is measured, never asserted.** `activated_blocks` comes off the
  `RecurrenceAdapterActivation` the forward pass filled in. A caller cannot
  hand this function a block list.

The function does not decide whether the run is good. It reports what ran;
`latent_execution_verdict` judges it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Final

from core.brain.llm.latent_cortex.penultimate_execution_receipt import (
    RECURRENT_LATENT,
    penultimate_execution_receipt,
    recurrent_pass,
)

RECURRENT_RECEIPT_PRODUCER: Final = "aura.rlc.intrinsic_recurrence_receipt.v1"


class RecurrentReceiptError(ValueError):
    """A forward pass could not be turned into an honest receipt."""


def state_digest(state: Any) -> str:
    """Canonical digest of one hidden-state tensor.

    Shape is hashed with the bytes: two different states that happen to share
    a byte length must not share a digest.
    """

    import mlx.core as mx

    try:
        array = mx.array(state).astype(mx.float32)
        mx.eval(array)
        payload = bytes(memoryview(array))
        shape = tuple(int(dimension) for dimension in array.shape)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RecurrentReceiptError("recurrent_receipt_state_not_digestible") from exc
    digest = hashlib.sha256()
    digest.update(repr(shape).encode("ascii"))
    digest.update(b"|")
    digest.update(payload)
    return digest.hexdigest()


def _l2_delta(earlier: Any, later: Any) -> float:
    import mlx.core as mx

    difference = mx.array(later).astype(mx.float32) - mx.array(earlier).astype(
        mx.float32
    )
    value = mx.sqrt(mx.sum(difference * difference))
    mx.eval(value)
    return float(value)


def build_recurrent_execution_receipt(
    *,
    trajectory: Sequence[Any],
    activation: Any,
    wiring: Mapping[str, Any],
    identity: Mapping[str, Any],
    window_start: int,
    window_stop: int,
    answer_sha256: str,
    decoded_token_count: int,
    adapter_sha256: str | None,
) -> dict[str, Any]:
    """Turn one recurrent forward pass into a receipt of what actually ran.

    ``trajectory`` is the per-iteration list `recurrent_hidden_states` returns.
    ``activation`` is the `RecurrenceAdapterActivation` the same pass filled.
    ``wiring`` is `attach_adapters`' return value, which names the blocks the
    attachment claimed to wrap.
    """

    if not isinstance(trajectory, Sequence) or isinstance(trajectory, (str, bytes)):
        raise RecurrentReceiptError("recurrent_receipt_trajectory_invalid")
    if not trajectory:
        # A recurrence with no passes is not a latent execution to describe.
        raise RecurrentReceiptError("recurrent_receipt_trajectory_empty")
    if not isinstance(wiring, Mapping):
        raise RecurrentReceiptError("recurrent_receipt_wiring_invalid")

    expected_blocks = wiring.get("adapted_block_indices")
    if expected_blocks is None:
        raise RecurrentReceiptError("recurrent_receipt_wiring_missing_blocks")

    activated = getattr(activation, "activated_blocks", None)
    if not callable(activated):
        raise RecurrentReceiptError("recurrent_receipt_activation_invalid")
    measured_blocks = list(activated())
    attached = bool(measured_blocks)

    digests = [state_digest(state) for state in trajectory]
    passes = [
        recurrent_pass(
            ordinal=index,
            state_sha256=digest,
            delta_l2=(
                0.0 if index == 0 else _l2_delta(trajectory[index - 1], trajectory[index])
            ),
        )
        for index, digest in enumerate(digests)
    ]

    return penultimate_execution_receipt(
        mechanism=RECURRENT_LATENT,
        identity=identity,
        adapter={
            "adapter_sha256": adapter_sha256 if attached else None,
            "attached": attached,
            "expected_blocks": list(expected_blocks) if attached else [],
            # Measured off the pass that ran. Not an argument.
            "activated_blocks": measured_blocks,
        },
        window={
            "start": window_start,
            "stop": window_stop,
            "layer_count": identity.get("layer_count"),
        },
        passes=passes,
        # The state the coda consumed is the window's last output. Using the
        # post-coda hidden here would make the "decode consumed the recurrent
        # state" check true for every forward pass ever run.
        decode_state_sha256=digests[-1],
        decoded_token_count=decoded_token_count,
        answer_sha256=answer_sha256,
        fallback_occurred=False,
        fallback_reason=None,
    )


def run_and_receipt(
    model: Any,
    tokens: Any,
    plan: Any,
    *,
    wiring: Mapping[str, Any],
    identity: Mapping[str, Any],
    answer_sha256: str,
    decoded_token_count: int,
    adapter_sha256: str | None,
) -> tuple[Any, dict[str, Any]]:
    """Run the recurrent forward inside a scope and receipt it in one step.

    Returns ``(final_hidden, receipt)``. The scope is opened here rather than
    left to the caller because a caller who forgets it gets a dark adapter and
    a receipt that says so only if the identity happens to be wired — and the
    whole point is that forgetting must not be possible on this path.
    """

    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_scope,
    )
    from core.learning.intrinsic_recurrence import recurrent_hidden_states

    with recurrence_adapter_scope() as activation:
        hidden, trajectory = recurrent_hidden_states(model, tokens, plan)

    receipt = build_recurrent_execution_receipt(
        trajectory=trajectory,
        activation=activation,
        wiring=wiring,
        identity=identity,
        window_start=int(plan.prelude_end),
        window_stop=int(plan.coda_start),
        answer_sha256=answer_sha256,
        decoded_token_count=decoded_token_count,
        adapter_sha256=adapter_sha256,
    )
    return hidden, receipt


__all__ = [
    "RECURRENT_RECEIPT_PRODUCER",
    "RecurrentReceiptError",
    "build_recurrent_execution_receipt",
    "run_and_receipt",
    "state_digest",
]
