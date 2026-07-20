"""Fuse the workspace INTO the block instead of bolting it alongside (CP225).

Why the RLC has been weak, structurally:

The latent slots are ordinary sequence positions. Four slots sit alongside
~200 prompt tokens inside the SAME attention softmax, which is a normalized
competition -- so the workspace's share of attention mass shrinks as the
prompt grows, and no amount of refining its contents changes that. The
recurrence is likewise a Python loop wrapped AROUND the layer stack: the
model is never informed that it is recurring and has no dedicated route to
its own scratchpad. Measured consequences: slots are causal (destroying
them destroys the answer) yet depth is flat (25/29/25/25 across an 8x
compute range). The model reads the workspace and declines to route
reasoning through it, because routing through it was never architecturally
cheaper than solving from the prompt.

This module gives each recurrent-window layer a SECOND, dedicated read of
the workspace that does not compete with the prompt:

    h <- h + out_proj( softmax(q(h) @ k(W)^T / sqrt(d)) @ v(W) )

The workspace gets its own queries, keys and values and its own softmax, so
its bandwidth is independent of sequence length. ``out_proj`` is
zero-initialized, making the fused block EXACTLY the base block until
trained -- the same discipline that makes a LoRA identity at attach, and
the reason this can be added to a working checkpoint without risking it.

Honest framing: this ADDS parameters rather than modifying stored weights.
The base checkpoint is untouched and its fingerprint still verifies; what
changes is the computation graph. This is no longer "the same model with a
clever loop" and should not be described as such.
"""
from __future__ import annotations

import math
from typing import Any

FUSED_WORKSPACE_SCHEMA = "aura.fused_workspace_attention.v1"


class WorkspaceReadHead:
    """A dedicated, non-competing read of the latent workspace.

    Kept deliberately small: it is a routing mechanism, not a second model.
    Rank-style narrow projections mean the added parameter count is close to
    a LoRA adapter's, so this stays trainable on one machine.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        head_dim: int = 64,
        scale: float = 1.0,
    ) -> None:
        if type(hidden_size) is not int or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer")
        if type(head_dim) is not int or not 1 <= head_dim <= hidden_size:
            raise ValueError("head_dim must be inside [1, hidden_size]")
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not 0.0 <= float(scale) <= 10.0
        ):
            raise ValueError("scale must be inside [0, 10]")
        import mlx.core as mx

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.scale = float(scale)
        bound = 1.0 / math.sqrt(hidden_size)
        key = mx.random.key(0)
        self.q_proj = mx.random.uniform(
            low=-bound, high=bound, shape=(hidden_size, head_dim), key=key
        )
        self.k_proj = mx.random.uniform(
            low=-bound, high=bound, shape=(hidden_size, head_dim), key=key
        )
        self.v_proj = mx.random.uniform(
            low=-bound, high=bound, shape=(hidden_size, head_dim), key=key
        )
        # Zero output projection => exact identity at init. The fused block
        # is bit-identical to the base block until this is trained.
        self.out_proj = mx.zeros((head_dim, hidden_size))

    def is_identity(self) -> bool:
        """True while this head contributes exactly nothing."""
        import mlx.core as mx

        return bool(mx.all(self.out_proj == 0))

    def __call__(self, hidden: Any, workspace: Any) -> Any:
        """Read the workspace and return the residual contribution.

        ``hidden`` is (batch, positions, hidden); ``workspace`` is
        (batch, slots, hidden). The returned delta is added to the residual
        stream by the caller.
        """
        import mlx.core as mx

        if hidden.shape[-1] != self.hidden_size:
            raise ValueError("hidden width does not match the read head")
        if workspace.shape[-1] != self.hidden_size:
            raise ValueError("workspace width does not match the read head")
        queries = hidden @ self.q_proj
        keys = workspace @ self.k_proj
        values = workspace @ self.v_proj
        weights = mx.softmax(
            (queries @ mx.swapaxes(keys, -1, -2)) / math.sqrt(self.head_dim),
            axis=-1,
        )
        return self.scale * ((weights @ values) @ self.out_proj)

    def parameters(self) -> dict[str, Any]:
        return {
            "q_proj": self.q_proj,
            "k_proj": self.k_proj,
            "v_proj": self.v_proj,
            "out_proj": self.out_proj,
        }

    def parameter_count(self) -> int:
        return int(sum(p.size for p in self.parameters().values()))

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema": FUSED_WORKSPACE_SCHEMA,
            "hidden_size": self.hidden_size,
            "head_dim": self.head_dim,
            "scale": self.scale,
            "parameters": self.parameter_count(),
            "identity": self.is_identity(),
        }


class FusedWorkspaceLayer:
    """Wraps one transformer block so it reads the workspace directly.

    The base block runs unchanged; the read head's contribution is added to
    its output. With a zero ``out_proj`` the wrapper is transparent, so a
    model can be fused first and trained later without a risky cutover.
    """

    def __init__(self, block: Any, head: WorkspaceReadHead) -> None:
        self.block = block
        self.head = head
        self.workspace: Any = None

    def bind_workspace(self, workspace: Any) -> None:
        """Publish the current latent state for this forward pass."""
        self.workspace = workspace

    def __call__(self, hidden: Any, mask: Any = None, cache: Any = None) -> Any:
        output = self.block(hidden, mask, cache)
        if self.workspace is None:
            return output
        return output + self.head(output, self.workspace).astype(output.dtype)


def fuse_workspace_path(
    model: Any,
    *,
    start_layer: int,
    stop_layer: int,
    head_dim: int = 64,
    scale: float = 1.0,
) -> dict[int, FusedWorkspaceLayer]:
    """Give every layer in [start, stop) a dedicated workspace read.

    Returns the fused layers by index. Every head is identity at init, so
    the model's behavior is unchanged until training makes it otherwise --
    verified by a contract test rather than assumed.
    """
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if not layers:
        raise ValueError("model has no transformer layers")
    if not 0 <= start_layer < stop_layer <= len(layers):
        raise ValueError("layer window is out of range")
    hidden_size = None
    for attribute in ("hidden_size", "dim"):
        hidden_size = getattr(getattr(model, "args", None), attribute, None)
        if hidden_size:
            break
    if not hidden_size:
        raise ValueError("could not determine hidden size from the model")

    fused: dict[int, FusedWorkspaceLayer] = {}
    for index in range(start_layer, stop_layer):
        head = WorkspaceReadHead(
            int(hidden_size), head_dim=head_dim, scale=scale
        )
        wrapper = FusedWorkspaceLayer(layers[index], head)
        layers[index] = wrapper
        fused[index] = wrapper
    return fused


def bind_workspace(fused: dict[int, FusedWorkspaceLayer], workspace: Any) -> None:
    """Publish one workspace state to every fused layer."""
    for wrapper in fused.values():
        wrapper.bind_workspace(workspace)


__all__ = [
    "FUSED_WORKSPACE_SCHEMA",
    "FusedWorkspaceLayer",
    "WorkspaceReadHead",
    "bind_workspace",
    "fuse_workspace_path",
]
