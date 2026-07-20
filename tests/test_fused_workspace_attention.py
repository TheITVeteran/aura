"""Fused workspace-read contracts (CP225).

The RLC's structural weakness: slots are ordinary sequence positions
competing inside the SAME attention softmax as ~200 prompt tokens, so the
workspace's share shrinks as the prompt grows. Measured result — slots are
causal, yet depth is flat. A dedicated read path gives the workspace
bandwidth that does not dilute with sequence length.

The load-bearing safety property: identity at init, so a working
checkpoint can be fused before it is trained.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.learning.fused_workspace_attention import (  # noqa: E402
    FusedWorkspaceLayer,
    WorkspaceReadHead,
    bind_workspace,
    fuse_workspace_path,
)

HIDDEN = 32


def _model() -> Model:
    args = ModelArgs(
        model_type="qwen2", hidden_size=HIDDEN, num_hidden_layers=6,
        intermediate_size=64, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=64, num_key_value_heads=2, max_position_embeddings=128,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


# ── Identity at init: fusing must be safe on a working checkpoint ───────


def test_read_head_is_exactly_identity_before_training():
    head = WorkspaceReadHead(HIDDEN, head_dim=16)
    assert head.is_identity()
    hidden = mx.random.normal((1, 5, HIDDEN), key=mx.random.key(1))
    workspace = mx.random.normal((1, 4, HIDDEN), key=mx.random.key(2))
    delta = head(hidden, workspace)
    assert float(mx.max(mx.abs(delta))) == 0.0
    assert head.to_receipt()["identity"] is True


def test_fusing_a_model_does_not_change_its_output():
    model = _model()
    tokens = mx.array([[5, 9, 17, 3]])
    before = model(tokens)
    fused = fuse_workspace_path(model, start_layer=1, stop_layer=4, head_dim=16)
    bind_workspace(fused, mx.random.normal((1, 4, HIDDEN), key=mx.random.key(3)))
    after = model(tokens)
    assert bool(mx.allclose(before, after, atol=1e-6)), (
        "an untrained fusion must be transparent, or it cannot be added to "
        "a working checkpoint safely"
    )


# ── Once trained, the workspace genuinely drives the block ──────────────


def test_trained_head_makes_the_workspace_change_the_output():
    model = _model()
    fused = fuse_workspace_path(model, start_layer=1, stop_layer=3, head_dim=16)
    for wrapper in fused.values():
        wrapper.head.out_proj = mx.ones_like(wrapper.head.out_proj) * 0.05
    tokens = mx.array([[5, 9, 17, 3]])

    bind_workspace(fused, mx.zeros((1, 4, HIDDEN)))
    with_zeros = model(tokens)
    bind_workspace(fused, mx.ones((1, 4, HIDDEN)))
    with_ones = model(tokens)
    assert not bool(mx.allclose(with_zeros, with_ones, atol=1e-6)), (
        "workspace content must reach the residual stream"
    )


def test_bandwidth_does_not_dilute_with_sequence_length():
    """The whole point: a competing softmax loses the workspace as the
    prompt grows; a dedicated head does not."""
    head = WorkspaceReadHead(HIDDEN, head_dim=16)
    head.out_proj = mx.ones_like(head.out_proj) * 0.05
    workspace = mx.ones((1, 4, HIDDEN))

    short = head(mx.ones((1, 8, HIDDEN)), workspace)
    long = head(mx.ones((1, 512, HIDDEN)), workspace)
    short_magnitude = float(mx.mean(mx.abs(short)))
    long_magnitude = float(mx.mean(mx.abs(long)))
    assert long_magnitude == pytest.approx(short_magnitude, rel=0.05), (
        "workspace influence must not shrink as the prompt grows"
    )


def test_unbound_workspace_leaves_the_block_untouched():
    model = _model()
    tokens = mx.array([[5, 9, 17, 3]])
    before = model(tokens)
    fused = fuse_workspace_path(model, start_layer=1, stop_layer=3, head_dim=16)
    for wrapper in fused.values():
        wrapper.head.out_proj = mx.ones_like(wrapper.head.out_proj) * 0.05
    # No bind_workspace call: nothing to read.
    assert bool(mx.allclose(before, model(tokens), atol=1e-6))


# ── Budget and validation ───────────────────────────────────────────────


def test_added_parameters_stay_adapter_sized():
    head = WorkspaceReadHead(5120, head_dim=64)
    # 4 projections x 5120 x 64 = 1.31M — comparable to a LoRA, trainable
    # on one machine rather than a second model.
    assert head.parameter_count() == 4 * 5120 * 64
    assert head.parameter_count() < 2_000_000


def test_configuration_and_shape_errors_fail_closed():
    with pytest.raises(ValueError, match="head_dim"):
        WorkspaceReadHead(HIDDEN, head_dim=HIDDEN + 1)
    with pytest.raises(ValueError, match="hidden_size"):
        WorkspaceReadHead(0)
    with pytest.raises(ValueError, match="scale"):
        WorkspaceReadHead(HIDDEN, head_dim=16, scale=99.0)
    head = WorkspaceReadHead(HIDDEN, head_dim=16)
    with pytest.raises(ValueError, match="hidden width"):
        head(mx.ones((1, 4, HIDDEN + 1)), mx.ones((1, 4, HIDDEN)))
    with pytest.raises(ValueError, match="workspace width"):
        head(mx.ones((1, 4, HIDDEN)), mx.ones((1, 4, HIDDEN + 1)))
    with pytest.raises(ValueError, match="layer window"):
        fuse_workspace_path(_model(), start_layer=4, stop_layer=2, head_dim=16)
