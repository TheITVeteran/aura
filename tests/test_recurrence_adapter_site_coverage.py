"""SPARK-066: a treatment that is only partly attached is a partial treatment.

CP227's gate was voided because the adapter never fired at all, and the repair
added an aggregate call count. This file covers the failure one level down: the
scope fires, `calls` is healthy, and some wrapped projections never ran. The
arms then differ by a fraction of the adapter, and the comparison measures
something nobody designed.

These run the real `attach_adapters` path against a real MLX module tree, not a
stub of it, so the identity being asserted is the identity the campaign gets.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx import nn  # noqa: E402

from core.brain.llm.latent_cortex.recurrence_adapter import (  # noqa: E402
    ScopedLoRALinear,
    recurrence_adapter_scope,
)
from core.learning.intrinsic_recurrence_objective import (  # noqa: E402
    IntrinsicTrainingSpec,
)
from tools.train_intrinsic_recurrence import attach_adapters  # noqa: E402

_DIM = 8
_LAYERS = 6


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.o_proj = nn.Linear(_DIM, _DIM, bias=False)
        self.v_proj = nn.Linear(_DIM, _DIM, bias=False)

    def __call__(self, x):
        return self.o_proj(self.v_proj(x))


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()

    def __call__(self, x):
        return self.self_attn(x)


class _Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = [_Layer() for _ in range(_LAYERS)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Inner()

    def __call__(self, x):
        return self.model(x)


def _spec() -> IntrinsicTrainingSpec:
    return IntrinsicTrainingSpec(prelude_end=1, coda_start=4, depths=(1, 2))


def _attached() -> tuple[_Model, dict]:
    model = _Model()
    wiring = attach_adapters(
        model,
        _spec(),
        rank=2,
        targets=("o_proj", "v_proj"),
        depth_conditioned=False,
    )
    # Non-zero deltas so an application is observable in the output too.
    for layer in model.model.layers:
        for name in ("o_proj", "v_proj"):
            projection = getattr(layer.self_attn, name)
            if isinstance(projection, ScopedLoRALinear):
                projection.lora_a = mx.ones_like(projection.lora_a)
                projection.lora_b = mx.ones_like(projection.lora_b)
    mx.eval(model.parameters())
    return model, wiring


# --- the attachment reports exactly what it wrapped -------------------------


def test_attachment_returns_every_site_it_wrapped():
    _, wiring = _attached()
    # window [1,4) plus coda [4,6), two projections each.
    assert wiring["adapted_projections"] == 10
    assert wiring["adapted_block_indices"] == [1, 2, 3, 4, 5]
    assert wiring["adapted_sites"] == sorted(
        f"model.layers.{index}.self_attn.{target}"
        for index in range(1, _LAYERS)
        for target in ("o_proj", "v_proj")
    )


def test_the_unadapted_prelude_is_not_claimed():
    _, wiring = _attached()
    assert not any(
        site.startswith("model.layers.0.") for site in wiring["adapted_sites"]
    )


# --- a healthy run fires every wrapped site ---------------------------------


def test_a_full_forward_fires_every_adapted_site():
    model, wiring = _attached()
    x = mx.ones((1, 3, _DIM))
    with recurrence_adapter_scope() as activation:
        model(x)
    mx.eval(model(x))
    assert activation.unfired_sites(wiring["adapted_sites"]) == []
    assert activation.activated_blocks() == wiring["adapted_block_indices"]
    assert activation.calls == wiring["adapted_projections"]


# --- the failure the aggregate cannot see -----------------------------------


def test_one_silently_unwrapped_projection_is_named():
    model, wiring = _attached()
    # Simulate the wrap that did not take: one site reverts to a bare Linear,
    # exactly what a target-name mismatch or an ordering bug produces.
    model.model.layers[3].self_attn.o_proj = nn.Linear(_DIM, _DIM, bias=False)
    mx.eval(model.parameters())

    x = mx.ones((1, 3, _DIM))
    with recurrence_adapter_scope() as activation:
        model(x)

    # The aggregate still looks entirely healthy.
    assert activation.calls == wiring["adapted_projections"] - 1
    assert activation.calls > 0
    # The per-site record names the exact projection that stayed dark.
    assert activation.unfired_sites(wiring["adapted_sites"]) == [
        "model.layers.3.self_attn.o_proj"
    ]


def test_a_whole_dark_block_is_named_site_by_site():
    model, wiring = _attached()
    for name in ("o_proj", "v_proj"):
        setattr(
            model.model.layers[5].self_attn, name, nn.Linear(_DIM, _DIM, bias=False)
        )
    mx.eval(model.parameters())

    x = mx.ones((1, 3, _DIM))
    with recurrence_adapter_scope() as activation:
        model(x)

    assert activation.unfired_sites(wiring["adapted_sites"]) == [
        "model.layers.5.self_attn.o_proj",
        "model.layers.5.self_attn.v_proj",
    ]
    assert 5 not in activation.activated_blocks()


def test_the_cp227_shape_leaves_every_site_unfired():
    model, wiring = _attached()
    x = mx.ones((1, 3, _DIM))
    # No scope: the whole adapter is dark, which is what voided CP227.
    bare = model(x)
    with recurrence_adapter_scope() as activation:
        pass
    mx.eval(bare)
    assert activation.calls == 0
    assert activation.unfired_sites(wiring["adapted_sites"]) == wiring["adapted_sites"]


def test_a_dark_adapter_produces_the_same_output_as_the_base_path():
    # The reason a dark adapter is dangerous rather than merely wrong: the
    # forward still runs and still answers.
    model, _ = _attached()
    x = mx.ones((1, 3, _DIM))
    outside = model(x)
    with recurrence_adapter_scope():
        inside = model(x)
    mx.eval(outside, inside)
    assert not bool(mx.array_equal(outside, inside))


# --- the identity survives into a receipt ----------------------------------


def test_activation_identity_feeds_the_penultimate_receipt():
    from core.brain.llm.latent_cortex.penultimate_execution_receipt import (
        PROVEN,
        RECURRENT_LATENT,
        latent_execution_verdict,
        penultimate_execution_receipt,
        recurrent_pass,
    )

    model, wiring = _attached()
    x = mx.ones((1, 3, _DIM))
    states = []
    with recurrence_adapter_scope() as activation:
        for _ in range(3):
            x = model(x)
            mx.eval(x)
            states.append(
                __import__("hashlib")
                .sha256(bytes(memoryview(mx.array(x).astype(mx.float32))))
                .hexdigest()
            )

    passes = [
        recurrent_pass(ordinal=index, state_sha256=digest, delta_l2=1.0)
        for index, digest in enumerate(states)
    ]
    receipt = penultimate_execution_receipt(
        mechanism=RECURRENT_LATENT,
        identity={
            "checkpoint_sha256": "a" * 64,
            "tokenizer_sha256": "b" * 64,
            "parameter_count": 1_000,
            "quantization": "fp32",
            "layer_count": _LAYERS + 2,
        },
        adapter={
            "adapter_sha256": "c" * 64,
            "attached": True,
            "expected_blocks": wiring["adapted_block_indices"],
            # Measured, not asserted.
            "activated_blocks": activation.activated_blocks(),
        },
        layer_index=_LAYERS,
        passes=passes,
        decode_state_sha256=states[-1],
        decoded_token_count=3,
        answer_sha256="d" * 64,
        fallback_occurred=False,
        fallback_reason=None,
    )
    assert latent_execution_verdict(receipt, require_adapter=True)["verdict"] == PROVEN


def test_a_partial_activation_cannot_build_a_latent_receipt():
    from core.brain.llm.latent_cortex.penultimate_execution_receipt import (
        PenultimateReceiptError,
        penultimate_execution_receipt,
        recurrent_pass,
    )

    model, wiring = _attached()
    model.model.layers[5].self_attn.o_proj = nn.Linear(_DIM, _DIM, bias=False)
    model.model.layers[5].self_attn.v_proj = nn.Linear(_DIM, _DIM, bias=False)
    mx.eval(model.parameters())

    x = mx.ones((1, 3, _DIM))
    with recurrence_adapter_scope() as activation:
        model(x)

    with pytest.raises(PenultimateReceiptError) as excinfo:
        penultimate_execution_receipt(
            mechanism="recurrent_latent",
            identity={
                "checkpoint_sha256": "a" * 64,
                "tokenizer_sha256": "b" * 64,
                "parameter_count": 1_000,
                "quantization": "fp32",
                "layer_count": _LAYERS + 2,
            },
            adapter={
                "adapter_sha256": "c" * 64,
                "attached": True,
                "expected_blocks": wiring["adapted_block_indices"],
                "activated_blocks": activation.activated_blocks(),
            },
            layer_index=_LAYERS,
            passes=[recurrent_pass(ordinal=0, state_sha256="e" * 64, delta_l2=1.0)],
            decode_state_sha256="e" * 64,
            decoded_token_count=3,
            answer_sha256="d" * 64,
            fallback_occurred=False,
            fallback_reason=None,
        )
    assert "adapter_did_not_activate" in str(excinfo.value)
