"""The accuracy gate's decode path must actually apply the adapter.

The first CP227 accuracy-gate run (artifacts/closeout/latent_cortex/
cp227_accuracy_gate) reported on@d == off@d exactly — 6/2/0 correct in
both arms at every depth — because ``_decode`` ran ``recurrent_logits``
outside ``recurrence_adapter_scope``. ``ScopedLoRALinear`` is dark outside
the scope, so both arms decoded the bare base model and the "training does
not convert to accuracy" verdict measured base against base.

These tests pin the repaired instrument:

* the eval decode CONSULTS the adapter (scope fires, calls > 0), and
* the adapter is CAUSAL on the decoded tokens (a large delta changes the
  output), so a dark treatment can never again masquerade as a null result.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.recurrence_adapter import (  # noqa: E402
    ScopedLoRALinear,
)
from core.learning.intrinsic_recurrence import RecurrentDepthPlan  # noqa: E402
from tools.eval_intrinsic_accuracy import _decode  # noqa: E402


class _Tokenizer:
    """Just enough tokenizer for _decode: ids -> non-JSON text."""

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(97 + (token % 26)) for token in ids)


def _model() -> Model:
    mx.random.seed(7)
    args = ModelArgs(
        model_type="qwen2", hidden_size=32, num_hidden_layers=4,
        intermediate_size=64, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=64, num_key_value_heads=2, max_position_embeddings=128,
        rope_theta=10000.0,
    )
    model = Model(args)
    model.freeze()
    for index in (1, 2):
        attention = model.model.layers[index].self_attn
        attention.o_proj = ScopedLoRALinear.from_base(attention.o_proj, r=2)
    mx.eval(model.parameters())
    return model


_PLAN = RecurrentDepthPlan(prelude_end=1, coda_start=3, iterations=2)


def test_decode_consults_the_adapter_scope():
    model = _model()
    totals: dict[str, int] = {}
    _decode(
        model, _Tokenizer(), [1, 2, 3], _PLAN,
        max_tokens=2, eos=None, activation_totals=totals,
    )
    assert totals.get("calls", 0) > 0, (
        "the eval decode ran with the adapter dark — the arms would "
        "compare base against base, the original gate defect"
    )
    assert totals.get("adapted_positions", 0) > 0


def test_adapter_delta_is_causal_on_decoded_tokens():
    model = _model()
    baseline = _decode(
        model, _Tokenizer(), [1, 2, 3], _PLAN, max_tokens=4, eos=None,
    )
    for index in (1, 2):
        scoped = model.model.layers[index].self_attn.o_proj
        scoped.lora_a = mx.ones_like(scoped.lora_a)
        scoped.lora_b = mx.ones_like(scoped.lora_b) * 5.0
    mx.eval(model.parameters())
    treated = _decode(
        model, _Tokenizer(), [1, 2, 3], _PLAN, max_tokens=4, eos=None,
    )
    assert treated != baseline, (
        "a large adapter delta must change greedy decode output through "
        "the eval path; identical output means the treatment is dark"
    )


def test_zeroed_adapter_stays_base_equivalent_under_the_scope():
    """The off arm's control validity: zero factors + live scope == base."""
    model = _model()
    baseline = _decode(
        model, _Tokenizer(), [1, 2, 3], _PLAN, max_tokens=4, eos=None,
    )
    for index in (1, 2):
        scoped = model.model.layers[index].self_attn.o_proj
        scoped.lora_a = mx.zeros_like(scoped.lora_a)
        scoped.lora_b = mx.zeros_like(scoped.lora_b)
    mx.eval(model.parameters())
    zeroed = _decode(
        model, _Tokenizer(), [1, 2, 3], _PLAN, max_tokens=4, eos=None,
    )
    assert zeroed == baseline
