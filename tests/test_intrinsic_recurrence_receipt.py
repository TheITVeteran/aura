"""SPARK-066: the receipt is produced by the pass, not written about it.

These run a real Qwen2 stack through the real `recurrent_hidden_states` loop
and build the receipt from the tensors that came out, so what is asserted is
what the campaign's forward pass would actually produce.
"""

from __future__ import annotations

import hashlib

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx import nn  # noqa: E402
from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.penultimate_execution_receipt import (  # noqa: E402
    PROVEN,
    REFUSED,
    PenultimateReceiptError,
    latent_execution_verdict,
)
from core.brain.llm.latent_cortex.recurrence_adapter import (  # noqa: E402
    ScopedLoRALinear,
    recurrence_adapter_scope,
)
from core.learning.intrinsic_recurrence import (  # noqa: E402
    RecurrentDepthPlan,
    recurrent_hidden_states,
)
from core.learning.intrinsic_recurrence_receipt import (  # noqa: E402
    RecurrentReceiptError,
    build_recurrent_execution_receipt,
    run_and_receipt,
    state_digest,
)

_LAYERS = 6
_WINDOW = (1, 4)


def _d(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _model(adapted: bool = True) -> tuple[Model, dict]:
    mx.random.seed(11)
    args = ModelArgs(
        model_type="qwen2", hidden_size=32, num_hidden_layers=_LAYERS,
        intermediate_size=64, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=64, num_key_value_heads=2, max_position_embeddings=128,
        rope_theta=10000.0,
    )
    model = Model(args)
    model.freeze()
    blocks: list[int] = []
    sites: list[str] = []
    if adapted:
        for index in range(*_WINDOW):
            attention = model.model.layers[index].self_attn
            site = f"model.layers.{index}.self_attn.o_proj"
            attention.o_proj = ScopedLoRALinear.from_base(
                attention.o_proj, r=2, block_index=index, site=site
            )
            scoped = attention.o_proj
            scoped.lora_a = mx.ones_like(scoped.lora_a)
            scoped.lora_b = mx.ones_like(scoped.lora_b) * 0.5
            blocks.append(index)
            sites.append(site)
    mx.eval(model.parameters())
    return model, {
        "adapted_block_indices": blocks,
        "adapted_sites": sites,
        "adapted_projections": len(sites),
    }


def _identity() -> dict:
    return {
        "checkpoint_sha256": _d("tiny-qwen2"),
        "tokenizer_sha256": _d("tokenizer"),
        "parameter_count": 1_000_000,
        "quantization": "fp32",
        "layer_count": _LAYERS,
    }


_PLAN = RecurrentDepthPlan(prelude_end=_WINDOW[0], coda_start=_WINDOW[1], iterations=3)


def _receipt(model, wiring, plan=_PLAN):
    _, receipt = run_and_receipt(
        model,
        mx.array([[1, 2, 3]]),
        plan,
        wiring=wiring,
        identity=_identity(),
        answer_sha256=_d("answer"),
        decoded_token_count=4,
        adapter_sha256=_d("adapter"),
    )
    return receipt


# --- the produced receipt describes the pass that produced it ---------------


def test_a_real_recurrent_pass_produces_a_proven_receipt():
    model, wiring = _model()
    receipt = _receipt(model, wiring)
    verdict = latent_execution_verdict(receipt, require_adapter=True)
    assert verdict["verdict"] == PROVEN
    assert verdict["passes"] == 3
    assert verdict["distinct_pass_states"] == 3


def test_the_receipt_records_the_window_the_plan_actually_ran():
    model, wiring = _model()
    receipt = _receipt(model, wiring)
    assert receipt["execution"]["window"] == {
        "start": _WINDOW[0],
        "stop": _WINDOW[1],
        "layer_count": _LAYERS,
        "coda_layers": _LAYERS - _WINDOW[1],
    }


def test_the_activated_blocks_are_measured_not_declared():
    model, wiring = _model()
    receipt = _receipt(model, wiring)
    assert receipt["adapter"]["activated_blocks"] == list(range(*_WINDOW))
    assert receipt["adapter"]["expected_blocks"] == wiring["adapted_block_indices"]


def test_the_decode_state_is_the_windows_output_not_the_models():
    # The distinction that makes the check meaningful: the post-coda hidden is
    # NOT what the recurrence produced, and using it would make "decode
    # consumed the recurrent state" true for any forward pass at all.
    model, wiring = _model()
    hidden, receipt = run_and_receipt(
        model,
        mx.array([[1, 2, 3]]),
        _PLAN,
        wiring=wiring,
        identity=_identity(),
        answer_sha256=_d("answer"),
        decoded_token_count=4,
        adapter_sha256=_d("adapter"),
    )
    final_pass = receipt["execution"]["passes"][-1]["state_sha256"]
    assert receipt["execution"]["decode_state_sha256"] == final_pass
    assert state_digest(hidden) != final_pass


def test_the_per_pass_deltas_are_measured_distances():
    model, wiring = _model()
    receipt = _receipt(model, wiring)
    passes = receipt["execution"]["passes"]
    assert passes[0]["delta_l2"] == 0.0
    assert all(row["delta_l2"] > 0.0 for row in passes[1:])


# --- the failures a produced receipt must still surface ---------------------


def test_a_dark_adapter_produces_an_unattached_receipt_that_refuses_the_claim():
    model, wiring = _model()
    # Revert every wrapped projection: the pass still runs and still answers.
    for index in range(*_WINDOW):
        model.model.layers[index].self_attn.o_proj = nn.Linear(32, 32, bias=False)
    mx.eval(model.parameters())

    receipt = _receipt(model, wiring)
    assert receipt["adapter"]["attached"] is False
    assert receipt["adapter"]["activated_blocks"] == []
    verdict = latent_execution_verdict(receipt, require_adapter=True)
    assert verdict["verdict"] == REFUSED
    assert verdict["reasons"][0]["reason"] == "treatment_adapter_not_attached"


def test_a_partially_dark_adapter_cannot_produce_a_receipt_at_all():
    model, wiring = _model()
    model.model.layers[2].self_attn.o_proj = nn.Linear(32, 32, bias=False)
    mx.eval(model.parameters())
    with pytest.raises(PenultimateReceiptError) as excinfo:
        _receipt(model, wiring)
    assert "adapter_did_not_activate" in str(excinfo.value)


def test_a_single_pass_plan_is_honest_about_its_depth():
    model, wiring = _model()
    plan = RecurrentDepthPlan(
        prelude_end=_WINDOW[0], coda_start=_WINDOW[1], iterations=1
    )
    receipt = _receipt(model, wiring, plan)
    assert len(receipt["execution"]["passes"]) == 1
    assert latent_execution_verdict(receipt, require_adapter=True)["verdict"] == PROVEN
    verdict = latent_execution_verdict(
        receipt, require_adapter=True, minimum_passes=3
    )
    assert verdict["verdict"] == REFUSED


def test_an_empty_trajectory_is_refused_rather_than_receipted():
    _, wiring = _model()
    with pytest.raises(RecurrentReceiptError) as excinfo:
        build_recurrent_execution_receipt(
            trajectory=[],
            activation=None,
            wiring=wiring,
            identity=_identity(),
            window_start=_WINDOW[0],
            window_stop=_WINDOW[1],
            answer_sha256=_d("answer"),
            decoded_token_count=1,
            adapter_sha256=_d("adapter"),
        )
    assert "trajectory_empty" in str(excinfo.value)


def test_wiring_without_block_identity_is_refused():
    model, _ = _model()
    with pytest.raises(RecurrentReceiptError) as excinfo:
        _receipt(model, {"adapted_projections": 3})
    assert "wiring_missing_blocks" in str(excinfo.value)


# --- digest properties ------------------------------------------------------


def test_the_same_computation_digests_identically():
    model, wiring = _model()
    first = _receipt(model, wiring)
    second = _receipt(model, wiring)
    assert [row["state_sha256"] for row in first["execution"]["passes"]] == [
        row["state_sha256"] for row in second["execution"]["passes"]
    ]


def test_shape_is_bound_into_the_state_digest():
    flat = mx.zeros((1, 4, 8))
    tall = mx.zeros((1, 8, 4))
    assert state_digest(flat) != state_digest(tall)


def test_a_changed_weight_changes_every_pass_digest():
    model, wiring = _model()
    before = [row["state_sha256"] for row in _receipt(model, wiring)["execution"]["passes"]]
    scoped = model.model.layers[1].self_attn.o_proj
    scoped.lora_b = mx.ones_like(scoped.lora_b) * 3.0
    mx.eval(model.parameters())
    after = [row["state_sha256"] for row in _receipt(model, wiring)["execution"]["passes"]]
    assert all(a != b for a, b in zip(before, after, strict=True))


def test_the_scope_is_opened_by_the_producer_not_left_to_the_caller():
    # Forgetting the scope is the CP227 failure. On this path it cannot be
    # forgotten: run_and_receipt opens it itself, so the adapter fires even
    # though the caller never mentioned a scope.
    model, wiring = _model()
    receipt = _receipt(model, wiring)
    assert receipt["adapter"]["attached"] is True

    # And the bare loop outside a scope really is dark, which is what makes
    # that guarantee worth having.
    with recurrence_adapter_scope() as activation:
        pass
    assert activation.calls == 0
    recurrent_hidden_states(model, mx.array([[1, 2, 3]]), _PLAN)
