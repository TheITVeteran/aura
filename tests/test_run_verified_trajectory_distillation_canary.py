from __future__ import annotations

import numpy as np
import pytest

from tools.run_verified_trajectory_distillation_canary import (
    _permuted_recurrence_adapter,
    _report_score,
    _validate_receipt_payload,
    _write_receipt,
    _zeroed_recurrence_adapter,
)


def _model_with_scoped_projection(*, coda: bool = False):
    import mlx.core as mx
    import mlx.nn as nn

    from core.brain.llm.latent_cortex.recurrence_adapter import (
        ScopedCodaLoRALinear,
        ScopedLoRALinear,
    )

    class _Attention:
        pass

    class _Layer:
        def __init__(self) -> None:
            self.self_attn = _Attention()
            wrapper_type = ScopedCodaLoRALinear if coda else ScopedLoRALinear
            projection = wrapper_type.from_base(
                nn.Linear(5, 5),
                r=2,
                scale=1.0,
                block_index=0,
                site="model.layers.0.self_attn.o_proj",
            )
            projection.lora_b = mx.arange(10, dtype=mx.float32).reshape((2, 5))
            mx.eval(projection.lora_b)
            self.self_attn.o_proj = projection

    class _Inner:
        def __init__(self) -> None:
            self.layers = [_Layer()]

    class _Model:
        def __init__(self) -> None:
            self.model = _Inner()

    return _Model()


def test_zeroed_recurrence_control_restores_after_exception() -> None:
    model = _model_with_scoped_projection()
    projection = model.model.layers[0].self_attn.o_proj
    before = np.asarray(projection.lora_b).copy()

    with pytest.raises(RuntimeError, match="probe failed"):
        with _zeroed_recurrence_adapter(model):
            assert not np.any(np.asarray(projection.lora_b))
            raise RuntimeError("probe failed")

    assert np.array_equal(np.asarray(projection.lora_b), before)


def test_permuted_recurrence_control_preserves_norm_and_restores() -> None:
    model = _model_with_scoped_projection()
    projection = model.model.layers[0].self_attn.o_proj
    before = np.asarray(projection.lora_b).copy()

    with _permuted_recurrence_adapter(model):
        observed = np.asarray(projection.lora_b)
        assert not np.array_equal(observed, before)
        assert np.linalg.norm(observed) == pytest.approx(np.linalg.norm(before))

    assert np.array_equal(np.asarray(projection.lora_b), before)


def test_lesion_controls_accept_decode_scoped_tissue() -> None:
    model = _model_with_scoped_projection(coda=True)
    projection = model.model.layers[0].self_attn.o_proj
    before = np.asarray(projection.lora_b).copy()

    with _zeroed_recurrence_adapter(model):
        assert not np.any(np.asarray(projection.lora_b))
    with _permuted_recurrence_adapter(model):
        assert not np.array_equal(np.asarray(projection.lora_b), before)

    assert np.array_equal(np.asarray(projection.lora_b), before)


def test_report_score_requires_integer_total() -> None:
    assert _report_score({"total_correct": 3}) == 3
    with pytest.raises(KeyError):
        _report_score({})


def test_receipt_writer_hashes_the_exact_persisted_body(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    body = {
        "schema": "aura.test.receipt.v1",
        "nested": {7: (np.float64(0.36725152632439967), "evidence")},
    }

    receipt = _write_receipt(path, body)

    assert path.stat().st_mode & 0o777 == 0o600
    assert _validate_receipt_payload(path.read_bytes()) == receipt
    assert receipt["nested"] == {"7": [0.36725152632439967, "evidence"]}


def test_receipt_validator_rejects_post_write_mutation(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    _write_receipt(path, {"schema": "aura.test.receipt.v1", "admitted": True})
    payload = path.read_bytes().replace(b'"admitted":true', b'"admitted":false')

    with pytest.raises(RuntimeError, match="hash does not bind"):
        _validate_receipt_payload(payload)
