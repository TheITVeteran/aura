from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from core.brain.llm.latent_cortex.fast_weight_learning import (  # noqa: E402
    token_sequence_sha256,
)
from core.brain.llm.latent_cortex.semantic_output_adapter import (  # noqa: E402
    SemanticOutputAdapter,
    SemanticOutputEmbeddingProxy,
    build_semantic_output_transfer_receipt,
    deterministic_sham_tokens,
    validate_semantic_output_transfer_receipt,
)


def _fit(*, sham: bool = False) -> SemanticOutputAdapter:
    keys = np.asarray(
        [
            [1.0, 0.1, 0.0],
            [0.9, -0.1, 0.0],
            [-1.0, 0.1, 0.0],
            [-0.9, -0.1, 0.0],
            [0.0, 1.0, 0.1],
            [0.0, 0.9, -0.1],
        ],
        dtype=np.float32,
    )
    targets = (2, 2, 3, 3, 4, 4)
    incumbents = (3, 3, 4, 4, 2, 2)
    task_ids = ("train-a", "train-b", "train-c", "train-d", "train-e", "train-f")
    if sham:
        targets = deterministic_sham_tokens(
            targets,
            task_ids=task_ids,
            incumbent_tokens=incumbents,
        )
    return SemanticOutputAdapter.fit(
        keys,
        targets,
        incumbents,
        task_ids=task_ids,
        ridge=0.1,
        logit_scale=8.0,
    )


def _test_row(task_id: str, baseline: float, treatment: float, sham: float):
    return {
        "task_id_sha256": hashlib.sha256(task_id.encode()).hexdigest(),
        "baseline_score": baseline,
        "treatment_score": treatment,
        "sham_score": sham,
        "baseline_tokens_sha256": token_sequence_sha256([0]),
        "treatment_tokens_sha256": token_sequence_sha256([1]),
        "sham_tokens_sha256": token_sequence_sha256([2]),
    }


def test_fit_generalizes_signed_correction_to_unseen_hidden_rows():
    adapter = _fit()
    adapter.reset(gain=1.0)
    logits = mx.zeros((1, 1, 5))

    positive = adapter.apply(mx.array([[[0.95, 0.0, 0.0]]]), logits)
    negative = adapter.apply(mx.array([[[-0.95, 0.0, 0.0]]]), logits)

    assert int(mx.argmax(positive[0, -1])) == 2
    assert int(mx.argmax(negative[0, -1])) == 3
    assert adapter.applications == 2


def test_zero_gain_is_identity_and_erase_removes_private_tissue():
    adapter = _fit()
    adapter.reset(gain=0.0)
    logits = mx.array([[[0.0, 1.0, -1.0, 0.5]]])
    hidden = mx.array([[[1.0, 0.0, 0.0]]])
    assert bool(mx.array_equal(adapter.apply(hidden, logits), logits))
    receipt = adapter.receipt()
    assert receipt["weights_sha256"]
    assert receipt["task_sha256s"] == sorted(receipt["task_sha256s"])

    adapter.erase()
    assert adapter.receipt() == {
        "schema": "aura.rlc.semantic_output_adapter.v1",
        "erased": True,
        "token_count": 0,
        "sample_count": 6,
    }
    with pytest.raises(RuntimeError, match="erased"):
        adapter.reset(gain=1.0)


def test_tied_embedding_proxy_preserves_input_and_intercepts_only_output():
    class Embedding:
        marker = "delegated"

        def __call__(self, value):
            return value + 1

        def as_linear(self, hidden):
            return mx.zeros((1, 1, 5))

    proxy = SemanticOutputEmbeddingProxy(Embedding())
    assert int(proxy(mx.array(1))) == 2
    assert proxy.marker == "delegated"
    proxy.capture = True
    adapter = _fit()
    adapter.reset(gain=1.0)
    proxy.attach(adapter)
    logits = proxy.as_linear(mx.array([[[1.0, 0.0, 0.0]]]))
    assert int(mx.argmax(logits[0, -1])) == 2
    assert proxy.last_hidden is not None
    assert proxy.detach() is adapter
    with pytest.raises(RuntimeError, match="no adapter"):
        proxy.detach()


def test_fit_refuses_non_corrections_and_single_task_memorization():
    keys = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="corrections"):
        SemanticOutputAdapter.fit(
            keys,
            [2, 3],
            [2, 4],
            task_ids=["a", "b"],
        )
    with pytest.raises(ValueError, match="multiple tasks"):
        SemanticOutputAdapter.fit(
            keys,
            [2, 3],
            [3, 2],
            task_ids=["same", "same"],
        )


def test_sham_is_deterministic_and_changes_labels():
    targets = (2, 2, 3, 3, 4, 4)
    incumbents = (3, 3, 4, 4, 2, 2)
    tasks = ("a", "b", "c", "d", "e", "f")
    first = deterministic_sham_tokens(
        targets,
        task_ids=tasks,
        incumbent_tokens=incumbents,
    )
    second = deterministic_sham_tokens(
        targets,
        task_ids=tasks,
        incumbent_tokens=incumbents,
    )
    assert first == second
    assert sorted(first) == sorted(targets)
    assert first != targets
    assert all(value != incumbents[index] for index, value in enumerate(first))


def test_transfer_receipt_freezes_validation_gain_and_accepts_clean_test_lift():
    treatment = _fit()
    sham = _fit(sham=True)
    validation = [
        {
            "gain": gain,
            "baseline_mean": 0.25,
            "treatment_mean": score,
            "sham_mean": sham_score,
        }
        for gain, score, sham_score in (
            (0.0, 0.25, 0.25),
            (0.5, 0.50, 0.30),
            (1.0, 0.75, 0.35),
            (2.0, 0.70, 0.40),
        )
    ]
    receipt = build_semantic_output_transfer_receipt(
        treatment_identity=treatment.receipt(),
        sham_identity=sham.receipt(),
        validation_task_ids=("validation-a", "validation-b"),
        test_task_ids=("test-a", "test-b"),
        validation_rows=validation,
        test_rows=(
            _test_row("test-a", 0.25, 0.75, 0.30),
            _test_row("test-b", 0.50, 1.00, 0.40),
        ),
        erase_proven=True,
    )
    assert receipt["selected_gain"] == 1.0
    assert receipt["accepted"] is True
    assert receipt["teacher_available_during_test"] is False
    assert validate_semantic_output_transfer_receipt(receipt) == receipt

    tampered = copy.deepcopy(receipt)
    tampered["test_treatment_mean"] = 1.0
    with pytest.raises(ValueError, match="commitment"):
        validate_semantic_output_transfer_receipt(tampered)


def test_transfer_receipt_rejects_overlap_and_regression():
    treatment = _fit()
    sham = _fit(sham=True)
    validation = [
        {
            "gain": gain,
            "baseline_mean": 0.25,
            "treatment_mean": 0.5 if gain else 0.25,
            "sham_mean": 0.3 if gain else 0.25,
        }
        for gain in (0.0, 0.5, 1.0, 2.0)
    ]
    with pytest.raises(ValueError, match="overlap"):
        build_semantic_output_transfer_receipt(
            treatment_identity=treatment.receipt(),
            sham_identity=sham.receipt(),
            validation_task_ids=("validation-a",),
            test_task_ids=("train-a",),
            validation_rows=validation,
            test_rows=(_test_row("train-a", 0.25, 0.5, 0.3),),
            erase_proven=True,
        )

    receipt = build_semantic_output_transfer_receipt(
        treatment_identity=treatment.receipt(),
        sham_identity=sham.receipt(),
        validation_task_ids=("validation-a",),
        test_task_ids=("test-a", "test-b"),
        validation_rows=validation,
        test_rows=(
            _test_row("test-a", 0.5, 0.4, 0.3),
            _test_row("test-b", 0.2, 0.8, 0.3),
        ),
        erase_proven=True,
    )
    assert receipt["test_treatment_mean"] > receipt["test_baseline_mean"]
    assert receipt["test_regressions"] == 1
    assert receipt["accepted"] is False
