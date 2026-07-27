from __future__ import annotations

import copy

import pytest

from core.learning.recurrent_sft_falsification import sha256_json
from tools import train_recurrent_sft_controls as controls


def _projected() -> dict:
    body = {
        "schema": controls.PROJECTED_DATASET_SCHEMA,
        "candidate_identity_sha256": "1" * 64,
        "train": [
            {
                "example_id": "2" * 64,
                "family": "logic",
                "target_kind": "answer",
                "prompt_tokens": [1, 2],
                "answer_tokens": [3, 4],
                "full_token_count": 4,
            },
            {
                "example_id": "3" * 64,
                "family": "tool",
                "target_kind": "answer",
                "prompt_tokens": [5, 6],
                "answer_tokens": [7, 8],
                "full_token_count": 4,
            },
        ],
        "validation": [{"sealed": "candidate-only"}],
        "holdout": None,
        "verified_replay": None,
    }
    return {**body, "dataset_sha256": sha256_json(body)}


def test_projected_dataset_accepts_bound_candidate_only_projection() -> None:
    document = _projected()
    assert controls._projected_dataset(document) == document


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("holdout", [], "projected_dataset_invalid"),
        ("verified_replay", {}, "projected_dataset_invalid"),
        ("dataset_sha256", "f" * 64, "commitment_mismatch"),
    ],
)
def test_projected_dataset_rejects_leakage_and_tampering(
    field: str,
    value: object,
    match: str,
) -> None:
    document = _projected()
    document[field] = value
    with pytest.raises(controls.RecurrentSFTControlTrainingError, match=match):
        controls._projected_dataset(document)


def test_reference_checkpoint_requires_exact_single_epoch_workload() -> None:
    projected = _projected()
    authority = {
        "authority_sha256": "4" * 64,
        "model": {"identity_sha256": "5" * 64},
        "execution_spec": {"semantic_sha256": "6" * 64},
        "trainer": {"bound": True},
    }
    checkpoint = {
        "schema": controls.REFERENCE_COMPLETION_SCHEMA,
        "terminal": True,
        "last_step_committed": True,
        "authority_sha256": authority["authority_sha256"],
        "dataset_sha256": projected["dataset_sha256"],
        "model_identity_sha256": authority["model"]["identity_sha256"],
        "execution_spec_sha256": authority["execution_spec"]["semantic_sha256"],
        "trainer_config_sha256": sha256_json(authority["trainer"]),
        "step": 2,
        "optimizer_updates": 2,
        "epoch": 0,
        "cursor": 2,
        "order": [1, 0],
        "adapter": {
            "path": "adapter.safetensors",
            "sha256": "7" * 64,
            "size_bytes": 10,
        },
        "optimizer": {
            "path": "optimizer.safetensors",
            "sha256": "8" * 64,
            "size_bytes": 20,
        },
    }
    assert (
        controls._reference_checkpoint(
            checkpoint,
            authority=authority,
            projected_dataset=projected,
        )
        == checkpoint
    )
    for field, value in (
        ("optimizer_updates", 1),
        ("epoch", 1),
        ("cursor", 1),
        ("order", [0, 0]),
    ):
        tampered = copy.deepcopy(checkpoint)
        tampered[field] = value
        with pytest.raises(
            controls.RecurrentSFTControlTrainingError,
            match="reference_checkpoint_invalid",
        ):
            controls._reference_checkpoint(
                tampered,
                authority=authority,
                projected_dataset=projected,
            )


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert text == "x"
        assert add_special_tokens is False
        return [17]

    def decode(self, tokens: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        return f"token-{tokens[0]}"


def test_tokenizer_contract_builds_atomic_neutral_and_complete_surfaces() -> None:
    tokenizer = _Tokenizer()
    assert controls._neutral_token_id(tokenizer) == 17
    assert controls._token_surfaces(tokenizer, _projected()["train"]) == {
        3: "token-3",
        4: "token-4",
        7: "token-7",
        8: "token-8",
    }


def test_neutral_token_must_be_atomic() -> None:
    tokenizer = _Tokenizer()
    tokenizer.encode = lambda *_args, **_kwargs: [1, 2]  # type: ignore[method-assign]
    with pytest.raises(
        controls.RecurrentSFTControlTrainingError,
        match="neutral_token_not_atomic",
    ):
        controls._neutral_token_id(tokenizer)


def test_tensor_fingerprint_binds_name_shape_dtype_and_values() -> None:
    import numpy as np

    baseline = controls._tensor_fingerprint(
        {"layer.lora_a": np.asarray([[1.0, 2.0]], dtype=np.float32)}
    )
    assert baseline == controls._tensor_fingerprint(
        {"layer.lora_a": np.asarray([[1.0, 2.0]], dtype=np.float32)}
    )
    assert baseline != controls._tensor_fingerprint(
        {"layer.lora_a": np.asarray([[1.0, 3.0]], dtype=np.float32)}
    )
    assert baseline != controls._tensor_fingerprint(
        {"other.lora_a": np.asarray([[1.0, 2.0]], dtype=np.float32)}
    )


def test_save_adapter_uses_real_mlx_filename_contract(tmp_path) -> None:
    import mlx.core as mx

    output = tmp_path / "control.safetensors"
    receipt = controls._save_adapter(
        output,
        {"layer.lora_a": mx.array([[1.0, 2.0]])},
    )

    assert output.is_file()
    assert receipt["filename"] == output.name
    assert receipt["size_bytes"] == output.stat().st_size
    assert not list(tmp_path.glob(".*.tmp.safetensors"))


def test_control_source_closure_is_exact_and_hash_bound() -> None:
    closure = controls.control_source_closure()
    assert closure["schema"].endswith("control_source_closure.v1")
    assert len(closure["files"]) == 13
    assert [row["role"] for row in closure["files"]] == sorted(
        row["role"] for row in closure["files"]
    )
    assert closure["closure_sha256"] == sha256_json(
        {
            "schema": closure["schema"],
            "files": closure["files"],
        }
    )
