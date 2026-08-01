from __future__ import annotations

from pathlib import Path

import pytest

from core.learning.resident_recurrent_sft_bootstrap_authority import sha256_json
from tools import resident_recurrent_sft_bootstrap_identity as identity
from tools.validate_structured_sft_tokenization import TokenizerValidationError


def test_absent_personality_identity_is_explicit_and_self_bound() -> None:
    observed = identity.absent_personality_identity()
    body = dict(observed)
    claimed = body.pop("identity_sha256")

    assert observed["present"] is False
    assert observed["file_count"] == 0
    assert claimed == sha256_json(body)


def test_runtime_identity_binds_installed_dependency_bytes() -> None:
    observed = identity.resident_bootstrap_runtime_identity()
    body = dict(observed)
    claimed = body.pop("identity_sha256")

    assert claimed == sha256_json(body)
    for name in ("mlx", "mlx-lm", "numpy"):
        dependency = observed["dependencies"][name]
        assert dependency["distribution"] == name
        assert dependency["file_count"] == len(dependency["files"])
        assert dependency["total_bytes"] == sum(
            record["size_bytes"] for record in dependency["files"]
        )
        assert len(dependency["tree_sha256"]) == 64


def test_tokenizer_identity_binds_artifact_and_runtime(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        identity,
        "resident_tokenizer_artifact_identity",
        lambda _path: {"sha256": "a" * 64, "files": []},
    )
    monkeypatch.setattr(
        identity,
        "resident_tokenizer_runtime_identity",
        lambda _tokenizer: {"sha256": "b" * 64, "callables": []},
    )

    observed = identity.resident_bootstrap_tokenizer_identity(tmp_path, object())
    body = dict(observed)
    claimed = body.pop("identity_sha256")

    assert observed["artifact_sha256"] == "a" * 64
    assert observed["runtime_sha256"] == "b" * 64
    assert claimed == sha256_json(body)


def test_tokenizer_only_loader_preserves_configured_eos_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text('{"eos_token_id":[2,3]}')
    observed: dict[str, object] = {}

    def load(path: Path, *, eos_token_ids: object) -> object:
        observed.update(path=path, eos=eos_token_ids)
        return object()

    monkeypatch.setattr("mlx_lm.utils.load_tokenizer", load)
    tokenizer = identity.load_resident_bootstrap_tokenizer(tmp_path)

    assert tokenizer is not None
    assert observed == {"path": tmp_path.resolve(), "eos": [2, 3]}


@pytest.mark.parametrize("eos", [None, True, -1, [], [2, "3"]])
def test_tokenizer_only_loader_rejects_invalid_eos_contract(
    tmp_path: Path,
    eos: object,
) -> None:
    import json

    (tmp_path / "config.json").write_text(json.dumps({"eos_token_id": eos}))
    with pytest.raises(TokenizerValidationError, match="tokenizer_eos_contract_invalid"):
        identity.load_resident_bootstrap_tokenizer(tmp_path)
