from __future__ import annotations

from pathlib import Path

from core.learning.resident_recurrent_sft_bootstrap_authority import sha256_json
from tools import resident_recurrent_sft_bootstrap_identity as identity


def test_absent_personality_identity_is_explicit_and_self_bound() -> None:
    observed = identity.absent_personality_identity()
    body = dict(observed)
    claimed = body.pop("identity_sha256")

    assert observed["present"] is False
    assert observed["file_count"] == 0
    assert claimed == sha256_json(body)


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
