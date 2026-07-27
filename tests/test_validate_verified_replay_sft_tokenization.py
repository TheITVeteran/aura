from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from test_verified_replay_sft import _bundles, _recommit, _sha
from test_verified_replay_sft_publication import _publish

from core.learning.verified_replay_sft import (
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    VerifiedReplaySFTError,
    canonical_json_bytes,
    validate_verified_replay_sft_candidate_artifacts,
    validate_verified_replay_sft_tokenization,
)
from core.learning.verified_replay_sft_publication import (
    read_candidate_publication_with_attestation,
)
from tools import validate_verified_replay_sft_tokenization as validator

_CANDIDATE_MANIFEST = "verified_replay_candidate_manifest.json"


class _ExactChatTokenizer:
    chat_template = "fixture-exact-final-assistant-template"

    @staticmethod
    def _prompt_tokens(messages: list[dict[str, Any]], tools: list[Any]) -> list[int]:
        payload = canonical_json_bytes({"messages": messages, "tools": tools})
        return [17, *payload, 18]

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[Any],
        add_generation_prompt: bool = False,
        return_dict: bool,
    ) -> list[int]:
        assert return_dict is False
        if add_generation_prompt:
            return [*self._prompt_tokens(messages, tools), 19]
        assert messages[-1]["role"] == "assistant"
        target = canonical_json_bytes(messages[-1])
        return [
            *self._prompt_tokens(messages[:-1], tools),
            19,
            *target,
            20,
        ]


class _BadPrefixTokenizer(_ExactChatTokenizer):
    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[Any],
        add_generation_prompt: bool = False,
        return_dict: bool,
    ) -> list[int]:
        if add_generation_prompt:
            return [999]
        return [1, 2, 3]


class _OverlongTokenizer(_ExactChatTokenizer):
    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[Any],
        add_generation_prompt: bool = False,
        return_dict: bool,
    ) -> list[int]:
        if add_generation_prompt:
            return [7]
        return [7] * 4_097


@pytest.fixture(scope="module")
def candidate_artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, bytes]:
    bundles, _protector, _store, _clearances, _payloads = _bundles(
        tmp_path_factory.mktemp("verified-replay-tokenization")
    )
    return dict(bundles.candidate_artifacts)


@pytest.fixture(scope="module")
def published_candidate(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, Any]]:
    (
        _protector,
        _buffer,
        _store,
        _clearances,
        _payloads,
        root,
        report,
    ) = _publish(tmp_path_factory.mktemp("verified-replay-publication"))
    return root, report


def _mutate_trainer_contract(
    artifacts: dict[str, bytes],
    **updates: Any,
) -> dict[str, bytes]:
    mutated = dict(artifacts)
    manifest = json.loads(mutated[_CANDIDATE_MANIFEST])
    manifest["trainer_contract"].update(updates)
    _recommit(manifest, "candidate_package_sha256")
    mutated[_CANDIDATE_MANIFEST] = canonical_json_bytes(manifest)
    return mutated


def _expected_receipts(
    candidate: dict[str, Any],
    tokenizer: _ExactChatTokenizer,
) -> list[dict[str, Any]]:
    receipts = []
    for split, rows in (
        (TRAIN_SPLIT, candidate["train_rows"]),
        (VALIDATION_SPLIT, candidate["validation_rows"]),
    ):
        for row in rows:
            full = tokenizer.apply_chat_template(
                row["messages"],
                tools=row["tools"],
                return_dict=False,
            )
            prefix = tokenizer.apply_chat_template(
                row["messages"][:-1],
                tools=row["tools"],
                add_generation_prompt=True,
                return_dict=False,
            )
            target = full[len(prefix) :]
            receipts.append(
                {
                    "example_sha256": row["example_sha256"],
                    "lineage_root_sha256": row["_meta"]["lineage_root_sha256"],
                    "split": split,
                    "error_class": row["_meta"]["error_class"],
                    "full_tokens_sha256": _sha(full),
                    "prefix_tokens_sha256": _sha(prefix),
                    "target_tokens_sha256": _sha(target),
                    "full_token_count": len(full),
                    "masked_prefix_token_count": len(prefix),
                    "supervised_target_token_count": len(target),
                    "target_start_index": len(prefix),
                    "prefix_exact": True,
                    "chat_dataset_process_exact": True,
                    "within_max_seq_length": True,
                }
            )
    return receipts


def _chat_dataset_process(
    tokenizer: _ExactChatTokenizer,
) -> Callable[[dict[str, Any]], tuple[list[int], int]]:
    def process(row: dict[str, Any]) -> tuple[list[int], int]:
        full = tokenizer.apply_chat_template(
            row["messages"],
            tools=row["tools"],
            return_dict=False,
        )
        prefix = tokenizer.apply_chat_template(
            row["messages"][:-1],
            tools=row["tools"],
            add_generation_prompt=True,
            return_dict=False,
        )
        return full, len(prefix)

    return process


def _install_tool_tokenizer_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    runtime_identities: tuple[dict[str, Any], dict[str, Any]] | None = None,
    final_snapshot_identity: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = tmp_path / "resident-tokenizer"
    source.mkdir(exist_ok=True)
    snapshot = tmp_path / "tokenizer-snapshots" / ("b" * 64)
    snapshot.mkdir(parents=True, exist_ok=True)
    identity = {
        "directory": str(source),
        "files": [
            {
                "name": "tokenizer_config.json",
                "size_bytes": 2,
                "sha256": "a" * 64,
            }
        ],
        "sha256": "b" * 64,
    }
    snapshot_manifest = {
        "schema": "aura.rlc.resident_tokenizer_snapshot.v1",
        "tokenizer_identity_sha256": identity["sha256"],
        "snapshot_manifest_sha256": "c" * 64,
    }
    runtime = {
        "sha256": "d" * 64,
        "tokenizer_class": {"module": __name__, "qualname": "_ExactChatTokenizer"},
    }
    tokenizer = _ExactChatTokenizer()

    @contextmanager
    def fake_snapshot(directory: Path, snapshot_root: Path):
        assert directory == source
        assert snapshot_root == tmp_path / "snapshot-root"
        yield snapshot, identity, snapshot_manifest

    if runtime_identities is not None:
        identities = iter(runtime_identities)

    def runtime_identity(_tokenizer: Any) -> dict[str, Any]:
        if runtime_identities is None:
            return runtime
        return next(identities)

    monkeypatch.setattr(validator, "resident_tokenizer_snapshot", fake_snapshot)
    monkeypatch.setattr(
        validator,
        "load_resident_tokenizer",
        lambda path: tokenizer if path == snapshot else None,
    )
    monkeypatch.setattr(
        validator,
        "resident_tokenizer_runtime_identity",
        runtime_identity,
    )
    monkeypatch.setattr(
        validator,
        "resident_tokenizer_artifact_identity",
        lambda path: (
            final_snapshot_identity or identity
            if path == snapshot
            else pytest.fail(f"unexpected snapshot path: {path}")
        ),
    )
    return source, identity, runtime, snapshot_manifest


def test_exact_full_prefix_target_receipts_cover_every_candidate_row(
    candidate_artifacts: dict[str, bytes],
) -> None:
    tokenizer = _ExactChatTokenizer()
    candidate = validate_verified_replay_sft_candidate_artifacts(candidate_artifacts)
    expected = _expected_receipts(candidate, tokenizer)

    report = validate_verified_replay_sft_tokenization(
        candidate_artifacts,
        tokenizer=tokenizer,
        chat_dataset_process=_chat_dataset_process(tokenizer),
    )

    assert report["rows_checked"] == len(expected)
    assert report["expected_rows"] == len(expected)
    assert report["rows_with_truncation"] == 0
    assert report["chat_dataset_process_mismatches"] == 0
    assert report["holdout_tokenized"] is False
    assert report["projection_receipts_sha256"] == _sha(expected)
    assert all(
        receipt["full_token_count"] > receipt["masked_prefix_token_count"]
        for receipt in expected
    )
    assert all(
        receipt["supervised_target_token_count"] > 0 for receipt in expected
    )
    assert all(
        receipt["target_start_index"] == receipt["masked_prefix_token_count"]
        for receipt in expected
    )


def test_bad_masked_prefix_is_refused(
    candidate_artifacts: dict[str, bytes],
) -> None:
    with pytest.raises(
        VerifiedReplaySFTError,
        match="verified_replay_sft_masked_prefix_not_exact",
    ):
        tokenizer = _BadPrefixTokenizer()
        validate_verified_replay_sft_tokenization(
            candidate_artifacts,
            tokenizer=tokenizer,
            chat_dataset_process=_chat_dataset_process(tokenizer),
        )


def test_chat_dataset_projection_mismatch_is_refused(
    candidate_artifacts: dict[str, bytes],
) -> None:
    tokenizer = _ExactChatTokenizer()

    with pytest.raises(
        VerifiedReplaySFTError,
        match="verified_replay_sft_chat_dataset_projection_mismatch",
    ):
        validate_verified_replay_sft_tokenization(
            candidate_artifacts,
            tokenizer=tokenizer,
            chat_dataset_process=lambda row: (
                tokenizer.apply_chat_template(
                    row["messages"],
                    tools=row["tools"],
                    return_dict=False,
                ),
                0,
            ),
        )


def test_sequence_over_contract_max_is_refused(
    candidate_artifacts: dict[str, bytes],
) -> None:
    with pytest.raises(
        VerifiedReplaySFTError,
        match="verified_replay_sft_sequence_would_truncate",
    ):
        tokenizer = _OverlongTokenizer()
        validate_verified_replay_sft_tokenization(
            candidate_artifacts,
            tokenizer=tokenizer,
            chat_dataset_process=_chat_dataset_process(tokenizer),
        )


def test_candidate_trainer_contract_mutation_is_refused_after_recommit(
    candidate_artifacts: dict[str, bytes],
) -> None:
    mutated = _mutate_trainer_contract(candidate_artifacts, mask_prompt=False)

    with pytest.raises(
        VerifiedReplaySFTError,
        match="verified_replay_sft_candidate_manifest_invalid",
    ):
        tokenizer = _ExactChatTokenizer()
        validate_verified_replay_sft_tokenization(
            mutated,
            tokenizer=tokenizer,
            chat_dataset_process=_chat_dataset_process(tokenizer),
        )


def test_tool_bundle_binds_custody_and_stable_tokenizer_without_evaluator_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (
        _protector,
        _buffer,
        _store,
        _clearances,
        _payloads,
        root,
        publication_report,
    ) = _publish(tmp_path)
    shutil.rmtree(root / "evaluator")
    source, identity, runtime, snapshot_manifest = _install_tool_tokenizer_fakes(
        monkeypatch,
        tmp_path,
    )

    report = validator.validate(
        candidate_directory=root / "candidate",
        tokenizer_directory=source,
        snapshot_root=tmp_path / "snapshot-root",
    )
    _artifacts, custody_commit = read_candidate_publication_with_attestation(
        root / "candidate"
    )

    assert not (root / "evaluator").exists()
    assert report["candidate_package_sha256"] == publication_report[
        "candidate_package_sha256"
    ]
    assert report["custody_root_sha256"] == publication_report["custody_root_sha256"]
    assert report["source_store_sha256"] == custody_commit["source_store_sha256"]
    assert report["candidate_custody_attestation"]["commit_sha256"] == (
        publication_report["publication_commit_sha256"]
    )
    assert report["candidate_custody_attestation"][
        "evaluator_filesystem_accessed"
    ] is False
    assert report["tokenizer"]["sha256"] == identity["sha256"]
    assert report["tokenizer"]["runtime"] == runtime
    assert report["tokenizer"]["snapshot_manifest"] == snapshot_manifest
    assert report["trainer_binding_contract"]["tokenizer_identity_sha256"] == (
        identity["sha256"]
    )
    assert report["trainer_binding_contract"]["runtime_identity_sha256"] == (
        runtime["sha256"]
    )
    assert report["trainer_binding_contract"]["snapshot_manifest_sha256"] == (
        snapshot_manifest["snapshot_manifest_sha256"]
    )
    assert report["trainer_binding_contract"][
        "evaluator_filesystem_access_required"
    ] is False
    assert report["evaluator_filesystem_accessed"] is False
    assert report["holdout_tokenized"] is False


def test_tool_refuses_runtime_identity_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    published_candidate: tuple[Path, dict[str, Any]],
) -> None:
    root, _publication_report = published_candidate
    source, _identity, _runtime, _snapshot_manifest = (
        _install_tool_tokenizer_fakes(
            monkeypatch,
            tmp_path,
            runtime_identities=(
                {"sha256": "1" * 64},
                {"sha256": "2" * 64},
            ),
        )
    )

    with pytest.raises(
        validator.TokenizerValidationError,
        match="tokenizer_runtime_identity_changed_during_validation",
    ):
        validator.validate(
            candidate_directory=root / "candidate",
            tokenizer_directory=source,
            snapshot_root=tmp_path / "snapshot-root",
        )


def test_tool_refuses_snapshot_identity_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    published_candidate: tuple[Path, dict[str, Any]],
) -> None:
    root, _publication_report = published_candidate
    source, _identity, _runtime, _snapshot_manifest = (
        _install_tool_tokenizer_fakes(
            monkeypatch,
            tmp_path,
            final_snapshot_identity={
                "directory": str(tmp_path / "changed"),
                "files": [],
                "sha256": "e" * 64,
            },
        )
    )

    with pytest.raises(
        validator.TokenizerValidationError,
        match="tokenizer_snapshot_changed_during_validation",
    ):
        validator.validate(
            candidate_directory=root / "candidate",
            tokenizer_directory=source,
            snapshot_root=tmp_path / "snapshot-root",
        )


def test_tool_refuses_candidate_commit_binding_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    published_candidate: tuple[Path, dict[str, Any]],
) -> None:
    root, _publication_report = published_candidate
    artifacts, commit = read_candidate_publication_with_attestation(root / "candidate")
    mismatched_commit = {
        **commit,
        "candidate_package_sha256": "f" * 64,
    }
    monkeypatch.setattr(
        validator,
        "read_candidate_publication_with_attestation",
        lambda _path: (artifacts, mismatched_commit),
    )
    source, _identity, _runtime, _snapshot_manifest = (
        _install_tool_tokenizer_fakes(monkeypatch, tmp_path)
    )

    with pytest.raises(
        validator.TokenizerValidationError,
        match="replay_candidate_custody_binding_invalid",
    ):
        validator.validate(
            candidate_directory=root / "candidate",
            tokenizer_directory=source,
            snapshot_root=tmp_path / "snapshot-root",
        )


def test_cli_success_report_remains_non_authoritative(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    published_candidate: tuple[Path, dict[str, Any]],
) -> None:
    root, _publication_report = published_candidate
    source, _identity, _runtime, _snapshot_manifest = (
        _install_tool_tokenizer_fakes(monkeypatch, tmp_path)
    )

    result = validator.main(
        [
            "--candidate-dir",
            str(root / "candidate"),
            "--tokenizer-dir",
            str(source),
            "--snapshot-root",
            str(tmp_path / "snapshot-root"),
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert result == 0
    assert report["status"] == "passed_exact_resident_tokenizer_masked_prefix"
    assert report["trainer_ready"] is False
    assert (
        report["training_authority"]
        == "none_pending_external_audit_and_trainer_admission"
    )
