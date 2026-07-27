from __future__ import annotations

import importlib
import json
import os
from contextlib import contextmanager

import pytest

from tools import validate_structured_sft_tokenization as validator


def test_bound_tokenizer_loader_binds_validated_eos(monkeypatch, tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"eos_token_id": [151643, 151645]}),
        encoding="utf-8",
    )
    observed = {}
    expected = object()

    def fake_load_tokenizer(path, *, eos_token_ids):
        observed["path"] = path
        observed["eos_token_ids"] = eos_token_ids
        return expected

    monkeypatch.setattr("mlx_lm.utils.load_tokenizer", fake_load_tokenizer)

    assert validator._load_bound_tokenizer(model) is expected
    assert observed == {
        "path": model,
        "eos_token_ids": [151643, 151645],
    }


@pytest.mark.parametrize(
    "eos_token_id",
    (None, [], -1, "151643", [151643, "151645"]),
)
def test_bound_tokenizer_loader_rejects_invalid_eos(tmp_path, eos_token_id):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"eos_token_id": eos_token_id}),
        encoding="utf-8",
    )

    with pytest.raises(
        validator.TokenizerValidationError,
        match="tokenizer_eos_contract_invalid",
    ):
        validator._load_bound_tokenizer(model)


def test_tokenizer_identity_refuses_symlink_directory(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    link = tmp_path / "model-link"
    os.symlink(model, link)

    with pytest.raises(OSError):
        validator._tokenizer_identity(link)


def test_tokenizer_identity_covers_every_loader_eligible_file(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    files = {
        "tokenizer_config.json": b"{}",
        "tokenization_custom.py": b"VERSION = 1\n",
        "tekken.json": b"{}",
        "fixtures.jsonl": b'{"x":1}\n',
        "template.jinja": b"{{ messages }}",
        "merges.txt": b"a b\n",
        "tokenizer.model": b"sentencepiece",
        "vocab.tiktoken": b"token",
    }
    for name, payload in files.items():
        (model / name).write_bytes(payload)

    first = validator._tokenizer_identity(model)
    assert {row["name"] for row in first["files"]} == set(files)
    (model / "tokenization_custom.py").write_text("VERSION = 2\n")
    second = validator._tokenizer_identity(model)
    assert second["sha256"] != first["sha256"]


def test_tokenizer_identity_rejects_eligible_symlink(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer_config.json").write_text("{}")
    outside = tmp_path / "outside.py"
    outside.write_text("SECRET = True\n")
    os.symlink(outside, model / "tokenization_custom.py")

    with pytest.raises(
        validator.TokenizerValidationError,
        match="tokenizer_identity_symlink_rejected",
    ):
        validator._tokenizer_identity(model)


def _install_validate_fakes(monkeypatch, tmp_path):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    identity = {
        "directory": str(model),
        "files": [
            {
                "name": "tokenizer_config.json",
                "size_bytes": 2,
                "sha256": "a" * 64,
            }
        ],
        "sha256": "b" * 64,
    }
    tokenizer = object()
    artifacts = {"candidate_train.jsonl": b"rows"}
    custody_attestation = {
        "schema": "aura.rlc.structured_sft_custody_commit.v1",
        "generation_id": "1" * 32,
        "candidate_package_sha256": "c" * 64,
        "evaluator_package_sha256": "2" * 64,
        "custody_root_sha256": "3" * 64,
        "custody_report_sha256": "4" * 64,
        "commit_sha256": "5" * 64,
    }
    monkeypatch.setattr(
        validator,
        "read_candidate_dataset_directory_with_attestation",
        lambda _path: (artifacts, custody_attestation),
    )
    monkeypatch.setattr(
        validator,
        "validate_candidate_dataset_artifacts",
        lambda _artifacts: {
            "package_sha256": "c" * 64,
            "validation_scope": "train_validation_replay_only",
            "curriculum_manifest": {
                "curriculum_sha256": "0" * 64,
                "spec": {
                    "seed": 1,
                    "train_cases_per_family": 1,
                    "validation_cases_per_family": 1,
                    "holdout_cases_per_family": 1,
                    "max_seq_length": 4096,
                }
            },
        },
    )
    monkeypatch.setattr(
        validator,
        "build_structured_sft_curriculum",
        lambda _spec, *, holdout_seed: {"visible": holdout_seed.hex()},
    )

    snapshot_manifest = {
        "schema": validator._SNAPSHOT_SCHEMA,
        "snapshot_manifest_sha256": "f" * 64,
    }

    @contextmanager
    def fake_snapshot(_path, _snapshot_root):
        yield model, identity, snapshot_manifest

    monkeypatch.setattr(validator, "_tokenizer_snapshot", fake_snapshot)
    monkeypatch.setattr(
        validator,
        "_load_bound_tokenizer",
        lambda _path: tokenizer,
    )
    monkeypatch.setattr(
        validator,
        "_tokenizer_identity",
        lambda _path: identity,
    )
    monkeypatch.setattr(
        validator,
        "validate_trainer_tokenization",
        lambda _curriculum, *, tokenizer: {
            "schema": "test",
            "curriculum_sha256": "1" * 64,
            "report_sha256": "d" * 64,
        },
    )
    return model, identity, tokenizer


def test_validate_binds_stable_tokenizer_and_runtime_identity(
    monkeypatch,
    tmp_path,
):
    model, identity, _tokenizer = _install_validate_fakes(
        monkeypatch,
        tmp_path,
    )
    runtime = {
        "sha256": "e" * 64,
        "tokenizer_class": {"module": "test", "qualname": "Fake"},
    }
    monkeypatch.setattr(
        validator,
        "_dependency_identity",
        lambda _tokenizer: runtime,
    )

    report = validator.validate(
        candidate_directory=tmp_path / "candidate",
        tokenizer_directory=model,
        snapshot_root=tmp_path / "snapshots",
    )

    assert report["candidate_package_sha256"] == "c" * 64
    assert report["candidate_validation_scope"] == "train_validation_replay_only"
    assert report["tokenizer"]["sha256"] == identity["sha256"]
    assert report["tokenizer"]["runtime"] == runtime
    assert (
        report["tokenizer"][
            "loaded_from_persistent_content_addressed_snapshot"
        ]
        is True
    )
    assert report["trainer_binding_contract"][
        "revalidate_in_trainer_process"
    ] is True
    assert report["trainer_binding_contract"][
        "evaluator_filesystem_access_required"
    ] is False
    assert report["candidate_custody_attestation"][
        "evaluator_filesystem_accessed"
    ] is False
    assert (
        report["candidate_curriculum_commitment_sha256"]
        == "0" * 64
    )
    assert len(report["validation_bundle_sha256"]) == 64


def test_validate_refuses_runtime_identity_change(monkeypatch, tmp_path):
    model, _identity, _tokenizer = _install_validate_fakes(
        monkeypatch,
        tmp_path,
    )
    identities = iter(({"sha256": "a" * 64}, {"sha256": "b" * 64}))
    monkeypatch.setattr(
        validator,
        "_dependency_identity",
        lambda _tokenizer: next(identities),
    )

    with pytest.raises(
        validator.TokenizerValidationError,
        match="tokenizer_runtime_identity_changed_during_validation",
    ):
        validator.validate(
            candidate_directory=tmp_path / "candidate",
            tokenizer_directory=model,
            snapshot_root=tmp_path / "snapshots",
        )


def test_tokenizer_snapshot_is_persistent_and_content_addressed(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer_config.json").write_text("{}")
    (model / "config.json").write_text('{"eos_token_id":1}')
    snapshots = tmp_path / "snapshots"

    with validator._tokenizer_snapshot(model, snapshots) as (
        first_path,
        first_identity,
        first_manifest,
    ):
        assert first_path.is_dir()
        assert first_path.name == first_identity["sha256"]
        assert first_manifest["tokenizer_identity_sha256"] == first_identity[
            "sha256"
        ]
    assert first_path.is_dir()

    with validator._tokenizer_snapshot(model, snapshots) as (
        second_path,
        second_identity,
        second_manifest,
    ):
        assert second_path == first_path
        assert second_identity == first_identity
        assert second_manifest == first_manifest


def test_callable_identity_binds_loaded_code_not_only_changed_source(
    monkeypatch,
    tmp_path,
):
    module_path = tmp_path / "identity_fixture.py"
    module_path.write_text("def result():\n    return 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module("identity_fixture")
    first = validator._callable_identity(module.result)

    module_path.write_text("def result():\n    return 2\n")
    second = validator._callable_identity(module.result)

    assert module.result() == 1
    assert second["runtime_code_sha256"] == first["runtime_code_sha256"]
    assert (
        second["disk_source"]["source_file_sha256"]
        != first["disk_source"]["source_file_sha256"]
    )


def test_effective_chat_template_identity_binds_override_closure() -> None:
    def template_for(label):
        def template(*_args, **_kwargs):
            return label

        return template

    class Wrapper:
        pass

    wrapper = Wrapper()
    wrapper._chat_template = template_for("first")
    first = validator._effective_chat_template_identity(wrapper)
    wrapper._chat_template = template_for("second")
    second = validator._effective_chat_template_identity(wrapper)

    assert first["kind"] == "callable"
    assert first["callable"]["runtime_code_sha256"] == second["callable"][
        "runtime_code_sha256"
    ]
    assert first["callable"]["closure_sha256"] != second["callable"][
        "closure_sha256"
    ]


def test_effective_chat_template_identity_binds_nested_callable_closure() -> None:
    def template_for(label):
        def inner():
            return label

        def template(*_args, **_kwargs):
            return inner()

        return template

    class Wrapper:
        pass

    wrapper = Wrapper()
    wrapper._chat_template = template_for("first")
    first = validator._effective_chat_template_identity(wrapper)
    wrapper._chat_template = template_for("second")
    second = validator._effective_chat_template_identity(wrapper)

    assert first != second
