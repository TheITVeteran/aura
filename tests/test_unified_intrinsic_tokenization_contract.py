"""Frozen dataset and tokenizer contracts for resident recurrence training."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from core.learning.recurrence_curriculum import task_battery
from tools.unified_intrinsic_tokenization_contract import (
    SOURCE_DATASET_FILENAME,
    TOKENIZED_DATASET_FILENAME,
    UnifiedTokenizationContractError,
    freeze_source_dataset,
    freeze_tokenized_dataset,
    load_source_dataset,
    verify_tokenized_dataset,
)


class ByteTokenizer:
    eos_token_id = 300

    def __init__(self, *, offset: int = 0) -> None:
        self.offset = offset

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        tokens = [ord(character) + self.offset for character in text]
        return ([299] if add_special_tokens else []) + tokens

    def decode(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del clean_up_tokenization_spaces
        return "".join(chr(token_id - self.offset) for token_id in token_ids)

    @staticmethod
    def apply_chat_template(
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str:
        assert add_generation_prompt is True
        assert tokenize is False
        return f"<user>{messages[0]['content']}</user><assistant>"


def _tasks() -> tuple[list, list]:
    train = task_battery(("khop",), (1,), 2, seed=101)
    holdout = task_battery(
        ("khop",),
        (1,),
        1,
        seed=202,
        excluded_prompts=tuple(task.prompt for task in train),
        excluded_task_ids=tuple(task.task_id for task in train),
    )
    return train, holdout


def test_source_dataset_roundtrips_exact_private_programs(tmp_path: Path) -> None:
    train, holdout = _tasks()
    path = tmp_path / SOURCE_DATASET_FILENAME
    first = freeze_source_dataset(path, train, holdout)
    second = freeze_source_dataset(path, train, holdout)
    restored_train, restored_holdout = load_source_dataset(path)

    assert first == second
    assert restored_train == train
    assert restored_holdout == holdout
    assert first["partition_overlap"] == 0
    assert stat.S_IMODE(path.stat().st_mode) == 0o400


def test_source_dataset_rejects_writable_or_symlinked_input(tmp_path: Path) -> None:
    train, holdout = _tasks()
    path = tmp_path / SOURCE_DATASET_FILENAME
    freeze_source_dataset(path, train, holdout)
    path.chmod(0o600)

    with pytest.raises(UnifiedTokenizationContractError, match="unreadable"):
        load_source_dataset(path)

    link = tmp_path / "linked" / SOURCE_DATASET_FILENAME
    link.parent.mkdir()
    link.symlink_to(path)
    with pytest.raises(UnifiedTokenizationContractError, match="symlink"):
        load_source_dataset(link)


def test_tokenized_dataset_binds_every_model_facing_token(tmp_path: Path) -> None:
    train, holdout = _tasks()
    dataset = freeze_source_dataset(
        tmp_path / SOURCE_DATASET_FILENAME,
        train,
        holdout,
    )
    path = tmp_path / TOKENIZED_DATASET_FILENAME
    tokenizer = ByteTokenizer()
    first = freeze_tokenized_dataset(
        path,
        tokenizer,
        train,
        holdout,
        bridge="\n\nFINAL_ANSWER: ",
        dataset_identity=dataset,
        tokenizer_identity_sha256="a" * 64,
    )
    second = verify_tokenized_dataset(
        path,
        tokenizer,
        train,
        holdout,
        bridge="\n\nFINAL_ANSWER: ",
        dataset_identity=dataset,
        tokenizer_identity_sha256="a" * 64,
    )

    assert first == second
    assert first["train_count"] == 2
    assert first["holdout_count"] == 1
    assert first["grounding_count"] == 462
    with pytest.raises(UnifiedTokenizationContractError, match="differs"):
        verify_tokenized_dataset(
            path,
            ByteTokenizer(offset=1),
            train,
            holdout,
            bridge="\n\nFINAL_ANSWER: ",
            dataset_identity=dataset,
            tokenizer_identity_sha256="a" * 64,
        )


def test_tokenized_dataset_refuses_create_once_drift(tmp_path: Path) -> None:
    train, holdout = _tasks()
    dataset = freeze_source_dataset(
        tmp_path / SOURCE_DATASET_FILENAME,
        train,
        holdout,
    )
    path = tmp_path / TOKENIZED_DATASET_FILENAME
    freeze_tokenized_dataset(
        path,
        ByteTokenizer(),
        train,
        holdout,
        bridge="\n\nFINAL_ANSWER: ",
        dataset_identity=dataset,
        tokenizer_identity_sha256="b" * 64,
    )
    path.chmod(0o600)
    path.write_text("{}\n", encoding="ascii")

    with pytest.raises(UnifiedTokenizationContractError, match="unreadable|differs"):
        freeze_tokenized_dataset(
            path,
            ByteTokenizer(),
            train,
            holdout,
            bridge="\n\nFINAL_ANSWER: ",
            dataset_identity=dataset,
            tokenizer_identity_sha256="b" * 64,
        )
