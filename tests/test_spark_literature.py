"""SPARK-002: the literature dossier stays typed, covered, and rendered."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.brain.llm.latent_cortex.epistemic_state import canonical_sha256
from core.brain.llm.latent_cortex.literature import (
    ENTRIES,
    REQUIRED_MECHANISMS,
    LiteratureEntry,
    LiteratureError,
    render_literature_markdown,
    validate_literature,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_registry_validates_and_digest_is_canonical() -> None:
    receipt = validate_literature()
    assert receipt["entry_count"] == len(ENTRIES) >= 20
    body = {key: value for key, value in receipt.items() if key != "registry_sha256"}
    assert receipt["registry_sha256"] == canonical_sha256(body)


def test_every_required_mechanism_is_grounded() -> None:
    assert {entry.mechanism for entry in ENTRIES} == set(REQUIRED_MECHANISMS)


def test_replication_discipline_is_explicit_and_mixed() -> None:
    statuses = {entry.claim_status for entry in ENTRIES}
    assert "replicated" in statuses
    assert "reported" in statuses
    for entry in ENTRIES:
        assert entry.license_declared
        assert entry.spark_items


def test_negative_result_is_first_class() -> None:
    negative = next(
        entry
        for entry in ENTRIES
        if entry.entry_id == "huang_2023_cannot_self_correct"
    )
    assert negative.claim_status == "replicated"
    assert 1 in negative.spark_items


def test_committed_doc_matches_registry_render() -> None:
    committed = (REPO_ROOT / "docs" / "RLC_SPARK_LITERATURE.md").read_text(
        encoding="utf-8"
    )
    assert committed == render_literature_markdown()


def test_entry_contract_fails_closed() -> None:
    with pytest.raises(LiteratureError):
        LiteratureEntry(
            entry_id="bad id",
            mechanism="self_consistency",
            title="A Title Long Enough",
            authors="Someone",
            year=2022,
            venue="ICLR",
            arxiv_id="2203.11171",
            claim_status="replicated",
            license_declared="arXiv license",
            supports="A grounding sentence long enough to pass.",
            spark_items=(14,),
        )
    with pytest.raises(LiteratureError):
        LiteratureEntry(
            entry_id="ok_entry_id",
            mechanism="self_consistency",
            title="A Title Long Enough",
            authors="Someone",
            year=2022,
            venue="ICLR",
            arxiv_id="not-an-arxiv-id",
            claim_status="replicated",
            license_declared="arXiv license",
            supports="A grounding sentence long enough to pass.",
            spark_items=(14,),
        )
    with pytest.raises(LiteratureError):
        LiteratureEntry(
            entry_id="ok_entry_id",
            mechanism="self_consistency",
            title="A Title Long Enough",
            authors="Someone",
            year=2022,
            venue="ICLR",
            arxiv_id="2203.11171",
            claim_status="rumor",
            license_declared="arXiv license",
            supports="A grounding sentence long enough to pass.",
            spark_items=(14,),
        )
    with pytest.raises(LiteratureError):
        LiteratureEntry(
            entry_id="ok_entry_id",
            mechanism="self_consistency",
            title="A Title Long Enough",
            authors="Someone",
            year=2022,
            venue="ICLR",
            arxiv_id="2203.11171",
            claim_status="replicated",
            license_declared="arXiv license",
            supports="A grounding sentence long enough to pass.",
            spark_items=(99,),
        )
