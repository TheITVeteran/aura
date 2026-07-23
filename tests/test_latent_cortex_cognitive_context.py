from __future__ import annotations

import hashlib

import pytest

from core.brain.llm.latent_cortex.cognitive_context import (
    CognitiveContextError,
    knowledge_metadata,
    normalize_cognitive_context,
)


def _evidence_item(text: str = "The local reference reports a stable scheduler."):
    return {
        "source": "reference",
        "text": text,
        "context_role": "evidence_observation",
        "instruction_authority": False,
        "evidence_id": "evidence-1234567890abcdef12345678",
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "retrieval_receipt_sha256": "2" * 64,
        "evidence_kind": "offline_reference",
        "evidence_origin": "core.knowledge.local_corpus",
        "source_version": "test-v1",
    }


def test_evidence_observation_is_normalized_with_public_provenance():
    item = _evidence_item()

    assert normalize_cognitive_context([item]) == [item]
    assert knowledge_metadata(item) == {
        "knowledge_class": "offline_reference",
        "source_owner": "core.knowledge.local_corpus",
        "source_version": "test-v1",
        "instruction_authority": False,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("instruction_authority", True),
        ("text", "altered evidence"),
        ("content_sha256", "0" * 64),
        ("retrieval_receipt_sha256", "not-a-digest"),
        ("evidence_kind", "web_search"),
        ("evidence_origin", ""),
        ("source_version", ""),
    ],
)
def test_evidence_observation_tampering_is_rejected(field, value):
    item = _evidence_item()
    item[field] = value

    with pytest.raises(
        CognitiveContextError,
        match="evidence cognitive context authority",
    ):
        normalize_cognitive_context([item])


def test_untyped_context_cannot_smuggle_reserved_authority_fields():
    item = {
        "source": "world_model",
        "text": "A live world hypothesis.",
        "instruction_authority": True,
    }

    with pytest.raises(CognitiveContextError, match="reserved fields"):
        normalize_cognitive_context([item])


def test_context_wire_rejects_non_list_sequences():
    with pytest.raises(CognitiveContextError, match="must be a list"):
        normalize_cognitive_context((_evidence_item(),))  # type: ignore[arg-type]
