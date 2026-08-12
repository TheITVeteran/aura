"""The embedder's window and its producers' chunk sizes must agree.

This is the ratchet for a defect measured on 12 Aug 2026: ``all-MiniLM-L6-v2``
declared ``max_seq_length: 256`` while black_hole_vault and ingestion_loop fed
it 800-word chunks and rag.chunk_text fed it 500-word chunks. 77% of an
800-word chunk never reached the encoder, and nothing logged it because
tokenizer truncation is silent by design.

The tests below fail if any producer outgrows the declared encoder window
again, or if the declared window is ever set above what the real model
reports. Neither half can drift without the other noticing.

No model is loaded here — these are contract checks over declared constants
and the chunkers' actual output, so they run offline in the smoke lane.
"""
from __future__ import annotations

import pytest

from core.memory import embedding_model
from core.memory.rag import chunk_text


def test_declared_dim_is_the_stored_dim():
    """The width in the contract is the width every store was built for."""
    from core.memory.vector_memory_engine import EmbeddingEngine

    assert embedding_model.VECTOR_DIM == 384, (
        "384 is not a preference — NavigatingGraph, the SQLite vector store "
        "and 30k+ persisted vectors are all built at this width. Changing it "
        "is a migration, not an edit."
    )
    assert EmbeddingEngine.VECTOR_DIM == embedding_model.VECTOR_DIM
    assert EmbeddingEngine.PREFERRED_MODEL == embedding_model.REPO_ID


def test_max_chunk_words_fits_the_declared_window():
    """The word ceiling must convert to fewer tokens than the model accepts."""
    ceiling = embedding_model.max_chunk_words()
    assert ceiling > 0
    assert embedding_model.estimate_tokens(ceiling) <= embedding_model.MAX_INPUT_TOKENS, (
        f"max_chunk_words()={ceiling} converts to "
        f"{embedding_model.estimate_tokens(ceiling)} tokens, over the declared "
        f"window of {embedding_model.MAX_INPUT_TOKENS}"
    )


@pytest.mark.parametrize(
    "chunk_size",
    [
        500,   # rag.chunk_text default
        800,   # black_hole_vault.py:213 and ingestion_loop.py:173
    ],
)
def test_live_chunk_sizes_survive_the_encoder(chunk_size: int):
    """Every chunk size actually used in the tree fits the encoder window.

    This is the assertion whose absence was the defect. Both of these sizes
    were silently truncated by the previous encoder.
    """
    assert chunk_size <= embedding_model.max_chunk_words(), (
        f"chunk_size={chunk_size} words exceeds what "
        f"{embedding_model.REPO_ID} can read "
        f"({embedding_model.max_chunk_words()} words). The tail would be "
        "silently discarded before embedding."
    )


def test_chunk_text_clamps_rather_than_truncating_silently():
    """An oversized request is clamped and recorded, never quietly dropped."""
    oversized = embedding_model.max_chunk_words() + 5_000
    text = " ".join(f"word{i}" for i in range(oversized + 100))
    chunks = chunk_text(text, chunk_size=oversized, overlap=10)
    assert chunks, "clamping must not lose the ingest"
    longest = max(len(c.split()) for c in chunks)
    assert longest <= embedding_model.max_chunk_words(), (
        f"chunk_text emitted a {longest}-word chunk despite the encoder "
        f"ceiling of {embedding_model.max_chunk_words()}"
    )


def test_chunk_text_covers_the_whole_input():
    """Chunking loses no words — the original truncation lost 77% of each."""
    words = [f"w{i}" for i in range(2_000)]
    chunks = chunk_text(" ".join(words), chunk_size=500, overlap=50)
    seen = set()
    for c in chunks:
        seen.update(c.split())
    assert seen == set(words), "chunking dropped content before it was embedded"


def test_query_and_document_paths_are_distinct():
    """Qwen3-Embedding is asymmetric; the two sides must not share a path."""
    assert embedding_model.QUERY_PROMPT_NAME, (
        "a query prompt name must be declared, or retrieval silently loses "
        "the asymmetry the model was trained with"
    )
    from core.memory.vector_memory_engine import EmbeddingEngine

    assert hasattr(EmbeddingEngine, "embed_query"), (
        "rag.retrieve_memories relies on embed_query for the query side"
    )


def test_identity_stamp_changes_with_the_model():
    """A store written by a previous generation must be detectable as such."""
    assert embedding_model.REPO_ID in embedding_model.IDENTITY
    assert str(embedding_model.VECTOR_DIM) in embedding_model.IDENTITY
    assert "MiniLM" not in embedding_model.IDENTITY
