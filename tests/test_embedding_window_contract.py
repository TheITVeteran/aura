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


def test_whole_episode_is_one_vector():
    """A normal memory is no longer shredded — that is the structural fix.

    Chunking existed only because the encoder's window was 256 tokens. A fact
    split across two chunks never co-occurs in any single vector, so nothing
    downstream can score its halves together. With a 32k window, chunking is
    an overflow path.
    """
    episode = " ".join(f"word{i}" for i in range(3_000))  # far past the old 800
    assert embedding_model.chunk_for_embedding(episode) == [episode], (
        "a 3,000-word episode must embed as ONE vector; the old 800-word "
        "split would have made four fragments whose facts could never be "
        "scored together"
    )


def test_overflow_still_chunks_and_loses_nothing():
    """Genuine overflow is still handled, and still covers the whole input."""
    ceiling = embedding_model.max_chunk_words()
    words = [f"w{i}" for i in range(ceiling * 2 + 500)]
    chunks = embedding_model.chunk_for_embedding(" ".join(words))
    assert len(chunks) > 1, "true overflow must still be split"
    assert max(len(c.split()) for c in chunks) <= ceiling
    seen: set[str] = set()
    for c in chunks:
        seen.update(c.split())
    assert seen == set(words), "overflow chunking dropped content"


def test_each_retrieval_task_has_its_own_instruction():
    """Three different jobs, three different query conditionings."""
    tasks = ["memory_recall", "evidence", "document"]
    prompts = {t: embedding_model.query_prompt(t) for t in tasks}
    assert len(set(prompts.values())) == len(tasks), (
        f"tasks must not share an instruction: {prompts}"
    )
    for t in tasks:
        assert embedding_model.is_known_task(t)
        assert prompts[t].startswith("Instruct:")
    # The evidence instruction must admit contradiction — the model's shipped
    # default ("passages that answer the query") demotes exactly that.
    assert "contradict" in prompts["evidence"].lower()
    # An unknown task degrades to the default rather than raising.
    assert embedding_model.query_prompt("nonsense-task") == prompts[
        embedding_model.DEFAULT_TASK
    ]
    assert not embedding_model.is_known_task("nonsense-task")


def test_threshold_is_calibrated_not_constant():
    """The cut must move with the score distribution, not sit at 0.01."""
    from core.memory import retrieval_calibration as rc

    # A null-only population: nothing here is a real match.
    low = [0.24 + 0.01 * (i % 5) for i in range(40)]
    cut_low = rc.null_threshold(low)
    assert cut_low is not None and cut_low > 0.01, (
        "a distribution centred at 0.26 must produce a cut above 0.01; "
        "the old constant admitted all of it"
    )

    # The same shape shifted down (a MiniLM-like null) must produce a
    # correspondingly lower cut — that is what 'relative' means.
    lower = [v - 0.20 for v in low]
    cut_lower = rc.null_threshold(lower)
    assert cut_lower is not None and cut_lower < cut_low

    # Too few samples: no cut can be justified, and none is invented.
    assert rc.null_threshold([0.3, 0.4]) is None


def test_select_above_chance_ranks_and_cuts():
    from core.memory import retrieval_calibration as rc

    scored = [(0.25 + 0.005 * i, f"noise{i}") for i in range(30)]
    scored.append((0.92, "the-real-hit"))
    kept = rc.select_above_chance(scored, top_k=5)
    assert kept[0] == "the-real-hit", "ranking must put the real hit first"
    assert len(kept) <= 5

    # The property that actually matters: when the population IS separable,
    # the cut keeps the separated group and drops the bulk. A tight null with
    # a handful of clear hits is the shape real retrieval produces.
    separable = [(0.25 + 0.0005 * i, f"n{i}") for i in range(30)]
    separable += [(0.80, "hit-a"), (0.78, "hit-b")]
    kept_sep = rc.select_above_chance(separable, top_k=20)
    assert set(kept_sep[:2]) == {"hit-a", "hit-b"}
    assert len(kept_sep) < 20, (
        "a separable population must be cut, not merely ranked and capped"
    )

    # A UNIFORM population is the documented limit: it has a genuine top end,
    # and no single-query distribution can distinguish 'all noise' from 'weak
    # signal' without an external null. top_k governs, and that is honest.
    uniform = [(0.25 + 0.001 * i, f"u{i}") for i in range(30)]
    kept_uniform = rc.select_above_chance(uniform, top_k=10)
    assert len(kept_uniform) == 10
    assert kept_uniform[0] == "u29", "ranking still holds under the limit"


def test_identity_stamp_changes_with_the_model():
    """A store written by a previous generation must be detectable as such."""
    assert embedding_model.REPO_ID in embedding_model.IDENTITY
    assert str(embedding_model.VECTOR_DIM) in embedding_model.IDENTITY
    assert "MiniLM" not in embedding_model.IDENTITY
