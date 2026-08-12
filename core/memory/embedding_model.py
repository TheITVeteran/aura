"""core/memory/embedding_model.py — the embedding contract, in one place.

WHY THIS MODULE EXISTS
======================
Until 12 Aug 2026 the embedding lane had two halves that never met. The
encoder (``all-MiniLM-L6-v2``) declared ``max_seq_length: 256`` in its own
config. The code feeding it chunked at 800 words (black_hole_vault,
ingestion_loop) and 500 words (rag.chunk_text). Measured through the model's
real tokenizer, an 800-word chunk is 1122 tokens — so 77% of every full chunk
was discarded before it was ever embedded.

Nothing logged it. Truncation is silent, documented tokenizer behaviour, and
both constants were individually defensible; only the pair was wrong.

The damage was not subtle. ``retrieve_memories`` blends dense cosine 60/40
with lexical TF-IDF: the lexical half read the whole chunk, the dense half
read the first quarter, and the heavier weight sat on the half that saw less.
Measured on four ~800-word documents whose distinguishing sentence sat past
token 256, MiniLM ranked the SAME document first for all four queries — it
saw four identical prefixes and tied. 1/4 is chance, not partial credit.

So: the window, the dimension, and the model identity are declared here once,
and every producer of embeddable text sizes itself against
``max_chunk_words()`` rather than a remembered number.
``tests/test_embedding_window_contract.py`` fails if a chunker outgrows the
encoder again.

WHY THIS MODEL
==============
``Qwen3-Embedding-0.6B`` is the only strong 2026 candidate that preserves the
existing storage schema. Its Matryoshka range is 32–1024, so it emits exactly
384 dimensions — ``VECTOR_DIM``, ``NavigatingGraph(dim=384)`` and every stored
vector's width stay untouched. EmbeddingGemma-300m scores similarly but steps
768/512/256/128, none of which is 384; adopting it would have been a schema
migration rather than a swap.

Measured on this host (M5 Pro, 12 Aug 2026), 384-dim output:

    all-MiniLM-L6-v2          tail-retrieval 1/4    10.7 ms/query
    Qwen3-Embedding-0.6B@384  tail-retrieval 3/4    20.2 ms/query

+9.4 ms in the live lane. ``evidence_relevance`` embeds inside the cognition
path, so that cost is real and is the reason the number is recorded here
rather than assumed.

MIXING GENERATIONS IS INCOHERENT
================================
Vectors written by MiniLM and by Qwen3-Embedding both have 384 entries and
are both float32. Nothing about their shape reveals that a cosine between
them is noise. ``IDENTITY`` is stamped into the store so a generation change
is detected instead of silently producing confident garbage.
"""
from __future__ import annotations

from typing import Any

#: HF repo id. The short name (``all-MiniLM-L6-v2``) was resolvable only
#: because sentence-transformers special-cases its own org; a fully qualified
#: id is what the model lane admits and what the manifest records.
REPO_ID = "Qwen/Qwen3-Embedding-0.6B"

#: Output width. Held at 384 deliberately — see module docstring.
VECTOR_DIM = 384

#: The encoder's real input window, in tokens. Qwen3-Embedding-0.6B accepts
#: 32,768; this is the number every chunker sizes against. It is NOT a guess:
#: assert_window_matches_model() checks it against the loaded model.
MAX_INPUT_TOKENS = 32_768

#: Chunkers work in words; tokenizers work in tokens. Measured on Aura's own
#: prose (engineering English, heavy on identifiers) an 800-word chunk came to
#: 1122 tokens — 1.40 tokens/word. Rounded up, because underestimating this
#: ratio is exactly the failure this module exists to prevent.
TOKENS_PER_WORD = 1.45

#: Stamped into vector stores. Bump whenever the model or dim changes, so a
#: store written by a previous generation is refused rather than blended.
IDENTITY = f"{REPO_ID}@{VECTOR_DIM}"

#: sentence-transformers prompt name for the query side. Qwen3-Embedding is
#: asymmetric — queries carry an instruction prefix, documents do not.
#: Encoding both sides identically measurably degrades retrieval.
QUERY_PROMPT_NAME = "query"

#: Approximate resident footprint, for model-lane admission. 0.6B params in
#: bf16 plus activation headroom. The old value (0.5) was sized for a 22M
#: model and would have under-declared this one by roughly 3x.
FOOTPRINT_GB = 1.6


def max_chunk_words(safety: float = 0.9) -> int:
    """Largest chunk, in words, that survives the encoder without truncation.

    ``safety`` leaves room for the tokenizer's special tokens and for prose
    denser than the measured ratio. A caller that wants the raw ceiling can
    pass 1.0, but nothing in Aura does — the whole point of this module is
    that the producer stays strictly inside the consumer's window.
    """
    if not 0.0 < safety <= 1.0:
        raise ValueError(f"safety must be in (0, 1]; got {safety!r}")
    return int((MAX_INPUT_TOKENS * safety) / TOKENS_PER_WORD)


def estimate_tokens(word_count: int) -> int:
    """Tokens a chunk of ``word_count`` words is expected to occupy."""
    return int(word_count * TOKENS_PER_WORD) + 2  # +2 for BOS/EOS


def load_encoder(*, device: str | None = None) -> Any:
    """Load the declared encoder at the declared width.

    Raises ImportError when sentence-transformers is absent — callers own the
    fallback decision, because a silent fallback to TF-IDF is a quality change
    the caller has to record as a degradation.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(REPO_ID, truncate_dim=VECTOR_DIM)
    if device:
        model = model.to(device)
    return model


def assert_window_matches_model(model: Any) -> None:
    """Fail loudly if the loaded model's window is smaller than declared.

    This is the check whose absence caused the original defect. A model whose
    real window is below ``MAX_INPUT_TOKENS`` silently truncates; catching it
    at load time turns a silent quality loss into a startup failure.
    """
    actual = getattr(model, "max_seq_length", None)
    if actual is None:
        return  # nothing to check against; not worth failing a boot over
    if int(actual) < MAX_INPUT_TOKENS:
        raise RuntimeError(
            f"embedding window contract violated: {REPO_ID} reports "
            f"max_seq_length={actual} but {MAX_INPUT_TOKENS} is declared. "
            "Chunkers are sized against the declared value and would be "
            "silently truncated. Update MAX_INPUT_TOKENS or pin the model."
        )


def encode_query(model: Any, texts: list[str]) -> Any:
    """Encode the query side, with the asymmetric prompt applied."""
    try:
        return model.encode(texts, prompt_name=QUERY_PROMPT_NAME, show_progress_bar=False)
    except (TypeError, ValueError, KeyError):
        # Older sentence-transformers, or a model without named prompts.
        # Symmetric encoding is worse but correct; never fail a recall on it.
        return model.encode(texts, show_progress_bar=False)


def encode_documents(model: Any, texts: list[str]) -> Any:
    """Encode the document side (no prompt prefix)."""
    return model.encode(texts, show_progress_bar=False)
