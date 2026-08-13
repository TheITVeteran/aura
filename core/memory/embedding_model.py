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

# ── Per-task retrieval instructions ──────────────────────────────────────
#
# Qwen3-Embedding conditions the query embedding on a natural-language task
# description. The shipped default is
#
#     "Given a web search query, retrieve relevant passages that answer the
#      query"
#
# which describes nobody's job here. Aura does not run web search; she runs
# three distinguishable retrieval jobs that until now shared one encoding
# path because MiniLM had no notion of a task at all:
#
#   MEMORY_RECALL  black_hole_vault.retrieve / rag.retrieve_memories —
#                  answering a question about something that happened.
#   EVIDENCE       cognition/evidence_relevance — scoring candidate evidence
#                  against a claim, where contradicting evidence is as
#                  relevant as supporting evidence. A "find passages that
#                  answer this" instruction actively suppresses the
#                  contradictions, which is the opposite of what an audit
#                  wants.
#   DOCUMENT       memory/ingestion_loop + document RAG — locating a passage
#                  inside ingested material.
#
# The instruction is part of the query text, so it costs one forward pass and
# nothing at storage time. Documents are NEVER instructed — the asymmetry is
# the point.
TASK_INSTRUCTIONS: dict[str, str] = {
    "memory_recall": (
        "Given a question about a past conversation or event, retrieve the "
        "remembered moments that answer it"
    ),
    "evidence": (
        "Given a claim, retrieve passages that support or contradict it"
    ),
    "document": (
        "Given a question, retrieve passages from the provided documents that "
        "answer it"
    ),
}

#: Used when a call site names no task. Deliberately the recall instruction
#: rather than the model's web-search default: every current caller of the
#: unnamed path is recalling something Aura already holds.
DEFAULT_TASK = "memory_recall"

#: Qwen3-Embedding's instruction wire format. Documented on the model card;
#: kept here so a call site never hand-rolls it.
_INSTRUCT_TEMPLATE = "Instruct: {instruction}\nQuery: "


def query_prompt(task: str | None = None) -> str:
    """The instruction prefix for ``task``.

    Unknown task names fall back to DEFAULT_TASK rather than raising — a
    retrieval is never worth losing to a typo — but the fallback is visible
    to callers that care via ``is_known_task``.
    """
    key = (task or DEFAULT_TASK).strip().lower()
    instruction = TASK_INSTRUCTIONS.get(key) or TASK_INSTRUCTIONS[DEFAULT_TASK]
    return _INSTRUCT_TEMPLATE.format(instruction=instruction)


def is_known_task(task: str | None) -> bool:
    return (task or "").strip().lower() in TASK_INSTRUCTIONS

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


def load_encoder(*, model_lane_lease: Any, device: str | None = None) -> Any:
    """Load the declared encoder under an active model-lane ownership fence.

    Raises ImportError when sentence-transformers is absent — callers own the
    fallback decision, because a silent fallback to TF-IDF is a quality change
    the caller has to record as a degradation.
    """
    from core.runtime.model_lane_control import (
        require_active_synchronous_in_process_model_lane,
    )

    require_active_synchronous_in_process_model_lane(model_lane_lease)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(REPO_ID, truncate_dim=VECTOR_DIM)
    assert_window_matches_model(model)
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


def encode_query(model: Any, texts: list[str], task: str | None = None) -> Any:
    """Encode the query side, conditioned on the retrieval task.

    ``task`` is one of TASK_INSTRUCTIONS. Passing None uses DEFAULT_TASK.
    """
    prompt = query_prompt(task)
    try:
        return model.encode(texts, prompt=prompt, show_progress_bar=False)
    except (TypeError, ValueError, KeyError):
        pass
    try:
        # Older sentence-transformers: named prompts only, no free-form prompt.
        return model.encode(texts, prompt_name=QUERY_PROMPT_NAME, show_progress_bar=False)
    except (TypeError, ValueError, KeyError):
        # No prompt support at all. Symmetric encoding is worse but correct;
        # never fail a recall over a missing template.
        return model.encode(texts, show_progress_bar=False)


def chunk_for_embedding(text: str, *, overlap_ratio: float = 0.1) -> list[str]:
    """Split ``text`` only if it genuinely overflows the encoder.

    THIS IS THE STRUCTURAL CHANGE, not a tuning knob.

    Chunking existed because the encoder's window was 256 tokens — roughly
    190 words — so any real memory had to be shredded to be embedded at all.
    The window is now 32,768 tokens (~20,000 words), which is larger than
    essentially every episode, document section, or conversation turn Aura
    stores. So chunking becomes an OVERFLOW path rather than the default one.

    Why that matters beyond convenience: a fact split across two chunks never
    co-occurs in any single vector, so no amount of scoring can relate its
    halves. That is the composition failure — correct at layer N, lost at
    N+1 — expressed in the retrieval layer. Embedding a whole episode as one
    vector is the only way those facts can be scored together.

    The size here is DERIVED from the declared window, not chosen: a chunk is
    whatever fits. Overlap is a fraction of the chunk rather than a constant,
    so it stays proportionate at any window.
    """
    if not text:
        return []
    words = text.split()
    ceiling = max_chunk_words()
    if len(words) <= ceiling:
        # The common case now, and the whole point.
        return [text]
    overlap = max(1, int(ceiling * overlap_ratio))
    stride = max(1, ceiling - overlap)
    return [" ".join(words[i:i + ceiling]) for i in range(0, len(words), stride)]


def encode_documents(model: Any, texts: list[str]) -> Any:
    """Encode the document side (no prompt prefix)."""
    return model.encode(texts, show_progress_bar=False)
