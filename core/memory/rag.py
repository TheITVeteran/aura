from __future__ import annotations

from typing import Any


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in (text or "").split() if t.strip()]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Basic text chunking for RAG ingestion."""
    if not text:
        return []
    words = text.split()
    chunks: list[str] = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


def compute_term_freq(tokens: list[str]) -> dict[str, float]:
    if not tokens:
        return {}
    counts: dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    total = float(len(tokens))
    return {k: v / total for k, v in counts.items()}


def compute_cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = float(sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys))
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))


def _idf_weights(doc_token_sets: list[set[str]]) -> dict[str, float]:
    """Smoothed inverse document frequency over the candidate memories."""
    import math

    n_docs = len(doc_token_sets)
    df: dict[str, int] = {}
    for tokens in doc_token_sets:
        for tok in tokens:
            df[tok] = df.get(tok, 0) + 1
    return {tok: math.log(1.0 + n_docs / (1.0 + count)) for tok, count in df.items()}


def retrieve_memories(
    query: str,
    memories: list[dict[str, Any]],
    top_k: int = 5,
    threshold: float = 0.01,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """TF-IDF cosine retrieval over memory texts.

    This was a bare substring match while ``compute_term_freq`` and
    ``compute_cosine_similarity`` sat unused directly above it — the exact
    'memory retrieval backbone undercuts the amplifiers' finding from the
    July 2026 external review. Scoring is now real: term-frequency vectors
    weighted by smoothed IDF over the candidate set, cosine-compared, with a
    bounded bonus when the memory contains the query as an exact phrase
    (verbatim evidence should outrank thematic overlap at equal cosine).
    """
    import math

    query_tokens = tokenize(query)
    if not query_tokens or not memories:
        return []

    doc_tokens = [tokenize(str(m.get("text", ""))) for m in memories]
    idf = _idf_weights([set(toks) for toks in doc_tokens])
    # unseen query terms get max-rarity weight instead of vanishing
    default_idf = math.log(1.0 + len(doc_tokens))

    def weigh(tf: dict[str, float]) -> dict[str, float]:
        return {tok: freq * idf.get(tok, default_idf) for tok, freq in tf.items()}

    query_vec = weigh(compute_term_freq(query_tokens))
    query_lower = (query or "").lower().strip()

    scored: list[dict[str, Any]] = []
    for memory, tokens in zip(memories, doc_tokens):
        doc_vec = weigh(compute_term_freq(tokens))
        score = compute_cosine_similarity(query_vec, doc_vec)
        if query_lower and query_lower in str(memory.get("text", "")).lower():
            score = min(1.0, score + 0.25)
        if score >= threshold:
            item = dict(memory)
            item["score"] = round(float(score), 6)
            scored.append(item)
    scored.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return scored[:top_k]


def retrieve_memories_v2(
    query: str,
    memories: list[dict[str, Any]],
    top_k: int = 5,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    return retrieve_memories(query, memories, top_k=top_k, **kwargs)
