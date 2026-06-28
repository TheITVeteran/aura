"""Ingestion — populate the non-parametric memory from Aura's TRUSTED knowledge.

The datastore's capacity is only as sound as what's in it, so we ingest only
verifier-clean answers and grounded beliefs — never a raw corpus that could teach
errors. Each (context -> answer) pair becomes a datastore entry: the model's hidden
state at the context is the key, the first answer token is the value.

Encoder-pluggable so this is fully testable without a GPU (a fake encoder in tests; the
real MLXEncoder lives in nonparametric_generation). Flag-gated (AURA_NONPARAMETRIC_INGEST),
bounded, deduplicated across runs.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

import numpy as np

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.NonParametricIngest")


def _flag_on(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "on", "yes", "enabled"}


@runtime_checkable
class Encoder(Protocol):
    """Turns text into a datastore key + the first continuation token."""

    dim: int

    def encode_hidden(self, text: str) -> np.ndarray: ...
    def first_token(self, continuation: str) -> int: ...


def _pair_hash(context: str, answer: str) -> str:
    return hashlib.sha256(f"{context}\x00{answer}".encode("utf-8")).hexdigest()[:16]


def collect_trusted_pairs(
    *, limit: int = 500, sources: list[tuple[Path, str]] | None = None
) -> list[tuple[str, str]]:
    """Read (context, answer) pairs from the persisted trusted stores (cache + traces).

    Decoupled from the live objects — reads the JSON on disk so ingestion can run as a
    background job without holding references. Only source-independent task types are
    included (math/code/logic), matching what the cache itself is willing to store.
    """
    pairs: list[tuple[str, str]] = []
    if sources is None:
        sources = [
            (Path(os.path.expanduser("~/.aura/data/runtime/reasoning_solved_cache.json")), "entries"),
            (Path(os.path.expanduser("~/.aura/data/runtime/reasoning_traces.json")), "traces"),
        ]
    for path, key in sources:
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            for entry in (raw.get(key, {}) or {}).values():
                obj = str(entry.get("objective", "")).strip()
                ans = str(entry.get("answer", "")).strip()
                if obj and ans:
                    pairs.append((obj, ans))
                if len(pairs) >= limit:
                    return pairs
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            record_degradation("nonparametric_ingest_collect", exc)
    return pairs


class NonParametricIngestor:
    """Encode trusted (context -> answer) pairs into the non-parametric datastore."""

    def __init__(self, memory: Any, *, dedup_path: str | Path | None = None) -> None:
        self._mem = memory
        self._dedup_path = Path(
            dedup_path or os.path.expanduser("~/.aura/data/runtime/nonparametric_ingested.json")
        )
        self._seen: set[str] = set()
        self._load_seen()

    def ingest_pair(self, context: str, answer: str, encoder: Encoder, *, weight: float = 1.0) -> bool:
        context, answer = str(context or "").strip(), str(answer or "").strip()
        if not context or not answer:
            return False
        h = _pair_hash(context, answer)
        if h in self._seen:
            return False
        try:
            key = encoder.encode_hidden(context)
            token_id = encoder.first_token(" " + answer)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("nonparametric_ingest_encode", exc)
            return False
        if not self._mem.add(key, token_id, token=answer, weight=weight):
            return False
        self._seen.add(h)
        return True

    def ingest_sequence(self, context: str, answer: str, encoder: Encoder, *, weight: float = 1.0) -> int:
        """Full-sequence ingestion: store (prefix-hidden → next token) for every answer position.

        Single-token ingestion only lets the model recall the FIRST answer token; generation
        of a multi-token answer needs a datastore entry at each position so the chain can be
        followed. Falls back to first-token ingestion if the encoder can't encode from ids.
        Returns the number of positions added (0 if skipped/duplicate).
        """
        context, answer = str(context or "").strip(), str(answer or "").strip()
        if not context or not answer:
            return 0
        enc_ids = getattr(encoder, "encode_hidden_ids", None)
        enc_tokens = getattr(encoder, "encode_tokens", None)
        if not callable(enc_ids) or not callable(enc_tokens):
            return 1 if self.ingest_pair(context, answer, encoder, weight=weight) else 0
        h = _pair_hash(context, answer)
        if h in self._seen:
            return 0
        try:
            ctx_ids = enc_tokens(context)
            full_ids = enc_tokens(context + " " + answer)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("nonparametric_ingest_tokenize", exc)
            return 0
        added = 0
        for p in range(max(0, len(ctx_ids) - 1), len(full_ids) - 1):
            try:
                key = enc_ids(full_ids[: p + 1])
                if self._mem.add(key, int(full_ids[p + 1]), token=answer if p == len(ctx_ids) - 1 else "", weight=weight):
                    added += 1
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("nonparametric_ingest_sequence", exc)
        if added:
            self._seen.add(h)
            self._save_seen()
        return added

    def ingest_pairs(self, pairs: Iterable[tuple[str, str]], encoder: Encoder, *, weight: float = 1.0) -> int:
        n = 0
        for ctx, ans in pairs:
            if self.ingest_pair(ctx, ans, encoder, weight=weight):
                n += 1
        if n:
            self._save_seen()
            try:
                self._mem.persist()
            except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("nonparametric_ingest_persist", exc)
        return n

    def ingest_from_trusted_stores(self, encoder: Encoder, *, limit: int = 500) -> int:
        """Pull verifier-clean pairs from the on-disk trusted stores and ingest them."""
        if not _flag_on("AURA_NONPARAMETRIC_INGEST"):
            return 0
        return self.ingest_pairs(collect_trusted_pairs(limit=limit), encoder)

    # ── dedup persistence ───────────────────────────────────────────────────────
    def _load_seen(self) -> None:
        if not self._dedup_path.exists():
            return
        try:
            self._seen = set(json.loads(self._dedup_path.read_text(encoding="utf-8")).get("seen", []))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            record_degradation("nonparametric_ingest_load_seen", exc)

    def _save_seen(self) -> None:
        try:
            self._dedup_path.parent.mkdir(parents=True, exist_ok=True)
            # keep the dedup ledger bounded
            seen = list(self._seen)[-50_000:]
            self._dedup_path.write_text(json.dumps({"seen": seen}), encoding="utf-8")
        except (OSError, ValueError, TypeError) as exc:
            record_degradation("nonparametric_ingest_save_seen", exc)
