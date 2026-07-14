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
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from core.runtime.atomic_writer import atomic_write_text
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
    return hashlib.sha256(f"{context}\x00{answer}".encode()).hexdigest()[:16]


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

    def has_seen(self, context: str, answer: str) -> bool:
        """Return whether a trusted pair already has a committed ingest receipt."""

        return _pair_hash(str(context or "").strip(), str(answer or "").strip()) in self._seen

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

    def ingest_sequence(
        self,
        context: str,
        answer: str,
        encoder: Encoder,
        *,
        weight: float = 1.0,
        max_positions: int | None = None,
        max_sequence_tokens: int | None = None,
        should_continue: Callable[[], bool] | None = None,
    ) -> int:
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
            added = self.ingest_pair(context, answer, encoder, weight=weight)
            return 1 if added else 0
        h = _pair_hash(context, answer)
        if h in self._seen:
            return 0
        if should_continue is not None and not should_continue():
            return 0
        try:
            ctx_ids = list(enc_tokens(context))
            full_ids = list(enc_tokens(context + " " + answer))
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("nonparametric_ingest_tokenize", exc)
            return 0
        start = max(0, len(ctx_ids) - 1)
        positions = list(range(start, len(full_ids) - 1))
        if not positions:
            return 0
        if max_sequence_tokens is not None and len(full_ids) > max(
            1, int(max_sequence_tokens)
        ):
            return 0
        if max_positions is not None and len(positions) > max(1, int(max_positions)):
            return 0

        prepared: list[tuple[np.ndarray, int, str]] = []
        batch_encoder = getattr(encoder, "encode_hidden_sequence_ids", None)
        try:
            if callable(batch_encoder):
                keys = np.asarray(batch_encoder(full_ids), dtype=np.float32)
                if keys.ndim != 2 or keys.shape[0] < len(full_ids):
                    raise ValueError(
                        "sequence encoder returned fewer hidden states than input tokens"
                    )
                prepared = [
                    (
                        keys[position],
                        int(full_ids[position + 1]),
                        answer if position == start else "",
                    )
                    for position in positions
                ]
            else:
                # Compatibility path for lightweight/test encoders.  Compute
                # every key before mutating the datastore so cancellation can
                # never publish a partial pair.
                for position in positions:
                    if should_continue is not None and not should_continue():
                        return 0
                    prepared.append(
                        (
                            np.asarray(
                                enc_ids(full_ids[: position + 1]),
                                dtype=np.float32,
                            ),
                            int(full_ids[position + 1]),
                            answer if position == start else "",
                        )
                    )
        except (RuntimeError, AttributeError, TypeError, ValueError, IndexError) as exc:
            record_degradation("nonparametric_ingest_sequence", exc)
            return 0

        if should_continue is not None and not should_continue():
            return 0

        added = 0
        for key, token_id, token_text in prepared:
            try:
                if self._mem.add(
                    key,
                    token_id,
                    token=token_text,
                    weight=weight,
                ):
                    added += 1
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("nonparametric_ingest_sequence", exc)
        if added:
            self._seen.add(h)
        return added

    def sequence_within_budget(
        self,
        context: str,
        answer: str,
        encoder: Encoder,
        *,
        max_positions: int | None = None,
        max_sequence_tokens: int | None = None,
    ) -> bool:
        """Check sequence budgets without running a model forward or mutating memory."""

        context, answer = str(context or "").strip(), str(answer or "").strip()
        if not context or not answer:
            return False
        enc_tokens = getattr(encoder, "encode_tokens", None)
        if not callable(enc_tokens):
            return True
        try:
            ctx_ids = list(enc_tokens(context))
            full_ids = list(enc_tokens(context + " " + answer))
            positions = max(0, (len(full_ids) - 1) - max(0, len(ctx_ids) - 1))
            if positions <= 0:
                return False
            if max_sequence_tokens is not None and len(full_ids) > max(
                1, int(max_sequence_tokens)
            ):
                return False
            if max_positions is not None and positions > max(1, int(max_positions)):
                return False
            return True
        except (RuntimeError, AttributeError, TypeError, ValueError, OverflowError) as exc:
            record_degradation("nonparametric_ingest_budget", exc)
            return False

    def ingest_pairs(self, pairs: Iterable[tuple[str, str]], encoder: Encoder, *, weight: float = 1.0) -> int:
        n = 0
        for ctx, ans in pairs:
            if self.ingest_pair(ctx, ans, encoder, weight=weight):
                n += 1
        if n:
            try:
                memory_persisted = bool(self._mem.persist())
            except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("nonparametric_ingest_persist", exc)
                memory_persisted = False
            if not memory_persisted:
                return 0
            if not self.persist_seen():
                return 0
        return n

    def ingest_from_trusted_stores(self, encoder: Encoder, *, limit: int = 500) -> int:
        """Pull verifier-clean pairs from the on-disk trusted stores and ingest them.

        Default ON since the July end-to-end proof; the background job that
        calls this stays pressure-guarded. Kill switch: AURA_NONPARAMETRIC_INGEST=0.
        """
        if not _flag_on("AURA_NONPARAMETRIC_INGEST", "1"):
            return 0
        return self.ingest_pairs(collect_trusted_pairs(limit=limit), encoder)

    def persist_seen(self) -> bool:
        """Durably publish dedup receipts after the datastore is durable."""

        return self._save_seen()

    # ── dedup persistence ───────────────────────────────────────────────────────
    def _load_seen(self) -> None:
        if not self._dedup_path.exists():
            return
        try:
            self._seen = set(json.loads(self._dedup_path.read_text(encoding="utf-8")).get("seen", []))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            record_degradation("nonparametric_ingest_load_seen", exc)

    def _save_seen(self) -> bool:
        try:
            self._dedup_path.parent.mkdir(parents=True, exist_ok=True)
            # keep the dedup ledger bounded
            seen = list(self._seen)[-50_000:]
            atomic_write_text(self._dedup_path, json.dumps({"seen": seen}), encoding="utf-8")
            return True
        except (OSError, ValueError, TypeError) as exc:
            record_degradation("nonparametric_ingest_save_seen", exc)
            return False
