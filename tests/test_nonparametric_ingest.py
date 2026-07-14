"""Tests for non-parametric memory ingestion (trusted knowledge -> datastore)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from core.brain.nonparametric_ingest import NonParametricIngestor, collect_trusted_pairs
from core.brain.nonparametric_memory import NonParametricMemory


class FakeEncoder:
    dim = 8

    def encode_hidden(self, text: str) -> np.ndarray:
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        return np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)

    def first_token(self, continuation: str) -> int:
        return int(hashlib.sha256(continuation.encode()).hexdigest()[:4], 16) % 1000


class FakeSeqEncoder(FakeEncoder):
    """Adds the id-level hooks ingest_sequence needs (prefix-consistent tokenization)."""

    def encode_tokens(self, text: str) -> list[int]:
        return [int(hashlib.sha1(w.encode()).hexdigest()[:6], 16) % 5000 for w in text.split()]

    def encode_hidden_ids(self, ids: list[int]) -> np.ndarray:
        seed = int(hashlib.sha256(str(list(ids)).encode()).hexdigest()[:8], 16)
        v = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
        return (v / np.linalg.norm(v)).astype(np.float32)

    def encode_hidden(self, text: str) -> np.ndarray:
        # consistent with encode_hidden_ids (as the real MLXEncoder is)
        return self.encode_hidden_ids(self.encode_tokens(text))


class FakeBatchSeqEncoder(FakeSeqEncoder):
    def __init__(self) -> None:
        self.batch_calls = 0
        self.prefix_calls = 0

    def encode_hidden_ids(self, ids: list[int]) -> np.ndarray:
        self.prefix_calls += 1
        return super().encode_hidden_ids(ids)

    def encode_hidden_sequence_ids(self, ids: list[int]) -> np.ndarray:
        self.batch_calls += 1
        return np.vstack(
            [super(FakeBatchSeqEncoder, self).encode_hidden_ids(ids[: index + 1])
             for index in range(len(ids))]
        )


def _ingestor(tmp_path):
    mem = NonParametricMemory(dim=8, path=tmp_path / "npm")
    ing = NonParametricIngestor(mem, dedup_path=tmp_path / "seen.json")
    return mem, ing, FakeEncoder()


def test_ingest_pair_adds_and_recalls(tmp_path):
    mem, ing, enc = _ingestor(tmp_path)
    assert ing.ingest_pair("what is the capital of Vorth", "Myrrhal", enc) is True
    assert len(mem) == 1
    nbrs = mem.query(enc.encode_hidden("what is the capital of Vorth"))
    assert nbrs[0].token_id == enc.first_token(" Myrrhal")
    assert nbrs[0].token == "Myrrhal"


def test_ingest_pair_dedups(tmp_path):
    mem, ing, enc = _ingestor(tmp_path)
    assert ing.ingest_pair("q", "a", enc) is True
    assert ing.ingest_pair("q", "a", enc) is False    # same pair → skipped
    assert len(mem) == 1


def test_ingest_pair_rejects_empty(tmp_path):
    mem, ing, enc = _ingestor(tmp_path)
    assert ing.ingest_pair("", "a", enc) is False
    assert ing.ingest_pair("q", "", enc) is False
    assert len(mem) == 0


def test_ingest_pairs_counts_and_persists(tmp_path):
    mem, ing, enc = _ingestor(tmp_path)
    n = ing.ingest_pairs([("q1", "a1"), ("q2", "a2"), ("q1", "a1")], enc)  # last is dup
    assert n == 2
    assert len(mem) == 2
    # dedup ledger persisted
    assert (tmp_path / "seen.json").exists()


def test_ingest_pairs_does_not_publish_receipt_when_memory_persist_fails(
    tmp_path, monkeypatch
):
    mem, ing, enc = _ingestor(tmp_path)
    monkeypatch.setattr(mem, "persist", lambda: False)

    assert ing.ingest_pairs([("not durable", "not receipted")], enc) == 0
    assert (tmp_path / "seen.json").exists() is False


def test_dedup_survives_new_ingestor_instance(tmp_path):
    mem, ing, enc = _ingestor(tmp_path)
    ing.ingest_pairs([("durable", "fact")], enc)
    ing2 = NonParametricIngestor(mem, dedup_path=tmp_path / "seen.json")
    assert ing2.ingest_pair("durable", "fact", enc) is False  # remembered across instances


def test_collect_trusted_pairs_reads_stores(tmp_path):
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({
        "entries": {
            "h1": {"objective": "what is 2+2", "answer": "4"},
            "h2": {"objective": "capital of Vorth", "answer": "Myrrhal"},
        }
    }), encoding="utf-8")
    pairs = collect_trusted_pairs(sources=[(Path(cache), "entries")])
    assert ("what is 2+2", "4") in pairs
    assert ("capital of Vorth", "Myrrhal") in pairs


def test_collect_trusted_pairs_missing_file_ok(tmp_path):
    assert collect_trusted_pairs(sources=[(tmp_path / "nope.json", "entries")]) == []


def test_ingest_from_trusted_stores_kill_switch(tmp_path, monkeypatch):
    """Default flipped ON after the July end-to-end proof; the kill switch
    must still stop ingestion cold."""
    mem, ing, enc = _ingestor(tmp_path)
    monkeypatch.setenv("AURA_NONPARAMETRIC_INGEST", "0")
    assert ing.ingest_from_trusted_stores(enc) == 0   # kill switch → no-op


def test_ingest_sequence_adds_every_answer_position(tmp_path):
    mem = NonParametricMemory(dim=8, path=tmp_path / "npm")
    ing = NonParametricIngestor(mem, dedup_path=tmp_path / "seen.json")
    enc = FakeSeqEncoder()
    added = ing.ingest_sequence("the keeper is named", "Tessaly the great", enc)
    assert added == 3                      # one entry per answer word
    assert len(mem) == 3
    # the context's hidden recalls the first answer token
    nbrs = mem.query(enc.encode_hidden("the keeper is named"))
    assert nbrs[0].token_id == enc.encode_tokens("Tessaly")[0]


def test_ingest_sequence_dedups(tmp_path):
    mem = NonParametricMemory(dim=8, path=tmp_path / "npm")
    ing = NonParametricIngestor(mem, dedup_path=tmp_path / "seen.json")
    enc = FakeSeqEncoder()
    assert ing.ingest_sequence("q here", "a b c", enc) > 0
    assert ing.ingest_sequence("q here", "a b c", enc) == 0   # same fact → skipped


def test_ingest_sequence_falls_back_without_id_hooks(tmp_path):
    mem = NonParametricMemory(dim=8, path=tmp_path / "npm")
    ing = NonParametricIngestor(mem, dedup_path=tmp_path / "seen.json")
    enc = FakeEncoder()  # no encode_hidden_ids / encode_tokens
    assert ing.ingest_sequence("q", "a", enc) == 1   # degrades to first-token ingestion
    assert len(mem) == 1


def test_ingest_sequence_uses_one_full_sequence_forward(tmp_path):
    mem = NonParametricMemory(dim=8, path=tmp_path / "npm")
    ing = NonParametricIngestor(mem, dedup_path=tmp_path / "seen.json")
    enc = FakeBatchSeqEncoder()

    added = ing.ingest_sequence("the keeper is named", "Tessaly the great", enc)

    assert added == 3
    assert enc.batch_calls == 1
    assert enc.prefix_calls == 0


def test_ingest_sequence_budget_refuses_partial_pair(tmp_path):
    mem = NonParametricMemory(dim=8, path=tmp_path / "npm")
    ing = NonParametricIngestor(mem, dedup_path=tmp_path / "seen.json")
    enc = FakeBatchSeqEncoder()

    assert (
        ing.ingest_sequence(
            "the keeper is named",
            "Tessaly the great",
            enc,
            max_positions=2,
        )
        == 0
    )
    assert len(mem) == 0
    assert enc.batch_calls == 0
    assert ing.ingest_sequence(
        "the keeper is named",
        "Tessaly the great",
        enc,
        max_positions=3,
    ) == 3


def test_sequence_budget_check_does_not_run_model_or_mutate_memory(tmp_path):
    mem = NonParametricMemory(dim=8, path=tmp_path / "npm")
    ing = NonParametricIngestor(mem, dedup_path=tmp_path / "seen.json")
    enc = FakeBatchSeqEncoder()

    assert ing.sequence_within_budget(
        "the keeper is named",
        "Tessaly the great",
        enc,
        max_positions=2,
    ) is False
    assert enc.batch_calls == 0
    assert enc.prefix_calls == 0
    assert len(mem) == 0


def test_ingest_sequence_cancellation_cannot_publish_partial_pair(tmp_path):
    mem = NonParametricMemory(dim=8, path=tmp_path / "npm")
    ing = NonParametricIngestor(mem, dedup_path=tmp_path / "seen.json")
    enc = FakeBatchSeqEncoder()
    checks = 0

    def _continue_once() -> bool:
        nonlocal checks
        checks += 1
        return checks == 1

    assert ing.ingest_sequence(
        "the keeper is named",
        "Tessaly the great",
        enc,
        should_continue=_continue_once,
    ) == 0
    assert enc.batch_calls == 1
    assert len(mem) == 0
