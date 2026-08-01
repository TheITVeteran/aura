"""Content fetcher: a cache that never expired and eviction that lied."""
from __future__ import annotations

import time

import pytest

from core.autonomy.content_fetcher import ContentFetcher

pytestmark = pytest.mark.unit


def _fetcher(tmp_path, **kwargs):
    return ContentFetcher(cache_dir=tmp_path / "cache", **kwargs)


def _put(fetcher, key="k1", *, stored_at=None, cache_path=None):
    fetcher._index[key] = {
        "method": "http", "target": "https://example.test",
        "text": "cached body", "transcript": "", "metadata": {},
        "cache_path": str(cache_path) if cache_path else None,
        "bytes_fetched": 11, "sources": [],
        "stored_at": time.time() if stored_at is None else stored_at,
    }
    return key


# ── freshness ──────────────────────────────────────────────────────────────


def test_fresh_entry_is_served(tmp_path):
    fetcher = _fetcher(tmp_path)
    key = _put(fetcher)

    assert fetcher._get_cached(key) is not None


def test_expired_entry_is_refetched(tmp_path):
    """stored_at was written and never read, so a page fetched once was served
    as a successful fetch forever — Aura could reason about a source that had
    since changed or been retracted and never notice."""
    fetcher = _fetcher(tmp_path, cache_ttl_seconds=60)
    key = _put(fetcher, stored_at=time.time() - 3600)

    assert fetcher._get_cached(key) is None
    assert key not in fetcher._index, "the stale record should be dropped"


def test_missing_stored_at_is_treated_as_stale(tmp_path):
    fetcher = _fetcher(tmp_path)
    key = _put(fetcher)
    del fetcher._index[key]["stored_at"]

    assert fetcher._get_cached(key) is None


def test_future_stored_at_is_treated_as_stale(tmp_path):
    """A clock skew forward must not grant an entry indefinite life."""
    fetcher = _fetcher(tmp_path)
    key = _put(fetcher, stored_at=time.time() + 10_000)

    assert fetcher._get_cached(key) is None


def test_ttl_has_a_floor(tmp_path):
    fetcher = _fetcher(tmp_path, cache_ttl_seconds=0)

    assert fetcher._cache_ttl_seconds >= 60.0


# ── consistency with what is actually on disk ──────────────────────────────


def test_entry_whose_files_are_gone_is_refetched(tmp_path):
    """Budget eviction removes cache DIRECTORIES without touching index
    records, so reads returned success with a cache_path pointing at nothing."""
    fetcher = _fetcher(tmp_path)
    key = _put(fetcher, cache_path=tmp_path / "cache" / "deleted-dir")

    assert fetcher._get_cached(key) is None
    assert key not in fetcher._index


def test_entry_whose_files_exist_is_served(tmp_path):
    fetcher = _fetcher(tmp_path)
    real = tmp_path / "cache" / "kept"
    real.mkdir(parents=True)
    key = _put(fetcher, cache_path=real)

    assert fetcher._get_cached(key) is not None


def test_eviction_prunes_the_index(tmp_path):
    """The core of the consistency bug: directories deleted, records left."""
    fetcher = _fetcher(tmp_path, total_cache_bytes=1)
    victim = tmp_path / "cache" / "victim"
    victim.mkdir(parents=True)
    (victim / "blob.bin").write_bytes(b"x" * 4096)
    key = _put(fetcher, cache_path=victim)

    fetcher._enforce_cache_budget()

    assert not victim.exists()
    assert key not in fetcher._index, "eviction must remove the record too"
