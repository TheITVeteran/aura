"""CP126 hardening contracts for core/brain/execution.py.

Covers finite timeout/retry validation, the retry cap, monotonic duration,
metadata copy+redaction, success evaluation (error envelopes + predicate),
the unexpected-fault receipt, and the uncertain-timeout marker.
"""
from __future__ import annotations

import math

import pytest

from core.brain.execution import _MAX_RETRIES, ExecutionManager, _looks_successful


class _Trace:
    def __init__(self):
        self.events: list[dict] = []

    def log(self, event):
        self.events.append(event)


def _mgr():
    return ExecutionManager(_Trace())


# ── 98f5776c: finite timeout / retry_delay ─────────────────────────────────


@pytest.mark.asyncio
async def test_nan_timeout_is_rejected():
    with pytest.raises(ValueError):
        await _mgr().execute("a", lambda: 1, timeout_seconds=math.nan)


@pytest.mark.asyncio
async def test_nan_retry_delay_does_not_crash():
    res = await _mgr().execute("a", lambda: "ok", retry_delay=math.nan)
    assert res.ok is True


# ── 4a56d16a: retry cap ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retries_are_capped():
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise RuntimeError("x")

    res = await _mgr().execute("boom", boom, retries=1000, retry_delay=0.0)
    assert res.ok is False
    assert calls["n"] == _MAX_RETRIES  # not 1000


# ── 5e3d665b: monotonic duration is non-negative ───────────────────────────


@pytest.mark.asyncio
async def test_duration_is_non_negative():
    res = await _mgr().execute("a", lambda: "ok")
    assert res.duration >= 0.0


# ── 7a725c8f: metadata is copied and secrets redacted ──────────────────────


@pytest.mark.asyncio
async def test_metadata_is_copied_not_returned_by_reference():
    caller_meta = {"trace": "abc"}
    res = await _mgr().execute("a", lambda: "ok", metadata=caller_meta)
    assert res.metadata is not caller_meta
    assert "operation_id" not in caller_meta  # caller's dict was not mutated


@pytest.mark.asyncio
async def test_metadata_secrets_are_redacted():
    res = await _mgr().execute("a", lambda: "ok", metadata={"api_key": "sk-secret"})
    assert res.metadata["api_key"] == "***redacted***"


@pytest.mark.asyncio
async def test_operation_id_is_present():
    res = await _mgr().execute("a", lambda: "ok")
    assert "operation_id" in res.metadata


# ── aa8b9910: a returned object is not automatically success ────────────────


@pytest.mark.asyncio
async def test_error_envelope_result_is_failure():
    res = await _mgr().execute("a", lambda: {"ok": False, "error": "nope"}, retries=1)
    assert res.ok is False and res.error == "nope"


@pytest.mark.asyncio
async def test_void_result_is_still_success():
    # An ordinary side-effecting action returning None stays successful.
    res = await _mgr().execute("a", lambda: None)
    assert res.ok is True


@pytest.mark.asyncio
async def test_success_predicate_is_honored():
    res = await _mgr().execute("a", lambda: 3, success_predicate=lambda r: r > 5, retries=1)
    assert res.ok is False


def test_looks_successful_defaults():
    assert _looks_successful("ok", None) is True
    assert _looks_successful(None, None) is True
    assert _looks_successful({"ok": False}, None) is False
    assert _looks_successful({"error": "x"}, None) is False


# ── 05973278: unexpected faults become a result receipt, never escape ──────


@pytest.mark.asyncio
async def test_unexpected_exception_becomes_a_result():
    class WeirdError(Exception):
        pass

    def boom():
        raise WeirdError("surprise")

    res = await _mgr().execute("a", boom, retries=1)
    assert res.ok is False and "unexpected" in (res.error or "")


# ── 6e82df46: timeout marks the outcome uncertain ──────────────────────────


@pytest.mark.asyncio
async def test_timeout_marks_uncertain():
    import asyncio

    async def slow():
        await asyncio.sleep(5)

    res = await _mgr().execute("a", slow, timeout_seconds=0.05, retries=1)
    assert res.ok is False
    assert res.metadata.get("outcome") == "uncertain_timeout"
