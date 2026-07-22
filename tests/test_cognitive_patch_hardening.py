"""CP126 hardening contracts for core/brain/cognitive_patch.py.

Covers removal of the eval-gaming magic string, fence-preserving extraction,
newline/secret sanitization of proposal metadata, the custody manifest, uuid
filenames, broadened error handling, and the honest not-applied status.
The brain and the file-write gateway are faked — nothing is executed.
"""
from __future__ import annotations

import pytest

import core.runtime.file_write_gateway as fwg
import core.utils.paths as paths_mod
from core.brain.cognitive_patch import CognitivePatchStrategy


class _Thought:
    def __init__(self, content):
        self.content = content


class _FakeBrain:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    async def think(self, prompt):
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return _Thought(self.response)


class _FakeGateway:
    def __init__(self, fail=False):
        self.writes: list[tuple[str, str]] = []
        self.fail = fail

    async def write_text_async(self, path, text, source="unknown"):
        if self.fail:
            raise OSError("disk full")
        self.writes.append((str(path), text))


@pytest.fixture
def harness(tmp_path, monkeypatch):
    gw = _FakeGateway()
    monkeypatch.setattr(fwg, "get_file_write_gateway", lambda: gw)
    monkeypatch.setattr(paths_mod, "DATA_DIR", tmp_path)

    def _make(response):
        strat = CognitivePatchStrategy.__new__(CognitivePatchStrategy)
        strat.brain = _FakeBrain(response)
        return strat, gw

    return _make


# ── 963abd63: match requires a real failure ────────────────────────────────


def test_match_requires_nonempty_reason():
    s = CognitivePatchStrategy.__new__(CognitivePatchStrategy)
    assert s.match("something broke") is True
    assert s.match("") is False
    assert s.match("   ") is False


# ── 43fab1bc: the eval-gaming magic string is gone ─────────────────────────


@pytest.mark.asyncio
async def test_magic_failure_string_no_longer_shortcuts(harness):
    strat, gw = harness("echo real-brain-fix")
    await strat.apply("test_failure_code_123", goal="g")
    assert strat.brain.calls == 1  # the brain WAS consulted, not bypassed
    _path, text = gw.writes[-1]
    assert "real-brain-fix" in text
    assert "Cognitive Fix Applied" not in text  # the old fixture command


# ── 8f35ade1: fenced code is extracted, not deleted ────────────────────────


@pytest.mark.asyncio
async def test_fenced_code_is_preserved(harness):
    strat, gw = harness("```bash\necho hello\n```")
    await strat.apply("boom", goal="g")
    assert gw.writes, "a proposal should have been saved"
    assert "echo hello" in gw.writes[-1][1]


# ── df48e5d5: saving a proposal is NOT reported as applied ──────────────────


@pytest.mark.asyncio
async def test_apply_returns_false_when_only_proposing(harness):
    strat, gw = harness("echo fix")
    result = await strat.apply("boom", goal="g")
    assert result is False  # a proposal was saved, but nothing was applied
    assert gw.writes  # ...yet the proposal exists for review


# ── df409ed3 + f7a5ee5e: metadata is single-line and redacted ──────────────


@pytest.mark.asyncio
async def test_proposal_metadata_is_sanitized(harness):
    strat, gw = harness("echo fix")
    await strat.apply("line1\nrm -rf /\napi_key=sk-secret", goal="do\nthing")
    manifest = gw.writes[-1][1].split("echo fix")[0]
    assert "\nrm -rf /" not in manifest  # newline injection collapsed
    assert "sk-secret" not in manifest  # secret redacted


# ── cb705768 + 60aedb3c: custody manifest + unique filenames ───────────────


@pytest.mark.asyncio
async def test_manifest_and_unique_filenames(harness):
    strat, gw = harness("echo fix")
    await strat.apply("boom-a", goal="g")
    strat.brain.response = "echo different"  # avoid the loop guard
    await strat.apply("boom-b", goal="g")
    assert gw.writes[0][0] != gw.writes[1][0]  # uuid filenames differ
    text = gw.writes[0][1]
    assert "proposal_id:" in text and "content_sha256:" in text


# ── 5f2a4ca8: a write OSError is caught (bool contract holds) ───────────────


@pytest.mark.asyncio
async def test_write_oserror_is_caught(tmp_path, monkeypatch):
    gw = _FakeGateway(fail=True)
    monkeypatch.setattr(fwg, "get_file_write_gateway", lambda: gw)
    monkeypatch.setattr(paths_mod, "DATA_DIR", tmp_path)
    strat = CognitivePatchStrategy.__new__(CognitivePatchStrategy)
    strat.brain = _FakeBrain("echo fix")
    result = await strat.apply("boom", goal="g")
    assert result is False  # OSError did not escape the bool contract


@pytest.mark.asyncio
async def test_empty_brain_response_aborts(harness):
    strat, gw = harness("   ")
    assert await strat.apply("boom", goal="g") is False
    assert gw.writes == []
