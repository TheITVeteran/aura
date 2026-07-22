"""CP126 hardening contracts for core/actuators/doc_ingest.py.

Document ingestion pulls external bytes into semantic memory, so the rails
matter: allowed-root confinement + sensitive-path denylist + symlink
resolution, size bounds, honest lossy-decode marking, a real HTML parser,
fail-closed PDF/image extraction, importance clamping, untrusted-content
labeling, a single checked write path, and a partial receipt on failure.
"""
from __future__ import annotations

import builtins
import math

import pytest

import core.actuators.doc_ingest as di
from core.actuators.doc_ingest import (
    DocumentIngestActuator,
    _clamp_importance,
    _validate_ingest_path,
)


class _FakeFacade:
    def __init__(self, results=None):
        self.calls: list[tuple[str, dict]] = []
        self._results = results

    def add_memory(self, text, metadata):
        self.calls.append((text, metadata))
        if self._results is None:
            return True
        return self._results[len(self.calls) - 1]


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_DOC_INGEST_ROOTS", str(tmp_path))
    return tmp_path


def _ingest(path, facade, monkeypatch, **extra):
    monkeypatch.setattr(di, "get_runtime_service", lambda name, default=None: facade if name == "memory_facade" else default)
    act = DocumentIngestActuator()
    params = {"action": "ingest", "path": str(path), "_aura_authorized": True}
    params.update(extra)
    return act.execute(params)


# ── 973ba2c8 + 95e3ce23: path confinement, denylist, size ──────────────────


def test_path_outside_roots_is_refused(sandbox, tmp_path):
    outside = tmp_path.parent / "outside.txt"
    ok, _, err = _validate_ingest_path(str(outside))
    assert ok is None and "allowed" in err


def test_symlink_escape_is_refused(sandbox):
    link = sandbox / "link.txt"
    try:
        link.symlink_to("/etc/passwd")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    resolved, _, err = _validate_ingest_path(str(link))
    assert resolved is None  # realpath escapes the allowed root


def test_oversize_file_is_refused(sandbox, monkeypatch):
    monkeypatch.setattr(di, "_MAX_FILE_BYTES", 10)
    big = sandbox / "big.txt"
    big.write_text("x" * 100)
    ok, _, err = _validate_ingest_path(str(big))
    assert ok is None and "limit" in err


def test_empty_file_is_refused(sandbox):
    empty = sandbox / "empty.txt"
    empty.write_text("")
    ok, _, err = _validate_ingest_path(str(empty))
    assert ok is None and "empty" in err


# ── b87896bd: importance is finite and in range ────────────────────────────


@pytest.mark.parametrize("value,expected", [(math.nan, 0.5), (5.0, 1.0), (-1.0, 0.0), ("x", 0.5), (0.3, 0.3)])
def test_importance_is_clamped(value, expected):
    assert _clamp_importance(value) == expected


# ── 65531b6c: real HTML parsing drops script/style, unescapes ──────────────


def test_html_parsing_drops_scripts_and_unescapes():
    html = "<html><head><style>x{}</style></head><body>Hello <script>alert(1)</script>&amp; bye</body></html>"
    text = DocumentIngestActuator._parse_html(html)
    assert "alert" not in text
    assert "x{}" not in text
    assert "Hello" in text and "& bye" in text


# ── 54a18eea: PDF extraction fails closed without pypdf ─────────────────────


def test_pdf_without_library_fails_closed(sandbox, monkeypatch):
    real_import = builtins.__import__

    def _blocked(name, *a, **k):
        if name == "pypdf":
            raise ImportError("blocked for test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(RuntimeError, match="pypdf is required"):
        DocumentIngestActuator._parse_pdf(str(sandbox / "x.pdf"))


# ── authority contract ─────────────────────────────────────────────────────


def test_execute_requires_authorization(sandbox):
    act = DocumentIngestActuator()
    assert act.execute({"action": "ingest", "path": str(sandbox / "a.txt")}).success is False


# ── 9a319ab7 + c6df268b + 7aa3789d + 952240a4: single labeled write ────────


def test_ingest_writes_once_with_untrusted_labels_and_digest(sandbox, monkeypatch):
    doc = sandbox / "note.txt"
    doc.write_text("hello world this is a short document")
    facade = _FakeFacade()
    res = _ingest(doc, facade, monkeypatch, importance=0.9)

    assert res.success is True
    assert len(facade.calls) == 1  # one write path, not two
    text, meta = facade.calls[0]
    assert meta["trust_tier"] == "untrusted_external"
    assert meta["contains_instructions"] is False
    assert meta["file_name"] == "note.txt"
    assert "file_path" not in meta  # absolute local path is not stored
    assert len(meta["file_sha256"]) == 64
    assert meta["importance"] == 0.9
    assert "UNTRUSTED EXTERNAL DOCUMENT" in text
    assert res.updates["file_sha256"] == meta["file_sha256"]


# ── 8be140ad + fd9483b5: count reflects confirmed writes; partial receipt ──


def test_partial_failure_yields_a_resume_cursor(sandbox, monkeypatch):
    doc = sandbox / "big.txt"
    doc.write_text(" ".join(f"word{i}" for i in range(5000)))  # many chunks
    facade = _FakeFacade(results=[True, True, False, True, True] + [True] * 40)
    res = _ingest(doc, facade, monkeypatch)

    assert res.success is False
    assert res.updates["chunks_indexed"] == 2  # only confirmed writes counted
    assert res.updates["resume_from_chunk"] == 2


def test_decode_lossy_is_marked(sandbox, monkeypatch):
    doc = sandbox / "latin.txt"
    doc.write_bytes(b"caf\xe9 not utf8")  # invalid utf-8 byte
    facade = _FakeFacade()
    res = _ingest(doc, facade, monkeypatch)
    assert res.success is True
    assert res.updates["decode_lossy"] is True
    assert facade.calls[0][1]["decode_lossy"] is True
