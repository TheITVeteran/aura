"""CP126 hardening contracts for core/actuators/artifact_builder.py.

Covers path containment of caller stems, CSV formula-injection neutralization,
size bounds, uuid-suffixed default names, partial-set cleanup, write receipts,
program provenance banners, and the open_artifact containment policy.
"""
from __future__ import annotations

import asyncio
import csv
from pathlib import Path

import pytest

from core.actuators import artifact_builder as ab


@pytest.fixture(autouse=True)
def _redirect_output(monkeypatch, tmp_path):
    monkeypatch.setattr(ab, "_output_dir", lambda: tmp_path)
    return tmp_path


# ── c77d5f5c: caller stems cannot escape the artifact dir ──────────────────


def test_traversal_stem_is_contained(tmp_path):
    built = ab.build_table([["x"]], headers=["h"], stem="../../etc/passwd")
    assert built.ok
    for p in built.paths:
        assert Path(p).resolve().parent == tmp_path.resolve()
    assert "/" not in Path(built.primary).stem


def test_stem_with_separators_is_sanitized(tmp_path):
    built = ab.build_program("print(1)", stem="a/b\\c")
    assert built.ok
    assert Path(built.primary).parent == tmp_path.resolve() or Path(built.primary).parent == tmp_path


# ── 6d4878e2: CSV formula injection is neutralized ─────────────────────────


@pytest.mark.parametrize("payload", ["=SUM(A1)", "+1+1", "-2+3", "@cmd", "=1+1"])
def test_csv_formula_cells_are_neutralized(payload):
    cell = ab._csv_cell(payload)
    assert cell.lstrip('"').startswith("'")  # leading apostrophe neutralizes it


def test_benign_csv_cells_unchanged():
    assert ab._csv_cell("hello") == "hello"
    assert ab._csv_cell("a,b") == '"a,b"'  # still quoted, not neutralized


def test_special_characters_still_round_trip(tmp_path):
    built = ab.build_table([['a,b', 'quote"x', "line\nbreak"]], headers=["c1", "c2", "c3"])
    with Path(built.primary).open() as fh:
        rows = list(csv.reader(fh))
    assert rows[1] == ['a,b', 'quote"x', "line\nbreak"]


# ── 9aee5e52: size bounds ──────────────────────────────────────────────────


def test_too_many_rows_refused(monkeypatch):
    monkeypatch.setattr(ab, "_MAX_ROWS", 5)
    built = ab.build_table([["x"]] * 10)
    assert built.ok is False and "many rows" in built.detail


def test_oversize_doc_refused(monkeypatch):
    monkeypatch.setattr(ab, "_MAX_BODY_CHARS", 10)
    built = ab.build_doc("z" * 100)
    assert built.ok is False and "too large" in built.detail


# ── 6a886ca0: default names carry a uuid (no sub-second collisions) ────────


def test_default_stems_are_unique():
    a = ab.build_table([["1"]]).primary
    b = ab.build_table([["1"]]).primary
    assert a != b


# ── f3ad7624 + 29bf4231: partial-set cleanup + write receipts ──────────────


def test_partial_write_failure_rolls_back(monkeypatch, tmp_path):
    calls = {"n": 0}
    real_write = ab._write

    def _flaky(path, text):
        calls["n"] += 1
        if calls["n"] == 2:  # fail the second file (HTML)
            return False
        return real_write(path, text)

    monkeypatch.setattr(ab, "_write", _flaky)
    built = ab.build_table([["x"]], headers=["h"])
    assert built.ok is False
    # The first (CSV) file must have been cleaned up — no half-built set.
    assert list(tmp_path.glob("*.csv")) == []


def test_write_receipt_required_for_ok(monkeypatch):
    monkeypatch.setattr(ab, "_write", lambda path, text: False)
    built = ab.build_program("print(1)")
    assert built.ok is False


# ── f563d767: program provenance banner ────────────────────────────────────


def test_program_carries_provenance_banner():
    built = ab.build_program("print('hi')", language="python")
    text = Path(built.primary).read_text()
    assert "unverified" in text.lower()
    assert "print('hi')" in text  # original source preserved


# ── 833d069f: open_artifact containment ────────────────────────────────────


def test_open_refuses_path_outside_artifact_dir(tmp_path):
    assert asyncio.run(ab.open_artifact("/etc/passwd")) is False


def test_open_refuses_url_like_argument():
    assert asyncio.run(ab.open_artifact("https://evil.example.com")) is False


def test_open_refuses_nonexistent_artifact(tmp_path):
    assert asyncio.run(ab.open_artifact(str(tmp_path / "nope.csv"))) is False
