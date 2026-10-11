"""A missing original or changed emulator cannot become a successful campaign case."""
from __future__ import annotations

import hashlib
import json
import zlib

import pytest

from tools.prepare_offline_flash_bundle import swf_metadata
from tools.serve_offline_flash_bundle import check_bundle


def _program(compressed=False):
    body = b"\x08\x00\x00\x1e\x01\x00\x00\x00"
    header = (b"CWS" if compressed else b"FWS") + b"\x09" + (len(body) + 8).to_bytes(4, "little")
    return header + (zlib.compress(body) if compressed else body)


@pytest.mark.parametrize("compressed", [False, True])
def test_whole_original_bytes_and_timeline_are_retained(compressed):
    data = _program(compressed)
    measured = swf_metadata(data)
    assert measured["sha256"] == hashlib.sha256(data).hexdigest()
    assert measured["frame_rate"] == 30. and measured["frames"] == 1


@pytest.mark.parametrize("data", [b"<html>not a game</html>", _program()[:-1], _program(True)[:-1], _program(True) + b"junk"])
def test_partial_or_foreign_programs_do_not_satisfy_the_asset_contract(data):
    with pytest.raises((ValueError, zlib.error)):
        swf_metadata(data)


def _bundle(tmp_path):
    programs = tmp_path / "programs"
    programs.mkdir()
    data = _program()
    (programs / "0.swf").write_bytes(data)
    manifest = {"schema_version": 1, "emulator": {"files": {}},
                "cases": [{"id": 0, "status": "asset_verified", "program": swf_metadata(data)},
                          {"id": 1, "status": "unresolved", "error": "missing original"}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_missing_cases_remain_in_the_campaign_and_never_acquire_a_pass(tmp_path):
    _bundle(tmp_path)
    cases = check_bundle(tmp_path)["cases"]
    assert len(cases) == 2 and cases[1]["status"] == "unresolved"
    assert not any(case.get("passed") for case in cases)


def test_replaced_original_cannot_run_under_the_old_identity(tmp_path):
    _bundle(tmp_path)
    (tmp_path / "programs" / "0.swf").write_bytes(_program() + b"new")
    with pytest.raises(ValueError, match="differs"):
        check_bundle(tmp_path)


def test_duplicate_case_numbers_are_not_a_complete_campaign(tmp_path):
    manifest = _bundle(tmp_path)
    manifest["cases"][1]["id"] = 0
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate"):
        check_bundle(tmp_path)
