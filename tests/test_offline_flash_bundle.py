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


def test_a_programs_other_files_are_fetched_from_its_archive_item_and_kept_with_their_hashes(tmp_path):
    """Offline 2026-10-11 a trap game asked for its levels, which its archive item holds and the bundle did not."""
    from tools.prepare_offline_flash_bundle import fetch_dependencies
    from tools.serve_offline_flash_bundle import check_bundle

    _bundle(tmp_path)
    bundle = tmp_path
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["cases"][0].update(archive_identifier="a-trap-game", catalog_source="2003/trap/trap-2003.swf")
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    listed = [{"name": "game.swf", "size": str(manifest["cases"][0]["program"]["bytes"])},
              {"name": "include/level1.txt", "size": "5"}, {"name": "a-trap-game_meta.xml", "size": "9"},
              {"name": "../escape.txt", "size": "3"}]
    fetched = fetch_dependencies(bundle, list_files=lambda _id: listed, fetch=lambda url: b"lvl 1")
    assert fetched["cases"][0]["dependencies"] == {"include/level1.txt": hashlib.sha256(b"lvl 1").hexdigest()}
    assert (bundle / "games" / "0" / "include" / "level1.txt").read_bytes() == b"lvl 1"
    check_bundle(bundle)
    (bundle / "games" / "0" / "include" / "level1.txt").write_bytes(b"changed")
    with pytest.raises(ValueError):
        check_bundle(bundle)


@pytest.mark.parametrize("size,data", [(5, b"bad"), (2, b"oversized"), (-1, b"bad"), ("unknown", b"bad")])
def test_dependency_extent_failure_remains_unresolved(tmp_path, size, data):
    from tools.prepare_offline_flash_bundle import fetch_dependencies

    manifest = _bundle(tmp_path)
    manifest["cases"][0]["archive_identifier"] = "a-program"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    fetched = fetch_dependencies(tmp_path, list_files=lambda _id: [{"name": "level.txt", "size": size}], fetch=lambda _url: data)
    assert not fetched["cases"][0]["dependencies"]
    assert "level.txt" in fetched["cases"][0]["dependency_errors"]
    assert fetched["dependency_completeness"].startswith("unestablished")
    assert not (tmp_path / "games" / "0" / "level.txt").exists()


def test_a_dependency_cannot_borrow_another_cases_file(tmp_path):
    manifest = _bundle(tmp_path)
    other = tmp_path / "games" / "1"
    other.mkdir(parents=True)
    data = b"different case"
    (other / "level.txt").write_bytes(data)
    manifest["cases"][0]["dependencies"] = {"../1/level.txt": hashlib.sha256(data).hexdigest()}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differs"):
        check_bundle(tmp_path)
