"""A gate whose own config does not parse is a gate that is not running.

A rebase left conflict markers inside
``config/aura_effect_ownership_baseline.json``, and that file was committed
and pushed. The governance gate then failed with

    configuration error: effect ownership baseline is unreadable

which is not a governance verdict. It is the gate declining to run, and for
as long as it stayed that way nothing was being checked — while `make
governance-lint` was still in the pipeline looking like coverage.

Two agents share this checkout, and both regenerate these baselines, so this
collision is structural rather than a one-off mistake. This test is the
cheapest possible thing that turns "the gate silently stopped working" into
a failing test.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"

_CONFLICT_MARKERS = ("<<<<<<<", ">>>>>>>", "=======")


def _config_json_files() -> list[Path]:
    if not CONFIG.is_dir():
        return []
    return sorted(path for path in CONFIG.glob("*.json") if path.is_file())


def test_there_are_config_files_to_check():
    """A glob that matches nothing would make every test below vacuous."""
    assert _config_json_files(), "no config JSON found; this suite proves nothing"


@pytest.mark.parametrize(
    "path", _config_json_files(), ids=lambda path: path.name
)
def test_every_gate_config_parses(path: Path):
    """If it does not parse, the gate reading it is not enforcing anything."""
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        pytest.fail(
            f"{path.name} is not valid JSON ({exc}). Any gate that reads it is "
            "reporting a configuration error rather than a verdict, and is "
            "therefore checking nothing."
        )


@pytest.mark.parametrize(
    "path", _config_json_files(), ids=lambda path: path.name
)
def test_no_config_carries_conflict_markers(path: Path):
    """The specific way it broke, pinned.

    A marker can survive in a file that still parses — inside a string, or in
    a section the parser reaches after the damage — so this is checked
    separately from parseability rather than assumed to be covered by it.
    """
    body = path.read_text(encoding="utf-8", errors="replace")
    found = [marker for marker in _CONFLICT_MARKERS if f"\n{marker}" in body]

    assert not found, (
        f"{path.name} contains merge conflict markers {found}. Two agents "
        "share this checkout and both regenerate these baselines; resolve by "
        "REGENERATING the baseline, not by picking a side, or the recorded "
        "inventory hash will not match the tree it claims to describe."
    )


def test_the_effect_ownership_baseline_has_its_expected_shape():
    """Parseable is not the same as usable.

    A file can be valid JSON and still be the wrong document — an empty
    object parses cleanly and would silently reset the ratchet to zero,
    which reads as "no debt" rather than "no baseline".
    """
    path = CONFIG / "aura_effect_ownership_baseline.json"
    if not path.exists():
        pytest.skip("baseline not present in this checkout")

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert isinstance(payload.get("buckets"), list)
    assert payload["buckets"], "an empty ratchet is not a ratchet"
    assert isinstance(payload.get("inventory_sha256"), str)
    assert len(payload["inventory_sha256"]) == 64, (
        "the inventory hash is missing or truncated, so the baseline cannot "
        "be shown to describe the current tree"
    )
