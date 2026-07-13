from dataclasses import dataclass

import pytest

from core.resilience import stem_cell
from core.resilience.stem_cell import StemCellRegistry


@dataclass
class _Snapshot:
    name: str
    version: int


@pytest.fixture
def isolated_stem_cells(monkeypatch, tmp_path):
    monkeypatch.setattr(stem_cell, "_STEM_DIR", tmp_path)
    monkeypatch.setattr(stem_cell, "_STEM_KEY_FILE", tmp_path / "stem_cells.key")
    return tmp_path


def test_stem_cell_defaults_to_signed_json_snapshot(isolated_stem_cells):
    registry = StemCellRegistry()
    registry.capture("identity", _Snapshot(name="aura", version=2), schema_version="v1")

    restored = registry.revert("identity", schema_version="v1")

    assert restored == {"name": "aura", "version": 2}
    snapshot_path = next(isolated_stem_cells.glob("identity_v1_*.signed"))
    assert snapshot_path.read_bytes()[4:5] == b"{"


def test_stem_cell_rejects_pathlike_identifiers(isolated_stem_cells):
    registry = StemCellRegistry()

    with pytest.raises(ValueError):
        registry.capture("../identity", {"ok": True})

    with pytest.raises(ValueError):
        registry.latest("../identity")


def test_stem_cell_refuses_tampered_payload(isolated_stem_cells):
    registry = StemCellRegistry()
    registry.capture("will", {"policy": "steady"}, schema_version="v1")
    path = next(isolated_stem_cells.glob("will_v1_*.signed"))
    raw = path.read_bytes()
    path.write_bytes(raw + b"tamper")

    assert registry.latest("will") is None


def test_stem_cell_writes_survive_production_governance(isolated_stem_cells, monkeypatch):
    """The signing-key write is internal maintenance and must carry its own
    governed scope. Without one, a production-governance runtime refuses the
    key material and every stem-cell signature dies with it — found live as
    an in-chunk order-dependence failure when a preceding test left
    production mode enabled."""
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "production")
    registry = StemCellRegistry()
    registry.capture("identity", _Snapshot(name="aura", version=3), schema_version="v1")
    assert registry.revert("identity", schema_version="v1") == {"name": "aura", "version": 3}
