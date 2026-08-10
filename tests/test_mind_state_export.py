"""A backup that silently restored almost nothing, and reported success.

The module advertised "a complete mind state" over eight subsystems. Every one
was fetched behind `hasattr(service, "...")`, and for most the named method
exists nowhere in the codebase — export_snapshot, get_value_weights,
export_goals, get_drive_state, load_from_dict, behavioral_scars,
attachment_history. A guard that is permanently False is not graceful
degradation; it is a silent permanent no-op.

Worse, only 3 of 9 components were integrity-hashed, and verify_integrity
iterated the hashes that existed and returned valid: True — so a tampered
beliefs.json passed the tamper check on a mind backup.

These pin the capability claim to what the code can actually do.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from core.self.mind_state_export import MindStateExporter

pytestmark = pytest.mark.unit


# ── doubles ────────────────────────────────────────────────────────────────


class _Belief:
    def __init__(self, value, confidence=0.9, source="observed"):
        self.value, self.confidence, self.source = value, confidence, source
        self.expired = False


class _WorldState:
    def __init__(self):
        self._beliefs = {"sky": _Belief("blue")}
        self.restored = {}

    def set_belief(self, key, value, confidence=0.7, source="inferred", ttl=1800.0):
        self.restored[key] = (value, confidence, source)


class _Broken:
    """A service registered but missing the API the exporter needs."""


@pytest.fixture
def exporter(monkeypatch):
    exp = MindStateExporter()
    services: dict[str, object] = {}

    monkeypatch.setattr(
        MindStateExporter, "_service", staticmethod(lambda name: services.get(name))
    )
    exp._services = services  # test handle
    return exp


# ── the capability claim is now checkable ──────────────────────────────────


def test_capability_report_names_every_component(exporter):
    report = exporter.capability_report()

    assert len(report["components"]) == 9
    assert "memories" in report["components"]


def test_an_unregistered_service_is_reported_not_hidden(exporter):
    report = exporter.capability_report()

    assert "memories" in report["cannot_export"]
    assert "not registered" in report["cannot_export"]["memories"]


def test_a_registered_service_missing_the_api_says_which_attribute(exporter):
    exporter._services["memory_system"] = _Broken()

    report = exporter.capability_report()

    assert "export_snapshot" in report["cannot_export"]["memories"]


def test_round_trips_lists_only_what_can_go_both_ways(exporter):
    exporter._services["world_state"] = _WorldState()

    report = exporter.capability_report()

    assert "beliefs" in report["round_trips"]
    assert "memories" not in report["round_trips"]


def test_a_component_that_can_export_but_not_restore_is_not_a_round_trip(exporter):
    """canonical_self.to_dict exists; load_from_dict does not."""

    class _OnlyExports:
        def to_dict(self):
            return {"name": "Aura"}

    exporter._services["canonical_self"] = _OnlyExports()

    report = exporter.capability_report()

    assert "canonical_self" in report["can_export"]
    assert "canonical_self" in report["cannot_restore"]
    assert "canonical_self" not in report["round_trips"]


# ── export reports what it could not take ──────────────────────────────────


@pytest.mark.asyncio
async def test_export_names_unavailable_components(exporter, tmp_path):
    exporter._services["world_state"] = _WorldState()

    result = await exporter.export_mind(str(tmp_path / "a.aura-mind"))

    assert result["success"] is True
    assert result["complete"] is False
    assert "memories" in result["unavailable"]


@pytest.mark.asyncio
async def test_a_partial_export_is_distinguishable_from_a_full_one(exporter, tmp_path):
    """Silently omitting components made a 3-component archive look like a
    9-component one."""
    exporter._services["world_state"] = _WorldState()

    result = await exporter.export_mind(str(tmp_path / "a.aura-mind"))

    assert result["components"] == ["beliefs"]
    assert len(result["unavailable"]) == 8


@pytest.mark.asyncio
async def test_the_manifest_records_unavailability(exporter, tmp_path):
    exporter._services["world_state"] = _WorldState()
    path = tmp_path / "a.aura-mind"
    await exporter.export_mind(str(path))

    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json"))

    assert "memories" in manifest["unavailable"]


# ── every component is hashed ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_every_written_component_gets_an_integrity_hash(exporter, tmp_path):
    """Three of nine used to. The other six were unprotected."""
    exporter._services["world_state"] = _WorldState()
    path = tmp_path / "a.aura-mind"
    await exporter.export_mind(str(path))

    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json"))

    assert set(manifest["components"]) == set(manifest["integrity"])


@pytest.mark.asyncio
async def test_verify_integrity_passes_a_clean_archive(exporter, tmp_path):
    exporter._services["world_state"] = _WorldState()
    path = tmp_path / "a.aura-mind"
    await exporter.export_mind(str(path))

    assert (await exporter.verify_integrity(str(path)))["valid"] is True


@pytest.mark.asyncio
async def test_a_tampered_component_fails_verification(exporter, tmp_path):
    exporter._services["world_state"] = _WorldState()
    path = tmp_path / "a.aura-mind"
    await exporter.export_mind(str(path))
    _rewrite_member(path, "beliefs.json", '{"sky": {"value": "green"}}')

    assert (await exporter.verify_integrity(str(path)))["valid"] is False


@pytest.mark.asyncio
async def test_an_unhashed_component_is_unverified_not_valid(exporter, tmp_path):
    """The hole that let a tampered beliefs.json pass: verify iterated only the
    hashes that existed."""
    exporter._services["world_state"] = _WorldState()
    path = tmp_path / "a.aura-mind"
    await exporter.export_mind(str(path))
    _strip_hash(path, "beliefs")

    result = await exporter.verify_integrity(str(path))

    assert result["valid"] is False
    assert "beliefs" in result["unverified"]


@pytest.mark.asyncio
async def test_import_refuses_an_unverifiable_archive(exporter, tmp_path):
    exporter._services["world_state"] = _WorldState()
    path = tmp_path / "a.aura-mind"
    await exporter.export_mind(str(path))
    _strip_hash(path, "beliefs")

    result = await exporter.import_mind(str(path))

    assert result["success"] is False
    assert "unverifiable" in result["error"]


@pytest.mark.asyncio
async def test_import_refuses_a_tampered_archive(exporter, tmp_path):
    exporter._services["world_state"] = _WorldState()
    path = tmp_path / "a.aura-mind"
    await exporter.export_mind(str(path))
    _rewrite_member(path, "beliefs.json", '{"sky": {"value": "green"}}')

    result = await exporter.import_mind(str(path))

    assert result["success"] is False
    assert "Integrity check failed" in result["error"]


# ── import reports what it could not put back ──────────────────────────────


@pytest.mark.asyncio
async def test_beliefs_actually_round_trip(exporter, tmp_path):
    """There was no belief importer at all; beliefs were exported and dropped."""
    source = _WorldState()
    exporter._services["world_state"] = source
    path = tmp_path / "a.aura-mind"
    await exporter.export_mind(str(path))

    target = _WorldState()
    exporter._services["world_state"] = target
    result = await exporter.import_mind(str(path))

    assert "beliefs" in result["imported"]
    assert target.restored["sky"][0] == "blue"


@pytest.mark.asyncio
async def test_a_component_that_cannot_be_restored_is_reported_skipped(
    exporter, tmp_path
):
    """'Restored' must never silently mean 'partially restored'."""
    exporter._services["world_state"] = _WorldState()
    path = tmp_path / "a.aura-mind"
    await exporter.export_mind(str(path))

    exporter._services["world_state"] = _Broken()
    result = await exporter.import_mind(str(path))

    assert result["imported"] == []
    assert "beliefs" in result["skipped"]
    assert result["complete"] is False


@pytest.mark.asyncio
async def test_a_restorer_that_raises_is_recorded_not_swallowed(exporter, tmp_path):
    exporter._services["world_state"] = _WorldState()
    path = tmp_path / "a.aura-mind"
    await exporter.export_mind(str(path))

    class _Explodes:
        def set_belief(self, *a, **k):
            raise RuntimeError("nope")

    exporter._services["world_state"] = _Explodes()
    result = await exporter.import_mind(str(path))

    assert "beliefs" in result["skipped"]
    assert "RuntimeError" in result["skipped"]["beliefs"]


@pytest.mark.asyncio
async def test_a_complete_round_trip_reports_complete(exporter, tmp_path):
    exporter._components = tuple(
        c for c in exporter._components if c.name == "beliefs"
    )
    exporter._services["world_state"] = _WorldState()
    path = tmp_path / "a.aura-mind"

    export = await exporter.export_mind(str(path))
    exporter._services["world_state"] = _WorldState()
    result = await exporter.import_mind(str(path))

    assert export["complete"] is True
    assert result["complete"] is True


@pytest.mark.asyncio
async def test_a_missing_archive_is_an_error_not_a_crash(exporter, tmp_path):
    result = await exporter.import_mind(str(tmp_path / "nope.aura-mind"))

    assert result["success"] is False


# ── security ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sensitive_fields_are_scrubbed_from_the_self_export(exporter, tmp_path):
    class _Self:
        def to_dict(self):
            return {"name": "Aura", "api_key": "sk-secret", "private_key": "x"}

    exporter._services["canonical_self"] = _Self()
    path = tmp_path / "a.aura-mind"
    await exporter.export_mind(str(path))

    with zipfile.ZipFile(path) as zf:
        data = json.loads(zf.read("canonical_self.json"))

    assert data == {"name": "Aura"}


# ── helpers ────────────────────────────────────────────────────────────────


def _members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as zf:
        return {n: zf.read(n) for n in zf.namelist()}


def _write(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in members.items():
            zf.writestr(name, content)


def _rewrite_member(path: Path, name: str, content: str) -> None:
    members = _members(path)
    members[name] = content.encode()
    _write(path, members)


def _strip_hash(path: Path, component: str) -> None:
    members = _members(path)
    manifest = json.loads(members["manifest.json"])
    manifest["integrity"].pop(component, None)
    members["manifest.json"] = json.dumps(manifest).encode()
    _write(path, members)
