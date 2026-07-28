"""Persisted immune state is a repair-capable, behaviour-evolving artifact.

CP126 5c214831: it was loaded with no schema version, no digest, and no size
limit — cells, behavioural rules, lineage fitness and expansion axes parsed
straight into a live population that can restart components and revoke tools.
"""
from __future__ import annotations

import json

import pytest

from core.adaptation import adaptive_immunity as mod
from core.adaptation.adaptive_immunity import (
    AdaptiveImmuneConfig,
    AdaptiveImmuneSystem,
)


@pytest.fixture()
def seeded(tmp_path):
    cfg = AdaptiveImmuneConfig(population_size=2, max_population=4)
    system = AdaptiveImmuneSystem(config=cfg, state_dir=tmp_path, rng_seed=3)
    system._save_state(force=True)
    return system, tmp_path


def _state_file(tmp_path):
    files = list(tmp_path.rglob("*.json"))
    assert files, "no state file written"
    return files[0]


def test_saved_state_declares_its_schema_and_digest(seeded):
    _system, tmp_path = seeded
    payload = json.loads(_state_file(tmp_path).read_text())

    assert payload["schema_version"] == mod.IMMUNE_STATE_SCHEMA_VERSION
    assert payload["integrity"]["digest"]
    assert payload["integrity"]["algorithm"] == "sha256-unkeyed"


def test_a_round_trip_loads(seeded):
    _system, tmp_path = seeded
    cfg = AdaptiveImmuneConfig(population_size=2, max_population=4)

    reloaded = AdaptiveImmuneSystem(config=cfg, state_dir=tmp_path, rng_seed=3)

    assert reloaded._cells


def test_a_foreign_schema_quarantines_to_a_reseed(seeded):
    _system, tmp_path = seeded
    path = _state_file(tmp_path)
    payload = json.loads(path.read_text())
    payload["schema_version"] = 999
    path.write_text(json.dumps(payload))

    cfg = AdaptiveImmuneConfig(population_size=2, max_population=4)
    reloaded = AdaptiveImmuneSystem(config=cfg, state_dir=tmp_path, rng_seed=3)

    # Reseeded rather than parsed field-by-field into a live population.
    assert reloaded._cells


def test_tampered_state_is_rejected(seeded):
    _system, tmp_path = seeded
    path = _state_file(tmp_path)
    payload = json.loads(path.read_text())
    payload["lineage_stats"] = {"injected": {"successes": 9999, "failures": 0}}
    path.write_text(json.dumps(payload))

    cfg = AdaptiveImmuneConfig(population_size=2, max_population=4)
    reloaded = AdaptiveImmuneSystem(config=cfg, state_dir=tmp_path, rng_seed=3)

    assert "injected" not in reloaded._lineage_stats


def test_an_oversized_state_file_is_refused(seeded, monkeypatch):
    _system, tmp_path = seeded
    monkeypatch.setattr(mod, "MAX_IMMUNE_STATE_BYTES", 10)

    cfg = AdaptiveImmuneConfig(population_size=2, max_population=4)
    reloaded = AdaptiveImmuneSystem(config=cfg, state_dir=tmp_path, rng_seed=3)

    assert reloaded._cells      # reseeded, did not parse


def test_malformed_json_does_not_abort_construction(seeded):
    _system, tmp_path = seeded
    _state_file(tmp_path).write_text("{not json")

    cfg = AdaptiveImmuneConfig(population_size=2, max_population=4)
    reloaded = AdaptiveImmuneSystem(config=cfg, state_dir=tmp_path, rng_seed=3)

    assert reloaded._cells


def test_a_non_object_payload_is_refused(seeded):
    _system, tmp_path = seeded
    _state_file(tmp_path).write_text("[1, 2, 3]")

    cfg = AdaptiveImmuneConfig(population_size=2, max_population=4)
    reloaded = AdaptiveImmuneSystem(config=cfg, state_dir=tmp_path, rng_seed=3)

    assert reloaded._cells


def test_the_digest_is_labelled_as_corruption_detection_not_trust():
    """An unkeyed digest cannot resist someone who can write the file, and
    the state says so rather than implying a trust root."""
    import inspect

    source = inspect.getsource(mod.AdaptiveImmuneSystem._save_state)
    assert "unkeyed" in source
    assert "not tampering" in source or "not as a trust root" in source
