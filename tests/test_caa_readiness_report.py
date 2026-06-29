"""Tests for CAA readiness verification from on-disk provenance."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from core.consciousness.affective_steering import AFFECTIVE_DIMENSIONS as RUNTIME_DIMENSIONS
from core.consciousness.caa.readiness_report import scan_vector_files, verify_readiness
from training.extract_steering_vectors import AFFECTIVE_DIMENSIONS, ALL_AFFECTIVE_DIMENSIONS


def _sha256(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _vec(path, *, source, extracted, derived_at=1000.0, model_path=None, model_config_sha256=None):
    np.savez(
        path,
        v=np.zeros(8, dtype=np.float32),
        source=source,
        extracted=extracted,
        derived_at=derived_at,
        requested_layer=11,
        selected_layer=11,
        selection_reason=source,
        model_path=model_path or "",
        model_config_sha256=model_config_sha256 or "",
    )


def _setup(tmp_path, specs):
    vdir = tmp_path / "vectors"
    fdir = tmp_path / "fused-model"
    vdir.mkdir()
    fdir.mkdir()
    for i, (source, extracted) in enumerate(specs):
        _vec(vdir / f"vec_layer{i}.npz", source=source, extracted=extracted)
    (fdir / "active.json").write_text(json.dumps({"active_model_path": "/m/active", "fused_at": 500.0}))
    return vdir, fdir


def test_scan_reads_provenance(tmp_path):
    vdir, _ = _setup(tmp_path, [("runtime_derived_caa", False), ("extracted_contrastive", True)])
    scan = scan_vector_files(vdir)
    assert scan["files"] == 2
    assert scan["extracted"] == 1
    assert scan["runtime_derived"] == 1


def test_runtime_derived_is_bootstrap_below_capacity(tmp_path):
    vdir, fdir = _setup(tmp_path, [("runtime_derived_caa", False)] * 6)
    r = verify_readiness(vectors_dir=vdir, fused_model_dir=fdir)
    assert r["level"] == "bootstrap"
    assert r["below_design_capacity"] is True
    assert r["steering_capacity_pct"] < 100
    assert "NOT extracted" in r["detail"]


def test_all_extracted_is_production_full_capacity(tmp_path):
    vdir, fdir = _setup(tmp_path, [("extracted_contrastive", True)] * 6)
    r = verify_readiness(vectors_dir=vdir, fused_model_dir=fdir)
    assert r["level"] == "production"
    assert r["below_design_capacity"] is False
    assert r["steering_capacity_pct"] == 100.0


def test_mixed_when_some_extracted(tmp_path):
    vdir, fdir = _setup(
        tmp_path,
        [("extracted_contrastive", True)] * 3 + [("runtime_derived_caa", False)] * 3,
    )
    r = verify_readiness(vectors_dir=vdir, fused_model_dir=fdir)
    assert r["level"] == "mixed"
    assert 0.0 < r["extracted_ratio"] < 1.0


def test_no_vectors_is_bootstrap(tmp_path):
    vdir = tmp_path / "empty"
    vdir.mkdir()
    r = verify_readiness(vectors_dir=vdir, fused_model_dir=tmp_path)
    assert r["level"] == "bootstrap"


def test_extractor_defaults_to_live_runtime_dimensions():
    runtime_keys = {spec["key"] for spec in RUNTIME_DIMENSIONS}
    assert set(AFFECTIVE_DIMENSIONS) == runtime_keys
    assert {"valence_positive", "arousal", "curiosity", "frustration", "energy"} <= set(
        AFFECTIVE_DIMENSIONS
    )
    assert "confidence" in ALL_AFFECTIVE_DIMENSIONS
    assert "warmth" in ALL_AFFECTIVE_DIMENSIONS


def test_readiness_uses_runtime_contract_and_ignores_stale_nonruntime_vectors(tmp_path):
    vdir = tmp_path / "vectors"
    fdir = tmp_path / "fused-model"
    active = fdir / "active-model"
    vdir.mkdir()
    active.mkdir(parents=True)
    active_config = active / "config.json"
    active_config.write_text(json.dumps({"num_hidden_layers": 64}), encoding="utf-8")
    active_hash = _sha256(active_config)
    fdir.mkdir(exist_ok=True)
    (fdir / "active.json").write_text(
        json.dumps({"active_model_path": str(active), "fused_at": 500.0}),
        encoding="utf-8",
    )

    for key in {spec["key"] for spec in RUNTIME_DIMENSIONS}:
        for layer in (25, 30, 35):
            _vec(
                vdir / f"{key}_layer{layer}.npz",
                source="extracted_caa",
                extracted=True,
                model_path=str(active),
                model_config_sha256=active_hash,
            )

    # This file is real directory drift from older derivation attempts; it
    # should be surfaced as ignored drift, not reduce production readiness.
    _vec(vdir / "warmth_layer99.npz", source="runtime_derived_caa", extracted=False)

    r = verify_readiness(vectors_dir=vdir, fused_model_dir=fdir)
    assert r["level"] == "production"
    assert r["below_design_capacity"] is False
    assert r["runtime_contract"]["expected_total"] == 15
    assert r["runtime_contract"]["expected_extracted"] == 15
    assert r["runtime_contract"]["ignored_file_count"] == 1


def test_extracted_vectors_from_previous_active_model_do_not_count_as_production(tmp_path):
    vdir = tmp_path / "vectors"
    fdir = tmp_path / "fused-model"
    active = fdir / "active-model"
    previous = fdir / "previous-model"
    vdir.mkdir()
    active.mkdir(parents=True)
    previous.mkdir(parents=True)
    active_config = active / "config.json"
    previous_config = previous / "config.json"
    active_config.write_text(json.dumps({"num_hidden_layers": 64, "revision": "new"}), encoding="utf-8")
    previous_config.write_text(json.dumps({"num_hidden_layers": 64, "revision": "old"}), encoding="utf-8")
    previous_hash = _sha256(previous_config)
    fdir.mkdir(exist_ok=True)
    (fdir / "active.json").write_text(
        json.dumps({"active_model_path": str(active), "fused_at": 600.0}),
        encoding="utf-8",
    )

    for key in {spec["key"] for spec in RUNTIME_DIMENSIONS}:
        for layer in (25, 30, 35):
            _vec(
                vdir / f"{key}_layer{layer}.npz",
                source="extracted_caa",
                extracted=True,
                model_path=str(previous),
                model_config_sha256=previous_hash,
            )

    r = verify_readiness(vectors_dir=vdir, fused_model_dir=fdir)
    assert r["level"] == "bootstrap"
    assert r["below_design_capacity"] is True
    assert r["runtime_contract"]["expected_total"] == 15
    assert r["runtime_contract"]["expected_extracted"] == 0
    assert r["runtime_contract"]["expected_extracted_unbound"] == 15
