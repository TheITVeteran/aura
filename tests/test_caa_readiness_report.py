"""Tests for CAA readiness verification from on-disk provenance."""
from __future__ import annotations

import json

import numpy as np

from core.consciousness.caa.readiness_report import scan_vector_files, verify_readiness


def _vec(path, *, source, extracted, derived_at=1000.0):
    np.savez(
        path,
        v=np.zeros(8, dtype=np.float32),
        source=source,
        extracted=extracted,
        derived_at=derived_at,
        requested_layer=11,
        selected_layer=11,
        selection_reason=source,
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
