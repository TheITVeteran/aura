from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tools.verify_semantic_neural_decode_canary import (
    REPO_ROOT,
    SOURCE_PATHS,
    _sha,
    verify_canary,
)

ARTIFACT = (
    REPO_ROOT
    / "artifacts/closeout/latent_cortex/cp550_semantic_decode_canary/result.json"
)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_artifact(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    payload["source_commit"] = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload["source_sha256s"] = {
        path: _file_sha(REPO_ROOT / path) for path in SOURCE_PATHS
    }
    payload["model_identity"] = {
        "path": str(model.resolve()),
        "config_sha256": _file_sha(model / "config.json"),
        "weights_index_sha256": _file_sha(model / "model.safetensors.index.json"),
    }
    payload["receipt_sha256"] = _sha(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    artifact = tmp_path / "result.json"
    artifact.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return artifact, model


def test_semantic_decode_verifier_regrades_and_replays_frozen_canary(tmp_path):
    artifact, model = _portable_artifact(tmp_path)
    report = verify_canary(artifact, model_path=model)
    assert report["verified"] is True
    assert report["independent_exact_by_arm"] == {
        "ordinary_base": 0,
        "matched_wire_base": 0,
        "treatment": 27,
        "coefficient_lesion": 9,
        "matched_wrong_state": 0,
    }
    assert report["gain_count"] == 27
    assert report["regression_count"] == 0
    assert report["treatment_state_replay_count"] == 27


def test_semantic_decode_verifier_rejects_resealed_response_tamper(tmp_path):
    artifact, model = _portable_artifact(tmp_path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    row = next(row for row in payload["raw_outputs"] if row["arm"] == "treatment")
    row["response"] = "FINAL_ANSWER: {}"
    payload["receipt_sha256"] = _sha(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    artifact.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="summary disagrees with raw output"):
        verify_canary(artifact, model_path=model)
