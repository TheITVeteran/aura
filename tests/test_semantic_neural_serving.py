from __future__ import annotations

import copy
import hashlib
import json

from core.brain.llm.semantic_neural_serving import (
    DEFAULT_ACTIVATION_PATH,
    semantic_neural_activation_errors,
    semantic_neural_serving_status,
)


def _activation():
    return json.loads(DEFAULT_ACTIVATION_PATH.read_text(encoding="utf-8"))


def test_materialized_semantic_activation_reopens_source_and_evidence():
    activation = _activation()
    assert semantic_neural_activation_errors(
        activation,
        verify_live_identity=False,
    ) == []


def test_semantic_activation_rejects_resealed_source_or_evidence_drift():
    activation = _activation()
    tampered = copy.deepcopy(activation)
    relative = next(iter(tampered["source_sha256s"]))
    tampered["source_sha256s"][relative] = "0" * 64
    assert "activation_sha256" in semantic_neural_activation_errors(
        tampered,
        verify_live_identity=False,
    )

    tampered = copy.deepcopy(activation)
    tampered["evidence"]["result_path"] = "../outside.json"
    body = {key: value for key, value in tampered.items() if key != "activation_sha256"}
    tampered["activation_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    assert "evidence_invalid" in semantic_neural_activation_errors(
        tampered,
        verify_live_identity=False,
    )


def test_semantic_serving_kill_switch_is_fail_closed(monkeypatch):
    activation = _activation()
    model_path = activation["model_identity"]["path"]
    monkeypatch.setenv("AURA_SEMANTIC_NEURAL_SERVING", "0")
    status = semantic_neural_serving_status(model_path)
    assert status == {
        "active": False,
        "reason": "semantic_neural_serving_disabled",
    }
