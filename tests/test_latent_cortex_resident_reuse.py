from core.brain.latent_cortex_service import _resident_state_reusable


def _receipt() -> dict:
    return {
        "episode_id": "episode-1",
        "input_tokens_sha256": "a" * 64,
        "params_unchanged": True,
        "worker_identity": {"worker_pid": 41},
        "runtime_integrity": {"schema": "test"},
        "fast_weights_applied": True,
        "fast_weights_attach_attempted": True,
        "checkpoint_fingerprint": "b" * 64,
        "checkpoint_fingerprint_method": "sha256",
        "checkpoint_file_count": 3,
    }


def test_resident_reuse_requires_worker_integrity(monkeypatch):
    observed = {}

    def _safe(value, **kwargs):
        observed["value"] = value
        observed.update(kwargs)
        return True

    monkeypatch.setattr(
        "core.brain.llm.latent_cortex.runtime_integrity.runtime_integrity_safe",
        _safe,
    )

    assert _resident_state_reusable(_receipt()) is True
    assert observed["require_worker"] is True
    assert observed["expected_episode_id"] == "episode-1"
    assert observed["expected_fast_weights_applied"] is True
    assert observed["expected_fast_weights_attach_attempted"] is True


def test_resident_reuse_rejects_changed_parameters_before_integrity_check(monkeypatch):
    called = False

    def _safe(*_args, **_kwargs):
        nonlocal called
        called = True
        return True

    monkeypatch.setattr(
        "core.brain.llm.latent_cortex.runtime_integrity.runtime_integrity_safe",
        _safe,
    )
    receipt = _receipt()
    receipt["params_unchanged"] = False

    assert _resident_state_reusable(receipt) is False
    assert called is False
