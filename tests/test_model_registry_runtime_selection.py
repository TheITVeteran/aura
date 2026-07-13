import pytest

import core.brain.llm.model_registry as model_registry


def test_default_deep_model_prefers_mlx_artifact_for_mlx_backend():
    assert model_registry._default_deep_model_name(backend="mlx") == "Qwen2.5-72B-Instruct-4bit"


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("mlx", "Qwen2.5-72B-Instruct-4bit"),
        ("retired", "Qwen2.5-72B-Instruct-4bit"),
    ],
)
def test_normalize_runtime_model_name_respects_backend(backend, expected):
    assert (
        model_registry.normalize_runtime_model_name(
            "Qwen2.5-72B-Instruct-Q4",
            backend=backend,
        )
        == expected
    )


def test_external_backend_env_is_ignored(monkeypatch):
    monkeypatch.setenv("AURA_LOCAL_BACKEND", "retired")

    assert model_registry.get_local_backend() == "mlx"
    assert model_registry.local_backend_is_mlx() is True


def test_mlx_client_refuses_retired_external_artifact(monkeypatch, tmp_path):
    from core.brain.llm.mlx_client import get_mlx_client

    monkeypatch.setenv("AURA_LOCAL_BACKEND", "mlx")

    gguf_path = tmp_path / "qwen2.5-32b-instruct-q5_k_m.gguf"

    with pytest.raises(RuntimeError, match="external_cortex_disabled"):
        get_mlx_client(model_path=str(gguf_path))


def test_get_model_path_maps_q4_alias_to_existing_mlx_model_dir(monkeypatch, tmp_path):
    model_dir = tmp_path / "models" / "Qwen2.5-72B-Instruct-4bit"
    model_dir.mkdir(parents=True)

    monkeypatch.setattr(model_registry, "BASE_DIR", tmp_path)
    monkeypatch.setattr(model_registry, "LOCAL_BACKEND", "mlx")
    monkeypatch.setitem(
        model_registry.MODEL_PATHS,
        "Qwen2.5-72B-Instruct-4bit",
        model_dir,
    )
    monkeypatch.setitem(
        model_registry.MODEL_PATHS,
        "Qwen2.5-72B-Instruct-Q4",
        tmp_path / "models" / "Qwen2.5-72B-Instruct-Q4",
    )

    resolved = model_registry.get_model_path("Qwen2.5-72B-Instruct-Q4")

    assert resolved == str(model_dir.resolve())


def test_get_model_path_preserves_missing_absolute_paths(monkeypatch, tmp_path):
    missing = tmp_path / "missing-model"
    monkeypatch.setattr(model_registry, "LOCAL_BACKEND", "mlx")

    assert model_registry.get_model_path(str(missing)) == str(missing)


def _reset_lane_audit_cache():
    model_registry._LANE_AUDIT_CACHE.update(key=None, at=0.0, result=None)


def test_audit_lane_assignments_caches_filesystem_work(monkeypatch):
    # The autouse resource_observer fixture zeroes the audit-cache TTL for
    # hermeticity; this test asserts the CACHE, so it pins a real TTL.
    monkeypatch.setenv("AURA_LANE_AUDIT_CACHE_TTL_S", "60")
    _reset_lane_audit_cache()
    calls = {"n": 0}
    real_realpath = model_registry.os.path.realpath

    def _counting_realpath(path, *args, **kwargs):
        calls["n"] += 1
        return real_realpath(path, *args, **kwargs)

    monkeypatch.setattr(model_registry.os.path, "realpath", _counting_realpath)
    try:
        first = model_registry.audit_lane_assignments(force_refresh=True)
        after_first = calls["n"]
        second = model_registry.audit_lane_assignments()

        assert second == first
        assert calls["n"] == after_first, "cached call must not hit the filesystem"
    finally:
        _reset_lane_audit_cache()


def test_audit_lane_assignments_cache_returns_copies(monkeypatch):
    _reset_lane_audit_cache()
    try:
        first = model_registry.audit_lane_assignments(force_refresh=True)
        first["lanes"].clear()
        second = model_registry.audit_lane_assignments()
        assert second["lanes"], "callers mutating a result must not poison the cache"
    finally:
        _reset_lane_audit_cache()


def test_audit_lane_assignments_invalidates_on_assignment_change(monkeypatch):
    _reset_lane_audit_cache()
    try:
        model_registry.audit_lane_assignments(force_refresh=True)
        monkeypatch.setattr(model_registry, "ACTIVE_MODEL", "totally-new-model")
        refreshed = model_registry.audit_lane_assignments()
        assert (
            refreshed["lanes"][model_registry.PRIMARY_ENDPOINT]["model"]
            == "totally-new-model"
        )
    finally:
        _reset_lane_audit_cache()


def test_audit_lane_assignments_ttl_zero_disables_cache(monkeypatch):
    _reset_lane_audit_cache()
    monkeypatch.setenv("AURA_LANE_AUDIT_CACHE_TTL_S", "0")
    calls = {"n": 0}
    real_uncached = model_registry._audit_lane_assignments_uncached

    def _counting_uncached():
        calls["n"] += 1
        return real_uncached()

    monkeypatch.setattr(
        model_registry, "_audit_lane_assignments_uncached", _counting_uncached
    )
    try:
        model_registry.audit_lane_assignments()
        model_registry.audit_lane_assignments()
        assert calls["n"] == 2
    finally:
        _reset_lane_audit_cache()
