import importlib


def test_optional_startup_validation_uses_spec_discovery_for_native_packages(monkeypatch):
    import core.startup.validator as validator

    imported: list[str] = []
    original_import_module = importlib.import_module

    def guarded_import_module(name: str, *args, **kwargs):
        imported.append(name)
        if name == "mlx_whisper":
            raise AssertionError("optional mlx_whisper check must not execute native import")
        return original_import_module(name, *args, **kwargs)

    def fake_find_spec(name: str):
        if name in {"mlx_whisper", "yaml"}:
            return object()
        return None

    monkeypatch.setattr(importlib, "import_module", guarded_import_module)
    monkeypatch.setattr(validator.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(
        "core.brain.llm.model_registry.get_local_backend",
        lambda: "llama_cpp",
    )
    monkeypatch.setattr(
        "core.brain.llm.model_registry.find_llama_server_bin",
        lambda: "/usr/local/bin/llama-server",
    )

    results = validator.check_optional_packages()
    by_name = {result.name: result for result in results}

    assert by_name["Optional: mlx_whisper"].passed is True
    assert by_name["Optional: yaml"].passed is True
    assert "mlx_whisper" not in imported
