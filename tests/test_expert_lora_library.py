"""Tests for the expert-LoRA library (capacity loophole: disk-resident specialists)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from core.brain.expert_lora_library import ExpertLoRALibrary, LoRAAdapter


class _RecordingApplier:
    def __init__(self):
        self.loaded: list[str] = []
        self.unloaded: list[str] = []

    def load(self, adapter):
        self.loaded.append(adapter.name)
        return True

    def unload(self, adapter):
        self.unloaded.append(adapter.name)
        return True


# CP126 70c50967: registration now requires an artifact that actually exists
# and looks like an adapter, so fixtures must materialize one instead of
# naming a path that was never on disk.
_ARTIFACT_ROOT: Path | None = None


@pytest.fixture
def lib(tmp_path):
    global _ARTIFACT_ROOT
    _ARTIFACT_ROOT = tmp_path / "adapters"
    _ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    return ExpertLoRALibrary(tmp_path / "lib.json", max_resident=2, applier=_RecordingApplier())


def _make_artifact(name: str) -> str:
    root = _ARTIFACT_ROOT if _ARTIFACT_ROOT is not None else Path(tempfile.mkdtemp())
    adapter_dir = Path(root) / name
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    return str(adapter_dir)


def _adapter(name, task_types, keywords, quality=0.5, base_model="Qwen2.5-32B", size_mb=1.0):
    return LoRAAdapter(
        name=name, path=_make_artifact(name), base_model=base_model,
        task_types=set(task_types), keywords=set(keywords), quality=quality,
        size_mb=size_mb,
    )


def test_register_and_list(lib):
    assert lib.register(_adapter("math_pro", ["math"], ["algebra", "calculus"]))
    assert len(lib.list()) == 1
    assert lib.get("math_pro").quality == 0.5


def test_register_rejects_incomplete(lib):
    assert lib.register(LoRAAdapter(name="", path="/x")) is False


def test_select_by_task_type_and_keywords(lib):
    lib.register(_adapter("math_pro", ["math"], ["integral", "derivative"], quality=0.6))
    lib.register(_adapter("code_pro", ["code"], ["python", "function"], quality=0.6))
    sel = lib.select_for("compute the derivative of x^2", "math")
    assert sel is not None and sel.name == "math_pro"
    sel2 = lib.select_for("write a python function", "code")
    assert sel2.name == "code_pro"


def test_select_respects_base_model(lib):
    lib.register(_adapter("wrong_base", ["math"], ["x"], base_model="Llama-3-8B"))
    assert lib.select_for("anything math", "math", base_model="Qwen2.5-32B") is None


def test_select_quality_breaks_ties(lib):
    lib.register(_adapter("low", ["math"], ["x"], quality=0.3))
    lib.register(_adapter("high", ["math"], ["x"], quality=0.9))
    sel = lib.select_for("solve x", "math")
    assert sel.name == "high"


def test_select_returns_none_when_no_match(lib):
    lib.register(_adapter("math_pro", ["math"], ["x"]))
    assert lib.select_for("audit the repo", "repo_audit") is None


def test_activate_loads_and_tracks_residency(lib):
    lib.register(_adapter("a", ["math"], ["x"]))
    assert lib.activate("a") is True
    assert "a" in lib.resident()
    assert "a" in lib._applier.loaded


def test_residency_lru_eviction(lib):
    for n in ("a", "b", "c"):
        lib.register(_adapter(n, ["math"], [n]))
    lib.activate("a")
    lib.activate("b")
    lib.activate("c")  # exceeds max_resident=2 -> evicts LRU "a"
    resident = lib.resident()
    assert len(resident) == 2
    assert "a" not in resident
    assert "a" in lib._applier.unloaded


def test_select_and_activate_off_by_default(lib, monkeypatch):
    monkeypatch.delenv("AURA_EXPERT_LORA_LIBRARY", raising=False)
    lib.register(_adapter("math_pro", ["math"], ["x"]))
    assert lib.select_and_activate("solve x", "math") is None  # disabled => no-op


def test_select_and_activate_when_enabled(lib, monkeypatch):
    monkeypatch.setenv("AURA_EXPERT_LORA_LIBRARY", "1")
    lib.register(_adapter("math_pro", ["math"], ["derivative"]))
    sel = lib.select_and_activate("compute the derivative", "math")
    assert sel is not None and sel.name == "math_pro"
    assert "math_pro" in lib.resident()


def test_scan_discovers_adapters(tmp_path):
    adir = tmp_path / "adapters" / "math_specialist"
    adir.mkdir(parents=True)
    (adir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adir / "adapters.safetensors").write_bytes(b"\x00" * 1024)
    lib = ExpertLoRALibrary(tmp_path / "lib.json")
    found = lib.scan(tmp_path / "adapters", base_model="Qwen2.5-32B")
    assert found == 1
    assert lib.get("math_specialist") is not None


def test_manifest_persistence(tmp_path):
    path = tmp_path / "lib.json"
    l1 = ExpertLoRALibrary(path)
    l1.register(_adapter("durable", ["code"], ["x"]))
    l2 = ExpertLoRALibrary(path)
    assert l2.get("durable") is not None
