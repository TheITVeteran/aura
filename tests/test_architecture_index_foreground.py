def test_architecture_index_defers_build_during_foreground_quiet_window(monkeypatch, tmp_path):
    from core.self.architecture_index import ArchitectureIndex

    (tmp_path / "core").mkdir()
    (tmp_path / "interface").mkdir()
    (tmp_path / "core" / "sample.py").write_text(
        '"""sample module"""\nclass Sample:\n    """sample class"""\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AURA_FOREGROUND_ONLY", "1")
    monkeypatch.setenv("AURA_FOREGROUND_ARCHITECTURE_INDEX_QUIET_S", "180")

    index = ArchitectureIndex(project_root=tmp_path)

    assert index.build() == 0
    assert index.query("sample module") == ""
    assert str(index._deferred_reason).startswith("foreground_quiet_window:")


def test_architecture_index_force_build_overrides_foreground_deferral(monkeypatch, tmp_path):
    from core.self.architecture_index import ArchitectureIndex

    (tmp_path / "core").mkdir()
    (tmp_path / "interface").mkdir()
    (tmp_path / "core" / "sample.py").write_text(
        '"""sample module"""\nclass Sample:\n    """sample class"""\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AURA_FOREGROUND_ONLY", "1")

    index = ArchitectureIndex(project_root=tmp_path)

    assert index.build(force=True) == 1
    assert "sample module" in index.query("sample module").lower()


def test_architecture_index_getter_does_not_start_foreground_build(monkeypatch):
    from core.self import architecture_index as module

    monkeypatch.setenv("AURA_FOREGROUND_ONLY", "1")
    monkeypatch.setenv("AURA_FOREGROUND_ARCHITECTURE_INDEX_QUIET_S", "180")
    module._index = None

    try:
        index = module.get_architecture_index()
        assert index._index == {}
        assert str(index._deferred_reason).startswith("foreground_quiet_window:")
    finally:
        module._index = None
