"""Model lifecycle: 'any nonempty directory is a model', and names that escape
the model root."""
from __future__ import annotations

import pytest

from core.brain.llm.model_lifecycle import ModelLifecycleManager, _dir_is_populated

pytestmark = pytest.mark.unit


def _model_dir(root, name="good-model", *, weights=True, config=True,
               extra=None):
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    if weights:
        (path / "model.safetensors").write_bytes(b"\x00" * 16)
    if config:
        (path / "config.json").write_text("{}")
    for filename in extra or []:
        (path / filename).write_text("x")
    return path


# ── a directory with something in it is not a model ────────────────────────


def test_a_complete_model_directory_is_present(tmp_path):
    assert _dir_is_populated(_model_dir(tmp_path)) is True


def test_sharded_weights_one_level_down_still_count(tmp_path):
    path = tmp_path / "sharded"
    (path / "shard-0").mkdir(parents=True)
    (path / "shard-0" / "model-00001.safetensors").write_bytes(b"\x00")
    (path / "config.json").write_text("{}")

    assert _dir_is_populated(path) is True


@pytest.mark.parametrize("case,kwargs", [
    ("weights but no config", {"config": False}),
    ("config but no weights", {"weights": False}),
])
def test_incomplete_model_directories_are_not_present(tmp_path, case, kwargs):
    """Presence required only one directory entry, so a metadata-only or
    weights-only directory satisfied inventory and all_present."""
    assert _dir_is_populated(_model_dir(tmp_path, **kwargs)) is False, case


def test_unrelated_files_do_not_make_a_model(tmp_path):
    path = tmp_path / "junk"
    path.mkdir()
    (path / "README.md").write_text("notes")

    assert _dir_is_populated(path) is False


def test_an_interrupted_download_is_not_a_present_model(tmp_path):
    """A partial transfer left a marker beside real-looking files and was
    reported production-ready."""
    path = _model_dir(tmp_path, "partial", extra=["model.safetensors.incomplete"])

    assert _dir_is_populated(path) is False


def test_a_lock_file_disqualifies_the_directory(tmp_path):
    path = _model_dir(tmp_path, "locked", extra=[".lock"])

    assert _dir_is_populated(path) is False


def test_empty_and_missing_directories_are_not_present(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    assert _dir_is_populated(empty) is False
    assert _dir_is_populated(tmp_path / "nope") is False


# ── names must not escape the governed model root ──────────────────────────


def test_model_names_cannot_traverse_out_of_the_root(tmp_path):
    """Plan names flowed straight into `models_dir / name`, so traversal
    redirected presence checks, directory creation, and download targets
    outside the governed root."""
    manager = ModelLifecycleManager(base_dir=tmp_path)

    for hostile in ("../escape", "../../etc/passwd", "a/../../out"):
        with pytest.raises(ValueError, match="traverse|outside"):
            manager._model_path(hostile)


def test_absolute_model_names_are_rejected(tmp_path):
    manager = ModelLifecycleManager(base_dir=tmp_path)

    with pytest.raises(ValueError, match="relative|outside"):
        manager._model_path("/tmp/somewhere-else")


def test_empty_model_name_is_rejected(tmp_path):
    manager = ModelLifecycleManager(base_dir=tmp_path)

    with pytest.raises(ValueError, match="empty"):
        manager._model_path("   ")


def test_ordinary_names_resolve_inside_the_root(tmp_path):
    manager = ModelLifecycleManager(base_dir=tmp_path)
    root = (tmp_path / "models").resolve()

    resolved = manager._model_path("cortex-32b")

    assert resolved.parent == root
    assert resolved.name == "cortex-32b"


def test_nested_names_are_allowed_inside_the_root(tmp_path):
    manager = ModelLifecycleManager(base_dir=tmp_path)
    root = (tmp_path / "models").resolve()

    resolved = manager._model_path("org/model-v2")

    assert root in resolved.parents
