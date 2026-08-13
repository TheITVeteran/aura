from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools.python_source_closure import (
    PythonSourceClosureError,
    local_python_source_sha256s,
)


def _write(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_source_closure_binds_transitive_and_relative_local_imports(tmp_path: Path) -> None:
    _write(tmp_path, "app/main.py", "from app import service\n")
    _write(tmp_path, "app/__init__.py", "")
    _write(tmp_path, "app/service.py", "from .helpers import answer\n")
    _write(tmp_path, "app/helpers.py", "answer = 42\n")

    measured = local_python_source_sha256s(tmp_path, ["app/main.py"])

    assert set(measured) == {
        "app/__init__.py",
        "app/helpers.py",
        "app/main.py",
        "app/service.py",
    }
    assert measured["app/helpers.py"] == hashlib.sha256(b"answer = 42\n").hexdigest()


def test_source_closure_excludes_external_imports(tmp_path: Path) -> None:
    _write(tmp_path, "entry.py", "import json\nimport package_that_is_not_local\n")

    assert set(local_python_source_sha256s(tmp_path, ["entry.py"])) == {"entry.py"}


def test_source_closure_refuses_missing_or_invalid_entries(tmp_path: Path) -> None:
    with pytest.raises(PythonSourceClosureError, match="entry is unavailable"):
        local_python_source_sha256s(tmp_path, ["missing.py"])

    _write(tmp_path, "broken.py", "this is not python !\n")
    with pytest.raises(PythonSourceClosureError, match="cannot parse"):
        local_python_source_sha256s(tmp_path, ["broken.py"])


def test_source_closure_cache_rebinds_changed_dependency_graph(tmp_path: Path) -> None:
    _write(tmp_path, "entry.py", "import first\n")
    _write(tmp_path, "first.py", "value = 1\n")
    _write(tmp_path, "second.py", "value = 2\n")
    assert set(local_python_source_sha256s(tmp_path, ["entry.py"])) == {
        "entry.py",
        "first.py",
    }

    _write(tmp_path, "entry.py", "import second\n")
    assert set(local_python_source_sha256s(tmp_path, ["entry.py"])) == {
        "entry.py",
        "second.py",
    }
