from pathlib import Path

from utils.bundler import MAX_SOURCE_FILE_BYTES, iter_source_files


def _relative_files(root: Path) -> set[str]:
    return {str(path.relative_to(root)) for path in iter_source_files(root, lite=True)}


def test_bundler_preserves_legitimate_audit_and_bundle_source(tmp_path):
    files = {
        "core/audit_engine.py": "AUDIT = True\n",
        "tests/test_output_contract.py": "def test_contract():\n    assert True\n",
        "utils/bundler.py": "def bundle():\n    return True\n",
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    assert _relative_files(tmp_path) == set(files)


def test_bundler_excludes_generated_source_bundles_and_oversized_files(tmp_path):
    generated = tmp_path / "aura_source_part_1.txt"
    generated.write_text("recursive output", encoding="utf-8")
    oversized = tmp_path / "core" / "oversized.py"
    oversized.parent.mkdir(parents=True, exist_ok=True)
    oversized.write_bytes(b"x" * (MAX_SOURCE_FILE_BYTES + 1))
    included = tmp_path / "core" / "normal.py"
    included.write_text("VALUE = 1\n", encoding="utf-8")

    assert _relative_files(tmp_path) == {"core/normal.py"}
