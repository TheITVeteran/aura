from pathlib import Path

from tools import run_test_chunks
from tools.run_test_chunks import parse_failed_node_ids


def test_chunk_runner_ignores_empty_failure_ids():
    output = """
FAILED
ERROR
FAILED tests/test_example.py::test_real_failure - AssertionError
==== short test summary info ====
"""

    assert parse_failed_node_ids(output) == ["tests/test_example.py::test_real_failure"]


def test_chunk_runner_stops_after_first_failed_chunk_by_default(monkeypatch, tmp_path):
    files = [tmp_path / "test_a.py", tmp_path / "test_b.py"]
    calls: list[int] = []

    monkeypatch.setattr(run_test_chunks, "discover_test_files", lambda _tests_dir: files)
    monkeypatch.setattr(
        run_test_chunks,
        "split_chunks",
        lambda _files, _chunks: [[Path("test_a.py")], [Path("test_b.py")]],
    )

    def fake_run_chunk(index, *_args, **_kwargs):
        calls.append(index)
        return False, f"chunk {index} failed", []

    monkeypatch.setattr(run_test_chunks, "run_chunk", fake_run_chunk)

    assert run_test_chunks.main(["--tests-dir", str(tmp_path)]) == 1
    assert calls == [1]


def test_chunk_runner_can_continue_after_failed_chunk(monkeypatch, tmp_path):
    files = [tmp_path / "test_a.py", tmp_path / "test_b.py"]
    calls: list[int] = []

    monkeypatch.setattr(run_test_chunks, "discover_test_files", lambda _tests_dir: files)
    monkeypatch.setattr(
        run_test_chunks,
        "split_chunks",
        lambda _files, _chunks: [[Path("test_a.py")], [Path("test_b.py")]],
    )

    def fake_run_chunk(index, *_args, **_kwargs):
        calls.append(index)
        return False, f"chunk {index} failed", []

    monkeypatch.setattr(run_test_chunks, "run_chunk", fake_run_chunk)

    assert (
        run_test_chunks.main(["--tests-dir", str(tmp_path), "--continue-on-failure"])
        == 1
    )
    assert calls == [1, 2]
