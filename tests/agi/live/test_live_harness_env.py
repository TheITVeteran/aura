import sys

from tests.agi.live.live_harness import LiveAuraHarness


def test_live_harness_strips_pytest_current_test_from_subprocess_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/example.py::test_case (call)")

    harness = LiveAuraHarness(tmp_path)
    result = harness.run_command(
        tmp_path,
        [
            sys.executable,
            "-c",
            "import os; print(os.environ.get('PYTEST_CURRENT_TEST', '<missing>'))",
        ],
        timeout_s=30,
    )

    assert result.ok
    assert result.stdout.strip() == "<missing>"
