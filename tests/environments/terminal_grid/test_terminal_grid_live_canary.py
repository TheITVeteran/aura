import sys

from core.environment.replay import EnvironmentTraceReplay
from core.runtime.subprocess_gateway import get_subprocess_gateway


def test_live_10_step_no_crash_trace_replay(tmp_path):
    trace_path = tmp_path / "canary.jsonl"
    result = get_subprocess_gateway().run(
        [
            sys.executable,
            "scripts/run_environment_canary.py",
            "--env",
            "terminal_grid:nethack",
            "--steps",
            "10",
            "--safe-mode",
            "--trace",
            str(trace_path),
        ],
        capture_output=True,
        timeout=120,
        offline_tooling=True,
        source="certification_tooling:terminal_grid_live_canary",
        accelerator_capability="none",
    )
    assert result.returncode == 0, result.stderr
    replay = EnvironmentTraceReplay().load(trace_path)
    assert replay.ok
    assert len(replay.rows) >= 10
