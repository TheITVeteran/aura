from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tools import (
    launch_structured_sft_research as containment,
)
from tools import run_detached_step

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or not containment.SANDBOX_PATH.is_file(),
    reason="kernel containment requires macOS sandbox-exec",
)


def _python() -> Path:
    return Path(sys.executable)


def _system_reads() -> tuple[Path, ...]:
    return (
        Path("/System"),
        Path("/Library"),
        Path("/usr"),
        Path("/bin"),
        Path("/sbin"),
        Path("/opt"),
        Path("/private/etc"),
        Path("/private/var/db/timezone"),
        Path(sys.prefix).resolve(strict=True),
        Path(__file__).resolve().parents[1],
    )


def test_profile_is_deny_default_and_names_forbidden_roots(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed"
    forbidden = tmp_path / "evaluator"
    lane = tmp_path / "lane" / "model_lane_control.json"
    allowed.mkdir()
    forbidden.mkdir()
    lane.parent.mkdir()
    profile = containment.build_sandbox_profile(
        python=_python(),
        read_paths=(*_system_reads(), tmp_path),
        write_paths=(tmp_path,),
        forbidden_roots=(forbidden,),
        model_lane_state=lane,
    )
    assert "(deny default)" in profile
    assert "(deny network*)" in profile
    assert "(deny process-fork)" in profile
    assert str(forbidden) in profile
    assert str(lane) in profile


def test_kernel_profile_denies_forbidden_read_write_network_and_fork(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed"
    forbidden = tmp_path / "evaluator"
    lane = tmp_path / "lane" / "model_lane_control.json"
    allowed.mkdir()
    forbidden.mkdir()
    lane.parent.mkdir()
    (allowed / "input.txt").write_text("allowed", encoding="ascii")
    (forbidden / "secret.txt").write_text("secret", encoding="ascii")
    profile = containment.build_sandbox_profile(
        python=_python(),
        read_paths=(*_system_reads(), tmp_path),
        write_paths=(tmp_path,),
        forbidden_roots=(forbidden,),
        model_lane_state=lane,
    )
    profile_path = tmp_path / "profile.sb"
    profile_path.write_text(profile, encoding="utf-8")
    script = """
import json
import os
import socket
import sys
from pathlib import Path

allowed = Path(sys.argv[1])
forbidden = Path(sys.argv[2])
port = int(sys.argv[3])
result = {
    "allowed_read": (allowed / "input.txt").read_text(encoding="ascii"),
}
(allowed / "output.txt").write_text("written", encoding="ascii")
result["allowed_write"] = (allowed / "output.txt").read_text(encoding="ascii")
for name, operation in (
    ("forbidden_read", lambda: (forbidden / "secret.txt").read_text()),
    ("forbidden_write", lambda: (forbidden / "new.txt").write_text("bad")),
    (
        "network",
        lambda: socket.create_connection(("127.0.0.1", port), timeout=1.0),
    ),
):
    try:
        operation()
    except OSError:
        result[name] = "kernel_denied"
    else:
        result[name] = "unexpectedly_allowed"
try:
    child = os.fork()
except OSError:
    result["fork"] = "kernel_denied"
else:
    if child == 0:
        os._exit(0)
    os.waitpid(child, 0)
    result["fork"] = "unexpectedly_allowed"
import mlx.core as mx
value = mx.sum(mx.array([1, 2, 3]))
mx.eval(value)
result["mlx"] = int(value.item())
print(json.dumps(result, sort_keys=True))
"""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        completed = subprocess.run(
            [
                str(containment.SANDBOX_PATH),
                "-f",
                str(profile_path),
                str(_python()),
                "-c",
                script,
                str(allowed),
                str(forbidden),
                str(port),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result == {
        "allowed_read": "allowed",
        "allowed_write": "written",
        "forbidden_read": "kernel_denied",
        "forbidden_write": "kernel_denied",
        "fork": "kernel_denied",
        "mlx": 6,
        "network": "kernel_denied",
    }


def test_profile_is_frozen_as_a_detached_execution_input(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "policy.sb"
    profile.write_text("(version 1) (allow default)\n", encoding="ascii")
    manifest = run_detached_step._build_execution_manifest(
        [str(containment.SANDBOX_PATH), "-f", str(profile), "/usr/bin/true"],
        Path(__file__).resolve().parents[1],
    )
    profile_roots = [
        item
        for item in manifest["roots"]
        if item.get("path") == str(profile.resolve())
    ]
    assert len(profile_roots) == 1
    assert profile_roots[0]["sha256"]


def test_sanitized_environment_excludes_model_and_custody_inputs(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    for relative in (
        "home",
        "tmp",
        "cache/huggingface/hub",
        "cache/mlx",
        "cache/transformers",
    ):
        (runtime / relative).mkdir(parents=True, exist_ok=True)
    lane = tmp_path / "lane" / "model_lane_control.json"
    lane.parent.mkdir()
    environment = containment._sanitized_environment(
        python=_python(),
        runtime_root=runtime,
        model_lane_state=lane,
    )
    assert "AURA_MODEL_PATH" not in environment
    assert all("EVALUATOR" not in key and "REPLAY" not in key for key in environment)
    assert environment["AURA_MODEL_LANE_STATE_PATH"] == str(lane)
    assert "AURA_MODEL_LANE_STATE_PATH" in run_detached_step._SAFE_ENVIRONMENT_KEYS


def test_forbidden_path_cannot_appear_in_target_command(tmp_path: Path) -> None:
    evaluator = tmp_path / "evaluator"
    command = ["/usr/bin/python3", "trainer.py", "--data", str(evaluator)]
    assert containment._command_contains_forbidden(command, (evaluator,))
    assert not containment._command_contains_forbidden(
        ["/usr/bin/python3", "trainer.py", "--data", str(tmp_path / "candidate")],
        (evaluator,),
    )
