from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools import run_detached_step as detached

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="strong containment requires macOS")


def _safe_resume_verifier() -> list[str]:
    return [
        sys.executable,
        "-c",
        (
            "import hashlib,json,os,pathlib; "
            "plan=os.environ['AURA_DETACHED_PLAN_SHA256']; "
            "command=os.environ['AURA_DETACHED_COMMAND_SHA256']; "
            "attempt=int(os.environ['AURA_DETACHED_PRIOR_ATTEMPT']); "
            "head=os.environ['AURA_DETACHED_PRIOR_JOURNAL_HEAD_SHA256']; "
            "path=pathlib.Path(os.environ['AURA_DETACHED_RESUME_EVIDENCE_PATH']); "
            "e={'schema':'aura.detached_step.resume_evidence.v1','plan_sha256':plan,"
            "'command_sha256':command,'prior_attempt':attempt,"
            "'prior_journal_head_sha256':head,'checkpoint_sequence':0,"
            "'checkpoint_state':'test-safe'}; "
            "raw=(json.dumps(e,sort_keys=True,separators=(',',':'))+'\\n').encode(); "
            "path.write_bytes(raw); esha=hashlib.sha256(raw).hexdigest(); "
            "identity=hashlib.sha256(json.dumps({'prior_attempt':attempt,"
            "'prior_journal_head_sha256':head,'checkpoint_sequence':0,"
            "'evidence_sha256':esha},sort_keys=True,separators=(',',':')).encode()).hexdigest(); "
            "print(json.dumps({'schema':'aura.detached_step.resume_verdict.v2',"
            "'plan_sha256':plan,'command_sha256':command,'prior_attempt':attempt,"
            "'prior_journal_head_sha256':head,'checkpoint_sequence':0,"
            "'checkpoint_identity':identity,'verdict':'safe_to_resume',"
            "'evidence_path':str(path),'evidence_sha256':esha,'evidence':e}))"
        ),
    ]


def _indeterminate_resume_verifier() -> list[str]:
    command = _safe_resume_verifier()
    command[-1] = command[-1].replace("'safe_to_resume'", "'indeterminate'")
    return command


def _replaying_resume_verifier(state_path: Path) -> list[str]:
    return [
        sys.executable,
        "-c",
        (
            "import hashlib,json,os,pathlib\n"
            f"state=pathlib.Path({str(state_path)!r})\n"
            "if state.exists():\n"
            "    print(state.read_text())\n"
            "else:\n"
            "    plan=os.environ['AURA_DETACHED_PLAN_SHA256']\n"
            "    command=os.environ['AURA_DETACHED_COMMAND_SHA256']\n"
            "    attempt=int(os.environ['AURA_DETACHED_PRIOR_ATTEMPT'])\n"
            "    head=os.environ['AURA_DETACHED_PRIOR_JOURNAL_HEAD_SHA256']\n"
            "    path=pathlib.Path(os.environ['AURA_DETACHED_RESUME_EVIDENCE_PATH'])\n"
            "    evidence={'schema':'aura.detached_step.resume_evidence.v1',"
            "'plan_sha256':plan,'command_sha256':command,'prior_attempt':attempt,"
            "'prior_journal_head_sha256':head,'checkpoint_sequence':0}\n"
            "    raw=(json.dumps(evidence,sort_keys=True,separators=(',',':'))+'\\n').encode()\n"
            "    path.write_bytes(raw)\n"
            "    evidence_sha=hashlib.sha256(raw).hexdigest()\n"
            "    identity=hashlib.sha256(json.dumps({'prior_attempt':attempt,"
            "'prior_journal_head_sha256':head,'checkpoint_sequence':0,"
            "'evidence_sha256':evidence_sha},sort_keys=True,separators=(',',':')).encode()).hexdigest()\n"
            "    verdict={'schema':'aura.detached_step.resume_verdict.v2',"
            "'plan_sha256':plan,'command_sha256':command,'prior_attempt':attempt,"
            "'prior_journal_head_sha256':head,'checkpoint_sequence':0,"
            "'checkpoint_identity':identity,'verdict':'safe_to_resume',"
            "'evidence_path':str(path),'evidence_sha256':evidence_sha,'evidence':evidence}\n"
            "    state.write_text(json.dumps(verdict,sort_keys=True,separators=(',',':')))\n"
            "    print(json.dumps(verdict))\n"
        ),
    ]


def _wait_for(path: Path, timeout_s: float = 8.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {path}")


def _wait_for_text(path: Path, expected: str, timeout_s: float = 8.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.is_file() and path.read_text(encoding="utf-8") == expected:
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {path} to contain {expected!r}")


def _wait_for_state(path: Path, expected: str, timeout_s: float = 8.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                value = {}
            if value.get("state") == expected:
                return value
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {path} state {expected!r}")


def _launch(
    run_dir: Path,
    command: list[str],
    *,
    timeout_s: float = 5.0,
    resume: bool = False,
    resume_contract: str = "none",
    resume_verifier: list[str] | None = None,
) -> dict:
    resume_args = ["--resume"] if resume else []
    if resume_contract == "target_checkpoint" and resume_verifier is None:
        resume_verifier = _safe_resume_verifier()
    verifier_args = (
        ["--resume-verifier-json", json.dumps(resume_verifier)]
        if resume_verifier is not None
        else []
    )
    result = subprocess.run(
        [
            sys.executable,
            str(Path(detached.__file__).resolve()),
            "launch",
            "--run-dir",
            str(run_dir),
            "--name",
            "test-step",
            "--cwd",
            str(run_dir.parent),
            "--timeout",
            str(timeout_s),
            "--resume-contract",
            resume_contract,
            *verifier_args,
            *resume_args,
            "--",
            *command,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    return json.loads(result.stdout)


def test_nonzero_target_runs_once_and_survives_launcher(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    counter = tmp_path / "counter.txt"
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import sys; "
            f"p=Path({str(counter)!r}); "
            "p.write_text((p.read_text() if p.exists() else '')+'once\\n'); "
            "sys.exit(7)"
        ),
    ]
    launch = _launch(run_dir, command)
    assert launch["restart_policy"] == "never"
    receipt_path = run_dir / detached.RECEIPT_FILE
    receipt = _wait_for(receipt_path)
    first_receipt = receipt_path.read_bytes()
    assert receipt["status"] == "failed"
    assert receipt["returncode"] == 7
    assert receipt["restart_count"] == 0
    assert receipt["supervisor_attempt"] == 1
    assert counter.read_text(encoding="utf-8") == "once\n"

    time.sleep(0.4)
    assert receipt_path.read_bytes() == first_receipt
    assert counter.read_text(encoding="utf-8") == "once\n"
    inspection = detached._status(run_dir)
    assert inspection["terminal"] is True
    assert inspection["supervisor_alive"] is False


def test_timeout_kills_target_group_and_writes_terminal_receipt(tmp_path: Path) -> None:
    run_dir = tmp_path / "timeout"
    _launch(
        run_dir,
        [sys.executable, "-c", "import time; time.sleep(30)"],
        timeout_s=0.25,
    )
    receipt = _wait_for(run_dir / detached.RECEIPT_FILE)
    assert receipt["status"] == "timed_out"
    assert receipt["timed_out"] is True
    assert receipt["returncode"] == 124
    assert receipt["restart_count"] == 0


def test_duplicate_run_directory_is_immutable(tmp_path: Path) -> None:
    run_dir = tmp_path / "immutable"
    command = [sys.executable, "-c", "pass"]
    _launch(run_dir, command)
    _wait_for(run_dir / detached.RECEIPT_FILE)
    result = subprocess.run(
        [
            sys.executable,
            str(Path(detached.__file__).resolve()),
            "launch",
            "--run-dir",
            str(run_dir),
            "--name",
            "test-step",
            "--cwd",
            str(tmp_path),
            "--timeout",
            "5",
            "--",
            *command,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    assert result.returncode == 2
    assert "terminal receipt already exists" in result.stderr


def test_plan_and_receipt_hashes_replay(tmp_path: Path) -> None:
    run_dir = tmp_path / "hashes"
    _launch(run_dir, [sys.executable, "-c", "pass"])
    receipt = _wait_for(run_dir / detached.RECEIPT_FILE)
    plan = json.loads((run_dir / detached.PLAN_FILE).read_text(encoding="utf-8"))
    plan_body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    receipt_body = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    assert plan["plan_sha256"] == detached._sha256(plan_body)
    assert receipt["plan_sha256"] == plan["plan_sha256"]
    assert receipt["receipt_sha256"] == detached._sha256(receipt_body)
    attempts = detached._read_attempts(run_dir)
    assert [event["event"] for event in attempts] == [
        "LAUNCHED",
        "CONTROL_READY",
        "TARGET_STARTED",
        "TERMINAL",
    ]
    assert attempts[0]["previous_event_sha256"] == ""
    assert attempts[1]["previous_event_sha256"] == attempts[0]["event_sha256"]
    assert attempts[2]["previous_event_sha256"] == attempts[1]["event_sha256"]
    assert attempts[3]["previous_event_sha256"] == attempts[2]["event_sha256"]


def test_explicit_resume_reaps_stale_child_and_increments_attempt(tmp_path: Path) -> None:
    run_dir = tmp_path / "resumed"
    counter = tmp_path / "resume-counter.txt"
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import time; "
            f"p=Path({str(counter)!r}); "
            "n=int(p.read_text())+1 if p.exists() else 1; "
            "p.write_text(str(n)); "
            "time.sleep(30) if n == 1 else None"
        ),
    ]
    first = _launch(run_dir, command, timeout_s=60.0, resume_contract="target_checkpoint")
    status = _wait_for_state(run_dir / detached.STATUS_FILE, "running")
    assert status["supervisor_attempt"] == 1
    _wait_for_text(counter, "1")

    os.kill(first["supervisor_pid"], signal.SIGKILL)
    deadline = time.time() + 5.0
    while time.time() < deadline and detached._pid_matches(
        first["supervisor_pid"], first["supervisor_start_token"]
    ):
        time.sleep(0.05)
    assert not detached._pid_matches(first["supervisor_pid"], first["supervisor_start_token"])

    resumed = _launch(
        run_dir,
        command,
        timeout_s=60.0,
        resume=True,
        resume_contract="target_checkpoint",
    )
    assert resumed["resumed"] is True
    assert resumed["recovered_stale_child"] is True
    assert resumed["supervisor_attempt"] == 2
    receipt = _wait_for(run_dir / detached.RECEIPT_FILE)
    assert receipt["status"] == "passed"
    assert receipt["supervisor_attempt"] == 2
    assert counter.read_text(encoding="utf-8") == "2"
    events = detached._read_attempts(run_dir)
    assert [(event["event"], event["attempt"]) for event in events] == [
        ("LAUNCHED", 1),
        ("CONTROL_READY", 1),
        ("TARGET_STARTED", 1),
        ("LAUNCHED", 2),
        ("CONTROL_READY", 2),
        ("TARGET_STARTED", 2),
        ("TERMINAL", 2),
    ]


def test_resume_is_rejected_while_supervisor_is_alive(tmp_path: Path) -> None:
    run_dir = tmp_path / "live"
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    _launch(run_dir, command, timeout_s=60.0, resume_contract="target_checkpoint")
    _wait_for_state(run_dir / detached.STATUS_FILE, "running")
    result = subprocess.run(
        [
            sys.executable,
            str(Path(detached.__file__).resolve()),
            "launch",
            "--run-dir",
            str(run_dir),
            "--name",
            "test-step",
            "--cwd",
            str(tmp_path),
            "--timeout",
            "60",
            "--resume-contract",
            "target_checkpoint",
            "--resume-verifier-json",
            json.dumps(_safe_resume_verifier()),
            "--resume",
            "--",
            *command,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    assert result.returncode == 2
    assert "supervisor is already alive" in result.stderr
    stopped = detached._stop(run_dir)
    assert stopped["stopped"] is True
    assert stopped["control"] == "authenticated_socket"
    _wait_for(run_dir / detached.RECEIPT_FILE)


def test_resume_requires_existing_plan(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing"
    command = [sys.executable, "-c", "pass"]
    with pytest.raises(subprocess.CalledProcessError) as raised:
        _launch(run_dir, command, resume=True)
    assert "--resume requires an existing detached plan" in raised.value.stderr


def test_generic_incomplete_execution_cannot_be_replayed(tmp_path: Path) -> None:
    run_dir = tmp_path / "generic"
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    first = _launch(run_dir, command, timeout_s=60.0)
    _wait_for_state(run_dir / detached.STATUS_FILE, "running")
    os.kill(first["supervisor_pid"], signal.SIGKILL)
    deadline = time.time() + 5.0
    while time.time() < deadline and detached._pid_matches(
        first["supervisor_pid"], first["supervisor_start_token"]
    ):
        time.sleep(0.05)

    result = subprocess.run(
        [
            sys.executable,
            str(Path(detached.__file__).resolve()),
            "launch",
            "--run-dir",
            str(run_dir),
            "--name",
            "test-step",
            "--cwd",
            str(tmp_path),
            "--timeout",
            "60",
            "--resume-contract",
            "none",
            "--resume",
            "--",
            *command,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    assert result.returncode == 2
    assert "completion-indeterminate" in result.stderr
    target = next(
        event
        for event in detached._read_attempts(run_dir)
        if event["event"] == "TARGET_STARTED"
    )
    assert detached._terminate_stale_target(target) is True


def test_checkpoint_resume_requires_verifier_safe_verdict(tmp_path: Path) -> None:
    run_dir = tmp_path / "verifier-refusal"
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    verifier = _indeterminate_resume_verifier()
    first = _launch(
        run_dir,
        command,
        timeout_s=60.0,
        resume_contract="target_checkpoint",
        resume_verifier=verifier,
    )
    _wait_for_state(run_dir / detached.STATUS_FILE, "running")
    os.kill(first["supervisor_pid"], signal.SIGKILL)
    deadline = time.time() + 5.0
    while time.time() < deadline and detached._pid_matches(
        first["supervisor_pid"], first["supervisor_start_token"]
    ):
        time.sleep(0.05)
    with pytest.raises(subprocess.CalledProcessError) as raised:
        _launch(
            run_dir,
            command,
            timeout_s=60.0,
            resume=True,
            resume_contract="target_checkpoint",
            resume_verifier=verifier,
        )
    assert "verifier returned indeterminate" in raised.value.stderr


def test_authoritative_terminal_journal_recreates_receipt(tmp_path: Path) -> None:
    run_dir = tmp_path / "reconcile"
    _launch(run_dir, [sys.executable, "-c", "pass"])
    receipt_path = run_dir / detached.RECEIPT_FILE
    _wait_for(receipt_path)
    expected = receipt_path.read_bytes()
    receipt_path.unlink()

    inspection = detached._status(run_dir)
    assert inspection["terminal"] is True
    assert receipt_path.read_bytes() == expected


def test_terminal_journal_crash_boundary_reconciles_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "terminal-crash"
    monkeypatch.setenv("AURA_DETACHED_TEST_CRASH_POINT", "after_terminal_journal")
    launch = _launch(
        run_dir,
        [sys.executable, "-c", "import time; time.sleep(0.2)"],
    )
    deadline = time.time() + 5.0
    while time.time() < deadline and detached._pid_matches(
        launch["supervisor_pid"], launch["supervisor_start_token"]
    ):
        time.sleep(0.05)
    assert not (run_dir / detached.RECEIPT_FILE).exists()
    terminal = [
        event for event in detached._read_attempts(run_dir) if event["event"] == "TERMINAL"
    ]
    assert len(terminal) == 1

    inspection = detached._status(run_dir)
    assert inspection["terminal"] is True
    assert (run_dir / detached.RECEIPT_FILE).is_file()


@pytest.mark.parametrize(
    "crash_point",
    ("after_supervisor_fork_before_reservation", "after_reservation_before_release"),
)
def test_handoff_crash_boundaries_never_duplicate_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_point: str,
) -> None:
    run_dir = tmp_path / crash_point
    counter = tmp_path / f"{crash_point}.txt"
    command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(counter)!r}).write_text('once')",
    ]
    monkeypatch.setenv("AURA_DETACHED_TEST_CRASH_POINT", crash_point)
    with pytest.raises(subprocess.CalledProcessError):
        _launch(
            run_dir,
            command,
            timeout_s=30.0,
            resume_contract="target_checkpoint",
        )
    assert not counter.exists()
    monkeypatch.delenv("AURA_DETACHED_TEST_CRASH_POINT")
    status_path = run_dir / detached.STATUS_FILE
    if status_path.is_file():
        stale = json.loads(status_path.read_text(encoding="utf-8"))
        deadline = time.time() + 5.0
        while time.time() < deadline and detached._pid_matches(
            stale["supervisor_pid"], stale["supervisor_start_token"]
        ):
            time.sleep(0.05)

    resumed = _launch(
        run_dir,
        command,
        timeout_s=30.0,
        resume=True,
        resume_contract="target_checkpoint",
    )
    assert resumed["resumed"] is True
    receipt = _wait_for(run_dir / detached.RECEIPT_FILE)
    assert receipt["status"] == "passed"
    assert counter.read_text(encoding="utf-8") == "once"


def test_forged_status_target_identity_is_rejected(tmp_path: Path) -> None:
    run_dir = tmp_path / "forged-status"
    _launch(run_dir, [sys.executable, "-c", "pass"])
    _wait_for(run_dir / detached.RECEIPT_FILE)
    status_path = run_dir / detached.STATUS_FILE
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["child_process_group_id"] += 1
    detached._atomic_write(status_path, status)

    with pytest.raises(detached.DetachedStepError, match="status target identity mismatch"):
        detached._status(run_dir)


def test_log_symlink_is_rejected_without_touching_target(tmp_path: Path) -> None:
    run_dir = tmp_path / "symlink-log"
    run_dir.mkdir(mode=0o700)
    victim = tmp_path / "victim.txt"
    victim.write_text("unchanged", encoding="utf-8")
    (run_dir / detached.LOG_FILE).symlink_to(victim)

    _launch(run_dir, [sys.executable, "-c", "print('unsafe')"])
    receipt = _wait_for(run_dir / detached.RECEIPT_FILE)
    assert receipt["status"] == "supervisor_failed"
    assert receipt["child_pid"] == 0
    assert victim.read_text(encoding="utf-8") == "unchanged"


def test_supervisor_fault_after_target_release_cleans_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "fault-cleanup"
    monkeypatch.setenv("AURA_DETACHED_TEST_FAULT_POINT", "after_target_release")
    _launch(run_dir, [sys.executable, "-c", "import time; time.sleep(30)"], timeout_s=60.0)
    receipt = _wait_for(run_dir / detached.RECEIPT_FILE)
    assert receipt["status"] == "supervisor_failed"
    assert receipt["process_group_empty"] is True
    assert receipt["descendant_cleanup_performed"] is True
    assert detached._identity_state(receipt["child_pid"], receipt["child_start_token"]) == "dead"


def test_kernel_policy_rejects_term_ignoring_grandchild(tmp_path: Path) -> None:
    run_dir = tmp_path / "grandchild-rejected"
    grandchild_pid = tmp_path / "grandchild.pid"
    command = [
        sys.executable,
        "-c",
        (
            "import pathlib,signal,subprocess,sys; "
            "p=subprocess.Popen([sys.executable,'-c',"
            "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)']); "
            f"pathlib.Path({str(grandchild_pid)!r}).write_text(str(p.pid))"
        ),
    ]
    _launch(run_dir, command, timeout_s=60.0)
    receipt = _wait_for(run_dir / detached.RECEIPT_FILE, timeout_s=15.0)
    assert receipt["status"] == "failed"
    assert receipt["returncode"] != 0
    assert receipt["fork_policy"] == "kernel_denied"
    assert receipt["containment_verified"] is True
    assert receipt["process_group_empty"] is True
    assert not grandchild_pid.exists()


def test_environment_stripping_new_session_descendant_is_kernel_denied(tmp_path: Path) -> None:
    run_dir = tmp_path / "escaped-lineage-denied"
    outcome = tmp_path / "escaped.outcome"
    command = [
        sys.executable,
        "-c",
        (
            "exec(\"import os, pathlib, subprocess, sys\\n"
            f"outcome = pathlib.Path({str(outcome)!r})\\n"
            "try:\\n"
            "    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
            "start_new_session=True, env={'PATH': os.environ['PATH']})\\n"
            "except PermissionError:\\n"
            "    outcome.write_text('kernel-denied')\\n"
            "else:\\n"
            "    outcome.write_text(f'escaped:{child.pid}')\")"
        ),
    ]
    _launch(run_dir, command, timeout_s=60.0)
    receipt = _wait_for(run_dir / detached.RECEIPT_FILE, timeout_s=15.0)
    assert receipt["status"] == "passed"
    assert receipt["containment_verified"] is True
    assert receipt["lineage_empty"] is True
    assert receipt["fork_policy"] == "kernel_denied"
    assert outcome.read_text(encoding="utf-8") == "kernel-denied"


def test_terminal_duration_uses_monotonic_clock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = detached._build_plan("clock", [sys.executable, "-c", "pass"], tmp_path, 5.0, "none")
    monkeypatch.setattr(detached.time, "time", lambda: 10.0)
    monkeypatch.setattr(detached.time, "monotonic_ns", lambda: 9_000_000_000)
    receipt = detached._terminal_receipt(
        plan=plan,
        attempt=1,
        supervisor_pid=100,
        supervisor_start_token="token",
        child_pid=0,
        child_process_group_id=0,
        child_start_token="",
        started_at=999.0,
        started_monotonic_ns=1_000_000_000,
        returncode=0,
        timed_out=False,
        stop_signal=None,
        descendant_cleanup_performed=False,
        lineage_cleanup_count=0,
        containment_verified=True,
        supervisor_error=None,
    )
    assert receipt["finished_at"] == 10.0
    assert receipt["duration_s"] == 8.0


def test_checkpoint_contract_reconciles_indeterminate_completed_side_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "completed-before-receipt"
    invocations = tmp_path / "invocations.txt"
    durable_effect = tmp_path / "durable-effect.txt"
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; "
            f"calls=Path({str(invocations)!r}); effect=Path({str(durable_effect)!r}); "
            "calls.write_text((calls.read_text() if calls.exists() else '')+'call\\n'); "
            "effect.write_text('once') if not effect.exists() else None"
        ),
    ]
    monkeypatch.setenv("AURA_DETACHED_TEST_CRASH_POINT", "after_target_exit")
    first = _launch(
        run_dir,
        command,
        timeout_s=60.0,
        resume_contract="target_checkpoint",
    )
    _wait_for_text(durable_effect, "once")
    deadline = time.time() + 5.0
    while time.time() < deadline and detached._pid_matches(
        first["supervisor_pid"], first["supervisor_start_token"]
    ):
        time.sleep(0.05)
    assert not (run_dir / detached.RECEIPT_FILE).exists()
    assert detached._status(run_dir)["completion_indeterminate"] is True

    monkeypatch.delenv("AURA_DETACHED_TEST_CRASH_POINT")
    resumed = _launch(
        run_dir,
        command,
        timeout_s=60.0,
        resume=True,
        resume_contract="target_checkpoint",
    )
    assert resumed["prior_completion_indeterminate"] is True
    receipt = _wait_for(run_dir / detached.RECEIPT_FILE)
    assert receipt["status"] == "passed"
    assert invocations.read_text(encoding="utf-8") == "call\ncall\n"
    assert durable_effect.read_text(encoding="utf-8") == "once"


def test_unobservable_process_identity_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        detached,
        "_inspect_process",
        lambda _pid: detached.ProcessObservation("unknown"),
    )
    assert detached._identity_state(123, "token") == "unknown"
    with pytest.raises(detached.DetachedStepError, match="unobservable"):
        detached._wait_for_pid_exit(123, "token", 0.01)

    monkeypatch.setattr(
        detached,
        "_inspect_process",
        lambda _pid: detached.ProcessObservation("alive", token=""),
    )
    assert detached._identity_state(123, "token") == "unknown"


def test_attempt_journal_tampering_is_rejected(tmp_path: Path) -> None:
    run_dir = tmp_path / "tampered-journal"
    _launch(run_dir, [sys.executable, "-c", "pass"])
    _wait_for(run_dir / detached.RECEIPT_FILE)
    journal_path = run_dir / detached.ATTEMPTS_FILE
    lines = journal_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["supervisor_pid"] += 1
    lines[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
    journal_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(detached.DetachedStepError, match="journal hash mismatch"):
        detached._status(run_dir)


def test_plan_freezes_absolute_executable_and_secret_free_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-cross-boundary")
    plan = detached._build_plan(
        "frozen-execution",
        [sys.executable, "-c", "pass"],
        tmp_path,
        5.0,
        "none",
    )
    assert Path(plan["command"][0]).is_absolute()
    assert plan["executable_sha256"] == detached._sha256_file(Path(plan["command"][0]))
    assert "ANTHROPIC_API_KEY" not in plan["execution_environment"]
    assert "must-not-cross-boundary" not in json.dumps(plan)


def test_execution_manifest_detects_interpreted_script_mutation(tmp_path: Path) -> None:
    script = tmp_path / "target.py"
    script.write_text("print('first')\n", encoding="utf-8")
    plan = detached._build_plan(
        "script-freeze",
        [sys.executable, str(script)],
        tmp_path,
        5.0,
        "none",
    )
    detached._verify_execution_manifest_current(plan["target_execution_manifest"])

    script.write_text("print('mutated')\n", encoding="utf-8")
    with pytest.raises(detached.DetachedStepError, match="execution source changed"):
        detached._verify_execution_manifest_current(plan["target_execution_manifest"])


def test_mutated_resume_verifier_is_rejected_before_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "mutated-verifier"
    verifier_script = tmp_path / "resume_verifier.py"
    verifier_script.write_text("raise SystemExit(2)\n", encoding="utf-8")
    resume_verifier = [sys.executable, str(verifier_script)]
    monkeypatch.setenv("AURA_DETACHED_TEST_CRASH_POINT", "after_target_exit")
    launched = _launch(
        run_dir,
        [sys.executable, "-c", "pass"],
        timeout_s=30.0,
        resume_contract="target_checkpoint",
        resume_verifier=resume_verifier,
    )
    deadline = time.time() + 8.0
    while time.time() < deadline and detached._pid_matches(
        launched["supervisor_pid"], launched["supervisor_start_token"]
    ):
        time.sleep(0.05)
    monkeypatch.delenv("AURA_DETACHED_TEST_CRASH_POINT")
    verifier_script.write_text("print('replacement approved')\n", encoding="utf-8")

    with pytest.raises(subprocess.CalledProcessError) as raised:
        _launch(
            run_dir,
            [sys.executable, "-c", "pass"],
            timeout_s=30.0,
            resume=True,
            resume_contract="target_checkpoint",
            resume_verifier=resume_verifier,
        )
    assert "existing detached plan differs" in raised.value.stderr


def test_stale_resume_verdict_cannot_authorize_later_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "stale-verdict"
    resume_verifier = _replaying_resume_verifier(tmp_path / "stale-verdict.state")
    command = [sys.executable, "-c", "pass"]
    monkeypatch.setenv("AURA_DETACHED_TEST_CRASH_POINT", "after_target_exit")
    first = _launch(
        run_dir,
        command,
        timeout_s=30.0,
        resume_contract="target_checkpoint",
        resume_verifier=resume_verifier,
    )
    deadline = time.time() + 8.0
    while time.time() < deadline and detached._pid_matches(
        first["supervisor_pid"], first["supervisor_start_token"]
    ):
        time.sleep(0.05)

    second = _launch(
        run_dir,
        command,
        timeout_s=30.0,
        resume=True,
        resume_contract="target_checkpoint",
        resume_verifier=resume_verifier,
    )
    deadline = time.time() + 8.0
    while time.time() < deadline and detached._pid_matches(
        second["supervisor_pid"], second["supervisor_start_token"]
    ):
        time.sleep(0.05)
    monkeypatch.delenv("AURA_DETACHED_TEST_CRASH_POINT")

    with pytest.raises(subprocess.CalledProcessError) as raised:
        _launch(
            run_dir,
            command,
            timeout_s=30.0,
            resume=True,
            resume_contract="target_checkpoint",
            resume_verifier=resume_verifier,
        )
    assert "verdict binding is invalid" in raised.value.stderr
