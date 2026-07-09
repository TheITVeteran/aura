"""Contract tests for the crash-triage categorizer.

Synthetic forensic fixtures in the REAL on-disk formats (stall dumps,
sentinel log lines, faulthandler segments) — the fingerprints, window
filtering, ranking, and hard-death accounting are the contracts.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from tools.crash_triage import Triage, render_table

pytestmark = pytest.mark.unit

NOW = 1_783_500_000.0  # fixed 'now' so window math is deterministic


def make_root(tmp_path: Path) -> Path:
    root = tmp_path / "error_logs"
    for sub in ("stalls", "memory", "crash"):
        (root / sub).mkdir(parents=True)
    return root


def write_stall(root: Path, name: str, *, seconds: float, frames: list[str], age_s: float) -> None:
    body = [f"STALL DETECTED: {seconds}s", "=" * 20, "", "Thread ID: 1"]
    for f in frames:
        body.append(f'  File "{f}", line 42, in worker_fn')
    p = root / "stalls" / name
    p.write_text("\n".join(body), encoding="utf-8")
    import os

    os.utime(p, (NOW - age_s, NOW - age_s))


def local_ts(age_s: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(NOW - age_s))


def test_stall_fingerprint_skips_infrastructure_frames(tmp_path):
    root = make_root(tmp_path)
    write_stall(
        root, "stall_1.txt", seconds=6.1, age_s=3600,
        frames=[
            "/x/core/runtime/runtime_hygiene.py",     # wrapper — must be skipped
            "/x/core/memory/gateway_record_index.py", # the real site
        ],
    )
    report = Triage(root, window_days=7, now=NOW).run()
    stalls = [c for c in report["classes"] if c["kind"] == "stall"]
    assert len(stalls) == 1
    assert stalls[0]["fingerprint"] == "stall:gateway_record_index.py:worker_fn:5-10s"


def test_same_anatomy_is_one_class_with_count(tmp_path):
    root = make_root(tmp_path)
    for i in range(3):
        write_stall(
            root, f"stall_{i}.txt", seconds=7.0, age_s=1000 + i,
            frames=["/x/core/memory/gateway_record_index.py"],
        )
    report = Triage(root, window_days=7, now=NOW).run()
    stalls = [c for c in report["classes"] if c["kind"] == "stall"]
    assert len(stalls) == 1 and stalls[0]["count"] == 3


def test_sentinel_separates_hard_deaths_from_orderly_exits(tmp_path):
    root = make_root(tmp_path)
    lines = [
        f"[{local_ts(500)}] pid=1 exiting: SIGTERM received while guarding target pid=2",
        f"[{local_ts(400)}] pid=3 exiting: target pid=4 vanished; capturing death syslog",
        f"[{local_ts(300)}] pid=5 exiting: target pid=6 vanished; capturing death syslog",
        f"[{local_ts(200)}] pid=7 armed: target pid=8 lethal_mb=43008",
    ]
    (root / "memory" / "sentinel.log").write_text("\n".join(lines), encoding="utf-8")
    report = Triage(root, window_days=7, now=NOW).run()
    by_fp = {c["fingerprint"]: c for c in report["classes"]}
    assert by_fp["process_death:target_vanished"]["count"] == 2
    assert by_fp["orderly_exit:sigterm_guard"]["count"] == 1
    assert report["hard_death_total"] == 2


def test_window_excludes_old_incidents(tmp_path):
    root = make_root(tmp_path)
    write_stall(root, "stall_old.txt", seconds=6.0, age_s=30 * 86400,
                frames=["/x/core/memory/gateway_record_index.py"])
    lines = [f"[{local_ts(40 * 86400)}] pid=3 exiting: target pid=4 vanished; capturing death syslog"]
    (root / "memory" / "sentinel.log").write_text("\n".join(lines), encoding="utf-8")
    report = Triage(root, window_days=7, now=NOW).run()
    assert report["class_count"] == 0
    assert report["hard_death_total"] == 0


def test_faulthandler_fatal_errors_classed_by_kind(tmp_path):
    root = make_root(tmp_path)
    fh = root / "crash" / "faulthandler.log"
    fh.write_text(
        "Fatal Python error: Segmentation fault\n...stack...\n"
        "Fatal Python error: Segmentation fault\n...stack...\n"
        "Fatal Python error: Aborted\n...stack...\n",
        encoding="utf-8",
    )
    import os

    os.utime(fh, (NOW - 100, NOW - 100))
    report = Triage(root, window_days=7, now=NOW).run()
    by_fp = {c["fingerprint"]: c for c in report["classes"]}
    assert by_fp["fatal_error:Segmentation fault"]["count"] == 2
    assert by_fp["fatal_error:Aborted"]["count"] == 1


def test_ranking_puts_process_deaths_first_and_table_renders(tmp_path):
    root = make_root(tmp_path)
    write_stall(root, "stall_1.txt", seconds=6.0, age_s=100,
                frames=["/x/core/memory/gateway_record_index.py"])
    (root / "memory" / "sentinel.log").write_text(
        f"[{local_ts(50)}] pid=3 exiting: target pid=4 vanished; capturing death syslog\n",
        encoding="utf-8",
    )
    report = Triage(root, window_days=7, now=NOW).run()
    assert report["classes"][0]["kind"] == "process_death"
    table = render_table(report)
    assert "hard deaths" in table and "process_death:target_vanished" in table


def test_empty_root_is_a_clean_zero_report(tmp_path):
    report = Triage(tmp_path / "nonexistent", window_days=7, now=NOW).run()
    assert report["class_count"] == 0
    assert report["hard_death_total"] == 0
    assert report["collector_errors"] == []


def test_runtime_contract_doc_never_drifts_from_code():
    """docs/RUNTIME_CONTRACT.md is GENERATED; if health_contract.py changes
    and the doc doesn't, this fails — the contract cannot silently drift."""
    from tools.render_health_contract import DOC_PATH, render

    assert DOC_PATH.is_file(), "run: python tools/render_health_contract.py"
    assert DOC_PATH.read_text(encoding="utf-8") == render()


def test_faulthandler_segments_dated_by_boot_marker_not_mtime(tmp_path):
    """The log is append-only across boots; mtime is always 'now'. A June
    segfault must not surface in this week's window just because the file
    got a fresh boot marker appended tonight."""
    import os

    root = make_root(tmp_path)
    fh = root / "crash" / "faulthandler.log"
    old = NOW - 30 * 86400
    recent = NOW - 3600
    fh.write_text(
        f"===== boot pid=111 at={old} =====\n"
        "Fatal Python error: Segmentation fault\n...old stack...\n"
        f"===== boot pid=222 at={recent} =====\n"
        "Fatal Python error: Aborted\n...fresh stack...\n",
        encoding="utf-8",
    )
    os.utime(fh, (NOW, NOW))  # file freshly touched — must NOT matter
    report = Triage(root, window_days=7, now=NOW).run()
    by_fp = {c["fingerprint"]: c for c in report["classes"]}
    assert "fatal_error:Segmentation fault" not in by_fp, "June crash leaked into the window"
    assert by_fp["fatal_error:Aborted"]["count"] == 1
    assert "pid=222" in by_fp["fatal_error:Aborted"]["example_receipt"]
