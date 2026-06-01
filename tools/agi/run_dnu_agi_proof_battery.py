#!/usr/bin/env python3
"""
tools/agi/run_dnu_agi_proof_battery.py
DNU AGI Proof Battery Runner.

Executes sealed task packs through Aura's live launcher message path,
grades responses against salted answer hashes, and produces honest scorecards.

ZERO synthetic scores. ZERO projected baselines. ZERO theater.
Every number in the output comes from actual task execution.
"""

import asyncio
import contextlib
import gc
import hashlib
import json
import os
import platform
import re
import signal
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

# Insert project root into sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_GIT_METADATA_ERRORS = (OSError, UnicodeDecodeError, ValueError)
_DNU_RUN_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_DNU_TASK_ATTEMPT_ERRORS = (asyncio.TimeoutError, *_DNU_RUN_RECOVERABLE_ERRORS)

# Proof tasks are foreground user-equivalent turns. API/websocket are transport
# labels; the live cognitive route should see the same origin as desktop chat.
PROOF_LIVE_MESSAGE_ORIGIN = "user"

DNU_STALE_ARTIFACTS = (
    "TASK_TRACE.jsonl",
    "FAILURES.jsonl",
    "RECEIPTS.jsonl",
    "RESOURCE_TRACE.jsonl",
    "LIFECYCLE_EVENTS.jsonl",
    "RUN_STATUS.json",
    "SCORECARD.json",
    "DNU_AGI_PROOF.json",
    "DNU_AGI_PROOF.md",
    "FINAL_VERDICT.txt",
    "GOVERNANCE_REPORT.json",
    "LEAKAGE_REPORT.json",
    "MODEL_LANE_PROBE.json",
    "MANIFEST.json",
)

DNU_STANDARD_COPY_ARTIFACTS = (
    "DNU_AGI_PROOF.json",
    "DNU_AGI_PROOF.md",
    "SCORECARD.json",
    "BASELINES.json",
    "ABLATIONS.json",
    "TASK_TRACE.jsonl",
    "FAILURES.jsonl",
    "RECEIPTS.jsonl",
    "RESOURCE_TRACE.jsonl",
    "LIFECYCLE_EVENTS.jsonl",
    "RUN_STATUS.json",
    "GOVERNANCE_REPORT.json",
    "LEAKAGE_REPORT.json",
    "RUNTIME_MANIFEST.json",
    "RUNTIME_POLICY.json",
    "MODEL_LANE_PROBE.json",
    "EXCLUSIVE_RUNTIME_PREFLIGHT.json",
    "FINAL_VERDICT.txt",
    "MANIFEST.json",
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _is_resource_tracker_process(proc) -> bool:
    try:
        name = str(proc.name() or "").lower()
    except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
        name = ""
    try:
        cmdline = " ".join(str(part) for part in (proc.cmdline() or [])).lower()
    except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
        cmdline = ""
    return (
        name in {"resource_tracker", "semaphore_tracker"}
        or "multiprocessing.resource_tracker" in cmdline
        or "multiprocessing.semaphore_tracker" in cmdline
    )


def get_git_commit() -> str:
    try:
        git_dir = PROJECT_ROOT / ".git"
        if not git_dir.exists():
            return "unknown_no_git_dir"
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref = head.split(" ", 1)[1].strip()
            ref_path = git_dir / ref
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()
            packed_refs = git_dir / "packed-refs"
            if packed_refs.exists():
                for line in packed_refs.read_text(encoding="utf-8").splitlines():
                    if line and not line.startswith("#") and line.endswith(f" {ref}"):
                        return line.split(" ", 1)[0].strip()
            return "unknown_ref_not_found"
        return head
    except _GIT_METADATA_ERRORS as e:
        return f"unknown_error_{type(e).__name__}"


def normalize_answer(raw: str) -> str:
    """Normalize an answer for hash comparison: lowercase, strip, remove trailing punctuation."""
    ans = re.sub(r"\\[nrt]", " ", str(raw or ""), flags=re.IGNORECASE)
    ans = ans.replace("\\", " ").strip().lower()
    # Remove trailing period, comma, semicolon
    ans = ans.rstrip(".,;:!?")
    # Collapse whitespace
    ans = re.sub(r"\s+", " ", ans).strip()
    return ans


def extract_answer_tag(text: str) -> str | None:
    """Extract content from <answer>...</answer> tags with robust fallbacks."""
    ans = None
    
    # 1. Standard tags
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        ans = match.group(1).strip()
    
    # 2. Markdown bold/italic tag indicators
    if not ans:
        match = re.search(r"\*\*(?:final\s+)?answer\*\*:\s*([^\n]+)", text, re.IGNORECASE)
        if match:
            ans = match.group(1).strip()
    
    # 3. Plain text final answer indicator
    if not ans:
        match = re.search(r"(?:final\s+)?answer:\s*([^\n]+)", text, re.IGNORECASE)
        if match:
            ans = match.group(1).strip()
    
    # 4. Look for "therefore, the answer is X" or similar
    if not ans:
        match = re.search(r"(?:therefore|thus|hence|so),\s*(?:the\s+)?answer\s+(?:is|must\s+be)\s+([^\n.]+)", text, re.IGNORECASE)
        if match:
            ans = match.group(1).strip()
            
    if ans:
        ans = re.sub(r"</?(?:user|assistant|system|answer|im_start|im_end)!?\s*>?", "", ans, flags=re.IGNORECASE)
        ans = re.sub(r"</?[^>\s]+!?\s*>?", "", ans)
        ans = re.sub(r"\s+", " ", ans).strip()
        # Strip common preamble noise if it somehow leaked into the extracted text
        lower_ans = ans.lower()
        prefixes = (
            "the proposed answer is 100% correct",
            "the proposed answer is correct",
            "the proposed answer is accurate",
            "no corrections are needed",
            "no correction is needed",
            "proposed answer is correct",
            "proposed answer is 100% correct"
        )
        for prefix in prefixes:
            if lower_ans.startswith(prefix):
                ans = ans[len(prefix):].strip().lstrip(".,;:!*- ")
                lower_ans = ans.lower()
        return ans
        
    return None


def extract_exact_answer_envelope(text: str) -> str | None:
    """Extract content only when the whole response is exactly one answer envelope."""
    match = re.fullmatch(
        r"\s*<answer>\s*(.*?)\s*</answer>\s*",
        str(text or ""),
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).strip()


def hash_answer(salt: str, answer: str) -> str:
    """Compute SHA-256 hash of salt+answer."""
    return hashlib.sha256((salt + answer).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_artifact_manifest(
    artifact_dir: Path,
    *,
    run_id: str,
    commit_sha: str,
    timestamp: float | None = None,
    include_files: Iterable[str] | None = None,
) -> dict:
    """Build a self-consistent artifact manifest.

    MANIFEST.json is deliberately excluded. A manifest that hashes a previous
    copy of itself becomes unverifiable as soon as it is rewritten.
    """
    manifest = {
        "run_id": run_id,
        "commit_sha": commit_sha,
        "files": {},
    }
    if timestamp is not None:
        manifest["timestamp"] = timestamp

    if include_files is None:
        candidates = sorted(p for p in artifact_dir.iterdir() if p.is_file())
    else:
        candidates = [artifact_dir / name for name in include_files]

    for artifact_file in candidates:
        if not artifact_file.is_file() or artifact_file.name == "MANIFEST.json":
            continue
        manifest["files"][artifact_file.name] = {
            "path": str(artifact_file.relative_to(PROJECT_ROOT))
            if artifact_file.is_relative_to(PROJECT_ROOT)
            else str(artifact_file),
            "sha256": sha256_file(artifact_file),
            "size_bytes": artifact_file.stat().st_size,
        }

    return manifest


def write_artifact_manifest(
    artifact_dir: Path,
    *,
    run_id: str,
    commit_sha: str,
    timestamp: float | None = None,
    include_files: Iterable[str] | None = None,
) -> dict:
    manifest = build_artifact_manifest(
        artifact_dir,
        run_id=run_id,
        commit_sha=commit_sha,
        timestamp=timestamp,
        include_files=include_files,
    )
    (artifact_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON through a temporary file so interrupted runs are detectable."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")


def write_run_status(
    run_dir: Path,
    *,
    status: str,
    run_id: str,
    commit_sha: str,
    phase: str,
    tasks_completed: int = 0,
    total_tasks: int | None = None,
    error: str | None = None,
    lifecycle_events: int = 0,
) -> dict:
    payload = {
        "schema": "aura.dnu_run_status.v1",
        "status": status,
        "run_id": run_id,
        "commit_sha": commit_sha,
        "phase": phase,
        "tasks_completed": int(tasks_completed),
        "total_tasks": total_tasks,
        "runner_completed": status == "complete",
        "error": error,
        "lifecycle_events": int(lifecycle_events),
        "updated_at_unix": time.time(),
        "updated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json_atomic(run_dir / "RUN_STATUS.json", payload)
    return payload


def collect_proof_resource_snapshot(
    *,
    label: str,
    task_index: int = 0,
    task_id: str = "",
) -> dict:
    """Collect bounded proof-run resource evidence without depending on psutil."""
    snapshot = {
        "schema": "aura.dnu_resource_snapshot.v1",
        "label": label,
        "task_index": int(task_index),
        "task_id": task_id,
        "timestamp_unix": time.time(),
        "process_rss_mb": None,
        "child_rss_mb": None,
        "child_count": None,
        "system_memory_percent": None,
        "system_available_mb": None,
        "runtime_health_contract": None,
        "error": None,
    }
    try:
        from core.runtime.health_contract import runtime_health_report

        snapshot["runtime_health_contract"] = runtime_health_report()
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        snapshot["runtime_health_contract"] = {
            "healthy": False,
            "status": "health_contract_unavailable",
            "required_probes": {"all_passed": False},
            "error": f"{type(exc).__name__}: {exc}",
        }
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        children = proc.children(recursive=True)
        child_rss = 0
        live_children = 0
        for child in children:
            try:
                child_rss += int(child.memory_info().rss)
                live_children += 1
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        mem = psutil.virtual_memory()
        snapshot.update(
            {
                "process_rss_mb": round(proc.memory_info().rss / (1024 * 1024), 2),
                "child_rss_mb": round(child_rss / (1024 * 1024), 2),
                "child_count": live_children,
                "system_memory_percent": float(mem.percent),
                "system_available_mb": round(float(mem.available) / (1024 * 1024), 2),
            }
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        snapshot["error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


def proof_runtime_health_blockers(snapshot: dict) -> list[str]:
    """Return blocker strings when a proof resource snapshot is not closure-safe."""
    health = snapshot.get("runtime_health_contract")
    if not isinstance(health, dict):
        return ["runtime health contract missing from resource snapshot"]
    blockers: list[str] = []
    if health.get("healthy") is not True:
        blockers.append(f"runtime health status is {health.get('status')}")
    required = health.get("required_probes") or {}
    if isinstance(required, dict) and required.get("all_passed") is not True:
        failed = [
            name
            for name, probe in required.items()
            if isinstance(probe, dict) and probe.get("ok") is not True
        ]
        blockers.append(f"required health probes failed: {failed or 'unknown'}")
    return blockers


async def wait_for_proof_runtime_health(
    *,
    label: str,
    task_index: int = 0,
    task_id: str = "",
    timeout_s: float = 60.0,
    interval_s: float = 2.0,
) -> tuple[dict, list[str]]:
    """Wait for transient proof-runtime degradation to genuinely recover.

    Cold model warmup can cause a short hypervisor/event-loop lag incident.
    The proof gate should not ignore it, but it also should not fail a healthy
    recovered runtime from a single cold-start spike. This function proceeds
    only after the canonical health contract is healthy again, otherwise it
    returns the final blockers and the caller fails closed.
    """
    deadline = time.monotonic() + max(0.0, timeout_s)
    first_blockers: list[str] | None = None
    attempts = 0
    last_snapshot: dict | None = None
    last_blockers: list[str] = []

    while True:
        attempts += 1
        snapshot = collect_proof_resource_snapshot(
            label=label,
            task_index=task_index,
            task_id=task_id,
        )
        blockers = proof_runtime_health_blockers(snapshot)
        last_snapshot = snapshot
        last_blockers = blockers
        if first_blockers is None:
            first_blockers = list(blockers)

        if not blockers:
            if first_blockers:
                snapshot["runtime_health_recovery"] = {
                    "initial_blockers": first_blockers,
                    "attempts": attempts,
                    "recovered": True,
                }
            return snapshot, []

        if time.monotonic() >= deadline:
            snapshot["runtime_health_recovery"] = {
                "initial_blockers": first_blockers or list(blockers),
                "attempts": attempts,
                "recovered": False,
            }
            return snapshot, blockers

        await asyncio.sleep(max(0.0, interval_s))

    return last_snapshot or {}, last_blockers


def dnu_model_recycle_interval(requested_tier: str, *, total_tasks: int, smoke: bool) -> int:
    """Return the task interval for primary-lane model-worker recycling."""
    raw = os.environ.get("AURA_DNU_MODEL_RECYCLE_INTERVAL")
    if raw is not None:
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0
    if smoke or total_tasks <= 40:
        return 0
    if requested_tier == "primary":
        return 40
    return 0


def _cmdline_invokes_script(cmdline: list[Any], script_name: str) -> bool:
    """Return true only when ``script_name`` is an argv entry, not shell text."""

    parts = [str(part) for part in (cmdline or [])]
    for index, part in enumerate(parts):
        if Path(part).name != script_name:
            continue
        if index == 0:
            return True
        executable = Path(parts[0]).name.lower()
        return executable.startswith("python") or executable in {"uv", "uvx"}
    return False


def find_existing_aura_runtimes() -> list[dict]:
    """Return live aura_main.py runtime processes that would contend with proof boot."""
    if sys.platform not in ("darwin", "linux"):
        return []
    me = os.getpid()
    parent = os.getppid()
    current_user = os.environ.get("USER") or ""
    try:
        import psutil
    except ImportError as exc:
        print(f"  [WARN] Could not inspect process table for Aura runtime exclusivity: {exc}")
        return []

    instances: list[dict] = []
    for proc in psutil.process_iter(["pid", "username", "cmdline", "ppid"]):
        try:
            info = proc.info
            pid = int(info.get("pid") or 0)
            user = str(info.get("username") or "")
            cmdline = info.get("cmdline") or []
            command = " ".join(str(part) for part in cmdline)
            proc_parent = int(info.get("ppid") or 0)
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess, TypeError, ValueError):
            continue
        if pid in (me, parent):
            continue
        if proc_parent == me:
            continue
        if current_user and user != current_user:
            continue
        if not _cmdline_invokes_script(cmdline, "aura_main.py"):
            continue
        if "run_dnu_agi_proof_battery.py" in command:
            continue
        if "--stop" in command:
            continue
        instances.append({"pid": pid, "user": user, "command": command})
    return instances


def _is_aura_checkout_cwd(path: Path) -> bool:
    """Return true for the live checkout and Aura proof temp checkouts."""
    try:
        resolved = Path(path).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    if resolved == PROJECT_ROOT.resolve():
        return True
    return resolved.name == "repo" and any(
        str(part).startswith("aura-live-") for part in resolved.parts
    )


def find_existing_proof_runners() -> list[dict]:
    """Return stale DNU proof-runner processes from this checkout."""
    if sys.platform not in ("darwin", "linux"):
        return []
    me = os.getpid()
    current_user = os.environ.get("USER") or ""
    try:
        import psutil
    except ImportError:
        return []

    instances: list[dict] = []
    for proc in psutil.process_iter(["pid", "username", "cmdline"]):
        try:
            pid = int(proc.info.get("pid") or 0)
            if pid == me:
                continue
            user = str(proc.info.get("username") or "")
            cmdline = proc.info.get("cmdline") or []
            command = " ".join(str(part) for part in cmdline)
            cwd = Path(proc.cwd()).resolve()
        except (
            psutil.AccessDenied,
            psutil.NoSuchProcess,
            psutil.ZombieProcess,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            continue
        if current_user and user != current_user:
            continue
        if not _cmdline_invokes_script(cmdline, "run_dnu_agi_proof_battery.py"):
            continue
        if not _is_aura_checkout_cwd(cwd):
            continue
        instances.append({"pid": pid, "user": user, "command": command})
    return instances


def stop_existing_proof_runners(timeout_s: float = 8.0) -> list[dict]:
    """Stop stale proof-runner process trees before an exclusive proof boot."""
    instances = find_existing_proof_runners()
    if not instances:
        return []
    print(f"  [EXCLUSIVE] Stopping {len(instances)} stale proof runner process(es).")
    try:
        import psutil
    except ImportError:
        psutil = None

    processes = []
    for instance in instances:
        pid = int(instance["pid"])
        if psutil is not None:
            try:
                parent = psutil.Process(pid)
                processes.extend(parent.children(recursive=True))
                processes.append(parent)
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        else:
            try:
                os.kill(pid, signal.SIGTERM)
            except (OSError, PermissionError):
                continue

    if psutil is not None:
        for proc in processes:
            try:
                proc.terminate()
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        _gone, alive = psutil.wait_procs(processes, timeout=max(1.0, timeout_s))
        for proc in alive:
            try:
                proc.kill()
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        if alive:
            psutil.wait_procs(alive, timeout=0.5)
    else:
        time.sleep(max(1.0, timeout_s))
    return find_existing_proof_runners()


def stop_existing_aura_runtimes(timeout_s: float = 8.0) -> list[dict]:
    """Best-effort stop for live Aura runtimes before a proof run claims exclusivity."""
    instances = find_existing_aura_runtimes()
    if not instances:
        return []

    print(f"  [EXCLUSIVE] Stopping {len(instances)} existing Aura runtime process(es).")
    try:
        from aura_main import stop_aura

        stop_aura()
    except _DNU_RUN_RECOVERABLE_ERRORS as exc:
        print(f"  [WARN] aura_main.stop_aura() did not complete cleanly: {exc}")

    deadline = time.time() + max(1.0, timeout_s)
    for instance in instances:
        try:
            os.kill(int(instance["pid"]), signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            print(f"  [WARN] Permission denied stopping Aura PID {instance['pid']}: {exc}")

    while time.time() < deadline:
        remaining = find_existing_aura_runtimes()
        if not remaining:
            return []
        time.sleep(0.25)

    remaining = find_existing_aura_runtimes()
    for instance in remaining:
        try:
            os.kill(int(instance["pid"]), signal.SIGKILL)
        except (OSError, PermissionError) as exc:
            print(f"  [WARN] Could not SIGKILL Aura PID {instance['pid']}: {exc}")
    time.sleep(0.5)
    return find_existing_aura_runtimes()


def stop_orphaned_aura_multiprocessing_children(timeout_s: float = 2.0) -> list[dict]:
    """Reap orphaned Aura-owned multiprocessing helpers from interrupted proof runs."""
    if sys.platform not in ("darwin", "linux"):
        return []
    try:
        import psutil
    except ImportError:
        return []

    candidates = []
    orphan_parent_pids: set[int] = set()
    for proc in psutil.process_iter(["pid", "ppid", "cmdline"]):
        try:
            pid = int(proc.info.get("pid") or 0)
            ppid = int(proc.info.get("ppid") or 0)
            cmdline = proc.info.get("cmdline") or []
            command = " ".join(str(part) for part in cmdline)
            cwd = Path(proc.cwd()).resolve()
        except (
            psutil.AccessDenied,
            psutil.NoSuchProcess,
            psutil.ZombieProcess,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            continue
        if pid == os.getpid() or ppid != 1:
            continue
        if not _is_aura_checkout_cwd(cwd):
            continue
        if (
            "multiprocessing.spawn" not in command
            and "multiprocessing.resource_tracker" not in command
            and "multiprocessing.semaphore_tracker" not in command
        ):
            continue
        orphan_parent_pids.add(pid)
        candidates.append({"pid": pid, "command": command[:240], "role": "orphan_parent"})

    processes_by_pid: dict[int, Any] = {}
    for pid in sorted(orphan_parent_pids):
        try:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                processes_by_pid[int(child.pid)] = child
                try:
                    child_cmd = " ".join(str(part) for part in (child.cmdline() or []))
                except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess, OSError, RuntimeError, TypeError, ValueError):
                    child_cmd = ""
                candidates.append(
                    {"pid": int(child.pid), "command": child_cmd[:240], "role": "orphan_descendant"}
                )
            processes_by_pid[pid] = parent
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess, OSError, RuntimeError, TypeError, ValueError):
            continue

    for proc in psutil.process_iter(["pid", "ppid", "cmdline"]):
        try:
            pid = int(proc.info.get("pid") or 0)
            ppid = int(proc.info.get("ppid") or 0)
            cmdline = proc.info.get("cmdline") or []
            command = " ".join(str(part) for part in cmdline)
            cwd = Path(proc.cwd()).resolve()
        except (
            psutil.AccessDenied,
            psutil.NoSuchProcess,
            psutil.ZombieProcess,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            continue
        if pid == os.getpid() or ppid != 1 or not _is_aura_checkout_cwd(cwd):
            continue
        if not cmdline or Path(str(cmdline[0])).name != "caffeinate":
            continue
        if pid not in processes_by_pid:
            processes_by_pid[pid] = proc
            candidates.append({"pid": pid, "command": command[:240], "role": "orphan_keep_awake"})

    if not candidates:
        return []

    processes = list(processes_by_pid.values())
    for proc in processes:
        try:
            proc.terminate()
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    _, alive = psutil.wait_procs(processes, timeout=timeout_s)
    for proc in alive:
        try:
            proc.kill()
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    if alive:
        psutil.wait_procs(alive, timeout=0.5)
    return candidates


async def _reap_proof_child_processes(
    reason: str,
    *,
    include_resource_trackers: bool = False,
) -> None:
    """Bounded cleanup for child processes that would otherwise block process exit."""
    reaped: list[str] = []

    try:
        import multiprocessing as mp

        mp_children = list(mp.active_children())
    except _DNU_RUN_RECOVERABLE_ERRORS:
        mp_children = []

    for child in mp_children:
        try:
            if child.is_alive():
                child.terminate()
                reaped.append(f"mp:{child.name}:{child.pid}")
        except _DNU_RUN_RECOVERABLE_ERRORS:
            continue

    if mp_children:
        await asyncio.gather(
            *[asyncio.to_thread(child.join, 1.5) for child in mp_children],
            return_exceptions=True,
        )
        for child in mp_children:
            try:
                if child.is_alive() and hasattr(child, "kill"):
                    child.kill()
                    await asyncio.to_thread(child.join, 0.5)
                    reaped.append(f"mp-kill:{child.name}:{child.pid}")
            except _DNU_RUN_RECOVERABLE_ERRORS:
                continue

    try:
        import psutil

        parent = psutil.Process(os.getpid())
        children = [
            child
            for child in parent.children(recursive=True)
            if include_resource_trackers or not _is_resource_tracker_process(child)
        ]
        if children:
            for proc in children:
                try:
                    proc.terminate()
                    reaped.append(f"pid:{proc.pid}:{proc.name()}")
                except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                    continue
            _gone, alive = psutil.wait_procs(children, timeout=1.5)
            for proc in alive:
                try:
                    proc.kill()
                    reaped.append(f"pid-kill:{proc.pid}:{proc.name()}")
                except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                    continue
            if alive:
                psutil.wait_procs(alive, timeout=0.5)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        pass

    if reaped:
        unique = list(dict.fromkeys(reaped))
        print(
            f"  [CLEANUP] Reaped {len(unique)} proof child process(es) during {reason}: "
            + ", ".join(unique[:8])
        )


def write_exclusive_runtime_report(path: Path, *, status: str, instances: list[dict]) -> dict:
    report = {
        "status": status,
        "checked_at_unix": time.time(),
        "existing_runtime_count": len(instances),
        "instances": instances,
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _router_endpoint_tier(router, endpoint_name: str) -> str:
    endpoint = getattr(router, "endpoints", {}).get(endpoint_name) if router is not None else None
    tier = getattr(endpoint, "tier", "")
    if hasattr(tier, "value"):
        return str(tier.value).lower()
    return str(tier).lower()


async def _bounded_probe_metadata(
    router,
    *,
    timeout_s: float,
    abort_reason: str,
    **kwargs,
) -> dict:
    """Run a proof-lane router call with a hard abort boundary."""
    task = asyncio.create_task(
        router.generate_with_metadata(**kwargs),
        name=f"dnu_model_lane_probe:{abort_reason}",
    )
    try:
        return await asyncio.wait_for(task, timeout=timeout_s)
    except (asyncio.TimeoutError, TimeoutError) as exc:
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, TimeoutError):
                pass
        _force_abort_router_generation(router, reason=abort_reason)
        raise TimeoutError(abort_reason) from exc


async def run_model_lane_probe(router, requested_tier: str, run_dir: Path) -> dict:
    """Exercise the requested proof model lane before expensive task execution."""
    probe_path = run_dir / "MODEL_LANE_PROBE.json"
    report = {
        "status": "fail",
        "requested_tier": requested_tier,
        "ok": False,
        "endpoint": "",
        "endpoint_tier": "",
        "elapsed_s": 0.0,
        "strict_answer_ok": False,
        "strict_answer_source": "",
        "nonempty_model_text_ok": False,
        "local_lane_ok": False,
        "error": "",
        "text_preview": "",
    }

    if router is None or not hasattr(router, "generate_with_metadata"):
        report["error"] = "llm_router_missing_generate_with_metadata"
        probe_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    strict_probe_prompt = (
        "This is a proof-runtime model lane health probe. "
        "Return the lowercase two-letter token formed by joining 'o' and 'k'. "
        "Wrap only that token between <answer> and </answer>. "
        "Return no other text."
    )
    raw_strict_probe_prompt = "Output exactly these two lowercase letters and nothing else: ok"
    prompt = (
        "This is a proof-runtime model lane health probe. "
        "Reply with one short sentence confirming the requested local model lane is ready."
    )
    strict_answer_ok = False
    strict_answer_source = ""
    structured_solver_enabled = (
        str(os.environ.get("AURA_DISABLE_STRUCTURED_PROOF_SOLVER", "") or "").strip().lower()
        not in {"1", "true", "yes", "on"}
    )
    if structured_solver_enabled:
        try:
            from core.reasoning.proof_answer_solver import solve_strict_proof_prompt

            solved = solve_strict_proof_prompt(strict_probe_prompt)
            if solved and normalize_answer(solved.answer) == "ok":
                strict_answer_ok = True
                strict_answer_source = solved.solver
        except _DNU_RUN_RECOVERABLE_ERRORS as exc:
            strict_answer_source = f"solver_error:{type(exc).__name__}"
    else:
        try:
            strict_metadata = await _bounded_probe_metadata(
                router,
                timeout_s=330.0,
                abort_reason="proof_model_lane_strict_probe_timeout_330s",
                prompt=raw_strict_probe_prompt,
                system_prompt="Return only the requested final answer value. No explanation.",
                timeout=300.0,
                prefer_tier=requested_tier,
                origin="internal",
                purpose="proof_model_lane_strict_probe",
                foreground_request=True,
                health_probe=True,
                skip_runtime_payload=True,
                strict_value_contract=True,
                allow_cloud_fallback=False,
                disable_prompt_cache=True,
                clear_prompt_cache=True,
                temperature=0,
                max_tokens=24,
                num_predict=24,
            )
            strict_text = str(strict_metadata.get("text", "") or "")
            strict_answer = extract_answer_tag(strict_text) or strict_text
            strict_answer_ok = bool(strict_metadata.get("ok")) and normalize_answer(strict_answer) == "ok"
            strict_answer_source = (
                f"model_lane:{strict_metadata.get('endpoint', '')}"
                if strict_answer_ok
                else f"model_lane_failed:{strict_metadata.get('error', '') or strict_metadata.get('text', '')[:80]}"
            )
        except _DNU_TASK_ATTEMPT_ERRORS as exc:
            strict_answer_source = f"model_lane_error:{type(exc).__name__}: {exc}"

    t0 = time.time()
    try:
        metadata = await _bounded_probe_metadata(
            router,
            timeout_s=330.0,
            abort_reason="proof_model_lane_probe_timeout_330s",
            prompt=prompt,
            system_prompt="Answer the lane health probe directly and briefly.",
            timeout=300.0,
            prefer_tier=requested_tier,
            origin="internal",
            purpose="proof_model_lane_probe",
            foreground_request=True,
            health_probe=True,
            skip_runtime_payload=True,
            allow_cloud_fallback=False,
            disable_prompt_cache=True,
            clear_prompt_cache=True,
            temperature=0,
            max_tokens=24,
            num_predict=24,
        )
    except _DNU_TASK_ATTEMPT_ERRORS as exc:
        report["elapsed_s"] = time.time() - t0
        report["error"] = f"{type(exc).__name__}: {exc}"
        probe_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    text = str(metadata.get("text", "") or "")
    endpoint = str(metadata.get("endpoint", "") or "")
    endpoint_tier = _router_endpoint_tier(router, endpoint)
    nonempty_model_text_ok = bool(text.strip())
    if requested_tier == "primary":
        local_lane_ok = endpoint_tier == "local"
    else:
        local_lane_ok = endpoint_tier in {"local_fast", "emergency"}
    lane_status = {}
    recurrent_depth = {"active": False, "config": None}
    try:
        endpoint_obj = getattr(router, "endpoints", {}).get(endpoint)
        client = getattr(endpoint_obj, "client", None)
        if client is not None and hasattr(client, "get_lane_status"):
            lane_status = client.get_lane_status()
        elif client is not None and hasattr(client, "get_conversation_status"):
            lane_status = client.get_conversation_status()
        if lane_status:
            recurrent_depth = lane_status.get("recurrent_depth", recurrent_depth)
    except _DNU_RUN_RECOVERABLE_ERRORS as exc:
        lane_status = {"error": f"{type(exc).__name__}: {exc}"}

    probe_ok = bool(metadata.get("ok")) and nonempty_model_text_ok and strict_answer_ok and local_lane_ok
    report.update(
        {
            "status": "pass" if probe_ok else "fail",
            "ok": probe_ok,
            "endpoint": endpoint,
            "endpoint_tier": endpoint_tier,
            "elapsed_s": time.time() - t0,
            "strict_answer_ok": strict_answer_ok,
            "strict_answer_source": strict_answer_source,
            "structured_proof_solver_enabled": structured_solver_enabled,
            "nonempty_model_text_ok": nonempty_model_text_ok,
            "local_lane_ok": local_lane_ok,
            "lane_status": lane_status,
            "recurrent_depth": recurrent_depth,
            "error": str(metadata.get("error", "") or ""),
            "text_preview": text[:240],
        }
    )
    probe_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


async def recycle_proof_model_lane(
    router,
    requested_tier: str,
    *,
    run_dir: Path,
    reason: str,
    task_index: int,
) -> dict:
    """Recycle the local proof model worker between task checkpoints.

    The proof path remains the same canonical runtime and requested model tier.
    This only bounds resident worker growth and stale shared-memory/semaphore
    state during long primary-lane proof batteries.
    """
    event = {
        "schema": "aura.dnu_lifecycle_event.v1",
        "event": "model_lane_recycle",
        "status": "started",
        "reason": reason,
        "requested_tier": requested_tier,
        "task_index": int(task_index),
        "timestamp_unix": time.time(),
        "before": collect_proof_resource_snapshot(
            label="before_model_lane_recycle",
            task_index=task_index,
        ),
        "after": None,
        "recycled_clients": [],
        "error": None,
    }
    lifecycle_path = run_dir / "LIFECYCLE_EVENTS.jsonl"
    append_jsonl(lifecycle_path, event)

    if router is None:
        event["status"] = "failed"
        event["error"] = "llm_router_missing"
        event["after"] = collect_proof_resource_snapshot(
            label="after_model_lane_recycle_failed",
            task_index=task_index,
        )
        append_jsonl(lifecycle_path, event)
        return event

    def _lane_matches(candidate) -> bool:
        text_parts = [type(candidate).__name__]
        for getter_name in ("get_lane_status", "get_conversation_status"):
            getter = getattr(candidate, getter_name, None)
            if not callable(getter):
                continue
            try:
                status = getter() or {}
            except _DNU_RUN_RECOVERABLE_ERRORS:
                continue
            if isinstance(status, dict):
                text_parts.extend(str(value) for value in status.values())
        text = " ".join(text_parts).lower()
        if requested_tier == "primary":
            return "32b" in text or "cortex" in text or "qwen2.5-32b" in text
        return "7b" in text or "brainstem" in text or "fast" in text

    recycled = 0
    recycled_candidate: Any | None = None
    candidates: list[tuple[str, Any]] = []
    seen: set[int] = set()

    def _add_candidate(label: str, candidate) -> None:
        if candidate is None:
            return
        for obj in (
            candidate,
            getattr(candidate, "client", None),
            getattr(candidate, "_client", None),
            getattr(candidate, "_mlx_client", None),
            getattr(candidate, "mlx_client", None),
            getattr(candidate, "_local_client", None),
        ):
            if obj is None:
                continue
            ident = id(obj)
            if ident in seen:
                continue
            seen.add(ident)
            candidates.append((label, obj))

    _add_candidate("router", router)
    endpoints = getattr(router, "endpoints", {}) or {}
    for endpoint_name, endpoint in list(endpoints.items()):
        _add_candidate(str(endpoint_name), endpoint)
    try:
        from core.container import ServiceContainer

        _add_candidate("inference_gate", ServiceContainer.get("inference_gate", default=None))
    except _DNU_RUN_RECOVERABLE_ERRORS:
        pass

    for label, candidate in candidates:
        reboot = getattr(candidate, "reboot_worker", None)
        if not callable(reboot):
            continue
        if not _lane_matches(candidate):
            continue
        try:
            result = reboot(reason=reason, mark_failed=False)
        except TypeError:
            result = reboot()
        if asyncio.iscoroutine(result):
            await asyncio.wait_for(result, timeout=180.0)
        recycled += 1
        recycled_candidate = candidate
        event["recycled_clients"].append(
            {
                "owner": label,
                "client": type(candidate).__name__,
            }
        )
        break

    gc.collect()
    if recycled == 0:
        event["status"] = "failed"
        event["error"] = "no_rebootable_model_client_for_requested_tier"
    else:
        warmup = getattr(recycled_candidate, "warmup", None)
        if not callable(warmup):
            warmup = getattr(recycled_candidate, "warm_up", None)
        if callable(warmup):
            try:
                try:
                    warmup_result = warmup(
                        foreground_request=requested_tier == "primary",
                        skip_swap_cooldown=True,
                    )
                except TypeError:
                    warmup_result = warmup()
                if asyncio.iscoroutine(warmup_result):
                    await asyncio.wait_for(warmup_result, timeout=240.0)
            except _DNU_RUN_RECOVERABLE_ERRORS as exc:
                event["status"] = "failed"
                event["error"] = f"model_lane_rewarm_failed:{type(exc).__name__}: {exc}"
                event["after"] = collect_proof_resource_snapshot(
                    label="after_model_lane_recycle_failed",
                    task_index=task_index,
                )
                append_jsonl(lifecycle_path, event)
                return event
        live_check = getattr(recycled_candidate, "is_alive", None)
        lane_live = bool(callable(live_check) and live_check())
        if not lane_live:
            event["status"] = "failed"
            event["error"] = "recycled_model_lane_not_live_after_warmup"
        else:
            event["status"] = "complete"
    event["after"] = collect_proof_resource_snapshot(
        label="after_model_lane_recycle",
        task_index=task_index,
    )
    append_jsonl(lifecycle_path, event)
    return event


async def shutdown_proof_runtime(orchestrator) -> None:
    """Tear down the canonical proof boot so local model workers do not linger."""
    from core.container import ServiceContainer
    from core.runtime.shutdown_coordinator import get_shutdown_coordinator, request_shutdown

    request_shutdown("dnu_agi_proof_battery_complete")
    orchestrator_shutdown_timeout_s = max(
        60.0,
        float(os.environ.get("AURA_PROOF_ORCHESTRATOR_SHUTDOWN_TIMEOUT_S", "60.0") or 60.0),
    )

    async def _bounded_call(label: str, callback, *, timeout: float = 8.0) -> None:
        if not callable(callback):
            return
        try:
            result = callback()
            if asyncio.iscoroutine(result):
                await asyncio.wait_for(result, timeout=timeout)
        except _DNU_RUN_RECOVERABLE_ERRORS as exc:
            print(f"  [WARN] Shutdown step {label} failed or timed out: {type(exc).__name__}: {exc}")

    doctor = ServiceContainer.get("flagship_doctor_daemon", default=None)
    if doctor is not None:
        await _bounded_call("flagship_doctor_daemon.stop", getattr(doctor, "stop", None), timeout=3.0)

    router = ServiceContainer.get("llm_router", default=None)
    if router is not None and hasattr(router, "endpoints"):
        for endpoint in list(router.endpoints.values()):
            client = getattr(endpoint, "client", None)
            candidates = [client]
            lazy_client = getattr(client, "_client", None)
            if lazy_client is not None:
                candidates.append(lazy_client)
            for candidate in candidates:
                if candidate is None:
                    continue
                for close_name in ("aclose", "close", "cleanup", "on_stop"):
                    close_method = getattr(candidate, close_name, None)
                    if callable(close_method):
                        await _bounded_call(f"model_client.{close_name}", close_method, timeout=10.0)
                        break
                else:
                    close_method = None
                if close_method is not None:
                    continue
                reboot_worker = getattr(candidate, "reboot_worker", None)
                if callable(reboot_worker):
                    try:
                        await asyncio.wait_for(
                            reboot_worker(reason="dnu_proof_runtime_shutdown", mark_failed=False),
                            timeout=8.0,
                        )
                    except _DNU_RUN_RECOVERABLE_ERRORS as exc:
                        print(
                            "  [WARN] Shutdown step model_worker.reboot_worker failed or timed out: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    continue
                aclose = getattr(candidate, "aclose", None)
                await _bounded_call("model_client.aclose", aclose, timeout=5.0)

    stop_method = getattr(orchestrator, "stop", None)
    await _bounded_call(
        "orchestrator.stop",
        stop_method,
        timeout=orchestrator_shutdown_timeout_s,
    )

    try:
        await get_shutdown_coordinator().shutdown(timeout_per_phase=10.0)
    finally:
        await _reap_proof_child_processes(
            "dnu_proof_runtime_shutdown",
            include_resource_trackers=False,
        )


# ---------------------------------------------------------------------------
# Anti-Theater Checks
# ---------------------------------------------------------------------------

def anti_theater_pre_check(tasks: list[dict], grader_data: dict) -> list[str]:
    """Pre-flight anti-theater validation. Returns list of violations."""
    violations = []

    # Check 1: No task contains golden_answer
    for task in tasks:
        if "golden_answer" in task:
            violations.append(f"THEATER: Task {task.get('task_id', '?')} contains golden_answer in task pack")

    # Check 2: Grader salts exist for all tasks
    for task in tasks:
        tid = task.get("task_id", "")
        if tid not in grader_data:
            violations.append(f"INTEGRITY: Task {tid} missing from grader salts")

    # Check 3: All hashes are valid SHA-256 hex
    for tid, entry in grader_data.items():
        h = entry.get("answer_hash", "")
        if not re.match(r"^[0-9a-f]{64}$", h):
            violations.append(f"INTEGRITY: Invalid hash format for {tid}: {h}")

    return violations


def anti_theater_post_check(results: list[dict]) -> list[str]:
    """Post-execution anti-theater validation. Returns list of violations."""
    violations = []

    # Check: No result has a score that wasn't computed from actual execution
    for r in results:
        if r.get("status") == "pass" and not r.get("response_text"):
            violations.append(f"THEATER: Task {r.get('task_id', '?')} marked pass but has no response text")

    # Check: neither the battery nor the evaluated path should need numerical
    # projection libraries for score computation. We ensure numpy is not imported
    # or referenced in the runner's namespace itself to prevent false positives
    # from CognitiveEngine's own authentic internal modules importing it under the hood.
    if "numpy" in globals() or "np" in globals():
        violations.append("THEATER: numpy directly imported in battery runner namespace")

    return violations


# ---------------------------------------------------------------------------
# Task Loading
# ---------------------------------------------------------------------------

TASK_CATEGORIES = ["reasoning", "coding", "planning", "self_debug", "transfer", "research"]
DIR_TO_CAT = {
    "reasoning": "novel_reasoning",
    "coding": "coding",
    "planning": "planning",
    "self_debug": "self_debug",
    "transfer": "transfer",
    "research": "research",
}
MINIMUM_COUNTS = {
    "novel_reasoning": 50,
    "coding": 10,
    "planning": 5,
    "self_debug": 5,
    "transfer": 10,
    "research": 10,
}

def load_task_packs(fixture_dir: Path) -> tuple[list[dict], dict]:
    """Load all task packs and grader salts. Returns (tasks, grader_data)."""
    all_tasks = []

    for category in TASK_CATEGORIES:
        cat_dir = fixture_dir / category
        tasks_file = cat_dir / "tasks.json"
        if not tasks_file.exists():
            print(f"  [WARN] Task file not found: {tasks_file}")
            continue

        tasks = json.loads(tasks_file.read_text(encoding="utf-8"))
        if isinstance(tasks, list):
            for t in tasks:
                t.setdefault("category", category)
            all_tasks.extend(tasks)
            print(f"  [OK] Loaded {len(tasks)} tasks from {category}/")
        else:
            print(f"  [WARN] Invalid task format in {tasks_file}")

    # Load grader salts from ALL salt files
    grader_data = {}
    for salt_file in fixture_dir.glob(".grader_salts*.json"):
        try:
            data = json.loads(salt_file.read_text(encoding="utf-8"))
            grader_data.update(data)
            print(f"  [OK] Loaded {len(data)} grader entries from {salt_file.name}")
        except (OSError, json.JSONDecodeError) as e:
            print(f"  [WARN] Failed to load {salt_file}: {e}")

    return all_tasks, grader_data


# ---------------------------------------------------------------------------
# Baselines & Ablations Utilities
# ---------------------------------------------------------------------------

def _baseline_timeout_seconds() -> float:
    """Bound comparison baselines so final-proof cannot hang behind Aura's live run."""
    raw = os.environ.get("AURA_DNU_BASELINE_TIMEOUT_SECONDS", "90")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 90.0
    return min(180.0, max(30.0, value))


DNU_BASELINE_MAX_TOKENS = 160


def _iter_router_generation_clients(router):
    """Yield router/client objects that may own an active baseline generation."""

    yielded: set[int] = set()

    def _yield(candidate):
        if candidate is None:
            return []
        objects = [candidate]
        for attr in ("client", "_client", "_mlx_client"):
            nested = getattr(candidate, attr, None)
            if nested is not None:
                objects.append(nested)
        fresh = []
        for obj in objects:
            ident = id(obj)
            if ident in yielded:
                continue
            yielded.add(ident)
            fresh.append(obj)
        return fresh

    for obj in _yield(router):
        yield obj

    endpoints = getattr(router, "endpoints", None)
    if isinstance(endpoints, dict):
        endpoint_iter = endpoints.values()
    elif isinstance(endpoints, (list, tuple, set)):
        endpoint_iter = endpoints
    else:
        endpoint_iter = ()

    for endpoint in endpoint_iter:
        for obj in _yield(endpoint):
            yield obj


def _force_abort_router_generation(router, *, reason: str) -> int:
    """Best-effort emergency abort for a router/client stuck past cancellation."""

    aborted = 0
    for client in _iter_router_generation_clients(router):
        abort = getattr(client, "force_abort_active_generation", None)
        if not callable(abort):
            continue
        try:
            if abort(reason=reason):
                aborted += 1
        except _DNU_RUN_RECOVERABLE_ERRORS as exc:
            print(f"  [WARN] Baseline watchdog abort skipped for one client: {exc}", flush=True)
    return aborted


async def _recover_router_after_baseline_abort(router, *, reason: str) -> int:
    """Recover clients that accepted an emergency baseline abort."""

    recovered = 0
    for client in _iter_router_generation_clients(router):
        reboot = getattr(client, "reboot_worker", None)
        if not callable(reboot):
            continue
        try:
            try:
                result = reboot(reason=f"baseline_abort_recovery:{reason}", mark_failed=False)
            except TypeError:
                result = reboot()
            if asyncio.iscoroutine(result):
                await result
            recovered += 1
        except _DNU_RUN_RECOVERABLE_ERRORS as exc:
            print(f"  [WARN] Baseline watchdog recovery skipped for one client: {exc}", flush=True)
    return recovered


async def _cancel_baseline_task(task: asyncio.Task) -> bool:
    """Cancel a timed-out comparison baseline without killing the live model lane."""

    if task.done():
        return True
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=3.0)
        return True
    except (asyncio.CancelledError, asyncio.TimeoutError, TimeoutError):
        return task.done()


async def _generate_baseline_response(
    router,
    *,
    prompt: str,
    system_prompt: str,
    purpose: str,
) -> str:
    """Run a baseline model call with bounded, non-destructive isolation.

    The comparison still uses the requested live 32B lane, but a baseline
    timeout must not kill or reboot Aura's shared foreground-capable worker.
    Timed-out baselines are recorded as baseline failures while the runtime
    remains available for the proof battery and normal launch path.
    """
    timeout_s = _baseline_timeout_seconds()
    loop = asyncio.get_running_loop()
    task = loop.create_task(
        router.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            origin="baseline",
            purpose=purpose,
            benchmark_request=True,
            foreground_request=False,
            skip_runtime_payload=True,
            timeout=timeout_s,
            prefer_tier=os.environ.get("AURA_PROOF_MODEL_TIER", "primary"),
            proof_primary_lane_required=True,
            allow_cloud_fallback=False,
            disable_prompt_cache=True,
            clear_prompt_cache=True,
            temperature=0.15,
            temp=0.15,
            top_p=0.85,
            min_p=0.03,
            repetition_penalty=1.35,
            repetition_context_size=1024,
            stop_sequences=["\n\n", "\\n", "User:", "Assistant:", "<|im_end|>", "<|endoftext|>"],
            max_tokens=DNU_BASELINE_MAX_TOKENS,
            num_predict=DNU_BASELINE_MAX_TOKENS,
        ),
        name=f"dnu_baseline:{purpose}",
    )
    reason = f"{purpose}_hard_timeout_{timeout_s:.0f}s"
    watchdog_fired = threading.Event()

    def _watchdog_abort() -> None:
        watchdog_fired.set()

    watchdog = threading.Timer(timeout_s, _watchdog_abort)
    watchdog.daemon = True
    watchdog.start()
    try:
        return await asyncio.wait_for(task, timeout=timeout_s)
    except (asyncio.TimeoutError, TimeoutError) as exc:
        cancelled = await _cancel_baseline_task(task)
        if not cancelled:
            abort_reason = reason if watchdog_fired.is_set() else f"{reason}_cancel_stuck"
            aborted = _force_abort_router_generation(router, reason=abort_reason)
            if aborted:
                await _recover_router_after_baseline_abort(router, reason=abort_reason)
        print(
            f"  [WARN] Baseline timed out cooperatively for {purpose} "
            f"after {timeout_s:.0f}s; shared model lane preserved.",
            flush=True,
        )
        raise TimeoutError(reason) from exc
    finally:
        watchdog.cancel()


@contextlib.contextmanager
def lesion_services(names: list[str]):
    """Dynamically unregister or replace services in ServiceContainer."""
    from core.container import ServiceContainer, ServiceDescriptor, ServiceLifetime
    from core.runtime.ablation_policy import mark_services_lesioned

    original = {}
    with mark_services_lesioned(names):
        with ServiceContainer._lock:
            for name in names:
                resolved_name = ServiceContainer._resolve_name(name)
                if resolved_name in ServiceContainer._services:
                    original[resolved_name] = ServiceContainer._services[resolved_name]
                    # Replace with a descriptor that returns None.
                    ServiceContainer._services[resolved_name] = ServiceDescriptor(
                        name=resolved_name,
                        factory=lambda *args, **kwargs: None,
                        lifetime=ServiceLifetime.SINGLETON,
                        instance=None,
                        required=False,
                        initialized=True,
                    )
        try:
            yield
        finally:
            # Restore original descriptors.
            with ServiceContainer._lock:
                for resolved_name, desc in original.items():
                    ServiceContainer._services[resolved_name] = desc


async def execute_raw_llm_task(router, task: dict, grader_data: dict, sem: asyncio.Semaphore) -> dict:
    task_id = task.get("task_id", "unknown")
    prompt = task.get("task_prompt", "")
    system_prompt = (
        "You are a helpful assistant. Solve the user's problem. "
        "Think step-by-step. Put your final answer strictly inside <answer>...</answer> tags. "
        "For example, <answer>Alice</answer> or <answer>5</answer>."
    )
    result = {
        "task_id": task_id,
        "category": task.get("category", "unknown"),
        "difficulty": task.get("difficulty", "unknown"),
        "status": "error",
        "response_text": "",
        "extracted_answer": None,
        "normalized_answer": None,
        "answer_hash": None,
        "elapsed_s": 0.0,
        "error": None,
    }
    t0 = time.time()
    try:
        async with sem:
            response = await _generate_baseline_response(
                router,
                prompt=prompt,
                system_prompt=system_prompt,
                purpose="raw_llm_baseline",
            )
        result["response_text"] = response
        result["elapsed_s"] = time.time() - t0
        result["status"] = "success"

        extracted = extract_answer_tag(response)
        if extracted:
            result["extracted_answer"] = extracted
            result["normalized_answer"] = normalize_answer(extracted)
        else:
            if len(response.strip()) < 200:
                result["extracted_answer"] = response.strip()
                result["normalized_answer"] = normalize_answer(response)
            else:
                result["status"] = "no_answer"
                result["error"] = "No <answer> tags found in response"
    except TimeoutError:
        result["status"] = "timeout"
        result["error"] = f"Task exceeded baseline time budget of {_baseline_timeout_seconds():.0f}s"
        result["elapsed_s"] = time.time() - t0
    except _DNU_RUN_RECOVERABLE_ERRORS as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {str(e)}"
        result["elapsed_s"] = time.time() - t0

    # Grade the result
    result = grade_result(result, grader_data)
    return result


async def execute_react_task(router, task: dict, grader_data: dict, sem: asyncio.Semaphore) -> dict:
    task_id = task.get("task_id", "unknown")
    prompt = task.get("task_prompt", "")
    system_prompt = (
        "You are a ReAct reasoning agent. Solve the task step-by-step by generating "
        "Thought, Action, Observation steps. You do not have actual tool access, so you should "
        "generate the Actions and the corresponding Observations yourself to structure your thinking. "
        "Finally, wrap your final answer strictly inside <answer>...</answer> tags."
    )
    result = {
        "task_id": task_id,
        "category": task.get("category", "unknown"),
        "difficulty": task.get("difficulty", "unknown"),
        "status": "error",
        "response_text": "",
        "extracted_answer": None,
        "normalized_answer": None,
        "answer_hash": None,
        "elapsed_s": 0.0,
        "error": None,
    }
    t0 = time.time()
    try:
        async with sem:
            response = await _generate_baseline_response(
                router,
                prompt=prompt,
                system_prompt=system_prompt,
                purpose="react_agent_baseline",
            )
        result["response_text"] = response
        result["elapsed_s"] = time.time() - t0
        result["status"] = "success"

        extracted = extract_answer_tag(response)
        if extracted:
            result["extracted_answer"] = extracted
            result["normalized_answer"] = normalize_answer(extracted)
        else:
            if len(response.strip()) < 200:
                result["extracted_answer"] = response.strip()
                result["normalized_answer"] = normalize_answer(response)
            else:
                result["status"] = "no_answer"
                result["error"] = "No <answer> tags found in response"
    except TimeoutError:
        result["status"] = "timeout"
        result["error"] = f"Task exceeded baseline time budget of {_baseline_timeout_seconds():.0f}s"
        result["elapsed_s"] = time.time() - t0
    except _DNU_RUN_RECOVERABLE_ERRORS as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {str(e)}"
        result["elapsed_s"] = time.time() - t0

    # Grade the result
    result = grade_result(result, grader_data)
    return result


async def execute_llm_with_tools_task(router, task: dict, grader_data: dict, sem: asyncio.Semaphore) -> dict:
    task_id = task.get("task_id", "unknown")
    prompt = task.get("task_prompt", "")
    system_prompt = (
        "You are an agent equipped with direct tools. Solve the user's problem step-by-step.\n\n"
        "Available Tools:\n"
        "- `read_file(path)`: Read file contents.\n"
        "- `write_file(path, content)`: Write file contents.\n"
        "- `execute_command(cmd)`: Execute a shell command.\n"
        "- `web_search(query)`: Search the web.\n"
        "- `read_web_page(url)`: Read web page content.\n\n"
        "You may invoke tools by printing: `Tool Call: <tool_name>(<args>)`. "
        "The system will return simulated success. "
        "Finally, think step-by-step and wrap your final answer strictly inside <answer>...</answer> tags."
    )
    result = {
        "task_id": task_id,
        "category": task.get("category", "unknown"),
        "difficulty": task.get("difficulty", "unknown"),
        "status": "error",
        "response_text": "",
        "extracted_answer": None,
        "normalized_answer": None,
        "answer_hash": None,
        "elapsed_s": 0.0,
        "error": None,
    }
    t0 = time.time()
    try:
        async with sem:
            response = await _generate_baseline_response(
                router,
                prompt=prompt,
                system_prompt=system_prompt,
                purpose="llm_with_tools_baseline",
            )
        result["response_text"] = response
        result["elapsed_s"] = time.time() - t0
        result["status"] = "success"

        extracted = extract_answer_tag(response)
        if extracted:
            result["extracted_answer"] = extracted
            result["normalized_answer"] = normalize_answer(extracted)
        else:
            if len(response.strip()) < 200:
                result["extracted_answer"] = response.strip()
                result["normalized_answer"] = normalize_answer(response)
            else:
                result["status"] = "no_answer"
                result["error"] = "No <answer> tags found in response"
    except TimeoutError:
        result["status"] = "timeout"
        result["error"] = f"Task exceeded baseline time budget of {_baseline_timeout_seconds():.0f}s"
        result["elapsed_s"] = time.time() - t0
    except _DNU_RUN_RECOVERABLE_ERRORS as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {str(e)}"
        result["elapsed_s"] = time.time() - t0

    # Grade the result
    result = grade_result(result, grader_data)
    return result


async def run_ablation_suite(runtime, tasks: list[dict], grader_data: dict, services_to_lesion: list[str]) -> tuple[float, bool]:
    from core.container import ServiceContainer
    
    # Programmatically verify the lesion is active and behaviorally load-bearing.
    lesion_verified = True
    services_disabled: set[str] = set()
    ablation_results = []
    with lesion_services(services_to_lesion):
        for s_name in services_to_lesion:
            try:
                inst = ServiceContainer.get(s_name, default=None)
                if inst is not None:
                    print(f"  [ERROR] Lesion failed to verify for: {s_name} (got {inst})")
                    lesion_verified = False
                else:
                    services_disabled.add(s_name)
            except Exception:
                # Exception means the service cannot be successfully fetched/resolved.
                services_disabled.add(s_name)

        for task in tasks:
            try:
                await isolate_live_runtime_for_dnu_task(task)
            except _DNU_RUN_RECOVERABLE_ERRORS as exc:
                print(f"  [WARN] Failed to reset state for ablation isolation: {exc}")
            res = await execute_task(runtime, task, timeout_s=max(240, task.get("time_budget_s", 240)))
            res = grade_result(res, grader_data)
            ablation_results.append(res)
    scorecard = compute_scorecard(ablation_results)
    pass_rate = scorecard["overall_pass_rate"]
    behavior_degraded = pass_rate < 1.0
    if not behavior_degraded:
        print(
            "  [ERROR] Ablation did not degrade behavior; disabled services were not load-bearing: "
            f"{sorted(services_disabled)}"
        )
    return pass_rate, lesion_verified and bool(services_disabled) and behavior_degraded


def _scrub_dnu_state_for_task(state, task: dict):
    """Clear turn-local proof residue while preserving the canonical runtime state shape."""

    from core.runtime.proof_policy import clear_transient_response_modifiers

    prompt = str(task.get("task_prompt", "") or "")
    task_id = str(task.get("task_id", "unknown") or "unknown")
    cognition = getattr(state, "cognition", None)
    if cognition is not None:
        cognition.working_memory = []
        cognition.long_term_memory = []
        cognition.rolling_summary = ""
        cognition.current_objective = None
        cognition.current_origin = PROOF_LIVE_MESSAGE_ORIGIN
        cognition.attention_focus = ""
        cognition.last_response = None
        cognition.discourse_topic = None
        cognition.discourse_branches = []
        if hasattr(cognition, "active_goals"):
            cognition.active_goals = []
        if hasattr(cognition, "pending_intents"):
            cognition.pending_intents = []
        if hasattr(cognition, "pending_initiatives"):
            cognition.pending_initiatives = []
        if hasattr(cognition, "phenomenal_state"):
            cognition.phenomenal_state = ""
        if hasattr(cognition, "modifiers"):
            cognition.modifiers = {}
    if not isinstance(getattr(state, "response_modifiers", None), dict):
        state.response_modifiers = {}
    clear_transient_response_modifiers(state.response_modifiers, strict=True)
    state.response_modifiers["proof_task_id"] = task_id
    state.response_modifiers["proof_task_prompt_hash"] = hashlib.sha256(prompt.encode()).hexdigest()
    return state


async def isolate_live_runtime_for_dnu_task(task: dict) -> None:
    """Reset live proof-turn state across repository and kernel before each DNU task."""

    from core.container import ServiceContainer

    state_repo = ServiceContainer.get("state_repository", default=None)
    if state_repo:
        state = await state_repo.get_current()
        if state:
            derived = state.derive("task_isolation_reset", origin="dnu_agi_proof_battery")
            _scrub_dnu_state_for_task(derived, task)
            current = getattr(state_repo, "_current", None)
            current_version = int(getattr(current, "version", 0) or 0)
            if int(getattr(derived, "version", 0) or 0) <= current_version:
                derived.version = current_version + 1
                derived.parent_state_id = getattr(current, "state_id", getattr(derived, "parent_state_id", None))
            await state_repo.commit(derived, "task_isolation_reset")

    try:
        from core.kernel.kernel_interface import KernelInterface

        ki = KernelInterface.get_instance()
        kernel = getattr(ki, "_kernel", None)
        kernel_state = getattr(kernel, "state", None)
        if kernel is not None and kernel_state is not None:
            derived = kernel_state.derive(
                "dnu_kernel_task_isolation",
                origin="dnu_agi_proof_battery",
            )
            kernel.state = _scrub_dnu_state_for_task(derived, task)
    except _DNU_RUN_RECOVERABLE_ERRORS:
        raise


# ---------------------------------------------------------------------------
# Task Execution
# ---------------------------------------------------------------------------

async def execute_task(runtime, task: dict, timeout_s: int = 240) -> dict:
    """Execute a single task through the canonical live message path and return result."""
    task_id = task.get("task_id", "unknown")
    prompt = task.get("task_prompt", "")
    budget = max(240, task.get("time_budget_s", timeout_s))

    result = {
        "task_id": task_id,
        "category": task.get("category", "unknown"),
        "difficulty": task.get("difficulty", "unknown"),
        "status": "error",
        "response_text": "",
        "extracted_answer": None,
        "normalized_answer": None,
        "answer_hash": None,
        "elapsed_s": 0.0,
        "error": None,
    }

    t0 = time.time()
    
    async def _run_live_path() -> str:
        if hasattr(runtime, "process_user_input_priority"):
            if hasattr(runtime, "_last_emitted_fingerprint"):
                runtime._last_emitted_fingerprint = ""
            response = await runtime.process_user_input_priority(
                prompt,
                origin=PROOF_LIVE_MESSAGE_ORIGIN,
                timeout_sec=float(budget),
            )
            return str(response or "")
        thought = await runtime.think(
            objective=prompt,
            origin="test",
            prefer_tier=os.environ.get("AURA_PROOF_MODEL_TIER", "primary"),
        )
        return str(getattr(thought, "content", "") or "")

    def _is_non_answer(text: str) -> bool:
        if not str(text or "").strip():
            return True
        try:
            from core.conversation.response_reliability import is_non_answer_repair_floor_reply

            if is_non_answer_repair_floor_reply(text):
                return True
        except _DNU_RUN_RECOVERABLE_ERRORS:
            pass
        return False

    # Milestone 1: Soft timeout (200s) and one live-path recovery retry.
    soft_budget = min(200, int(budget * 0.85))
    try:
        response_text = await asyncio.wait_for(_run_live_path(), timeout=soft_budget)
        if _is_non_answer(response_text):
            raise RuntimeError("live_path_returned_no_answer")
    except _DNU_TASK_ATTEMPT_ERRORS as exc:
        print(f"\n  [WARN] First attempt for {task_id} failed or soft-timed out ({type(exc).__name__}). Retrying through live path...", end="", flush=True)
        try:
            response_text = await asyncio.wait_for(
                _run_live_path(),
                timeout=max(15, budget - (time.time() - t0) - 2)
            )
            if _is_non_answer(response_text):
                raise RuntimeError("live_path_returned_no_answer")
        except Exception as retry_exc:
            result["status"] = "timeout" if isinstance(retry_exc, asyncio.TimeoutError) else "error"
            result["error"] = f"Retry failed: {type(retry_exc).__name__}: {str(retry_exc)}"
            result["elapsed_s"] = time.time() - t0
            return result

    result["response_text"] = response_text
    result["elapsed_s"] = time.time() - t0
    result["status"] = "success"
    result["answer_source"] = "model_or_runtime"
    result["structured_proof_solver"] = None
    try:
        from core.container import ServiceContainer

        state_repo = ServiceContainer.get("state_repository", default=None)
        state = await state_repo.get_current() if state_repo else None
        modifiers = getattr(state, "response_modifiers", {}) if state is not None else {}
        solver_payload = (
            modifiers.get("structured_proof_solver")
            if isinstance(modifiers, dict)
            else None
        )
        if solver_payload:
            result["answer_source"] = "structured_proof_solver"
            result["structured_proof_solver"] = solver_payload
    except _DNU_RUN_RECOVERABLE_ERRORS as exc:
        result["answer_source_error"] = f"{type(exc).__name__}: {exc}"

    # Extract answer from <answer> tags
    extracted = extract_answer_tag(result["response_text"])
    if extracted:
        result["extracted_answer"] = extracted
        result["normalized_answer"] = normalize_answer(extracted)
    else:
        result["status"] = "no_answer"
        result["error"] = "No <answer> tags found in response"

    return result


def grade_result(result: dict, grader_data: dict) -> dict:
    """Grade a task result against the salted answer hash."""
    task_id = result["task_id"]

    if task_id not in grader_data:
        result["status"] = "ungraded"
        result["error"] = f"No grader entry for {task_id}"
        return result

    if result["status"] in ("timeout", "error"):
        return result

    if result["normalized_answer"] is None:
        result["status"] = "no_answer"
        return result

    entry = grader_data[task_id]
    salt = entry["salt"]
    expected_hash = entry["answer_hash"]

    computed_hash = hash_answer(salt, result["normalized_answer"])
    result["answer_hash"] = computed_hash

    if computed_hash == expected_hash:
        result["status"] = "pass"
    else:
        result["status"] = "fail"
        # Don't leak the golden answer - just note the hash mismatch
        result["error"] = f"Hash mismatch: computed {computed_hash[:16]}... != expected {expected_hash[:16]}..."

    return result


def _read_jsonl(path: Path) -> tuple[list[dict], int]:
    records: list[dict] = []
    invalid = 0
    if not path.exists():
        return records, invalid
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
            else:
                invalid += 1
        except json.JSONDecodeError:
            invalid += 1
    return records, invalid


def build_governance_report(receipts_file: Path, *, expected_tasks: int) -> dict:
    """Build an evidence-backed governance report from DNU receipts and live negatives."""

    records, invalid_lines = _read_jsonl(receipts_file)
    dnu_records = [
        record for record in records if record.get("source") == "dnu_agi_proof_battery"
    ]
    receipt_count = sum(1 for record in records if str(record.get("receipt_id", "")).startswith("will_"))
    pre_action_count = sum(
        1 for record in dnu_records if record.get("authorization_phase") == "pre_action"
    )
    effect_proof_count = sum(
        1
        for record in dnu_records
        if isinstance(record.get("effect_verified"), bool)
        and isinstance(record.get("telemetry_logged"), bool)
        and record.get("closure_verified") is True
    )
    pre_action_missing = max(0, expected_tasks - pre_action_count)
    missing_effect_proof = max(0, expected_tasks - effect_proof_count)
    invalid_receipts = invalid_lines + sum(
        1 for record in records if not str(record.get("receipt_id", "")).startswith("will_")
    )

    negative_tests: dict[str, bool]
    try:
        from tools.receipt_coverage_validator import run_negative_tests

        negative_tests = run_negative_tests()
    except _DNU_RUN_RECOVERABLE_ERRORS as exc:
        negative_tests = {
            "negative_test_harness_executed": False,
            "error": False,
            "error_detail": f"{type(exc).__name__}: {exc}",
        }

    negative_tests_passed = all(value is True for value in negative_tests.values())
    surface_counts: dict[str, int] = {}
    for record in records:
        domain = str(record.get("domain", "unknown") or "unknown")
        surface_counts[domain] = surface_counts.get(domain, 0) + 1

    status = (
        "pass"
        if (
            receipt_count > 0
            and invalid_receipts == 0
            and pre_action_missing == 0
            and missing_effect_proof == 0
            and negative_tests_passed
        )
        else "fail"
    )
    return {
        "schema": "aura.dnu_governance_report.v2",
        "generated_by": "tools/agi/run_dnu_agi_proof_battery.py::build_governance_report",
        "status": status,
        "receipt_count": receipt_count,
        "dnu_pre_action_receipt_count": pre_action_count,
        "expected_dnu_tasks": expected_tasks,
        "pre_action_authorization_missing": pre_action_missing,
        "missing_effect_proof_count": missing_effect_proof,
        "invalid_receipts": invalid_receipts,
        "surface_counts": surface_counts,
        "negative_tests": negative_tests,
        "negative_tests_passed": negative_tests_passed,
        "bypass_count": pre_action_missing + missing_effect_proof + invalid_receipts,
        "forged_receipt_result": (
            "pass" if negative_tests.get("forged_receipt_rejected") is True else "fail"
        ),
        "missing_effect_proof_result": (
            "pass" if negative_tests.get("missing_effect_proof_rejected") is True else "fail"
        ),
        "disabled_will_result": (
            "pass" if negative_tests.get("disabled_will_blocks_action") is True else "fail"
        ),
    }


def build_leakage_report(
    *,
    pre_violations: list[str],
    post_violations: list[str],
    run_dir: Path,
) -> dict:
    """Build an anti-theater/proof-integrity report from actual scans."""

    try:
        from tools.proof_integrity_lint import run_lint

        proof_integrity = run_lint(PROJECT_ROOT, "production")
    except _DNU_RUN_RECOVERABLE_ERRORS as exc:
        proof_integrity = {
            "passed": False,
            "findings": [
                {
                    "kind": "proof_integrity_lint_error",
                    "file": "tools/proof_integrity_lint.py",
                    "line": 0,
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            ],
        }

    task_trace = run_dir / "TASK_TRACE.jsonl"
    trace_records, trace_invalid = _read_jsonl(task_trace)
    structured_solver_tasks = [
        record.get("task_id")
        for record in trace_records
        if record.get("answer_source") == "structured_proof_solver"
    ]
    status = (
        "pass"
        if (
            not pre_violations
            and not post_violations
            and proof_integrity.get("passed") is True
            and trace_invalid == 0
        )
        else "fail"
    )
    return {
        "schema": "aura.dnu_leakage_report.v2",
        "generated_by": "tools/agi/run_dnu_agi_proof_battery.py::build_leakage_report",
        "status": status,
        "answer_leak_result": "pass" if not pre_violations else "fail",
        "salt_leak_result": "pass" if not pre_violations else "fail",
        "hidden_test_leak_result": "pass" if proof_integrity.get("passed") is True else "fail",
        "grader_leak_result": "pass" if not pre_violations else "fail",
        "canary_result": "pass" if not post_violations else "fail",
        "pre_check_violations": pre_violations,
        "post_check_violations": post_violations,
        "proof_integrity_lint": proof_integrity,
        "trace_invalid_lines": trace_invalid,
        "structured_solver_task_count": len(structured_solver_tasks),
        "structured_solver_tasks": structured_solver_tasks,
    }


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def compute_scorecard(results: list[dict]) -> dict:
    """Compute scorecard from actual results. No synthetic scores."""
    scorecard = {
        "total_tasks": len(results),
        "total_pass": sum(1 for r in results if r["status"] == "pass"),
        "total_fail": sum(1 for r in results if r["status"] == "fail"),
        "total_timeout": sum(1 for r in results if r["status"] == "timeout"),
        "total_error": sum(1 for r in results if r["status"] == "error"),
        "total_no_answer": sum(1 for r in results if r["status"] == "no_answer"),
        "total_ungraded": sum(1 for r in results if r["status"] == "ungraded"),
        "categories": {},
    }

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"total": 0, "pass": 0, "fail": 0, "timeout": 0, "error": 0, "no_answer": 0}
        categories[cat]["total"] += 1
        if r["status"] in categories[cat]:
            categories[cat][r["status"]] += 1

    for cat, stats in categories.items():
        attempted = stats["total"]
        passed = stats["pass"]
        rate = passed / attempted if attempted > 0 else 0.0
        scorecard["categories"][cat] = {
            "attempted": attempted,
            "passed": passed,
            "failed": stats["fail"],
            "timed_out": stats["timeout"],
            "errors": stats["error"],
            "no_answer": stats["no_answer"],
            "pass_rate": round(rate, 4),
        }

    # Overall pass rate
    attempted = scorecard["total_tasks"]
    passed = scorecard["total_pass"]
    scorecard["overall_pass_rate"] = round(passed / attempted, 4) if attempted > 0 else 0.0

    return scorecard


def assign_tier(pass_rate: float, has_unsupported_claims: bool = False) -> dict:
    """Assign tier strictly from pass rate. No inflation. Max cap is Tier 5: Expert."""
    if pass_rate <= 0.0:
        base_tier = 0
        label = "No Capability"
    elif pass_rate <= 0.20:
        base_tier = 1
        label = "Minimal"
    elif pass_rate <= 0.40:
        base_tier = 2
        label = "Emergent"
    elif pass_rate <= 0.60:
        base_tier = 3
        label = "Competent"
    elif pass_rate <= 0.80:
        base_tier = 4
        label = "Proficient"
    else:
        # Cap strictly at Tier 5: Expert as a proof-bearing AGI-candidate architecture
        # to respect the scientific boundary of unproven metaphysical or AGI claims.
        base_tier = 5
        label = "Expert"

    if has_unsupported_claims and base_tier > 2:
        return {"tier": 2, "label": "Emergent (Capped)", "pass_rate": pass_rate}
    return {"tier": base_tier, "label": label, "pass_rate": pass_rate}


def generate_markdown_report(
    sys_info: dict,
    scorecard: dict,
    tier: dict,
    anti_theater: dict,
    results: list[dict],
    baselines: dict,
    ablations: dict,
) -> str:
    """Generate human-readable markdown report."""
    lines = []
    lines.append("# DNU AGI Proof Battery Report")
    lines.append("")
    lines.append(f"**Run ID:** `{sys_info['run_id']}`")
    lines.append(f"**Timestamp:** `{sys_info['timestamp']}`")
    lines.append(f"**Commit SHA:** `{sys_info['commit_sha']}`")
    lines.append(f"**Platform:** `{sys_info['platform']}`")
    lines.append(f"**Python:** `{sys_info['python_version']}`")
    lines.append("")

    # Tier
    lines.append("## Assigned Tier")
    lines.append("")
    lines.append(f"**Tier {tier['tier']}: {tier['label']}** (Overall Pass Rate: {tier['pass_rate']:.1%})")
    lines.append("")

    # Anti-Theater Status
    lines.append("## Anti-Theater Controls")
    lines.append("")
    pre = anti_theater.get("pre_check_violations", [])
    post = anti_theater.get("post_check_violations", [])
    if not pre and not post:
        lines.append("✅ All anti-theater checks passed. No synthetic scores detected.")
    else:
        lines.append("⚠️ **Anti-theater violations detected:**")
        for v in pre + post:
            lines.append(f"- {v}")
    lines.append("")

    # Scorecard
    lines.append("## Scorecard")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Tasks | {scorecard['total_tasks']} |")
    lines.append(f"| Passed | {scorecard['total_pass']} |")
    lines.append(f"| Failed | {scorecard['total_fail']} |")
    lines.append(f"| Timed Out | {scorecard['total_timeout']} |")
    lines.append(f"| Errors | {scorecard['total_error']} |")
    lines.append(f"| No Answer | {scorecard['total_no_answer']} |")
    lines.append(f"| **Overall Pass Rate** | **{scorecard['overall_pass_rate']:.1%}** |")
    lines.append("")

    # Per-category breakdown
    lines.append("## Category Breakdown")
    lines.append("")
    lines.append("| Category | Attempted | Passed | Failed | Timeout | Pass Rate |")
    lines.append("|----------|-----------|--------|--------|---------|-----------|")
    for cat, stats in sorted(scorecard["categories"].items()):
        lines.append(
            f"| {cat} | {stats['attempted']} | {stats['passed']} | "
            f"{stats['failed']} | {stats['timed_out']} | {stats['pass_rate']:.1%} |"
        )
    lines.append("")

    # Baselines
    lines.append("## Baselines")
    lines.append("")
    lines.append("| Baseline | Status | Pass Rate / Notes |")
    lines.append("|----------|--------|-------------------|")
    for name, data in baselines.items():
        status = data.get("status", "NOT_RUN")
        if status == "RUN":
            pr = data.get("pass_rate", 0.0)
            lines.append(f"| {name} | RUN | {pr:.1%} pass rate ({data.get('passed')}/{data.get('total_tasks')}) |")
        else:
            lines.append(f"| {name} | NOT_RUN | {data.get('reason', 'N/A')} |")
    lines.append("")

    # Ablations
    lines.append("## Ablations")
    lines.append("")
    lines.append("| Configuration | Status | Pass Rate / Notes |")
    lines.append("|---------------|--------|-------------------|")
    for name, data in ablations.items():
        status = data.get("status", "NOT_RUN")
        if status == "RUN":
            pr = data.get("pass_rate", 0.0)
            verified = "Yes" if data.get("lesion_effect_verified", False) else "No"
            # Keep full_aura as N/A since it has no lesion
            if name == "full_aura":
                verified = "N/A"
            lines.append(f"| {name} | RUN | {pr:.1%} pass rate (Lesion Verified: {verified}) |")
        else:
            lines.append(f"| {name} | NOT_RUN | {data.get('reason', 'N/A')} |")
    lines.append("")

    # Failed Tasks Sample
    failures = [r for r in results if r["status"] in ("fail", "error", "timeout", "no_answer")]
    if failures:
        lines.append("## Failed Tasks (Sample)")
        lines.append("")
        for f in failures[:20]:  # Cap at 20
            lines.append(f"- **{f['task_id']}** ({f['category']}): {f['status']} — {f.get('error', 'N/A')}")
        if len(failures) > 20:
            lines.append(f"- ... and {len(failures) - 20} more failures")
        lines.append("")

    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append(f"# Commit: {sys_info['commit_sha']}")
    lines.append(f"# Python: {sys_info['python_version']}")
    lines.append("python tools/agi/run_dnu_agi_proof_battery.py")
    lines.append("```")
    lines.append("")

    lines.append("---")
    lines.append("*This report was generated by the DNU AGI Proof Battery Runner.*")
    lines.append("*All scores are computed from actual task execution. No synthetic projections.*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="DNU AGI Proof Battery")
    parser.add_argument("--full", action="store_true", help="Run full battery")
    parser.add_argument("--out", default="", help="Output directory")
    parser.add_argument("--smoke", action="store_true", help="Smoke run")
    parser.add_argument(
        "--model-tier",
        choices=("primary", "tertiary"),
        default=None,
        help=(
            "Proof model lane. Defaults to primary/32B Cortex for acceptance; "
            "use tertiary/7B only for diagnostic fast-lane isolation."
        ),
    )
    parser.add_argument(
        "--stop-existing-runtime",
        action="store_true",
        help="Stop any already running Aura runtime before booting the exclusive proof profile.",
    )
    parser.add_argument(
        "--allow-coexisting-runtime",
        action="store_true",
        help="Diagnostic escape hatch only: allow proof boot while another Aura runtime is running.",
    )
    parser.add_argument(
        "--disable-structured-proof-solver",
        action="store_true",
        help=(
            "Force strict proof tasks through the requested model lane instead of "
            "allowing Aura's symbolic proof solver to answer directly."
        ),
    )
    # ignore unknown args to prevent failing on extra options
    args, unknown = parser.parse_known_args()

    print("=" * 60)
    print("         DNU AGI PROOF BATTERY RUNNER")
    print("         No Synthetic Scores. No Theater.")
    print("=" * 60)

    run_id = str(uuid.uuid4())
    commit_sha = get_git_commit()
    os.environ.setdefault("AURA_PROOF_RUN", "1")
    if args.disable_structured_proof_solver:
        os.environ["AURA_DISABLE_STRUCTURED_PROOF_SOLVER"] = "1"
    else:
        os.environ.pop("AURA_DISABLE_STRUCTURED_PROOF_SOLVER", None)
        os.environ.pop("AURA_DISABLE_MLX_STRICT_ANSWER_CONTRACT", None)
    requested_proof_model_tier = (
        args.model_tier or os.environ.get("AURA_PROOF_MODEL_TIER") or "primary"
    ).strip().lower()
    if requested_proof_model_tier not in {"primary", "tertiary"}:
        print(
            f"  [WARN] Invalid AURA_PROOF_MODEL_TIER={requested_proof_model_tier!r}; "
            "using primary."
        )
        requested_proof_model_tier = "primary"
    os.environ["AURA_PROOF_MODEL_TIER"] = requested_proof_model_tier

    if args.out:
        artifacts_base = Path(args.out).resolve()
    else:
        artifacts_base = Path(os.environ.get("AURA_ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts" / "agi_live")))

    run_dir = artifacts_base
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale_artifact in DNU_STALE_ARTIFACTS:
        try:
            (run_dir / stale_artifact).unlink(missing_ok=True)
        except OSError as exc:
            print(f"  [WARN] Could not remove stale artifact {stale_artifact}: {exc}")

    sys_info = {
        "run_id": run_id,
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit_sha": commit_sha,
        "python_version": sys.version,
        "platform": platform.platform(),
        "proof_model_tier": requested_proof_model_tier,
        "proof_live_message_origin": PROOF_LIVE_MESSAGE_ORIGIN,
        "structured_proof_solver_enabled": not args.disable_structured_proof_solver,
    }

    print(f"Run ID: {run_id}")
    print(f"Commit SHA: {commit_sha}")
    print(f"Run Directory: {run_dir}")
    print(f"Proof Model Tier: {requested_proof_model_tier}")
    lifecycle_events = 0
    write_run_status(
        run_dir,
        status="running",
        run_id=run_id,
        commit_sha=commit_sha,
        phase="startup",
        lifecycle_events=lifecycle_events,
    )
    (run_dir / "RESOURCE_TRACE.jsonl").touch()
    (run_dir / "LIFECYCLE_EVENTS.jsonl").touch()

    def fail_run_status(
        *,
        phase: str,
        error: str,
        tasks_completed: int = 0,
        total_tasks: int | None = None,
    ) -> int:
        """Persist a terminal failed status before returning from a fatal gate."""
        write_run_status(
            run_dir,
            status="failed",
            run_id=run_id,
            commit_sha=commit_sha,
            phase=phase,
            tasks_completed=tasks_completed,
            total_tasks=total_tasks,
            error=error,
            lifecycle_events=lifecycle_events,
        )
        return 1

    # -----------------------------------------------------------------------
    # 0. Proof runtime exclusivity
    # -----------------------------------------------------------------------
    print("\n[0/8] Checking proof runtime exclusivity...")
    exclusive_report_path = run_dir / "EXCLUSIVE_RUNTIME_PREFLIGHT.json"
    existing_proof_runners = find_existing_proof_runners()
    if existing_proof_runners and args.stop_existing_runtime:
        remaining_proof_runners = stop_existing_proof_runners()
        if remaining_proof_runners:
            write_exclusive_runtime_report(
                exclusive_report_path,
                status="proof_runner_stop_failed",
                instances=remaining_proof_runners,
            )
            print("  [FATAL] Stale proof runner remained alive after stop request:")
            for instance in remaining_proof_runners:
                print(f"    PID {instance['pid']}: {instance['command'][:180]}")
            return fail_run_status(
                phase="runtime_exclusivity",
                error="proof_runner_stop_failed",
            )
    elif existing_proof_runners and not args.allow_coexisting_runtime:
        write_exclusive_runtime_report(
            exclusive_report_path,
            status="blocked_existing_proof_runner",
            instances=existing_proof_runners,
        )
        print("  [FATAL] Existing DNU proof runner detected. Proof runs require exclusivity.")
        print("          Re-run with --stop-existing-runtime to stop it first.")
        for instance in existing_proof_runners:
            print(f"    PID {instance['pid']}: {instance['command'][:180]}")
        return fail_run_status(
            phase="runtime_exclusivity",
            error="blocked_existing_proof_runner",
        )

    existing_runtimes = find_existing_aura_runtimes()
    if existing_runtimes and args.stop_existing_runtime:
        remaining = stop_existing_aura_runtimes()
        exclusive_report = write_exclusive_runtime_report(
            exclusive_report_path,
            status="stopped_existing_runtime" if not remaining else "stop_failed",
            instances=remaining,
        )
        if remaining:
            print("  [FATAL] Existing Aura runtime remained alive after stop request:")
            for instance in remaining:
                print(f"    PID {instance['pid']}: {instance['command'][:180]}")
            return fail_run_status(
                phase="runtime_exclusivity",
                error="aura_runtime_stop_failed",
            )
    elif existing_runtimes and not args.allow_coexisting_runtime:
        exclusive_report = write_exclusive_runtime_report(
            exclusive_report_path,
            status="blocked_existing_runtime",
            instances=existing_runtimes,
        )
        print("  [FATAL] Existing Aura runtime detected. Proof runs require exclusive runtime.")
        print("          Re-run with --stop-existing-runtime to stop it first.")
        for instance in existing_runtimes:
            print(f"    PID {instance['pid']}: {instance['command'][:180]}")
        return fail_run_status(
            phase="runtime_exclusivity",
            error="blocked_existing_aura_runtime",
        )
    else:
        status = "coexisting_runtime_allowed" if existing_runtimes else "exclusive"
        exclusive_report = write_exclusive_runtime_report(
            exclusive_report_path,
            status=status,
            instances=existing_runtimes,
        )
        print(f"  [OK] Runtime exclusivity status: {status}.")
    reaped_orphans = stop_orphaned_aura_multiprocessing_children()
    if reaped_orphans:
        exclusive_report["orphaned_multiprocessing_children_reaped"] = reaped_orphans
        exclusive_report_path.write_text(json.dumps(exclusive_report, indent=2), encoding="utf-8")
        print(
            "  [CLEANUP] Reaped "
            f"{len(reaped_orphans)} orphaned Aura multiprocessing helper(s) from prior interrupted runs."
        )
    if not args.allow_coexisting_runtime:
        from aura_main import bootstrap_lock

        bootstrap_lock(skip_lock=False)
        exclusive_report["canonical_runtime_lock_claimed_by_runner_pid"] = os.getpid()
        exclusive_report_path.write_text(json.dumps(exclusive_report, indent=2), encoding="utf-8")
        print("  [OK] Canonical runtime lock claimed for proof runner.")
    sys_info["exclusive_runtime_preflight"] = exclusive_report

    # -----------------------------------------------------------------------
    # 1. Load Task Packs
    # -----------------------------------------------------------------------
    print("\n[1/8] Loading sealed task packs...")
    fixture_dir = PROJECT_ROOT / "tests" / "agi" / "fixtures" / "dnu_tasks"
    if not fixture_dir.exists():
        print(f"  [FATAL] Fixture directory not found: {fixture_dir}")
        return fail_run_status(
            phase="load_task_packs",
            error=f"fixture_directory_not_found:{fixture_dir}",
        )

    all_tasks, grader_data = load_task_packs(fixture_dir)
    print(f"  Total tasks loaded: {len(all_tasks)}")
    print(f"  Grader entries loaded: {len(grader_data)}")

    if args.smoke:
        print("  [LIMIT] Smoke run enabled: limiting execution to first 1 task.")
        all_tasks = all_tasks[:1]
    else:
        max_tasks_env = os.environ.get("AURA_AGI_MAX_TASKS")
        if max_tasks_env:
            try:
                max_tasks = int(max_tasks_env)
                print(f"  [LIMIT] Limiting execution to first {max_tasks} tasks (AURA_AGI_MAX_TASKS={max_tasks})")
                all_tasks = all_tasks[:max_tasks]
            except ValueError:
                print(f"  [WARN] Invalid AURA_AGI_MAX_TASKS value: {max_tasks_env}")

    if len(all_tasks) == 0:
        write_run_status(
            run_dir,
            status="failed",
            run_id=run_id,
            commit_sha=commit_sha,
            phase="load_task_packs",
            error="no_tasks_loaded",
            lifecycle_events=lifecycle_events,
        )
        print("  [FATAL] No tasks loaded. Cannot run battery.")
        return 1
    write_run_status(
        run_dir,
        status="running",
        run_id=run_id,
        commit_sha=commit_sha,
        phase="task_packs_loaded",
        total_tasks=len(all_tasks),
        lifecycle_events=lifecycle_events,
    )

    # -----------------------------------------------------------------------
    # 2. Anti-Theater Pre-Check
    # -----------------------------------------------------------------------
    print("\n[2/8] Running anti-theater pre-checks...")
    pre_violations = anti_theater_pre_check(all_tasks, grader_data)
    if pre_violations:
        for v in pre_violations:
            print(f"  [VIOLATION] {v}")
    else:
        print("  [OK] All pre-checks passed.")

    # -----------------------------------------------------------------------
    # 3. Boot canonical Aura runtime
    # -----------------------------------------------------------------------
    print("\n[3/8] Booting canonical AuraRuntime(profile='proof')...")

    from aura_main import boot_aura_runtime
    from core.container import ServiceContainer

    orch = await boot_aura_runtime(
        profile="proof",
        ready_label="Proof-DNU",
        readiness_context="dnu_agi_proof",
        artifact_root=PROJECT_ROOT / "artifacts" / "current",
    )
    print("  [OK] Canonical Aura runtime booted via aura_main.boot_aura_runtime.")

    router = ServiceContainer.get("llm_router", default=None)
    removed_ep = None
    if router is not None and hasattr(router, "endpoints"):
        # Keep the proof run on the same live runtime while preventing a second
        # heavyweight local lane from resident-coexisting with Cortex.
        removed_ep = router.endpoints.pop("Solver", None)
    if removed_ep:
        print("  [OK] Programmatically quarantined local heavyweight Solver (72B) endpoint to prevent timeouts.")

    engine = (
        ServiceContainer.get("cognitive_engine", default=None)
        or getattr(orch, "cognitive_engine", None)
        or getattr(orch, "cognition", None)
    )
    if engine is None:
        print("  [FATAL] Canonical boot completed without cognitive_engine.")
        return fail_run_status(
            phase="boot_runtime",
            error="canonical_boot_completed_without_cognitive_engine",
            total_tasks=len(all_tasks),
        )
    if hasattr(engine, "setup") and not getattr(engine, "_phases", None):
        engine.setup()
    print(f"  [OK] CognitiveEngine resolved from canonical boot. Lobotomized: {getattr(engine, 'lobotomized', False)}")
    print(f"  [OK] Phases loaded: {len(getattr(engine, '_phases', []))}")

    if getattr(engine, "lobotomized", False):
        print("  [FATAL] CognitiveEngine is lobotomized. Cannot run battery.")
        return fail_run_status(
            phase="boot_runtime",
            error="cognitive_engine_lobotomized",
            total_tasks=len(all_tasks),
        )

    runtime_manifest_src = PROJECT_ROOT / "artifacts" / "current" / "runtime_manifest.json"
    if not runtime_manifest_src.exists():
        print(f"  [FATAL] Canonical boot did not emit runtime manifest: {runtime_manifest_src}")
        return fail_run_status(
            phase="boot_runtime",
            error=f"runtime_manifest_missing:{runtime_manifest_src}",
            total_tasks=len(all_tasks),
        )
    runtime_manifest_copy = run_dir / "RUNTIME_MANIFEST.json"
    runtime_manifest_copy.write_text(runtime_manifest_src.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"  [OK] Runtime manifest captured in {runtime_manifest_copy.name}.")
    runtime_policy = {
        "proof_model_tier": requested_proof_model_tier,
        "live_message_origin": PROOF_LIVE_MESSAGE_ORIGIN,
        "canonical_boot_profile": "proof",
        "strict_answer_tags_required": True,
        "structured_proof_solver_enabled": not args.disable_structured_proof_solver,
        "exclusive_runtime_required": not args.allow_coexisting_runtime,
        "exclusive_runtime_preflight_status": exclusive_report.get("status"),
        "model_recycle_interval_tasks": dnu_model_recycle_interval(
            requested_proof_model_tier,
            total_tasks=len(all_tasks),
            smoke=args.smoke,
        ),
        "comparisons_mode": "skipped_for_smoke" if args.smoke else "run",
    }
    runtime_policy_path = run_dir / "RUNTIME_POLICY.json"
    runtime_policy_path.write_text(json.dumps(runtime_policy, indent=2), encoding="utf-8")
    print(f"  [OK] Runtime policy captured in {runtime_policy_path.name}.")

    print("  [PROBE] Exercising requested proof model lane before task execution...")
    model_lane_probe = await run_model_lane_probe(router, requested_proof_model_tier, run_dir)
    runtime_policy["model_lane_probe"] = model_lane_probe
    runtime_policy_path.write_text(json.dumps(runtime_policy, indent=2), encoding="utf-8")
    if not model_lane_probe.get("ok"):
        print(
            "  [FATAL] Requested proof model lane failed probe: "
            f"{model_lane_probe.get('error') or model_lane_probe.get('text_preview')}"
        )
        await shutdown_proof_runtime(orch)
        return fail_run_status(
            phase="model_lane_probe",
            error=str(
                model_lane_probe.get("error")
                or model_lane_probe.get("text_preview")
                or "model_lane_probe_failed"
            ),
            total_tasks=len(all_tasks),
        )
    print(
        "  [OK] Model lane probe passed via "
        f"{model_lane_probe.get('endpoint')} ({model_lane_probe.get('endpoint_tier')})."
    )
    initial_resource_snapshot, initial_health_blockers = await wait_for_proof_runtime_health(
        label="after_model_lane_probe",
        timeout_s=90.0 if requested_proof_model_tier == "primary" else 45.0,
        interval_s=2.0,
    )
    append_jsonl(run_dir / "RESOURCE_TRACE.jsonl", initial_resource_snapshot)
    if initial_health_blockers:
        print("  [FATAL] Proof runtime health failed after model lane probe:")
        for blocker in initial_health_blockers:
            print(f"    - {blocker}")
        await shutdown_proof_runtime(orch)
        return fail_run_status(
            phase="model_lane_probe",
            error="; ".join(initial_health_blockers),
            total_tasks=len(all_tasks),
        )

    # -----------------------------------------------------------------------
    # 4. Execute Tasks
    # -----------------------------------------------------------------------
    print(f"\n[4/8] Executing {len(all_tasks)} tasks through CognitiveEngine...")
    results = []
    trace_file = run_dir / "TASK_TRACE.jsonl"
    receipts_file = run_dir / "RECEIPTS.jsonl"
    resource_trace_file = run_dir / "RESOURCE_TRACE.jsonl"
    recycle_interval = dnu_model_recycle_interval(
        requested_proof_model_tier,
        total_tasks=len(all_tasks),
        smoke=args.smoke,
    )
    if recycle_interval:
        print(
            "  [LIFECYCLE] Primary proof model worker will be recycled every "
            f"{recycle_interval} tasks to bound long-run resident resource growth."
        )
    from core.will import get_will
    will = get_will()
    await will.start()
    write_run_status(
        run_dir,
        status="running",
        run_id=run_id,
        commit_sha=commit_sha,
        phase="task_execution",
        total_tasks=len(all_tasks),
        lifecycle_events=lifecycle_events,
    )

    with trace_file.open("w", encoding="utf-8") as trace_fh, receipts_file.open("w", encoding="utf-8") as receipts_fh:
        for i, task in enumerate(all_tasks, 1):
            tid = task.get("task_id", "?")
            cat = task.get("category", "?")
            print(f"  [{i}/{len(all_tasks)}] {tid} ({cat})...", end=" ", flush=True)

            before_len = len(will._audit_trail) if will else 0
            pre_action_receipt_ids: set[str] = set()

            try:
                await isolate_live_runtime_for_dnu_task(task)
            except _DNU_RUN_RECOVERABLE_ERRORS as e:
                print(f"  [WARN] Failed to reset state for task isolation: {e}")

            if will:
                try:
                    from core.will import ActionDomain, WillOutcome

                    decision = will.decide(
                        content=task.get("task_prompt", ""),
                        source="dnu_agi_proof_battery",
                        domain=ActionDomain.RESPONSE,
                        priority=1.0,
                        is_critical=True,
                    )
                    pre_action_receipt_ids.add(decision.receipt_id)
                    if decision.outcome == WillOutcome.REFUSE:
                        result = {
                            "task_id": tid,
                            "category": task.get("category", "unknown"),
                            "difficulty": task.get("difficulty", "unknown"),
                            "status": "error",
                            "response_text": "",
                            "extracted_answer": None,
                            "normalized_answer": None,
                            "answer_hash": None,
                            "elapsed_s": 0.0,
                            "error": f"pre_action_authorization_refused:{decision.reason}",
                        }
                        results.append(result)
                        trace_fh.write(json.dumps(result, default=str) + "\n")
                        trace_fh.flush()
                        print(f"⚠ error ({result['error']})")
                        continue
                except _DNU_RUN_RECOVERABLE_ERRORS as ex:
                    result = {
                        "task_id": tid,
                        "category": task.get("category", "unknown"),
                        "difficulty": task.get("difficulty", "unknown"),
                        "status": "error",
                        "response_text": "",
                        "extracted_answer": None,
                        "normalized_answer": None,
                        "answer_hash": None,
                        "elapsed_s": 0.0,
                        "error": f"pre_action_authorization_failed:{type(ex).__name__}: {ex}",
                    }
                    results.append(result)
                    trace_fh.write(json.dumps(result, default=str) + "\n")
                    trace_fh.flush()
                    print(f"⚠ error ({result['error']})")
                    continue

            result = await execute_task(orch, task, timeout_s=max(240, task.get("time_budget_s", 240)))

            result = grade_result(result, grader_data)
            results.append(result)

            # Write trace
            trace_fh.write(json.dumps(result, default=str) + "\n")
            trace_fh.flush()

            # Record receipts
            if will:
                new_decisions = list(will._audit_trail)[before_len:]
                for d in new_decisions:
                    domain_val = d.domain.value if hasattr(d.domain, "value") else str(d.domain)
                    outcome_val = d.outcome.value if hasattr(d.outcome, "value") else str(d.outcome)
                    is_pre_action = d.receipt_id in pre_action_receipt_ids
                    effect_verified = result["status"] in {"pass", "fail", "no_answer"}
                    telemetry_logged = True
                    closure_ok = (
                        will.verify_closure(
                            d.receipt_id,
                            effect_verified=effect_verified,
                            telemetry_logged=telemetry_logged,
                        )
                        if is_pre_action
                        else False
                    )
                    vol_hash = hashlib.sha256(f"{tid}:{d.receipt_id}:{domain_val}:{outcome_val}:{d.reason}".encode()).hexdigest()
                    receipt_entry = {
                        "task_id": tid,
                        "receipt_id": d.receipt_id,
                        "domain": domain_val,
                        "outcome": outcome_val,
                        "reason": d.reason,
                        "source": getattr(d, "source", ""),
                        "volition_hash": vol_hash,
                        "authorization_phase": "pre_action" if is_pre_action else "internal_runtime",
                        "effect_verified": effect_verified,
                        "telemetry_logged": telemetry_logged,
                        "closure_verified": closure_ok,
                    }
                    receipts_fh.write(json.dumps(receipt_entry, default=str) + "\n")
                receipts_fh.flush()

            status_icon = {
                "pass": "✓",
                "fail": "✗",
                "timeout": "⏱",
                "error": "⚠",
                "no_answer": "∅",
                "ungraded": "?",
            }.get(result["status"], "?")
            print(f"{status_icon} {result['status']} ({result['elapsed_s']:.1f}s)")
            task_resource_snapshot = collect_proof_resource_snapshot(
                label="after_task",
                task_index=i,
                task_id=tid,
            )
            append_jsonl(resource_trace_file, task_resource_snapshot)
            task_health_blockers = proof_runtime_health_blockers(task_resource_snapshot)
            if task_health_blockers:
                print("  [FATAL] Proof runtime health failed during task execution:")
                for blocker in task_health_blockers:
                    print(f"    - {blocker}")
                await shutdown_proof_runtime(orch)
                return fail_run_status(
                    phase="task_execution",
                    error="; ".join(task_health_blockers),
                    tasks_completed=len(results),
                    total_tasks=len(all_tasks),
                )
            write_run_status(
                run_dir,
                status="running",
                run_id=run_id,
                commit_sha=commit_sha,
                phase="task_execution",
                tasks_completed=len(results),
                total_tasks=len(all_tasks),
                lifecycle_events=lifecycle_events,
            )
            if recycle_interval and i < len(all_tasks) and i % recycle_interval == 0:
                recycle_event = await recycle_proof_model_lane(
                    router,
                    requested_proof_model_tier,
                    run_dir=run_dir,
                    reason=f"dnu_checkpoint_after_task_{i}",
                    task_index=i,
                )
                lifecycle_events += 1
                write_run_status(
                    run_dir,
                    status="running",
                    run_id=run_id,
                    commit_sha=commit_sha,
                    phase="model_lane_recycle",
                    tasks_completed=len(results),
                    total_tasks=len(all_tasks),
                    error=recycle_event.get("error"),
                    lifecycle_events=lifecycle_events,
                )
                if recycle_event.get("status") != "complete":
                    print(
                        "  [FATAL] Proof model lane recycle failed: "
                        f"{recycle_event.get('error')}"
                    )
                    await shutdown_proof_runtime(orch)
                    return fail_run_status(
                        phase="model_lane_recycle",
                        error=str(recycle_event.get("error") or "model_lane_recycle_failed"),
                        tasks_completed=len(results),
                        total_tasks=len(all_tasks),
                    )
                recycle_health_blockers = proof_runtime_health_blockers(
                    recycle_event.get("after") or {}
                )
                if recycle_health_blockers:
                    print("  [FATAL] Proof runtime health failed after model lane recycle:")
                    for blocker in recycle_health_blockers:
                        print(f"    - {blocker}")
                    await shutdown_proof_runtime(orch)
                    return fail_run_status(
                        phase="model_lane_recycle",
                        error="; ".join(recycle_health_blockers),
                        tasks_completed=len(results),
                        total_tasks=len(all_tasks),
                    )

    # -----------------------------------------------------------------------
    # 5. Anti-Theater Post-Check
    # -----------------------------------------------------------------------
    print("\n[5/8] Running anti-theater post-checks...")
    write_run_status(
        run_dir,
        status="running",
        run_id=run_id,
        commit_sha=commit_sha,
        phase="anti_theater_post_check",
        tasks_completed=len(results),
        total_tasks=len(all_tasks),
        lifecycle_events=lifecycle_events,
    )
    post_violations = anti_theater_post_check(results)
    if post_violations:
        for v in post_violations:
            print(f"  [VIOLATION] {v}")
    else:
        print("  [OK] All post-checks passed.")

    anti_theater = {
        "pre_check_violations": pre_violations,
        "post_check_violations": post_violations,
        "all_passed": len(pre_violations) == 0 and len(post_violations) == 0,
    }

    # -----------------------------------------------------------------------
    # 6. Compute Scorecard and Enforce Tier-Capping
    # -----------------------------------------------------------------------
    print("\n[6/8] Computing scorecard from actual results...")
    write_run_status(
        run_dir,
        status="running",
        run_id=run_id,
        commit_sha=commit_sha,
        phase="scorecard",
        tasks_completed=len(results),
        total_tasks=len(all_tasks),
        lifecycle_events=lifecycle_events,
    )
    scorecard = compute_scorecard(results)

    # Check minimum task counts and record violations
    unsupported_claims = []
    for cat, min_count in MINIMUM_COUNTS.items():
        actual = scorecard["categories"].get(cat, {}).get("attempted", 0)
        if actual < min_count:
            unsupported_claims.append(
                f"Category '{cat}' has {actual} tasks, below minimum of {min_count}"
            )

    tier = assign_tier(scorecard["overall_pass_rate"], has_unsupported_claims=len(unsupported_claims) > 0)

    print(f"  Overall Pass Rate: {scorecard['overall_pass_rate']:.1%}")
    print(f"  Assigned Tier: {tier['tier']} ({tier['label']})")
    for cat, stats in sorted(scorecard["categories"].items()):
        print(f"  {cat}: {stats['passed']}/{stats['attempted']} ({stats['pass_rate']:.1%})")

    # -----------------------------------------------------------------------
    # 6.5. Run Baselines & Ablations
    # -----------------------------------------------------------------------
    if args.smoke:
        print("\n[SMOKE] Skipping baseline and ablation comparisons.")
        print("  Smoke verifies boot, model lane, live message path, grading, and artifacts only.")
        baselines = {
            "raw_llm": {"status": "SKIPPED_SMOKE", "reason": "smoke runs avoid expensive comparison loops"},
            "llm_with_tools": {"status": "SKIPPED_SMOKE", "reason": "smoke runs avoid expensive comparison loops"},
            "react_agent": {"status": "SKIPPED_SMOKE", "reason": "smoke runs avoid expensive comparison loops"},
        }
        full_aura_comparison_rate = scorecard["overall_pass_rate"]
        ablations = {
            "full_aura": {"status": "RUN", "pass_rate": full_aura_comparison_rate},
            "no_persistent_memory": {"status": "SKIPPED_SMOKE", "lesion_effect_verified": None},
            "no_volition": {"status": "SKIPPED_SMOKE", "lesion_effect_verified": None},
            "no_will_authority": {"status": "SKIPPED_SMOKE", "lesion_effect_verified": None},
            "no_system2": {"status": "SKIPPED_SMOKE", "lesion_effect_verified": None},
            "no_self_repair": {"status": "SKIPPED_SMOKE", "lesion_effect_verified": None},
            "no_affect_steering": {"status": "SKIPPED_SMOKE", "lesion_effect_verified": None},
            # Compatibility aliases for historical report consumers.
            "aura_minus_memory": {"status": "SKIPPED_SMOKE", "lesion_effect_verified": None},
            "aura_minus_volition": {"status": "SKIPPED_SMOKE", "lesion_effect_verified": None},
            "aura_minus_will": {"status": "SKIPPED_SMOKE", "lesion_effect_verified": None},
            "aura_minus_system2": {"status": "SKIPPED_SMOKE", "lesion_effect_verified": None},
            "aura_minus_self_repair": {"status": "SKIPPED_SMOKE", "lesion_effect_verified": None},
            "aura_minus_affect_steering": {"status": "SKIPPED_SMOKE", "lesion_effect_verified": None},
        }
    else:
        print("\nRunning raw LLM, LLM with direct tools, and ReAct agent baselines...")
        write_run_status(
            run_dir,
            status="running",
            run_id=run_id,
            commit_sha=commit_sha,
            phase="baselines",
            tasks_completed=len(results),
            total_tasks=len(all_tasks),
            lifecycle_events=lifecycle_events,
        )
        # Cap tasks for baseline and ablation comparisons to keep execution highly efficient
        comparison_tasks = all_tasks[:10]
        print(f"  Using first {len(comparison_tasks)} tasks for representative ablation comparisons.")

        sem = asyncio.Semaphore(1)
        raw_llm_tasks = [execute_raw_llm_task(router, task, grader_data, sem) for task in comparison_tasks]
        llm_with_tools_tasks = [execute_llm_with_tools_task(router, task, grader_data, sem) for task in comparison_tasks]
        react_tasks = [execute_react_task(router, task, grader_data, sem) for task in comparison_tasks]

        raw_llm_results = await asyncio.gather(*raw_llm_tasks)
        llm_with_tools_results = await asyncio.gather(*llm_with_tools_tasks)
        react_results = await asyncio.gather(*react_tasks)
        baseline_resource_snapshot, baseline_health_blockers = await wait_for_proof_runtime_health(
            label="after_baselines",
            timeout_s=60.0,
            interval_s=2.0,
        )
        append_jsonl(resource_trace_file, baseline_resource_snapshot)
        if baseline_health_blockers:
            print("  [FATAL] Proof runtime health failed after baseline comparison:")
            for blocker in baseline_health_blockers:
                print(f"    - {blocker}")
            write_run_status(
                run_dir,
                status="failed",
                run_id=run_id,
                commit_sha=commit_sha,
                phase="baselines",
                tasks_completed=len(results),
                total_tasks=len(all_tasks),
                error="; ".join(baseline_health_blockers),
                lifecycle_events=lifecycle_events,
            )
            await shutdown_proof_runtime(orch)
            return 1

        raw_llm_scorecard = compute_scorecard(raw_llm_results)
        llm_with_tools_scorecard = compute_scorecard(llm_with_tools_results)
        react_scorecard = compute_scorecard(react_results)

        baselines = {
            "raw_llm": {
                "status": "RUN",
                "pass_rate": raw_llm_scorecard["overall_pass_rate"],
                "total_tasks": len(comparison_tasks),
                "passed": raw_llm_scorecard["total_pass"],
            },
            "llm_with_tools": {
                "status": "RUN",
                "pass_rate": llm_with_tools_scorecard["overall_pass_rate"],
                "total_tasks": len(comparison_tasks),
                "passed": llm_with_tools_scorecard["total_pass"],
            },
            "react_agent": {
                "status": "RUN",
                "pass_rate": react_scorecard["overall_pass_rate"],
                "total_tasks": len(comparison_tasks),
                "passed": react_scorecard["total_pass"],
            },
        }

        print("\nRunning dynamic system ablations sequentially...")
        write_run_status(
            run_dir,
            status="running",
            run_id=run_id,
            commit_sha=commit_sha,
            phase="ablations",
            tasks_completed=len(results),
            total_tasks=len(all_tasks),
            lifecycle_events=lifecycle_events,
        )

        # Compute full_aura pass rate on the same comparison subset for honest ablation comparison.
        full_aura_comparison_results = results[:len(comparison_tasks)]
        full_aura_comparison_scorecard = compute_scorecard(full_aura_comparison_results)
        full_aura_comparison_rate = full_aura_comparison_scorecard["overall_pass_rate"]

        print("  Running ablation: no_persistent_memory...")
        aura_minus_memory_rate, memory_verified = await run_ablation_suite(orch, comparison_tasks, grader_data, ["memory_facade", "memory_coordinator"])

        print("  Running ablation: no_volition...")
        aura_minus_volition_rate, volition_verified = await run_ablation_suite(orch, comparison_tasks, grader_data, ["volition_engine"])

        print("  Running ablation: no_will_authority...")
        aura_minus_will_rate, will_verified = await run_ablation_suite(orch, comparison_tasks, grader_data, ["unified_will"])

        print("  Running ablation: no_system2...")
        aura_minus_system2_rate, system2_verified = await run_ablation_suite(orch, comparison_tasks, grader_data, ["native_system2"])

        print("  Running ablation: no_self_repair...")
        aura_minus_self_repair_rate, self_repair_verified = await run_ablation_suite(orch, comparison_tasks, grader_data, ["self_repair", "skill_library"])

        print("  Running ablation: no_affect_steering...")
        aura_minus_affect_steering_rate, affect_verified = await run_ablation_suite(orch, comparison_tasks, grader_data, ["affective_steering_engine", "affect_engine", "affect_facade"])

        ablations = {
            "full_aura": {"status": "RUN", "pass_rate": full_aura_comparison_rate},
            "no_persistent_memory": {"status": "RUN", "pass_rate": aura_minus_memory_rate, "lesion_effect_verified": memory_verified},
            "no_volition": {"status": "RUN", "pass_rate": aura_minus_volition_rate, "lesion_effect_verified": volition_verified},
            "no_will_authority": {"status": "RUN", "pass_rate": aura_minus_will_rate, "lesion_effect_verified": will_verified},
            "no_system2": {"status": "RUN", "pass_rate": aura_minus_system2_rate, "lesion_effect_verified": system2_verified},
            "no_self_repair": {"status": "RUN", "pass_rate": aura_minus_self_repair_rate, "lesion_effect_verified": self_repair_verified},
            "no_affect_steering": {"status": "RUN", "pass_rate": aura_minus_affect_steering_rate, "lesion_effect_verified": affect_verified},
            # Compatibility aliases for historical report consumers.
            "aura_minus_memory": {"status": "RUN", "pass_rate": aura_minus_memory_rate, "lesion_effect_verified": memory_verified},
            "aura_minus_volition": {"status": "RUN", "pass_rate": aura_minus_volition_rate, "lesion_effect_verified": volition_verified},
            "aura_minus_will": {"status": "RUN", "pass_rate": aura_minus_will_rate, "lesion_effect_verified": will_verified},
            "aura_minus_system2": {"status": "RUN", "pass_rate": aura_minus_system2_rate, "lesion_effect_verified": system2_verified},
            "aura_minus_self_repair": {"status": "RUN", "pass_rate": aura_minus_self_repair_rate, "lesion_effect_verified": self_repair_verified},
            "aura_minus_affect_steering": {"status": "RUN", "pass_rate": aura_minus_affect_steering_rate, "lesion_effect_verified": affect_verified},
        }
        ablation_resource_snapshot, ablation_health_blockers = await wait_for_proof_runtime_health(
            label="after_ablations",
            timeout_s=60.0,
            interval_s=2.0,
        )
        append_jsonl(resource_trace_file, ablation_resource_snapshot)
        if ablation_health_blockers:
            print("  [FATAL] Proof runtime health failed after ablation recovery:")
            for blocker in ablation_health_blockers:
                print(f"    - {blocker}")
            write_run_status(
                run_dir,
                status="failed",
                run_id=run_id,
                commit_sha=commit_sha,
                phase="ablations",
                tasks_completed=len(results),
                total_tasks=len(all_tasks),
                error="; ".join(ablation_health_blockers),
                lifecycle_events=lifecycle_events,
            )
            await shutdown_proof_runtime(orch)
            return 1

    # -----------------------------------------------------------------------
    # 7. Write Artifacts
    # -----------------------------------------------------------------------
    print("\n[7/8] Writing artifacts...")
    write_run_status(
        run_dir,
        status="running",
        run_id=run_id,
        commit_sha=commit_sha,
        phase="write_artifacts",
        tasks_completed=len(results),
        total_tasks=len(all_tasks),
        lifecycle_events=lifecycle_events,
    )

    # Granular verification checklist
    baselines_complete = (
        all(b.get("status") == "RUN" for b in baselines.values())
        if not args.smoke
        else all(b.get("status") == "SKIPPED_SMOKE" for b in baselines.values())
    )
    ablations_verified = (
        all(a.get("lesion_effect_verified", False) for name, a in ablations.items() if name != "full_aura")
        if not args.smoke
        else all(
            a.get("status") == "SKIPPED_SMOKE"
            for name, a in ablations.items()
            if name != "full_aura"
        )
    )
    
    category_thresholds_passed = True
    for cat_name, cat_stats in scorecard["categories"].items():
        if "transfer" in cat_name.lower():
            if cat_stats.get("pass_rate", 0.0) < 0.75:
                category_thresholds_passed = False
                
    receipt_count = 0
    if receipts_file.exists():
        try:
            receipt_count = len(receipts_file.read_text(encoding="utf-8").strip().splitlines())
        except _DNU_RUN_RECOVERABLE_ERRORS as exc:
            print(f"  [WARN] Failed to count governance receipts: {exc}")
    gov_report = build_governance_report(receipts_file, expected_tasks=len(all_tasks))
    leakage_report = build_leakage_report(
        pre_violations=pre_violations,
        post_violations=post_violations,
        run_dir=run_dir,
    )
    structured_solver_task_count = int(leakage_report.get("structured_solver_task_count", 0) or 0)

    verification_checklist = {
        "runner_completed": True,
        "score_threshold_passed": scorecard["overall_pass_rate"] >= 0.85,
        "category_thresholds_passed": category_thresholds_passed,
        "baselines_complete": baselines_complete,
        "ablations_verified": ablations_verified,
        "governance_receipts_verified": gov_report.get("status") == "pass",
        "leakage_checks_passed": leakage_report.get("status") == "pass",
        "model_lane_probe_passed": bool(model_lane_probe.get("ok")),
        "model_lane_answered_without_structured_solver": (
            structured_solver_task_count == 0
            if args.disable_structured_proof_solver
            else True
        ),
    }

    # Main proof bundle
    proof_bundle = {
        "system_info": sys_info,
        "scorecard": scorecard,
        "tier": tier,
        "anti_theater": anti_theater,
        "baselines": baselines,
        "ablations": ablations,
        "task_count": len(all_tasks),
        "grader_entry_count": len(grader_data),
        "category_summary": {cat: scorecard["categories"].get(cat, {}) for cat in DIR_TO_CAT.values()},
        "unsupported_claims": unsupported_claims,
        "verification_checklist": verification_checklist,
        "runtime_policy": runtime_policy,
        "model_lane_probe": model_lane_probe,
        "structured_solver_task_count": structured_solver_task_count,
        "lifecycle_event_count": lifecycle_events,
        "resource_trace": {
            "path": str((run_dir / "RESOURCE_TRACE.jsonl").relative_to(PROJECT_ROOT))
            if (run_dir / "RESOURCE_TRACE.jsonl").is_relative_to(PROJECT_ROOT)
            else str(run_dir / "RESOURCE_TRACE.jsonl"),
        },
        "governance_report_status": gov_report.get("status"),
        "leakage_report_status": leakage_report.get("status"),
        "passed": all(verification_checklist.values()) and anti_theater["all_passed"],
    }

    # Write DNU_AGI_PROOF.json
    proof_path = run_dir / "DNU_AGI_PROOF.json"
    proof_path.write_text(json.dumps(proof_bundle, indent=2, default=str), encoding="utf-8")
    print(f"  [OK] {proof_path.name}")

    # Write SCORECARD.json
    scorecard_path = run_dir / "SCORECARD.json"
    scorecard_path.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
    print(f"  [OK] {scorecard_path.name}")

    # Write BASELINES.json
    baselines_path = run_dir / "BASELINES.json"
    baselines_path.write_text(json.dumps(baselines, indent=2), encoding="utf-8")
    print(f"  [OK] {baselines_path.name}")

    # Write ABLATIONS.json
    ablations_path = run_dir / "ABLATIONS.json"
    ablations_path.write_text(json.dumps(ablations, indent=2), encoding="utf-8")
    print(f"  [OK] {ablations_path.name}")

    # Write FAILURES.jsonl
    failures_path = run_dir / "FAILURES.jsonl"
    with failures_path.open("w", encoding="utf-8") as f:
        for r in results:
            if r["status"] != "pass":
                f.write(json.dumps(r, default=str) + "\n")
    print(f"  [OK] {failures_path.name}")

    # Write markdown report
    md_content = generate_markdown_report(sys_info, scorecard, tier, anti_theater, results, baselines, ablations)
    md_path = run_dir / "DNU_AGI_PROOF.md"
    md_path.write_text(md_content, encoding="utf-8")
    print(f"  [OK] {md_path.name}")

    # Write REPRODUCTION.md
    repro_content = f"""# Reproduction Instructions

## Environment
- **Commit SHA:** `{commit_sha}`
- **Python Version:** `{sys_info['python_version']}`
- **Platform:** `{sys_info['platform']}`
- **Run ID:** `{run_id}`

## Prerequisites
- Aura source code at the specified commit
- LLM model server running (check port configuration)
- Python environment with all dependencies

## Commands
```bash
cd /path/to/aura-source
git checkout {commit_sha}
python tools/agi/run_dnu_agi_proof_battery.py
```

## Verification
```bash
python -m pytest tests/agi/live/test_dnu_agi_proof_battery.py -q
```

## Notes
- Task fixtures are sealed under `tests/agi/fixtures/dnu_tasks/`
- Grader salts are in `.grader_salts*.json` files (not task packs)
- Results depend on model server availability and response quality
- Different model versions will produce different results
"""
    repro_path = run_dir / "REPRODUCTION.md"
    repro_path.write_text(repro_content, encoding="utf-8")
    print(f"  [OK] {repro_path.name}")

    # Write GOVERNANCE_REPORT.json
    gov_report_path = run_dir / "GOVERNANCE_REPORT.json"
    gov_report_path.write_text(json.dumps(gov_report, indent=2), encoding="utf-8")
    print(f"  [OK] {gov_report_path.name}")

    # Write LEAKAGE_REPORT.json
    leakage_report_path = run_dir / "LEAKAGE_REPORT.json"
    leakage_report_path.write_text(json.dumps(leakage_report, indent=2), encoding="utf-8")
    print(f"  [OK] {leakage_report_path.name}")

    # Write FINAL_VERDICT.txt
    verdict_text = "DNU AGI NOT PROVEN"
    if tier["tier"] == 6 and len(unsupported_claims) == 0:
        verdict_text = "DNU AGI PROVEN"
    
    verdict_path = run_dir / "FINAL_VERDICT.txt"
    verdict_path.write_text(verdict_text, encoding="utf-8")
    print(f"  [OK] {verdict_path.name}")
    write_run_status(
        run_dir,
        status="complete",
        run_id=run_id,
        commit_sha=commit_sha,
        phase="complete",
        tasks_completed=len(results),
        total_tasks=len(all_tasks),
        lifecycle_events=lifecycle_events,
    )

    # -----------------------------------------------------------------------
    # 8. Write Manifest
    # -----------------------------------------------------------------------
    print("\n[8/8] Writing manifest...")
    write_artifact_manifest(
        run_dir,
        run_id=run_id,
        commit_sha=commit_sha,
        timestamp=sys_info["timestamp"],
    )
    print("  [OK] MANIFEST.json")

    # Also copy key artifacts to the standard agi_live directory for pytest
    std_dest = artifacts_base
    std_dest.mkdir(parents=True, exist_ok=True)
    for fname in DNU_STANDARD_COPY_ARTIFACTS:
        src = run_dir / fname
        if src.exists():
            (std_dest / fname).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    # Recompute manifest for the standard location
    write_artifact_manifest(
        std_dest,
        run_id=run_id,
        commit_sha=commit_sha,
        include_files=DNU_STANDARD_COPY_ARTIFACTS,
    )

    # -----------------------------------------------------------------------
    # Final Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  DNU AGI PROOF BATTERY: COMPLETE")
    print(f"  Tasks Executed: {scorecard['total_tasks']}")
    print(f"  Overall Pass Rate: {scorecard['overall_pass_rate']:.1%}")
    print(f"  Assigned Tier: {tier['tier']} ({tier['label']})")
    print(f"  Anti-Theater: {'CLEAN' if anti_theater['all_passed'] else 'VIOLATIONS DETECTED'}")
    print("=" * 60)

    if not anti_theater["all_passed"]:
        print("\n[!] Anti-theater violations detected. Review report.")
        await shutdown_proof_runtime(orch)
        return fail_run_status(
            phase="anti_theater_post_check",
            error="anti_theater_violations_detected",
            tasks_completed=len(results),
            total_tasks=len(all_tasks),
        )

    await shutdown_proof_runtime(orch)
    print("\n[+] DNU AGI Proof Battery: COMPLETE")
    return 0


if __name__ == "__main__":
    code = asyncio.run(main())
    raise SystemExit(code)
