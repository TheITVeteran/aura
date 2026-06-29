#!/usr/bin/env python3
"""Unattended orchestrator for Aura's LoRA pipeline.

Drives training/train_and_fuse.py without modifying it. Adds:
  * Resume via training/resume_training.py if a partial adapter exists.
  * State persistence to training_state.json (idempotent re-spawns).
  * SIGTERM/SIGINT: writes final snapshot, then exits cleanly.
  * Dry-run short-circuit: --skip-train + --skip-dataset + --tag dryrun*
    exits 0 without touching the trainer (smoke-test).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from subprocess import TimeoutExpired

try:
    import psutil
except ImportError:  # pragma: no cover - production requirements include psutil.
    psutil = None  # type: ignore[assignment]

TRAINING_DIR = Path(__file__).resolve().parent
REPO_DIR = TRAINING_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402

ADAPTER_DIR = TRAINING_DIR / "adapters" / "aura-personality"
STATE_FILE = ADAPTER_DIR / "training_state.json"
TRAIN_AND_FUSE = TRAINING_DIR / "train_and_fuse.py"
RESUME_SCRIPT = TRAINING_DIR / "resume_training.py"
CHECKPOINT_GLOB = "*_adapters.safetensors"
_STATE_RECOVERABLE_ERRORS = (
    OSError,
    UnicodeDecodeError,
    json.JSONDecodeError,
    TypeError,
    ValueError,
)

_shutdown = threading.Event()
_GIB = 1024**3


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def latest_checkpoint() -> tuple[Path | None, int]:
    """Highest-numbered N_adapters.safetensors → (path, iter)."""
    if not ADAPTER_DIR.exists():
        return None, 0
    cands = [(int(c.stem.split("_", 1)[0]), c) for c in ADAPTER_DIR.glob(CHECKPOINT_GLOB)
             if c.stem.split("_", 1)[0].isdigit()]
    if not cands:
        return None, 0
    n, path = max(cands)
    return path, n


def has_partial_run() -> bool:
    return latest_checkpoint()[0] is not None


def update_state(*, started_at: str, **extra: object) -> dict:
    """Snapshot checkpoint progress + extras to STATE_FILE atomically."""
    ckpt, last_iter = latest_checkpoint()
    state: dict = {}
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text())
        except _STATE_RECOVERABLE_ERRORS as exc:
            state = {"state_read_error": f"{type(exc).__name__}: {exc}"}
    state.update({
        "started_at": state.get("started_at") or started_at,
        "last_iter": last_iter,
        "last_checkpoint_path": str(ckpt) if ckpt else None,
        "last_heartbeat": _now_iso(),
    })
    state.update(extra)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    os.replace(tmp, STATE_FILE)
    return state


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _process_tree_rss_gb(pid: int) -> float:
    if psutil is None:
        return 0.0
    try:
        root = psutil.Process(pid)
        procs = [root, *root.children(recursive=True)]
    except (psutil.Error, AttributeError, RuntimeError, TypeError, ValueError):
        return 0.0
    total = 0
    for proc in procs:
        try:
            total += int(proc.memory_info().rss)
        except (psutil.Error, AttributeError, RuntimeError, TypeError, ValueError):
            continue
    return total / _GIB


def _memory_guard_reason(pid: int) -> str | None:
    if psutil is None:
        return "psutil_unavailable"
    max_tree_rss_gb = _env_float("AURA_TRAINING_MAX_PROCESS_TREE_RSS_GB", 56.0)
    max_host_percent = _env_float("AURA_TRAINING_MAX_HOST_MEMORY_PERCENT", 94.0)
    tree_rss_gb = _process_tree_rss_gb(pid)
    if tree_rss_gb >= max_tree_rss_gb:
        return f"process_tree_rss:{tree_rss_gb:.1f}GB/{max_tree_rss_gb:.1f}GB"
    try:
        host = psutil.virtual_memory()
        percent = float(getattr(host, "percent", 100.0) or 100.0)
        if percent >= max_host_percent:
            return f"host_memory_pressure:{percent:.1f}%/{max_host_percent:.1f}%"
    except (psutil.Error, AttributeError, RuntimeError, TypeError, ValueError):
        return "host_memory_probe_failed"
    return None


def _terminate_process_tree(proc) -> None:  # noqa: ANN001 - subprocess.Popen-compatible.
    if psutil is not None:
        try:
            root = psutil.Process(proc.pid)
            children = root.children(recursive=True)
            for child in reversed(children):
                try:
                    child.terminate()
                except (psutil.Error, RuntimeError, TypeError, ValueError):
                    pass
            root.terminate()
            gone, alive = psutil.wait_procs([*children, root], timeout=15)
            for child in alive:
                try:
                    child.kill()
                except (psutil.Error, RuntimeError, TypeError, ValueError):
                    pass
            return
        except (psutil.Error, AttributeError, RuntimeError, TypeError, ValueError):
            pass
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except TimeoutExpired:
        proc.kill()


def _spawn(cmd: list[str], *, started_at: str) -> int:
    """Run subprocess, heartbeat state, honour _shutdown."""
    print(f"[orch] $ {' '.join(cmd)}", flush=True)
    proc = get_subprocess_gateway().spawn(
        cmd,
        cwd=str(REPO_DIR),
        offline_tooling=True,
        source="training_tooling:run_unattended",
    )
    watchdog_interval = max(2.0, _env_float("AURA_TRAINING_WATCHDOG_INTERVAL_S", 10.0))
    try:
        while not _shutdown.is_set():
            reason = _memory_guard_reason(proc.pid)
            if reason:
                print(f"[orch] memory guard tripped — {reason}; terminating training tree")
                update_state(started_at=started_at, phase="memory_guard_kill", memory_guard_reason=reason)
                _terminate_process_tree(proc)
                return 137
            try:
                return proc.wait(timeout=watchdog_interval)
            except TimeoutExpired:
                update_state(started_at=started_at, phase="running")
        print("[orch] shutdown — terminating subprocess")
        _terminate_process_tree(proc)
        try:
            return proc.wait(timeout=30)
        except TimeoutExpired:
            proc.kill()
            return proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.terminate()


def run_train_and_fuse(args: argparse.Namespace, *, started_at: str) -> int:
    if not TRAIN_AND_FUSE.exists():
        print(f"[orch] {TRAIN_AND_FUSE.name} missing; cannot proceed.")
        return 3
    cmd: list[str] = [sys.executable, str(TRAIN_AND_FUSE)]
    if args.skip_dataset:
        cmd.append("--skip-dataset")
    if args.skip_train:
        cmd.append("--skip-train")
    if getattr(args, "crsm_delta", False):
        cmd.append("--crsm-delta")
    if getattr(args, "resume", False):
        cmd.append("--resume")
    if getattr(args, "preflight_only", False):
        cmd.append("--preflight-only")
    if args.base_model:
        cmd += ["--base-model", args.base_model]
    if args.tag:
        cmd += ["--tag", args.tag]
    update_state(started_at=started_at, phase="train_and_fuse")
    rc = _spawn(cmd, started_at=started_at)
    update_state(started_at=started_at, phase="train_and_fuse_done", last_pipeline_rc=rc)
    return rc


def run_fuse_publish(args: argparse.Namespace, *, started_at: str) -> int:
    if not TRAIN_AND_FUSE.exists():
        print(f"[orch] {TRAIN_AND_FUSE.name} missing; cannot publish fused model.")
        return 3
    cmd: list[str] = [
        sys.executable,
        str(TRAIN_AND_FUSE),
        "--skip-dataset",
        "--skip-train",
        "--mark-crsm-consumed",
    ]
    if args.base_model:
        cmd += ["--base-model", args.base_model]
    if args.tag:
        cmd += ["--tag", args.tag]
    update_state(started_at=started_at, phase="fuse_publish")
    rc = _spawn(cmd, started_at=started_at)
    update_state(started_at=started_at, phase="fuse_publish_done", last_pipeline_rc=rc)
    return rc


def run_resume(*, started_at: str) -> int:
    if not RESUME_SCRIPT.exists():
        print(f"[orch] {RESUME_SCRIPT.name} missing; falling back to train_and_fuse.")
        return -1
    cmd = [sys.executable, str(RESUME_SCRIPT)]
    update_state(started_at=started_at, phase="resume")
    rc = _spawn(cmd, started_at=started_at)
    update_state(started_at=started_at, phase="resume_done", last_resume_rc=rc)
    return rc


def _install_signal_handlers(started_at: str) -> None:
    def _handler(signum, _frame):  # noqa: ANN001
        print(f"[orch] signal {signum} — writing final snapshot")
        _shutdown.set()
        update_state(started_at=started_at, phase="signal_exit", last_signal=int(signum))

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="")
    p.add_argument("--base-model", default="")
    p.add_argument("--skip-dataset", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--crsm-delta", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    return p.parse_args(argv)


def is_dryrun(args: argparse.Namespace) -> bool:
    return args.skip_train and args.skip_dataset and args.tag.startswith("dryrun")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started_at = _now_iso()
    _install_signal_handlers(started_at)
    print(f"[orch] started_at={started_at} args={vars(args)}")
    update_state(started_at=started_at, phase="boot", args=vars(args))

    if is_dryrun(args):
        print("[orch] dryrun mode (skip-train + skip-dataset + dryrun* tag) — clean exit.")
        update_state(started_at=started_at, phase="dryrun_done")
        return 0

    if args.preflight_only:
        print("[orch] preflight-only mode — checking train/fuse safety without launching training.")
        rc = run_train_and_fuse(args, started_at=started_at)
        update_state(started_at=started_at, phase="preflight_done", last_pipeline_rc=rc)
        return rc

    if has_partial_run() and not args.skip_train:
        print("[orch] partial run detected — resuming via resume_training.py")
        rc = run_resume(started_at=started_at)
        # If resume_training.py fails to find a valid resume state (rc=1 typically),
        # fall back to train_and_fuse.py --resume which is more generic.
        if rc == 0:
            print("[orch] resume completed cleanly; fusing and publishing model.")
            rc = run_fuse_publish(args, started_at=started_at)
            if rc != 0:
                print(f"[orch] fuse/publish failed (rc={rc}) — wrapper will retry.")
                return rc
            update_state(started_at=started_at, phase="complete")
            print("[orch] resume-based pipeline completed and published cleanly.")
            return 0
        if rc != 0 and rc != -1: # rc -1 means script missing
            print(f"[orch] resume failed (rc={rc}) — falling back to train_and_fuse --resume")

        args.resume = True
    else:
        args.resume = False

    rc = run_train_and_fuse(args, started_at=started_at)
    if rc != 0:
        print(f"[orch] pipeline failed (rc={rc}) — wrapper will retry.")
        return rc

    update_state(started_at=started_at, phase="complete")
    print("[orch] pipeline completed cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
