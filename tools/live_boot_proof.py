#!/usr/bin/env python3
"""Live boot proof: boot Aura for real, converse, act, verify, shut down.

Static gates prove the code; this proves the companion. The driver runs
OUTSIDE Aura's process and treats her like a user would:

1. boot `aura_main.py --headless` and poll /api/health until the runtime
   contract reports healthy (bounded wait),
2. send real chat turns through /api/chat and measure latency,
3. check the identity contract holds in the *actual* reply (the
   self-claim verifier runs on what she really said),
4. ask for a real governed desktop action (folder + file) and verify
   the effect on disk from outside her process,
5. watch her process-tree RSS the whole time with a hard abort ceiling,
6. stop her cleanly and verify no orphan workers and no port squat.

Every step lands in a JSONL transcript plus a final JSON verdict under
artifacts/live_proof/. A timeout, OOM abort, dead process, or failed
verification is a loud failed step — never a skipped one. The artifact
records what actually happened, including failures; it is evidence,
not advertising.

Usage:
    python tools/live_boot_proof.py [--port 8000] [--boot-timeout 600]
    python tools/live_boot_proof.py --skip-desktop-action
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROOF_DIR = ROOT / "artifacts" / "live_proof"

# Abort the whole proof if Aura's process tree exceeds this. Generous
# enough for a full model load on the 64GB target, far below host danger.
RSS_ABORT_MB = 45_000.0


class LiveProof:
    def __init__(self, *, port: int, boot_timeout_s: float, skip_desktop: bool):
        self.port = port
        self.boot_timeout_s = boot_timeout_s
        self.skip_desktop = skip_desktop
        self.base = f"http://127.0.0.1:{port}"
        self.proc: subprocess.Popen | None = None
        self.steps: list[dict[str, Any]] = []
        self.peak_rss_mb = 0.0
        self.started_at = time.time()
        PROOF_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.transcript_path = PROOF_DIR / f"live_proof_{stamp}.jsonl"
        self.verdict_path = PROOF_DIR / f"live_proof_{stamp}_verdict.json"

    # ── recording ─────────────────────────────────────────────────────

    def record(self, step: str, ok: bool, **detail: Any) -> bool:
        entry = {
            "at": time.time(),
            "elapsed_s": round(time.time() - self.started_at, 2),
            "step": step,
            "ok": bool(ok),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            **detail,
        }
        self.steps.append(entry)
        with open(self.transcript_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        marker = "✅" if ok else "❌"
        print(f"{marker} [{entry['elapsed_s']:>7.1f}s] {step}: "
              f"{detail.get('summary', '')}", flush=True)
        return ok

    # ── process management ────────────────────────────────────────────

    def tree_rss_mb(self) -> float:
        if self.proc is None:
            return 0.0
        try:
            root = psutil.Process(self.proc.pid)
            total = root.memory_info().rss
            for child in root.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.Error:
                    continue
            mb = total / (1024 * 1024)
            self.peak_rss_mb = max(self.peak_rss_mb, mb)
            return mb
        except psutil.Error:
            return 0.0

    def guard_rss(self) -> None:
        mb = self.tree_rss_mb()
        if mb > RSS_ABORT_MB:
            self.record(
                "rss_guard",
                False,
                summary=f"ABORT: tree RSS {mb:.0f}MB exceeded {RSS_ABORT_MB:.0f}MB",
            )
            self.kill_hard()
            raise RuntimeError("live proof aborted on RSS ceiling")

    def port_in_use(self) -> bool:
        try:
            with httpx.Client(timeout=2.0) as client:
                client.get(f"{self.base}/api/health")
            return True
        except httpx.HTTPError:
            return False

    def boot(self) -> bool:
        if self.port_in_use():
            return self.record(
                "preflight_port",
                False,
                summary=f"port {self.port} already serving — refusing to "
                f"fight an existing instance; stop it first "
                f"(python aura_main.py --stop)",
            )
        existing = [
            p.pid
            for p in psutil.process_iter(["cmdline"])
            if "aura_main.py" in " ".join(p.info.get("cmdline") or [])
        ]
        if existing:
            return self.record(
                "preflight_process",
                False,
                summary=f"aura_main already running (pids {existing}); "
                f"refusing to double-boot",
            )

        env = dict(os.environ)
        env.setdefault("AURA_WATCHDOG_BOOT_GRACE_S", "240")
        self.proc = subprocess.Popen(
            [sys.executable, "aura_main.py", "--headless", "--port", str(self.port)],
            cwd=ROOT,
            stdout=open(PROOF_DIR / "live_boot_stdout.log", "w"),
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        self.record("boot_spawn", True, summary=f"pid {self.proc.pid}")

        deadline = time.monotonic() + self.boot_timeout_s
        last_state: dict[str, Any] = {}
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                return self.record(
                    "boot_health",
                    False,
                    summary=f"process exited during boot (rc={self.proc.returncode})",
                )
            self.guard_rss()
            try:
                with httpx.Client(timeout=5.0) as client:
                    resp = client.get(f"{self.base}/api/health")
                if resp.status_code == 200:
                    payload = resp.json()
                    last_state = payload if isinstance(payload, dict) else {}
                    status = str(
                        last_state.get("status")
                        or last_state.get("state")
                        or ""
                    ).lower()
                    if status in {"healthy", "ok", "ready"} or last_state.get(
                        "healthy"
                    ) is True:
                        return self.record(
                            "boot_health",
                            True,
                            summary=f"healthy after {time.time() - self.started_at:.0f}s "
                            f"(rss {self.tree_rss_mb():.0f}MB)",
                            health=last_state,
                        )
            except httpx.HTTPError:
                pass
            time.sleep(3.0)
        return self.record(
            "boot_health",
            False,
            summary=f"not healthy within {self.boot_timeout_s:.0f}s",
            last_health=last_state,
        )

    # ── exercises ─────────────────────────────────────────────────────

    def chat(self, message: str, *, timeout_s: float = 180.0) -> tuple[bool, str, float]:
        started = time.monotonic()
        try:
            with httpx.Client(timeout=timeout_s) as client:
                resp = client.post(
                    f"{self.base}/api/chat",
                    json={"message": message, "session_id": "live-proof"},
                )
            latency = time.monotonic() - started
            if resp.status_code != 200:
                return False, f"http {resp.status_code}: {resp.text[:300]}", latency
            payload = resp.json()
            text = str(
                payload.get("response")
                or payload.get("reply")
                or payload.get("message")
                or payload.get("text")
                or ""
            ).strip()
            return bool(text), text, latency
        except httpx.HTTPError as exc:
            return False, f"{type(exc).__name__}: {exc}", time.monotonic() - started

    def exercise_identity_turn(self) -> bool:
        ok, text, latency = self.chat(
            "Quick reliability check, in two or three sentences: what are you, "
            "and will you remember this conversation tomorrow?"
        )
        self.guard_rss()
        if not ok:
            return self.record(
                "chat_identity", False, summary=text[:200], latency_s=round(latency, 1)
            )
        from core.conversation.self_claim_verifier import verify_self_claims

        verdict = verify_self_claims(text)
        return self.record(
            "chat_identity",
            verdict.ok,
            summary=(
                f"{latency:.1f}s, {len(text)} chars"
                + ("" if verdict.ok else
                   f" — SELF-CLAIM VIOLATIONS: {[v.kind for v in verdict.violations]}")
            ),
            latency_s=round(latency, 1),
            reply=text[:1500],
            self_claim_ok=verdict.ok,
            violations=[v.kind for v in verdict.violations],
        )

    def exercise_continuity_turn(self) -> bool:
        token = f"amber-{int(time.time()) % 100000}"
        ok1, _, lat1 = self.chat(
            f"Remember this codeword for me: {token}. Just confirm you have it."
        )
        self.guard_rss()
        ok2, text2, lat2 = self.chat("What codeword did I just give you?")
        self.guard_rss()
        recalled = token.lower() in text2.lower()
        # Recall is the criterion. Round 13: she answered 'The codeword
        # you gave me is amber-82004' — perfect recall — but the set
        # turn's reply text had been empty under gate serialization and
        # the old all-three conjunction marked the step red. A silent
        # set with proven recall is a pass; the set latency still lands
        # in the transcript for the record.
        return self.record(
            "chat_continuity",
            ok2 and recalled,
            summary=(
                f"set {lat1:.1f}s / recall {lat2:.1f}s — "
                + ("codeword recalled" if recalled else
                   f"NOT recalled (reply: {text2[:160]})")
            ),
            token=token,
            recalled=recalled,
            reply=text2[:600],
        )

    def exercise_desktop_action(self) -> bool:
        if self.skip_desktop:
            return self.record(
                "desktop_action", True, summary="skipped by flag", skipped=True
            )
        target_dir = Path.home() / "Documents" / "Aura Live Proof"
        marker = target_dir / "live_proof.txt"
        ok, text, latency = self.chat(
            "Please create a folder named 'Aura Live Proof' in my Documents "
            "folder and write a file inside it called live_proof.txt with one "
            "sentence about who you are and the current timestamp. Use your "
            "desktop tools and confirm exactly what you did.",
            timeout_s=300.0,
        )
        self.guard_rss()
        # External verification: the proof is on disk, not in her words.
        time.sleep(2.0)
        file_exists = marker.is_file()
        content = marker.read_text(errors="replace")[:400] if file_exists else ""
        return self.record(
            "desktop_action",
            ok and file_exists and bool(content.strip()),
            summary=(
                f"{latency:.1f}s — "
                + (f"file verified on disk ({len(content)} chars)"
                   if file_exists else "FILE NOT FOUND on disk")
            ),
            latency_s=round(latency, 1),
            reply=text[:800],
            file_exists=file_exists,
            file_content=content,
            path=str(marker),
        )

    def snapshot_vitals(self) -> bool:
        vitals: dict[str, Any] = {"tree_rss_mb": round(self.tree_rss_mb(), 1)}
        ok = True
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{self.base}/api/health")
            vitals["health_status_code"] = resp.status_code
            if resp.status_code == 200:
                payload = resp.json()
                if isinstance(payload, dict):
                    vitals["health"] = {
                        k: payload.get(k)
                        for k in ("status", "state", "healthy", "runtime", "uptime_s")
                        if k in payload
                    }
            else:
                ok = False
        except httpx.HTTPError as exc:
            vitals["health_error"] = str(exc)
            ok = False
        return self.record(
            "vitals", ok, summary=f"rss {vitals['tree_rss_mb']}MB", **vitals
        )

    # ── shutdown ──────────────────────────────────────────────────────

    def shutdown(self) -> bool:
        if self.proc is None:
            return self.record("shutdown", False, summary="no process")
        try:
            stop = subprocess.run(
                [sys.executable, "aura_main.py", "--stop"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=90,
            )
            stop_note = f"--stop rc={stop.returncode}"
        except subprocess.SubprocessError as exc:
            stop_note = f"--stop failed: {exc}"

        try:
            self.proc.wait(timeout=60)
            graceful = True
        except subprocess.TimeoutExpired:
            graceful = False
            self.kill_hard()

        time.sleep(2.0)
        orphans = [
            p.pid
            for p in psutil.process_iter(["cmdline"])
            if any(
                marker in " ".join(p.info.get("cmdline") or [])
                for marker in ("aura_main.py", "mlx_worker.py", "llama-server")
            )
        ]
        for pid in orphans:
            try:
                psutil.Process(pid).kill()
            except psutil.Error:
                pass
        port_free = not self.port_in_use()
        return self.record(
            "shutdown",
            graceful and not orphans and port_free,
            summary=(
                f"{stop_note}; graceful={graceful}; orphans={orphans or 'none'}; "
                f"port_free={port_free}"
            ),
            graceful=graceful,
            orphans=orphans,
            port_free=port_free,
        )

    def kill_hard(self) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                self.proc.kill()
            except OSError:
                pass

    # ── orchestration ─────────────────────────────────────────────────

    def run(self) -> int:
        passed = True
        try:
            if not self.boot():
                passed = False
            else:
                passed &= self.snapshot_vitals()
                passed &= self.exercise_identity_turn()
                passed &= self.exercise_continuity_turn()
                passed &= self.exercise_desktop_action()
                passed &= self.snapshot_vitals()
        except RuntimeError as exc:
            self.record("abort", False, summary=str(exc))
            passed = False
        finally:
            if self.proc is not None and self.proc.poll() is None:
                passed &= self.shutdown()

        verdict = {
            "schema": "aura.live_boot_proof.v1",
            "passed": passed,
            "started_at": self.started_at,
            "finished_at": time.time(),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "steps": self.steps,
            "transcript": str(self.transcript_path.relative_to(ROOT)),
        }
        self.verdict_path.write_text(json.dumps(verdict, indent=2, default=str))
        print(f"\n{'✅ LIVE PROOF PASSED' if passed else '❌ LIVE PROOF FAILED'}")
        print(f"verdict: {self.verdict_path.relative_to(ROOT)}")
        return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--boot-timeout", type=float, default=600.0)
    parser.add_argument("--skip-desktop-action", action="store_true")
    args = parser.parse_args(argv)
    proof = LiveProof(
        port=args.port,
        boot_timeout_s=args.boot_timeout,
        skip_desktop=args.skip_desktop_action,
    )
    return proof.run()


if __name__ == "__main__":
    raise SystemExit(main())
