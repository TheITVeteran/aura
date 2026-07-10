#!/usr/bin/env python3
"""tools/crash_triage.py — categorize the forensic record into incident classes.

"Crashes are rare and categorized" is a maturity claim; this makes the
categorization real. It sweeps the crash-forensics surfaces —

  * data/error_logs/stalls/stall_*.txt        (event-loop stall dumps)
  * data/error_logs/memory/sentinel.log       (guard lifecycle: orderly SIGTERMs
                                               vs targets that VANISHED)
  * data/error_logs/memory/death_syslog_*.log (hard-death syslog captures)
  * data/error_logs/crash/faulthandler.log    (fatal Python errors)
  * data/error_logs/crash/memory_spike_stacks.log (RSS spike snapshots)

— into FINGERPRINTED incident classes: a stable signature, a count, first/last
seen, and an example receipt path. Two deaths with the same anatomy are one
class with count=2, not two mysteries. The report is JSON (for the incident
narrator and dashboards) plus a human table.

Usage:
  python tools/crash_triage.py [--root data/error_logs] [--window-days 7]
      [--out artifacts/reliability/triage.json]
  make triage
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_SENTINEL_LINE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]\s+pid=\d+\s+(?P<verb>armed|exiting):\s*(?P<detail>.*)$"
)
_STALL_HEADER = re.compile(r"STALL DETECTED:\s*(?P<seconds>[0-9.]+)s")
_REPO_FRAME = re.compile(r'File "(?P<path>[^"]*/(?:core|interface|tools)/[^"]+)", line \d+, in (?P<fn>\w+)')
# Wrapper/plumbing frames that appear in EVERY thread dump and say nothing
# about where the stall actually lives — skip past them to the first frame
# that names real work.
_INFRA_FRAMES = ("runtime_hygiene.py", "aura_logging.py", "task_tracker.py", "concurrency.py")
_FATAL = re.compile(r"^Fatal Python error: (?P<what>.+)$", re.MULTILINE)
_BOOT_MARKER = re.compile(r"^===== boot pid=(?P<pid>\d+) at=(?P<at>[\d.]+)")


@dataclass
class IncidentClass:
    fingerprint: str
    kind: str                    # stall | process_death | orderly_exit | fatal_error | memory_spike
    count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    example_receipt: str = ""
    detail: dict = field(default_factory=dict)

    def observe(self, at: float, receipt: str) -> None:
        self.count += 1
        if self.first_seen == 0.0 or at < self.first_seen:
            self.first_seen = at
        if at > self.last_seen:
            self.last_seen = at
            self.example_receipt = receipt

    def to_dict(self) -> dict:
        return {
            "fingerprint": self.fingerprint,
            "kind": self.kind,
            "count": self.count,
            "first_seen": self.first_seen,
            "first_seen_iso": _iso(self.first_seen),
            "last_seen": self.last_seen,
            "last_seen_iso": _iso(self.last_seen),
            "example_receipt": self.example_receipt,
            "detail": self.detail,
        }


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds") if ts else ""


def _parse_local_ts(raw: str) -> float:
    """Sentinel timestamps look like 2026-07-08T18:59:39-0700."""
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw.strip(), fmt).timestamp()
        except ValueError:
            continue
    return 0.0


class Triage:
    def __init__(self, root: Path, *, window_days: float = 7.0, now: float | None = None) -> None:
        self.root = Path(root)
        self.now = float(now if now is not None else time.time())
        self.cutoff = self.now - window_days * 86400.0
        self.classes: dict[str, IncidentClass] = {}
        self.errors: list[str] = []

    def _cls(self, fingerprint: str, kind: str, **detail) -> IncidentClass:
        found = self.classes.get(fingerprint)
        if found is None:
            found = IncidentClass(fingerprint=fingerprint, kind=kind, detail=dict(detail))
            self.classes[fingerprint] = found
        return found

    # ── collectors ────────────────────────────────────────────────────────────

    def collect_stalls(self) -> None:
        for dump in sorted((self.root / "stalls").glob("stall_*.txt")):
            at = dump.stat().st_mtime
            if at < self.cutoff:
                continue
            try:
                text = dump.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                self.errors.append(f"stall_read:{dump.name}:{exc}")
                continue
            header = _STALL_HEADER.search(text)
            seconds = float(header.group("seconds")) if header else 0.0
            where = "unknown_frame"
            for frame in _REPO_FRAME.finditer(text):
                name = Path(frame.group("path")).name
                if name in _INFRA_FRAMES:
                    continue
                where = f"{name}:{frame.group('fn')}"
                break
            bucket = "5-10s" if seconds < 10 else ("10-30s" if seconds < 30 else "30s+")
            cls = self._cls(f"stall:{where}:{bucket}", "stall", top_frame=where, bucket=bucket)
            cls.observe(at, str(dump))

    def collect_sentinel(self) -> None:
        path = self.root / "memory" / "sentinel.log"
        if not path.is_file():
            return
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            self.errors.append(f"sentinel_read:{exc}")
            return
        for line in lines:
            m = _SENTINEL_LINE.match(line.strip())
            if not m or m.group("verb") != "exiting":
                continue
            at = _parse_local_ts(m.group("ts"))
            if at < self.cutoff:
                continue
            detail = m.group("detail")
            if "vanished" in detail:
                # A target that VANISHED is a hard, uncommanded process death.
                cls = self._cls("process_death:target_vanished", "process_death")
            elif "SIGTERM" in detail:
                # Commanded teardown (launch cycles, reboots) — noise unless
                # the rate is absurd, but the rate itself is worth seeing.
                cls = self._cls("orderly_exit:sigterm_guard", "orderly_exit")
            elif "lethal" in detail.lower():
                cls = self._cls("process_death:lethal_memory", "process_death")
            else:
                cls = self._cls("orderly_exit:other", "orderly_exit")
            cls.observe(at, f"{path}#{m.group('ts')}")

    def collect_death_syslogs(self) -> None:
        for cap in sorted((self.root / "memory").glob("death_syslog_*.log")):
            at = cap.stat().st_mtime
            if at < self.cutoff:
                continue
            cls = self._cls("process_death:syslog_capture", "process_death")
            cls.observe(at, str(cap))

    def collect_faulthandler(self) -> None:
        """Date each fatal segment by its PRECEDING boot marker, never by file
        mtime — the log is append-only across every boot, so mtime is always
        "now" and would pull months-old segfaults into the current window
        (which is exactly the false alarm the first live run of this tool
        raised: six June segfaults reported as this week's)."""
        path = self.root / "crash" / "faulthandler.log"
        if not path.is_file():
            return
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            self.errors.append(f"faulthandler_read:{exc}")
            return
        segment_at = path.stat().st_mtime  # fallback for logs predating markers
        segment_pid = "?"
        for line in lines:
            marker = _BOOT_MARKER.match(line)
            if marker:
                segment_at = float(marker.group("at"))
                segment_pid = marker.group("pid")
                continue
            fatal = _FATAL.match(line)
            if fatal and segment_at >= self.cutoff:
                what = fatal.group("what").strip()[:60]
                cls = self._cls(f"fatal_error:{what}", "fatal_error")
                cls.observe(segment_at, f"{path}#pid={segment_pid}:'{what}'")

    def collect_memory_spikes(self) -> None:
        path = self.root / "crash" / "memory_spike_stacks.log"
        if not path.is_file():
            return
        at = path.stat().st_mtime
        if at < self.cutoff:
            return
        try:
            spikes = sum(
                1 for line in path.open(encoding="utf-8", errors="replace")
                if "MEMORY SPIKE" in line or "SPIKE DETECTED" in line
            )
        except OSError as exc:
            self.errors.append(f"spike_read:{exc}")
            return
        if spikes:
            cls = self._cls("memory_spike:rss_spike", "memory_spike", events_in_file=spikes)
            cls.observe(at, str(path))
            cls.count = spikes  # count = spike events, not files

    # ── report ────────────────────────────────────────────────────────────────

    def run(self) -> dict:
        for collector in (
            self.collect_stalls,
            self.collect_sentinel,
            self.collect_death_syslogs,
            self.collect_faulthandler,
            self.collect_memory_spikes,
        ):
            try:
                collector()
            except Exception as exc:  # noqa: BLE001 - triage must survive any single bad surface
                self.errors.append(f"{collector.__name__}:{type(exc).__name__}:{exc}")

        ranked = sorted(
            self.classes.values(),
            key=lambda c: ({"process_death": 0, "fatal_error": 1, "stall": 2,
                            "memory_spike": 3, "orderly_exit": 4}.get(c.kind, 9), -c.count),
        )
        return {
            "schema": "aura.crash_triage.v1",
            "generated_at": self.now,
            "generated_at_iso": _iso(self.now),
            "root": str(self.root),
            "window_days": round((self.now - self.cutoff) / 86400.0, 2),
            "class_count": len(ranked),
            "hard_death_total": sum(c.count for c in ranked if c.kind == "process_death"),
            "classes": [c.to_dict() for c in ranked],
            "collector_errors": self.errors,
        }


def compute_trend(report: dict, previous: dict) -> dict:
    """Crashpad-style trend: what appeared, what resolved, what moved.

    Keyed by fingerprint so a class is tracked across runs even as its
    count and last-seen change.
    """
    prev_classes = {c["fingerprint"]: c for c in previous.get("classes", [])}
    cur_classes = {c["fingerprint"]: c for c in report.get("classes", [])}
    shared = set(cur_classes) & set(prev_classes)
    return {
        "previous_generated_at": previous.get("generated_at"),
        "previous_generated_at_iso": previous.get("generated_at_iso"),
        "new_classes": sorted(set(cur_classes) - set(prev_classes)),
        "resolved_classes": sorted(set(prev_classes) - set(cur_classes)),
        "count_deltas": {
            fp: cur_classes[fp]["count"] - prev_classes[fp]["count"]
            for fp in sorted(shared)
            if cur_classes[fp]["count"] != prev_classes[fp]["count"]
        },
    }


def append_history(out_path: Path, report: dict) -> None:
    """One compact line per triage run — the long-term trend record."""
    history_path = out_path.with_name("triage_history.jsonl")
    trend = report.get("trend") or {}
    line = json.dumps(
        {
            "generated_at": report["generated_at"],
            "generated_at_iso": report["generated_at_iso"],
            "window_days": report["window_days"],
            "class_count": report["class_count"],
            "hard_death_total": report["hard_death_total"],
            "new_classes": trend.get("new_classes", []),
            "resolved_classes": trend.get("resolved_classes", []),
        },
        sort_keys=True,
    )
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def render_table(report: dict) -> str:
    lines = [
        f"crash triage — window {report['window_days']}d, "
        f"{report['class_count']} classes, "
        f"{report['hard_death_total']} hard deaths",
        f"{'KIND':<14} {'COUNT':>5}  {'LAST SEEN':<19}  FINGERPRINT",
    ]
    for c in report["classes"]:
        lines.append(
            f"{c['kind']:<14} {c['count']:>5}  {c['last_seen_iso']:<19}  {c['fingerprint']}"
        )
    if report["collector_errors"]:
        lines.append(f"collector errors: {report['collector_errors']}")
    trend = report.get("trend")
    if trend:
        lines.append(
            f"trend vs {trend.get('previous_generated_at_iso', 'previous run')}: "
            f"{len(trend.get('new_classes', []))} new, "
            f"{len(trend.get('resolved_classes', []))} resolved, "
            f"{len(trend.get('count_deltas', {}))} moved"
        )
        for fp in trend.get("new_classes", []):
            lines.append(f"  NEW      {fp}")
        for fp in trend.get("resolved_classes", []):
            lines.append(f"  RESOLVED {fp}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO_ROOT / "data" / "error_logs"))
    parser.add_argument("--window-days", type=float, default=7.0)
    parser.add_argument("--out", default="", help="optional JSON report path")
    args = parser.parse_args()

    report = Triage(Path(args.root), window_days=args.window_days).run()
    if args.out:
        out = Path(args.out)
        if out.exists():
            try:
                previous = json.loads(out.read_text(encoding="utf-8"))
                report["trend"] = compute_trend(report, previous)
            except (json.JSONDecodeError, OSError, KeyError, TypeError) as exc:
                print(f"trend unavailable (previous report unreadable): {exc}")
    print(render_table(report))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        try:
            append_history(out, report)
        except OSError as exc:
            print(f"history append failed: {exc}")
        print(f"\nreport: {out}")
    # Exit 1 when hard deaths exist in-window: wire-able into CI/cron alerts.
    return 1 if report["hard_death_total"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
