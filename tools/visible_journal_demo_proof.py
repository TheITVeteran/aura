#!/usr/bin/env python3
"""Visible journal demo proof: watch her do it, then verify the disk.

Bryan's demo part 1 with the "and I want to see you do it" clause: from
one request Aura must VISIBLY open Notes and stage the entry there
(open_app + new-note + paste — real UI motion on the real screen), and
ALSO land the durable artifacts: "Aura's Journal" folder, timestamped
self-describing entry, robot image found online, PDF inside the folder.

Boots the runtime in desktop mode (--desktop). Contains NO task-specific
execution logic — only the objective sentence and external verification:
- the same hostile disk checks as the journal proof (fresh PDF, text
  layer with timestamp + first-person self-description, embedded image);
- PLUS receipt checks that the visible staging actually dispatched:
  an open_app step targeting Notes and a paste hotkey step, both ok,
  pulled from the lane's skill receipts (claims are never trusted).

Usage:
    python tools/visible_journal_demo_proof.py [--port 8000] [--boot-timeout 600]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.journal_demo_proof import (  # noqa: E402
    _SELF_TOKENS,
    _TIMESTAMP_RE,
    JournalDemoProof,
    _pdf_has_image,
    _pdf_text,
)

VISIBLE_OBJECTIVE = (
    "Please open up my Notes app and write a short journal entry in your "
    "own words describing who and what you are — I want to see you do it. "
    "Include the current date and time inside the entry text. Find an "
    "image of a robot online and include it in the entry. Then save the "
    "finished entry as a PDF inside a new folder called 'Aura's Journal' "
    "in my Documents folder. Tell me exactly what you did at each step."
)


def _iter_dicts(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _iter_dicts(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _iter_dicts(item)


def _step_records(lane_data: Any) -> list[dict]:
    """Pull executed-step records (action + ok) out of the skill payload."""
    records = []
    for d in _iter_dicts(lane_data):
        action = str(d.get("action") or "").strip().lower()
        if action and ("ok" in d or "status" in d):
            records.append(d)
    return records


def _visible_staging_evidence(lane_data: Any) -> dict[str, bool]:
    notes_opened = False
    paste_dispatched = False
    for record in _step_records(lane_data):
        action = str(record.get("action") or "").lower()
        target = str(
            record.get("target")
            or record.get("app")
            or record.get("requested_target")
            or ""
        ).lower()
        ok = bool(record.get("ok")) or str(record.get("status") or "").lower() in {
            "ok",
            "success",
            "completed",
        }
        if action == "open_app" and "notes" in target and ok:
            notes_opened = True
        if action == "hotkey" and ("v" in target or "paste" in target) and ok:
            paste_dispatched = True
    return {"notes_opened": notes_opened, "paste_dispatched": paste_dispatched}


class VisibleJournalProof(JournalDemoProof):
    def __init__(self, *, port: int, boot_timeout_s: float):
        super().__init__(port=port, boot_timeout_s=boot_timeout_s, mode="desktop")

    def exercise_visible_journal_chain(self) -> bool:
        journal_dir = Path.home() / "Documents" / "Aura's Journal"
        step_started = time.time()
        ok, reply, latency, payload = self.chat_full(VISIBLE_OBJECTIVE, timeout_s=600.0)
        self.guard_rss()
        time.sleep(2.0)

        pdfs = sorted(
            journal_dir.glob("*.pdf"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ) if journal_dir.is_dir() else []
        fresh_pdfs = [p for p in pdfs if p.stat().st_mtime >= step_started - 1.0]
        pdf = fresh_pdfs[0] if fresh_pdfs else None

        text = _pdf_text(pdf) if pdf else ""
        lowered = text.lower()
        has_timestamp = bool(_TIMESTAMP_RE.search(text))
        has_self = any(tok in lowered for tok in _SELF_TOKENS) and len(text) > 120
        has_image = _pdf_has_image(pdf) if pdf else False
        staging = _visible_staging_evidence(payload.get("data"))

        verified = bool(
            ok
            and pdf
            and has_timestamp
            and has_self
            and has_image
            and staging["notes_opened"]
            and staging["paste_dispatched"]
        )
        return self.record(
            "visible_journal_chain",
            verified,
            summary=(
                f"{latency:.1f}s — "
                + (
                    f"PDF {pdf.name} ({pdf.stat().st_size // 1024}KB) "
                    f"text={len(text)}ch ts={has_timestamp} self={has_self} "
                    f"image={has_image} staging={staging}"
                    if pdf
                    else f"NO FRESH PDF (staging={staging})"
                )
            ),
            latency_s=round(latency, 1),
            reply=reply[:1200],
            lane_data=payload.get("data"),
            lane_status=payload.get("status"),
            pdf_path=str(pdf) if pdf else "",
            pdf_bytes=pdf.stat().st_size if pdf else 0,
            pdf_text_chars=len(text),
            pdf_text_head=text[:400],
            has_timestamp=has_timestamp,
            has_self_description=has_self,
            has_embedded_image=has_image,
            notes_opened=staging["notes_opened"],
            paste_dispatched=staging["paste_dispatched"],
        )

    def run(self) -> int:  # noqa: D102 - sequence mirrors JournalDemoProof.run
        try:
            if not self.boot():
                return 1
            self.snapshot_vitals()
            chain_ok = self.exercise_visible_journal_chain()
            self.snapshot_vitals()
            shutdown_ok = self.shutdown()
            verdict = {
                "proof": "visible_journal_demo",
                "passed": bool(chain_ok and shutdown_ok),
                "steps": self.steps,
            }
            self.verdict_path.write_text(json.dumps(verdict, indent=2, default=str))
            print(
                (
                    "✅ VISIBLE JOURNAL DEMO PROOF PASSED"
                    if verdict["passed"]
                    else "❌ VISIBLE JOURNAL DEMO PROOF FAILED"
                )
                + f" → {self.verdict_path}"
            )
            return 0 if verdict["passed"] else 1
        finally:
            self.kill_hard()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--boot-timeout", type=float, default=600.0)
    args = parser.parse_args(argv)
    return VisibleJournalProof(port=args.port, boot_timeout_s=args.boot_timeout).run()


if __name__ == "__main__":
    sys.exit(main())
