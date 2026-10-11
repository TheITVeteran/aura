"""Read a place's counters and stage changes into its current guide.

A recognized readout is attached to a known measured mechanic. Stage changes
retain their observed chronology and the first stage establishes a baseline.
"""
from __future__ import annotations

import re
from typing import Any

#: A short label shown with a value: what is read off the screen, not pressed.
_A_READOUT = re.compile(r"^\W*([A-Za-z][A-Za-z .'-]{1,24}?)\s*[:=]?\s*(?:[\d.,/%]+|\.\.\.|…)?\W*$")
#: Writing that says which stage she is on.
_A_STAGE = re.compile(r"\b(level|stage|wave|round|mission|world|chapter)\s*:?\s*(\d{1,3})\b", re.I)


def readout_from(guide: Any, label: str) -> None:
    match = _A_READOUT.match(label or "")
    if not match or len(label.split()) > 4:
        return
    words = " ".join(match.group(1).lower().split())
    measured = guide.measures(words)
    if measured and words not in guide.readouts:
        guide.readouts[words] = measured


def stage_from(guide: Any, text: str, at: float) -> list[str]:
    found = _A_STAGE.search(text)
    if not found:
        return []
    stage = f"{found.group(1).lower()} {found.group(2)}"
    if stage == guide.stage:
        return []
    before, guide.stage = guide.stage, stage
    if not before:
        return []
    guide.changes.append((at, f"now on {stage}"))
    return [f"Now on {stage}."]
