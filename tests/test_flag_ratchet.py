"""Ratchet: raw AURA_* env reads only shrink (roadmap C1).

The typed layer is core/runtime/flags.py — declared name, type, default,
description, owner, and an enumerable live value with resolution source.
Raw ``os.environ.get("AURA_*")`` reads are untyped, undiscoverable, and
individually parsed; ~686 distinct knobs accumulated that way. This
ratchet caps the raw-read count so the sprawl can only shrink: migrate a
call site to a declared flag, lower the budget. Never raise it without
the same scrutiny as widening a permission.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]

# Exact occurrence count at ratchet introduction (July 9, 2026). ONLY goes down.
RAW_ENV_READ_BUDGET = 585

# The flag layer itself and the settings store are the sanctioned readers.
SANCTIONED = {
    "core/runtime/flags.py",
    "core/runtime/runtime_settings.py",
}

_READ_RE = re.compile(r"(?:os\.environ\.get|os\.getenv|environ\.get|\bgetenv)\(\s*[\"']AURA_")


def _count_raw_reads() -> tuple[int, dict[str, int]]:
    per_file: dict[str, int] = {}
    for scope in ("core", "interface", "api", "tools"):
        scope_path = REPO_ROOT / scope
        if not scope_path.exists():
            continue
        for path in scope_path.rglob("*.py"):
            rel = str(path.relative_to(REPO_ROOT))
            if rel in SANCTIONED:
                continue
            hits = len(_READ_RE.findall(path.read_text(encoding="utf-8", errors="ignore")))
            if hits:
                per_file[rel] = hits
    return sum(per_file.values()), per_file


def test_raw_env_reads_do_not_grow():
    total, per_file = _count_raw_reads()
    assert total <= RAW_ENV_READ_BUDGET, (
        f"raw AURA_* env reads grew to {total} (budget {RAW_ENV_READ_BUDGET}). "
        "Declare the knob in core/runtime/flags.py (declare(...)) and read "
        "through the Flag object instead. Top offenders: "
        + ", ".join(f"{f}({n})" for f, n in sorted(per_file.items(), key=lambda kv: -kv[1])[:5])
    )


def test_new_reliability_modules_are_raw_read_free():
    """The modules born in this pass must stay fully on the typed layer."""
    for rel in (
        "core/brain/lane_admission.py",
        "core/runtime/lane_reconciler.py",
        "core/runtime/fmea.py",
    ):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert not _READ_RE.search(text), f"{rel} regressed to raw env reads"
