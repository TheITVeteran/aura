"""Ratchet: the repo root stays a front door, not a lab notebook.

July critique: 'the repo still blends real runtime, tests, proof bundles,
scoping notes, and research narrative.' Historical/closeout/research docs now
live in docs/evidence/. This ratchet pins the root to an explicit allowlist —
adding a root document is a conscious decision reviewed here, and moving one
back from docs/evidence/ fails loudly.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# The front door: first-contact, operational policy, cards, and the ACTIVE
# claims/evidence surface. Nothing point-in-time, nothing narrative.
ALLOWED_ROOT_DOCS = frozenset({
    "README.md", "ARCHITECTURE.md", "HOW_IT_WORKS.md", "INSTALL.md",
    "TESTING.md", "CONTRIBUTING.md", "CLAUDE.md", "SECURITY.md", "ROADMAP.md",
    "EVALUATE_AURA.md", "ARTIFACT_INDEX.md",
    "CLAIMS_MATRIX.md", "CLAIMS_SUPPORTED.md", "CLAIMS_NOT_SUPPORTED.md",
    "MODEL_CARD.md", "DATA_CARD.md", "AI_SYSTEM_CARD.md", "MEMORY_CARD.md",
    "HARDWARE_PROFILES.md", "KNOWN_FAILURE_MODES.md",
    "AUTONOMY_BOUNDARIES.md", "HUMAN_OVERRIDE_POLICY.md", "TOOL_USE_POLICY.md",
    "OWNERSHIP.md", "SERVICE_OWNERSHIP.md",
    # Front-door by definition, and deliberately allowlisted rather than moved:
    # AGENTS.md is the instruction surface an agent reads before touching this
    # repo (the sibling of CLAUDE.md), and a CHANGELOG belongs at the root of
    # any project that ships releases — filing either under docs/evidence/
    # would hide it from the reader it exists for.
    "AGENTS.md", "CHANGELOG.md",
})

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_root_markdown_is_allowlisted():
    actual = {p.name for p in REPO_ROOT.glob("*.md")}
    strays = actual - ALLOWED_ROOT_DOCS
    assert not strays, (
        f"unexpected root doc(s) {sorted(strays)} — historical/proof/research "
        "documents belong in docs/evidence/; if this is genuinely a front-door "
        "document, add it to ALLOWED_ROOT_DOCS deliberately"
    )


def test_evidence_directory_is_indexed():
    evidence = REPO_ROOT / "docs" / "evidence"
    assert (evidence / "README.md").exists()
    index = (evidence / "README.md").read_text(encoding="utf-8")
    for doc in evidence.glob("*.md"):
        if doc.name == "README.md":
            continue
        assert doc.name in index, f"docs/evidence/{doc.name} missing from the index"


def test_moved_docs_left_no_dangling_references():
    """No file outside docs/evidence still links to the old root paths."""
    moved = [p.name for p in (REPO_ROOT / "docs" / "evidence").glob("*.md")
             if p.name != "README.md"]
    offenders: list[str] = []
    for scope in ("*.md", "tools/*.py", "core/**/*.py", "interface/**/*.py"):
        for path in REPO_ROOT.glob(scope):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for name in moved:
                # a bare root-relative reference (not the new docs/evidence path)
                if f"`{name}`" in text or f"]({name})" in text or f'"{name}"' in text:
                    offenders.append(f"{path.relative_to(REPO_ROOT)} -> {name}")
    assert not offenders, f"dangling references to moved docs: {offenders[:10]}"
