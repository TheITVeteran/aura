"""mypy strict ratchet — the allowlist only grows.

config/mypy_strict_files.txt is the set of files proven clean under mypy
strict; `make typecheck` reads it. This test enforces the ratchet: every
listed file exists and typechecks clean (via mypy.api, in-process), and
the list never silently shrinks — lowering MIN_STRICT_FILES is a loud,
reviewed act, exactly like RAW_GET_BUDGET and the async-write allowlist.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = REPO_ROOT / "config" / "mypy_strict_files.txt"

# Only grows. 10 original + 21 maturity-pass spine/tool files + 3 quantum
# files + 5 persistent-world files + 4 incident/device-boundary files +
# self-code transaction + desktop effect verification/host automation skill +
# bounded multimodal event-time fusion + canonical legacy sensory adapter +
# consented visual-only speech recognition + calibrated live social situation +
# identity-scoped relational-memory authority and compatibility adapters +
# authority-backed conversational-profile, dialogue-cognition, relational-
# intelligence, social-imagination, delivered-outcome humor, and receipt-confirmed
# user output transport, exact-agent user profiles, semantic fact provenance,
# principal-bound paired devices, authenticated principal resolution, and
# exact-agent chat-turn profile scheduling, canonical relationship topology,
# and the exact-agent person compatibility view.
MIN_STRICT_FILES = 71

MYPY_FLAGS = ["--follow-imports=skip", "--explicit-package-bases"]


def _allowlisted_files() -> list[str]:
    lines = ALLOWLIST.read_text().splitlines()
    return [
        ln.strip() for ln in lines
        if ln.strip() and not ln.strip().startswith("#")
    ]


def test_allowlist_meets_minimum():
    files = _allowlisted_files()
    assert len(files) >= MIN_STRICT_FILES, (
        f"mypy strict allowlist shrank to {len(files)} (< {MIN_STRICT_FILES}). "
        "The ratchet only grows: fix the file instead of delisting it, or "
        "shrink MIN_STRICT_FILES here with justification in the same commit."
    )


def test_allowlisted_files_exist():
    missing = [f for f in _allowlisted_files() if not (REPO_ROOT / f).exists()]
    assert not missing, f"allowlisted files missing from the tree: {missing}"


def test_no_duplicate_entries():
    files = _allowlisted_files()
    assert len(files) == len(set(files)), "duplicate entries pad the count"


def test_every_allowlisted_file_typechecks_strict():
    """The actual enforcement: one in-process mypy run over the whole list.

    mypy.api avoids a subprocess (enterprise gate) and reuses the local
    .mypy_cache, so warm runs are cheap.
    """
    from mypy import api as mypy_api

    files = [str(REPO_ROOT / f) for f in _allowlisted_files()]
    stdout, stderr, exit_code = mypy_api.run(MYPY_FLAGS + files)
    assert exit_code == 0, (
        "mypy strict regression in the allowlist:\n"
        f"{stdout}\n{stderr}\n"
        "Fix the types — do not delist the file."
    )
