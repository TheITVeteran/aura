"""Repair admission identifies work by content, never by array position.

Six consecutive admission failures on the 32B were one defect wearing
different names. A repair replaces a span, the replacement decomposes into
its own atoms, and every index after the failure shifts -- so any check that
locates something by ordinal in the REPAIRED candidate is testing position
while claiming to test identity:

  preserved_prefix_changed        prefix atoms compared with the wrong shape
  unrelated_atom_changed          atom 16 spliced back VERBATIM, rejected
                                  because it had moved to index 17
  failed_verifier_not_rechecked   index 15 now held a different claim, so the
                                  verifier looked like it never ran

Each cost a full 32B probe cycle to diagnose. These pin the invariant.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.local_repair import (  # noqa: E402
    _failed_verifier_recheck,
    _unrelated_work_unchanged,
)


def _atom(atom_id: str, text_sha: str, ordinal: int) -> dict:
    return {
        "atom_id": atom_id,
        "atom_sha256": f"{ordinal:064d}",
        "kind": "assertion",
        "text_sha256": text_sha,
        "dependency_cues": [],
        "start": ordinal * 10,
        "end": ordinal * 10 + 9,
    }


def test_trailing_work_survives_a_shifted_index():
    """The repair replaced one atom with two, so the preserved trailing atom
    moved from ordinal 2 to ordinal 3. It is byte-identical and must pass."""
    request = {
        "failed_atom_ordinal": 1,
        "preserved_unrelated_atoms": [
            {"ordinal": 0, "atom_id": "a000", "kind": "assertion",
             "text_sha256": "aa", "dependency_cues": []},
            {"ordinal": 2, "atom_id": "a002", "kind": "assertion",
             "text_sha256": "cc", "dependency_cues": []},
        ],
    }
    repaired = {
        "atoms": [
            _atom("a000", "aa", 0),
            _atom("r001", "r1", 1),
            _atom("r002", "r2", 2),
            _atom("a002", "cc", 3),
        ]
    }
    assert _unrelated_work_unchanged(repaired, request) is True


def test_trailing_work_that_actually_changed_is_still_caught():
    """Content is the identity. A rewritten trailing atom must fail."""
    request = {
        "failed_atom_ordinal": 1,
        "preserved_unrelated_atoms": [
            {"ordinal": 2, "atom_id": "a002", "kind": "assertion",
             "text_sha256": "cc", "dependency_cues": []},
        ],
    }
    repaired = {"atoms": [_atom("a000", "aa", 0), _atom("x", "zz", 1)]}
    assert _unrelated_work_unchanged(repaired, request) is False


def test_prefix_atoms_keep_the_exact_index_comparison():
    """Before the failure the prefix is byte-identical, so ordinals hold and
    the stricter test still applies."""
    request = {
        "failed_atom_ordinal": 2,
        "preserved_unrelated_atoms": [
            {"ordinal": 0, "atom_id": "a000", "kind": "assertion",
             "text_sha256": "aa", "dependency_cues": []},
        ],
    }
    moved = {"atoms": [_atom("zzz", "different", 0), _atom("a000", "aa", 1)]}
    assert _unrelated_work_unchanged(moved, request) is False


def test_verifier_recheck_follows_the_class_not_the_ordinal():
    request = {"failed_atom_ordinal": 15, "required_verifier": "exact_integer_arithmetic"}
    routes = {"routes": [
        {"atom_id": "a000", "verifier": "none", "outcome": "unknown"},
        {"atom_id": "r001", "verifier": "exact_integer_arithmetic", "outcome": "verified"},
    ]}
    rechecked, passed = _failed_verifier_recheck(routes, request)
    assert rechecked is True and passed is True


def test_a_repair_that_is_still_wrong_does_not_pass():
    """The gate that finally fired on the 32B: structurally admissible, and
    the arithmetic still refuted."""
    request = {"failed_atom_ordinal": 15, "required_verifier": "exact_integer_arithmetic"}
    routes = {"routes": [
        {"atom_id": "r001", "verifier": "exact_integer_arithmetic", "outcome": "refuted"},
    ]}
    rechecked, passed = _failed_verifier_recheck(routes, request)
    assert rechecked is True and passed is False
