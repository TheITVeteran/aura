"""A refuted claim is repairable even when both branches are wrong alike.

Every repair request used to descend from a DISPUTE, and a dispute exists
only where two branches' atom sequences differ -- identical branches
short-circuit to decoded_claim_graphs_exactly_equal. So when both branches
made the same mistake, an exact verifier could refute the answer and nothing
was repaired. Measured on the 32B: refuted=1, repair_requests=0, and answer
replacement reporting known_refutation_has_no_dominant_repair. The baseline
was known wrong, the system knew it, and had no way to act.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import inspect  # noqa: E402

from core.brain.llm.latent_cortex import local_repair  # noqa: E402


def test_refutation_alone_can_request_a_repair():
    src = inspect.getsource(local_repair._repair_requests)
    assert "if not repair_requests:" in src, (
        "refutation-driven repair is gone; a refuted claim is unrepairable "
        "again whenever both branches are wrong the same way"
    )
    # It must be built from a refuted EXACT route, not any refutation.
    assert 'route["outcome"] != "refuted"' in src
    assert 'route["verifier"] not in _EXACT_VERIFIERS' in src


def test_modular_refutations_are_repairable():
    """exact_modular_arithmetic must be in the allowlist, or the whole
    `modular` task family stays unrepairable -- its prompts literally say
    "modulo 19"."""
    assert "exact_modular_arithmetic" in local_repair._EXACT_VERIFIERS
    assert "exact_integer_arithmetic" in local_repair._EXACT_VERIFIERS


def test_dispute_driven_requests_still_take_precedence():
    """Refutation-driven repair is a fallback, not a replacement: when a real
    inter-branch dispute exists it still produces the request, so the pair
    evidence is preserved."""
    src = inspect.getsource(local_repair._repair_requests)
    dispute_loop = src.index("for plan in plans:")
    fallback = src.index("if not repair_requests:")
    assert dispute_loop < fallback, (
        "the refutation fallback must run after the dispute loop"
    )
