"""Domain truth-engines for Aura's reasoning amplifier.

Frontier-class local reasoning does not come from a bigger network — it comes from
refusing to accept an unverified answer. These verifiers are the *truth engines* the
amplifier dispatches to per task type: code (compile/AST/lint), math (sympy/z3/exact
arithmetic), logic (natural deduction), repo-evidence (the answer's file/symbol
references must actually exist), citation (factual claims need grounding) and planning
(a plan's structure/preconditions). Each returns a uniform :class:`VerificationResult`
so the amplifier can filter candidates and attach a reasoning receipt.

Everything here is pure-or-read-only: static analysis, exact symbolic checks, and
filesystem evidence. Untrusted *execution* lives in :mod:`core.brain.symbolic_sandbox`
behind effect governance — these engines gather evidence, they do not act.
"""
from __future__ import annotations

from .base import VerificationResult, Verifier, combine_results
from .registry import VerifierRegistry, get_verifier_registry, verify_candidate

__all__ = [
    "VerificationResult",
    "Verifier",
    "combine_results",
    "VerifierRegistry",
    "get_verifier_registry",
    "verify_candidate",
]
