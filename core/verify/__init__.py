"""core/verify — structural verification of the runtime's own shape.

See :mod:`core.verify.invariants` for the registry and the contract.
"""

from core.verify.invariants import (
    InvariantSpec,
    Severity,
    VerifyReport,
    Violation,
    invariant,
    verify,
    verify_after,
)

__all__ = [
    "InvariantSpec",
    "Severity",
    "VerifyReport",
    "Violation",
    "invariant",
    "verify",
    "verify_after",
]
