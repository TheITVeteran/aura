"""Request-scoped exact relational principal context."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_RELATIONAL_PRINCIPAL: ContextVar[str | None] = ContextVar(
    "aura_relational_principal",
    default=None,
)


def normalize_relational_principal(value: object) -> str:
    return " ".join(str(value or "").strip().split())[:160]


def current_relational_principal() -> str:
    return _RELATIONAL_PRINCIPAL.get() or ""


def relational_principal_scope_is_bound() -> bool:
    """Return whether this causal task explicitly established a principal scope."""
    return _RELATIONAL_PRINCIPAL.get() is not None


@contextmanager
def relational_principal_scope(principal: object) -> Iterator[str]:
    """Bind an exact principal for one causal request and restore on exit."""
    normalized = normalize_relational_principal(principal)
    token = _RELATIONAL_PRINCIPAL.set(normalized)
    try:
        yield normalized
    finally:
        _RELATIONAL_PRINCIPAL.reset(token)


__all__ = [
    "current_relational_principal",
    "normalize_relational_principal",
    "relational_principal_scope_is_bound",
    "relational_principal_scope",
]
