"""Classify cloud-LLM call failures so optional cloud calls degrade, not crash.

Aura is local-first. Cloud enhancements — Gemini grounded search, cloud
fallback tiers — are optional. When the provider is unreachable, unauthorized,
or out of quota (429 RESOURCE_EXHAUSTED), the turn must fall back to the local
search/inference path, never surface an unhandled exception to the request
handler (observed live, July 2026: a grounded-search turn hit a Gemini 429 and
the ``google.genai`` ``ClientError`` was not in the caller's except tuple, so
it propagated as an "Unhandled exception [req=…]").

These call sites use delayed imports, so the provider error classes are
resolved lazily and the resolution never fails if the SDK is absent.
"""
from __future__ import annotations


def cloud_call_error_types() -> tuple[type[BaseException], ...]:
    """Exception types that mean 'the cloud call failed — degrade locally'.

    Always includes the transport/timeout/value errors a network call can
    raise; adds the ``google.genai`` error hierarchy (base ``APIError`` covers
    429/ClientError and 5xx/ServerError) when the SDK is importable.
    """
    types_: list[type[BaseException]] = [
        OSError,
        TimeoutError,
        ConnectionError,
        ValueError,
    ]
    try:
        from google.genai import errors as genai_errors  # type: ignore

        for name in ("APIError", "ClientError", "ServerError"):
            cls = getattr(genai_errors, name, None)
            if isinstance(cls, type) and issubclass(cls, BaseException):
                types_.append(cls)
    except ImportError:
        pass
    # De-duplicate while preserving order (APIError may already cover subclasses).
    seen: set[type] = set()
    unique: list[type[BaseException]] = []
    for cls in types_:
        if cls not in seen:
            seen.add(cls)
            unique.append(cls)
    return tuple(unique)


def is_cloud_call_error(exc: BaseException) -> bool:
    """True when *exc* is a cloud-call failure the caller should degrade on."""
    return isinstance(exc, cloud_call_error_types())


__all__ = ["cloud_call_error_types", "is_cloud_call_error"]
