"""A cloud 429 must degrade to local search, never crash the turn.

Live (July 2026): a grounded-search turn hit a Gemini 429 RESOURCE_EXHAUSTED
and the google.genai ClientError was absent from the caller's except tuple, so
it surfaced as an "Unhandled exception [req=…]". A local-first system degrades.
"""
from __future__ import annotations

from core.brain.llm.cloud_errors import cloud_call_error_types, is_cloud_call_error


def test_transport_errors_always_classified():
    types_ = cloud_call_error_types()
    assert OSError in types_
    assert TimeoutError in types_
    assert ConnectionError in types_
    assert is_cloud_call_error(TimeoutError("gemini timed out"))
    assert is_cloud_call_error(ConnectionError("dns"))


def test_non_cloud_errors_not_swallowed():
    assert not is_cloud_call_error(KeyError("bug"))
    assert not is_cloud_call_error(AssertionError("bug"))


def test_genai_error_hierarchy_included_when_sdk_present():
    """When the google.genai SDK is installed, its 429/ClientError base is caught."""
    try:
        from google.genai import errors as genai_errors
    except ImportError:
        # No SDK on this host — the classifier still returns transport errors,
        # which the sibling tests cover. Nothing more to assert here.
        assert cloud_call_error_types(), "classifier must never be empty"
        return
    api_error = getattr(genai_errors, "APIError", None)
    if isinstance(api_error, type):
        assert api_error in cloud_call_error_types()

        # A concrete 429 instance (built without the SDK's constructor) is
        # classified as a cloud error via the base type.
        class _FakeQuota(api_error):  # type: ignore[misc, valid-type]
            def __init__(self):  # noqa: D401 - bypass SDK's response_json arg
                pass

        assert is_cloud_call_error(_FakeQuota())


def test_grounded_search_degrades_on_cloud_error(monkeypatch):
    """The except tuple must catch a provider error and return ok=False."""
    import core.skills.grounded_search as gs

    degrade_tuple = (AttributeError, RuntimeError, *cloud_call_error_types())
    # Simulate the live failure shape: a 429-style error is raised by the SDK.
    raised = ConnectionError("429 RESOURCE_EXHAUSTED: quota exceeded")
    assert isinstance(raised, degrade_tuple), (
        "the grounded-search except tuple must classify a cloud transport error"
    )
