"""A real answer must not be destroyed by its envelope.

Measured live on the desktop path. Aura was asked to open Notes and write a
three-sentence note about orcas. She DID it — the note is on disk:

    Orcas are apex predators known for their intelligence and complex social
    structures. They communicate through a variety of clicks, whistles, and
    pulsed calls. Each pod has its own distinct dialect.

The salvage path then returned the reply as a bare `str`, the delivery boundary
checked `isinstance(response, JSONResponse)`, and the person received:

    The chat route returned an unsupported response format.

Worst possible polarity: the REFUSAL path in the same function was correctly
formed, so a turn that failed reported cleanly while a turn that SUCCEEDED
reported a transport error.
"""

from __future__ import annotations

import inspect


def test_the_salvage_path_returns_a_json_response():
    """`_servable_draft_or_none` returns a str; the handler must not pass it through."""
    from interface.routes import chat as chat_routes

    source = inspect.getsource(chat_routes)
    start = source.index("async def _fail_closed_degraded_desktop_reply")
    # The salvage function ends at the next sibling definition.
    end = source.index("async def _finalize_fastpath", start)
    body = source[start:end]

    assert "return salvaged" not in body, (
        "the salvaged draft is a bare str; returning it raw makes the delivery "
        "boundary replace a real answer with a 500"
    )
    assert "served_repairable_draft" in body
    # Every exit carries an envelope.
    returns = [
        line.strip()
        for line in body.splitlines()
        if line.strip().startswith("return ")
    ]
    assert returns, "the salvage function must still return something"
    for statement in returns:
        assert (
            statement.startswith("return JSONResponse")
            or statement == "return served"  # already a JSONResponse | None
        ), f"non-JSONResponse exit from the salvage path: {statement!r}"


def test_the_delivery_boundary_delivers_text_instead_of_erasing_it():
    """The class-level guard: an envelope defect must not cost the answer."""
    from interface.routes import chat as chat_routes

    source = inspect.getsource(chat_routes)
    marker = "if not isinstance(response, JSONResponse):"
    assert marker in source
    guard = source[source.index(marker) : source.index(marker) + 2600]

    assert "chat_response_envelope_coerced" in guard, (
        "a handler that returned reply TEXT has produced an answer; the boundary "
        "must deliver it rather than ship a 500"
    )
    assert "chat.response_envelope" in guard, (
        "the envelope defect must still be recorded, not silently tolerated"
    )
    # A genuinely unusable return still fails closed.
    assert "chat_response_format_rejected" in guard
