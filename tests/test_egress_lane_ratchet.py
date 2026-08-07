"""Ratchet: no NEW outbound HTTP in ``core/`` outside the network gateway.

``core/runtime/network_gateway.py`` is where an outbound request meets
governance, the defensive preflight, web-content provenance, and — since
the egress privacy boundary landed — the only read of what is actually
inside the body. All of that is worth exactly as much as the share of
traffic that goes through it.

It was not all of it. Four modules held their own ``aiohttp.ClientSession``
and reached the network directly: peer belief and drive-state broadcast
(``core/collective/belief_sync.py``, four call paths, carrying Aura's own
beliefs to addresses that arrive from discovery), web search
(``core/agency/tool_orchestrator.py``, carrying the user's words to the open
web), a localhost health probe, and a shared session in the API adapter.
None of them were wrong about anything except the one thing that mattered:
they were outside the boundary, so the boundary was a boundary with doors in
it.

A convention cannot hold this. Each of those sites was written by someone
who simply reached for the HTTP client they knew, and the next one will be
too. This is the gate instead.

If this test fails on code you just wrote: call
``get_network_gateway().request()`` / ``.request_async()`` instead of an
HTTP client directly. If a vendor SDK builds its own HTTP and cannot be
routed, screen the content at the call site with
``core.security.egress_privacy.filter_model_prompt`` and add the module here
with that reason stated.
"""
from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Calls that put bytes on a socket without passing the gateway.
_DIRECT_EGRESS_CALLS = {
    "ClientSession",  # aiohttp
    "urlopen",  # urllib.request
    "build_opener",
}

#: Module attributes that are an HTTP verb on a client library.
_DIRECT_EGRESS_ROOTS = {"httpx", "requests", "urllib3", "aiohttp"}
_HTTP_VERBS = {"get", "post", "put", "patch", "delete", "head", "options", "stream", "request"}

#: Vendor SDKs that build and send their own HTTP. These do not look like
#: network calls at all — ``client.aio.models.generate_content(...)`` reads
#: like a method on an object — which is exactly why the API adapter's cloud
#: path sat outside the gateway without anyone noticing. The constructor is
#: the honest place to catch them: holding one of these clients IS holding a
#: way out of the machine.
_VENDOR_CLIENT_ROOTS = {"genai", "openai", "anthropic", "cohere", "mistralai"}
_VENDOR_CLIENT_CALLS = {"Client", "AsyncClient", "AsyncAnthropic", "AsyncOpenAI"}

#: Files that legitimately hold direct HTTP, with the reason. This list only
#: shrinks.
ALLOWED: dict[str, str] = {
    # The gateway itself is the thing everyone else routes through.
    "core/runtime/network_gateway.py": "is the gateway",
    # Holds a google.genai client that builds its own HTTP and cannot be
    # routed through the gateway. Screened instead at the call site by
    # APIAdapter._screen_for_egress, which applies the same egress privacy
    # tiers and refuses the cloud leg rather than send an unscreened prompt.
    "core/adapters/api_adapter.py": (
        "vendor SDK builds its own transport; content is screened by "
        "_screen_for_egress before it is handed to the client"
    ),
    # Both hold a google.genai client for grounded search, and both screen the
    # query with filter_model_prompt before it is sent — a refusal drops the
    # cloud leg and the local search pipeline answers instead.
    "core/brain/react_loop.py": (
        "vendor SDK for grounded search; query screened by filter_model_prompt, "
        "refusal falls through to the local Sovereign/DDG pipeline"
    ),
    "core/skills/grounded_search.py": (
        "vendor SDK for grounded search; query screened by filter_model_prompt, "
        "refusal returns a fallback instruction to the caller"
    ),
}


def _scan() -> dict[str, set[str]]:
    """Every direct-egress call in ``core/``, by file."""
    found: dict[str, set[str]] = {}
    for path in (PROJECT_ROOT / "core").rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        rel = str(path.relative_to(PROJECT_ROOT))

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue

            # aiohttp.ClientSession(...) / urllib.request.urlopen(...)
            if func.attr in _DIRECT_EGRESS_CALLS:
                found.setdefault(rel, set()).add(func.attr)
                continue

            root = func.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if not isinstance(root, ast.Name):
                continue

            # httpx.post(...) / requests.get(...) — a verb on a client root.
            if func.attr in _HTTP_VERBS and root.id in _DIRECT_EGRESS_ROOTS:
                found.setdefault(rel, set()).add(f"{root.id}.{func.attr}")

            # genai.Client(...) — a vendor SDK that carries its own transport.
            elif func.attr in _VENDOR_CLIENT_CALLS and root.id in _VENDOR_CLIENT_ROOTS:
                found.setdefault(rel, set()).add(f"{root.id}.{func.attr}")
    return found


def test_no_new_direct_egress_outside_the_gateway():
    found = _scan()
    offenders = {
        rel: sorted(calls) for rel, calls in found.items() if rel not in ALLOWED
    }
    assert not offenders, (
        "outbound HTTP in core/ that skips core/runtime/network_gateway.py — and "
        "therefore skips governance, the outbound preflight, and the egress "
        f"privacy boundary: {offenders}. Route it through "
        "get_network_gateway().request_async()."
    )


def test_the_allowlist_only_shrinks():
    """An allowlist entry that no longer needs to be there is debt, not policy."""
    found = _scan()
    stale = sorted(set(ALLOWED) - set(found))
    assert not stale, (
        f"these files no longer contain direct egress and should be removed "
        f"from ALLOWED: {stale}"
    )
