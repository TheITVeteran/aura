"""The forced tool handoff died on a duplicate keyword before it dispatched.

`should_force_tool_handoff` decides that a turn needs tools, builds the map,
and delegates to `think_and_act(prompt, system_prompt=..., tools=..., ...,
**kwargs)`. If `kwargs` already carries any of those names — and the
non-streaming path writes `system_prompt` into `kwargs` a few lines earlier —
Python rejects the call with a duplicate-keyword TypeError BEFORE the coroutine
runs. Nothing around it catches that, so the entire forced-tool route died on
the turns that most needed a tool.

The non-streaming path was fixed; the streaming one still passed `**kwargs`
straight through, one caller keyword away from the same crash.
"""
from __future__ import annotations

import ast
import inspect


def _handoff_calls():
    import core.brain.llm.llm_router as router_mod

    source = inspect.getsource(router_mod)
    tree = ast.parse(source)
    calls = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "think_and_act"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            calls.append((node, ast.get_source_segment(source, node) or ""))
    return calls


def test_both_handoff_sites_exist():
    assert len(_handoff_calls()) == 2, (
        "the forced tool handoff no longer has both a streaming and a "
        "non-streaming call site"
    )


def test_no_handoff_splats_unfiltered_kwargs():
    """**kwargs must be a copy with the consumed names removed."""
    for _node, rendered in _handoff_calls():
        assert "**kwargs," not in rendered, (
            "a forced tool handoff splats the caller's kwargs unfiltered, so "
            "one duplicate keyword raises TypeError before dispatch"
        )
        assert "**handoff_kwargs," in rendered


def test_every_explicitly_passed_name_is_popped_first():
    import core.brain.llm.llm_router as router_mod

    source = inspect.getsource(router_mod)

    for name in ("tools", "context", "prefer_tier", "_contract_tool_handoff"):
        assert f'handoff_kwargs.pop("{name}", None)' in source or (
            f'"{name}",' in source
        ), f"{name} is passed explicitly but never removed from the kwargs copy"
    assert source.count('handoff_kwargs.pop("system_prompt", "")') == 2


def test_the_streaming_handoff_keeps_its_system_prompt():
    """Popping must not silently discard the prompt the stream computed."""
    import core.brain.llm.llm_router as router_mod

    source = inspect.getsource(router_mod)

    assert 'handoff_kwargs.pop("system_prompt", "") or system_prompt or ""' in source
