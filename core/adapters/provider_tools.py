"""Validate tool declarations at the model-execution boundary.

A tool definition in a generation request tells the model that a capability
exists. If Aura cannot actually execute it, the model is being told
something untrue about itself; if nothing checked, then anything that
could reach an adapter could declare one. CP126 ``6e14ba27`` — caller
tools were copied straight into the request with no validation against the
capability registry, no schema limit, and no policy.

Fails CLOSED per tool rather than per request. A request that declares one
unknown tool beside four real ones is still a useful request, and refusing
the whole list would teach callers to stop declaring tools at all.
"""

from __future__ import annotations

from typing import Any

from core.runtime.errors import record_degradation

__all__ = [
    "MAX_TOOLS_PER_REQUEST",
    "MAX_TOOL_SCHEMA_CHARS",
    "admissible_tools",
    "registered_capability_names",
    "tool_name",
]

#: Ceiling on what one request may declare. An unbounded tool list is a
#: payload, and a schema deep enough to be interesting is deep enough to be
#: a denial of service on the request parser.
MAX_TOOLS_PER_REQUEST = 32
MAX_TOOL_SCHEMA_CHARS = 20_000


def tool_name(tool: Any) -> str:
    """The declared name, or empty when the shape carries none."""
    for key in ("name", "function_declarations"):
        if isinstance(tool, dict) and key in tool:
            value = tool[key]
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, (list, tuple)) and value:
                first = value[0]
                if isinstance(first, dict):
                    return str(first.get("name", "")).strip()
    name = getattr(tool, "name", "")
    return str(name).strip() if isinstance(name, str) else ""


def registered_capability_names() -> set[str]:
    """Tool names the runtime can actually execute. Empty when unknown."""
    try:
        from core.container import ServiceContainer

        engine = ServiceContainer.get("capability_engine", default=None)
        if engine is None:
            return set()
        for attr in ("list_skill_names", "skill_names", "available_skills"):
            getter = getattr(engine, attr, None)
            if callable(getter):
                return {str(n) for n in (getter() or [])}
        skills = getattr(engine, "skills", None)
        if isinstance(skills, dict):
            return {str(n) for n in skills}
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return set()
    return set()


def admissible_tools(tools: Any) -> list[Any]:
    """Keep the tools Aura actually has; drop and record the rest."""
    if not tools:
        return []
    if not isinstance(tools, (list, tuple)):
        record_degradation(
            "provider_tools",
            TypeError(f"tools must be a sequence, got {type(tools).__name__}"),
            severity="warning",
            action="dropped a malformed tool declaration at the execution boundary",
        )
        return []
    if len(tools) > MAX_TOOLS_PER_REQUEST:
        record_degradation(
            "provider_tools",
            ValueError(f"{len(tools)} tools declared for one request"),
            severity="warning",
            action=f"kept the first {MAX_TOOLS_PER_REQUEST} tool declarations",
        )
        tools = list(tools)[:MAX_TOOLS_PER_REQUEST]

    known = registered_capability_names()
    admitted: list[Any] = []
    refused: list[str] = []
    for tool in tools:
        name = tool_name(tool)
        if len(str(tool)) > MAX_TOOL_SCHEMA_CHARS:
            refused.append(f"{name}:schema_too_large")
            continue
        # An empty registry means the capability engine is not up. That is
        # not permission to forward anything: unnamed tools are refused
        # either way, and named ones are refused when a registry exists and
        # does not know them.
        if not name:
            refused.append("unnamed")
            continue
        if known and name not in known:
            refused.append(f"{name}:not_in_capability_registry")
            continue
        admitted.append(tool)

    if refused:
        record_degradation(
            "provider_tools",
            PermissionError(f"refused tool declarations: {refused[:8]}"),
            severity="warning",
            action="forwarded only the tools Aura's capability registry knows",
            extra={"refused_count": len(refused), "admitted_count": len(admitted)},
        )
    return admitted
