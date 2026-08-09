"""Aura's MCP reach, measured against a real server rather than a mock.

`mcp_client` was registered, routable, and documented as connecting Aura to
"enterprise data connectors" — and it could not have connected to anything.
The `mcp` package was not installed, so every call returned an install
error; there was no registry, so reaching a connector meant guessing a
working `npx -y @scope/server` command line from nothing; and the whole path
asked no authority at all.

A mocked transport would prove none of that was fixed. These tests launch
`tests/fixtures/mcp_echo_server.py` as a real subprocess, complete the real
protocol handshake, and call real tools — through the same skill, gate, and
registry the live runtime uses. If the MCP path breaks, this fails.

Marked `slow` because each case spawns a process and runs a handshake.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

mcp = pytest.importorskip("mcp", reason="the mcp package is the capability under test")

from core.capabilities import mcp_connectors  # noqa: E402
from core.skills.mcp_client import MCPClientSkill, MCPInput  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ECHO_SERVER = ROOT / "tests" / "fixtures" / "mcp_echo_server.py"


@pytest.fixture
def echo_registry(tmp_path, monkeypatch):
    """Register the echo server as a connector named "echo"."""
    registry = tmp_path / "mcp_connectors.json"
    registry.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "echo": {
                        "command": sys.executable,
                        "args": [str(ECHO_SERVER)],
                        "purpose": "test echo server",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mcp_connectors, "_AURA_REGISTRY", registry)
    # Do not let the developer's own host config leak into the assertions.
    monkeypatch.setattr(mcp_connectors, "_HOST_CONFIGS", ())
    return registry


@pytest.fixture
def permissive_authority(monkeypatch):
    """Approve execution, so these tests measure the MCP path, not the gate.

    The gate itself is tested in
    `test_general_execution_surfaces_are_governed.py`. Here it must be out
    of the way, but it must still be CALLED — the counter at the end of this
    file asserts that.
    """
    calls: list[str] = []

    class _Decision:
        approved = True
        reason = "test"
        outcome = "approved"
        constraints: dict = {}
        capability_token_id = "tok"
        executive_intent_id = "intent"
        standing_authority_token = "lease"
        signed_capability = "SIGNED"

    class _Gateway:
        async def authorize_tool_execution(self, tool, args, **kw):
            calls.append(args.get("kind", "?"))
            return _Decision()

        def finalize_tool_execution(self, **kw):
            return {"closed": True}

    class _Result:
        ok = True
        detail = ""

    class _Verifier:
        def verify(self, cap, **kw):
            return _Result()

    import core.executive.authority_gateway as ag
    import core.governance.capability_chain as cc

    monkeypatch.setattr(ag, "get_authority_gateway", lambda: _Gateway())
    monkeypatch.setattr(cc, "get_capability_verifier", lambda: _Verifier())
    return calls


# ───────────────────────────────────────────── the round trip actually runs


@pytest.mark.asyncio
async def test_discovery_reaches_a_real_server(echo_registry, permissive_authority):
    """A real process, a real handshake, real tool metadata coming back."""
    result = await MCPClientSkill().execute(
        MCPInput(action="discover", connector="echo"), {}
    )

    assert result["ok"] is True, result.get("error")
    names = {t["name"] for t in result["tools"]}
    assert {"echo", "add"} <= names, f"discovered {names}"


@pytest.mark.asyncio
async def test_a_tool_call_returns_the_servers_real_answer(
    echo_registry, permissive_authority
):
    result = await MCPClientSkill().execute(
        MCPInput(
            action="execute",
            connector="echo",
            tool_name="echo",
            tool_args={"message": "hello"},
        ),
        {},
    )

    assert result["ok"] is True, result.get("error")
    assert "echo:hello" in json.dumps(result["result"])


@pytest.mark.asyncio
async def test_typed_arguments_survive_the_wire(echo_registry, permissive_authority):
    """A string round trip could pass while numbers silently stringify."""
    result = await MCPClientSkill().execute(
        MCPInput(
            action="execute",
            connector="echo",
            tool_name="add",
            tool_args={"a": 17, "b": 25},
        ),
        {},
    )

    assert result["ok"] is True, result.get("error")
    assert "42" in json.dumps(result["result"])


@pytest.mark.asyncio
async def test_both_grants_were_actually_requested(
    echo_registry, permissive_authority
):
    """The gate is not bypassed on the path that works.

    A governance check that only fires on the error path is not a gate.
    """
    await MCPClientSkill().execute(
        MCPInput(
            action="execute",
            connector="echo",
            tool_name="echo",
            tool_args={"message": "x"},
        ),
        {},
    )

    assert "mcp_server" in permissive_authority
    assert "mcp_tool" in permissive_authority


# ─────────────────────────────────────────────────── the registry is honest


def test_a_registered_connector_is_resolvable_by_name(echo_registry):
    connector = mcp_connectors.resolve_connector("echo")

    assert connector is not None
    assert connector.command == sys.executable
    assert str(ECHO_SERVER) in connector.args


def test_resolution_is_case_insensitive(echo_registry):
    assert mcp_connectors.resolve_connector("ECHO") is not None


def test_an_unknown_connector_resolves_to_nothing(echo_registry):
    assert mcp_connectors.resolve_connector("nope") is None


@pytest.mark.asyncio
async def test_an_unknown_connector_names_what_does_exist(
    echo_registry, permissive_authority
):
    """A refusal that does not say what IS available is a dead end."""
    result = await MCPClientSkill().execute(
        MCPInput(action="discover", connector="sentry"), {}
    )

    assert result["ok"] is False
    assert "echo" in result["error"]


@pytest.mark.asyncio
async def test_listing_connectors_launches_nothing(echo_registry, permissive_authority):
    """Answering "what can you reach?" must not start every server."""
    result = await MCPClientSkill().execute(MCPInput(action="list_connectors"), {})

    assert result["ok"] is True
    assert result["count"] == 1
    assert not permissive_authority, "listing spawned something"


def test_an_entry_with_no_command_is_dropped(tmp_path, monkeypatch):
    """A name that cannot be launched must not appear in the reach.

    Listing it would put a connector in Aura's answer that fails on use —
    a claim outliving the thing that made it true.
    """
    registry = tmp_path / "r.json"
    registry.write_text(
        json.dumps({"mcpServers": {"broken": {"args": ["x"]}, "fine": {"command": "true"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mcp_connectors, "_AURA_REGISTRY", registry)
    monkeypatch.setattr(mcp_connectors, "_HOST_CONFIGS", ())

    names = [c.name for c in mcp_connectors.available_connectors()]
    assert names == ["fine"]


def test_credentials_never_leave_the_config(tmp_path, monkeypatch):
    """`to_dict` feeds Aura's answer. Values must not be in it."""
    registry = tmp_path / "r.json"
    registry.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "svc": {"command": "true", "env": {"SVC_TOKEN": "super-secret"}}
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mcp_connectors, "_AURA_REGISTRY", registry)
    monkeypatch.setattr(mcp_connectors, "_HOST_CONFIGS", ())

    payload = json.dumps(mcp_connectors.describe_reach())

    assert "SVC_TOKEN" in payload, "the key name is useful and should be shown"
    assert "super-secret" not in payload, "a credential value reached Aura's answer"


@pytest.mark.asyncio
async def test_a_missing_credential_fails_with_the_real_reason(
    tmp_path, monkeypatch, permissive_authority
):
    """Otherwise the server starts and returns an opaque auth error."""
    registry = tmp_path / "r.json"
    registry.write_text(
        json.dumps(
            {"mcpServers": {"svc": {"command": "true", "env": {"ABSENT_KEY_XYZ": ""}}}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mcp_connectors, "_AURA_REGISTRY", registry)
    monkeypatch.setattr(mcp_connectors, "_HOST_CONFIGS", ())
    monkeypatch.delenv("ABSENT_KEY_XYZ", raising=False)

    result = await MCPClientSkill().execute(
        MCPInput(action="discover", connector="svc"), {}
    )

    assert result["ok"] is False
    assert "ABSENT_KEY_XYZ" in result["error"]
    assert not permissive_authority, "a connector with no credentials was launched"


def test_the_shipped_registry_is_empty_and_parseable():
    """The repo must not ship connectors nobody configured.

    A plausible default list would make `list_connectors` describe reach
    Aura does not have.
    """
    shipped = json.loads((ROOT / "config" / "mcp_connectors.json").read_text("utf-8"))

    assert shipped["mcpServers"] == {}


def test_the_host_configs_are_read_not_written():
    """Aura's reach widens when the owner configures a server in the tools
    they already use. She must never write to those files."""
    source = (ROOT / "core" / "capabilities" / "mcp_connectors.py").read_text("utf-8")

    assert ".claude.json" in source
    assert "claude_desktop_config.json" in source
    for forbidden in ("write_text", "open(", "json.dump"):
        assert forbidden not in source, f"the connector registry writes: {forbidden}"
