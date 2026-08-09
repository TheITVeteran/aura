#!/usr/bin/env python3
"""A real MCP server, used to prove Aura's MCP reach actually works.

Without this, every MCP test would mock the transport and prove only that
Aura calls a mock correctly — which is exactly the failure this codebase
keeps finding, where a faculty reporting that it ran is taken as evidence
that it worked.

This is a genuine stdio MCP server: `tests/test_mcp_reach_is_real.py`
launches it as a subprocess through the same skill, gate, and registry the
live runtime uses, completes the protocol handshake, and calls its tools.

It has no dependency on Aura and imports nothing from the repo, so a failure
here is a failure of the MCP path and not of the harness.
"""

from mcp.server import MCPServer

server = MCPServer(name="aura-test-echo", version="1.0.0")


@server.tool(description="Echo a message back, to prove the round trip.")
def echo(message: str) -> str:
    return f"echo:{message}"


@server.tool(description="Add two integers, to prove typed arguments arrive.")
def add(a: int, b: int) -> int:
    return a + b


if __name__ == "__main__":
    server.run()
