from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHELL_APP = ROOT / "interface/static/shell/src/App.jsx"


def _source() -> str:
    return SHELL_APP.read_text(encoding="utf-8")


def test_react_shell_marks_chat_requests_as_desktop_cognitive_engine_required():
    source = _source()
    assert '"X-Aura-Surface": "desktop-ui"' in source
    assert '"X-Aura-Require-CognitiveEngine": "true"' in source
    assert "headers: desktopChatHeaders()" in source


def test_react_shell_renders_fail_closed_chat_response_body_before_generic_error():
    source = _source()
    send_start = source.index("async function sendMessage")
    regen_start = source.index("async function regenerate")
    send_block = source[send_start:regen_start]

    payload_read = send_block.index("const payload = await readApiPayload(response);")
    status_check = send_block.index("if (!response.ok)")
    assert payload_read < status_check
    assert 'role: payload.response ? "assistant" : "system"' in send_block
    assert 'apiFailureMessage(payload, `Chat failed (${response.status})`)' in send_block
