from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHELL_APP = ROOT / "interface/static/shell/src/App.jsx"
LEGACY_INDEX = ROOT / "interface/static/index.html"
LEGACY_JS = ROOT / "interface/static/aura.js"


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


def test_legacy_shell_inline_scripts_are_syntax_valid():
    html = LEGACY_INDEX.read_text(encoding="utf-8")
    scripts = [
        match.group(1)
        for match in re.finditer(
            r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>",
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
    ]
    assert scripts

    matching = {")": "(", "]": "[", "}": "{"}
    opening = set(matching.values())
    closing = set(matching)
    for index, script in enumerate(scripts, start=1):
        stack: list[tuple[str, int]] = []
        quote = ""
        escaped = False
        line_comment = False
        block_comment = False
        for pos, char in enumerate(script):
            next_char = script[pos + 1] if pos + 1 < len(script) else ""
            if line_comment:
                if char == "\n":
                    line_comment = False
                continue
            if block_comment:
                if char == "*" and next_char == "/":
                    block_comment = False
                continue
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = ""
                continue
            if char == "/" and next_char == "/":
                line_comment = True
                continue
            if char == "/" and next_char == "*":
                block_comment = True
                continue
            if char in {"'", '"', "`"}:
                quote = char
                continue
            if char in opening:
                stack.append((char, pos))
                continue
            if char in closing:
                assert stack, f"inline script {index} has unmatched {char!r} at {pos}"
                opener, opener_pos = stack.pop()
                assert opener == matching[char], (
                    f"inline script {index} closes {opener!r} from {opener_pos} with {char!r} at {pos}"
                )
        assert not quote, f"inline script {index} has an unterminated string/template literal"
        assert not block_comment, f"inline script {index} has an unterminated block comment"
        assert not stack, f"inline script {index} has unclosed delimiter {stack[-1][0]!r}"


def test_legacy_splash_auto_reveals_shell_after_timeout():
    source = LEGACY_JS.read_text(encoding="utf-8")
    dismiss_start = source.index("function dismissSplash")
    dismiss_block = source[dismiss_start: source.index("document.addEventListener('visibilitychange'", dismiss_start)]

    assert "autoRevealMs" in dismiss_block
    assert "const revealShell = () =>" in dismiss_block
    assert "setTimeout(revealShell, autoRevealMs)" in dismiss_block
    assert "Loading interface...', { autoRevealMs: 900 })" in source
