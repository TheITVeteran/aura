from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


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
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required for inline JavaScript syntax verification")

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

    with tempfile.TemporaryDirectory() as tmp:
        for index, script in enumerate(scripts, start=1):
            path = Path(tmp) / f"inline_{index}.js"
            path.write_text(script, encoding="utf-8")
            result = subprocess.run(
                [node, "--check", str(path)],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            assert result.returncode == 0, result.stderr or result.stdout


def test_legacy_splash_auto_reveals_shell_after_timeout():
    source = LEGACY_JS.read_text(encoding="utf-8")
    dismiss_start = source.index("function dismissSplash")
    dismiss_block = source[dismiss_start: source.index("document.addEventListener('visibilitychange'", dismiss_start)]

    assert "autoRevealMs" in dismiss_block
    assert "const revealShell = () =>" in dismiss_block
    assert "setTimeout(revealShell, autoRevealMs)" in dismiss_block
    assert "Loading interface...', { autoRevealMs: 900 })" in source
