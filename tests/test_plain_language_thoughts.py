"""Engineering on the inside, plain English on the face.

The neural feed gives every card a lay channel label already, but the BODY
was still the raw log line:

    Router: Queueing background inference until admission clears for
    origin=stream_narrative reason=foreground_headroom_reserved after
    suppressing 11 repeated notices.

Someone watching their own mind think should not have to parse that. The
rewrite touches the PREVIEW only — fullMessage and the COPY payload stay
byte-for-byte raw, so FULL and COPY remain the debugging surface they
already were. Accessibility, not information loss.

The second half of the corpus matters as much as the first: ordinary speech
must pass through untouched, or a translator meant to clarify telemetry
would start rewriting Aura.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "tests" / "js" / "plain_language_thoughts.mjs"
AURA_JS = REPO / "interface" / "static" / "aura.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_telemetry_reads_as_english_and_speech_is_left_alone():
    result = subprocess.run(
        [shutil.which("node"), str(HARNESS), str(AURA_JS)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "all checks passed" in result.stdout


def test_the_raw_payload_is_still_carried():
    """FULL and COPY are the escape hatch that makes this safe."""
    source = AURA_JS.read_text(encoding="utf-8")
    assert "preview.text = plainLanguageThought(preview.text);" in source
    # The raw payload must not be routed through the translator.
    assert "escHtml(fullMsg)" in source
    assert "sanitizeThoughtMessage(rawFull)" in source
