"""A dead runtime must not look like a slow thought.

The browser's delivery loop is deliberately patient: a turn survives a
transport blip and resumes rather than failing on the first error. It had no
ceiling. Observed live 2026-07-27, the runtime exited mid-turn and the chat
window showed

    ● ● ● Aura is reconciling the current turn…

for seventeen minutes, over a process that had already gone. That is the
session's recurring theme in its purest form — a failure and a wait were
indistinguishable to the person watching, so the person had no way to know
anything was wrong.

The fix has to hold both directions at once: bound the case where nothing
answers, without ever cutting off a turn that is merely taking a long time.
Only consecutive TRANSPORT failures count against the ceiling; a server that
answers "still working" is contact and resets the clock.

The JS runs under node against the real aura.js source, so editing the loop
back to an unbounded retry fails this test rather than quietly shipping.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "tests" / "js" / "chat_delivery_ceiling.mjs"
AURA_JS = REPO / "interface" / "static" / "aura.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_delivery_loop_gives_up_on_an_unreachable_runtime_but_not_a_slow_one():
    result = subprocess.run(
        [shutil.which("node"), str(HARNESS), str(AURA_JS)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"chat delivery ceiling checks failed:\n{result.stdout}\n{result.stderr}"
    )
    assert "all checks passed" in result.stdout


def test_the_ceiling_is_declared_in_the_source():
    """Readable without node, and the reason it exists stays next to it."""
    source = AURA_JS.read_text(encoding="utf-8")
    assert "CHAT_DELIVERY_UNREACHABLE_MS" in source
    assert "unreachableSince = 0;" in source, (
        "contact must reset the clock, or a long turn eventually gets cut off"
    )
