"""A card that hides the numbers must offer a way to read them.

Live, thought cards rendered:

    "Voice listening — internal measurements (open FULL for the numbers)."
    "Sustained anomaly detected — internal measurements (open FULL ...)."  [WARNING]

Two defects behind that. The text named a control called FULL, and no
control called FULL has ever been rendered — the expander is labelled
SHOW ALL. And `longThought` counted only *clipping*, never *redaction*, so a
SHORT telemetry line had its numbers replaced on the face while no expander
rendered at all. A WARNING card named an anomaly it could not then describe.

The general rule: any transformation that hides content must report hidden
content, the same way clipping does.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

import pytest

AURA_JS = pathlib.Path(__file__).resolve().parent.parent / "interface" / "static" / "aura.js"

TELEMETRY_LINE = "Voice listening | rms_gate=0.01 conf_gate=-0.7 threshold=0.55 state=armed"


def test_no_card_text_promises_a_control_that_does_not_exist():
    source = AURA_JS.read_text(encoding="utf-8")
    # The only surviving mention may be the comment recording the old bug.
    offenders = [
        line.strip()
        for line in source.splitlines()
        if ("open FULL" in line or "FULL and COPY" in line or "FULL or COPY" in line)
        and 'said "open FULL"' not in line
    ]
    assert not offenders, offenders
    assert "SHOW ALL" in source


def test_the_expander_control_is_named_show_all():
    source = AURA_JS.read_text(encoding="utf-8")
    assert ">SHOW ALL</button>" in source


def test_redaction_marks_the_card_as_having_hidden_content():
    source = AURA_JS.read_text(encoding="utf-8")
    assert "redactsMeasurements" in source
    assert "|| measurementsRedacted" in source, (
        "redaction must feed longThought, or a short redacted line renders "
        "with no expander and the numbers are unreachable"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_redaction_behaviour_executes_correctly():
    """Run the real functions rather than trusting a grep."""
    source = AURA_JS.read_text(encoding="utf-8")

    start = source.index("const PLAIN_LANGUAGE_RULES")
    end = source.index("function redactsMeasurements")
    end = source.index("}", source.index("return plainLanguageThought", end)) + 1
    snippet = source[start:end]

    harness = (
        snippet
        + "\n"
        + f"const line = {json.dumps(TELEMETRY_LINE)};\n"
        + "const out = plainLanguageThought(line);\n"
        + "console.log(JSON.stringify({"
        + "redacted: redactsMeasurements(line),"
        + "text: out,"
        + "plainUntouched: redactsMeasurements('She is thinking about the weather.')"
        + "}));\n"
    )
    result = subprocess.run(
        ["node", "-e", harness], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])

    assert payload["redacted"] is True, "a dense telemetry line must report redaction"
    assert "SHOW ALL" in payload["text"]
    assert "open FULL" not in payload["text"]
    assert payload["plainUntouched"] is False, "ordinary prose must not be marked redacted"
