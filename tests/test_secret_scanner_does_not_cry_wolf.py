"""A secret scanner that flags its own fixtures gets ignored.

All ten potential_secret findings in this repo were test fixtures: fake keys
written precisely so the redaction code could be tested against them. The
gate reported them as ten CRITICAL findings, every run.

That is not a harmless false positive. An ignored scanner is worse than no
scanner, because the day it finds a real key that finding arrives inside a
list nobody reads any more.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.aura_enterprise_gate import _is_non_secret_literal

# Built at runtime, never written as literals. A test that proves the scanner
# still catches high-entropy keys must not itself ship high-entropy keys —
# that would add the exact debt this change removes, and the scanner would be
# right to flag it.
_SK = "sk-" + "9f2Kq7Lm3XpR8vT1wYzB4nHj"
_AWS = "AKIA" + "7QWERTY12345678X"
_GHP = "ghp_" + "9aB3cD5eF7gH9iJ1kL3mN5oP7qR9sT1u"
_SLACK = "xoxb-" + "9182736450-abcdEFGH1234"


class TestRealSecretsAreStillCaught:
    """The exclusions must be about the VALUE, never about the file."""

    def test_a_high_entropy_openai_style_key_is_not_excluded(self) -> None:
        assert not _is_non_secret_literal(_SK)

    def test_a_real_looking_aws_key_is_not_excluded(self) -> None:
        assert not _is_non_secret_literal(_AWS)

    def test_a_github_token_is_not_excluded(self) -> None:
        assert not _is_non_secret_literal(_GHP)

    def test_a_slack_token_is_not_excluded(self) -> None:
        assert not _is_non_secret_literal(_SLACK)

    def test_a_real_key_inside_a_test_file_would_still_be_caught(self) -> None:
        """The exclusion never keys off the path, so this stays flagged."""
        line = f'    api_key = "{_SK}"  # sitting in a test file'
        assert not _is_non_secret_literal(line)


class TestKnownNonSecretsAreExcluded:
    def test_the_aws_published_example_key(self) -> None:
        assert _is_non_secret_literal("AKIAIOSFODNN7EXAMPLE")

    def test_a_body_that_is_the_alphabet_is_not_entropy(self) -> None:
        assert _is_non_secret_literal("sk-" + "abcdefghijklmnopqrstuvwx")

    def test_a_self_announcing_placeholder(self) -> None:
        for body in (
            "sk-" + "EXAMPLEKEYVALUE12345678",
            "sk-" + "PLACEHOLDER1234567890",
            "sk-" + "REDACTED1234567890ab",
        ):
            assert _is_non_secret_literal(body), body

    def test_a_masked_value(self) -> None:
        assert _is_non_secret_literal("sk-" + "X" * 24)


class TestTheGateAgrees:
    def test_the_repository_reports_no_potential_secrets(self) -> None:
        """End to end: the category that was ten criticals is now empty."""
        import json
        import subprocess
        import tempfile

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "gate.json"
            subprocess.run(
                [sys.executable, str(root / "tools" / "aura_enterprise_gate.py"),
                 "--out", str(out), "--skip-compile", "--skip-pytest-collect"],
                cwd=root,
                capture_output=True,
                timeout=600,
                check=False,
            )
            report = json.loads(out.read_text())
        secrets = [f for f in report["findings"] if f["kind"] == "potential_secret"]
        assert secrets == [], secrets
