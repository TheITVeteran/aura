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


class TestPassOnlyConstructorsAreJudgedByShape:
    """A test double's no-op __init__ is not an unimplemented function.

    All three pass_only_function findings were constructor overrides on test
    doubles — the standard way to stop a real constructor from running, where
    `pass` IS the correct implementation of "set nothing up".

    The exemption is shaped, not path-based: the class must define at least
    one other method with a real body. A class that is nothing BUT a
    pass-only __init__ is still scaffolding and is still reported, which is
    what stops this from being "skip tests/" under a better name.
    """

    @staticmethod
    def _findings(source: str) -> list[str]:
        import ast

        from tools.aura_enterprise_gate import AstGate, GateReport

        report = GateReport(root=Path("."), generated_at_unix=0.0)
        AstGate("tests/probe.py", report, source.splitlines()).visit(ast.parse(source))
        return [f.kind for f in report.findings]

    def test_a_double_with_real_methods_is_exempt(self) -> None:
        source = (
            "class Double:\n"
            "    def __init__(self) -> None:\n"
            "        pass\n"
            "    def is_ready(self) -> bool:\n"
            "        return True\n"
        )
        assert "pass_only_function" not in self._findings(source)

    def test_a_class_that_is_only_a_stub_is_still_reported(self) -> None:
        source = "class Stub:\n    def __init__(self) -> None:\n        pass\n"
        assert "pass_only_function" in self._findings(source)

    def test_a_class_whose_other_methods_are_also_stubs_is_reported(self) -> None:
        source = (
            "class Hollow:\n"
            "    def __init__(self) -> None:\n"
            "        pass\n"
            "    def later(self) -> None:\n"
            "        pass\n"
        )
        assert "pass_only_function" in self._findings(source)

    def test_a_pass_only_method_that_is_not_a_constructor_is_reported(self) -> None:
        source = (
            "class Real:\n"
            "    def work(self) -> None:\n"
            "        pass\n"
            "    def other(self) -> int:\n"
            "        return 1\n"
        )
        assert "pass_only_function" in self._findings(source)

    def test_a_bare_pass_only_function_is_reported(self) -> None:
        assert "pass_only_function" in self._findings("def nothing():\n    pass\n")
