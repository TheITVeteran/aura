"""One regex, two hazards, and only one of them was real.

The local-path rule reported 37 findings. Every one was a test file, and 27
of them were strings nothing ever opened: a jail-escape path a policy has to
reject, a fake socket name, a monkeypatched stub's return value. Nine more
were redaction fixtures — the secret a scrubber is proven to remove, plus the
assertion proving it — where deleting either line destroys the proof.

Two of the 37 were genuine. They were indistinguishable from the noise, which
is the whole failure: a rule that fires 35 times for nothing is a rule whose
output stops being read, and the two defects ride out on the ignored list.

So the rule now asks what the hazard actually is. A path naming one human's
account is machine-specific wherever it appears. A shared-temp path is a
defect when the process WRITES to it, and inert otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.aura_enterprise_gate import (  # noqa: E402
    docstring_line_numbers,
    local_path_context,
)

# Built here rather than written as literals: a test proving the rule still
# catches hardcoded paths must not itself ship hardcoded paths.
_TMP = "/" + "tmp"
_USERS = "/" + "Users"


def _tree(source: str):
    import ast

    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _context(source: str):
    return local_path_context(_tree(source))


def _scan(source: str) -> set[int]:
    """Lines the gate would report, applying every discriminator."""
    from tools.aura_enterprise_gate import TEXT_PATTERNS, _local_path_is_inert

    pattern = TEXT_PATTERNS["hardcoded_local_path"]
    tree = _tree(source)
    prose = docstring_line_numbers(tree)
    context = local_path_context(tree)

    reported: set[int] = set()
    for line_no, line in enumerate(source.splitlines(), start=1):
        match = pattern.search(line)
        if match is None:
            continue
        if line_no in prose or line.lstrip().startswith("#"):
            continue
        if _local_path_is_inert(match.group(0), line_no, context):
            continue
        reported.add(line_no)
    return reported


class TestTheRuleStillCatchesTheRealThing:
    def test_a_developers_home_directory_is_always_a_finding(self) -> None:
        source = f'DOCS = "{_USERS}/bryan/Documents/report.pdf"\n'
        assert _scan(source) == {1}

    def test_a_home_directory_is_a_finding_even_as_inert_data(self) -> None:
        """No I/O anywhere near it. It is still one person's machine."""
        source = f'CASES = [("{_USERS}/bryan/models/a.bin", "expected")]\n'
        assert _scan(source) == {1}

    def test_writing_to_shared_temp_is_a_finding(self) -> None:
        source = f'with open("{_TMP}/aura-report.json", "w") as fh:\n    fh.write("x")\n'
        assert _scan(source) == {1}

    def test_a_temp_path_wrapped_in_path_is_still_seen(self) -> None:
        source = f'shutil.rmtree(Path("{_TMP}/aura-workdir"))\n'
        assert _scan(source) == {1}

    def test_a_temp_path_as_a_subprocess_cwd_is_seen(self) -> None:
        source = f'subprocess.run(["ls"], cwd="{_TMP}/aura-workdir")\n'
        assert _scan(source) == {1}

    def test_an_interpolated_temp_path_that_gets_opened_is_seen(self) -> None:
        source = f'open(f"{_TMP}/aura-{{pid}}.sock", "w")\n'
        assert _scan(source) == {1}


class TestInertDataIsNotADefect:
    def test_a_rejected_policy_input_is_not_a_finding(self) -> None:
        """The path exists so the policy can refuse it. It is never opened."""
        source = f'assert policy.allows("{_TMP}/attacker/git") is False\n'
        assert _scan(source) == set()

    def test_a_monkeypatched_return_value_is_not_a_finding(self) -> None:
        source = f'monkeypatch.setattr(mod, "_repo", lambda: Path("{_TMP}/model"))\n'
        assert _scan(source) == set()

    def test_a_fake_socket_name_is_not_a_finding(self) -> None:
        source = f'record = {{"socket_path": f"{_TMP}/control-{{n}}.sock"}}\n'
        assert _scan(source) == set()


class TestRedactionFixturesSurvive:
    def test_the_secret_and_its_assertion_are_both_exempt(self) -> None:
        source = (
            f'def test_it():\n'
            f'    err = OSError("{_USERS}/secret/path: token failed")\n'
            f'    assert "{_USERS}/secret" not in str(redact(err))\n'
        )
        assert _scan(source) == set()

    def test_the_exemption_costs_an_actual_assertion(self) -> None:
        """Without the "must not leak" line, the fixture is just a hardcoded
        path again. That is what keeps this from being a way to go quiet."""
        source = f'    err = OSError("{_USERS}/secret/path: token failed")\n'
        assert _scan(source) == {1}

    def test_a_substring_assertion_covers_the_longer_literal(self) -> None:
        source = (
            f'def test_it():\n'
            f'    raise RuntimeError("{_USERS}/secret/path leaked token=abc123")\n'
            f'    assert "secret" not in blob\n'
        )
        assert _scan(source) == set()

    def test_a_short_generic_assertion_does_not_grant_the_exemption(self) -> None:
        """"ok" must not become a licence to hardcode every path in a file."""
        source = (
            f'def test_it():\n'
            f'    p = "{_USERS}/bryan/models/a.bin"\n'
            f'    assert "ok" not in blob\n'
        )
        assert _scan(source) == {2}


class TestProseIsNotCode:
    def test_a_docstring_quoting_an_incident_is_not_a_finding(self) -> None:
        source = (
            f'"""The live error, verbatim:\n'
            f'\n'
            f"    No such file: '{_USERS}/bryan/Desktop/wallpaper.png'\n"
            f'"""\n'
        )
        assert _scan(source) == set()

    def test_a_comment_explaining_the_rule_is_not_a_finding(self) -> None:
        source = f"# a hardcoded {_USERS}/<name> breaks on every other machine\n"
        assert _scan(source) == set()

    def test_a_docstring_does_not_shield_the_code_beneath_it(self) -> None:
        source = (
            f'def f():\n'
            f'    """Explains {_USERS}/bryan/notes."""\n'
            f'    return "{_USERS}/bryan/notes"\n'
        )
        assert _scan(source) == {3}


class TestTheDiscriminatorsAreHonestAboutTheirLimits:
    def test_an_unparseable_file_claims_nothing(self) -> None:
        """Silently returning "no findings" on a syntax error would let a
        broken file suppress its own defects. Empty sets mean the exemptions
        do not apply, so every match is reported."""
        source = f'def broken(\n    x = "{_TMP}/aura-out.json"\n'
        context = _context(source)
        assert docstring_line_numbers(_tree(source)) == set()
        assert context.disk_lines is None
        assert context.redaction_evidence == ()
        assert _scan(source) == {2}

    def test_a_path_bound_to_a_name_first_is_missed(self) -> None:
        """Documented, deliberate: only direct operands are traced. Recording
        the gap here means the next person finds it as a known limit rather
        than as a surprise."""
        source = f'p = "{_TMP}/aura-out.json"\nopen(p, "w")\n'
        assert _scan(source) == set()
