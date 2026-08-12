"""Two ways the architect's safety gate opened by failing.

The gate stops the autonomous architect from editing this repository when
conditions are wrong. Both of its blind spots resolved to "proceed".

**A corrupted state file ended a freeze.** `_load_state` replaced an
unparseable file with `_empty_state()`, which clears `frozen_until`, zeroes
`consecutive_failures`, and empties the promotion and observation lists.
Every one of those is a RESTRAINT. So damaging that file — or writing it
non-atomically and crashing mid-write — was a way to end a freeze early and
start again with a clean record. The gate's own memory of why it stopped is
the last thing that can afford a permissive default.

**A broken git read the tree as clean.** `_check_git_clean` inspected
`result.stdout` and never `result.returncode`. A git that exited non-zero
with nothing on stdout — not a repository, an index lock held by another
process, a permissions failure — produced an empty diff, and the check that
exists to prevent editing a repo with uncommitted work passed hardest
exactly when git was broken.

The second one had a witness: `tiny_repo` was a bare directory, so every
architect test had been passing this gate by not being a repository at all.
"""
from __future__ import annotations

import json

import pytest

from core.architect.safety_gate import UNREADABLE_STATE_FREEZE_S


@pytest.fixture
def gate(tmp_path):
    """A safety gate whose state file lives in a temp directory."""
    from core.architect.config import ASAConfig
    from core.architect.safety_gate import ASASafetyGate

    def _make():
        return ASASafetyGate(
            ASAConfig(repo_root=tmp_path, artifact_root=tmp_path / ".aura_architect")
        )

    return _make


# ────────────────────────── a damaged state file is not a fresh start


def test_an_unreadable_state_file_freezes_rather_than_clearing_the_freeze(
    gate, tmp_path
):
    instance = gate()
    instance.state_path.parent.mkdir(parents=True, exist_ok=True)
    instance.state_path.write_text("{ this is not json", encoding="utf-8")

    reloaded = gate()

    assert reloaded.state["frozen_until"] is not None, (
        "a corrupted state file cleared the freeze, which is how a freeze "
        "gets ended early"
    )
    assert "unreadable" in reloaded.state["freeze_reason"]


def test_the_freeze_cannot_be_waited_out_within_a_session(gate):
    """The right response to "the restraints are unreadable" is a person
    looking at it, not a timer expiring."""
    assert UNREADABLE_STATE_FREEZE_S >= 60 * 60


def test_an_absent_state_file_is_still_a_fresh_start(gate):
    """A first run must not be treated as damage."""
    instance = gate()

    assert instance.state["frozen_until"] is None
    assert instance.state["consecutive_failures"] == 0


def test_a_valid_state_file_is_read_normally(gate):
    instance = gate()
    instance.state_path.parent.mkdir(parents=True, exist_ok=True)
    instance.state_path.write_text(
        json.dumps({"frozen_until": None, "consecutive_failures": 3}),
        encoding="utf-8",
    )

    reloaded = gate()

    assert reloaded.state["consecutive_failures"] == 3


def test_a_json_file_that_is_not_an_object_is_not_trusted(gate):
    """`json.loads("[]")` parses fine and then every `.get` explodes or
    returns None, which reads as no restraints."""
    instance = gate()
    instance.state_path.parent.mkdir(parents=True, exist_ok=True)
    instance.state_path.write_text("[1, 2, 3]", encoding="utf-8")

    reloaded = gate()

    assert isinstance(reloaded.state, dict)


# ─────────────────────────── a failed git status is not a clean tree


def test_a_non_repository_is_not_reported_clean(gate, tmp_path):
    """`tiny_repo` was a bare directory for exactly this reason."""
    instance = gate()

    ok, reason = instance._check_git_clean()

    assert ok is False, (
        "a directory that is not a git repository was reported as a clean "
        "tree, so the architect would edit it with no rollback available"
    )
    assert "git status failed" in reason


def test_the_returncode_is_checked_before_stdout():
    """Order is the whole defect: an empty stdout from a failed git looks
    identical to an empty stdout from a clean tree."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "core"
        / "architect"
        / "safety_gate.py"
    ).read_text("utf-8")

    body = source[source.index("def _check_git_clean") :]
    body = body[: body.index("def _check_t3_observation_limit")]

    assert body.index("returncode") < body.index("result.stdout.strip()"), (
        "stdout is inspected before the exit status again"
    )


def test_the_failure_reason_names_git_rather_than_the_tree():
    """"git not clean: " for a git that never ran sends whoever reads the
    log looking for uncommitted files that do not exist."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "core"
        / "architect"
        / "safety_gate.py"
    ).read_text("utf-8")

    assert 'f"git status failed (exit {result.returncode})' in source
