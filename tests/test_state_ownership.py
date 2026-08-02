"""A test run must not be able to change what Aura believes.

Two live cases:

* Terminal-grid tests read and wrote the live user-global learning
  directory. Learned risk from live state reached 1.0 and vetoed a benign
  move; test episodes were written back into the organism's real memory.
* A fixture left a fail-closed Mycelium registered after its test ended,
  and dozens of later tests failed only inside the full suite.

The guard is not a warning. The damage above was silent.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.runtime.state_ownership import (
    RuntimeProfile,
    StateOwnershipViolation,
    assert_state_path_allowed,
    is_live_state_path,
    live_state_root,
    runtime_identity,
    runtime_instance_id,
    runtime_profile,
    stamp_record,
    state_root,
)

ROOT = Path(__file__).resolve().parents[1]


def test_this_very_test_run_is_not_a_live_runtime():
    """If this fails, everything below is testing the wrong thing."""
    assert runtime_profile() is RuntimeProfile.TEST
    assert not runtime_profile().may_touch_live_state


def test_the_test_root_is_not_inside_the_live_root():
    """A subdirectory is one `..` from the thing it was meant to be apart from."""
    root, live = state_root(), live_state_root()
    assert root != live
    assert live not in root.parents


def test_writing_live_state_from_a_test_raises(tmp_path):
    with pytest.raises(StateOwnershipViolation, match="live instance state"):
        assert_state_path_allowed(live_state_root() / "data" / "aura_memory.db")
    # The run's own root is fine.
    assert_state_path_allowed(state_root() / "data" / "anything.json")
    assert_state_path_allowed(tmp_path / "scratch.json")


def test_the_violation_says_how_to_fix_it():
    """An enforcement message that does not name the remedy just blocks work."""
    with pytest.raises(StateOwnershipViolation) as caught:
        assert_state_path_allowed(live_state_root() / "data" / "x.json", source="probe")
    message = str(caught.value)
    assert "AURA_STATE_ROOT" in message
    assert "probe" in message


def test_traversal_back_into_live_state_is_caught():
    """Resolved, not string-compared, or the check is decorative."""
    sneaky = state_root() / ".." / live_state_root().name / "data" / "memory.db"
    assert is_live_state_path(sneaky)
    with pytest.raises(StateOwnershipViolation):
        assert_state_path_allowed(sneaky)


def test_a_path_that_does_not_exist_yet_is_still_checked():
    """Writes create things; checking only existing paths would miss every write."""
    assert is_live_state_path(live_state_root() / "not" / "created" / "yet.json")


def test_the_write_gateway_enforces_it():
    """The guard has to sit at the chokepoint, not in a helper nobody calls."""
    from core.runtime.file_write_gateway import get_file_write_gateway

    with pytest.raises(StateOwnershipViolation):
        get_file_write_gateway().write_text(
            live_state_root() / "data" / "test_should_never_write_this.json",
            "{}",
            source="test_state_ownership",
        )


def test_the_gateway_guard_is_wired_at_the_shared_coercion_point():
    """AST: every write path funnels through _coerce_target, so it must guard."""
    source = (ROOT / "core" / "runtime" / "file_write_gateway.py").read_text("utf-8")
    tree = ast.parse(source)
    for name in ("_coerce_target", "_coerce_path_allow_dir"):
        func = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        calls = {
            node.func.id
            for node in ast.walk(func)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "assert_state_path_allowed" in calls, (
            f"{name} does not check state ownership; writes routed through it "
            "would reach live state unguarded"
        )


def test_identity_is_immutable_for_the_process():
    assert runtime_instance_id() == runtime_instance_id()
    assert runtime_instance_id().startswith("test-")


def test_persistent_records_name_the_runtime_that_wrote_them():
    stamped = stamp_record({"episode": 1}, model_identity="qwen-32b@abc123")
    assert stamped["episode"] == 1
    assert stamped["_runtime"]["runtime_instance_id"] == runtime_instance_id()
    assert stamped["_runtime"]["runtime_profile"] == "test"
    assert stamped["_runtime"]["model_identity"] == "qwen-32b@abc123"


def test_stamping_does_not_mutate_the_caller_or_overwrite_an_origin():
    original = {"episode": 1}
    stamp_record(original)
    assert "_runtime" not in original, "stamp_record mutated its input"

    forwarded = {"episode": 2, "_runtime": {"runtime_instance_id": "live-original"}}
    assert stamp_record(forwarded)["_runtime"]["runtime_instance_id"] == "live-original"


def test_runtime_identity_is_serializable_and_carries_no_secrets():
    import json

    identity = runtime_identity()
    json.dumps(identity)
    assert set(identity) == {
        "runtime_instance_id",
        "runtime_profile",
        "state_root",
        "started_at",
        "pid",
        "host",
    }


def test_config_paths_resolve_under_this_runs_root_not_the_live_one():
    """The structural fix: stores get their own world, not a blocked write."""
    from core.config import config

    assert not is_live_state_path(config.paths.data_dir)
    assert str(config.paths.data_dir).startswith(str(state_root()))


@pytest.mark.parametrize(
    "module,attribute",
    [
        ("core.security.trust_engine", "TRUST_LOG_PATH"),
        ("core.memory.conversation_persistence", "DEFAULT_PERSIST_DIR"),
        ("core.reaper", "DEFAULT_REAPER_MANIFEST_DIR"),
        ("core.memory.scar_formation", "_DATA_DIR"),
        ("core.agi.skill_synthesizer", "PERSIST_PATH"),
    ],
)
def test_module_level_state_paths_follow_the_profile(module, attribute):
    """These were `Path.home() / ".aura"`, evaluated at import, bypassing config."""
    import importlib

    value = getattr(importlib.import_module(module), attribute)
    assert not is_live_state_path(value), (
        f"{module}.{attribute} still resolves into live instance state"
    )


def test_no_module_resolves_the_live_root_for_itself_any_more():
    """The whole class, pinned. 243 sites did this before.

    A store that finds its own root from Path.home() cannot be redirected,
    only blocked — and a blocked subsystem is not a working one.
    """
    offenders: list[str] = []
    allowed = {"core/runtime/state_ownership.py"}
    for path in (ROOT / "core").rglob("*.py"):
        rel = str(path.relative_to(ROOT))
        if rel in allowed or "__pycache__" in rel:
            continue
        text = path.read_text("utf-8", errors="ignore")
        for line_no, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if 'Path.home() / ".aura"' in line or "expanduser(\"~/.aura\")" in line:
                offenders.append(f"{rel}:{line_no}")
    assert not offenders, (
        "these modules resolve the live state root themselves instead of "
        f"asking state_root(): {offenders[:10]}"
    )
