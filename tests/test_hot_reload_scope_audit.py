"""A hot-reload scope that names nothing may not report success.

Found by pressing UPDATE on the live desktop, 2026-07-26: the llm scope had
pointed at "core.brain.llm.inference_gate" for as long as it existed. There is
no such module — the gate is core.brain.inference_gate — so the entry matched
nothing, reloaded nothing, and the UI said "All changes are live."
"""

import importlib.util

import pytest

from core.ops.hot_reload import (
    LIVE_SAFE_ALL_SCOPES,
    RELOAD_SCOPES,
    HotReloader,
    ReloadResult,
)

pytestmark = pytest.mark.unit


def test_every_declared_scope_prefix_resolves_to_a_real_module():
    unresolvable = []
    for scope, prefixes in RELOAD_SCOPES.items():
        for prefix in prefixes:
            target = prefix.rstrip(".")
            try:
                found = importlib.util.find_spec(target) is not None
            except (ImportError, AttributeError, ValueError):
                found = False
            if not found:
                unresolvable.append(f"{scope}:{prefix}")
    assert not unresolvable, (
        "hot-reload scopes name modules that do not exist, so pressing UPDATE "
        f"silently reloads nothing for them: {unresolvable}"
    )


def test_the_audit_catches_the_defect_it_was_written_for():
    reloader = HotReloader()
    RELOAD_SCOPES["_audit_probe"] = ["core.brain.llm.inference_gate"]
    try:
        assert reloader._unmatched_prefixes(("_audit_probe",)) == [
            "_audit_probe:core.brain.llm.inference_gate"
        ]
    finally:
        RELOAD_SCOPES.pop("_audit_probe", None)


def test_a_real_but_unimported_module_is_not_an_error():
    """Lazily imported subsystems are normal; only unresolvable names count."""
    reloader = HotReloader()
    RELOAD_SCOPES["_audit_probe"] = ["core.ops.hot_reload"]
    try:
        assert reloader._unmatched_prefixes(("_audit_probe",)) == []
    finally:
        RELOAD_SCOPES.pop("_audit_probe", None)


def test_unmatched_prefixes_make_the_result_not_ok():
    result = ReloadResult(scope="llm", reloaded=["core.brain.inference_gate"])
    assert result.ok is True
    result.unmatched_prefixes = ["llm:core.brain.llm.inference_gate"]
    assert result.ok is False
    assert result.to_dict()["unmatched_prefixes"] == [
        "llm:core.brain.llm.inference_gate"
    ]


def test_the_reply_shaping_layer_is_reachable_by_some_scope():
    """A fix to reply reliability was previously in no scope at all."""
    covered = {
        prefix
        for scope in LIVE_SAFE_ALL_SCOPES
        for prefix in RELOAD_SCOPES.get(scope, [])
    }
    for module in (
        "core.conversation.",
        "core.brain.epistemic_firewall",
        "core.brain.inference_gate",
    ):
        assert module in covered, f"{module} cannot be hot-reloaded by any scope"
