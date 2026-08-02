"""A governance organ nobody can see is not governance.

Both of these were built, tested, and invisible: the durable-learning
gate's own ``report()`` had no caller, and nothing published which runtime
wrote which state. That is the residue shape this codebase keeps finding —
substantial, correct in isolation, and uninvoked.

Also pinned here: the streaming turn. chat_stream prefers ``think_stream``
(``hasattr`` is checked first), so it is THE live desktop turn, and it does
not route through ``think`` at all. A ledger bound only to ``think`` would
have looked wired while missing the real path.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _integrity() -> dict:
    from core.runtime.health_contract import _runtime_integrity_block

    return _runtime_integrity_block()


def test_the_health_surface_names_the_runtime_that_is_reporting():
    identity = _integrity()["runtime_identity"]
    assert identity["runtime_instance_id"]
    assert identity["runtime_profile"] == "test"
    assert identity["state_root"]


def test_the_health_surface_reports_what_aura_was_allowed_to_learn():
    learning = _integrity()["durable_learning"]
    for key in (
        "admissions",
        "durable_updates",
        "rolled_back",
        "quarantined",
        "quarantine_awaiting_review",
        "verifiers",
    ):
        assert key in learning, f"the health surface does not report {key}"


def test_durable_learning_visible_on_the_surface_reflects_real_admissions():
    """The report must track the gate, not be a static shape."""
    from core.governance.durable_learning import (
        VerificationGrade,
        get_durable_learning_gate,
    )

    before = _integrity()["durable_learning"]["admissions"]["session"]
    get_durable_learning_gate().admit(
        __import__(
            "core.governance.durable_learning", fromlist=["LearningUpdate"]
        ).LearningUpdate(
            subsystem="mycelium",
            key="health-surface-probe",
            operation="reinforce",
            success=True,
            grade=VerificationGrade.ASSERTED,
        )
    )
    after = _integrity()["durable_learning"]["admissions"]["session"]
    assert after == before + 1


def test_the_health_surface_says_whether_admission_is_measuring_yet():
    from core.brain.llm.measured_admission import get_throughput_estimator

    get_throughput_estimator()  # registers itself
    throughput = _integrity()["admission_throughput"]
    assert "shapes_measured" in throughput or throughput == {"registered": False}


def test_the_runtime_layer_does_not_import_cognition_to_do_it():
    """The foundation must boot and report on a mind that failed to start."""
    source = (ROOT / "core" / "runtime" / "health_contract.py").read_text("utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("core.brain"), (
                f"health_contract imports {node.module}; the runtime foundation "
                "may not depend on cognition"
            )


# ------------------------------------------------------- the streaming turn


def _think_stream() -> ast.AsyncFunctionDef:
    tree = ast.parse((ROOT / "core" / "brain" / "cognitive_engine.py").read_text("utf-8"))
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "think_stream"
    )


def test_the_streaming_turn_is_under_the_ledger_too():
    called = {
        node.func.id
        for node in ast.walk(_think_stream())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "bind_turn" in called, (
        "think_stream does not bind a turn; chat_stream prefers this path, so "
        "the live desktop turn would have no ledger"
    )
    assert "finalize_turn" in called, "the streaming turn has no terminal finalizer"
    assert "TurnOutcome" in called


def test_the_streaming_turn_records_what_the_person_received():
    called = {
        node.func.attr
        for node in ast.walk(_think_stream())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "mark_served" in called, (
        "the stream never records what was served, so an empty stream would "
        "finalize as though nothing had been attempted"
    )


def test_the_stream_still_yields_its_tokens():
    """The binding must not have turned a generator into a coroutine."""
    import inspect

    from core.brain.cognitive_engine import CognitiveEngine

    assert inspect.isasyncgenfunction(CognitiveEngine.think_stream), (
        "think_stream stopped being an async generator; chat_stream iterates it"
    )
