"""`generate` returned None for four different things, and None said nothing.

A background admission deferral, a proof contract that names Cortex, RAM
admission deferring a cold load, and critical memory pressure all exited with
`return None` after a log line. That is the same value the model returns when
it produces no text, so a caller could not tell "policy refused this" from
"the model said nothing" — and the two want opposite handling: one is retry
later, the other is a failure to investigate.

The return contract stays `str | None`, because every caller depends on it.
What changed is that None is no longer the whole message: a typed receipt goes
to the caller's own context, to the gate for health, and to the turn ledger,
whose terminal status is then a refusal rather than a turn that mysteriously
held no answer.

The second half of this file is the request clock. Every attempt used to open
a window of its own — the primary attempt, each repair at 30-60s, the
brainstem, the reflex, APIAdapter at 30s, then HealthRouter at another 30 — so
a caller asking for 45 seconds could wait several minutes while every
individual `wait_for` was "within budget".
"""
from __future__ import annotations

import ast
import inspect

import pytest

from core.brain.inference_gate import InferenceGate
from core.utils.deadlines import get_deadline


def _gate():
    gate = InferenceGate.__new__(InferenceGate)
    gate._last_refusal_receipt = {}
    return gate


# ────────────────────────────── the refusal is typed and reaches the caller


def test_a_refusal_reaches_the_caller_through_its_own_context():
    gate = _gate()
    context: dict = {}

    result = gate._refuse_generation(
        InferenceGate.REFUSAL_RESOURCE,
        "critical_memory_pressure",
        context=context,
        origin="desktop_user",
    )

    assert result is None, "the return contract changed"
    receipt = context["inference_refusal"]
    assert receipt["kind"] == InferenceGate.REFUSAL_RESOURCE
    assert receipt["reason"] == "critical_memory_pressure"
    assert receipt["origin"] == "desktop_user"


def test_the_gate_keeps_the_last_refusal_for_health():
    gate = _gate()

    gate._refuse_generation(
        InferenceGate.REFUSAL_DEFERRED, "foreground_quiet_window", context=None
    )

    assert gate.last_refusal_receipt()["reason"] == "foreground_quiet_window"


def test_the_receipt_is_a_copy_not_the_live_record():
    gate = _gate()
    gate._refuse_generation(InferenceGate.REFUSAL_DEFERRED, "reason", context=None)

    gate.last_refusal_receipt()["reason"] = "tampered"

    assert gate.last_refusal_receipt()["reason"] == "reason"


def test_an_unknown_retry_time_is_omitted_rather_than_zeroed():
    """A zero would read as "retry immediately", which is a claim nobody made."""
    gate = _gate()
    context: dict = {}

    gate._refuse_generation(
        InferenceGate.REFUSAL_DEFERRED, "reason", context=context, origin="x"
    )

    assert "retry_after_s" not in context["inference_refusal"]


def test_a_known_retry_time_is_carried():
    gate = _gate()
    context: dict = {}

    gate._refuse_generation(
        InferenceGate.REFUSAL_DEFERRED,
        "reason",
        context=context,
        retry_after_s=12.5,
    )

    assert context["inference_refusal"]["retry_after_s"] == 12.5


def test_the_turn_ledger_records_a_refusal_not_a_missing_answer():
    from core.runtime.turn_outcome import TurnOutcome, bind_turn

    gate = _gate()
    outcome = TurnOutcome(origin="user_chat")

    with bind_turn(outcome):
        gate._refuse_generation(
            InferenceGate.REFUSAL_PROOF_LANE, "primary_required", context=None
        )

    receipt = outcome.finalize()
    assert receipt.status.value == "refused"
    assert any(r["kind"] == "inference_refusal" for r in receipt.causal_receipts)


def test_no_turn_bound_is_not_an_error():
    """Background work and tools run with no turn; a refusal must still work."""
    gate = _gate()

    assert gate._refuse_generation(InferenceGate.REFUSAL_DEFERRED, "r", context=None) is None


def test_every_policy_exit_goes_through_the_refusal_receipt():
    """A bare `return None` added later would put the ambiguity straight back."""
    import core.brain.inference_gate as gate_mod

    source = inspect.getsource(gate_mod)
    tree = ast.parse(source)

    generate = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "generate"
        and node.args.args
        and [arg.arg for arg in node.args.args][:2] == ["self", "prompt"]
    )

    bare = [
        node.lineno
        for node in ast.walk(generate)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and node.value.value is None
    ]

    assert not bare, (
        f"generate still returns a bare None — indistinguishable from an empty "
        f"model answer — at line(s) {bare}"
    )


# ────────────────────────────── one clock for the whole request


def test_an_attempt_never_gets_more_than_the_request_has_left():
    deadline = get_deadline(10.0)

    assert InferenceGate._window_within(deadline, 30.0) <= 10.0


def test_an_attempt_asking_for_less_keeps_its_own_window():
    deadline = get_deadline(100.0)

    assert InferenceGate._window_within(deadline, 5.0) == 5.0


def test_an_exhausted_request_gets_no_window_at_all():
    """Not a small window — none. An attempt that starts after the caller's
    deadline cannot deliver anything the caller is still waiting for, and it
    holds the model lane while it fails."""
    deadline = get_deadline(0.0)

    assert InferenceGate._window_within(deadline, 30.0) == 0.0


def test_an_unbounded_request_leaves_the_attempt_alone():
    deadline = get_deadline(None)

    assert InferenceGate._window_within(deadline, 30.0) == 30.0


def test_a_missing_deadline_object_does_not_break_dispatch():
    assert InferenceGate._window_within(None, 30.0) == 30.0


@pytest.mark.parametrize(
    "variable",
    ["primary_deadline", "retry_deadline", "fallback_deadline", "reflex_deadline"],
)
def test_every_local_attempt_is_capped_by_the_request_deadline(variable):
    import core.brain.inference_gate as gate_mod

    source = inspect.getsource(gate_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if variable not in targets:
            continue
        rendered = ast.get_source_segment(source, node) or ""
        assert "_window_within" in rendered, (
            f"{variable} opens a window outside the caller's budget"
        )
        return
    raise AssertionError(f"{variable} was not found")


def test_both_cloud_paths_are_capped_by_the_request_deadline():
    import core.brain.inference_gate as gate_mod

    source = inspect.getsource(gate_mod)

    assert "cloud_window_s = self._window_within(" in source
    assert "router_window_s = self._window_within(" in source
    # And no hard 30-second cloud budget survives.
    assert "timeout=30.0," not in source, (
        "a cloud attempt still runs on its own thirty-second clock"
    )
