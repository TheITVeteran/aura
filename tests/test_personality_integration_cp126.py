"""CP126 contract tests for personality-systems integration.

The point: "integrated" must mean a filter is installed, and "aligned" must be
established by inspecting the live wiring.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.brain import personality_integration as module
from core.brain.personality_integration import (
    integrate_all_personality_systems,
    personality_integration_status,
    uninstall_personality_systems,
    verify_all_systems_aligned,
)


class _Engine:
    def __init__(self, prefix="[shaped] "):
        self.prefix = prefix
        self.calls = []

    def filter_response(self, text):
        self.calls.append(text)
        return f"{self.prefix}{text}"


class _Queue:
    def __init__(self):
        self.items = []

    def put_nowait(self, item):
        self.items.append(item)
        return item


class _Comm:
    def __init__(self):
        self.queued = []

    def queue_message(self, content, emotion, urgency, context=None):
        self.queued.append(content)
        return content


@pytest.fixture()
def engine(monkeypatch) -> _Engine:
    made = _Engine()
    monkeypatch.setattr(module, "record_degradation", lambda *a, **k: None)
    import core.brain.personality_engine as pe

    monkeypatch.setattr(pe, "get_personality_engine", lambda: made)
    return made


@pytest.fixture()
def orchestrator() -> SimpleNamespace:
    return SimpleNamespace(reply_queue=_Queue(), proactive_comm=_Comm())


# --- fdbc444b: success requires a real hook ------------------------------


def test_integration_with_no_hook_points_is_not_success(engine):
    bare = SimpleNamespace()

    receipt = integrate_all_personality_systems(bare)

    assert receipt.ok is False
    assert bool(receipt) is False
    assert receipt.installed == []
    assert any("no_hook_point" in item or "absent" in item for item in receipt.skipped)


def test_integration_with_a_reply_queue_is_success(engine, orchestrator):
    receipt = integrate_all_personality_systems(orchestrator)

    assert receipt.ok is True
    assert "reply_queue" in receipt.installed
    assert "proactive_comm" in receipt.installed


def test_the_receipt_serializes(engine, orchestrator):
    payload = integrate_all_personality_systems(orchestrator).to_dict()

    assert payload["ok"] is True
    assert payload["engine_available"] is True


def test_an_orchestrator_without_proactive_comm_is_partial(engine):
    only_queue = SimpleNamespace(reply_queue=_Queue())

    receipt = integrate_all_personality_systems(only_queue)

    assert receipt.ok is True
    assert receipt.installed == ["reply_queue"]
    assert "proactive_comm:absent" in receipt.skipped


# --- cd83e384: failure is explicit, not a quiet False -------------------


def test_an_unavailable_engine_is_a_typed_failure(monkeypatch):
    import core.brain.personality_engine as pe

    def boom():
        raise RuntimeError("engine missing")

    monkeypatch.setattr(pe, "get_personality_engine", boom)
    monkeypatch.setattr(module, "record_degradation", lambda *a, **k: None)

    receipt = integrate_all_personality_systems(SimpleNamespace(reply_queue=_Queue()))

    assert receipt.ok is False
    assert any("unavailable" in item for item in receipt.errors)


def test_an_engine_without_filter_response_is_refused(monkeypatch):
    import core.brain.personality_engine as pe

    monkeypatch.setattr(pe, "get_personality_engine", lambda: object())
    monkeypatch.setattr(module, "record_degradation", lambda *a, **k: None)

    receipt = integrate_all_personality_systems(SimpleNamespace(reply_queue=_Queue()))

    assert receipt.ok is False
    assert "personality_engine_has_no_filter_response" in receipt.errors


def test_a_failure_raises_a_degradation(monkeypatch, orchestrator):
    import core.brain.personality_engine as pe

    recorded = []
    monkeypatch.setattr(pe, "get_personality_engine", lambda: _Engine())
    monkeypatch.setattr(module, "record_degradation", lambda *a, **k: recorded.append(a))
    integrate_all_personality_systems(SimpleNamespace())

    assert recorded


# --- 4504ac11: idempotent installation with teardown --------------------


def test_repeated_integration_does_not_stack_filters(engine, orchestrator):
    integrate_all_personality_systems(orchestrator)
    integrate_all_personality_systems(orchestrator)
    integrate_all_personality_systems(orchestrator)

    orchestrator.reply_queue.put_nowait("hello")

    # One filter, so exactly one prefix.
    assert orchestrator.reply_queue.items[0] == "[shaped] hello"


def test_repeated_proactive_integration_does_not_stack(engine, orchestrator):
    integrate_all_personality_systems(orchestrator)
    integrate_all_personality_systems(orchestrator)

    orchestrator.proactive_comm.queue_message("hi", "calm", 0.1)

    assert orchestrator.proactive_comm.queued == ["[shaped] hi"]


def test_uninstall_restores_the_originals(engine, orchestrator):
    integrate_all_personality_systems(orchestrator)

    removed = uninstall_personality_systems(orchestrator)
    orchestrator.reply_queue.put_nowait("plain")
    orchestrator.proactive_comm.queue_message("plain", "calm", 0.1)

    assert set(removed) == {"reply_queue", "proactive_comm"}
    assert orchestrator.reply_queue.items[-1] == "plain"
    assert orchestrator.proactive_comm.queued[-1] == "plain"


def test_uninstall_on_an_uninstalled_orchestrator_is_a_no_op(orchestrator):
    assert uninstall_personality_systems(orchestrator) == []


def test_status_reports_what_is_installed(engine, orchestrator):
    assert personality_integration_status(orchestrator) == {
        "reply_queue_filtered": False, "proactive_comm_filtered": False,
    }

    integrate_all_personality_systems(orchestrator)

    assert personality_integration_status(orchestrator) == {
        "reply_queue_filtered": True, "proactive_comm_filtered": True,
    }


# --- 2ed17445: the caller's object is not mutated -----------------------


def test_the_callers_dict_is_not_modified_in_place(engine, orchestrator):
    integrate_all_personality_systems(orchestrator)
    original = {"message": "evidence text", "id": 7}

    orchestrator.reply_queue.put_nowait(original)

    assert original["message"] == "evidence text"
    assert orchestrator.reply_queue.items[0]["message"] == "[shaped] evidence text"
    assert orchestrator.reply_queue.items[0] is not original


def test_the_prefilter_text_is_preserved(engine, orchestrator):
    integrate_all_personality_systems(orchestrator)

    orchestrator.reply_queue.put_nowait({"message": "raw"})

    assert orchestrator.reply_queue.items[0]["unfiltered_message"] == "raw"


def test_other_keys_survive(engine, orchestrator):
    integrate_all_personality_systems(orchestrator)

    orchestrator.reply_queue.put_nowait({"message": "m", "trace_id": "abc"})

    assert orchestrator.reply_queue.items[0]["trace_id"] == "abc"


def test_a_failing_filter_enqueues_the_original(monkeypatch, orchestrator):
    class Broken:
        def filter_response(self, text):
            raise RuntimeError("filter down")

    import core.brain.personality_engine as pe

    monkeypatch.setattr(pe, "get_personality_engine", lambda: Broken())
    monkeypatch.setattr(module, "record_degradation", lambda *a, **k: None)
    integrate_all_personality_systems(orchestrator)

    orchestrator.reply_queue.put_nowait({"message": "keep me"})

    assert orchestrator.reply_queue.items[0]["message"] == "keep me"


def test_a_non_message_item_passes_through(engine, orchestrator):
    integrate_all_personality_systems(orchestrator)

    orchestrator.reply_queue.put_nowait({"payload": 1})

    assert orchestrator.reply_queue.items[0] == {"payload": 1}


# --- e36030ad: alignment is established by inspection -------------------


def test_alignment_is_false_without_any_installed_filter(engine, orchestrator):
    assert verify_all_systems_aligned(orchestrator) is False


def test_alignment_is_true_once_a_filter_is_installed(engine, orchestrator):
    integrate_all_personality_systems(orchestrator)

    assert verify_all_systems_aligned(orchestrator) is True


def test_alignment_goes_false_again_after_uninstall(engine, orchestrator):
    integrate_all_personality_systems(orchestrator)
    uninstall_personality_systems(orchestrator)

    assert verify_all_systems_aligned(orchestrator) is False


def test_alignment_is_false_when_the_engine_cannot_filter(monkeypatch, orchestrator):
    import core.brain.personality_engine as pe

    monkeypatch.setattr(pe, "get_personality_engine", lambda: _Engine())
    monkeypatch.setattr(module, "record_degradation", lambda *a, **k: None)
    integrate_all_personality_systems(orchestrator)

    monkeypatch.setattr(pe, "get_personality_engine", lambda: object())

    assert verify_all_systems_aligned(orchestrator) is False


def test_alignment_does_not_depend_on_a_version_constant(engine, orchestrator, monkeypatch):
    """BEHAVIOUR: a kernel reporting the blessed version proves nothing.

    The old check returned `kernel.version == "3.5.5"`, ignoring its
    orchestrator argument entirely — so it attested alignment with no
    integration installed at all.
    """
    import core.brain.personality_kernel as pk

    monkeypatch.setattr(
        pk, "get_kernel", lambda: SimpleNamespace(version="3.5.5"), raising=False
    )

    # Blessed version, no filter installed: alignment must still be False.
    assert verify_all_systems_aligned(orchestrator) is False

    integrate_all_personality_systems(orchestrator)
    assert verify_all_systems_aligned(orchestrator) is True
