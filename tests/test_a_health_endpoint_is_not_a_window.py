"""What the agency health endpoint publishes, and what liveness is derived from.

`get_status` began with `self.state.model_dump()` — the whole AgencyState — so
every health consumer received the ambient screen context, the perceptual
buffer, the queued visual and audio observations Aura had not yet chosen to
share, the pending goals, and the topics she was thinking about raising. A
health endpoint answers "is this subsystem working". It is not a window into
what she has seen and not mentioned.

And liveness was two sticky error strings. One earlier tool-routing failure
held `alive` false until a later successful tool call happened to clear it, and
nothing else could make it report degraded at all.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.agency.agency_core import _AGENCY_STATUS_FIELDS, AgencyCore, AgencyState


def _core(**overrides) -> AgencyCore:
    core = AgencyCore.__new__(AgencyCore)
    core.state = AgencyState()
    core.state.unshared_observations = ["he looked tired in the 3pm call"]
    core.state.topics_to_discuss = ["the thing he said about leaving"]
    core.state.pending_goals = [{"text": "ask about the interview"}]
    core.state.perceptual_buffer = {"screen": "a private message is open"}
    core.state.current_ambient_context = "bank statement on the second monitor"
    core._last_viability_error = None
    core._last_tool_routing_error = None
    core._pathway_registry = {"a": lambda *_a: None}
    core._action_queue = []
    core._pathway_hooks = {}
    core._cognitive_loop_task = None
    core._last_cognitive_loop_receipt = {}
    for key, value in overrides.items():
        setattr(core, key, value)
    return core


def test_the_unshared_observations_do_not_leave_through_health():
    status = _core().get_status()
    blob = repr(status)

    assert "he looked tired" not in blob
    assert "the thing he said about leaving" not in blob
    assert "a private message is open" not in blob
    assert "bank statement" not in blob


def test_the_counts_are_published_so_a_growing_queue_is_visible():
    status = _core().get_status()

    assert status["unshared_observation_count"] == 1
    assert status["topics_to_discuss_count"] == 1
    assert status["pending_goal_count"] == 1
    assert status["perceptual_buffer_keys"] == ["screen"]
    assert status["ambient_context_present"] is True


def test_the_published_field_list_holds_no_container():
    """A list or dict added to AgencyState must not reach the endpoint by
    being added to this tuple without thought."""
    state = AgencyState()
    for field in _AGENCY_STATUS_FIELDS:
        value = getattr(state, field)
        assert not isinstance(value, (list, dict, set)), field


def test_a_healthy_core_is_alive():
    core = _core()

    assert core.get_status()["alive"] is True
    assert core._degraded_reasons() == []
    assert core.is_alive() is True


def test_an_empty_pathway_registry_is_degraded():
    """Nothing but the two error strings could make this report degraded."""
    core = _core(_pathway_registry={})

    assert "no_pathways_registered" in core._degraded_reasons()
    assert core.get_status()["alive"] is False


def test_a_stopped_cognitive_loop_is_degraded():
    core = _core(_cognitive_loop_task=SimpleNamespace(done=lambda: True))

    assert "cognitive_loop_stopped" in core._degraded_reasons()


def test_a_running_cognitive_loop_is_not():
    core = _core(_cognitive_loop_task=SimpleNamespace(done=lambda: False))

    assert "cognitive_loop_stopped" not in core._degraded_reasons()


@pytest.mark.parametrize(
    ("attribute", "prefix"),
    [("_last_viability_error", "viability:"), ("_last_tool_routing_error", "tool_routing:")],
)
def test_each_error_string_names_itself_in_the_reason_list(attribute, prefix):
    core = _core(**{attribute: "something broke"})
    reasons = core._degraded_reasons()

    assert any(r.startswith(prefix) for r in reasons), reasons
    assert core.get_status()["degraded_reasons"] == reasons


def test_clearing_the_error_clears_the_reason():
    """The strings are sticky by design; what must not stick is the verdict
    once they are gone."""
    core = _core(_last_tool_routing_error="a failure")
    assert core.get_status()["alive"] is False

    core._last_tool_routing_error = None
    assert core.get_status()["alive"] is True
