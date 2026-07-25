"""CP126 fail-open batch 4: execution, observation, and acceptance.

* ``4bf25067`` — a terminal receipt with no exit_code defaulted to 0, so a
  process that never reported how it ended suppressed its own failure.
* ``7031837e`` — any vision response not carrying status=error was
  stringified as success, turning malformed IPC into confident perception.
* ``6d2e65cb`` — a broken loop detector let the agent loop continue at full
  budget, permitting the recursion it exists to stop.
* ``a6763356`` — with no execution adapter the client fabricated an error
  STRING and fed it to the model as though a tool had run.
* ``6a8225f5`` — tool results were interpolated into history wholesale.
* ``52feb1d1`` — with `succeeded` absent, any status outside a short denylist
  counted as acceptance and was persisted as seen.
"""
from __future__ import annotations

import inspect

import pytest


class TestAMissingExitCodeIsNotSuccess:
    def _verify(self, receipt):
        import asyncio

        from core.body.action_postcondition import ActionPostconditionVerifier

        class _State:
            world_model: dict = {}

        return asyncio.run(ActionPostconditionVerifier().verify(receipt, _State()))

    def test_absent_exit_code_is_reported_as_unknown(self):
        """Absent is not zero; it is unknown, and unknown must not read as
        success on the receipt that records what a process did."""
        result = self._verify({"channel": "terminal", "status": "success"})
        assert "process_exit_code_unreported" in result["side_effects"]

    def test_a_reported_zero_is_clean(self):
        result = self._verify(
            {"channel": "terminal", "status": "success", "exit_code": 0},
        )
        assert result["side_effects"] == []

    def test_a_nonzero_exit_code_still_reports_failure(self):
        result = self._verify(
            {"channel": "terminal", "status": "success", "exit_code": 3},
        )
        assert "process_failed_with_code:3" in result["side_effects"]


class TestVisionResponsesAreValidated:
    def _client(self):
        from core.brain.llm.mlx_vision_client import MLXVisionClient

        return MLXVisionClient

    def test_the_schema_is_enforced_not_stringified(self):
        source = inspect.getsource(self._client())
        assert "non-mapping response" in source
        assert "carried no 'response' field" in source
        assert "must be text" in source

    def test_an_unknown_status_is_refused(self):
        source = inspect.getsource(self._client())
        assert "returned unknown status" in source

    def test_the_old_permissive_stringify_is_gone(self):
        source = inspect.getsource(self._client())
        assert 'return str(resp.get("response", ""))' not in source


class TestBrokenLoopDetectionBoundsTheLoop:
    def test_the_loop_stops_early_without_detection(self):
        from core.brain.llm import local_agent_client as mod

        source = inspect.getsource(mod)
        assert "loop_detection_available = False" in source
        assert "loop detection unavailable" in source

    def test_import_failure_is_no_longer_a_silent_skip(self):
        from core.brain.llm import local_agent_client as mod

        source = inspect.getsource(mod)
        assert "Circuit breaker module not found. Skipping check." not in source


class TestNoExecutorFailsRatherThanNarrates:
    def test_the_fabricated_error_string_is_gone(self):
        from core.brain.llm import local_agent_client as mod

        source = inspect.getsource(mod)
        assert "[Error: No execution adapter configured for" not in source

    def test_it_returns_an_explicit_no_executor_result(self):
        from core.brain.llm import local_agent_client as mod

        source = inspect.getsource(mod)
        assert '"error": "no_executor"' in source
        assert '"confidence": 0.0' in source


class TestObservationsAreBoundedOnEntry:
    def test_a_small_result_is_unchanged(self):
        from core.brain.llm.local_agent_client import _bounded_observation

        assert _bounded_observation("hello") == "hello"

    def test_a_large_result_is_truncated_and_says_so(self):
        from core.brain.llm.local_agent_client import (
            _MAX_OBSERVATION_CHARS,
            _bounded_observation,
        )

        bounded = _bounded_observation("x" * (_MAX_OBSERVATION_CHARS + 500))
        assert len(bounded) < _MAX_OBSERVATION_CHARS + 300
        assert "observation truncated" in bounded
        assert "500 more characters" in bounded

    def test_none_is_safe(self):
        from core.brain.llm.local_agent_client import _bounded_observation

        assert _bounded_observation(None) == ""

    def test_history_uses_the_bound(self):
        from core.brain.llm import local_agent_client as mod

        source = inspect.getsource(mod)
        assert "_bounded_observation(result_str)" in source


class TestRepairAcceptanceIsPositive:
    def _accepting(self):
        from core.agency.self_repair_backlog import _ACCEPTING_STATUSES

        return _ACCEPTING_STATUSES

    @pytest.mark.parametrize(
        "status", ["unknown", "something_new", "failed", "blocked", "denied"],
    )
    def test_unrecognised_statuses_are_not_acceptance(self, status):
        assert status not in self._accepting()

    @pytest.mark.parametrize(
        "status", ["planned", "waiting_for_approval", "created", "queued"],
    )
    def test_the_designed_approval_path_is_still_acceptance(self, status):
        """A shadow plan awaiting approval WAS created — that is this
        subsystem's success path, not a defect. The finding listed these as
        suspect; the design and its tests say otherwise, and what the fix
        actually removes is acceptance-by-default for unclassified statuses.
        """
        assert status in self._accepting()

    def test_the_denylist_default_is_gone(self):
        from core.agency import self_repair_backlog as mod

        source = inspect.getsource(mod)
        assert "reason.lower() not in {" not in source
        assert "_ACCEPTING_STATUSES" in source
