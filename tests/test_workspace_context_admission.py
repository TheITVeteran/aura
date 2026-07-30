"""Context the workspace could not seat must not vanish from the record.

CP126 (high), core/brain/llm/latent_cortex/workspace.py: "Excess context
seeds are silently dropped."

The cap itself is structural and correct: slot 0 is the communication slot
and at least one hypothesis slot must stay private, so a four-slot workspace
can seat two evidence seeds and no more. What was missing is that the caller
admitted N pieces of cognitive context and the workspace kept fewer with
nothing saying so — a reasoning trace built on two of five memories looked
exactly like one built on all five.
"""
from __future__ import annotations

import pytest

from core.brain.llm.latent_cortex.types import WorkspaceConfig
from core.brain.llm.latent_cortex.workspace import LatentWorkspace


def _workspace(admission=None, n_slots=4):
    ws = LatentWorkspace.__new__(LatentWorkspace)
    ws.config = WorkspaceConfig(n_slots=n_slots)
    ws.context_slots = []
    ws.context_admission = admission or {
        "schema": "aura.workspace_context_admission.v1",
        "requested": 0, "admitted": 0, "dropped": 0,
        "dropped_sources": [], "n_slots": n_slots, "complete": True,
    }
    return ws


class TestTheAdmissionIsCarried:
    def test_a_workspace_reports_a_complete_admission_by_default(self):
        assert _workspace().context_admission["complete"] is True

    def test_the_schema_is_stated(self):
        assert _workspace().context_admission["schema"] == (
            "aura.workspace_context_admission.v1"
        )

    def test_a_partial_admission_is_marked_incomplete(self):
        ws = _workspace({
            "schema": "aura.workspace_context_admission.v1",
            "requested": 5, "admitted": 2, "dropped": 3,
            "dropped_sources": ["world_model", "self_model", "goal"],
            "n_slots": 4, "complete": False,
        })
        assert ws.context_admission["complete"] is False
        assert ws.context_admission["dropped"] == 3

    def test_the_dropped_sources_are_named(self):
        """Knowing three were dropped is less useful than knowing which."""
        ws = _workspace({
            "schema": "aura.workspace_context_admission.v1",
            "requested": 5, "admitted": 2, "dropped": 3,
            "dropped_sources": ["world_model", "self_model", "goal"],
            "n_slots": 4, "complete": False,
        })
        assert "world_model" in ws.context_admission["dropped_sources"]

    def test_requested_equals_admitted_plus_dropped(self):
        ws = _workspace({
            "schema": "aura.workspace_context_admission.v1",
            "requested": 5, "admitted": 2, "dropped": 3,
            "dropped_sources": [], "n_slots": 4, "complete": False,
        })
        a = ws.context_admission
        assert a["requested"] == a["admitted"] + a["dropped"]


class TestTheCapIsStillEnforced:
    """The structural reservation must not regress: seating context into
    the comm slot or the last hypothesis slot would break the workspace."""

    def test_the_seat_formula_reserves_two_slots(self):
        import inspect

        source = inspect.getsource(LatentWorkspace.from_prompt_embeddings)
        assert "m - 2" in source

    def test_the_drop_is_recorded_not_swallowed(self):
        import inspect

        source = inspect.getsource(LatentWorkspace.from_prompt_embeddings)
        block = source[source.index("dropped = requested_seeds[max_context:]"):]
        assert "record_degradation" in block[:600]

    def test_a_dropped_seed_never_silently_reduces_the_count(self):
        import inspect

        source = inspect.getsource(LatentWorkspace.from_prompt_embeddings)
        assert "context_admission" in source
        assert '"requested": len(requested_seeds)' in source
