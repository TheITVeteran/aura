"""Contract tests for the bidirectional Global Workspace ↔ RLC coupling.

Direction 1: live coalitions become typed cognitive-context items that seed
identifiable thought slots (organ items keep priority).
Direction 2: episode conclusions return to the workspace as competing
candidates priced by how they were earned — broadcast is won, never granted.
Lab/background episodes must stay fully decoupled from the live mind.
"""

import asyncio

import pytest

from core.brain import gwt_rlc_coupling as coupling
from core.consciousness.global_workspace import (
    CognitiveCandidate,
    ContentType,
    GlobalWorkspace,
)


class _FakeWinner:
    def __init__(self, content: str, source: str) -> None:
        self.content = content
        self.source = source


class _FakeWorkspace:
    def __init__(self, winner=None, coalitions=None, accept=True) -> None:
        self.last_winner = winner
        self._coalitions = list(coalitions or [])
        self._accept = accept
        self.submitted: list[CognitiveCandidate] = []

    def get_competing_coalitions(self, n: int = 3):
        return self._coalitions[: max(0, int(n))]

    async def submit(self, candidate: CognitiveCandidate) -> bool:
        self.submitted.append(candidate)
        return self._accept


# ── Direction 1: coalitions → slot candidates ───────────────────────────


def test_coalition_context_reads_broadcast_then_coalitions(monkeypatch):
    ws = _FakeWorkspace(
        winner=_FakeWinner("the plan is brittle", "anti_brittleness_impulse"),
        coalitions=[
            {"source": "memory", "content": "user asked about entropy"},
            {"source": "goal_system", "content": "finish the proof"},
        ],
    )
    monkeypatch.setattr(coupling, "_workspace", lambda: ws)
    items = coupling.workspace_coalition_context(max_items=3)
    assert items[0]["source"] == "workspace_broadcast"
    assert "anti_brittleness_impulse" in items[0]["text"]
    assert "the plan is brittle" in items[0]["text"]
    assert items[1]["source"] == "workspace_coalition"
    assert "memory" in items[1]["text"]
    assert len(items) == 3


def test_coalition_context_skips_own_echo_and_empty(monkeypatch):
    ws = _FakeWorkspace(
        winner=None,
        coalitions=[
            {"source": "latent_cortex", "content": "my own prior conclusion"},
            {"source": "", "content": "sourceless"},
            {"source": "affect", "content": ""},
            {"source": "world_model", "content": "rain is likely"},
        ],
    )
    monkeypatch.setattr(coupling, "_workspace", lambda: ws)
    items = coupling.workspace_coalition_context(max_items=4)
    texts = " | ".join(i["text"] for i in items)
    assert "my own prior conclusion" not in texts
    assert "rain is likely" in texts


def test_coalition_context_bounds_text_and_items(monkeypatch):
    ws = _FakeWorkspace(
        winner=_FakeWinner("x" * 5000, "flood"),
        coalitions=[
            {"source": f"s{i}", "content": "y" * 5000} for i in range(10)
        ],
    )
    monkeypatch.setattr(coupling, "_workspace", lambda: ws)
    items = coupling.workspace_coalition_context(max_items=2)
    assert len(items) == 2
    assert all(len(i["text"]) <= 400 for i in items)


def test_coalition_context_no_workspace_is_noop(monkeypatch):
    monkeypatch.setattr(coupling, "_workspace", lambda: None)
    assert coupling.workspace_coalition_context() == []


def test_merge_keeps_organ_priority_and_fills_remaining(monkeypatch):
    ws = _FakeWorkspace(
        winner=_FakeWinner("broadcast content", "drive"),
        coalitions=[{"source": "memory", "content": "coalition content"}],
    )
    monkeypatch.setattr(coupling, "_workspace", lambda: ws)
    organ = [
        {"source": f"organ_{i}", "text": f"organ item {i}"} for i in range(5)
    ]
    merged = coupling.merge_cognitive_context(organ, max_items=6)
    assert merged is not None
    assert [m["source"] for m in merged[:5]] == [o["source"] for o in organ]
    assert len(merged) == 6  # exactly one coalition slot left
    assert merged[5]["source"] == "workspace_broadcast"


def test_merge_full_organ_context_leaves_no_room(monkeypatch):
    called = {"n": 0}

    def _ws():
        called["n"] += 1
        return _FakeWorkspace(winner=_FakeWinner("w", "s"))

    monkeypatch.setattr(coupling, "_workspace", _ws)
    organ = [{"source": f"o{i}", "text": f"t{i}"} for i in range(6)]
    merged = coupling.merge_cognitive_context(organ, max_items=6)
    assert merged is not None and len(merged) == 6
    assert all(m["source"].startswith("o") for m in merged)


def test_merge_empty_everything_returns_none(monkeypatch):
    monkeypatch.setattr(coupling, "_workspace", lambda: None)
    assert coupling.merge_cognitive_context(None) is None
    assert coupling.merge_cognitive_context([]) is None


def test_real_workspace_get_competing_coalitions_sorted_and_bounded():
    gw = GlobalWorkspace()
    for name, prio in (("low", 0.2), ("high", 0.9), ("mid", 0.5)):
        gw._candidates.append(
            CognitiveCandidate(
                content=f"content-{name}" + "z" * 500,
                source=name,
                priority=prio,
                content_type=ContentType.META,
            )
        )
    rows = gw.get_competing_coalitions(2)
    assert len(rows) == 2
    assert rows[0]["source"] == "high"
    assert rows[1]["source"] == "mid"
    assert len(rows[0]["content"]) <= 400
    assert rows[0]["priority"] >= rows[1]["priority"]
    assert gw.get_competing_coalitions(0) == []


# ── Direction 2: conclusions → broadcast ────────────────────────────────


def test_priority_pricing_verified_beats_convergence_only():
    verified_receipt = {
        "verifier_guidance": {
            "evaluations": 3,
            "best_score": 0.9,
            "best_failures": [],
        }
    }
    unverified_receipt = {"verifier_guidance": {"evaluations": 0}}
    p_v, pricing_v = coupling._conclusion_priority(verified_receipt, 0.5)
    p_u, pricing_u = coupling._conclusion_priority(unverified_receipt, 0.5)
    assert p_v > p_u
    assert pricing_v["verified"] is True
    assert pricing_u["verified"] is False


def test_priority_pricing_failures_void_verification():
    receipt = {
        "verifier_guidance": {
            "evaluations": 2,
            "best_score": 0.9,
            "best_failures": ["contradiction"],
        }
    }
    _, pricing = coupling._conclusion_priority(receipt, 0.5)
    assert pricing["verified"] is False


@pytest.mark.parametrize(
    "bad_best",
    [float("nan"), float("inf"), True, "0.9", None],
)
def test_priority_pricing_rejects_junk_scores(bad_best):
    receipt = {
        "verifier_guidance": {"evaluations": 1, "best_score": bad_best}
    }
    priority, pricing = coupling._conclusion_priority(receipt, 0.5)
    assert pricing["verified"] is False
    assert 0.0 <= priority <= 0.9


def test_priority_capped_and_stakes_bounded():
    receipt = {
        "verifier_guidance": {
            "evaluations": 5,
            "best_score": 1.0,
            "best_failures": [],
        }
    }
    priority, pricing = coupling._conclusion_priority(receipt, 99.0)
    assert priority <= 0.9
    assert pricing["stakes"] == 1.0
    priority_low, _ = coupling._conclusion_priority({}, -5.0)
    assert priority_low == pytest.approx(0.5)


def test_broadcast_submits_competing_candidate(monkeypatch):
    ws = _FakeWorkspace(accept=True)
    monkeypatch.setattr(coupling, "_workspace", lambda: ws)
    receipt = {
        "episode_id": "ep-77",
        "verifier_guidance": {
            "evaluations": 2,
            "best_score": 0.8,
            "best_failures": [],
        },
    }
    out = asyncio.run(
        coupling.broadcast_episode_conclusion(
            "why is the sky blue", "Rayleigh scattering.", receipt, stakes=0.8
        )
    )
    assert out["submitted"] is True and out["accepted"] is True
    assert len(ws.submitted) == 1
    cand = ws.submitted[0]
    assert cand.source == "latent_cortex"
    assert cand.content_type is ContentType.META
    assert cand.content == "Rayleigh scattering."
    assert cand.metadata["schema"] == coupling.GWT_RLC_SCHEMA
    assert cand.metadata["episode_id"] == "ep-77"
    assert len(cand.metadata["objective_sha256"]) == 64
    assert cand.priority == out["priority"] <= 0.9
    assert cand.metadata["pricing"]["verified"] is True


def test_broadcast_no_workspace_and_empty_text_are_receipted_noops(monkeypatch):
    monkeypatch.setattr(coupling, "_workspace", lambda: None)
    out = asyncio.run(
        coupling.broadcast_episode_conclusion("q", "answer", {}, stakes=0.5)
    )
    assert out == {
        "schema": coupling.GWT_RLC_SCHEMA,
        "submitted": False,
        "reason": "workspace_absent",
    }
    ws = _FakeWorkspace()
    monkeypatch.setattr(coupling, "_workspace", lambda: ws)
    out = asyncio.run(
        coupling.broadcast_episode_conclusion("q", "   ", {}, stakes=0.5)
    )
    assert out["submitted"] is False
    assert out["reason"] == "empty_conclusion"
    assert ws.submitted == []


def test_broadcast_rejection_is_still_a_receipt(monkeypatch):
    ws = _FakeWorkspace(accept=False)
    monkeypatch.setattr(coupling, "_workspace", lambda: ws)
    out = asyncio.run(
        coupling.broadcast_episode_conclusion("q", "conclusion", {}, stakes=0.2)
    )
    assert out["submitted"] is True
    assert out["accepted"] is False  # competition can be lost — honestly


def test_broadcast_submit_failure_degrades_not_raises(monkeypatch):
    class _Exploding:
        last_winner = None

        async def submit(self, candidate):
            raise RuntimeError("workspace wedged")

    monkeypatch.setattr(coupling, "_workspace", lambda: _Exploding())
    out = asyncio.run(
        coupling.broadcast_episode_conclusion("q", "conclusion", {}, stakes=0.2)
    )
    assert out["submitted"] is False
    assert out["reason"] == "submit_failed:RuntimeError"
