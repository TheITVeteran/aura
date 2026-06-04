from __future__ import annotations

from core.world_model.belief_graph import BeliefGraph


def test_active_surprise_updates_belief_graph(tmp_path):
    graph = BeliefGraph(
        persist_path=str(tmp_path / "world_model.json"),
        causal_path=str(tmp_path / "causal_graph.json"),
    )

    target = "file:aura-missing-note.txt"
    graph.update_belief("aura", "expects_exists", target, confidence_score=0.7)
    contradiction = graph.detect_contradiction("aura", "observed_missing", target)

    assert contradiction is not None
    assert contradiction["relation"] == "expects_exists"

    graph.update_belief("aura", "observed_missing", target, confidence_score=0.9)
    beliefs = graph.get_beliefs_about("aura")

    assert beliefs
    assert any(
        belief["target"] == target and belief["relation"] == "observed_missing"
        for belief in beliefs
    )
