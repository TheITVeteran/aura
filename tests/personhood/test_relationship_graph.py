"""tests/personhood/test_relationship_graph.py
============================================
Unit tests verifying the RelationshipGraph:
  1. Node creation and mirroring of preferences.
  2. Sentiment scoring and interaction logging.
  3. Boundary flags and shared project mapping.
"""

import pytest
import os
import json
from pathlib import Path
from core.social.relationship_graph import RelationshipGraph


@pytest.fixture
def temp_graph(tmp_path: Path):
    """Fixture that initializes a RelationshipGraph with isolated folder."""
    graph_dir = tmp_path / "social_graph"
    graph = RelationshipGraph(storage_dir=graph_dir)
    return graph


def test_node_creation_and_persistence(temp_graph):
    """Verify nodes are successfully created and persisted to disk."""
    graph = temp_graph
    
    assert len(graph.nodes) == 0
    
    # Create a node
    node = graph.get_or_create_node("u-1", name="Bryan", node_type="user")
    assert node.name == "Bryan"
    assert node.node_type == "user"
    assert node.sentiment_score == 0.5
    
    # Verify file exists
    node_file = graph._path("u-1")
    assert node_file.exists()
    
    with open(node_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["name"] == "Bryan"
        assert data["node_id"] == "u-1"


def test_interaction_sentiment_modulation(temp_graph):
    """Verify that interactions modulate sentiment scores and append digests."""
    graph = temp_graph
    node = graph.get_or_create_node("u-1", name="Bryan")
    
    # Positive interaction
    graph.record_interaction("u-1", sentiment_delta=0.1, digest="Bryan praised the new actuators.")
    assert graph.nodes["u-1"].sentiment_score == pytest.approx(0.6)
    assert len(graph.nodes["u-1"].digests) == 1
    assert graph.nodes["u-1"].digests[0] == "Bryan praised the new actuators."
    
    # Negative interaction
    graph.record_interaction("u-1", sentiment_delta=-0.2, digest="Bryan encountered a deadlock.")
    assert graph.nodes["u-1"].sentiment_score == pytest.approx(0.4)
    assert len(graph.nodes["u-1"].digests) == 2


def test_boundaries_and_shared_projects(temp_graph):
    """Verify setting boundaries and linking projects to nodes."""
    graph = temp_graph
    node = graph.get_or_create_node("u-1", name="Bryan")
    
    # Set boundary
    graph.set_boundary_flag("u-1", "do_not_disturb_after_midnight", True)
    assert graph.nodes["u-1"].boundary_flags["do_not_disturb_after_midnight"] is True
    
    # Link project
    graph.link_project("u-1", "proj-aura3")
    assert "proj-aura3" in graph.nodes["u-1"].shared_projects
