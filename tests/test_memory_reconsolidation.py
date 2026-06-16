"""Tests for engram dynamics: reconsolidation, hippocampal pattern completion,
novelty, fidelity-vs-vividness, governance, and therapeutic reconsolidation.

These exercise the model from the "memory is rewritten on recall" view: a recalled
memory becomes labile and drifts toward the present context, gated by neuromodulatory
plasticity and constitutional governance; repeated recall makes a memory more vivid
yet less faithful.
"""
import sqlite3
import time

from core.memory.episodic_memory import EpisodicMemory
from core.memory.hippocampus import HippocampalIndex
from core.memory.reconsolidation import ReconsolidationEngine


# ---------------------------------------------------------------------------
# Pure reconsolidation engine
# ---------------------------------------------------------------------------

def _base_kwargs(**over):
    kw = dict(
        now=10_000.0,
        timestamp=0.0,                 # very old → past the consolidation window
        emotional_valence=0.0,
        original_valence=0.0,
        importance=0.4,
        decay_rate=0.01,
        fidelity=1.0,
        reconsolidation_count=0,
        last_reconsolidated=0.0,
        current_strength=0.4,
        qualia_snapshot={"q_norm": 0.2, "dominant_dim": "calm"},
        current_qualia={"valence": 0.8, "q_norm": 0.7, "dominant_dim": "joy"},
        lability=1.0,
    )
    kw.update(over)
    return kw


def test_refractory_window_blocks_recent_recall():
    eng = ReconsolidationEngine()
    out = eng.reconsolidate(**_base_kwargs(last_reconsolidated=9_999.0))  # < cooldown ago
    assert out.fired is False
    assert out.drifted is False
    assert out.emotional_valence == 0.0
    assert out.fidelity == 1.0
    assert out.reconsolidation_count == 0


def test_recall_drifts_toward_present_and_loses_fidelity():
    eng = ReconsolidationEngine()
    out = eng.reconsolidate(**_base_kwargs())
    assert out.fired is True
    assert out.drifted is True
    # Valence moved toward the present positive mood (0.8), but only partway.
    assert 0.0 < out.emotional_valence <= ReconsolidationEngine().max_valence_drift + 1e-9
    # Fidelity to the original dropped; the trace was rewritten.
    assert out.fidelity < 1.0
    assert out.reconsolidation_count == 1
    assert out.prediction_error > 0.0


def test_strong_emotional_memory_resists_change():
    eng = ReconsolidationEngine()
    mild = eng.reconsolidate(**_base_kwargs(emotional_valence=0.0, original_valence=0.0, current_strength=0.3))
    strong = eng.reconsolidate(**_base_kwargs(
        emotional_valence=-0.9, original_valence=-0.9, current_strength=0.95,
    ))
    mild_shift = abs(mild.emotional_valence - 0.0)
    strong_shift = abs(strong.emotional_valence - (-0.9))
    # Boundary condition: the strong, vivid, emotional memory barely moves.
    assert strong_shift < mild_shift
    assert (1.0 - strong.fidelity) < (1.0 - mild.fidelity)


def test_neuromodulatory_lability_scales_drift():
    eng = ReconsolidationEngine()
    low = eng.reconsolidate(**_base_kwargs(lability=0.3))
    high = eng.reconsolidate(**_base_kwargs(lability=2.0))
    assert abs(high.emotional_valence) > abs(low.emotional_valence)


def test_fresh_memory_rehearses_but_does_not_rewrite():
    eng = ReconsolidationEngine()
    # Age below the consolidation window: recall strengthens but content holds.
    out = eng.reconsolidate(**_base_kwargs(timestamp=9_500.0))  # only 500s old
    assert out.fired is True
    assert out.drifted is False
    assert out.emotional_valence == 0.0
    assert out.fidelity == 1.0
    assert out.importance >= 0.4  # rehearsal nudged importance up (or held)


def test_therapeutic_reconsolidation_moves_toward_target():
    eng = ReconsolidationEngine()
    out = eng.reconsolidate_in_context(
        now=10_000.0,
        emotional_valence=-0.6,
        qualia_snapshot={"q_norm": 0.3},
        importance=0.7,
        fidelity=1.0,
        reconsolidation_count=1,
        target_valence=0.4,
        intensity=0.5,
        safe_context={"valence": 0.5, "q_norm": 0.6},
    )
    assert out.emotional_valence > -0.6          # softened toward target
    assert out.fidelity < 1.0                    # a reframed memory is still rewritten
    assert out.reconsolidation_count == 2


# ---------------------------------------------------------------------------
# Hippocampal index / pattern completion
# ---------------------------------------------------------------------------

def test_extract_cues_filters_noise_and_types_cues():
    cues = HippocampalIndex.extract_cues(
        "The user asked about the squirrel and the crow",
        "searched memory",
        "found it",
        tools=["web_search"],
        qualia_snapshot={"dominant_dim": "curiosity"},
    )
    assert "squirrel" in cues and "crow" in cues
    assert "the" not in cues and "and" not in cues  # stopwords dropped
    assert "tool:web_search" in cues
    assert "dim:curiosity" in cues


def test_pattern_completion_ranks_by_overlap(tmp_path):
    db = str(tmp_path / "ep.db")

    def factory():
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        return conn

    idx = HippocampalIndex(factory)
    idx.bind("ep-crow", ["squirrel", "crow", "nut", "mouse"])
    idx.bind("ep-coffee", ["coffee", "monday", "tired"])
    matches = idx.pattern_complete(["crow", "squirrel"], limit=5)
    assert matches
    assert matches[0][0] == "ep-crow"
    assert all(eid != "ep-coffee" for eid, _ in matches)


# ---------------------------------------------------------------------------
# Full integration through EpisodicMemory
# ---------------------------------------------------------------------------

def _make_memory(tmp_path) -> EpisodicMemory:
    mem = EpisodicMemory(db_path=str(tmp_path / "episodic.db"))
    mem._RECORD_COOLDOWN = 0.0
    # Deterministic governance in tests: approve writes and reconsolidations.
    mem._approve_memory_write = lambda *a, **k: (True, None)
    mem._approve_reconsolidation = lambda ep, outcome: True
    return mem


def _age_episode(mem: EpisodicMemory, episode_id: str, *, age_seconds: float):
    """Push an episode's timestamp into the past and clear its labile cooldown."""
    with mem._get_conn() as conn:
        conn.execute(
            "UPDATE episodes SET timestamp = ?, last_reconsolidated = 0 WHERE episode_id = ?",
            (time.time() - age_seconds, episode_id),
        )
        conn.commit()


def test_encode_sets_engram_fields_and_binds_cues(tmp_path):
    mem = _make_memory(tmp_path)
    eid = mem.record_episode(
        context="a crow and a squirrel fought over a nut",
        action="watched",
        outcome="a mouse stole it",
        success=True,
        emotional_valence=0.3,
    )
    assert eid
    with mem._get_conn() as conn:
        row = conn.execute("SELECT * FROM episodes WHERE episode_id = ?", (eid,)).fetchone()
    assert row["fidelity"] == 1.0
    assert row["original_valence"] == 0.3
    assert row["reconsolidation_count"] == 0
    assert 0.0 <= row["novelty"] <= 1.0
    assert mem._hippocampus.stats()["indexed_engrams"] >= 1


def test_recall_reconsolidates_and_drifts(tmp_path):
    mem = _make_memory(tmp_path)
    eid = mem.record_episode(
        context="mildly interesting podcast on the bus",
        action="listened",
        outcome="arrived at work",
        success=True,
        emotional_valence=0.0,
    )
    _age_episode(mem, eid, age_seconds=3 * 86400)
    # Force a positive present context and high neuromodulatory lability.
    mem._current_qualia = lambda: {"valence": 0.9, "q_norm": 0.8, "dominant_dim": "joy"}
    mem._plasticity_gain = lambda: 1.6

    recalled = mem.recall_recent(limit=5)
    ep = next(e for e in recalled if e.episode_id == eid)
    assert ep.emotional_valence > 0.0          # tone drifted toward the present
    assert ep.current_fidelity() < 1.0         # and the trace was rewritten
    assert ep.reconsolidation_count >= 1

    # Persisted, not just on the in-memory object.
    with mem._get_conn() as conn:
        row = conn.execute("SELECT * FROM episodes WHERE episode_id = ?", (eid,)).fetchone()
    assert row["emotional_valence"] > 0.0
    assert row["fidelity"] < 1.0


def test_vividness_rises_as_accuracy_falls(tmp_path):
    mem = _make_memory(tmp_path)
    eid = mem.record_episode(
        context="the epic squirrel crow fight",
        action="retold the story",
        outcome="everyone laughed",
        success=True,
        emotional_valence=0.1,
    )
    mem._current_qualia = lambda: {"valence": 0.95, "q_norm": 0.85, "dominant_dim": "joy"}
    mem._plasticity_gain = lambda: 1.5

    fidelities = []
    for _ in range(5):
        _age_episode(mem, eid, age_seconds=3 * 86400)  # also clears cooldown
        mem.recall_recent(limit=5)
        with mem._get_conn() as conn:
            row = conn.execute(
                "SELECT fidelity, access_count FROM episodes WHERE episode_id = ?", (eid,)
            ).fetchone()
        fidelities.append(row["fidelity"])

    assert row["access_count"] >= 5             # vividness/rehearsal kept rising
    assert fidelities[-1] < fidelities[0]       # accuracy kept falling
    assert fidelities == sorted(fidelities, reverse=True)


def test_governance_veto_blocks_content_rewrite(tmp_path):
    mem = _make_memory(tmp_path)
    mem._approve_reconsolidation = lambda ep, outcome: False  # constitution vetoes drift
    eid = mem.record_episode(
        context="a tense argument",
        action="discussed",
        outcome="unresolved",
        success=False,
        emotional_valence=-0.2,
    )
    _age_episode(mem, eid, age_seconds=3 * 86400)
    mem._current_qualia = lambda: {"valence": 0.9, "q_norm": 0.8, "dominant_dim": "joy"}
    mem._plasticity_gain = lambda: 1.6

    mem.recall_recent(limit=5)
    with mem._get_conn() as conn:
        row = conn.execute("SELECT * FROM episodes WHERE episode_id = ?", (eid,)).fetchone()
    # Content held (governance vetoed the rewrite) but rehearsal still happened.
    assert row["emotional_valence"] == -0.2
    assert row["fidelity"] == 1.0
    assert row["reconsolidation_count"] == 0
    assert row["access_count"] >= 1


def test_pattern_complete_recall_path(tmp_path):
    mem = _make_memory(tmp_path)
    eid = mem.record_episode(
        context="debugging the quantum flux capacitor calibration",
        action="patched the resonance coil",
        outcome="stabilised",
        success=True,
    )
    hits = mem.pattern_complete("flux capacitor calibration resonance", limit=5)
    assert any(e.episode_id == eid for e in hits)


def test_therapeutic_reconsolidation_softens_memory(tmp_path):
    mem = _make_memory(tmp_path)
    eid = mem.record_episode(
        context="a humiliating mistake in front of everyone",
        action="froze",
        outcome="felt awful",
        success=False,
        emotional_valence=-0.8,
    )
    ok = mem.reconsolidate_memory_in_context(eid, target_valence=0.2, intensity=0.5)
    assert ok is True
    with mem._get_conn() as conn:
        row = conn.execute("SELECT * FROM episodes WHERE episode_id = ?", (eid,)).fetchone()
    assert row["emotional_valence"] > -0.8     # softened toward safety
    assert row["fidelity"] < 1.0               # reframing is still a rewrite
    assert row["reconsolidation_count"] >= 1


def test_summary_reports_engram_dynamics(tmp_path):
    mem = _make_memory(tmp_path)
    mem.record_episode(context="something novel happened", action="noted", outcome="ok", success=True)
    summary = mem.get_summary()
    for key in ("avg_fidelity", "reshaped_memories", "total_reconsolidations", "avg_novelty", "indexed_engrams"):
        assert key in summary
    assert summary["avg_fidelity"] <= 1.0
