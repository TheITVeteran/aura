import json
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest

from core.memory.episodic_memory import EpisodicMemory


def test_idempotent_episode_retry_returns_same_committed_row(tmp_path):
    memory = EpisodicMemory(db_path=str(tmp_path / "episodic-idempotent.db"))
    memory._approve_memory_write = lambda *args, **kwargs: (True, None)
    kwargs = {
        "context": "User asked: retain this",
        "action": "Generated response in session durable",
        "outcome": "I retained it.",
        "success": True,
        "source": "chat_turn_logger",
        "idempotency_key": "session:exchange:r1",
        "metadata": {
            "memory_log_operation_id": "session:exchange:r1",
            "conversation_revision": 2,
            "origin": "desktop_ui",
        },
    }

    first = memory.record_episode(**kwargs)
    second = memory.record_episode(**kwargs)

    assert first == second
    assert first.startswith("idem-")
    with memory._get_conn() as con:
        row = con.execute(
            "SELECT COUNT(*) AS count, source, source_ref, source_revision, "
            "source_metadata FROM episodes WHERE episode_id = ?",
            (first,),
        ).fetchone()
    assert row["count"] == 1
    assert row["source"] == "chat_turn_logger"
    assert row["source_ref"] == "session:exchange:r1"
    assert row["source_revision"] == 2
    assert json.loads(row["source_metadata"])["origin"] == "desktop_ui"


def test_idempotent_episode_insert_is_atomic_across_memory_instances(tmp_path):
    db_path = str(tmp_path / "episodic-concurrent.db")
    first_memory = EpisodicMemory(db_path=db_path)
    second_memory = EpisodicMemory(db_path=db_path)
    barrier = Barrier(2)

    def approve(*_args, **_kwargs):
        barrier.wait(timeout=5.0)
        return True, None

    first_memory._approve_memory_write = approve
    second_memory._approve_memory_write = approve
    kwargs = {
        "context": "User asked: retain one copy",
        "action": "Generated response in session concurrent",
        "outcome": "Only one durable row may exist.",
        "success": True,
        "idempotency_key": "session:concurrent:r1",
    }

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda memory: memory.record_episode(**kwargs),
                [first_memory, second_memory],
            )
        )

    assert results[0] == results[1]
    with first_memory._get_conn() as con:
        assert con.execute(
            "SELECT COUNT(*) FROM episodes WHERE episode_id = ?",
            (results[0],),
        ).fetchone()[0] == 1


def _record_conversation_revision(
    memory: EpisodicMemory,
    *,
    revision: int,
    response: str,
) -> str:
    operation_id = f"session-1:exchange-1:r{revision}"
    return memory.record_episode(
        context="User asked: revise the explanation",
        action="Generated response in session session-1",
        outcome=response,
        success=True,
        source="chat_turn_logger",
        metadata={
            "session_id": "session-1",
            "conversation_exchange_id": "exchange-1",
            "conversation_revision": revision,
            "memory_log_operation_id": operation_id,
        },
        idempotency_key=operation_id,
    )


@pytest.mark.parametrize("arrival_order", [(1, 2), (2, 1)])
def test_only_latest_conversation_revision_is_cognitively_authoritative(
    tmp_path,
    arrival_order,
):
    memory = EpisodicMemory(db_path=str(tmp_path / "episodic-revisions.db"))
    memory._approve_memory_write = lambda *args, **kwargs: (True, None)
    responses = {
        1: "The obsolete answer says the core is cold.",
        2: "The corrected answer explains why Earth's core is hot.",
    }

    episode_ids = {
        revision: _record_conversation_revision(
            memory,
            revision=revision,
            response=responses[revision],
        )
        for revision in arrival_order
    }

    with memory._get_conn() as con:
        rows = con.execute(
            "SELECT episode_id, source_revision, source_authoritative, superseded_by "
            "FROM episodes ORDER BY source_revision"
        ).fetchall()
    assert [(row["source_revision"], row["source_authoritative"]) for row in rows] == [
        (1, 0),
        (2, 1),
    ]
    assert rows[0]["superseded_by"] == episode_ids[2]
    assert rows[1]["superseded_by"] == ""
    assert [episode.outcome for episode in memory.recall_recent(limit=10)] == [
        responses[2]
    ]
    assert memory._keyword_search("obsolete cold", limit=5) == []
    assert [episode.outcome for episode in memory._keyword_search("corrected hot", limit=5)] == [
        responses[2]
    ]
    assert memory._fetch_by_ids([episode_ids[1]]) == []


def test_consolidation_never_rehearses_superseded_conversation_revision(tmp_path):
    memory = EpisodicMemory(db_path=str(tmp_path / "episodic-consolidation.db"))
    memory._approve_memory_write = lambda *args, **kwargs: (True, None)
    old_id = _record_conversation_revision(memory, revision=1, response="Old answer")
    new_id = _record_conversation_revision(memory, revision=2, response="Corrected answer")
    with memory._get_conn() as con:
        con.execute(
            "UPDATE episodes SET next_decay_eval=0, access_count=5, decay_rate=0.01 "
            "WHERE episode_id IN (?, ?)",
            (old_id, new_id),
        )
        con.commit()

    result = memory._consolidate_sync()

    with memory._get_conn() as con:
        rows = {
            row["episode_id"]: row
            for row in con.execute(
                "SELECT episode_id, next_decay_eval, decay_rate FROM episodes "
                "WHERE episode_id IN (?, ?)",
                (old_id, new_id),
            ).fetchall()
        }
    assert result["boosted"] == 1
    assert rows[old_id]["next_decay_eval"] == 0
    assert rows[old_id]["decay_rate"] == 0.01
    assert rows[new_id]["next_decay_eval"] > 0
    assert rows[new_id]["decay_rate"] < 0.01


def test_cp359_rows_are_migrated_to_one_authoritative_revision(tmp_path):
    db_path = str(tmp_path / "episodic-migration.db")
    memory = EpisodicMemory(db_path=db_path)
    memory._approve_memory_write = lambda *args, **kwargs: (True, None)
    old_id = _record_conversation_revision(memory, revision=1, response="Old answer")
    new_id = _record_conversation_revision(memory, revision=2, response="New answer")
    with memory._get_conn() as con:
        for episode_id in (old_id, new_id):
            row = con.execute(
                "SELECT source_metadata FROM episodes WHERE episode_id=?",
                (episode_id,),
            ).fetchone()
            metadata = json.loads(row["source_metadata"])
            metadata.pop("conversation_exchange_id", None)
            con.execute(
                "UPDATE episodes SET source_session_id='', source_exchange_id='', "
                "source_authoritative=1, superseded_at=0.0, superseded_by='', "
                "source_metadata=? WHERE episode_id=?",
                (json.dumps(metadata), episode_id),
            )
        con.commit()

    migrated = EpisodicMemory(db_path=db_path)

    with migrated._get_conn() as con:
        rows = con.execute(
            "SELECT episode_id, source_exchange_id, source_authoritative, "
            "superseded_by FROM episodes ORDER BY source_revision"
        ).fetchall()
    assert [row["source_exchange_id"] for row in rows] == ["exchange-1", "exchange-1"]
    assert [row["source_authoritative"] for row in rows] == [0, 1]
    assert rows[0]["superseded_by"] == new_id
    assert migrated.get_summary()["total_episodes"] == 1
    assert migrated.get_summary()["stored_episode_rows"] == 2
    assert migrated.get_summary()["superseded_conversation_revisions"] == 1


def test_idempotent_episode_rejects_payload_collision(tmp_path):
    memory = EpisodicMemory(db_path=str(tmp_path / "episodic-collision.db"))
    memory._approve_memory_write = lambda *args, **kwargs: (True, None)
    memory.record_episode(
        context="first",
        action="answer",
        outcome="one",
        success=True,
        idempotency_key="same-operation",
    )

    with pytest.raises(ValueError, match="idempotency identity collision"):
        memory.record_episode(
            context="second",
            action="answer",
            outcome="two",
            success=True,
            idempotency_key="same-operation",
        )


def test_cooldown_does_not_report_an_unpersisted_episode(tmp_path):
    memory = EpisodicMemory(db_path=str(tmp_path / "episodic-cooldown.db"))
    memory._approve_memory_write = lambda *args, **kwargs: (True, None)
    memory._RECORD_COOLDOWN = 60.0

    first = memory.record_episode("one", "answer", "stored", True)
    second = memory.record_episode("two", "answer", "not stored", True)

    assert first
    assert second == ""
    with memory._get_conn() as con:
        assert con.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 1


def test_recall_similar_skips_keyword_fallback_when_vector_results_suffice(tmp_path):
    class _VectorMemory:
        def search_similar(self, query, k, filter_metadata=None):
            return [
                {"metadata": {"episode_id": "ep-a"}},
                {"metadata": {"episode_id": "ep-b"}},
            ]

    memory = EpisodicMemory(db_path=str(tmp_path / "episodic.db"), vector_memory=_VectorMemory())
    episodes = [
        SimpleNamespace(episode_id="ep-a", importance=0.7, timestamp=time.time()),
        SimpleNamespace(episode_id="ep-b", importance=0.6, timestamp=time.time() - 1),
    ]

    memory._fetch_by_ids = lambda episode_ids: [ep for ep in episodes if ep.episode_id in episode_ids]
    memory._keyword_search = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("keyword fallback should be skipped"))
    observed = {}
    memory._observe_ranked_recall = lambda ranked, *, returned_count: observed.update(
        candidates=len(ranked), returned_count=returned_count
    )

    result = memory.recall_similar("summarize our recent continuity work", limit=2)

    assert [ep.episode_id for ep in result] == ["ep-a", "ep-b"]
    assert observed == {"candidates": 2, "returned_count": 2}


def test_recall_similar_keeps_keyword_fallback_for_exact_recall_queries(tmp_path):
    class _VectorMemory:
        def search_similar(self, query, k, filter_metadata=None):
            return [
                {"metadata": {"episode_id": "ep-a"}},
                {"metadata": {"episode_id": "ep-b"}},
            ]

    memory = EpisodicMemory(db_path=str(tmp_path / "episodic.db"), vector_memory=_VectorMemory())
    called = {"keyword": 0}
    episodes = [
        SimpleNamespace(episode_id="ep-a", importance=0.7, timestamp=time.time()),
        SimpleNamespace(episode_id="ep-b", importance=0.6, timestamp=time.time() - 1),
    ]

    memory._fetch_by_ids = lambda episode_ids: [ep for ep in episodes if ep.episode_id in episode_ids]

    def _keyword_search(*_args, **_kwargs):
        called["keyword"] += 1
        return []

    memory._keyword_search = _keyword_search

    memory.recall_similar('What did I tell you? Give me the exact words.', limit=2)

    assert called["keyword"] == 1
