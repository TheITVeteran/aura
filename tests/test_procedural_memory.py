"""Procedural memory: playbooks compound at inference; weights only when the
foundry admits the domain; transfer must be demonstrated, not assumed."""
from __future__ import annotations

import pytest

from core.brain.procedural_memory import ProceduralMemory
from core.brain.verifiers.foundry import VerifierFoundry

pytestmark = pytest.mark.unit


@pytest.fixture()
def memory(tmp_path):
    return ProceduralMemory(path=tmp_path / "playbooks.json")


def _win(memory, *, objective="implement a binary search over a sorted list",
         task_type="code", strategy="deep/self_consistency", **kw):
    return memory.record_win(
        objective=objective, task_type=task_type,
        answer="def bsearch(xs, t):\n    lo, hi = 0, len(xs)",
        strategy=strategy, verifiers=["code", "logic"],
        confidence=kw.pop("confidence", 0.85), **kw,
    )


def test_win_distills_into_a_playbook(memory):
    pid = _win(memory)
    assert pid
    status = memory.status()
    assert status["playbooks"] == 1
    assert status["by_task_type"] == {"code": 1}


def test_same_family_reinforces_instead_of_duplicating(memory):
    a = _win(memory)
    b = _win(memory, objective="implement binary search over sorted lists")
    assert a == b
    assert memory.status()["playbooks"] == 1


def test_recall_matches_by_shape_and_injects_text(memory):
    _win(memory)
    _win(memory, objective="parse a csv file into rows", task_type="code",
         strategy="fast/direct")
    text = memory.as_playbook_text("binary search in a sorted array",
                                   task_type="code")
    assert "Proven approaches" in text
    assert "self_consistency" in text
    assert "csv" not in text                     # shape-matched, not dumped


def test_recall_respects_task_type(memory):
    _win(memory)
    assert memory.recall("binary search a sorted list", task_type="math") == []


def test_transfer_credit_requires_retrieval_then_win(memory):
    pid = _win(memory)
    # a similar problem retrieves the playbook…
    books = memory.recall("binary search over a sorted array",
                          task_type="code", problem_key="p2")
    assert [b.playbook_id for b in books] == [pid]
    # …and only when THAT problem wins does transfer credit land
    memory.record_win(objective="binary search over a sorted array",
                      task_type="code", answer="def f(): ...",
                      strategy="deep/self_consistency", verifiers=["code"],
                      confidence=0.9, problem_key="p2")
    book = memory.recall("binary search sorted", task_type="code")[0]
    assert book.reuse_wins == 1


def test_distillation_requires_transfer_and_admission(memory, tmp_path, monkeypatch):
    foundry = VerifierFoundry(root=tmp_path / "foundry")
    try:
        monkeypatch.setattr(
            "core.runtime.service_access.optional_service",
            lambda name, default=None: foundry if name == "verifier_foundry"
            else default,
        )
        pid = _win(memory)
        # no transfer yet → nothing exports even in an admitted domain (code=seed)
        assert memory.export_distillation_batch(min_reuse_wins=1) == []

        memory.recall("binary search over a sorted array", task_type="code",
                      problem_key="p")
        memory.record_win(objective="binary search over a sorted array",
                          task_type="code", answer="x", strategy="deep/sc",
                          verifiers=["code"], confidence=0.9, problem_key="p")
        batch = memory.export_distillation_batch(min_reuse_wins=1)
        assert len(batch) == 1
        assert "Strategy:" in batch[0]["completion"]
        # marked distilled — not exported twice
        assert memory.export_distillation_batch(min_reuse_wins=1) == []
        assert pid  # sanity
    finally:
        foundry.close()


def test_distillation_blocked_for_unadmitted_domain(memory, tmp_path, monkeypatch):
    foundry = VerifierFoundry(root=tmp_path / "foundry")
    try:
        monkeypatch.setattr(
            "core.runtime.service_access.optional_service",
            lambda name, default=None: foundry if name == "verifier_foundry"
            else default,
        )
        _win(memory, objective="write a moving essay about rivers",
             task_type="writing", strategy="deep/courtroom")
        memory.recall("write an essay about rivers", task_type="writing",
                      problem_key="w")
        memory.record_win(objective="write an essay about rivers",
                          task_type="writing", answer="x", strategy="deep/co",
                          verifiers=["rubric"], confidence=0.9, problem_key="w")
        # transfer exists, but "writing" has no earned admission → no export
        assert memory.export_distillation_batch(min_reuse_wins=1) == []
    finally:
        foundry.close()


def test_distillation_never_runs_without_a_foundry(memory, monkeypatch):
    monkeypatch.setattr(
        "core.runtime.service_access.optional_service",
        lambda name, default=None: default,
    )
    _win(memory)
    memory.recall("binary search sorted list", task_type="code",
                  problem_key="p")
    memory.record_win(objective="binary search sorted list", task_type="code",
                      answer="x", strategy="s", verifiers=["code"],
                      confidence=0.9, problem_key="p")
    # weight-time compounding is NEVER ungated
    assert memory.export_distillation_batch(min_reuse_wins=1) == []


def test_persistence_roundtrip(tmp_path):
    m1 = ProceduralMemory(path=tmp_path / "pb.json")
    m1.record_win(objective="solve quadratic equations", task_type="math",
                  answer="x = (-b ± √(b²-4ac)) / 2a", strategy="fast/direct",
                  verifiers=["math"], confidence=0.9)
    m1.flush()
    m2 = ProceduralMemory(path=tmp_path / "pb.json")
    assert m2.status()["playbooks"] == 1
    assert m2.recall("solve a quadratic equation", task_type="math")


def test_eviction_keeps_proven_transferrers(tmp_path, monkeypatch):
    monkeypatch.setattr("core.brain.procedural_memory._MAX_PLAYBOOKS", 5)
    m = ProceduralMemory(path=tmp_path / "pb.json")
    keeper = m.record_win(objective="alpha beta gamma keeper problem",
                          task_type="code", answer="x", strategy="s",
                          verifiers=["code"], confidence=0.9)
    m.recall("alpha beta gamma keeper problem", task_type="code",
             problem_key="k")
    m.record_win(objective="alpha beta gamma keeper problem", task_type="code",
                 answer="x", strategy="s", verifiers=["code"],
                 confidence=0.9, problem_key="k")
    for i in range(8):
        m.record_win(objective=f"unique filler problem number {i} zzz{i}",
                     task_type="code", answer="x", strategy="s",
                     verifiers=["code"], confidence=0.7)
    with m._lock:
        assert keeper in m._books                 # transfer-proven survives
        assert len(m._books) <= 5
