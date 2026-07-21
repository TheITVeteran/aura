"""The organs-in-loop reasoning test (CP236).

The RLC Context requirement: prove reasoning, not recall. Tasks whose
answer is absent from the prompt and base memory, whose facts live only in
retrieval, that must be combined -- and where BOTH ablations bite:
disabling retrieval breaks it (knowledge was external) and disabling
recurrence breaks it (depth did the combining).
"""
from __future__ import annotations

import pytest

from core.learning.integrated_reasoning_eval import (
    REASONING_CRITERIA,
    RETRIEVAL_OFF,
    RETRIEVAL_ON,
    Fact,
    FixtureRetrieval,
    KnowledgeTask,
    assert_base_recall_guard,
    build_knowledge_tasks,
    run_factorial,
)


# ── The tasks cannot be answered by reading the prompt ──────────────────


def test_a_task_whose_answer_is_in_the_prompt_is_refused():
    """That measures reading comprehension, not reasoning or retrieval."""
    with pytest.raises(ValueError, match="answer appears in the prompt"):
        KnowledgeTask(
            task_id="bad", prompt="The value is 42. What is the value?",
            answer="42", facts=(), hops=1,
            criteria=("derive_unstored_answer",),
        )


def test_generated_chains_hide_the_answer_and_the_edges():
    tasks = build_knowledge_tasks(
        families=["transitive_chain"], hops=[2, 4], per_cell=3, seed=1
    )
    assert len(tasks) == 2 * 3
    for task in tasks:
        assert task.answer.lower() not in task.prompt.lower()
        # Every edge needed to reach the answer lives in retrieval, not text.
        assert len(task.facts) == task.hops
        for fact in task.facts:
            assert fact.obj not in task.prompt or fact.subject not in task.prompt


# ── A solver that only reads the prompt fails; one that retrieves and
#    reasons succeeds -- and the factorial shows both are necessary ──────


def _chain_solver(hops_needed_visible=True):
    """A controllable solver.

    It can follow a chain only as far as (retrieved edges) AND (depth). With
    empty context it has no edges; at depth 1 it can take only one hop. This
    is the behaviour the real recurrence is hypothesised to have, made
    deterministic so the harness itself can be tested.
    """
    def solve(prompt: str, context: list[str], depth: int) -> str:
        edges = {}
        for passage in context:
            parts = passage.replace(".", "").split(" points to ")
            if len(parts) == 2:
                edges[parts[0].strip()] = parts[1].strip()
        # start node is named in the prompt
        import re
        start = re.search(r"starting at (\S+)", prompt)
        if not start:
            return ""
        node = start.group(1)
        for _ in range(depth):  # one hop per unit of depth
            node = edges.get(node, node)
        return f"FINAL_ANSWER: {node}"
    return solve


def test_the_factorial_shows_both_retrieval_and_depth_are_causal():
    tasks = build_knowledge_tasks(
        families=["transitive_chain"], hops=[4], per_cell=6, seed=7
    )
    source = FixtureRetrieval()
    for task in tasks:
        source.plant(task)
    report = run_factorial(
        tasks, source, _chain_solver(), depths=(1, 2, 4), retrieval_limit=8
    )
    acc = report["accuracy"]
    # retrieval off => no edges => cannot solve at any depth
    assert acc[RETRIEVAL_OFF][4] == 0.0
    # retrieval on but depth 1 => only one hop of a four-hop chain
    assert acc[RETRIEVAL_ON][1] == 0.0
    # retrieval on AND depth 4 => solved
    assert acc[RETRIEVAL_ON][4] == 1.0

    verdicts = report["verdicts"]
    assert verdicts["retrieval_is_causal"] is True
    assert verdicts["recurrence_helps"] is True
    assert verdicts["both_required"] is True
    assert "could not produce alone" in verdicts["claim"]


def test_both_required_is_false_when_depth_one_already_solves():
    """If retrieval alone (no depth) works, the task did not need
    recurrence and must not be counted as proof of it."""
    tasks = build_knowledge_tasks(
        families=["transitive_chain"], hops=[1], per_cell=6, seed=3
    )
    source = FixtureRetrieval()
    for task in tasks:
        source.plant(task)
    report = run_factorial(tasks, source, _chain_solver(), depths=(1, 2, 4))
    # one-hop chains are solved at depth 1
    assert report["accuracy"][RETRIEVAL_ON][1] == 1.0
    assert report["verdicts"]["both_required"] is False


# ── The base-recall guard: disqualify anything answerable from memory ───


def test_base_recall_guard_disqualifies_memorized_answers():
    tasks = build_knowledge_tasks(
        families=["transitive_chain"], hops=[3], per_cell=4, seed=9
    )

    def cheating_solver(prompt, context, depth):
        # Pretends to "know" every answer without retrieval.
        import re
        # extract the gold from nowhere -- simulate a leaked/memorized answer
        return "FINAL_ANSWER: " + prompt  # never contains the hidden node

    guard = assert_base_recall_guard(tasks, cheating_solver, depth=4)
    # The gold nodes are random and absent from the prompt, so even this
    # "cheater" cannot recall them -- the guard passes honestly.
    assert guard["guard_passed"] is True

    def truly_leaking_solver(prompt, context, depth):
        # Answers correctly with NO retrieval => memorized => must be caught.
        for task in tasks:
            if task.task_id in prompt or task.prompt == prompt:
                return f"FINAL_ANSWER: {task.answer}"
        return ""

    leaked = assert_base_recall_guard(tasks, truly_leaking_solver, depth=4)
    assert leaked["guard_passed"] is False
    assert len(leaked["answered_from_memory"]) == len(tasks)


# ── Conflicting sources exercise evidence discrimination ────────────────


def test_conflicting_sources_reward_the_authoritative_fact():
    tasks = build_knowledge_tasks(
        families=["conflicting_sources"], hops=[2], per_cell=4, seed=5
    )
    source = FixtureRetrieval()
    for task in tasks:
        source.plant(task)

    def authority_solver(prompt, context, depth):
        # Trusts the FIRST retrieved passage (fixture ranks by authority).
        if not context:
            return ""
        first = context[0]
        # "current value is X" -> X
        if "current value is" in first:
            return "FINAL_ANSWER: " + first.split("current value is")[1].split(".")[0].strip()
        return ""

    report = run_factorial(tasks, source, authority_solver, depths=(1, 2, 4))
    assert report["accuracy"][RETRIEVAL_ON][2] == 1.0
    assert report["accuracy"][RETRIEVAL_OFF][2] == 0.0
    coverage = report["criteria_coverage"]
    assert coverage["distinguish_conflicting_evidence"] == len(tasks)


# ── Coverage of the nine reasoning marks is reported ────────────────────


def test_criteria_coverage_is_reported():
    tasks = build_knowledge_tasks(
        families=["transitive_chain", "conflicting_sources"],
        hops=[2], per_cell=2, seed=1,
    )
    source = FixtureRetrieval()
    for task in tasks:
        source.plant(task)
    report = run_factorial(tasks, source, _chain_solver(), depths=(1, 2))
    coverage = report["criteria_coverage"]
    assert set(coverage) == set(REASONING_CRITERIA)
    assert coverage["combine_unpresented_facts"] > 0
    assert coverage["distinguish_conflicting_evidence"] > 0


# ── Fail closed ─────────────────────────────────────────────────────────


def test_invalid_inputs_are_refused():
    with pytest.raises(ValueError, match="unknown families"):
        build_knowledge_tasks(families=["nope"], hops=[1], per_cell=1, seed=1)
    with pytest.raises(ValueError, match="no tasks"):
        run_factorial([], FixtureRetrieval(), lambda p, c, d: "")
    with pytest.raises(ValueError, match="unknown reasoning criteria"):
        KnowledgeTask(
            task_id="x", prompt="q?", answer="z", facts=(), hops=1,
            criteria=("not_a_criterion",),
        )
