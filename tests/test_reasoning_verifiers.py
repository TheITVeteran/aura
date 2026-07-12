"""Tests for the domain truth-engine verifiers (core/brain/verifiers)."""
from __future__ import annotations

import pytest

from core.brain.verifiers import get_verifier_registry, verify_candidate
from core.brain.verifiers.base import VerificationResult, combine_results
from core.brain.verifiers.code_engine import CodeTruthEngine, extract_code_blocks
from core.brain.verifiers.math_engine import MathTruthEngine
from core.brain.verifiers.planning_engine import PlanningEngine
from core.brain.verifiers.repo_engine import RepoEvidenceEngine


@pytest.mark.asyncio
async def test_code_engine_passes_clean_code():
    code = "```python\ndef add(a, b):\n    return a + b\n```"
    res = await CodeTruthEngine(run_ruff=False).verify(code)
    assert res.checked and res.ok
    assert res.detail["compiled_ok"] == 1


@pytest.mark.asyncio
async def test_code_engine_fails_syntax_error():
    code = "```python\ndef broken(:\n    return\n```"
    res = await CodeTruthEngine(run_ruff=False).verify(code)
    assert res.checked and not res.ok
    assert any("syntax" in i for i in res.issues)


@pytest.mark.asyncio
async def test_code_engine_noop_when_no_code():
    res = await CodeTruthEngine(run_ruff=False).verify("just prose, no code here at all")
    assert res.ok and not res.checked


def test_extract_code_blocks_fenced_and_inline():
    assert extract_code_blocks("```py\nimport os\n```") == ["import os"]
    assert extract_code_blocks("def f():\n    return 1") == ["def f():\n    return 1"]
    assert extract_code_blocks("hello world") == []


@pytest.mark.asyncio
async def test_math_engine_catches_arithmetic_error():
    res = await MathTruthEngine().verify("The total is 2 + 2 = 5, therefore done.")
    assert res.checked and not res.ok
    assert any("arithmetic" in i for i in res.issues)


@pytest.mark.asyncio
async def test_math_engine_accepts_correct_arithmetic():
    res = await MathTruthEngine().verify("Since 12 * 12 = 144 we are fine.")
    assert res.checked and res.ok


@pytest.mark.asyncio
async def test_math_engine_verify_expression_target():
    res = await MathTruthEngine().verify(
        "The answer is 42.", context={"verify_expression": "6 * 7"}
    )
    assert res.checked and res.ok


@pytest.mark.asyncio
async def test_repo_engine_flags_missing_file():
    res = await RepoEvidenceEngine().verify("This is handled in core/totally/madeup_file.py")
    assert res.checked and not res.ok
    assert any("not found" in i for i in res.issues)


@pytest.mark.asyncio
async def test_repo_engine_accepts_real_file():
    res = await RepoEvidenceEngine().verify("See core/brain/verifiers/base.py for the result type.")
    assert res.checked and res.ok


@pytest.mark.asyncio
async def test_planning_engine_requires_verification_step():
    plan = "1. Create the module\n2. Add the function\n3. Build the package"
    res = await PlanningEngine().verify(plan)
    assert res.checked
    assert any("verification" in i for i in res.issues)


@pytest.mark.asyncio
async def test_planning_engine_good_plan():
    plan = "1. Inspect the file\n2. Edit the function\n3. Run the tests to verify it passes"
    res = await PlanningEngine().verify(plan)
    assert res.checked and res.ok


def test_combine_results_hard_gate():
    good = VerificationResult(domain="x", ok=True, checked=True, score=0.9)
    bad = VerificationResult(domain="x", ok=False, checked=True, score=0.2)
    noop = VerificationResult(domain="x", ok=True, checked=False, score=0.5)
    assert combine_results("x", [good, noop]).ok
    assert not combine_results("x", [good, bad]).ok
    # only no-op checks → neutral, ok
    assert combine_results("x", [noop]).ok and not combine_results("x", [noop]).checked


@pytest.mark.asyncio
async def test_registry_dispatch_by_task_type():
    reg = get_verifier_registry()
    # Math task with an error must fail through the registry.
    res = await reg.verify("3 + 3 = 7", task_type="math")
    assert res.checked and not res.ok
    # Code task with clean code passes.
    res2 = await verify_candidate("```python\nx = 1\n```", task_type="code")
    assert res2.ok


@pytest.mark.asyncio
async def test_registry_always_runs_logic():
    reg = get_verifier_registry()
    verifiers = reg.select("generic")
    assert any(getattr(v, "name", "") == "logic" for v in verifiers)


# ── Citation engine: self-fetching evidence (July capability raise) ──────


class _FakeCorpusStore:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, limit=5):
        return self._hits[:limit]


class _FakeHit:
    def __init__(self, title, snippet):
        self.title = title
        self.snippet = snippet


@pytest.mark.asyncio
async def test_citation_engine_checks_caller_evidence():
    from core.brain.verifiers.citation_engine import CitationEngine

    result = await CitationEngine().verify(
        "The retry budget is unlimited and reboots forever.",
        context={"evidence": ["The retry budget is three attempts, then it fails closed."]},
    )
    assert result.checked
    assert not result.ok, "an absolute claim against a bounded fact must fail"


@pytest.mark.asyncio
async def test_citation_engine_self_fetches_when_caller_brings_nothing(monkeypatch):
    """The capability raise: no evidence pack → the engine pulls its own
    receipts from the local corpus and still catches contradictions."""
    from core.brain.verifiers import citation_engine
    from core.knowledge import local_corpus

    hits = [_FakeHit("retry policy", "The retry budget is three attempts, then it fails closed.")]
    monkeypatch.setattr(
        local_corpus, "get_local_corpus_store", lambda *a, **k: _FakeCorpusStore(hits)
    )
    result = await citation_engine.CitationEngine().verify(
        "The retry budget is unlimited and reboots forever.",
        context={"objective": "what is the retry budget policy"},
    )
    assert result.detail["self_fetched_evidence"] is True
    assert result.checked
    assert not result.ok, "self-fetched contradiction is a hard fail"


@pytest.mark.asyncio
async def test_self_fetched_absence_of_mention_is_not_wrongness(monkeypatch):
    """Partial-corpus semantics: a true claim the corpus never ingested
    must NOT fail — only contradictions do."""
    from core.brain.verifiers import citation_engine
    from core.knowledge import local_corpus

    hits = [_FakeHit("retry policy", "The retry budget is three attempts before backing off.")]
    monkeypatch.setattr(
        local_corpus, "get_local_corpus_store", lambda *a, **k: _FakeCorpusStore(hits)
    )
    result = await citation_engine.CitationEngine().verify(
        "The retry budget is three attempts. Jupiter is the largest planet.",
        context={"objective": "retry budget"},
    )
    assert result.ok, "unmentioned-but-unrelated claims are advisories, not failures"
    assert any("unconfirmed by local corpus" in issue for issue in result.issues)


@pytest.mark.asyncio
async def test_no_evidence_anywhere_stays_advisory(monkeypatch):
    from core.brain.verifiers import citation_engine
    from core.knowledge import local_corpus

    monkeypatch.setattr(
        local_corpus, "get_local_corpus_store", lambda *a, **k: _FakeCorpusStore([])
    )
    result = await citation_engine.CitationEngine().verify(
        "The retry budget is unlimited.",
        context={"objective": "retry budget"},
    )
    assert result.ok and not result.checked, "nothing to check against → advise only"
