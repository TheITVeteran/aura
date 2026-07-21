"""The one seam every cognitive source plugs into (CP241).

Retrieval proved the workspace amplifies fed material (0->56%). This is the
discipline that lets imagination, world-model, and talk-through join WITHOUT
becoming the RLC's seven unproven mechanisms: one interface, tagged trust,
budgeted composition, and an ablation harness that makes each source prove
it helps.
"""
from __future__ import annotations

import pytest

from core.learning.workspace_producers import (
    GROUNDED,
    HYPOTHETICAL,
    ImaginationProducer,
    RetrievalProducer,
    WorkspaceComposer,
    WorkspaceMaterial,
    ablation_variants,
)


class _Retrieval:
    def __init__(self, passages):
        self._p = passages

    def retrieve(self, query, *, limit):
        return self._p[:limit]


class _Imaginer:
    def __init__(self, scenarios):
        self._s = scenarios

    def imagine(self, query, limit):
        return self._s[:limit]


# ── Provenance and trust are mandatory ──────────────────────────────────


def test_material_carries_source_and_trust():
    m = WorkspaceMaterial(text="Paris is the capital", source="retrieval", trust=GROUNDED)
    assert m.as_line() == "Paris is the capital"
    with pytest.raises(ValueError, match="trust must be"):
        WorkspaceMaterial(text="x", source="s", trust="vibes")
    with pytest.raises(ValueError, match="provenance"):
        WorkspaceMaterial(text="x", source="", trust=GROUNDED)


def test_imagined_material_is_visibly_marked_a_hypothesis():
    """The model must never mistake an imagined scenario for a fact."""
    m = WorkspaceMaterial(text="what if X", source="imagination", trust=HYPOTHETICAL)
    assert m.as_line().startswith("[hypothesis] ")


# ── Producers adapt real sources to the seam ────────────────────────────


def test_retrieval_producer_tags_material_grounded():
    p = RetrievalProducer(_Retrieval(["fact one", "fact two"]))
    out = p.produce("q", limit=5)
    assert [m.text for m in out] == ["fact one", "fact two"]
    assert all(m.trust == GROUNDED for m in out)


def test_imagination_producer_tags_material_hypothetical():
    p = ImaginationProducer(_Imaginer(["scenario A", "scenario B"]))
    out = p.produce("q", limit=5)
    assert all(m.trust == HYPOTHETICAL for m in out)


def test_a_failing_generator_degrades_to_empty_not_fabrication():
    class Broken:
        def imagine(self, query, limit):
            raise RuntimeError("sim crashed")

    assert ImaginationProducer(Broken()).produce("q", limit=3) == []


# ── Composition: facts first, budget enforced, question never crowded ───


def test_grounded_material_is_offered_before_hypothetical():
    composer = WorkspaceComposer(
        producers=[
            ImaginationProducer(_Imaginer(["imagined"])),
            RetrievalProducer(_Retrieval(["real fact"])),
        ],
        total_limit=10,
    )
    result = composer.compose("q")
    # grounded fact appears before the hypothesis regardless of producer order
    assert result["lines"][0] == "real fact"
    assert result["lines"][1].startswith("[hypothesis]")
    assert result["grounded"] == 1 and result["hypothetical"] == 1


def test_budget_drops_hypothetical_before_grounded():
    """When the window is tight, unverified material is cut first -- never
    the grounded facts."""
    composer = WorkspaceComposer(
        producers=[
            RetrievalProducer(_Retrieval(["f1", "f2"])),
            ImaginationProducer(_Imaginer(["h1", "h2"])),
        ],
        per_source_limit=4, total_limit=2,
    )
    result = composer.compose("q")
    assert result["grounded"] == 2
    assert result["hypothetical"] == 0
    assert result["dropped"] == 2


def test_context_block_is_empty_when_no_source_produces():
    composer = WorkspaceComposer(producers=[RetrievalProducer(_Retrieval([]))])
    block, receipt = composer.context_block("q")
    assert block == ""
    assert receipt["grounded"] == 0


def test_context_block_prepends_material():
    composer = WorkspaceComposer(producers=[RetrievalProducer(_Retrieval(["a fact"]))])
    block, _ = composer.context_block("q")
    assert "Known context:" in block and "a fact" in block


# ── The ablation harness that makes each source prove itself ────────────


def test_ablation_variants_cover_all_off_and_each_source_off():
    composer = WorkspaceComposer(producers=[
        RetrievalProducer(_Retrieval(["f"]), name="retrieval"),
        ImaginationProducer(_Imaginer(["s"]), name="imagination"),
    ])
    variants = ablation_variants(composer)
    assert set(variants) == {"all", "none", "without_retrieval", "without_imagination"}
    # 'none' produces nothing; 'without_retrieval' drops only retrieval
    assert variants["none"].compose("q")["lines"] == []
    assert variants["without_retrieval"].compose("q")["grounded"] == 0
    assert variants["without_imagination"].compose("q")["hypothetical"] == 0
