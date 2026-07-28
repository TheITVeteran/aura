"""The module may only claim what its own code can establish.

CP126 (medium), core/brain/nonparametric_memory.py: "Implementation comments
overstate demonstrated capacity and safety. The module generalizes kNN
memory to matching a much larger model and says fail-open interpolation
never degrades generation, while its own code provides no benchmark or proof
contract... These claims exceed what the implementation can establish and
can mislead certification surfaces that treat comments as capability
evidence."

Two claims, both overstated, and they are wrong in different ways.

"A small model + a large datastore matches a much bigger model on knowledge"
is a real result from the kNN-LM literature and not a measurement of this
code. Nothing here benchmarks Aura against a larger model. The design is
BASED on that result; it does not demonstrate it.

"generation never degrades below the raw model" claimed more than the
implementation does. What fail-open actually guarantees is that three
specific paths — no neighbours above the gate, λ≈0, and any exception —
return the caller's distribution UNCHANGED. It says nothing about a blend
that fires, and a blend that fires can absolutely be wrong; the similarity
gate exists precisely because confidently-recalled neighbours from a
different fact once outvoted the exact match.

The narrow claim is worth having because it is checkable. These tests check
it, which is what turns a comment into evidence.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from core.brain import nonparametric_memory as npm


@pytest.fixture
def store(tmp_path):
    return npm.NonParametricMemory(dim=8, path=tmp_path / "npm.json")


def _probs() -> dict[int, float]:
    return {1: 0.5, 2: 0.3, 3: 0.2}


class TestTheGuaranteedPathsReturnTheInputUnchanged:
    """The narrow claim, tested. Each of these cannot degrade generation
    because it does not touch the distribution at all."""

    def test_no_neighbours_returns_the_input(self, store):
        lm = _probs()
        out = store.interpolate(lm, np.zeros(8, dtype=np.float32))
        assert out == lm

    def test_a_raising_query_returns_the_input(self, store, monkeypatch):
        lm = _probs()

        def _boom(*_args, **_kwargs):
            raise RuntimeError("index exploded")

        monkeypatch.setattr(store, "query", _boom)
        assert store.interpolate(lm, np.zeros(8, dtype=np.float32)) == lm

    def test_lambda_zero_returns_the_input(self, store):
        lm = _probs()
        out = store.interpolate(
            lm, np.zeros(8, dtype=np.float32), lam_override=0.0,
        )
        assert out == lm

    def test_a_malformed_key_returns_the_input(self, store):
        lm = _probs()
        assert store.interpolate(lm, np.array([], dtype=np.float32)) == lm

    def test_the_returned_mass_is_still_a_distribution(self, store):
        out = store.interpolate(_probs(), np.zeros(8, dtype=np.float32))
        assert all(np.isfinite(v) for v in out.values())
        assert all(v >= 0.0 for v in out.values())


class TestTheOverstatedClaimsAreGone:
    def test_the_never_degrades_claim_is_qualified(self):
        source = inspect.getsource(npm.NonParametricMemory.interpolate)
        # The unconditional promise must not stand on its own.
        assert "never degrades below the raw model" not in source.replace(
            'Saying "generation never degrades below the raw model" claimed the', "",
        )

    def test_the_docstring_separates_guaranteed_from_not_guaranteed(self):
        source = inspect.getsource(npm.NonParametricMemory.interpolate)
        assert "What IS guaranteed" in source
        assert "What is NOT guaranteed" in source

    def test_the_capacity_claim_is_attributed_to_the_literature(self):
        doc = npm.__doc__ or ""
        assert "literature" in doc
        assert "not a measurement of this implementation" in doc.replace("\n", " ")

    def test_the_module_says_it_claims_a_mechanism_not_a_capability(self):
        doc = npm.__doc__ or ""
        assert "mechanism, not a capability" in doc


class TestABlendThatFiresIsNotClaimedSafe:
    """The honest half: once mass moves, this module makes no promise about
    the result. Pinned so the unconditional claim cannot quietly return."""

    def test_the_docstring_admits_a_firing_blend_can_be_wrong(self):
        source = inspect.getsource(npm.NonParametricMemory.interpolate)
        assert "can be wrong" in source
