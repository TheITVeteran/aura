"""An isolation argument that can be empty provides no isolation.

`NonParametricMemory` implements provenance and principal isolation
properly: `add` takes an `EntryProvenance`, `query` takes a principal,
entries can be revoked by source and erased by principal. The storage
layer is right.

The live seams did not carry any of it. The foreground logits processor
called `memory.query(key, k=k)` — and the store's own docstring says an
empty principal searches EVERYTHING. The ingest path called
`_mem.add(key, token_id, ...)` with no provenance, so every entry the
resident worker wrote was `source_id="unattributed"`,
`trust=UNVERIFIED`, `principal="anonymous"`: exactly the three values
that make revocation and per-principal erasure impossible.

So the architecture was credited with isolation the primary live seam did
not enforce. That is a cross-layer invariant failure, and a unit test on
either layer alone could never see it.

This file walks the parse tree of the live seam modules. A call to
`query` or `add` that loses its keyword fails here, in CI, rather than
being noticed by the next reviewer.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

from core.brain.nonparametric_binding import MemoryBinding, binding_for_job
from core.brain.nonparametric_identity import EntryProvenance, StoreIdentity, TrustLevel
from core.brain.nonparametric_memory import SHARED_MEMORY_PRINCIPAL, NonParametricMemory

DIM = 8

#: Every module where a query or an add reaches the live datastore.
LIVE_SEAMS = (
    "core/brain/nonparametric_worker.py",
    "core/brain/nonparametric_ingest.py",
    "core/brain/llm/latent_cortex/nonparametric_context.py",
)


def _calls_named(path: str, method: str) -> list[ast.Call]:
    tree = ast.parse(pathlib.Path(path).read_text())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == method
    ]


# ── the structural gate ─────────────────────────────────────────────────────


@pytest.mark.parametrize("path", LIVE_SEAMS)
def test_no_live_query_omits_its_principal(path):
    """The one seam where a person's turn meets the datastore."""
    for call in _calls_named(path, "query"):
        keywords = {kw.arg for kw in call.keywords}
        assert "principal" in keywords, (
            f"{path}:{call.lineno} queries the datastore with no principal. "
            "The store's own docstring says an empty principal searches every "
            "entry, so this reads other principals' memory."
        )


@pytest.mark.parametrize("path", LIVE_SEAMS)
def test_no_live_add_omits_its_provenance(path):
    """An entry that cannot be attributed cannot be revoked."""
    for call in _calls_named(path, "add"):
        target = getattr(call.func, "value", None)
        # Only datastore writes; set.add and list.add are not this.
        if not (isinstance(target, ast.Attribute) and "mem" in target.attr.lower()):
            continue
        keywords = {kw.arg for kw in call.keywords}
        assert "provenance" in keywords, (
            f"{path}:{call.lineno} writes a datastore entry with no provenance, "
            "so it defaults to unattributed/unverified/anonymous"
        )


def test_interpolate_is_scoped_too():
    """It queries internally, so an unscoped call is the same leak."""
    for call in _calls_named("core/brain/nonparametric_worker.py", "interpolate"):
        keywords = {kw.arg for kw in call.keywords}
        assert "principal" in keywords, (
            "interpolate() runs a query of its own; an unscoped call reads "
            "every principal's entries into the blend"
        )


# ── the binding type refuses rather than defaulting ─────────────────────────


def test_a_binding_cannot_be_built_without_a_principal():
    with pytest.raises(ValueError, match="principal"):
        MemoryBinding(principal="", source_id="chat")
    with pytest.raises(ValueError, match="principal"):
        MemoryBinding(principal="   ", source_id="chat")


def test_a_binding_cannot_be_built_without_a_revocable_source():
    with pytest.raises(ValueError, match="revoke"):
        MemoryBinding(principal="bryan", source_id="")


def test_a_job_that_names_nobody_yields_no_binding():
    assert binding_for_job({}, source_id="foreground") is None
    assert binding_for_job(None, source_id="foreground") is None
    assert binding_for_job({"principal": "  "}, source_id="foreground") is None


@pytest.mark.parametrize("field", ["principal", "user_id", "subject", "principal_id"])
def test_a_job_that_names_someone_yields_a_binding(field):
    binding = binding_for_job({field: "bryan"}, source_id="foreground")
    assert binding is not None and binding.principal == "bryan"


# ── the foreground build refuses an unscoped install ────────────────────────


def test_the_foreground_processor_is_not_installed_without_a_principal(monkeypatch):
    """Recall is an enhancement. Running it unscoped is the leak itself."""
    import core.brain.nonparametric_worker as worker

    monkeypatch.setattr(worker, "foreground_enabled", lambda: True)
    monkeypatch.setattr(worker, "foreground_memory_admitted_for_job", lambda job: True)

    store = NonParametricMemory(DIM, path=None, identity=StoreIdentity(dim=DIM))
    store.add(
        np.ones(DIM, dtype=np.float32),
        3,
        "x",
        provenance=EntryProvenance(source_id="s", principal="someone_else"),
    )
    monkeypatch.setattr(worker, "get_nonparametric_memory", lambda dim: store)
    monkeypatch.setattr(worker, "_unusable_datastore_reason", lambda memory: "")

    model = type("M", (), {"args": type("A", (), {"hidden_size": DIM})()})()
    assert worker.maybe_build_foreground(model, job={}) is None, (
        "a job naming no principal was given a processor that reads every principal's entries"
    )
    outcome = worker.last_recall_outcome()
    assert outcome["status"] == "not_admitted"
    assert "principal" in outcome["detail"]


# ── the ingestor refuses an anonymous write at construction ─────────────────


def test_an_ingestor_cannot_be_built_without_provenance():
    from core.brain.nonparametric_ingest import NonParametricIngestor

    with pytest.raises(TypeError, match="provenance"):
        NonParametricIngestor(object())
    with pytest.raises(TypeError, match="EntryProvenance"):
        NonParametricIngestor(object(), provenance="reasoning_solved_cache")


@pytest.mark.parametrize(
    "provenance",
    [
        EntryProvenance(source_id="unattributed", principal="bryan"),
        EntryProvenance(source_id="", principal="bryan"),
        EntryProvenance(source_id="cache", principal="anonymous"),
        EntryProvenance(source_id="cache", principal=""),
    ],
)
def test_an_ingestor_refuses_placeholder_identity(provenance):
    from core.brain.nonparametric_ingest import NonParametricIngestor

    with pytest.raises(ValueError):
        NonParametricIngestor(object(), provenance=provenance)


def test_a_real_identity_builds(tmp_path):
    from core.brain.nonparametric_ingest import NonParametricIngestor, ingest_provenance

    ingestor = NonParametricIngestor(
        object(),
        provenance=ingest_provenance(
            principal=SHARED_MEMORY_PRINCIPAL,
            source_id="trusted_store:reasoning_traces",
            trust=TrustLevel.VERIFIED,
        ),
        dedup_path=str(tmp_path / "seen.json"),
    )
    assert ingestor._provenance.principal == SHARED_MEMORY_PRINCIPAL
    assert ingestor._provenance.trust == TrustLevel.VERIFIED


# ── the isolation actually holds end to end ─────────────────────────────────


def test_one_principals_entry_is_invisible_to_another():
    store = NonParametricMemory(DIM, path=None, identity=StoreIdentity(dim=DIM))
    key = np.ones(DIM, dtype=np.float32)
    store.add(key, 3, "mine", provenance=EntryProvenance(source_id="s", principal="bryan"))

    assert store.query(key, principal="bryan"), "the owner cannot see their own entry"
    assert store.query(key, principal="someone_else") == [], (
        "another principal read an entry that is not theirs"
    )


def test_shared_entries_are_visible_to_everyone():
    store = NonParametricMemory(DIM, path=None, identity=StoreIdentity(dim=DIM))
    key = np.ones(DIM, dtype=np.float32)
    store.add(
        key,
        3,
        "system fact",
        provenance=EntryProvenance(source_id="s", principal=SHARED_MEMORY_PRINCIPAL),
    )
    assert store.query(key, principal="anybody"), (
        "Aura's own verified reasoning results should be readable by any turn"
    )


def test_a_trusted_store_can_be_revoked_by_source():
    store = NonParametricMemory(DIM, path=None, identity=StoreIdentity(dim=DIM))
    for index in range(3):
        vec = np.zeros(DIM, dtype=np.float32)
        vec[index % DIM] = 1.0
        store.add(
            vec,
            index + 1,
            "t",
            provenance=EntryProvenance(
                source_id="trusted_store:reasoning_traces",
                principal=SHARED_MEMORY_PRINCIPAL,
            ),
        )
    assert store.revoke_source("trusted_store:reasoning_traces") == 3, (
        "entries written by the live ingest path could not be revoked as a "
        "group, because they all shared one indistinguishable origin"
    )


# ── the latent-cortex retrieval refuses rather than reading everything ──────


def test_recurrent_retrieval_refuses_without_a_principal():
    from core.brain.llm.latent_cortex.nonparametric_context import (
        retrieve_observation,
        validate_receipt,
    )

    observation, receipt = retrieve_observation(np.zeros(DIM, dtype=np.float32), None, enabled=True)
    assert observation is None
    assert receipt["status"] == "no_principal", (
        "a recurrent step pulled a clue from every principal's memory into this turn's evidence"
    )
    # The verdict is a real one the receipt validator recognises, not an
    # unreadable status that silently counts as something else.
    assert validate_receipt(receipt)["status"] == "no_principal"


def test_the_ingest_worker_attributes_every_entry_to_its_store():
    """CP-review: the resident worker's ingest job entered through the
    anonymous path, so nothing it wrote could be revoked per store."""
    import inspect

    from core.brain.llm import mlx_worker

    source = inspect.getsource(mlx_worker)
    assert "collect_trusted_pairs_by_source" in source, (
        "the worker flattened every trusted store into one origin"
    )
    assert "ingest_provenance(" in source
    assert "trusted_store:" in source
