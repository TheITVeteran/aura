"""CP126 ``core/brain/nonparametric_memory.py`` — fifteen findings, six critical.

The store holds hidden-state vectors keyed to next tokens and mixes them
into the model's own distribution. Six criticals, and each is a boundary
that was documented rather than implemented.

The store's whole identity was its hidden WIDTH, so two models of the same
width shared one — combining one model's vectors with the other's token
ids and reporting successful reuse. The module docstring says the store
holds verifier-clean trusted knowledge, and ``add`` took anything from
anyone. Entries carried no principal, so one person's query searched
another's memory. Keys and metadata were published as two independent
replacements. The loader validated rank, width and list lengths and
nothing else. And ``np.load`` materialised the whole file before the
entry cap applied.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from core.brain.nonparametric_identity import (
    EntryProvenance,
    StoreIdentity,
    TrustLevel,
)
from core.brain.nonparametric_memory import (
    NonParametricMemory,
    get_nonparametric_memory,
    reset_nonparametric_memory_for_test,
)

DIM = 8


def _identity(**over) -> StoreIdentity:
    fields = {
        "dim": DIM,
        "checkpoint": "ck-alpha",
        "architecture": "qwen2",
        "tokenizer_vocab_size": 1000,
        "quantization": "q4",
    }
    fields.update(over)
    return StoreIdentity(**fields)


def _store(tmp_path, name="s", **over) -> NonParametricMemory:
    return NonParametricMemory(
        DIM, path=str(tmp_path / name), identity=over.pop("identity", _identity(**over))
    )


def _vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=DIM).astype(np.float32)


# ── aba3eb39: a width is not an identity ────────────────────────────────────


def test_two_models_of_the_same_width_do_not_share_a_store(tmp_path):
    alpha = _store(tmp_path, "a", checkpoint="ck-alpha")
    beta = _store(tmp_path, "b", checkpoint="ck-beta")
    assert alpha._identity.slug() != beta._identity.slug(), (
        "two different checkpoints of the same width produced one store name"
    )


def test_an_incompatible_store_is_refused_rather_than_reused(tmp_path):
    written = _store(tmp_path, "shared", checkpoint="ck-alpha")
    written.add(_vec(1), 7, "seven", provenance=EntryProvenance(source_id="s"))
    assert written.persist()

    other_model = NonParametricMemory(
        DIM, path=str(tmp_path / "shared"), identity=_identity(checkpoint="ck-beta")
    )
    assert len(other_model) == 0, (
        "one model's vectors were loaded under another's identity, and the "
        "code reported successful reuse"
    )


@pytest.mark.parametrize(
    "field",
    ["checkpoint", "architecture", "quantization", "hidden_state_tap", "adapter"],
)
def test_every_identity_field_can_refuse_a_reuse(field):
    mine = _identity()
    theirs = _identity(**{field: "something-else"})
    compatible, why = mine.compatible_with(theirs)
    assert compatible is False and field in why


def test_a_tokenizer_change_refuses_reuse():
    compatible, why = _identity().compatible_with(_identity(tokenizer_vocab_size=999))
    assert compatible is False and "tokenizer" in why


def test_a_centering_change_refuses_reuse():
    mine = _identity()
    compatible, why = mine.compatible_with(_identity(centering_version=0))
    assert compatible is False and "centering" in why


# ── ff3a4505: an entry says why it was admitted, and can be revoked ─────────


def test_an_entry_records_its_source_and_trust(tmp_path):
    store = _store(tmp_path)
    store.add(
        _vec(1),
        7,
        "seven",
        provenance=EntryProvenance(
            source_id="verifier-run-3", trust=TrustLevel.VERIFIED, verifier="arith"
        ),
    )
    assert store._provenance[0].source_id == "verifier-run-3"
    assert store._provenance[0].trust_rank > TrustLevel.rank(TrustLevel.UNVERIFIED)


def test_a_discredited_source_can_be_revoked_entirely(tmp_path):
    store = _store(tmp_path)
    for seed in range(4):
        store.add(_vec(seed), seed + 1, "t", provenance=EntryProvenance(source_id="bad"))
    store.add(_vec(99), 9, "keep", provenance=EntryProvenance(source_id="good"))

    assert store.revoke_source("bad") == 4
    assert len(store) == 1
    assert store._provenance[0].source_id == "good"


def test_an_unattributed_add_is_recorded_as_unattributed(tmp_path):
    store = _store(tmp_path)
    store.add(_vec(1), 7, "seven")
    assert store._provenance[0].source_id == "unattributed"
    assert store._provenance[0].trust == TrustLevel.UNVERIFIED


# ── 62309dad: one principal cannot read another's memory ────────────────────


def test_a_query_is_scoped_to_its_principal(tmp_path):
    store = _store(tmp_path)
    store.add(_vec(1), 7, "mine", provenance=EntryProvenance(source_id="s", principal="bryan"))
    store.add(_vec(2), 8, "theirs", provenance=EntryProvenance(source_id="s", principal="other"))

    tokens = {nb.token for nb in store.query(_vec(1), k=8, principal="bryan")}
    assert tokens == {"mine"}, f"a query reached another principal's memory: {tokens}"


def test_shared_entries_are_visible_to_everyone(tmp_path):
    store = _store(tmp_path)
    store.add(_vec(1), 7, "system", provenance=EntryProvenance(source_id="s", principal="shared"))
    assert store.query(_vec(1), k=4, principal="anybody")


def test_a_principal_can_be_forgotten(tmp_path):
    store = _store(tmp_path)
    store.add(_vec(1), 7, "a", provenance=EntryProvenance(source_id="s", principal="bryan"))
    store.add(_vec(2), 8, "b", provenance=EntryProvenance(source_id="s", principal="other"))
    assert store.forget_principal("bryan") == 1
    assert len(store) == 1


# ── b8b9656b: one generation, or none ───────────────────────────────────────


def test_the_manifest_pins_the_keys_it_describes(tmp_path):
    store = _store(tmp_path)
    store.add(_vec(1), 7, "seven", provenance=EntryProvenance(source_id="s"))
    assert store.persist()

    meta = json.loads((tmp_path / "s.meta.json").read_text())
    assert meta["schema_version"] >= 3
    assert len(meta["keys_sha256"]) == 64
    assert "generation" in meta


def test_metadata_paired_with_the_wrong_keys_is_refused(tmp_path):
    store = _store(tmp_path)
    store.add(_vec(1), 7, "seven", provenance=EntryProvenance(source_id="s"))
    store.persist()

    # A keys file from another generation, as a torn publish would leave.
    np.save(tmp_path / "s.keys.npy", np.ones((1, DIM), dtype=np.float32))

    reloaded = _store(tmp_path)
    assert len(reloaded) == 0, (
        "vectors from one generation were paired with token metadata from "
        "another, with nothing able to detect it"
    )


def test_a_failed_write_leaves_no_temporary_files(tmp_path, monkeypatch):
    store = _store(tmp_path)
    store.add(_vec(1), 7, "seven", provenance=EntryProvenance(source_id="s"))

    import core.brain.nonparametric_memory as module

    monkeypatch.setattr(module.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("no")))
    assert store.persist() is False
    leftovers = [p for p in tmp_path.iterdir() if p.suffix in {".npy", ".json"} and "tmp" in p.name]
    assert leftovers == [], f"temporary files survived a failed write: {leftovers}"


# ── 848cf532: the loader validates what it reads ────────────────────────────


def _corrupt(tmp_path, mutate) -> NonParametricMemory:
    store = _store(tmp_path)
    for seed in range(3):
        store.add(_vec(seed), seed + 1, "t", provenance=EntryProvenance(source_id="s"))
    store.persist()
    meta = json.loads((tmp_path / "s.meta.json").read_text())
    mutate(meta)
    (tmp_path / "s.meta.json").write_text(json.dumps(meta))
    return _store(tmp_path)


def test_a_wrong_declared_dim_is_refused(tmp_path):
    assert len(_corrupt(tmp_path, lambda m: m.__setitem__("dim", DIM + 1))) == 0


def test_a_non_finite_timestamp_is_refused(tmp_path):
    assert len(_corrupt(tmp_path, lambda m: m.__setitem__("ts", [float("nan")] * 3))) == 0


def test_an_out_of_vocabulary_token_id_is_refused(tmp_path):
    assert len(_corrupt(tmp_path, lambda m: m.__setitem__("token_ids", [10**9, 1, 2]))) == 0


def test_a_store_without_an_identity_is_refused(tmp_path):
    assert len(_corrupt(tmp_path, lambda m: m.pop("identity"))) == 0


def test_a_pre_manifest_generation_is_not_loaded(tmp_path):
    assert len(_corrupt(tmp_path, lambda m: m.__setitem__("schema_version", 2))) == 0


# ── 1d1d0bfd: the entry bound applies before the allocation ─────────────────


def test_the_persisted_file_is_memory_mapped_before_the_cap(tmp_path):
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(NonParametricMemory._load).lstrip())
    loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "load"
    ]
    assert loads, "the loader does not read a keys file"
    for call in loads:
        assert any(kw.arg == "mmap_mode" for kw in call.keywords), (
            "np.load materialises the entire persisted matrix before the "
            "entry count is capped, so a replaced file allocates past the bound"
        )


def test_an_oversized_store_loads_only_its_bound(tmp_path):
    big = NonParametricMemory(
        DIM, path=str(tmp_path / "s"), max_entries=200, identity=_identity()
    )
    for seed in range(120):
        big.add(_vec(seed), seed % 900, "t", provenance=EntryProvenance(source_id="s"))
    big.persist()

    small = NonParametricMemory(
        DIM, path=str(tmp_path / "s"), max_entries=64, identity=_identity()
    )
    assert len(small) == 64


# ── 4d9d6616: restart retention matches live retention ──────────────────────


def test_reload_keeps_the_entries_eviction_would_have_kept(tmp_path):
    store = NonParametricMemory(
        DIM, path=str(tmp_path / "s"), max_entries=100, identity=_identity()
    )
    store.add(_vec(1), 11, "heavy", weight=10.0, provenance=EntryProvenance(source_id="s"))
    for seed in range(2, 12):
        store.add(_vec(seed), seed, "light", weight=0.01, provenance=EntryProvenance(source_id="s"))
    store.persist()

    reloaded = NonParametricMemory(
        DIM, path=str(tmp_path / "s"), max_entries=3, identity=_identity()
    )
    assert "heavy" in reloaded._tokens, (
        "loading an oversized store kept the LAST records while live eviction "
        "drops the lowest weight-times-recency, so retention differed across a restart"
    )


# ── 2979489a: magnitudes are bounded at admission ───────────────────────────


def test_a_vector_whose_square_overflows_is_refused(tmp_path):
    store = _store(tmp_path)
    huge = np.full(DIM, 1e30, dtype=np.float32)
    assert np.all(np.isfinite(huge))
    assert store.add(huge, 7, "seven") is False, (
        "a finite float32 key whose squared norm overflows was admitted, and "
        "that norm feeds every distance in the store"
    )


def test_a_non_finite_query_returns_nothing(tmp_path):
    store = _store(tmp_path)
    store.add(_vec(1), 7, "seven", provenance=EntryProvenance(source_id="s"))
    assert store.query(np.full(DIM, np.nan, dtype=np.float32)) == []


# ── cb95d7d9: a NaN clamps to the floor, never the ceiling ──────────────────


def test_a_nan_clamps_down():
    from core.brain.nonparametric_memory import _clamp

    assert _clamp(float("nan"), 0.0, 0.7) == 0.0, (
        "max(lo, min(hi, nan)) returns hi, so a NaN lambda handed the whole "
        "distribution to recall"
    )
    assert _clamp(float("inf"), 0.0, 0.7) == 0.0
    assert _clamp(0.5, 0.0, 0.7) == 0.5


def test_a_hostile_lambda_override_cannot_create_negative_model_mass(tmp_path):
    store = _store(tmp_path)
    store.add(_vec(1), 7, "seven", provenance=EntryProvenance(source_id="s"))
    for override in (4.0, float("nan"), -3.0):
        blended = store.interpolate({7: 0.6, 8: 0.4}, _vec(1), lam_override=override)
        assert all(math.isfinite(p) and p >= 0.0 for p in blended.values())


def test_a_caller_distribution_with_a_nan_does_not_propagate_it(tmp_path):
    store = _store(tmp_path)
    store.add(_vec(1), 7, "seven", provenance=EntryProvenance(source_id="s"))
    blended = store.interpolate({7: float("nan"), 8: 0.4}, _vec(1))
    assert all(math.isfinite(p) for p in blended.values())


# ── 957c060f: token ids are vocabulary-bounded ──────────────────────────────


def test_an_out_of_vocabulary_id_is_never_stored(tmp_path):
    store = _store(tmp_path)
    assert store.add(_vec(1), 10**9, "huge") is False
    assert store.add(_vec(2), -1, "negative") is False
    assert len(store) == 0


def test_an_unknown_vocabulary_still_refuses_a_negative_id(tmp_path):
    store = NonParametricMemory(
        DIM, path=str(tmp_path / "s"), identity=StoreIdentity(dim=DIM)
    )
    assert store.add(_vec(1), -5, "negative") is False
    assert store.add(_vec(2), 5, "ok") is True


# ── 4354909: a replay writes one fact once ──────────────────────────────────


def test_the_same_fact_added_twice_is_one_entry(tmp_path):
    store = _store(tmp_path)
    key = _vec(1)
    provenance = EntryProvenance(source_id="s", principal="bryan")
    assert store.add(key, 7, "seven", provenance=provenance)
    assert store.add(key, 7, "seven", provenance=provenance)
    assert len(store) == 1, (
        "a replay created a second independent nearest neighbour, so the "
        "duplicate's kNN vote amplified the same token"
    )


def test_two_different_facts_are_two_entries(tmp_path):
    store = _store(tmp_path)
    store.add(_vec(1), 7, "seven", provenance=EntryProvenance(source_id="s"))
    store.add(_vec(2), 8, "eight", provenance=EntryProvenance(source_id="s"))
    assert len(store) == 2


# ── 7a4bfedb: the gate cannot race the similarity mode ──────────────────────


def test_a_neighbour_carries_the_threshold_it_was_judged_against(tmp_path):
    store = _store(tmp_path)
    store.add(_vec(1), 7, "seven", provenance=EntryProvenance(source_id="s"))
    neighbour = store.query(_vec(1), k=1)[0]
    assert neighbour.gate_threshold in {store.MIN_SIM_RAW, store.MIN_SIM_CENTERED}
    assert neighbour.centered is store.similarity_ready()


def test_a_query_can_decline_to_move_the_persisted_mean(tmp_path):
    store = _store(tmp_path)
    store.add(_vec(1), 7, "seven", provenance=EntryProvenance(source_id="s"))
    before = store._query_mu_n
    store.query(_vec(2), k=1, update_mean=False)
    assert store._query_mu_n == before, (
        "every query, including unauthorized and unrelated ones, mutated the "
        "global persisted similarity transform"
    )


# ── da1019b0: a recall receipt says why nothing happened ────────────────────


def test_each_fallthrough_names_itself(tmp_path, monkeypatch):
    monkeypatch.delenv("AURA_NONPARAMETRIC_MEMORY", raising=False)
    store = _store(tmp_path)
    logits = np.zeros(16)

    store.apply_to_logits(logits, _vec(1))
    assert store.last_recall_receipt()["reason"] == "flag_off"

    monkeypatch.setenv("AURA_NONPARAMETRIC_MEMORY", "1")
    store.apply_to_logits(np.full(4, np.nan), _vec(1))
    assert store.last_recall_receipt()["reason"] == "invalid_logits"

    store.apply_to_logits(logits, _vec(1))
    assert store.last_recall_receipt()["reason"] == "no_neighbors"


def test_the_receipt_names_the_store_and_its_neighbours(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_NONPARAMETRIC_MEMORY", "1")
    store = _store(tmp_path)
    key = _vec(1)
    store.add(key, 7, "seven", provenance=EntryProvenance(source_id="verified-run"))
    store.apply_to_logits(np.zeros(16), key)

    receipt = store.last_recall_receipt()
    assert receipt["store"] == store._identity.slug()
    assert receipt["gate_mode"] in {"raw", "centered"}
    assert receipt["reason"], "the receipt has no reason"


def test_stats_reports_the_memory_it_actually_holds(tmp_path):
    store = _store(tmp_path)
    for seed in range(5):
        store.add(_vec(seed), seed, "a long token string", provenance=EntryProvenance(source_id="s"))
    stats = store.stats()
    assert stats["allocated_bytes"] > store._keys.nbytes, (
        "the figure counted the two matrices and none of the metadata"
    )
    assert stats["identity"]["dim"] == DIM


# ── 13a3ce91: the registry is read under its own lock ───────────────────────


def test_the_registry_is_keyed_by_identity_not_width():
    reset_nonparametric_memory_for_test()
    try:
        alpha = get_nonparametric_memory(identity=_identity(checkpoint="ck-alpha"))
        beta = get_nonparametric_memory(identity=_identity(checkpoint="ck-beta"))
        assert alpha is not beta, (
            "two checkpoints of one width were handed the same store"
        )
    finally:
        reset_nonparametric_memory_for_test()


def test_every_registry_read_happens_under_the_lock():
    import ast
    import inspect

    import core.brain.nonparametric_memory as module

    tree = ast.parse(inspect.getsource(module.get_nonparametric_memory).lstrip())
    body = [node for node in tree.body[0].body if not isinstance(node, ast.Expr)]
    assert body and isinstance(body[-1], ast.With), (
        "a registry read sits outside the lifecycle lock, so two callers can "
        "each build a store for the same path and persist over one another"
    )


def test_reset_drops_every_store():
    reset_nonparametric_memory_for_test()
    get_nonparametric_memory(identity=_identity())
    reset_nonparametric_memory_for_test()
    assert get_nonparametric_memory(0) is None
