"""Contract tests: layer-schedule programs + virtual-width branches."""
from __future__ import annotations

import json

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen2 import Model, ModelArgs

from core.brain.llm.latent_cortex.branches import BRANCH_ROLES, BranchEnsemble
from core.brain.llm.latent_cortex.recurrence import WindowRunner
from core.brain.llm.latent_cortex.schedules import (
    LayerSchedule,
    ScheduleLibrary,
    ScheduleSearch,
    StageOp,
)
from core.brain.llm.latent_cortex.types import (
    BranchConfig,
    ComputeBudget,
    RecurrenceConfig,
    WorkspaceConfig,
)

N_LAYERS, P_END, C_START = 8, 2, 6
PROMPT = [[5, 9, 17, 3, 42, 7, 11, 23, 2, 88]]


@pytest.fixture(scope="module")
def tiny_model():
    args = ModelArgs(
        model_type="qwen2",
        hidden_size=64,
        num_hidden_layers=N_LAYERS,
        intermediate_size=128,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


def _prefill(model):
    from mlx_lm.models.base import create_attention_mask

    inner = model.model
    prompt = mx.array(PROMPT)
    cache = [KVCache() for _ in inner.layers]
    h = inner.embed_tokens(prompt)
    mask = create_attention_mask(h, cache)
    for i, layer in enumerate(inner.layers):
        h = layer(h, mask, cache[i])
    mx.eval(h)
    return cache


# ── Schedule programs ───────────────────────────────────────────────────


def test_schedule_hash_is_canonical_and_name_free():
    a = LayerSchedule(ops=(StageOp(2, 6, 4),), name="foo")
    b = LayerSchedule(ops=(StageOp(2, 6, 4),), name="bar")
    c = LayerSchedule(ops=(StageOp(2, 6, 5),))
    assert a.schedule_hash == b.schedule_hash
    assert a.schedule_hash != c.schedule_hash
    round_trip = LayerSchedule.from_dict(json.loads(json.dumps(a.to_dict())))
    assert round_trip.schedule_hash == a.schedule_hash


def test_schedule_validation_rejects_escapes_and_degenerates():
    ok = LayerSchedule(ops=(StageOp(2, 6, 4), StageOp(3, 5, 2)))
    assert ok.validate(prelude_end=2, coda_start=6) == []
    assert LayerSchedule(ops=(StageOp(0, 6, 1),)).validate(prelude_end=2, coda_start=6)
    assert LayerSchedule(ops=(StageOp(2, 7, 1),)).validate(prelude_end=2, coda_start=6)
    assert LayerSchedule(ops=(StageOp(4, 4, 1),)).validate(prelude_end=2, coda_start=6)
    assert LayerSchedule(ops=(StageOp(2, 6, 0),)).validate(prelude_end=2, coda_start=6)
    assert LayerSchedule(ops=()).validate(prelude_end=2, coda_start=6)
    monster = LayerSchedule(ops=(StageOp(2, 6, 100_000),))
    assert monster.validate(prelude_end=2, coda_start=6)


def test_library_prefers_evidence_over_novelty(tmp_path):
    lib = ScheduleLibrary(tmp_path / "sched.json")
    default = LayerSchedule.single_window(2, 6, 4)
    exotic = LayerSchedule(ops=(StageOp(2, 4, 3), StageOp(4, 6, 3)), name="exotic")

    # Exotic schedule with too few trials must NOT displace the default.
    for _ in range(3):
        lib.record_outcome(exotic, "math", True)
    best = lib.best_for_domain("math", prelude_end=2, coda_start=6, default_repeats=4)
    assert best.schedule_hash == default.schedule_hash

    # With enough trials and a genuinely better Wilson LB, it wins.
    for _ in range(20):
        lib.record_outcome(exotic, "math", True)
    for _ in range(10):
        lib.record_outcome(default, "math", False)
    best = lib.best_for_domain("math", prelude_end=2, coda_start=6, default_repeats=4)
    assert best.schedule_hash == exotic.schedule_hash


def test_library_ignores_records_invalid_for_current_topology(tmp_path):
    lib = ScheduleLibrary(tmp_path / "sched.json")
    stale = LayerSchedule(ops=(StageOp(2, 30, 4),), name="from-bigger-model")
    for _ in range(20):
        lib.record_outcome(stale, "math", True)
    best = lib.best_for_domain("math", prelude_end=2, coda_start=6, default_repeats=4)
    assert best.schedule_hash == LayerSchedule.single_window(2, 6, 4).schedule_hash


def test_library_persistence_round_trip(tmp_path):
    path = tmp_path / "sched.json"
    lib = ScheduleLibrary(path)
    s = LayerSchedule(ops=(StageOp(2, 5, 2),), name="s")
    for _ in range(9):
        lib.record_outcome(s, "code", True, provenance="unit-test")
    assert lib.save() is True
    reloaded = ScheduleLibrary(path)
    best = reloaded.best_for_domain("code", prelude_end=2, coda_start=6, default_repeats=4)
    assert best.schedule_hash == s.schedule_hash
    assert reloaded.status()["records"] == 1


def test_search_is_deterministic_and_finds_planted_optimum():
    # Planted structure: score = repeats of the widest op, capped — the
    # search must climb toward more repeats without violating validity.
    def evaluator(s: LayerSchedule) -> float:
        widest = max(s.ops, key=lambda op: op.end - op.start)
        return min(widest.repeats, 6) - 0.01 * s.total_layer_repeats

    def run():
        search = ScheduleSearch(prelude_end=2, coda_start=6, max_repeats=8, seed=11)
        return search.run(evaluator, population=6, generations=5)

    r1, r2 = run(), run()
    assert r1.best.schedule_hash == r2.best.schedule_hash, "search must be deterministic"
    base = LayerSchedule.single_window(2, 6, 4)
    assert r1.best_score >= evaluator(base), "search must not lose to its seed"
    assert r1.best.validate(prelude_end=2, coda_start=6) == []
    assert r1.evaluated >= 6


# ── Branch ensemble ─────────────────────────────────────────────────────


def _ensemble(model, cache, n_branches=3, budget=None, exchange_interval=2):
    inner = model.model
    emb = inner.embed_tokens(mx.array(PROMPT))
    budget = budget or ComputeBudget()
    runner = WindowRunner(inner, budget)
    ensemble = BranchEnsemble.seed(
        emb,
        WorkspaceConfig(n_slots=4, seed=3),
        BranchConfig(n_branches=n_branches, exchange_interval=exchange_interval),
        RecurrenceConfig(max_steps=8, convergence_eps=0.02, min_steps=2),
        runner,
        cache,
        P_END,
    )
    return ensemble, runner, budget


def test_branch_seeding_gives_distinct_roles_and_clean_cache(tiny_model):
    cache = _prefill(tiny_model)
    prompt_len = len(PROMPT[0])
    ensemble, _, _ = _ensemble(tiny_model, cache)
    assert [b.role for b in ensemble.branches] == list(BRANCH_ROLES[:3])
    # Branch seeding must NOT persist anything: caches stay prompt-only.
    assert all(c.offset == prompt_len for c in cache)
    z0, z1 = ensemble.branches[0].z, ensemble.branches[1].z
    assert not bool(mx.allclose(z0, z1)), "role basins must differ"


def test_branches_step_exchange_and_halt(tiny_model):
    cache = _prefill(tiny_model)
    prompt_len = len(PROMPT[0])
    ensemble, runner, budget = _ensemble(tiny_model, cache)
    for _ in range(8):
        if ensemble.all_halted():
            break
        ensemble.step_all(runner, cache, P_END, C_START, budget=budget)
        assert all(c.offset == prompt_len for c in cache), "branch passes must rewind"
    assert ensemble.all_halted()
    assert ensemble.exchanges >= 1, "exchange must have occurred"
    receipt = ensemble.to_receipt()
    assert receipt["n_branches"] == 3
    assert all(b["halt_reason"] for b in receipt["branches"])


def test_exchange_blends_comm_slot_only(tiny_model):
    cache = _prefill(tiny_model)
    ensemble, _, _ = _ensemble(tiny_model, cache, n_branches=2)
    before = [b.z for b in ensemble.branches]
    ensemble.exchange()
    for b, prev in zip(ensemble.branches, before):
        delta = mx.abs(b.z - prev)
        comm_delta = float(mx.max(delta[:, 0, :]))
        other_delta = float(mx.max(delta[:, 1:, :]))
        assert comm_delta > 0, "comm slot must receive consensus"
        assert other_delta == 0, "non-comm slots must be untouched by exchange"


def test_diversity_jitter_decorrelates_parallel_branches(tiny_model):
    cache = _prefill(tiny_model)
    ensemble, _, _ = _ensemble(tiny_model, cache, n_branches=2)
    a, b = ensemble.branches
    b.z = a.z  # force collapse
    b.workspace.update(b.z)
    ensemble.maintain_diversity()
    assert not bool(mx.allclose(a.z, b.z)), "collapsed branches must be jittered apart"


def test_selection_prefers_external_score_then_convergence(tiny_model):
    cache = _prefill(tiny_model)
    ensemble, runner, budget = _ensemble(tiny_model, cache)
    for _ in range(8):
        if ensemble.all_halted():
            break
        ensemble.step_all(runner, cache, P_END, C_START, budget=budget)
    winner = ensemble.select(score_fn=lambda br: 42.0 if br.index == 1 else 0.0)
    assert winner.index == 1
    by_convergence = ensemble.select()
    trails = {b.index: b.halting.residual_trail[-1] for b in ensemble.branches}
    assert by_convergence.index == min(trails, key=trails.get)


def test_equal_flop_accounting_scales_with_branches(tiny_model):
    cache1 = _prefill(tiny_model)
    e1, r1, b1 = _ensemble(tiny_model, cache1, n_branches=1)
    e1.step_all(r1, cache1, P_END, C_START, budget=b1)
    single = b1.spent_layer_apps

    cache3 = _prefill(tiny_model)
    e3, r3, b3 = _ensemble(tiny_model, cache3, n_branches=3)
    e3.step_all(r3, cache3, P_END, C_START, budget=b3)
    assert b3.spent_layer_apps == 3 * single, "K branches must cost exactly K× per step"
