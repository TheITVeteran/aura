"""Contract tests: layer-schedule programs + virtual-width branches."""

from __future__ import annotations

import copy
import hashlib
import json
import stat

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.cache import KVCache  # noqa: E402
from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

import core.brain.llm.latent_cortex.branches as branches_mod  # noqa: E402
import core.brain.llm.latent_cortex.recurrence as recurrence_mod  # noqa: E402
from core.brain.llm.latent_cortex.branches import BRANCH_ROLES, BranchEnsemble  # noqa: E402
from core.brain.llm.latent_cortex.escape import (  # noqa: E402
    BranchEscapeLadder,
    EscapeConfig,
)
from core.brain.llm.latent_cortex.recurrence import WindowRunner  # noqa: E402
from core.brain.llm.latent_cortex.schedules import (  # noqa: E402
    LayerSchedule,
    PairedScheduleOutcome,
    ScheduleComputeReceipt,
    ScheduleLibrary,
    ScheduleSearch,
    StageOp,
)
from core.brain.llm.latent_cortex.telemetry import LatentTelemetry  # noqa: E402
from core.brain.llm.latent_cortex.types import (  # noqa: E402
    BranchConfig,
    ComputeBudget,
    RecurrenceConfig,
    WorkspaceConfig,
)
from core.brain.llm.recurrent_depth import CacheSnapshotError  # noqa: E402

N_LAYERS, P_END, C_START = 8, 2, 6
PROMPT = [[5, 9, 17, 3, 42, 7, 11, 23, 2, 88]]


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _paired_outcome(
    schedule: LayerSchedule,
    domain: str,
    index: int,
    *,
    candidate_success: bool = True,
    default_success: bool = False,
    run_order: str | None = None,
    held_out: bool = True,
    contamination_scan_passed: bool = True,
    candidate_layer_apps: int = 1_000,
    default_layer_apps: int = 1_000,
    candidate_estimator: str = "estimator-v1",
    default_estimator: str = "estimator-v1",
    evaluator_build: str = "evaluator-v1",
) -> PairedScheduleOutcome:
    return PairedScheduleOutcome.create(
        schedule_hash=schedule.schedule_hash,
        domain=domain,
        task_id=f"task-{index}",
        task_commitment_sha256=_digest(f"task-commitment-{index}"),
        candidate_success=candidate_success,
        default_success=default_success,
        candidate_compute=ScheduleComputeReceipt(
            layer_apps=candidate_layer_apps,
            estimator_sha256=_digest(candidate_estimator),
        ),
        default_compute=ScheduleComputeReceipt(
            layer_apps=default_layer_apps,
            estimator_sha256=_digest(default_estimator),
        ),
        run_order=run_order or ("candidate_first" if index % 2 == 0 else "default_first"),
        held_out=held_out,
        contamination_scan_passed=contamination_scan_passed,
        scorer_receipt_sha256=_digest(f"scorer-{index}"),
        verifier_receipt_sha256=_digest(f"verifier-{index}"),
        evaluation_run_id=f"evaluation-run-{index // 10}",
        evaluator_build_sha256=_digest(evaluator_build),
        model_checkpoint_sha256=_digest("checkpoint-v1"),
        evidence_protocol_sha256=_digest("schedule-protocol-v1"),
    )


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
    with pytest.raises(ValueError):
        LayerSchedule.from_dict({"ops": [{"start": "2", "end": 6}]})
    with pytest.raises(ValueError):
        LayerSchedule.from_dict({"ops": [{"start": 2, "end": 6, "alpha": float("nan")}]})
    with pytest.raises(ValueError, match="finite number"):
        LayerSchedule.from_dict({"ops": [{"start": 2, "end": 6, "alpha": 10**10_000}]})
    assert LayerSchedule(ops=(StageOp(2, 6, 1, 10**10_000),)).validate(
        prelude_end=2,
        coda_start=6,
    )
    with pytest.raises(ValueError):
        LayerSchedule.from_dict({"ops": [], "typo": True})


def test_library_prefers_evidence_over_novelty(tmp_path):
    lib = ScheduleLibrary(tmp_path / "sched.json")
    default = LayerSchedule.single_window(2, 6, 4)
    exotic = LayerSchedule(ops=(StageOp(2, 4, 3), StageOp(4, 6, 3)), name="exotic")

    for index in range(3):
        lib.record_paired_outcome(exotic, "math", _paired_outcome(exotic, "math", index))
    best = lib.best_for_domain("math", prelude_end=2, coda_start=6, default_repeats=4)
    assert best.schedule_hash == default.schedule_hash

    for index in range(3, 40):
        lib.record_paired_outcome(exotic, "math", _paired_outcome(exotic, "math", index))
    best = lib.best_for_domain("math", prelude_end=2, coda_start=6, default_repeats=4)
    assert best.schedule_hash == exotic.schedule_hash


def test_library_rejects_unpaired_replay_and_unbalanced_evidence(tmp_path):
    lib = ScheduleLibrary(tmp_path / "sched.json")
    candidate = LayerSchedule(ops=(StageOp(2, 5, 3),), name="candidate")
    assert not hasattr(lib, "record_outcome")

    first = _paired_outcome(candidate, "math", 0, run_order="candidate_first")
    lib.record_paired_outcome(candidate, "math", first)
    with pytest.raises(ValueError, match="duplicate or conflicting"):
        lib.record_paired_outcome(candidate, "math", first)
    for index in range(1, 20):
        lib.record_paired_outcome(
            candidate,
            "math",
            _paired_outcome(candidate, "math", index, run_order="candidate_first"),
        )
    best = lib.best_for_domain("math", prelude_end=2, coda_start=6, default_repeats=4)
    assert best.schedule_hash == LayerSchedule.single_window(2, 6, 4).schedule_hash


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"held_out": False}, "held out"),
        ({"contamination_scan_passed": False}, "contamination"),
        ({"run_order": "unknown"}, "run_order"),
        ({"candidate_estimator": "candidate", "default_estimator": "default"}, "estimators"),
        ({"candidate_layer_apps": 1_200, "default_layer_apps": 1_000}, "compute matching"),
    ],
)
def test_paired_schedule_evidence_rejects_invalid_trials(overrides, match):
    candidate = LayerSchedule(ops=(StageOp(2, 5, 3),), name="candidate")
    with pytest.raises(ValueError, match=match):
        _paired_outcome(candidate, "math", 0, **overrides)


def test_schedule_compute_receipt_rejects_unbounded_integer():
    with pytest.raises(ValueError, match="bounded positive integer"):
        ScheduleComputeReceipt(
            layer_apps=10**10_000,
            estimator_sha256=_digest("estimator"),
        )


def test_library_rejects_profile_drift_and_commitment_replay(tmp_path):
    lib = ScheduleLibrary(tmp_path / "sched.json")
    candidate = LayerSchedule(ops=(StageOp(2, 5, 3),), name="candidate")
    lib.record_paired_outcome(candidate, "math", _paired_outcome(candidate, "math", 0))
    with pytest.raises(ValueError, match="profile changed"):
        lib.record_paired_outcome(
            candidate,
            "math",
            _paired_outcome(candidate, "math", 1, evaluator_build="evaluator-v2"),
        )

    replay = _paired_outcome(candidate, "math", 2)
    replay_payload = replay.to_dict()
    replay_payload["task_commitment_sha256"] = _digest("task-commitment-0")
    replay_payload["evidence_binding_sha256"] = PairedScheduleOutcome.binding_sha256(
        schedule_hash=candidate.schedule_hash,
        domain="math",
        values={
            key: value for key, value in replay_payload.items() if key != "evidence_binding_sha256"
        },
    )
    replay = PairedScheduleOutcome.from_dict(
        replay_payload,
        schedule_hash=candidate.schedule_hash,
        domain="math",
    )
    with pytest.raises(ValueError, match="task commitment"):
        lib.record_paired_outcome(candidate, "math", replay)


def test_library_ignores_records_invalid_for_current_topology(tmp_path):
    lib = ScheduleLibrary(tmp_path / "sched.json")
    stale = LayerSchedule(ops=(StageOp(2, 30, 4),), name="from-bigger-model")
    for index in range(20):
        lib.record_paired_outcome(stale, "math", _paired_outcome(stale, "math", index))
    best = lib.best_for_domain("math", prelude_end=2, coda_start=6, default_repeats=4)
    assert best.schedule_hash == LayerSchedule.single_window(2, 6, 4).schedule_hash


def test_library_persistence_round_trip(tmp_path):
    path = tmp_path / "sched.json"
    lib = ScheduleLibrary(path)
    s = LayerSchedule(ops=(StageOp(2, 5, 2),), name="s")
    for index in range(25):
        lib.record_paired_outcome(s, "code", _paired_outcome(s, "code", index))
    assert lib.save() is True
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    reloaded = ScheduleLibrary(path)
    best = reloaded.best_for_domain("code", prelude_end=2, coda_start=6, default_repeats=4)
    assert best.schedule_hash == s.schedule_hash
    assert reloaded.status() == {
        "records": 1,
        "observations": 25,
        "domains": {"code": 1},
        "revision": 1,
    }


@pytest.mark.parametrize(
    "payload",
    [
        [],
        "not-a-library",
        {"version": 2, "revision": 0, "records": {}},
        {"version": 2, "revision": 0, "records": [{"schedule": {}}]},
    ],
)
def test_library_malformed_roots_fail_closed_without_crashing(tmp_path, payload):
    path = tmp_path / "sched.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    lib = ScheduleLibrary(path)
    assert lib.status() == {
        "records": 0,
        "observations": 0,
        "domains": {},
        "revision": 0,
    }
    assert lib.save() is False


def test_stale_schedule_writer_merges_without_overwriting_newer_evidence(tmp_path):
    path = tmp_path / "sched.json"
    candidate = LayerSchedule(ops=(StageOp(2, 5, 2),), name="candidate")
    first = ScheduleLibrary(path)
    delayed = ScheduleLibrary(path)
    first.record_paired_outcome(candidate, "code", _paired_outcome(candidate, "code", 0))
    delayed.record_paired_outcome(candidate, "code", _paired_outcome(candidate, "code", 1))

    assert first.save() is True
    assert delayed.save() is True

    reloaded = ScheduleLibrary(path)
    assert reloaded.status()["observations"] == 2
    assert reloaded.status()["revision"] == 2


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


def _ensemble(
    model,
    cache,
    n_branches=3,
    budget=None,
    exchange_interval=2,
    *,
    max_steps=8,
    fixed_depth=False,
    isolation_steps=1,
):
    inner = model.model
    emb = inner.embed_tokens(mx.array(PROMPT))
    budget = budget or ComputeBudget()
    runner = WindowRunner(inner, budget)
    ensemble = BranchEnsemble.seed(
        emb,
        WorkspaceConfig(n_slots=4, seed=3),
        BranchConfig(
            n_branches=n_branches,
            isolation_steps=isolation_steps,
            exchange_interval=exchange_interval,
        ),
        RecurrenceConfig(
            max_steps=max_steps,
            convergence_eps=0.02,
            min_steps=min(2, max_steps),
            fixed_depth=fixed_depth,
        ),
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
    assert ensemble.exchange_now(sync_kind="test", sync_id="before-candidates") is False, (
        "peer exposure must wait for candidates"
    )


def test_fresh_context_candidates_seal_before_first_exchange(tiny_model):
    cache = _prefill(tiny_model)
    ensemble, runner, budget = _ensemble(
        tiny_model,
        cache,
        n_branches=3,
        exchange_interval=1,
        isolation_steps=2,
    )
    original_second = ensemble.branches[1].z
    ensemble.branches[0].z = ensemble.branches[0].z * 1.01
    ensemble.branches[0].workspace.update(ensemble.branches[0].z)
    assert bool(mx.allclose(ensemble.branches[1].z, original_second))

    assert ensemble.step_all(runner, cache, P_END, C_START, budget=budget)
    assert ensemble.exchanges == 0
    assert ensemble.step_all(runner, cache, P_END, C_START, budget=budget)
    assert ensemble.exchanges == 1

    isolation = ensemble.isolation_receipt(runner.cache_discipline_receipt())
    assert isolation["certified"] is True
    assert isolation["first_exchange_step"] == 2
    assert isolation["blocked_cross_exposures"] >= 1
    assert len({row["candidate_sha256"] for row in isolation["candidates"]}) == 3
    assert len({row["rng_stream_sha256"] for row in isolation["candidates"]}) == 3
    assert len({row["context_sha256"] for row in isolation["candidates"]}) == 1
    assert isolation["cache_discipline"]["all_restored"] is True


def test_window_runner_fails_when_cache_restore_postcondition_is_false(tiny_model, monkeypatch):
    cache = _prefill(tiny_model)
    budget = ComputeBudget()
    runner = WindowRunner(tiny_model.model, budget)
    slots = mx.zeros((1, 4, 64))
    monkeypatch.setattr(
        recurrence_mod,
        "_restore_recurrent_caches",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(CacheSnapshotError, match="restore postcondition"):
        runner.run(slots, cache, P_END, C_START, persist=False)
    discipline = runner.cache_discipline_receipt()
    assert discipline == {
        "schema": "aura.rlc.cache_discipline.v1",
        "nonpersistent_calls": 1,
        "restored_calls": 0,
        "restore_failures": 1,
        "all_restored": False,
    }


def test_role_lesion_runs_but_cannot_claim_independent_candidates(tiny_model):
    cache = _prefill(tiny_model)
    inner = tiny_model.model
    embeddings = inner.embed_tokens(mx.array(PROMPT))
    budget = ComputeBudget()
    runner = WindowRunner(inner, budget)
    ensemble = BranchEnsemble.seed(
        embeddings,
        WorkspaceConfig(n_slots=4, seed=3),
        BranchConfig(
            n_branches=2,
            isolation_steps=1,
            roles=("analogy", "analogy"),
        ),
        RecurrenceConfig(max_steps=4, min_steps=1),
        runner,
        cache,
        P_END,
    )
    assert ensemble.step_all(runner, cache, P_END, C_START, budget=budget)
    isolation = ensemble.isolation_receipt(runner.cache_discipline_receipt())
    assert isolation["sealed"] is True
    assert isolation["configured_role_lesion"] is True
    assert isolation["certified"] is False
    assert isolation["reason"] == "configured_role_lesion"


def test_role_swap_moves_executable_labor_and_restoration_recovers(tiny_model):
    roles = ("constructive_solution", "counterexample_search")

    def seeded(assignment):
        cache = _prefill(tiny_model)
        inner = tiny_model.model
        embeddings = inner.embed_tokens(mx.array(PROMPT))
        budget = ComputeBudget()
        runner = WindowRunner(inner, budget)
        ensemble = BranchEnsemble.seed(
            embeddings,
            WorkspaceConfig(n_slots=4, seed=3),
            BranchConfig(
                n_branches=2,
                isolation_steps=1,
                roles=assignment,
            ),
            RecurrenceConfig(max_steps=2, min_steps=1),
            runner,
            cache,
            P_END,
        )
        receipts = ensemble.apply_cognitive_operators(
            mx.ones((1, 1, 64)),
            action="falsify",
            action_step=0,
        )
        return ensemble, receipts

    distinct, distinct_receipts = seeded(roles)
    swapped, swapped_receipts = seeded(tuple(reversed(roles)))
    lesioned, lesioned_receipts = seeded((roles[0], roles[0]))
    restored, restored_receipts = seeded(roles)

    assert [row["role"] for row in distinct_receipts] == list(roles)
    assert [row["operator"] for row in distinct_receipts] == [
        "constructive_solution",
        "counterexample",
    ]
    assert [row["role"] for row in swapped_receipts] == list(reversed(roles))
    assert [row["operator"] for row in swapped_receipts] == [
        "counterexample",
        "constructive_solution",
    ]
    assert [row["operator"] for row in lesioned_receipts] == [
        "constructive_solution",
        "constructive_solution",
    ]
    assert lesioned._configured_role_lesion is True
    assert [branch.seed_sha256 for branch in restored.branches] == [
        branch.seed_sha256 for branch in distinct.branches
    ]
    assert [row["output_sha256"] for row in restored_receipts] == [
        row["output_sha256"] for row in distinct_receipts
    ]
    assert [branch.seed_sha256 for branch in swapped.branches] != [
        branch.seed_sha256 for branch in distinct.branches
    ]


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


def test_branch_round_restores_every_branch_when_later_branch_fails(
    tiny_model,
    monkeypatch,
):
    cache = _prefill(tiny_model)
    prompt_len = len(PROMPT[0])
    ensemble, runner, budget = _ensemble(tiny_model, cache, n_branches=2)
    before = [
        {
            "z": branch.z,
            "steps": branch.steps,
            "halting": branch.halting.snapshot(),
            "grounding": list(branch.recurrent_grounding_trace),
            "stability": list(branch.loop_stability_trace),
            "acceptance": list(branch.update_acceptance_trace),
            "reflector": list(branch.reflector_trace),
        }
        for branch in ensemble.branches
    ]
    original = branches_mod.recurrence_step
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("forced second branch failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(branches_mod, "recurrence_step", fail_second)
    with pytest.raises(RuntimeError, match="forced second branch failure"):
        ensemble.step_all(
            runner,
            cache,
            P_END,
            C_START,
            budget=budget,
        )

    assert budget.spent_layer_apps > 0
    assert all(item.offset == prompt_len for item in cache)
    for branch, prior in zip(ensemble.branches, before, strict=True):
        assert bool(mx.array_equal(branch.z, prior["z"]))
        assert branch.steps == prior["steps"]
        assert branch.halting.snapshot() == prior["halting"]
        assert branch.recurrent_grounding_trace == prior["grounding"]
        assert branch.loop_stability_trace == prior["stability"]
        assert branch.update_acceptance_trace == prior["acceptance"]
        assert branch.reflector_trace == prior["reflector"]


def test_branch_round_restores_exchange_isolation_diversity_and_telemetry(
    tiny_model,
    monkeypatch,
):
    cache = _prefill(tiny_model)
    prompt_len = len(PROMPT[0])
    ensemble, runner, budget = _ensemble(
        tiny_model,
        cache,
        n_branches=2,
        exchange_interval=1,
        isolation_steps=1,
    )
    ensemble.config.collapse_cos_threshold = -1.0
    ensemble.telemetry = LatentTelemetry()
    before = ensemble.snapshot_ensemble_runtime()
    original_charge = budget.charge_tensor_work

    def fail_after_diversity_mutation(operation, **counters):
        if operation == "branch_diversity_jitter":
            raise RuntimeError("forced post-exchange diversity failure")
        return original_charge(operation, **counters)

    monkeypatch.setattr(budget, "charge_tensor_work", fail_after_diversity_mutation)
    with pytest.raises(RuntimeError, match="post-exchange diversity failure"):
        ensemble.step_all(
            runner,
            cache,
            P_END,
            C_START,
            budget=budget,
        )

    assert budget.spent_layer_apps > 0
    assert all(item.offset == prompt_len for item in cache)
    assert ensemble.exchanges == before["exchanges"]
    assert len(ensemble.exchange_receipts) == before["exchange_receipt_length"]
    assert ensemble._exchange_sync_points == before["exchange_sync_points"]
    assert ensemble._isolation_sealed is before["isolation_sealed"]
    assert ensemble._isolation_failure == before["isolation_failure"]
    assert ensemble._blocked_cross_exposures == before["blocked_cross_exposures"]
    assert ensemble._cross_exposure_started is before["cross_exposure_started"]
    assert ensemble._first_exchange_step == before["first_exchange_step"]
    assert vars(ensemble.telemetry) == before["telemetry_state"]
    for branch in ensemble.branches:
        prior = before["branches"][branch.index]
        assert bool(mx.array_equal(branch.z, prior["z"]))
        assert branch.steps == prior["steps"]
        assert branch.candidate_sha256 == prior["candidate_sha256"]
        assert branch.candidate_step == prior["candidate_step"]


def test_cognitive_operator_primitives_mutate_only_live_bounded_state(tiny_model):
    cache = _prefill(tiny_model)
    ensemble, runner, budget = _ensemble(tiny_model, cache, n_branches=2)
    before = [branch.z for branch in ensemble.branches]
    control = mx.ones((1, 1, int(before[0].shape[-1])))

    assert ensemble.inject_control(control) == 2
    for prior, branch in zip(before, ensemble.branches, strict=True):
        slot = ensemble.config.comm_slot
        assert bool(mx.allclose(branch.z[:, :slot, :], prior[:, :slot, :]))
        assert bool(mx.allclose(branch.z[:, slot + 1 :, :], prior[:, slot + 1 :, :]))
        assert not bool(mx.allclose(branch.z[:, slot : slot + 1, :], prior[:, slot : slot + 1, :]))

    disagreement = ensemble.disagreement()
    assert 0.0 <= disagreement <= 1.0
    assert ensemble.compress_state() == 0, "aggregation cannot precede isolation"
    assert ensemble.step_all(runner, cache, P_END, C_START, budget=budget)
    before_compression = [branch.z for branch in ensemble.branches]
    assert ensemble.compress_state() == 2
    assert any(
        not bool(mx.allclose(prior, branch.z))
        for prior, branch in zip(before_compression, ensemble.branches, strict=True)
    )
    assert ensemble.halt_all("value_controller_answer") == 2
    assert ensemble.all_halted()
    assert all(
        branch.halt_reason.startswith("value_controller_answer") for branch in ensemble.branches
    )


def test_step_score_observes_candidate_state_not_committed_predecessor(tiny_model):
    cache = _prefill(tiny_model)
    ensemble, runner, budget = _ensemble(tiny_model, cache, n_branches=1)
    branch = ensemble.branches[0]
    predecessor = branch.z
    observed = {}

    def score(candidate):
        observed["z"] = candidate.z
        observed["steps"] = candidate.steps
        return 7.25

    assert ensemble.step_all(
        runner,
        cache,
        P_END,
        C_START,
        budget=budget,
        score_fn=score,
    )

    assert not bool(mx.allclose(observed["z"], predecessor))
    assert observed["steps"] == 1
    assert branch.steps == 1
    assert branch.halting.score_trail == [pytest.approx(7.25)]
    assert bool(mx.allclose(branch.halting.best_state, observed["z"]))


def test_branch_savepoint_restores_full_state_machine(tiny_model):
    cache = _prefill(tiny_model)
    ensemble, runner, budget = _ensemble(tiny_model, cache, n_branches=1)
    branch = ensemble.branches[0]
    branch.escape = BranchEscapeLadder(EscapeConfig(), branch.index)

    assert ensemble.step_all(runner, cache, P_END, C_START, budget=budget)
    assert ensemble.savepoint_all() == 1
    saved_z = branch.z
    saved_role = branch.role
    saved_steps = branch.steps
    saved_residuals = list(branch.halting.residual_trail)
    saved_scores = list(branch.halting.score_trail)
    saved_best_step = branch.halting.best_step
    saved_best_score = branch.halting.best_score
    saved_best_state = branch.halting.best_state

    assert ensemble.step_all(runner, cache, P_END, C_START, budget=budget)
    branch.halting.head_halts = 4
    assert branch.escape.on_divergence(branch, "diverged_norm") == "escaped"
    branch.score = 99.0
    branch.halted = True
    branch.halt_reason = "after_savepoint"

    assert ensemble.revert_branch_to_savepoint(branch) is True
    assert bool(mx.allclose(branch.z, saved_z))
    assert branch.workspace.z is branch.z
    assert branch.role == saved_role
    assert branch.steps == saved_steps
    assert branch.score == 0.0
    assert branch.halted is False
    assert branch.halt_reason == ""
    assert branch.halting.residual_trail == saved_residuals
    assert branch.halting.score_trail == saved_scores
    assert branch.halting.best_step == saved_best_step
    assert branch.halting.best_score == pytest.approx(saved_best_score)
    assert branch.halting.best_state is saved_best_state
    assert branch.halting.head_halts == 0
    assert branch.escape.to_receipt() == {
        "attempts": [],
        "rungs_used": [],
        "on_probation": False,
    }


def test_fixed_depth_performs_terminal_exchange_before_halting(tiny_model):
    cache = _prefill(tiny_model)
    ensemble, runner, budget = _ensemble(
        tiny_model,
        cache,
        n_branches=2,
        exchange_interval=1,
        max_steps=2,
        fixed_depth=True,
    )

    assert ensemble.step_all(runner, cache, P_END, C_START, budget=budget)
    assert ensemble.exchanges == 1
    assert not ensemble.all_halted()
    assert ensemble.step_all(runner, cache, P_END, C_START, budget=budget)
    assert ensemble.exchanges == 2
    assert ensemble.all_halted()


def test_interval_exchange_waits_for_resynchronization_and_ignores_other_exchange_kinds(
    tiny_model,
):
    cache = _prefill(tiny_model)
    ensemble, runner, budget = _ensemble(
        tiny_model,
        cache,
        n_branches=2,
        exchange_interval=2,
        max_steps=6,
        fixed_depth=True,
        isolation_steps=1,
    )

    assert ensemble.step_all(runner, cache, P_END, C_START, budget=budget)
    assert ensemble.exchange_now(
        sync_kind="schedule_bytecode",
        sync_id="mixed-exchange-before-interval",
        budget=budget,
    )

    first, second = ensemble.branches
    second.halted = True
    assert ensemble.step_all(runner, cache, P_END, C_START, budget=budget)
    second.halted = False
    first.halted = True
    assert ensemble.step_all(runner, cache, P_END, C_START, budget=budget)
    first.halted = False
    assert {branch.steps for branch in ensemble.active()} == {2}
    assert [row["sync_kind"] for row in ensemble.exchange_receipts] == ["schedule_bytecode"]

    assert ensemble.step_all(runner, cache, P_END, C_START, budget=budget)
    assert ensemble.step_all(runner, cache, P_END, C_START, budget=budget)
    interval_rows = [row for row in ensemble.exchange_receipts if row["sync_kind"] == "interval"]
    assert [row["sync_id"] for row in interval_rows] == ["recurrent-step:4"]


def test_loop_diagnostics_reset_derivatives_after_exchange(tiny_model):
    continuous_cache = _prefill(tiny_model)
    continuous, runner, budget = _ensemble(
        tiny_model,
        continuous_cache,
        n_branches=1,
        exchange_interval=8,
        max_steps=2,
        fixed_depth=True,
    )
    assert continuous.step_all(runner, continuous_cache, P_END, C_START, budget=budget)
    assert continuous.step_all(runner, continuous_cache, P_END, C_START, budget=budget)
    assert [
        row["continuous_from_previous"] for row in continuous.branches[0].loop_stability_trace
    ] == [False, True]

    exchanged_cache = _prefill(tiny_model)
    exchanged, runner, budget = _ensemble(
        tiny_model,
        exchanged_cache,
        n_branches=2,
        exchange_interval=1,
        max_steps=2,
        fixed_depth=True,
    )
    assert exchanged.step_all(runner, exchanged_cache, P_END, C_START, budget=budget)
    assert exchanged.step_all(runner, exchanged_cache, P_END, C_START, budget=budget)
    for branch in exchanged.branches:
        trail = branch.loop_stability_trace
        assert [row["continuous_from_previous"] for row in trail] == [False, False]
        assert trail[1]["contraction_ratio"] is None
        assert trail[1]["delta_cosine"] is None


def test_exchange_blends_comm_slot_only(tiny_model):
    cache = _prefill(tiny_model)
    ensemble, runner, budget = _ensemble(tiny_model, cache, n_branches=2)
    assert ensemble.step_all(runner, cache, P_END, C_START, budget=budget)
    before = [b.z for b in ensemble.branches]
    assert ensemble.exchange(sync_kind="test", sync_id="comm-slot-only") is True
    for b, prev in zip(ensemble.branches, before, strict=True):
        delta = mx.abs(b.z - prev)
        comm_delta = float(mx.max(delta[:, 0, :]))
        other_delta = float(mx.max(delta[:, 1:, :]))
        assert comm_delta > 0, "comm slot must receive consensus"
        assert other_delta == 0, "non-comm slots must be untouched by exchange"


def test_exchange_consensus_is_causally_discounted_by_support_weights(tiny_model):
    def prepared():
        cache = _prefill(tiny_model)
        ensemble, runner, budget = _ensemble(
            tiny_model,
            cache,
            n_branches=3,
            isolation_steps=1,
            exchange_interval=8,
        )
        assert ensemble.step_all(runner, cache, P_END, C_START, budget=budget)
        for index, branch in enumerate(ensemble.branches, start=1):
            branch.z = mx.ones_like(branch.z) * float(index)
            branch.workspace.update(branch.z)
        return ensemble

    equal = prepared()
    discounted = prepared()
    discounted.set_support_weights({0: 1.0, 1: 0.2, 2: 0.2})

    assert equal.exchange(sync_kind="test", sync_id="equal-support") is True
    assert discounted.exchange(sync_kind="test", sync_id="discounted-support") is True

    assert not bool(mx.allclose(equal.branches[0].z, discounted.branches[0].z)), (
        "correlation weights must change the exchanged neural state"
    )


def test_exchange_trace_excludes_mailbox_and_context_and_marks_generations(
    tiny_model,
):
    from core.brain.llm.latent_cortex.branch_exchange import (
        build_branch_exchange_trace,
        validate_branch_exchange_trace,
    )

    cache = _prefill(tiny_model)
    inner = tiny_model.model
    embeddings = inner.embed_tokens(mx.array(PROMPT))
    budget = ComputeBudget()
    runner = WindowRunner(inner, budget)
    ensemble = BranchEnsemble.seed(
        embeddings,
        WorkspaceConfig(n_slots=4, seed=3),
        BranchConfig(n_branches=2, isolation_steps=1, exchange_interval=1),
        RecurrenceConfig(max_steps=4, min_steps=1, fixed_depth=True),
        runner,
        cache,
        P_END,
        context_seeds=[("world_model", mx.ones((64,)))],
    )

    assert ensemble.step_all(runner, cache, P_END, C_START, budget=budget)
    assert ensemble.step_all(runner, cache, P_END, C_START, budget=budget)
    isolation = ensemble.isolation_receipt(runner.cache_discipline_receipt())
    trace = build_branch_exchange_trace(
        exchanges=ensemble.exchange_receipts,
        n_branches=2,
        n_slots=4,
        comm_slot=0,
        exchange_gamma=0.35,
        branch_isolation=isolation,
        cognitive_slots=[{"slot": 1}],
        exchange_interval=1,
        schedule_hash="test-schedule",
        bytecode_events=[],
        cognitive_action_trace=[],
    )

    assert trace["exchange_count"] == 2
    assert trace["independent_support_generations"] == 1
    first, second = trace["exchanges"]
    assert first["generation"] == "independent_candidates"
    assert first["counts_as_independent_support"] is True
    assert first["prior_peer_context_possible"] is False
    assert second["generation"] == "cooperative_refinement"
    assert second["counts_as_independent_support"] is False
    assert second["prior_peer_context_possible"] is True
    assert all(row["source_slots"] == [2, 3] for row in first["source_rows"])
    assert all(row["excluded_slots"] == [0, 1] for row in first["source_rows"])
    assert first["first_answer_text_exposed"] is False
    assert first["message_representation"] == "latent_tensor_only"
    assert all(
        row["non_comm_pre_sha256"] == row["non_comm_post_sha256"] for row in first["recipient_rows"]
    )
    validate_branch_exchange_trace(
        trace,
        exchange_count=2,
        n_branches=2,
        n_slots=4,
        comm_slot=0,
        exchange_gamma=0.35,
        branch_isolation=isolation,
        cognitive_slots=[{"slot": 1}],
        exchange_interval=1,
        schedule_hash="test-schedule",
        bytecode_events=[],
        cognitive_action_trace=[],
    )

    tampered = copy.deepcopy(trace)
    tampered["exchanges"][0]["source_rows"][0]["source_slots"] = [0, 1]
    with pytest.raises(ValueError, match="source provenance"):
        validate_branch_exchange_trace(
            tampered,
            exchange_count=2,
            n_branches=2,
            n_slots=4,
            comm_slot=0,
            exchange_gamma=0.35,
            branch_isolation=isolation,
            cognitive_slots=[{"slot": 1}],
            exchange_interval=1,
            schedule_hash="test-schedule",
            bytecode_events=[],
            cognitive_action_trace=[],
        )


def test_exchange_rejects_replayed_sync_point(tiny_model):
    cache = _prefill(tiny_model)
    ensemble, runner, budget = _ensemble(
        tiny_model,
        cache,
        n_branches=2,
        exchange_interval=8,
    )
    assert ensemble.step_all(runner, cache, P_END, C_START, budget=budget)
    assert ensemble.exchange(sync_kind="test", sync_id="single-use") is True
    with pytest.raises(ValueError, match="already consumed"):
        ensemble.exchange(sync_kind="test", sync_id="single-use")


def test_diversity_jitter_decorrelates_parallel_branches(tiny_model):
    cache = _prefill(tiny_model)
    ensemble, runner, budget = _ensemble(tiny_model, cache, n_branches=2)
    assert ensemble.step_all(runner, cache, P_END, C_START, budget=budget)
    a, b = ensemble.branches
    b.z = a.z  # force collapse
    b.workspace.update(b.z)
    assert ensemble.maintain_diversity() is True
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


def test_branch_round_is_all_or_none_under_budget(tiny_model):
    cache = _prefill(tiny_model)
    ensemble, runner, _ = _ensemble(tiny_model, cache, n_branches=3)
    one_branch_cost = 4 * (C_START - P_END)
    budget = ComputeBudget(max_layer_apps=2 * one_branch_cost)
    admitted = ensemble.step_all(
        runner,
        cache,
        P_END,
        C_START,
        budget=budget,
    )
    assert admitted is False
    assert budget.spent_layer_apps == 0
    assert all(branch.steps == 0 for branch in ensemble.branches)


# ── Neural bytecode: typed non-window instructions ───────────────────────


def test_bytecode_ops_parse_validate_and_hash_stably():
    from core.brain.llm.latent_cortex.schedules import LayerSchedule, StageOp

    program = LayerSchedule.from_dict(
        {
            "name": "probe-guided",
            "ops": [
                {"start": 2, "end": 6, "repeats": 2},
                {"kind": "savepoint"},
                {"kind": "exchange"},
                {"start": 2, "end": 6, "repeats": 2, "alpha": 0.5},
                {"kind": "verify_probe", "revert_on_drop": True},
            ],
        }
    )
    assert program.validate(prelude_end=2, coda_start=6) == []
    assert program.total_layer_repeats == 16  # bytecode ops spend no layers
    # Window-only serialization is unchanged ⇒ legacy hashes survive.
    legacy = LayerSchedule(ops=(StageOp(2, 6, 2),))
    assert '"kind"' not in legacy.canonical_json()
    # Bytecode ops are covered by the hash.
    without_probe = LayerSchedule.from_dict(
        {
            "ops": [
                {"start": 2, "end": 6, "repeats": 2},
                {"kind": "savepoint"},
                {"kind": "exchange"},
                {"start": 2, "end": 6, "repeats": 2, "alpha": 0.5},
            ]
        }
    )
    assert program.schedule_hash != without_probe.schedule_hash


def test_bytecode_validation_rejects_malformed_programs():
    from core.brain.llm.latent_cortex.schedules import LayerSchedule

    with pytest.raises(ValueError, match="kind must be one of"):
        LayerSchedule.from_dict({"ops": [{"kind": "teleport"}]})
    with pytest.raises(ValueError, match="revert_on_drop only applies"):
        LayerSchedule.from_dict({"ops": [{"kind": "exchange", "revert_on_drop": True}]})
    with pytest.raises(ValueError, match="unknown keys"):
        LayerSchedule.from_dict({"ops": [{"kind": "savepoint", "start": 2}]})
    # revert_on_drop without a preceding savepoint is invalid.
    naked = LayerSchedule.from_dict(
        {
            "ops": [
                {"start": 2, "end": 6},
                {"kind": "verify_probe", "revert_on_drop": True},
            ]
        }
    )
    problems = naked.validate(prelude_end=2, coda_start=6)
    assert any("preceding savepoint" in p for p in problems)
    # A program of only bytecode ops computes nothing.
    inert = LayerSchedule.from_dict({"ops": [{"kind": "savepoint"}, {"kind": "exchange"}]})
    assert any("at least one window op" in p for p in inert.validate(prelude_end=2, coda_start=6))
    # Probe budget is bounded.
    flood = LayerSchedule.from_dict(
        {"ops": [{"start": 2, "end": 6}] + [{"kind": "verify_probe"} for _ in range(5)]}
    )
    assert any("probe budget" in p for p in flood.validate(prelude_end=2, coda_start=6))


def test_bytecode_program_executes_and_traces(tiny_model):
    from core.brain.llm.latent_cortex.engine import LatentCortexEngine
    from core.brain.llm.latent_cortex.task_verifiers import check_arithmetic_claims
    from core.brain.llm.latent_cortex.types import (
        BranchConfig,
        ComputeBudget,
        CortexConfig,
        RecurrenceConfig,
        WorkspaceConfig,
    )

    scores = iter([0.8, 0.2])  # second probe drops ⇒ backtrack

    def verifier(text: str) -> float:
        if text.startswith("Independent consistency check:"):
            return float(check_arithmetic_claims(text)["score"])
        return next(scores, 0.2)

    engine = LatentCortexEngine(
        tiny_model,
        _ProbeTokenizer(),
        config=CortexConfig(
            workspace=WorkspaceConfig(n_slots=4, seed=7),
            recurrence=RecurrenceConfig(max_steps=8, min_steps=1, convergence_eps=1e-9),
            branches=BranchConfig(n_branches=2),
            decode_max_tokens=4,
            schedule={
                "name": "probe-backtrack",
                "ops": [
                    {"start": 2, "end": 6, "repeats": 1},
                    {"kind": "savepoint"},
                    {"kind": "verify_probe", "revert_on_drop": True},
                    {"kind": "exchange"},
                    {"start": 2, "end": 6, "repeats": 1},
                    {"kind": "verify_probe", "revert_on_drop": True},
                ],
            },
        ),
    )
    result = engine.reason(token_ids=[5, 9, 17, 3, 42], budget=ComputeBudget(), verifier=verifier)
    assert result.ok
    events = result.receipt.bytecode_events
    kinds = [event["kind"] for event in events]
    assert kinds == ["savepoint", "verify_probe", "exchange", "verify_probe"]
    assert events[0]["branches"] == 2
    assert events[1]["ran"] is True and events[1]["score"] == 0.8
    assert events[2]["done"] is True
    assert events[3]["ran"] is True and events[3]["score"] == 0.2
    assert events[3]["branch"] == events[1]["branch"]
    assert events[3]["previous_score"] == 0.8
    assert events[3]["reverted_branches"] == 1
    assert "bytecode_probe_reverted" in result.receipt.honest_flags
    assert result.receipt.to_dict()["bytecode_events"] == events


class _ProbeTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 128 for c in text][:16]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)
