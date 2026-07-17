"""Contract tests: the learned per-problem execution controller.

Evidence-gated by construction: base allocation until Wilson separation on
graded verified outcomes, sparse exploration, bounded arm deltas, durable
ledger with corrupt-line tolerance, and a kill switch tests default to.
"""

from core.brain.llm.latent_cortex.execution_controller import (
    ARMS,
    ExecutionController,
    context_bucket,
    controller_enabled,
)


def _controller(tmp_path):
    return ExecutionController(root=tmp_path / "controller")


def test_context_bucket_is_coarse_and_deterministic():
    a = context_bucket(
        "Compare eager and lazy loading and choose one.", "general", 0.8, 0.5
    )
    b = context_bucket(
        "Compare eager and lazy loading and choose one.", "general", 0.8, 0.5
    )
    assert a == b
    assert "s:high" in a and "u:mid" in a
    assert "compare" in a and "select" in a
    # Different stakes band ⇒ different bucket; raw values never leak in.
    c = context_bucket(
        "Compare eager and lazy loading and choose one.", "general", 0.2, 0.5
    )
    assert c != a and "0.8" not in a


def test_observe_mode_until_evidence_exists(tmp_path):
    controller = _controller(tmp_path)
    decision = controller.choose(
        objective="why is the cache cold", domain="general", stakes=0.5, uncertainty=0.5
    )
    assert decision["arm"] == "base"
    assert decision["mode"] == "observe"


def test_exploration_is_sparse_and_round_robin(tmp_path):
    controller = _controller(tmp_path)
    modes = []
    for _ in range(8):
        decision = controller.choose(
            objective="q", domain="general", stakes=0.5, uncertainty=0.5
        )
        modes.append(decision["mode"])
    assert modes.count("explore") == 2  # every EXPLORE_EVERY=4 episodes
    assert modes.count("observe") == 6


def test_exploitation_requires_wilson_separation(tmp_path):
    controller = _controller(tmp_path)
    bucket = context_bucket("q", "general", 0.5, 0.5)
    # 12 mediocre base episodes, 12 strong deeper-recurrence episodes.
    for _ in range(12):
        controller.record_outcome(
            bucket=bucket, arm="base", verified_score=0.2, success=True
        )
        controller.record_outcome(
            bucket=bucket,
            arm="deeper_recurrence",
            verified_score=1.0,
            success=True,
        )
    decision = controller.choose(
        objective="q", domain="general", stakes=0.5, uncertainty=0.5
    )
    assert decision["arm"] == "deeper_recurrence"
    assert decision["mode"] == "exploit"
    assert decision["evidence"]["arm_lb"] > decision["evidence"]["base_ub"]


def test_no_exploitation_on_underpowered_or_overlapping_evidence(tmp_path):
    controller = _controller(tmp_path)
    bucket = context_bucket("q", "general", 0.5, 0.5)
    for _ in range(11):  # one short of MIN_TRIALS
        controller.record_outcome(
            bucket=bucket, arm="base", verified_score=0.2, success=True
        )
        controller.record_outcome(
            bucket=bucket, arm="wider_branches", verified_score=1.0, success=True
        )
    decision = controller.choose(
        objective="q", domain="general", stakes=0.5, uncertainty=0.5
    )
    assert decision["mode"] != "exploit"
    overlapping = _controller(tmp_path / "b")
    for _ in range(20):
        overlapping.record_outcome(
            bucket=bucket, arm="base", verified_score=0.6, success=True
        )
        overlapping.record_outcome(
            bucket=bucket, arm="wider_branches", verified_score=0.65, success=True
        )
    decision = overlapping.choose(
        objective="q", domain="general", stakes=0.5, uncertainty=0.5
    )
    assert decision["mode"] != "exploit"  # intervals overlap: not separated


def test_ledger_persists_and_tolerates_corruption(tmp_path):
    root = tmp_path / "controller"
    first = ExecutionController(root=root)
    bucket = context_bucket("q", "general", 0.5, 0.5)
    for _ in range(3):
        first.record_outcome(
            bucket=bucket, arm="base", verified_score=0.8, success=True
        )
    (root / "outcomes.jsonl").open("a").write("{corrupt\n")
    second = ExecutionController(root=root)
    status = second.status()
    assert status["episodes_seen"] == 3
    assert status["restore_errors"] == 1
    cell = next(c for c in status["cells"] if c["arm"] == "base")
    assert cell["n"] == 3 and cell["mean_verified"] == 0.8


def test_apply_arm_deltas_are_bounded(tmp_path):
    controller = _controller(tmp_path)
    config = {
        "max_steps": 15,
        "n_branches": 3,
        "fast_weights_max_layers": 8,
    }
    deeper = controller.apply_arm("deeper_recurrence", config)
    assert deeper["max_steps"] == 16  # capped, not 19
    assert deeper["n_branches"] == 2  # arm trades width for depth
    wider = controller.apply_arm("wider_branches", config)
    assert wider["n_branches"] == 4 and wider["max_steps"] == 13
    lean = controller.apply_arm("lean_fast_weights", config)
    assert lean["fast_weights_max_layers"] == 2
    assert controller.apply_arm("base", config) == config


def test_probe_guided_arm_emits_valid_bytecode_only_with_region(tmp_path):
    from core.brain.llm.latent_cortex.schedules import LayerSchedule

    controller = _controller(tmp_path)
    config = {"max_steps": 6, "n_branches": 2}
    without_region = controller.apply_arm("probe_guided_bytecode", config)
    assert "schedule" not in without_region
    with_region = controller.apply_arm(
        "probe_guided_bytecode", config, recurrent_region=(16, 48)
    )
    program = LayerSchedule.from_dict(with_region["schedule"])
    assert program.validate(prelude_end=16, coda_start=48) == []
    kinds = [op.kind for op in program.ops]
    assert kinds.count("verify_probe") == 2
    assert "savepoint" in kinds and "exchange" in kinds


def test_junk_outcomes_are_ignored(tmp_path):
    controller = _controller(tmp_path)
    controller.record_outcome(
        bucket="b", arm="base", verified_score=float("nan"), success=True
    )
    controller.record_outcome(
        bucket="b", arm="not_an_arm", verified_score=0.5, success=True
    )
    status = controller.status()
    assert status["episodes_seen"] == 1  # junk arm coerced to base, NaN dropped
    assert all(c["arm"] in ARMS for c in status["cells"])


def test_kill_switch_defaults_off_in_tests():
    assert controller_enabled() is False
