import asyncio
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import core.runtime.boot_safety as boot_safety_module
from core.brain.inference_gate import InferenceGate
from core.brain.llm_health_router import build_router_from_config
from core.config import PROJECT_ROOT, config
from core.container import ServiceContainer
from core.runtime.boot_safety import main_process_camera_policy, uvloop_allowed
from core.runtime.desktop_boot_safety import (
    compute_mlx_cache_limit,
    compute_mlx_memory_limit,
    compute_process_rss_limit,
    desktop_safe_boot_enabled,
    inprocess_mlx_metal_enabled,
)
from core.senses.continuous_vision import ContinuousSensoryBuffer
from core.sensory_motor_cortex import SensoryMotorCortex
from core.utils.memory_monitor import AppleSiliconMemoryMonitor

VISION_TEST_ROOT = Path(tempfile.gettempdir()) / "aura-test"


class AsyncCallRecorder:
    def __init__(self, result=None):
        self.result = result
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


def test_config_exports_project_root_alias():
    assert PROJECT_ROOT == config.paths.project_root


def test_uvloop_disabled_by_default_on_darwin(monkeypatch):
    monkeypatch.delenv("AURA_ENABLE_UVLOOP", raising=False)
    assert uvloop_allowed(platform="darwin") is False


def test_uvloop_can_be_forced_on_darwin(monkeypatch):
    monkeypatch.setenv("AURA_ENABLE_UVLOOP", "1")
    assert uvloop_allowed(platform="darwin") is True


def test_main_process_camera_policy_blocks_darwin_without_override(monkeypatch):
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_MAIN_PROCESS_CAMERA", raising=False)
    enabled, reason = main_process_camera_policy(True, platform="darwin")
    assert enabled is False
    assert "cv2/PyAV" in reason


def test_continuous_vision_blocks_forced_camera_on_darwin(monkeypatch):
    monkeypatch.setenv("AURA_FORCE_CAMERA", "1")
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_MAIN_PROCESS_CAMERA", raising=False)
    monkeypatch.setattr(boot_safety_module.sys, "platform", "darwin")

    buffer = ContinuousSensoryBuffer(VISION_TEST_ROOT)

    assert buffer.camera_enabled is False


def test_sensory_motor_cortex_blocks_forced_camera_on_darwin(monkeypatch):
    monkeypatch.setenv("AURA_FORCE_CAMERA", "1")
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_MAIN_PROCESS_CAMERA", raising=False)
    monkeypatch.setattr(boot_safety_module.sys, "platform", "darwin")

    cortex = SensoryMotorCortex()

    assert cortex.camera_enabled is False


def test_sensory_motor_cortex_syncs_user_activity_before_idle_trigger():
    orchestrator = SimpleNamespace(
        _last_user_interaction_time=200.0,
        status=SimpleNamespace(is_processing=False),
        _current_thought_task=None,
    )
    cortex = SensoryMotorCortex(orchestrator=orchestrator, config={"boredom_threshold": 120})
    cortex.last_interaction_time = 0.0

    assert cortex._should_trigger_volition(now=250.0) is False
    assert cortex.last_interaction_time == 200.0


def test_sensory_motor_cortex_skips_volition_while_processing():
    orchestrator = SimpleNamespace(
        _last_user_interaction_time=0.0,
        status=SimpleNamespace(is_processing=True),
        _current_thought_task=None,
    )
    cortex = SensoryMotorCortex(orchestrator=orchestrator, config={"boredom_threshold": 120})
    cortex.last_interaction_time = 0.0

    assert cortex._should_trigger_volition(now=500.0) is False
    assert cortex.last_interaction_time == 500.0


@pytest.mark.asyncio
async def test_sensory_motor_cortex_routes_idle_volition_into_autonomy():
    trigger_autonomous_thought = AsyncCallRecorder()
    generate_autonomous_thought = AsyncCallRecorder()
    emit_spontaneous_message = AsyncCallRecorder()
    orchestrator = SimpleNamespace(
        _trigger_autonomous_thought=trigger_autonomous_thought,
        generate_autonomous_thought=generate_autonomous_thought,
        emit_spontaneous_message=emit_spontaneous_message,
    )
    cortex = SensoryMotorCortex(orchestrator=orchestrator)

    await cortex._dispatch_idle_volition(reason="idle_timeout")

    assert trigger_autonomous_thought.calls == [((False,), {})]
    assert generate_autonomous_thought.calls == []
    assert emit_spontaneous_message.calls == []


def test_memory_monitor_uses_psutil_pressure_sample(monkeypatch):
    monitor = AppleSiliconMemoryMonitor()
    monkeypatch.setattr(
        "core.utils.memory_monitor.psutil.virtual_memory",
        lambda: SimpleNamespace(percent=57.8),
    )

    assert monitor._get_pressure_sysctl() == 57


def test_health_router_prefers_existing_inference_gate(monkeypatch):
    sentinel_gate = object()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(lambda cls, name, default="_SENTINEL": sentinel_gate if name == "inference_gate" else default),
    )

    router = build_router_from_config(config)

    from core.brain.llm.model_registry import PRIMARY_ENDPOINT
    assert router.endpoints[PRIMARY_ENDPOINT].client is sentinel_gate


@pytest.mark.asyncio
async def test_lazy_local_client_initializes_off_event_loop(monkeypatch):
    sentinel_gate = object()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(lambda cls, name, default="_SENTINEL": sentinel_gate if name == "inference_gate" else default),
    )

    router = build_router_from_config(config)

    from core.brain.llm.model_registry import BRAINSTEM_ENDPOINT

    client = router.endpoints[BRAINSTEM_ENDPOINT].client
    generate_text_async = AsyncCallRecorder("ok")
    downstream = SimpleNamespace(generate_text_async=generate_text_async)
    offloads = []

    async def fake_to_thread(fn):
        offloads.append(fn)
        return fn()

    monkeypatch.setattr(client, "_get_client", lambda: downstream)
    monkeypatch.setattr("core.brain.llm_health_router.asyncio.to_thread", fake_to_thread)

    assert await client.generate_text_async("hello") == "ok"
    assert len(offloads) == 1
    assert generate_text_async.calls == [(("hello",), {})]


def test_health_router_exposes_only_cortex_during_primary_proof(monkeypatch):
    sentinel_gate = object()
    monkeypatch.setattr("core.brain.llm_health_router.proof_run_active", lambda **_kwargs: True)
    monkeypatch.setattr("core.brain.llm_health_router.proof_model_tier", lambda: "primary")
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(lambda cls, name, default="_SENTINEL": sentinel_gate if name == "inference_gate" else default),
    )

    router = build_router_from_config(config)

    from core.brain.llm.model_registry import PRIMARY_ENDPOINT

    assert list(router.endpoints) == [PRIMARY_ENDPOINT]
    assert router.endpoints[PRIMARY_ENDPOINT].client is sentinel_gate


def test_desktop_safe_boot_tracks_app_launch_context(monkeypatch):
    monkeypatch.delenv("AURA_SAFE_BOOT_DESKTOP", raising=False)
    monkeypatch.setenv("AURA_LAUNCHED_FROM_APP", "1")

    assert desktop_safe_boot_enabled() is True


def test_inference_gate_disables_boot_prewarm_under_safe_desktop_boot(monkeypatch):
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")

    assert InferenceGate._boot_should_eager_warmup() is False
    assert InferenceGate._boot_should_schedule_deferred_prewarm() is False


def test_compute_mlx_cache_limit_uses_safer_cap_for_desktop_safe_boot(monkeypatch):
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.delenv("AURA_LAUNCHED_FROM_APP", raising=False)

    total = 64 * 1024 ** 3
    limit = compute_mlx_cache_limit(total)

    assert limit == 10 * 1024 ** 3


def test_compute_mlx_memory_limit_uses_desktop_safe_active_memory_ceiling(monkeypatch):
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.delenv("AURA_MLX_MEMORY_LIMIT_GB", raising=False)
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_MEMORY_LIMITS", raising=False)

    total = 64 * 1024 ** 3
    limit = compute_mlx_memory_limit(total)

    assert limit == 28 * 1024 ** 3


def test_compute_process_rss_limit_uses_desktop_safe_guard_ceiling(monkeypatch):
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.delenv("AURA_PROCESS_RSS_LIMIT_GB", raising=False)
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_MEMORY_LIMITS", raising=False)

    total = 64 * 1024 ** 3
    limit = compute_process_rss_limit(total)

    assert limit == int(total * 0.56)


def test_desktop_safe_boot_clamps_unsafe_inherited_model_limits(monkeypatch):
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.setenv("AURA_MLX_MEMORY_LIMIT_GB", "96")
    monkeypatch.setenv("AURA_PROCESS_RSS_LIMIT_GB", "120")
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_MEMORY_LIMITS", raising=False)

    total = 64 * 1024 ** 3

    assert compute_mlx_memory_limit(total) == 28 * 1024 ** 3
    assert compute_process_rss_limit(total) == 36 * 1024 ** 3


def test_desktop_safe_boot_allows_explicit_unsafe_memory_override(monkeypatch):
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.setenv("AURA_MLX_MEMORY_LIMIT_GB", "40")
    monkeypatch.setenv("AURA_PROCESS_RSS_LIMIT_GB", "42")
    monkeypatch.setenv("AURA_ALLOW_UNSAFE_MEMORY_LIMITS", "1")

    total = 64 * 1024 ** 3

    assert compute_mlx_memory_limit(total) == 40 * 1024 ** 3
    assert compute_process_rss_limit(total) == 42 * 1024 ** 3


def test_desktop_safe_boot_clamps_stale_floor_overrides(monkeypatch):
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.setenv("AURA_SAFE_BOOT_METAL_CACHE_FLOOR_GB", "80")
    monkeypatch.setenv("AURA_SAFE_BOOT_MLX_MEMORY_FLOOR_GB", "80")
    monkeypatch.setenv("AURA_SAFE_BOOT_PROCESS_RSS_FLOOR_GB", "80")
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_MEMORY_LIMITS", raising=False)

    total = 64 * 1024 ** 3

    assert compute_mlx_cache_limit(total) == 10 * 1024 ** 3
    assert compute_mlx_memory_limit(total) == 28 * 1024 ** 3
    assert compute_process_rss_limit(total) == 36 * 1024 ** 3


def test_live_boot_proof_inherits_safe_desktop_mlx_limits(monkeypatch):
    from tools.live_boot_proof import build_safe_boot_env, live_proof_rss_abort_mb

    monkeypatch.setattr(
        "tools.live_boot_proof.psutil.virtual_memory",
        lambda: SimpleNamespace(total=64 * 1024 ** 3),
    )

    env = build_safe_boot_env({})

    assert env["AURA_LOCAL_BACKEND"] == "mlx"
    assert env["AURA_SAFE_BOOT_DESKTOP"] == "1"
    assert env["AURA_HEADLESS"] == "1"
    assert env["AURA_DEFERRED_CORTEX_PREWARM"] == "1"
    assert env["AURA_LOCAL_RUNTIME_SINGLETON"] == "1"
    assert env["AURA_LOCAL_PARALLEL_SLOTS"] == "1"
    assert env["AURA_EAGER_LOCAL_SENSORY_BOOT"] == "0"
    assert env["AURA_ENABLE_PROACTIVE_VISION"] == "0"
    assert env["AURA_SAFE_BOOT_METAL_CACHE_RATIO"] == "0.16"
    assert env["AURA_SAFE_BOOT_METAL_CACHE_CAP_GB"] == "10"
    assert env["AURA_FOREGROUND_CHAT_MAX_TOKENS"] == "2048"
    assert env["AURA_MLX_MEMORY_LIMIT_GB"] == "28"
    assert env["AURA_PROCESS_RSS_LIMIT_GB"] == "36"
    assert live_proof_rss_abort_mb(env) == 38_000.0


def test_live_boot_proof_desktop_mode_mirrors_packaged_launcher(monkeypatch):
    from tools.live_boot_proof import build_safe_boot_env

    monkeypatch.setattr(
        "tools.live_boot_proof.psutil.virtual_memory",
        lambda: SimpleNamespace(total=64 * 1024 ** 3),
    )

    env = build_safe_boot_env({}, mode="desktop")

    assert env["AURA_LOCAL_BACKEND"] == "mlx"
    assert env["AURA_SAFE_BOOT_DESKTOP"] == "1"
    assert env["AURA_HEADLESS"] == "0"
    assert env["AURA_LAUNCHED_FROM_APP"] == "1"
    assert env["AURA_EXTERNAL_GUI_OWNER"] == "1"
    assert env["AURA_EAGER_CORTEX_WARMUP"] == "0"
    assert env["AURA_DEFERRED_CORTEX_PREWARM"] == "1"


def test_live_boot_proof_preserves_operator_mlx_limit(monkeypatch):
    from tools.live_boot_proof import build_safe_boot_env

    monkeypatch.setattr(
        "tools.live_boot_proof.psutil.virtual_memory",
        lambda: SimpleNamespace(total=64 * 1024 ** 3),
    )

    env = build_safe_boot_env({"AURA_MLX_MEMORY_LIMIT_GB": "28"})

    assert env["AURA_SAFE_BOOT_DESKTOP"] == "1"
    assert env["AURA_MLX_MEMORY_LIMIT_GB"] == "28"


def test_live_boot_proof_clamps_unsafe_parent_memory_limits(monkeypatch):
    from tools.live_boot_proof import build_safe_boot_env, live_proof_rss_abort_mb

    monkeypatch.setattr(
        "tools.live_boot_proof.psutil.virtual_memory",
        lambda: SimpleNamespace(total=64 * 1024 ** 3),
    )

    env = build_safe_boot_env(
        {
            "AURA_MLX_MEMORY_LIMIT_GB": "96",
            "AURA_PROCESS_RSS_LIMIT_GB": "120",
            "AURA_LIVE_PROOF_RSS_ABORT_MB": "90000",
        }
    )

    assert env["AURA_MLX_MEMORY_LIMIT_GB"] == "28"
    assert env["AURA_PROCESS_RSS_LIMIT_GB"] == "36"
    assert live_proof_rss_abort_mb(env) == 38_000.0


def test_live_boot_proof_uses_readiness_heartbeat_contract():
    source = (PROJECT_ROOT / "tools" / "live_boot_proof.py").read_text()

    assert "/api/health/heartbeat" in source
    assert "required_probes" in source
    assert "runtime_probe_healthy" in source
    assert "system_ready" in source
    assert "exercise_capability_inventory_turn" in source
    assert "X-Aura-Require-CognitiveEngine" in source


def test_live_boot_proof_runtime_stream_scan_fails_failure_markers(monkeypatch, tmp_path):
    import tools.live_boot_proof as live_boot_proof

    monkeypatch.setattr(live_boot_proof, "PROOF_DIR", tmp_path)
    proof = live_boot_proof.LiveProof(
        port=8999,
        mode="desktop",
        boot_timeout_s=1.0,
        skip_desktop=True,
        restart_continuity=False,
        conversation_soak_turns=0,
    )
    proof.stdout_path.write_text(
        "Cortex Warming...\nTraceback (most recent call last):\nRuntime: DEGRADED\n",
        encoding="utf-8",
    )

    assert proof.scan_runtime_stream() is False
    step = proof.steps[-1]
    assert step["step"] == "runtime_stream_scan"
    assert "Cortex Warming" in step["markers"]
    assert "Traceback" in step["markers"]
    assert "Runtime: DEGRADED" in step["markers"]


def test_live_boot_proof_verdict_records_commit_and_end_metadata():
    source = (PROJECT_ROOT / "tools" / "live_boot_proof.py").read_text()

    assert '"ended_at": finished_at' in source
    assert '"git_commit": git_commit' in source
    assert '"git_dirty": git_dirty' in source
    assert '"stdout_log": artifact_display_path(self.stdout_path)' in source
    assert "current_git_commit()" in source
    assert "current_git_dirty()" in source


def test_live_boot_proof_supports_stable_output_directory():
    source = (PROJECT_ROOT / "tools" / "live_boot_proof.py").read_text()

    assert "--out-dir" in source
    assert "self.latest_verdict_path" in source
    assert "LATEST_VERDICT.json" in source


def test_compute_mlx_cache_limit_defaults_to_standard_ratio_when_not_safe(monkeypatch):
    monkeypatch.delenv("AURA_SAFE_BOOT_DESKTOP", raising=False)
    monkeypatch.delenv("AURA_LAUNCHED_FROM_APP", raising=False)

    limit = compute_mlx_cache_limit(64 * 1024 ** 3)

    assert limit == int(64 * 1024 ** 3 * 0.75)


def test_rsi_lab_creates_data_dir_without_runtime_globals(monkeypatch, tmp_path):
    from research.meta_learning_loop import RSILab

    monkeypatch.setattr(type(config.paths), "_runtime_home_cache", tmp_path)

    lab = RSILab()

    assert lab.lab_dir == tmp_path / "data" / "rsi_lab"
    assert lab.lab_dir.exists()


@pytest.mark.asyncio
async def test_rsi_lab_requires_validation_evidence_for_promotion(monkeypatch, tmp_path):
    from research.meta_learning_loop import RSILab

    monkeypatch.setattr(type(config.paths), "_runtime_home_cache", tmp_path)
    lab = RSILab()
    weak_id = lab.submit_candidate(
        "heuristic",
        "always do the clever thing",
        "Too vague to promote because it has no validation or rollback evidence.",
    )
    strong_id = lab.submit_candidate(
        "skill",
        {
            "steps": ["inspect inputs", "run verifier", "emit receipt"],
            "tool_contract": {"input": "objective", "output": "verified_plan"},
            "evidence": {
                "provenance": "unit-test",
                "validation_command": "pytest tests/test_boot_runtime_safety.py",
                "validation_passed": True,
                "rollback_plan": "remove skill registration",
                "receipt_id": "receipt_rsi_validation_001",
                "risk": {"level": "bounded", "blast_radius": "skill registry only"},
            },
        },
        "Promote because the skill has explicit validation, provenance, rollback, and bounded risk.",
    )

    assert await lab.evaluate_pending_candidates() == 2

    assert lab.candidates[weak_id].status == "failed"
    assert "validation_passed" in lab.candidates[weak_id].evaluation_report["blocking_failures"]
    assert lab.candidates[strong_id].status == "passed"
    assert lab.candidates[strong_id].evaluation_report["checks"]["receipt_present"] is True
    assert lab.promote(weak_id) is False
    assert lab.candidates[weak_id].status == "failed"
    assert lab.promote("missing-candidate") is False
    assert lab.promote(strong_id) is True
    assert lab.candidates[strong_id].status == "promoted"


def test_rsi_lab_loads_valid_candidates_while_skipping_corrupt_records(monkeypatch, tmp_path):
    from research.meta_learning_loop import RSILab

    monkeypatch.setattr(type(config.paths), "_runtime_home_cache", tmp_path)
    lab_dir = tmp_path / "data" / "rsi_lab"
    lab_dir.mkdir(parents=True)
    (lab_dir / "candidates.json").write_text(
        json.dumps(
            {
                "valid": {
                    "id": "valid",
                    "artifact_type": "heuristic",
                    "content": {"rule": "prefer verified changes because they are reversible"},
                    "rationale": "Keep only candidates with enough evidence because promotion is risky.",
                    "status": "pending_eval",
                    "score": 0.0,
                    "evaluation_report": {},
                    "created_at": 1.0,
                },
                "corrupt": ["not", "a", "candidate"],
            }
        ),
        encoding="utf-8",
    )

    lab = RSILab()

    assert list(lab.candidates) == ["valid"]
    assert lab.candidates["valid"].artifact_type == "heuristic"


@pytest.mark.asyncio
async def test_chaos_fill_disk_creates_bounded_pressure_file(monkeypatch, tmp_path):
    from tools.chaos import injector

    scheduled = []

    class Tracker:
        def create_task(self, coro, name=None):
            scheduled.append((coro, name))
            return SimpleNamespace(done=lambda: False)

    monkeypatch.setenv("AURA_CHAOS_DISK_TARGET_DIR", str(tmp_path))
    monkeypatch.setenv("AURA_CHAOS_DISK_MAX_MB", "1")
    monkeypatch.setenv("AURA_CHAOS_DISK_RESTORE_SECONDS", "0")
    monkeypatch.setattr(injector, "get_task_tracker", lambda: Tracker())

    result = await injector._fill_disk()

    pressure_file = Path(result["target"])
    assert result["applied"] is True
    assert result["bytes_written"] == 1024 * 1024
    assert await asyncio.to_thread(pressure_file.exists)
    assert scheduled and scheduled[0][1] == "chaos.fill_disk.restore_pressure_file"

    await scheduled[0][0]

    assert not await asyncio.to_thread(pressure_file.exists)
    assert not await asyncio.to_thread(pressure_file.parent.exists)


def test_inprocess_mlx_metal_disabled_during_safe_boot(monkeypatch):
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.delenv("AURA_FORCE_INPROCESS_MLX_METAL", raising=False)
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_INPROCESS_MLX_METAL", raising=False)
    monkeypatch.delenv("AURA_DISABLE_INPROCESS_MLX_METAL", raising=False)

    enabled, reason = inprocess_mlx_metal_enabled(
        platform_name="darwin",
        mac_version="26.4",
    )

    assert enabled is False
    assert reason == "desktop_safe_boot"


def test_inprocess_mlx_metal_disabled_on_macos26_by_default(monkeypatch):
    monkeypatch.delenv("AURA_SAFE_BOOT_DESKTOP", raising=False)
    monkeypatch.delenv("AURA_LAUNCHED_FROM_APP", raising=False)
    monkeypatch.delenv("AURA_FORCE_INPROCESS_MLX_METAL", raising=False)
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_INPROCESS_MLX_METAL", raising=False)
    monkeypatch.delenv("AURA_DISABLE_INPROCESS_MLX_METAL", raising=False)

    enabled, reason = inprocess_mlx_metal_enabled(
        platform_name="darwin",
        mac_version="26.4",
    )

    assert enabled is False
    assert reason == "macos26_guard"


def test_inprocess_mlx_metal_can_be_forced_for_debugging(monkeypatch):
    monkeypatch.delenv("AURA_SAFE_BOOT_DESKTOP", raising=False)
    monkeypatch.delenv("AURA_LAUNCHED_FROM_APP", raising=False)
    monkeypatch.setenv("AURA_FORCE_INPROCESS_MLX_METAL", "1")
    monkeypatch.delenv("AURA_DISABLE_INPROCESS_MLX_METAL", raising=False)

    enabled, reason = inprocess_mlx_metal_enabled(
        platform_name="darwin",
        mac_version="26.4",
    )

    assert enabled is True
    assert reason == "forced"


def test_live_learner_autorun_training_requires_explicit_operator_policy(monkeypatch):
    from core.learning.live_learner import LiveLearner, TrainingPolicy

    monkeypatch.delenv("AURA_SELF_TRAIN_AUTORUN", raising=False)
    assert TrainingPolicy.from_env().autorun_enabled is False

    learner = LiveLearner.__new__(LiveLearner)
    learner._policy = TrainingPolicy(autorun_enabled=False)
    learner._training_in_progress = False
    learner._model_path = "aura-model"
    learner._buffer = [{} for _ in range(LiveLearner.MIN_EXAMPLES_FOR_TRAINING)]
    learner._last_train_time = 0.0

    assert learner._should_train() is False

    learner._policy = TrainingPolicy(autorun_enabled=True)
    assert learner._should_train() is True


def test_voice_engine_imports_current_data_dir_path(monkeypatch):
    """Voice boot must not regress to the historical core.common.paths DATA_DIR import."""
    monkeypatch.setenv("AURA_AUTO_LISTEN", "0")

    from core.self_model import DATA_FILE
    from core.senses.voice_engine import SovereignVoiceEngine
    from core.utils.paths import DATA_DIR

    engine = SovereignVoiceEngine()

    assert DATA_FILE == DATA_DIR / "self_model.json"
    assert str(engine.data_dir).endswith("voice_models")


@pytest.mark.asyncio
async def test_continuous_vision_defers_screen_backend_without_permission(monkeypatch):
    class _FakeMSSModule:
        def __init__(self):
            self.mss_calls = 0

        def mss(self):
            self.mss_calls += 1
            raise AssertionError("mss() should not be called without active permission")

    check_permission = AsyncCallRecorder({"granted": False, "status": "deferred"})
    guard = SimpleNamespace(check_permission=check_permission)

    fake_mss = _FakeMSSModule()
    monkeypatch.setitem(sys.modules, "mss", fake_mss)
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(lambda cls, name, default=None: guard if name == "permission_guard" else default),
    )

    buffer = ContinuousSensoryBuffer(VISION_TEST_ROOT)
    ready = await buffer._ensure_screen_backend()

    assert ready is False
    assert len(check_permission.calls) == 1
    assert fake_mss.mss_calls == 0
    assert buffer.sct is None
    assert buffer.monitor is None
