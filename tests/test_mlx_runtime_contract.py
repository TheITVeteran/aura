"""Runtime contract tests for MLX client/worker hardening."""

import asyncio
import os
import sys
import types

import pytest

from core.runtime.shutdown_coordinator import clear_shutdown_request, request_shutdown


def test_setup_worker_env_not_called_at_import():
    """_setup_worker_env must NOT run when the module is imported.

    If it does, the parent process inherits Metal-specific env vars that
    conflict with multi-process orchestration.
    """
    # Record current env state for key worker-only vars
    sentinel_keys = (
        "MLX_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "MLX_FORCE_SERIAL_COMPILE",
        "METAL_COMPILER_TIMEOUT_MS",
    )
    # Clear them so we can detect if import sets them
    saved = {}
    for key in sentinel_keys:
        saved[key] = os.environ.pop(key, None)

    try:
        # Force-reimport the module
        mod_name = "core.brain.llm.mlx_worker"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        # We can't fully reimport due to heavy deps, but we can verify the
        # module source doesn't call _setup_worker_env() at the top level
        import inspect

        from core.brain.llm import mlx_worker

        source = inspect.getsource(mlx_worker)
        # Find all top-level calls (not inside function bodies)
        # The fix moved _setup_worker_env() inside _mlx_worker_loop
        lines = source.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            # Top-level call (indent == 0) to _setup_worker_env
            if indent == 0 and stripped.startswith("_setup_worker_env("):
                raise AssertionError(
                    f"_setup_worker_env() called at module level (line {i}). "
                    "It must only be called inside _mlx_worker_loop()."
                )
    finally:
        for key, val in saved.items():
            if val is not None:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)


def test_bounded_max_tokens_narrowed_exceptions():
    """_bounded_max_tokens must not swallow arbitrary exceptions."""
    from core.brain.llm.mlx_client import _bounded_max_tokens

    # Normal operation
    assert _bounded_max_tokens(100, 200, 512) == 100
    assert _bounded_max_tokens(None, None, 256) == 256
    assert _bounded_max_tokens("", "", 128) == 128

    # Edge: non-numeric values should fallback, not crash
    assert _bounded_max_tokens("abc", 50, 64) == 50
    assert _bounded_max_tokens(50, object(), 64) == 50


def test_mlx_degradation_records_have_explicit_runtime_actions():
    """MLX failures must connect to recovery behavior, not only log telemetry."""
    from pathlib import Path

    from tools.audit_degradation import analyze_file

    assert analyze_file(Path("core/brain/llm/mlx_client.py")) == []


def test_mlx_runtime_probe_subprocess_is_bounded_and_reviewed():
    """The MLX probe subprocess must stay governed and bounded."""
    import inspect

    from core.brain.llm import mlx_client
    from tools.aura_enterprise_gate import ALLOW_SUBPROCESS

    source = inspect.getsource(mlx_client._probe_mlx_runtime)
    assert "core/brain/llm/mlx_client.py" in ALLOW_SUBPROCESS
    assert "get_subprocess_gateway().run(" in source
    assert "timeout=25.0" in source
    assert "source=\"runtime_probe:mlx_runtime_probe\"" in source
    assert "read_only=True" in source
    assert "AURA_TEST_MODE" not in source
    assert "shell=True" not in source


def test_worker_health_probe_bypasses_prompt_cache():
    from core.brain.llm import mlx_worker

    assert mlx_worker._job_requires_prompt_cache_bypass({"health_probe": True}) is True


def test_record_mlx_degradation_preserves_action_and_severity():
    from core.brain.llm.mlx_client import _record_mlx_degradation
    from core.runtime.errors import get_degradation_tracker

    tracker = get_degradation_tracker()
    tracker.reset()

    _record_mlx_degradation(
        RuntimeError("spawn wedged"),
        action="marked lane failed and applied spawn backoff",
        severity="error",
    )

    recent = tracker.recent(subsystem="mlx_client", limit=1)
    assert recent
    assert recent[0].severity == "error"
    assert recent[0].action == "marked lane failed and applied spawn backoff"


@pytest.mark.asyncio
async def test_mlx_warmup_refuses_before_worker_recovery_after_shutdown(monkeypatch):
    from core.brain.llm.mlx_client import MLXLocalClient

    clear_shutdown_request()
    client = MLXLocalClient("/models/Aura-32B-test")
    calls: list[str] = []

    async def _should_not_recover(*_args, **_kwargs):
        calls.append("recover")
        return True

    monkeypatch.setattr(client, "_ensure_worker_alive", _should_not_recover)

    try:
        request_shutdown("unit-test")
        assert await client.warmup(foreground_request=False) is False
    finally:
        clear_shutdown_request()

    assert calls == []
    assert client._warmup_attempted is False
    assert client._warmup_in_flight is False


def test_mlx_worker_spawn_refuses_before_orphan_scan_after_shutdown(monkeypatch):
    from core.brain.llm import mlx_client
    from core.brain.llm.mlx_client import MLXLocalClient

    clear_shutdown_request()
    client = MLXLocalClient("/models/Aura-32B-test")

    def _should_not_scan(*_args, **_kwargs):
        raise AssertionError("shutdown latch must stop spawn before process scans")

    monkeypatch.setattr(mlx_client.psutil, "process_iter", _should_not_scan)

    try:
        request_shutdown("unit-test")
        with pytest.raises(RuntimeError, match="runtime_shutdown"):
            client._spawn_worker_blocking()
    finally:
        clear_shutdown_request()


def test_mlx_worker_spawn_rechecks_shutdown_after_file_lock(monkeypatch, tmp_path):
    from core.brain.llm import mlx_client
    from core.brain.llm.mlx_client import MLXLocalClient

    client = MLXLocalClient.__new__(MLXLocalClient)
    client.model_path = "/models/Aura-32B-test"
    client._req_q = object()
    client._res_q = object()
    client.device = "gpu"
    client._substrate_mem = None
    client._steering_active = None
    client._cancel_seq = None
    client._mp_context = types.SimpleNamespace(
        Process=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("worker Process constructed after shutdown crossed file lock")
        )
    )

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(mlx_client.psutil, "process_iter", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(mlx_client, "_memory_pressure_blocks_worker_spawn", lambda *_args: None)
    monkeypatch.setattr(mlx_client, "_lane_admission_blocks_worker_spawn", lambda *_args: None)
    monkeypatch.setattr(mlx_client, "_probe_mlx_runtime", lambda: (True, "ok"))

    def _flock(_file, operation):
        if operation & mlx_client.fcntl.LOCK_NB:
            request_shutdown("crossed-file-lock")

    monkeypatch.setattr(mlx_client.fcntl, "flock", _flock)

    with pytest.raises(RuntimeError, match="runtime_shutdown"):
        client._spawn_worker_blocking()


@pytest.mark.asyncio
async def test_warmup_precompile_shutdown_cancellation_is_not_degradation(monkeypatch):
    from core.brain.llm import mlx_client
    from core.brain.llm.mlx_client import MLXLocalClient

    client = MLXLocalClient.__new__(MLXLocalClient)
    client.model_path = "/models/Aura-32B"

    async def cancelled_generation(*_args, **_kwargs):
        await asyncio.sleep(0)
        raise asyncio.CancelledError()

    monkeypatch.setattr(client, "_generate_inner", cancelled_generation)
    monkeypatch.setattr(mlx_client, "is_shutdown_requested", lambda: True)
    monkeypatch.setattr(
        mlx_client,
        "_record_mlx_degradation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("shutdown cancellation must not record MLX degradation")
        ),
    )

    with pytest.raises(asyncio.CancelledError):
        await client._run_warmup_precompile(
            request_is_background=False,
            foreground_request=True,
            owner_name="warmup:Aura-32B",
            warmup_timeout=1.0,
        )


def _install_worker_fakes(monkeypatch, mlx_worker, *, load_impl, steering_engine=None):
    class FakeQueue:
        def __init__(self, items=()):
            self.items = list(items)
            self.writes = []

        def get(self):
            if not self.items:
                raise AssertionError("worker read from an empty fake request queue")
            return self.items.pop(0)

        def put(self, item, *_, **__):
            self.writes.append(item)

    class FakeIPCWriter:
        def __init__(self, response_queue):
            self.response_queue = response_queue

        def start(self):
            return None

        def put(self, item):
            self.response_queue.put(item)

    class FakeWorkerThread:
        def __init__(self, *_, **__):
            return None

        def start(self):
            return None

        def start_job(self):
            return None

        def activity(self):
            return None

        def stop_job(self):
            return None

    mlx_core = types.ModuleType("mlx.core")
    mlx_core.set_cache_limit = lambda *_args, **_kwargs: None
    mlx_core.clear_cache = lambda *_args, **_kwargs: None
    mlx_core.metal = types.SimpleNamespace(
        set_cache_limit=lambda *_args, **_kwargs: None,
        clear_cache=lambda *_args, **_kwargs: None,
    )
    mlx_pkg = types.ModuleType("mlx")
    mlx_pkg.core = mlx_core
    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.load = load_impl
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda **_kwargs: object()

    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    monkeypatch.setattr(mlx_worker, "_setup_worker_env", lambda: None)
    monkeypatch.setattr(mlx_worker, "resolve_personality_adapter", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mlx_worker, "IPCWriterThread", FakeIPCWriter)
    monkeypatch.setattr(mlx_worker, "HeartbeatThread", FakeWorkerThread)
    monkeypatch.setattr(mlx_worker, "JobWatchdog", FakeWorkerThread)

    if steering_engine is not None:
        steering_mod = types.ModuleType("core.consciousness.affective_steering")
        steering_mod.get_steering_engine = lambda: steering_engine
        monkeypatch.setitem(sys.modules, "core.consciousness.affective_steering", steering_mod)

    return FakeQueue


def test_worker_init_failure_exits_before_accepting_jobs(monkeypatch):
    from core.brain.llm import mlx_worker
    from core.runtime.errors import get_degradation_tracker

    tracker = get_degradation_tracker()
    tracker.reset()

    load_failures = []

    def load_failure(*_args, **_kwargs):
        load_failures.append((_args, _kwargs))
        raise RuntimeError("model load failed")

    queue_factory = _install_worker_fakes(monkeypatch, mlx_worker, load_impl=load_failure)
    requests = queue_factory([{"action": "generate", "prompt": "must not be read"}])
    responses = queue_factory()

    mlx_worker._mlx_worker_loop("fake-model", requests, responses)

    assert len(load_failures) == 1
    assert requests.items == [{"action": "generate", "prompt": "must not be read"}]
    assert responses.writes[-1]["status"] == "error"
    assert responses.writes[-1]["action"] == "init"
    recent = tracker.recent(subsystem="mlx_worker", limit=1)
    assert recent
    assert recent[0].severity == "critical"
    assert recent[0].action == "reported initialization error and exited worker loop before accepting jobs"


def test_worker_blocks_generation_when_steering_liveness_drops(monkeypatch):
    from core.brain.llm import mlx_worker
    from core.runtime.errors import get_degradation_tracker

    tracker = get_degradation_tracker()
    tracker.reset()

    class FakeSteeringEngine:
        _alpha = 1.0

        def __init__(self):
            self.checks = 0

        def attach(self, *_args, **_kwargs):
            return None

        def is_active(self):
            self.checks += 1
            return self.checks == 1

    steering_engine = FakeSteeringEngine()
    queue_factory = _install_worker_fakes(
        monkeypatch,
        mlx_worker,
        load_impl=lambda *_args, **_kwargs: (object(), object()),
        steering_engine=steering_engine,
    )
    requests = queue_factory([{"id": "g1", "action": "generate", "prompt": "hello"}, None])
    responses = queue_factory()

    mlx_worker._mlx_worker_loop("fake-model", requests, responses)

    assert responses.writes[0]["status"] == "ok"
    generation_error = responses.writes[1]
    assert generation_error["id"] == "g1"
    assert generation_error["action"] == "generate"
    assert generation_error["status"] == "error"
    assert "steering" in generation_error["message"].lower()
    recent = tracker.recent(subsystem="mlx_worker", limit=1)
    assert recent
    assert recent[0].severity == "critical"
    assert recent[0].action == "blocked generation because steering liveness failed"


def test_response_listener_shutdown_awareness():
    """The response listener loop condition should check shutdown state."""
    import inspect

    from core.brain.llm.mlx_client import MLXLocalClient

    source = inspect.getsource(MLXLocalClient._response_listener_loop)
    forbidden_loop = "while " + "True"
    assert "_runtime_shutdown_requested()" in source, (
        "_response_listener_loop must check _runtime_shutdown_requested() "
        "instead of using an unbounded loop"
    )
    assert forbidden_loop not in source, (
        "_response_listener_loop should not use an unbounded loop; it must be shutdown-aware"
    )


def test_worker_docstring_placement():
    """_mlx_worker_loop must have its docstring as the first thing in the body."""
    from core.brain.llm.mlx_worker import _mlx_worker_loop

    assert _mlx_worker_loop.__doc__ is not None, "_mlx_worker_loop is missing its docstring"
    assert "subprocess" in _mlx_worker_loop.__doc__.lower(), (
        "_mlx_worker_loop docstring should mention subprocess isolation"
    )


def test_no_duplicate_consecutive_empty_reset():
    """_consecutive_empty should only be reset once on successful generation."""
    import inspect

    from core.brain.llm.mlx_client import MLXLocalClient

    source = inspect.getsource(MLXLocalClient._generate_inner)

    # The success path is the block between "res.get('status') == 'ok'" and
    # the next except clause. Count resets in the full method — there should
    # be exactly one on the success path, not two adjacent lines.
    # A duplicate was the original bug; verify it stays gone.
    lines_with_reset = [
        line.strip() for line in source.splitlines() if "self._consecutive_empty = 0" in line
    ]
    assert len(lines_with_reset) == 1, (
        f"Expected exactly 1 reset of _consecutive_empty in _generate_inner, "
        f"found {len(lines_with_reset)}. Duplicate resets obscure intent."
    )


def test_stream_path_no_mid_generation_cache_clear():
    """The stream path must NOT clear MLX cache every N tokens.

    Mid-generation cache clearing forces Metal to reallocate GPU memory,
    creating micro-stalls that degrade token throughput. Post-generation
    cleanup is sufficient.
    """
    import inspect

    from core.brain.llm.mlx_worker import _mlx_worker_loop

    source = inspect.getsource(_mlx_worker_loop)
    # Find the stream action handler
    stream_start = source.find('elif action == "stream":')
    assert stream_start > 0, "Could not find stream action handler"
    stream_section = source[stream_start:]

    # The old pattern was: if token_count % 10 == 0: _clear_mlx_cache(mx)
    assert "token_count % 10" not in stream_section, (
        "Stream path still contains mid-generation cache clearing every 10 tokens. "
        "This was identified as harmful to throughput and must be removed."
    )


def test_stream_path_uses_surface_generation_controls():
    """Streaming user-visible text must use the same controls as normal generation.

    The desktop UI commonly consumes streamed tokens. If this path bypasses
    the surface clamp, live chat can diverge from proof/backend generation.
    """
    import inspect

    from core.brain.llm.mlx_worker import _mlx_worker_loop

    source = inspect.getsource(_mlx_worker_loop)
    stream_start = source.find('elif action == "stream":')
    assert stream_start > 0, "Could not find stream action handler"
    stream_section = source[stream_start:]
    apply_idx = stream_section.find("_apply_surface_generation_controls")
    generate_idx = stream_section.find("stream_generate(")
    restore_idx = stream_section.find("_restore_surface_generation_controls")

    assert apply_idx > 0, "Stream path does not apply surface generation controls."
    assert generate_idx > 0, "Stream path does not call stream_generate."
    assert restore_idx > 0, "Stream path does not restore surface generation controls."
    assert apply_idx < generate_idx < restore_idx, (
        "Stream path must apply surface controls before token generation and "
        "restore them after generation exits."
    )
