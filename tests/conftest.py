"""Shared pytest fixtures for Aura smoke tests."""
import asyncio
import builtins
import contextlib
import inspect
import os
import shutil
import socket
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

# Log hermeticity: keep test logging out of the live ~/.aura/logs so suite
# noise (test doubles, induced failures) never pollutes the running
# instance's aura_json.log. Set before any core import can call
# setup_logging(); PID-scoped so parallel chunk runners don't share a file.
if not os.environ.get("AURA_LOG_DIR", "").strip():
    os.environ["AURA_LOG_DIR"] = str(
        Path(tempfile.gettempdir()) / f"aura-test-logs-{os.getpid()}"
    )

# State hermeticity: redirect the central Aura home before core.config can
# construct its process-global settings object. Report-only detection was not
# enough: immune singleton tests wrote evolved cells into the user's live
# ~/.aura/data store while still passing. Tests that intentionally exercise a
# caller-supplied path continue to pass that path directly.
if not os.environ.get("AURA_PATHS__HOME_DIR", "").strip():
    os.environ["AURA_PATHS__HOME_DIR"] = str(
        Path(tempfile.gettempdir()) / f"aura-test-home-{os.getpid()}"
    )
os.environ.setdefault("AURA_TEST_LIVE_DATA_GUARD", "fail")
os.environ.setdefault("AURA_TEST_STATE_GUARD", "fail")

# Ledger hermeticity: the latent execution controller learns from live
# episode outcomes and persists them under the real data dir. Tests running
# fake episodes must never pollute that evidence; tests that exercise the
# controller construct their own instance with a tmp root.
os.environ.setdefault("AURA_EXECUTION_CONTROLLER", "0")

# Determinism: token-progress budgets adapt to LIVE machine memory pressure
# by default (the host running this suite often has a 20GB model resident).
# Pin the adaptation off so timing assertions can't drift with the
# environment; targeted tests opt back in and inject their own snapshots.
os.environ.setdefault("AURA_FIRST_TOKEN_PRESSURE_ADAPT", "0")

# Determinism: hybrid semantic retrieval would load a real MiniLM backend
# and make ranking assertions environment-dependent. Pin it off; the
# targeted rag tests opt back in with an injected fake engine.
os.environ.setdefault("AURA_SEMANTIC_RAG", "0")

_CLEANUP_TIMEOUT_S = 2.0

# Ensure the project root is on sys.path so `core.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_collection_modifyitems(config, items):
    """Keep destructive resident-model gates opt-in without recording skips."""
    if os.environ.get("AURA_RUN_RLC_RESIDENT_1P5B_GATE") == "1":
        return
    selected = []
    deselected = []
    for item in items:
        if item.get_closest_marker("resident_model") is None:
            selected.append(item)
        else:
            deselected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


@dataclass(frozen=True)
class _ResourceLeakSnapshot:
    child_identities: frozenset[tuple[int, float]]
    listening_fds: frozenset[tuple[int, int, str, int]]
    open_files: frozenset[str]


# setup_logging() installs one RotatingFileHandler on aura_json.log and keeps
# it for the life of the process — that is the design, and this same conftest
# points it at a PID-scoped temp dir a few lines above so it never touches the
# live instance's log. A test that happens to trigger the first Aura import
# therefore "opens" a file it must not close, which the leak detector counted
# as a leak: test_ui_bootstrap_returns_state_and_tool_catalog failed in
# teardown for a handler working exactly as intended. Exempt that one sink by
# name, and nothing else, so a genuinely leaked file still fails.
def _is_process_lifetime_log_sink(path) -> bool:
    return os.path.basename(str(path)) == "aura_json.log"


class HermeticResourceSandbox:
    """Per-test host leak detector; never used as resource-policy evidence."""

    def __init__(self, *, root: Path):
        import psutil

        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        # Tests deliberately monkeypatch psutil's module attributes. Pin the
        # native constructors before the test body so teardown observation
        # cannot be redirected through the very double it is auditing.
        self._native_process = psutil.Process
        self._native_error = psutil.Error
        self._native_wait_procs = psutil.wait_procs
        self._leased_sockets: list[socket.socket] = []
        self.baseline = self.snapshot()

    def snapshot(self) -> _ResourceLeakSnapshot:
        try:
            process = self._native_process(os.getpid())
            children = frozenset(
                (child.pid, float(child.create_time()))
                for child in process.children(recursive=True)
                if child.status().lower() not in {"dead", "zombie"}
            )
            connections = process.net_connections(kind="inet")
            open_files = process.open_files()
        except (self._native_error, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(f"host leak observation unavailable: {exc}") from exc
        listeners = frozenset(
            (
                os.getpid(),
                int(connection.fd),
                str(getattr(connection.laddr, "ip", "") or ""),
                int(getattr(connection.laddr, "port", 0) or 0),
            )
            for connection in connections
            if str(connection.status).upper() == "LISTEN"
        )
        return _ResourceLeakSnapshot(
            child_identities=children,
            listening_fds=listeners,
            open_files=frozenset(
                str(item.path)
                for item in open_files
                if not _is_process_lifetime_log_sink(item.path)
            ),
        )

    def leaks(self) -> dict[str, set[object]]:
        current = self.snapshot()
        return {
            "children": set(current.child_identities - self.baseline.child_identities),
            "listeners": set(current.listening_fds - self.baseline.listening_fds),
            "open_files": set(current.open_files - self.baseline.open_files),
        }

    @contextlib.contextmanager
    def listening_socket(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        self._leased_sockets.append(listener)
        try:
            yield listener
        finally:
            with contextlib.suppress(OSError):
                listener.close()
            with contextlib.suppress(ValueError):
                self._leased_sockets.remove(listener)

    def close_and_assert_clean(self) -> None:
        leaked_leases = [sock for sock in self._leased_sockets if sock.fileno() >= 0]
        for listener in tuple(self._leased_sockets):
            with contextlib.suppress(OSError):
                listener.close()
        self._leased_sockets.clear()

        deadline = time.monotonic() + 0.75
        leaks = self.leaks()
        while leaks["children"] and time.monotonic() < deadline:
            time.sleep(0.05)
            leaks = self.leaks()

        child_pids = sorted(int(identity[0]) for identity in leaks["children"])
        if child_pids:
            handles = []
            for pid in child_pids:
                with contextlib.suppress(self._native_error):
                    handles.append(self._native_process(pid))
            for handle in handles:
                with contextlib.suppress(self._native_error):
                    handle.terminate()
            _gone, alive = self._native_wait_procs(handles, timeout=0.5)
            for handle in alive:
                with contextlib.suppress(self._native_error):
                    handle.kill()

        listener_leaks = list(leaks["listeners"])
        for pid, fd, _host, _port in listener_leaks:
            if int(pid) == os.getpid() and int(fd) >= 0:
                with contextlib.suppress(OSError):
                    os.close(int(fd))

        if leaked_leases or any(leaks.values()):
            pytest.fail(
                "hermetic resource leak detected: "
                f"leased_sockets={len(leaked_leases)} leaks={leaks}"
            )


@pytest.fixture(autouse=True)
def hermetic_resource_sandbox(tmp_path):
    sandbox = HermeticResourceSandbox(root=tmp_path / "resource-sandbox")
    try:
        yield sandbox
    finally:
        sandbox.close_and_assert_clean()


@pytest.fixture(autouse=True)
def _live_data_write_guard(request):
    """Hermeticity: flag any Python-level write into the real ~/.aura/data.

    2026-07-12: pin tests were found appending fixture pins to the LIVE
    session-memory ledger — phantom memories Aura could recall as real.
    Report-mode by default (ledger in AURA_LOG_DIR);
    AURA_TEST_LIVE_DATA_GUARD=fail escalates to hard failures.
    """
    import builtins as _builtins

    from tests.live_data_guard import make_guarded_open

    original = _builtins.open
    _builtins.open = make_guarded_open(request.node.nodeid)
    try:
        yield
    finally:
        _builtins.open = original


_TEST_SCOPED_SERVICE_KEYS = frozenset(
    {
        "advanced_cognition",
        "aura_now",
        "being_runtime",
        "dialogue_cognition",
        "epistemic_reach",
        "relational_memory",
        "scheduler",
        "social_imagination",
        "substrate_voice_engine",
        "thought_interoception",
        "unified_felt_state",
        "unified_will",
        # Registered as a side effect of CONSTRUCTING a GlobalWorkspace, so any
        # consciousness test leaks them and the next test to evict one is
        # blamed for removing shared state it never created.
        "inhibition_manager",
        "unity_runtime",
        "unity_workspace_frame",
        "world_state",
    }
)
_TEST_SCOPED_RESET_FUNCTIONS = (
    ("core.being.runtime", "reset_being_runtime_for_test"),
    ("core.epistemics.epistemic_reach", "reset_epistemic_reach_for_test"),
    ("core.identity.id_rag", "reset_identity_chronicle_for_test"),
    ("core.being.thought_interoception", "reset_thought_interoception_for_test"),
    ("core.being.unified_felt_state", "reset_unified_felt_state_for_test"),
    ("core.governance.will", "reset_unified_will_for_test"),
    ("core.resilience.inhibition_manager", "reset_inhibition_manager_for_test"),
    ("core.unity.runtime", "reset_unity_runtime_for_test"),
    ("core.social.dialogue_cognition", "reset_dialogue_cognition_for_test"),
    ("core.social.relational_memory", "reset_relational_memory_authority"),
    ("core.social.social_imagination", "reset_social_imagination_for_test"),
    ("core.voice.substrate_voice_engine", "reset_substrate_voice_engine_for_test"),
    ("core.world_state", "reset_world_state_for_test"),
)


def _reset_test_scoped_runtime_services() -> None:
    """Close lazy test-owned organs before comparing process-global state."""
    for module_name, function_name in _TEST_SCOPED_RESET_FUNCTIONS:
        module = sys.modules.get(module_name)
        reset = getattr(module, function_name, None) if module is not None else None
        if callable(reset):
            reset()

    scheduler_module = sys.modules.get("core.scheduler")
    scheduler_type = (
        getattr(scheduler_module, "Scheduler", None)
        if scheduler_module is not None
        else None
    )
    scheduler = getattr(scheduler_type, "_instance", None)
    task = getattr(scheduler, "_main_loop_task", None)
    if task is not None and not task.done():
        task.cancel()
    if scheduler_type is not None:
        scheduler_type._instance = None

    container_module = sys.modules.get("core.container")
    container = (
        getattr(container_module, "ServiceContainer", None)
        if container_module is not None
        else None
    )
    services = getattr(container, "_services", None)
    if not isinstance(services, dict):
        return
    keys = [
        key
        for key in list(services)
        if key in _TEST_SCOPED_SERVICE_KEYS or key.startswith("environment_kernel:")
    ]
    lock = getattr(container, "_lock", None)
    if lock is None:
        for key in keys:
            services.pop(key, None)
        return
    with lock:
        for key in keys:
            services.pop(key, None)


@pytest.fixture(autouse=True)
def _environment_learning_isolation(_global_state_contamination_guard):
    """Evict environment-kernel learning services between tests.

    ``EnvironmentKernel`` registers its ``AdvancedCognitionRuntime`` into the
    process-global ServiceContainer; without eviction, every later kernel in
    the same test process reuses the first test's runtime — including its
    accumulated in-memory episodes, so learned risk climbs across tests until
    the advanced-cognition gate starts vetoing benign actions (the in-memory
    twin of the on-disk contamination fixed via AURA_ENV_RUNTIME_DIR).
    Only the kernel-registered learning instances are evicted; the rest of
    the container is untouched.
    """

    _reset_test_scoped_runtime_services()
    yield
    _reset_test_scoped_runtime_services()


@pytest.fixture(autouse=True)
def resource_observer(
    request,
    monkeypatch,
    tmp_path,
    hermetic_resource_sandbox,
    _global_state_contamination_guard,
):
    """Keep ordinary tests independent from the developer host's pressure.

    Tests that genuinely inspect hardware must opt in with ``host_observation``
    (or an existing live/hardware marker).  The ordinary path installs a
    process-wide deterministic observer so worker threads inherit the same
    facts and every pressure result is labelled ``simulated``.
    """
    # These dependencies establish teardown ordering: resource resets first,
    # state comparison second, host leak observation last.
    del hermetic_resource_sandbox, _global_state_contamination_guard

    from core.runtime.resource_observation import (
        HostResourceObserver,
        ObservationSource,
        SimulatedResourceObserver,
        resource_observer_scope,
    )
    from core.runtime.thermal import reset_thermal_cache
    from core.utils.memory_monitor import clear_memory_pressure_snapshot_cache

    runtime_root = tmp_path / "aura-runtime"
    monkeypatch.setenv(
        "AURA_MODEL_LANE_STATE_PATH",
        str(runtime_root / "model_lane_control.json"),
    )
    monkeypatch.setenv("AURA_RECEIPT_ROOT", str(runtime_root / "receipts"))
    monkeypatch.setenv("AURA_MEMORY_SNAPSHOT_CACHE_TTL_S", "0")
    monkeypatch.setenv("AURA_LANE_AUDIT_CACHE_TTL_S", "0")
    monkeypatch.setenv("AURA_TEST_RUNTIME_ROOT", str(runtime_root))
    # 2026-07-23: environment learning sidecars (world model, zero-shot
    # transfer) live under the USER-GLOBAL data dir shared with the live
    # organism. Tests were inheriting learned risk from it and writing test
    # episodes back into it. Every test gets a disposable workspace; a test
    # that genuinely needs the live store must set AURA_ENV_RUNTIME_DIR
    # itself. core/environment/runtime_workspace.py enforces the same rule
    # process-wide for import-time calls no fixture can reach.
    monkeypatch.setenv("AURA_ENV_RUNTIME_DIR", str(runtime_root / "environment_runtime"))
    # Leader-election leases default to the shared data dir, so every parallel
    # chunk — and the live runtime — resolved to the same files. A test could
    # then lose an election to an unrelated process and fail for a reason
    # nothing in it could explain, which is precisely the pass-alone /
    # fail-together shape that makes an aggregate pass count untrustworthy.
    monkeypatch.setenv("AURA_RUNTIME_LEASE_DIR", str(runtime_root / "leases"))
    # 2026-07-28: the screen blueprint reads the real macOS window server
    # in-process, so it is not reachable by the AppleScript mocks the desktop
    # suite uses — a test that mocked "what is frontmost" silently started
    # getting the answer from whatever was actually on screen. Off by default;
    # a test that is genuinely about the blueprint sets it back to "1" itself
    # (see tests/test_screen_blueprint.py).
    monkeypatch.setenv("AURA_SCREEN_BLUEPRINT", "0")

    def _reset_resource_singletons():
        from core.agency.capability_token import reset_token_store
        from core.brain.lane_admission import reset_lane_admission_controller_for_test
        from core.brain.llm.model_registry import reset_model_registry_caches_for_test
        from core.conversation.surface_delivery import reset_route_delivery
        from core.executive.authority_gateway import reset_authority_gateway
        from core.executive.standing_authority import reset_standing_authority_manager
        from core.memory.memory_write_gateway import reset_memory_write_gateway
        from core.resource.resource_governor import reset_resource_governor_for_test
        from core.runtime.control_plane import reset_runtime_control_plane
        from core.runtime.model_lane_control import reset_model_lane_controller_for_test
        from core.runtime.receipts import reset_receipt_store
        from core.runtime.runtime_pressure import reset_unified_runtime_pressure_for_test
        from core.state.state_gateway import reset_state_gateway

        reset_runtime_control_plane()
        reset_authority_gateway()
        reset_standing_authority_manager()
        reset_token_store()
        reset_unified_runtime_pressure_for_test()
        reset_lane_admission_controller_for_test()
        reset_model_lane_controller_for_test()
        reset_resource_governor_for_test()
        reset_model_registry_caches_for_test()
        reset_receipt_store()
        reset_memory_write_gateway()
        reset_state_gateway()
        # "What the route already answered" is process-global, so one test's
        # reply suppressed the next test's identical one as a duplicate:
        # test_ordinary_speech_is_not_withheld passed alone and failed in a
        # chunk, which is the order-dependence shape, not a flake.
        reset_route_delivery()

    host_markers = ("host_observation", "live", "hardware", "longrun")
    host_backed = any(request.node.get_closest_marker(name) for name in host_markers)
    if host_backed:
        observer = HostResourceObserver(
            source=ObservationSource.HOST,
            scenario_id=f"pytest-host:{request.node.nodeid}",
        )
    else:
        observer = SimulatedResourceObserver(
            scenario_id=f"pytest:{request.node.nodeid}",
        )

    clear_memory_pressure_snapshot_cache()
    reset_thermal_cache()
    _reset_resource_singletons()
    with resource_observer_scope(observer):
        yield observer
    _reset_resource_singletons()
    clear_memory_pressure_snapshot_cache()
    reset_thermal_cache()


class _RecordedCall:
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        yield self.args
        yield self.kwargs


class _CallRecorder:
    def __init__(self, result=None, *, side_effect=None):
        self.return_value = result
        self.side_effect = side_effect
        self.calls = []
        self.call_args = None

    @property
    def called(self):
        return bool(self.calls)

    @property
    def call_count(self):
        return len(self.calls)

    def __call__(self, *args, **kwargs):
        call = _RecordedCall(args, kwargs)
        self.calls.append(call)
        self.call_args = call
        if isinstance(self.side_effect, BaseException):
            raise self.side_effect
        if callable(self.side_effect):
            return self.side_effect(*args, **kwargs)
        return self.return_value

    def assert_called_once(self):
        assert len(self.calls) == 1

    def assert_called_once_with(self, *args, **kwargs):
        self.assert_called_once()
        call = self.calls[0]
        assert call.args == args
        assert call.kwargs == kwargs

    def assert_not_called(self):
        assert not self.calls


class _AsyncCallRecorder:
    def __init__(self, result=None, *, side_effect=None):
        self.return_value = result
        self.side_effect = side_effect
        self.await_args_list = []
        self.await_args = None

    @property
    def await_count(self):
        return len(self.await_args_list)

    @property
    def called(self):
        return bool(self.await_args_list)

    def __call__(self, *args, **kwargs):
        call = _RecordedCall(args, kwargs)
        self.await_args_list.append(call)
        self.await_args = call

        async def _complete():
            if isinstance(self.side_effect, BaseException):
                raise self.side_effect
            if callable(self.side_effect):
                value = self.side_effect(*args, **kwargs)
            else:
                value = self.return_value
            if inspect.isawaitable(value):
                return await value
            return value

        return _complete()

    def assert_awaited_once(self):
        assert len(self.await_args_list) == 1

    def assert_not_called(self):
        assert not self.await_args_list


class _TestStorageGateway:
    def create_dir(self, path, *, cause: str = "test"):
        Path(path).mkdir(parents=True, exist_ok=True)

    def delete(self, path, *, cause: str = "test"):
        Path(path).unlink(missing_ok=True)

    def delete_tree(self, path, *, ignore_errors: bool = True, cause: str = "test"):
        shutil.rmtree(path, ignore_errors=ignore_errors)


class _TestTaskTracker:
    def create_task(self, awaitable, *args, **kwargs):
        if not inspect.isawaitable(awaitable):
            return awaitable
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)
        return loop.create_task(awaitable, name=kwargs.get("name"))

    track = create_task
    track_task = create_task


def _test_get_storage_gateway():
    return _TestStorageGateway()


def _test_get_task_tracker():
    return _TestTaskTracker()


builtins.get_storage_gateway = _test_get_storage_gateway
builtins.get_task_tracker = _test_get_task_tracker


@pytest.fixture
def service_container():
    """Provide a fresh ServiceContainer with cleared registry."""
    from core.container import ServiceContainer

    def _resolve_hook(instance, hook_name):
        try:
            inspect.getattr_static(instance, hook_name)
        except (NameError, AttributeError):
            return None
        try:
            hook = getattr(instance, hook_name)
        except (AttributeError, RuntimeError, TypeError):
            return None
        return hook if callable(hook) else None

    def _finish_cleanup(result):
        if inspect.isawaitable(result):
            async def _bounded_cleanup():
                await asyncio.wait_for(result, timeout=_CLEANUP_TIMEOUT_S)

            asyncio.run(_bounded_cleanup())

    def _close_service_instances():
        seen = set()
        for desc in list(getattr(ServiceContainer, "_services", {}).values()):
            instance = getattr(desc, "instance", None)
            if instance is None or id(instance) in seen:
                continue
            seen.add(id(instance))

            for method_name in ("shutdown", "stop", "close"):
                method = _resolve_hook(instance, method_name)
                if method is None:
                    continue
                try:
                    _finish_cleanup(method())
                except (RuntimeError, OSError, ValueError, TypeError, TimeoutError):
                    pass

            db = getattr(instance, "_db", None)
            db_close = _resolve_hook(db, "close") if db is not None else None
            if db_close is not None:
                try:
                    _finish_cleanup(db_close())
                except (RuntimeError, OSError, ValueError, TypeError, TimeoutError):
                    pass
    
    ServiceContainer.clear()
    
    # Snapshot existing registry to restore after test
    original = dict(ServiceContainer._registry) if hasattr(ServiceContainer, "_registry") else {}
    
    yield ServiceContainer

    try:
        from core.utils.task_tracker import get_task_tracker, task_tracker
        # AttributeError included: tests may monkeypatch the tracker with a
        # minimal object without .shutdown; teardown must tolerate it regardless
        # of fixture finalization order, matching the hygiene guard below.
        asyncio.run(get_task_tracker().shutdown(timeout=1.0))
        asyncio.run(task_tracker.shutdown(timeout=1.0))
    except (ImportError, RuntimeError, TimeoutError, AttributeError):
        pass

    try:
        from core.runtime.runtime_hygiene import get_runtime_hygiene

        hygiene = get_runtime_hygiene()
        asyncio.run(hygiene.stop())
        hygiene.reset_state()
    except (ImportError, RuntimeError, TimeoutError, AttributeError):
        pass

    _close_service_instances()
    ServiceContainer.clear()
    try:
        # The personality accessor latches the container's value into a
        # module global that ServiceContainer.clear() cannot purge — a
        # registered test double would otherwise leak into every later test
        # (2026-07-12 order-dependence register).
        from core.brain.personality_engine import reset_personality_engine_for_test

        reset_personality_engine_for_test()
    except ImportError:
        pass
    try:
        # A leaked primary-inference lease defers all background LLM work in
        # later tests for up to 90s (same register, phenomenology victim).
        from core.runtime.backpressure import reset_backpressure_for_test

        reset_backpressure_for_test()
    except ImportError:
        pass

    # Restore original registry
    if hasattr(ServiceContainer, "_registry"):
        ServiceContainer._registry.clear()
        ServiceContainer._registry.update(original)


@pytest.fixture(autouse=True)
def _shutdown_latch_hygiene():
    """Reset the process-global shutdown latch a leaking test leaves set.

    Production shutdown is deliberately MONOTONIC (76e5a71c): once latched,
    nothing may clear it in-process. In the suite that means one test that
    calls request_shutdown without a finally-clear poisons EVERY later test
    in the chunk — gateways/hygiene refuse resource creation, coordinators
    skip handlers — and the victims flap by seed (stem-cell signing,
    graceful-shutdown task tracking, coordinator replay all fell to this
    across three certification runs). Per-test isolation of a deliberately
    monotonic global is exactly a fixture's job; the polluter class is
    unbounded (any test may exercise shutdown), so hygiene lives here.
    """
    yield
    try:
        from core.runtime.shutdown_coordinator import (
            clear_shutdown_request,
            is_shutdown_requested,
        )

        if is_shutdown_requested():
            clear_shutdown_request()
    except (ImportError, AttributeError, RuntimeError):
        pass


@pytest.fixture(autouse=True)
def _disable_redis_event_bus_for_tests():
    """Keep the test suite local-only so Redis client coroutines don't leak warnings."""
    from core import event_bus as event_bus_module
    from core.config import config

    prev_use_for_events = bool(getattr(config.redis, "use_for_events", False))
    prev_bus_use_redis = bool(getattr(event_bus_module.get_event_bus(), "_use_redis", False))
    prev_bus_redis = getattr(event_bus_module.get_event_bus(), "_redis", None)

    config.redis.use_for_events = False
    event_bus_module.get_event_bus()._use_redis = False
    event_bus_module.get_event_bus()._redis = None

    yield

    config.redis.use_for_events = prev_use_for_events
    event_bus_module.get_event_bus()._use_redis = prev_bus_use_redis
    event_bus_module.get_event_bus()._redis = prev_bus_redis


@pytest.fixture(autouse=True)
def _cleanup_runtime_hygiene_after_test():
    yield

    try:
        from core.runtime.runtime_hygiene import get_runtime_hygiene

        hygiene = get_runtime_hygiene()
        asyncio.run(hygiene.stop())
        hygiene.reset_state()
    except (ImportError, RuntimeError, TimeoutError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def _reset_shutdown_request_between_tests():
    try:
        from core.runtime.shutdown_coordinator import clear_shutdown_request

        clear_shutdown_request()
    except (ImportError, RuntimeError, AttributeError):
        pass
    yield
    try:
        from core.runtime.shutdown_coordinator import clear_shutdown_request

        clear_shutdown_request()
    except (ImportError, RuntimeError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def _reset_observation_memory_between_tests():
    """A screen one test looked at is not a screen the next test can see.

    Retained perception is process-global on purpose — her senses outlive
    the turn that filled them — which makes it exactly the kind of state
    that leaks between tests. It did: a capture recorded by the perception
    suite rode into an unrelated desktop-lane test and appended itself to
    that turn's objective.
    """
    def _clear() -> None:
        try:
            from core.perception.observation_evidence import get_observation_memory

            get_observation_memory().clear()
        except (ImportError, RuntimeError, AttributeError):
            pass
        try:
            from core.self.source_excerpt import forget_shown_excerpt

            forget_shown_excerpt()
        except (ImportError, RuntimeError, AttributeError):
            pass

    _clear()
    yield
    _clear()


@pytest.fixture(autouse=True)
def _reset_runtime_degradation_state_between_tests():
    """Keep process-local incidents from contaminating later health assertions."""
    from core.runtime.errors import get_degradation_tracker, get_subsystem_registry

    get_degradation_tracker().reset()
    get_subsystem_registry().reset()
    yield
    get_degradation_tracker().reset()
    get_subsystem_registry().reset()


@pytest.fixture(autouse=True)
def _reset_startup_latch_between_tests():
    """Order-independence for the K2 startup latch.

    The latch is process-global and monotonic by design (a live process
    must never present as "booting" after first readiness). In tests that
    property inverts into cross-test contamination: any test that reaches
    a ready health report latches, and later boot-status tests would see
    "degraded" where they pinned "booting".
    """
    from core.runtime.health_contract import reset_startup_latch_for_test

    reset_startup_latch_for_test()
    yield
    reset_startup_latch_for_test()


@pytest.fixture(autouse=True)
def _reset_crash_loop_breaker_between_tests():
    """Order-independence for the K4 crash-loop breaker.

    Worker-lifecycle tests kill fake workers repeatedly; the process-global
    breaker would trip and refuse spawns in unrelated later tests.
    """
    from core.runtime.lane_reconciler import get_crash_loop_breaker

    get_crash_loop_breaker().reset_for_test()
    yield
    get_crash_loop_breaker().reset_for_test()


@pytest.fixture(autouse=True)
def _pin_measured_phi_off_between_tests(monkeypatch):
    """Order-independence for the unified felt state's measured-Φ track.

    The phi computer is process-global; a test that feeds its trajectory
    would inject a live measurement (and a phi-divergence axis) into any
    later reconcile() call. Tests that want the measured track pass
    measured_phi explicitly or monkeypatch the resolver themselves.
    """
    from core.being.unified_felt_state import UnifiedFeltStateEngine

    monkeypatch.setattr(
        UnifiedFeltStateEngine, "_measured_system_phi", staticmethod(lambda: None)
    )
    yield


@pytest.fixture(autouse=True)
def _reset_escalation_governor_and_conditions_between_tests():
    """Order-independence for the A4 escalation cap and K6 conditions.

    Both are process-global: a test that trips the cap would suppress an
    expected CRITICAL raise in a later test; stale conditions would leak
    into later condition assertions.
    """
    from core.runtime.conditions import reset_conditions_for_test
    from core.runtime.errors import get_escalation_governor

    get_escalation_governor().reset_for_test()
    reset_conditions_for_test()
    yield
    get_escalation_governor().reset_for_test()
    reset_conditions_for_test()


@pytest.fixture(autouse=True)
def _service_registry_state_guard():
    """Order-independence for the low-level runtime service registry.

    Registry resolvers/sinks are process-global. Two contamination
    directions were observed in-chunk (defect register, July 3):
    tests that install a fake resolver and leak it forward, and tests
    that "clean up" by installing None — ERASING the container-backed
    resolver later tests depend on. Snapshot-and-restore fixes both
    without touching individual call sites.
    """
    import core.runtime.service_registry as _registry

    guarded = [
        name for name in dir(_registry)
        if name.startswith("_") and name.endswith(("_resolver", "_sink"))
    ]
    snapshot = {name: getattr(_registry, name) for name in guarded}
    yield
    for name, value in snapshot.items():
        setattr(_registry, name, value)


@pytest.fixture(autouse=True)
def _contain_governance_strictness_between_tests():
    """Order-independence for governance enforcement.

    governance_runtime_active() flips strict once kernel-marker services
    exist or container registration locks. Tests that register those and
    don't clean up made every later gateway write in the same process
    fail with GovernanceViolationError (observed across whole chunks).
    This guard restores only the governance-flipping state — marker
    services added during the test and the registration lock — leaving
    all other registrations untouched.
    """
    from core.container import ServiceContainer

    markers = ("executive_core", "aura_kernel", "kernel_interface")
    before = {name: ServiceContainer.has(name) for name in markers}
    locked_before = bool(getattr(ServiceContainer, "_registration_locked", False))
    yield
    try:
        services = getattr(ServiceContainer, "_services", None)
        aliases = getattr(ServiceContainer, "_aliases", {})
        if services is not None:
            for name in markers:
                if not before[name] and ServiceContainer.has(name):
                    resolved = aliases.get(name, name)
                    services.pop(resolved, None)
                    services.pop(name, None)
        if not locked_before and getattr(
            ServiceContainer, "_registration_locked", False
        ):
            ServiceContainer._registration_locked = False
    except (AttributeError, RuntimeError, TypeError):
        pass


@pytest.fixture(autouse=True)
def _reset_foreground_guard_between_tests():
    """Order-independence: chat-route tests leave the module-global
    foreground quiet window armed, which made unrelated suites (e.g.
    flagship doctor idle-context assertions) fail when run together."""
    yield
    try:
        from core.runtime.foreground_guard import _reset_for_tests

        _reset_for_tests()
    except (ImportError, RuntimeError, AttributeError):
        pass


def pytest_sessionfinish(session, exitstatus):
    """Final cleanup for singleton executors that can keep pytest alive.

    The suite creates long-lived runtime services on purpose.  Unit tests should
    not leave their ThreadPool/ProcessPool workers attached to the pytest
    process after all assertions have completed.
    """
    try:
        from core.bus.local_pipe_bus import LocalPipeBus

        LocalPipeBus.shutdown_executor()
    except (ImportError, RuntimeError, AttributeError):
        pass


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Let the audit script exit after pytest has printed its real summary.

    Some integration tests intentionally touch long-lived runtime primitives
    whose atexit joins can keep the interpreter alive after all assertions have
    passed.  The audit runner opts into this hook so a green or red pytest
    status is preserved exactly, while leaked background threads cannot leave
    orphaned test processes.
    """
    if os.environ.get("AURA_PYTEST_FORCE_EXIT_AFTER_SUMMARY", "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(int(exitstatus))

    try:
        from core.utils.executor import shutdown_executors

        shutdown_executors()
    except (ImportError, RuntimeError, AttributeError):
        pass

    try:
        from core.consciousness.hierarchical_phi import get_hierarchical_phi

        get_hierarchical_phi().shutdown()
    except (ImportError, RuntimeError, AttributeError):
        pass

    try:
        from core.container import ServiceContainer

        asyncio.run(ServiceContainer.shutdown())
    except (ImportError, RuntimeError, TimeoutError, AttributeError):
        pass

@pytest.fixture
def mock_container(service_container):
    """Full architectural collaborator registry for Aura tests."""

    from core.container import ServiceContainer
    
    agency_bus = SimpleNamespace(submit=_CallRecorder(result=True))
    
    from core.agency_bus import AgencyBus

    original_get = AgencyBus.get
    AgencyBus.get = classmethod(lambda cls: agency_bus)
    try:
        # Brain / Cognitive Engine
        async def recorded_stream(*args, **kwargs):
            yield "Recorded "
            yield "stream"

        cognition = SimpleNamespace(
            record_interaction=_AsyncCallRecorder(),
            process_turn=_AsyncCallRecorder("Recorded response"),
            think=_AsyncCallRecorder(SimpleNamespace(content="Recorded thought")),
            think_stream=recorded_stream,
        )
        
        # Memory
        memory = SimpleNamespace(
            retrieve_unified_context=_AsyncCallRecorder("Memories"),
            commit_interaction=_AsyncCallRecorder(),
            run_maintenance=_AsyncCallRecorder(),
            get_hot_memory=_AsyncCallRecorder({}),
            get_cold_memory_context=_AsyncCallRecorder(""),
            store=_AsyncCallRecorder(),
        )
        
        # Meta-Learning
        meta = SimpleNamespace(
            recall_strategy=_AsyncCallRecorder({}),
            index_experience=_AsyncCallRecorder(),
            run_maintenance=_AsyncCallRecorder(),
        )

        personality = SimpleNamespace(
            update=_CallRecorder(),
            filter_response=_CallRecorder(side_effect=lambda text: text),
            get_emotional_context_for_response=_CallRecorder(
                {"mood": "neutral", "tone": "balanced", "emotional_state": {}}
            ),
            get_time_context=_CallRecorder({"formatted": "12:00 PM"}),
            get_sovereign_context=_CallRecorder(""),
            current_mood="balanced",
        )

        strategic_planner = SimpleNamespace(get_next_task=_CallRecorder())

        project_store = SimpleNamespace(
            get_active_projects=_CallRecorder([]),
            get_tasks_for_project=_CallRecorder([]),
        )

        knowledge_graph = SimpleNamespace(
            add_knowledge=_CallRecorder(),
            remember_person=_CallRecorder(),
            ask_question=_CallRecorder(),
        )
        
        # Senses & State
        liquid_state = SimpleNamespace(
            update=_AsyncCallRecorder(),
            get_status=_CallRecorder({"health": 1.0, "status": {"initialized": True, "running": True}}),
            current=SimpleNamespace(curiosity=0.5, frustration=0.1, energy=0.8),
        )
        
        affect = SimpleNamespace(
            state=SimpleNamespace(dominant_emotion="Joy"),
            get_current_state=_CallRecorder({"valence": 0.5}),
        )
        
        # Core Registry
        ServiceContainer.register_instance("cognitive_engine", cognition)
        ServiceContainer.register_instance("cognition", cognition)
        ServiceContainer.register_instance("memory", memory)
        ServiceContainer.register_instance("memory_facade", memory)
        ServiceContainer.register_instance("metacognition", meta)
        ServiceContainer.register_instance("meta_learning", meta)
        ServiceContainer.register_instance("personality_engine", personality)
        ServiceContainer.register_instance("strategic_planner", strategic_planner)
        ServiceContainer.register_instance("project_store", project_store)
        ServiceContainer.register_instance("knowledge_graph", knowledge_graph)
        ServiceContainer.register_instance("affect_engine", affect)
        ServiceContainer.register_instance("liquid_state", liquid_state)
        ServiceContainer.register_instance("conscious_substrate", liquid_state)
        
        # Infrastructure
        ServiceContainer.register_instance("watchdog", SimpleNamespace())
        ServiceContainer.register_instance("output_gate", SimpleNamespace(emit=_AsyncCallRecorder()))
        ServiceContainer.register_instance(
            "capability_engine",
            SimpleNamespace(execute=_AsyncCallRecorder({"ok": True})),
        )
        
        # Fallbacks for missing services identified in audit
        drives = SimpleNamespace(satisfy=_AsyncCallRecorder())
        alignment = SimpleNamespace(
            filter_response=_AsyncCallRecorder(side_effect=lambda x, *args, **kwargs: x)
        )
        for svc in ["homeostasis", "subsystem_audit", "lnn", "mortality", "identity", "curiosity",
                    "intent_router", "cognitive_router", "world_model",
                    "belief_graph"]:
            ServiceContainer.register_instance(svc, _AsyncCallRecorder())
        ServiceContainer.register_instance("mycelium", SimpleNamespace())
        ServiceContainer.register_instance("state_machine", SimpleNamespace())
        ServiceContainer.register_instance("drives", drives)
        ServiceContainer.register_instance("alignment_engine", alignment)
            
        yield ServiceContainer
    finally:
        AgencyBus.get = original_get

@pytest.fixture
def orchestrator(mock_container):
    """Hardened RobustOrchestrator fixture with full dependency injection."""
    import asyncio
    import time

    from core.orchestrator import RobustOrchestrator
    from core.orchestrator.orchestrator_types import SystemStatus

    # Initialize instance WITHOUT class patching
    orch = RobustOrchestrator()
    
    # Setup core status
    status_obj = SystemStatus()
    status_obj.initialized = True
    status_obj.running = True
    status_obj.cycle_count = 0
    status_obj.start_time = time.time()
    orch.status = status_obj
    
    # Ensure queues and locks exist
    orch.message_queue = asyncio.Queue()
    orch.reply_queue = asyncio.Queue()
    orch._lock = asyncio.Lock()
    orch._history_lock = asyncio.Lock()
    
    # Setup core dependencies from container
    for component in ["cognitive_engine", "memory", "capability_engine", 
                     "strategic_planner", "project_store", "intent_router",
                     "personality_engine", "world_model", "curiosity",
                     "knowledge_graph", "drives", "state_machine", 
                     "output_gate", "liquid_state", "mycelium"]:
        svc = mock_container.get(component)
        if component == "mycelium":
            # Mycelium has sync methods like match_hardwired and rooted_flow call
            from core.orchestrator.main import AsyncNullContext
            svc = SimpleNamespace(
                rooted_flow=_CallRecorder(AsyncNullContext()),
                match_hardwired=_CallRecorder(),
            )
        elif component == "state_machine":
             svc = SimpleNamespace(execute=_AsyncCallRecorder())
        elif component == "intent_router":
             svc = SimpleNamespace(classify=_AsyncCallRecorder("chitchat"))
        elif component == "output_gate":
             svc = SimpleNamespace(emit=_AsyncCallRecorder())
        setattr(orch, component, svc)
        setattr(orch, f"_{component}", svc)
    
    # Provide async test doubles expected by existing orchestrator tests.
    orch.hooks = SimpleNamespace(trigger=_AsyncCallRecorder())
    
    # Ensure _finalize_response and _handle_incoming_message remain real
    # unless a specific test replaces them.
    
    try:
        yield orch
    finally:
        status = getattr(orch, "status", None)
        if status is not None:
            if hasattr(status, "running"):
                status.running = False
            if hasattr(status, "is_processing"):
                status.is_processing = False

        stop_event = getattr(orch, "_stop_event", None)
        if stop_event is not None and hasattr(stop_event, "set"):
            stop_event.set()

        async def _cleanup_tasks():
            for attr in ("_current_thought_task", "_autonomous_task"):
                task = getattr(orch, attr, None)
                if isinstance(task, asyncio.Task) and not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.wait_for(task, timeout=_CLEANUP_TIMEOUT_S)

        try:
            asyncio.run(_cleanup_tasks())
        except (RuntimeError, TimeoutError, ValueError) as exc:
            warnings.warn(
                f"orchestrator fixture task cleanup did not complete cleanly: {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )


@pytest.fixture(autouse=True)
def _reset_health_caches_between_tests():
    """Health payloads are memoised for 5s; scenarios are not.

    Without this, a test that installs a ready boot snapshot could read a
    payload captured by an unrelated test moments earlier — passing alone and
    failing in company, which is the signature of order dependence rather than
    a defect in the code under test.
    """

    def _reset():
        try:
            from interface.routes.system import reset_health_caches

            reset_health_caches()
        except (ImportError, RuntimeError, AttributeError):
            pass

    _reset()
    yield
    _reset()


@pytest.fixture(autouse=True)
def _reset_working_memory_queue_load_between_tests():
    """The spiking-inference queue model accumulates load across calls.

    That is right for a running mind and wrong for a process running many
    independent scenarios: load left by one test turns the next one's
    admission decision from "accept" into "compress_foreground".
    """

    def _reset():
        try:
            import sys

            module = sys.modules.get("core.cognitive.spiking_active_inference")
            if module is None:
                return
            # Reset only an advisor that already exists. Constructing one here
            # would drag cognition into every test, including the ones that
            # assert a deterministic path never builds a CognitiveEngine.
            advisor = getattr(module, "_ADVISOR", None)
            if advisor is None:
                return
            queue = getattr(advisor, "_working_memory", None)
            if queue is not None and hasattr(queue, "reset"):
                queue.reset()
        except (ImportError, RuntimeError, AttributeError):
            pass

    _reset()
    yield
    _reset()


# ── Attribution for cross-test contamination ──────────────────────────────
#
# The chunk runner already reports order-dependence: a test that fails in a
# chunk and passes alone. What it cannot say is WHICH earlier test caused it.
# So the victim gets investigated and the polluter keeps running, and the only
# available remedy is to distrust the whole aggregate.
#
# This snapshots the process-global surfaces a test has no business changing
# and attributes any change to the test that made it. Report-mode by default —
# a wall of failures on first run teaches people to disable the guard — and
# AURA_TEST_STATE_GUARD=fail makes it enforcing, the same escalation the live
# data guard uses.
_STATE_GUARD_LEDGER: list[str] = []


def _global_state_fingerprint() -> dict[str, object]:
    fingerprint: dict[str, object] = {}
    try:
        from core.container import ServiceContainer

        services = getattr(ServiceContainer, "_services", None)
        if isinstance(services, dict):
            fingerprint["service_container"] = frozenset(services)
    except (ImportError, AttributeError, RuntimeError, TypeError):
        pass
    try:
        import os

        fingerprint["cwd"] = os.getcwd()
        # An AURA_* variable set without monkeypatch outlives the test and
        # silently reconfigures every later one. This is the leak that made a
        # latent-cortex authority test fail only when a governance suite ran
        # first, and it is invisible to the ServiceContainer snapshot.
        fingerprint["aura_env"] = frozenset(
            f"{key}={value}"
            for key, value in os.environ.items()
            if key.startswith("AURA_")
        )
    except OSError:
        pass
    try:
        # Installed resolvers and sinks are process-global by design: they are
        # how the runtime is wired once at boot. A test that installs one and
        # does not remove it rewires every later test's view of the runtime,
        # and nothing about the victim shows where it came from.
        from core.runtime import service_registry as _registry

        fingerprint["installed_resolvers"] = frozenset(
            name
            for name in dir(_registry)
            if name.startswith("_") and not name.startswith("__")
            and getattr(_registry, name, None) is not None
            and callable(getattr(_registry, name, None)) is False
        )
    except (ImportError, AttributeError, RuntimeError, TypeError):
        pass
    try:
        import sys

        # A test that leaves a mock in sys.modules under a real module name
        # silently rewires every later import of it.
        fingerprint["mocked_core_modules"] = frozenset(
            name
            for name, module in list(sys.modules.items())
            if name.startswith(("core.", "interface."))
            and module is not None
            and not hasattr(module, "__file__")
        )
    except (ImportError, AttributeError, RuntimeError):
        pass
    return fingerprint


@pytest.fixture(autouse=True)
def _global_state_contamination_guard(request, hermetic_resource_sandbox):
    """Name the test that dirtied shared state, not the one that tripped over it."""
    import os

    del hermetic_resource_sandbox  # host leak observation must run after this guard

    if request.node.get_closest_marker("mutates_global_state"):
        yield
        return

    _reset_test_scoped_runtime_services()
    before = _global_state_fingerprint()
    try:
        yield
    finally:
        _reset_test_scoped_runtime_services()
        after = _global_state_fingerprint()
        changes: list[str] = []
        for key in sorted(set(before) | set(after)):
            old_value, new_value = before.get(key), after.get(key)
            if old_value == new_value:
                continue
            if isinstance(old_value, frozenset) and isinstance(new_value, frozenset):
                added = sorted(new_value - old_value)[:6]
                removed = sorted(old_value - new_value)[:6]
                detail = []
                if added:
                    detail.append(f"added {', '.join(map(str, added))}")
                if removed:
                    detail.append(f"removed {', '.join(map(str, removed))}")
                if detail:
                    changes.append(f"{key}: {'; '.join(detail)}")
            else:
                changes.append(f"{key}: {old_value!r} -> {new_value!r}")
        if changes:
            message = (
                f"{request.node.nodeid} left shared state changed: " + " | ".join(changes)
            )
            _STATE_GUARD_LEDGER.append(message)
            if str(os.environ.get("AURA_TEST_STATE_GUARD", "")).strip().lower() == "fail":
                pytest.fail(message, pytrace=False)
            print(f"\n[state-guard] {message}")


@pytest.fixture(autouse=True)
def _mlx_clients_do_not_outlive_their_test(request):
    """Close MLX clients a test created, inside that test.

    An MLXLocalClient's finalizer releases its durable lane. A client left for
    the garbage collector releases whenever the collector happens to run —
    which is inside some LATER test, into whatever recorder that test
    installed. Measured: test_forced_abort_releases_exact_durable_lane_owner
    saw an extra release from a previous test's client,

        ('mlx:8733:/private/var/.../Qwen2.5-32B-Instruct-8bit', 1, 'client_close')

    and both it and test_mlx_force_abort_kills_worker_before_lifecycle_lock_
    cleanup passed alone and failed together — the pass-alone / fail-together
    shape that makes an aggregate green untrustworthy.

    Closing here, then collecting, keeps every finalizer inside the test that
    created the object.
    """
    yield
    # SCOPED BY WHO CAN CREATE ONE, not by what happens to be in the registry.
    #
    # The leaking client is built directly — MLXLocalClient(...) — so it never
    # enters _CLIENTS, which means gating on that registry skipped the very
    # case this exists for (measured: the failure came straight back). The
    # collect is what does the work.
    #
    # But an unconditional collect after every test costs real minutes across
    # ~7,400 of them: this sweep went from ~14 to 20+ when the fixture landed.
    # Only modules that can build one need paying for.
    module = str(getattr(request.node, "fspath", "") or "")
    if "mlx" not in module.lower() and "cortex" not in module.lower():
        return

    import gc

    try:
        from core.brain.llm import mlx_client as _mlx
    except Exception:  # noqa: BLE001 - the module may not be importable here
        return
    registry = getattr(_mlx, "_CLIENTS", None)
    if isinstance(registry, dict) and registry:
        for client in list(registry.values()):
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - teardown may never fail a test
                    pass
        registry.clear()
    gc.collect()
