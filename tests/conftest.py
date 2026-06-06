"""Shared pytest fixtures for Aura smoke tests."""
import asyncio
import builtins
import contextlib
import inspect
import os
import shutil
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

_CLEANUP_TIMEOUT_S = 2.0

# Ensure the project root is on sys.path so `core.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
        asyncio.run(get_task_tracker().shutdown(timeout=1.0))
        asyncio.run(task_tracker.shutdown(timeout=1.0))
    except (ImportError, RuntimeError, TimeoutError):
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

    # Restore original registry
    if hasattr(ServiceContainer, "_registry"):
        ServiceContainer._registry.clear()
        ServiceContainer._registry.update(original)


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
