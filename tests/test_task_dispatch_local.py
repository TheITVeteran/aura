"""tests/test_task_dispatch_local.py
========================================
The desktop/consumer build must run with NO Celery/Redis daemon. When no broker
is configured, background work must execute *locally* — never be handed to the
MockCelery shim, whose ``send_task`` only RECORDS the call and would silently
drop the task (the original consumer-path bug).

These tests lock in:
  - RedisConfig defaults OFF (no daemon required out of the box),
  - ``_broker_active()`` is False unless a real broker is both importable and
    explicitly enabled,
  - ``dispatch_user_input`` runs the work in-process when there is no broker,
  - ``dispatch_background`` runs the registered handler off the event loop and
    routes to Celery only when the broker is genuinely active,
  - an unknown task name is reported (``"skipped"``), never silently lost.
"""
from __future__ import annotations

import threading
import time

import core.tasks as tasks_module
from core.config import RedisConfig
from core.tasks import dispatch_background, dispatch_user_input


def test_redis_defaults_off(monkeypatch):
    # No env set → consumer default is no daemon.
    for var in ("AURA_REDIS_ENABLED", "AURA_REDIS_USE_FOR_EVENTS"):
        monkeypatch.delenv(var, raising=False)
    cfg = RedisConfig()
    assert cfg.enabled is False
    assert cfg.use_for_events is False


def test_redis_opt_in_via_env(monkeypatch):
    monkeypatch.setenv("AURA_REDIS_ENABLED", "1")
    monkeypatch.setenv("AURA_REDIS_USE_FOR_EVENTS", "true")
    cfg = RedisConfig()
    assert cfg.enabled is True
    assert cfg.use_for_events is True


def test_broker_inactive_by_default(monkeypatch):
    # Even if redis.enabled were flipped, no real Celery package → not active.
    monkeypatch.setattr(tasks_module, "CELERY_AVAILABLE", False)
    assert tasks_module._broker_active() is False


def test_dispatch_user_input_runs_locally_when_no_broker(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(tasks_module, "_broker_active", lambda: False)
    monkeypatch.setattr(tasks_module, "process_user_input", lambda msg: seen.append(msg))
    dispatch_user_input("hello world")
    assert seen == ["hello world"]


def test_dispatch_user_input_does_not_drop_to_mock_celery(monkeypatch):
    """With a non-executing shim active, the message must still be processed."""
    processed: list[str] = []
    sent: list[object] = []
    monkeypatch.setattr(tasks_module, "_broker_active", lambda: False)
    monkeypatch.setattr(tasks_module, "process_user_input", lambda msg: processed.append(msg))
    monkeypatch.setattr(
        tasks_module.celery_app, "send_task", lambda *a, **k: sent.append((a, k))
    )
    dispatch_user_input("do not drop me")
    assert processed == ["do not drop me"]
    assert sent == []  # send_task (the dropping shim path) was never called


def test_dispatch_background_runs_handler_off_event_loop(monkeypatch):
    ran = threading.Event()
    captured: dict[str, object] = {}

    def _handler(value, *, flag=None):
        captured["value"] = value
        captured["flag"] = flag
        captured["thread"] = threading.current_thread().name
        ran.set()

    monkeypatch.setattr(tasks_module, "_broker_active", lambda: False)
    monkeypatch.setitem(tasks_module._LOCAL_TASKS, "test.local_handler", _handler)

    route = dispatch_background("test.local_handler", args=["payload"], kwargs={"flag": True})
    assert route == "local"
    assert ran.wait(timeout=5.0), "local background handler did not run"
    assert captured["value"] == "payload"
    assert captured["flag"] is True
    # Ran on a dedicated daemon thread, not the caller's thread.
    assert captured["thread"].startswith("aura-task-")


def test_dispatch_background_routes_to_celery_when_active(monkeypatch):
    sent: list[tuple] = []
    monkeypatch.setattr(tasks_module, "_broker_active", lambda: True)
    monkeypatch.setattr(
        tasks_module.celery_app,
        "send_task",
        lambda name, args=None, kwargs=None: sent.append((name, args, kwargs)),
    )
    route = dispatch_background("core.tasks.run_self_update")
    assert route == "celery"
    assert sent == [("core.tasks.run_self_update", [], {})]


def test_dispatch_background_celery_failure_falls_back_local(monkeypatch):
    ran = threading.Event()
    send_attempts = []
    monkeypatch.setattr(tasks_module, "_broker_active", lambda: True)

    def _boom(*a, **k):
        send_attempts.append((a, k))
        raise RuntimeError("broker unreachable")

    monkeypatch.setattr(tasks_module.celery_app, "send_task", _boom)
    monkeypatch.setitem(tasks_module._LOCAL_TASKS, "test.fallback", lambda: ran.set())
    route = dispatch_background("test.fallback")
    assert route == "local"
    assert len(send_attempts) == 1
    assert ran.wait(timeout=5.0), "fallback to local handler did not run"


def test_dispatch_background_unknown_task_is_reported_not_dropped(monkeypatch, caplog):
    monkeypatch.setattr(tasks_module, "_broker_active", lambda: False)
    with caplog.at_level("ERROR"):
        route = dispatch_background("core.tasks.does_not_exist")
    assert route == "skipped"
    assert any("does_not_exist" in r.message for r in caplog.records)


def test_local_task_registry_matches_celery_task_names():
    """Every registered local handler maps to its real module-level function."""
    for name, fn in tasks_module._LOCAL_TASKS.items():
        assert callable(fn)
        short = name.rsplit(".", 1)[-1]
        assert getattr(tasks_module, short) is fn


def test_local_background_failure_does_not_raise(monkeypatch):
    """A failing local handler must be recorded as degradation, not crash."""
    degraded: list[tuple] = []
    handler_attempted = threading.Event()
    monkeypatch.setattr(tasks_module, "_broker_active", lambda: False)
    monkeypatch.setattr(
        tasks_module, "record_degradation", lambda *a, **k: degraded.append((a, k))
    )

    def _boom():
        handler_attempted.set()
        raise ValueError("handler exploded")

    monkeypatch.setitem(tasks_module._LOCAL_TASKS, "test.boom", _boom)
    route = dispatch_background("test.boom")
    assert route == "local"
    # Give the daemon thread a moment to run and record the degradation.
    deadline = time.time() + 5.0
    while not degraded and time.time() < deadline:
        time.sleep(0.02)
    assert handler_attempted.is_set()
    assert degraded, "local handler failure was not recorded as degradation"
