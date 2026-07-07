from __future__ import annotations

import time

from core.runtime.liveness import (
    clear_runtime_service_progress,
    get_runtime_service_progress,
    mark_runtime_service_progress,
)


def test_runtime_liveness_registry_records_and_filters_progress():
    clear_runtime_service_progress()

    mark_runtime_service_progress("web_interlocutor.job.one")
    time.sleep(0.001)
    mark_runtime_service_progress("api.health.heartbeat")
    mark_runtime_service_progress("web_interlocutor.job.one")

    web = get_runtime_service_progress("web_interlocutor")
    assert web["ok"] is True
    assert web["source"] == "web_interlocutor.job.one"
    assert web["count"] == 2
    assert web["matches"] == 1
    assert web["age_s"] >= 0.0

    any_progress = get_runtime_service_progress()
    assert any_progress["ok"] is True
    assert any_progress["source"] in {"web_interlocutor.job.one", "api.health.heartbeat"}


def test_runtime_liveness_registry_reports_empty_state():
    clear_runtime_service_progress()

    assert get_runtime_service_progress("missing") == {
        "ok": False,
        "source": "",
        "updated_at": 0.0,
        "age_s": None,
        "count": 0,
        "matches": 0,
    }
