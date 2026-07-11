from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from core.runtime.root_signal_owner import RootShutdownSignalOwner
from core.runtime.shutdown_coordinator import (
    clear_shutdown_request,
    publish_root_exit_verdict,
    publish_shutdown_verdict,
    request_shutdown,
    shutdown_request_snapshot,
)


@pytest.fixture(autouse=True)
def _clean_shutdown_latch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.setenv("HOME", str(tmp_path))
    clear_shutdown_request()
    yield
    clear_shutdown_request()


def test_repeated_root_signals_preserve_first_reason_and_count_every_request() -> None:
    observed: list[tuple[signal.Signals, dict[str, object]]] = []
    owner = RootShutdownSignalOwner(
        scope="desktop_signal",
        observer=lambda sig, snapshot: observed.append((sig, snapshot)),
    )

    owner._handle_signal(signal.SIGTERM)
    owner._handle_signal(signal.SIGINT)

    snapshot = shutdown_request_snapshot()
    assert owner.requested is True
    assert owner.first_reason == "desktop_signal:SIGTERM"
    assert snapshot["first_reason"] == "desktop_signal:SIGTERM"
    assert snapshot["last_reason"] == "desktop_signal:SIGINT"
    assert snapshot["request_count"] == 2
    assert [item[0] for item in observed] == [signal.SIGTERM, signal.SIGINT]


def test_bootstrap_handlers_remain_owned_until_explicit_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registrations: list[tuple[signal.Signals, object]] = []

    def _record_signal(sig: signal.Signals, handler: object) -> object:
        registrations.append((sig, handler))
        return signal.SIG_DFL

    monkeypatch.setattr(signal, "signal", _record_signal)
    owner = RootShutdownSignalOwner(scope="server_signal")

    assert owner.install_bootstrap() == 2
    assert owner.install_bootstrap() == 2
    owner.close()

    installed = registrations[:2]
    removed = registrations[2:]
    assert [item[0] for item in installed] == [signal.SIGINT, signal.SIGTERM]
    assert all(callable(item[1]) for item in installed)
    assert [item[0] for item in removed] == [signal.SIGINT, signal.SIGTERM]
    assert all(item[1] is signal.SIG_DFL for item in removed)


def test_event_loop_handlers_replace_bootstrap_owner_and_remove_after_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    added: list[
        tuple[signal.Signals, Callable[[signal.Signals], None], signal.Signals]
    ] = []
    removed: list[signal.Signals] = []

    class _Loop:
        def add_signal_handler(
            self,
            sig: signal.Signals,
            callback: Callable[[signal.Signals], None],
            callback_sig: signal.Signals,
        ) -> None:
            added.append((sig, callback, callback_sig))

        def remove_signal_handler(self, sig: signal.Signals) -> bool:
            removed.append(sig)
            return True

    owner = RootShutdownSignalOwner(scope="server_signal")
    monkeypatch.setattr(signal, "signal", lambda *_args: signal.SIG_DFL)
    owner.install_bootstrap()
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: _Loop())

    assert owner.install() == 2
    added[0][1](added[0][2])
    owner.close()

    assert owner.requested is True
    assert owner.first_reason == "server_signal:SIGINT"
    assert [item[0] for item in added] == [signal.SIGINT, signal.SIGTERM]
    assert removed == [signal.SIGINT, signal.SIGTERM]


def test_process_lifetime_owner_hands_signals_back_before_loop_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synchronous_handlers: list[tuple[signal.Signals, object]] = []
    removed: list[signal.Signals] = []

    class _Loop:
        def add_signal_handler(self, *_args) -> None:
            return None

        def remove_signal_handler(self, sig: signal.Signals) -> bool:
            removed.append(sig)
            return True

    monkeypatch.setattr(
        signal,
        "signal",
        lambda sig, handler: synchronous_handlers.append((sig, handler)) or signal.SIG_DFL,
    )
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: _Loop())
    owner = RootShutdownSignalOwner(scope="desktop_signal")
    owner.retain_for_process_exit()
    owner.install_bootstrap()
    owner.install()

    owner.finish_async_ownership()

    assert removed == [signal.SIGINT, signal.SIGTERM]
    handoff = synchronous_handlers[-2:]
    assert [item[0] for item in handoff] == [signal.SIGINT, signal.SIGTERM]
    assert all(callable(item[1]) for item in handoff)


def test_root_exit_receipt_advances_final_verdict_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "run" / "shutdown_report.json"
    monkeypatch.setenv("AURA_SHUTDOWN_REPORT_PATH", str(report_path))
    request_shutdown("desktop_signal:SIGTERM")
    prior = publish_shutdown_verdict(
        coordinator_report={"clean": True, "completed_phases": ["output_flush"]},
        container_report={"clean": True},
        runtime_hygiene_report={"clean": True},
        stage="graceful_shutdown_complete",
        final=True,
    )
    assert prior["verdict"] == {"clean": True, "blockers": []}

    terminal = publish_root_exit_verdict(
        lock_released=True,
        finalizers_completed=True,
        logging_shutdown_completed=True,
        root_resource_report={"clean": True, "blockers": []},
        exit_code=0,
    )

    assert terminal["stage"] == "root_process_exit"
    assert terminal["terminal_receipt_sequence"] == 1
    assert terminal["root_exit"] == {
        "lock_released": True,
        "multiprocessing_finalizers_completed": True,
        "logging_shutdown_completed": True,
        "exit_code": 0,
        "completed_at_unix": terminal["root_exit"]["completed_at_unix"],
        "resources": {"clean": True, "blockers": []},
    }
    assert terminal["verdict"] == {"clean": True, "blockers": []}
    assert len(list((report_path.parent / "shutdown_history").glob("*.json"))) == 1
    with pytest.raises(RuntimeError, match="already published"):
        publish_root_exit_verdict(
            lock_released=True,
            finalizers_completed=True,
            logging_shutdown_completed=True,
            root_resource_report={"clean": True, "blockers": []},
            exit_code=0,
        )


def test_shutdown_coordinator_exposes_stable_phase_start_marker() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "core"
        / "runtime"
        / "shutdown_coordinator.py"
    ).read_text(encoding="utf-8")

    assert "ShutdownCoordinator: phase started" in source
    assert "phase=%s handlers=%d timeout_seconds=%.3f" in source
