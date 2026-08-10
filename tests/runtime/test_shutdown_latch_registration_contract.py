"""A quit during boot must not look like a wiring bug.

LIVE DEFECT, seen in ~/.aura/logs/desktop-launch.log (2026-08-09 16:17).

The runtime received SIGTERM while it was still booting. The API server's
lifespan went on to register the whole service graph anyway; the shutdown
latch suppressed every registration SOFTLY (``register`` returns without
registering) while ``get`` still raised HARD, so the first register-then-get
pair produced:

    ServiceNotFoundError: Service 'defensive_runtime' not found in
    static registry.
    ERROR:    Application startup failed. Exiting.

Three separate faults in one line: the message named the wrong cause, the
handler could not catch it (ContainerError descends from AuraError, not
RuntimeError, so the except tuple was inert), and the lifespan should never
have been booting a runtime that had already been told to quit.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def shutdown_requested():
    """Hold the real shutdown latch for the duration of a test."""
    from core.runtime.shutdown_coordinator import (
        clear_shutdown_request,
        is_shutdown_requested,
        request_shutdown,
    )

    request_shutdown("test:boot_race")
    if not is_shutdown_requested():
        pytest.skip("shutdown latch not settable in this runtime")
    try:
        yield
    finally:
        clear_shutdown_request()


def test_container_error_is_not_a_runtime_error() -> None:
    """The premise of the dead except tuple, pinned.

    If this ever becomes true, the original handler would have worked and
    this whole contract can be revisited.
    """
    from core.exceptions import ContainerError, ServiceNotFoundError

    assert not issubclass(ContainerError, RuntimeError)
    assert issubclass(ServiceNotFoundError, ContainerError)


def test_absent_service_reports_wiring_bug_when_not_shutting_down() -> None:
    from core.container import ServiceContainer
    from core.exceptions import ServiceNotFoundError

    with pytest.raises(ServiceNotFoundError) as caught:
        ServiceContainer.get("a_service_nobody_ever_registered")

    assert "not found in static registry" in str(caught.value)


@pytest.mark.usefixtures("shutdown_requested")
def test_absent_service_blames_the_shutdown_latch_during_teardown() -> None:
    """The message must name the real cause, not "wiring bug"."""
    from core.container import ServiceContainer
    from core.exceptions import ContainerError

    with pytest.raises(ContainerError) as caught:
        ServiceContainer.get("a_service_suppressed_by_the_latch")

    message = str(caught.value)
    assert "shutdown" in message.lower()
    assert "not found in static registry" not in message


@pytest.mark.usefixtures("shutdown_requested")
def test_register_then_get_is_catchable_by_the_registration_handler() -> None:
    """The exact live sequence: register (suppressed) then get (raised).

    The handler in service_registration catches ContainerError, so the
    degradation path can run instead of the lifespan dying.
    """
    from core.container import ServiceContainer
    from core.exceptions import ContainerError

    ServiceContainer.register(
        "defensive_runtime_probe",
        lambda: object(),
        required=False,
    )

    with pytest.raises(ContainerError):
        ServiceContainer.get("defensive_runtime_probe")


@pytest.mark.usefixtures("shutdown_requested")
def test_default_still_wins_over_raising_during_shutdown() -> None:
    """Callers that pass a default must keep getting it, not an exception."""
    from core.container import ServiceContainer

    assert ServiceContainer.get("still_absent_service", default=None) is None


def test_registration_handler_catches_container_errors() -> None:
    """service_registration's except tuple must include ContainerError."""
    import inspect

    from core import service_registration

    source = inspect.getsource(service_registration)
    index = source.find('container.get("defensive_runtime")')
    assert index != -1
    handler = source[index : index + 900]
    assert "ContainerError" in handler


def test_lifespan_skips_boot_when_shutdown_already_requested() -> None:
    """The root cause: nothing should be booted for a runtime that quit."""
    import inspect

    from interface import server

    source = inspect.getsource(server.lifespan)
    registration = source.find("register_all_services(")
    guard = source.find("is_shutdown_requested()")

    assert guard != -1, "lifespan has no shutdown guard"
    assert guard < registration, "shutdown guard must precede service registration"
