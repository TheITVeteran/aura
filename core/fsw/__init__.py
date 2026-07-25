"""core/fsw — flight-software discipline.

Clean-room adoption of practices from F Prime (NASA JPL's flight software
framework), the Apollo Guidance Computer, and OpenMCT (NASA Ames' mission
control framework).

These systems solved a problem Aura has: keeping a complex autonomous
machine legible, recoverable, and honest while it runs unattended,
without a human able to attach a debugger. The specific inheritances:

* :mod:`core.fsw.telemetry_dictionary` — every number has an ID, a unit,
  and declared limits, and every notable occurrence is an event with a
  severity. Not "some logs and some metrics" — a dictionary.
* :mod:`core.fsw.rate_groups` — deterministic periodic execution with
  cycle-slip detection, so "we are running behind" is measured rather
  than inferred.
* :mod:`core.fsw.assertions` — FW_ASSERT: an assertion that records
  where and with what, and triggers a controlled response instead of
  either crashing or being silently compiled out.
* :mod:`core.fsw.command_dispatch` — a command dictionary and a
  sequencer, so an autonomous plan is a checkable artifact.
* :mod:`core.fsw.restart_protection` — the Apollo AGC's restart groups
  and its 1202 overload response, which is the single most relevant piece
  of engineering history here: a computer that shed low-priority work,
  announced the alarm, and kept the critical loop running while landing.
* :mod:`core.fsw.health_checker` — F Prime's Svc::Health, which pings
  every component on a schedule and declares the ones that stop
  answering.
"""

from core.fsw.telemetry_dictionary import (
    ChannelSpec,
    EventSeverity,
    LimitState,
    channel,
    emit_event,
    get_telemetry,
)

__all__ = [
    "ChannelSpec",
    "EventSeverity",
    "LimitState",
    "channel",
    "emit_event",
    "get_telemetry",
]
