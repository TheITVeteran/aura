"""Declared channels and events for the ontogenetic organ.

A channel id is a contract: anything reading Aura's telemetry can rely on
0x0501 meaning the organ's observation rate forever. Limits are declared here
so a crossing is a state transition the system announces, not a threshold
somebody remembered to check at a read site.

The limits chosen matter more than the channels. Two of them encode the
failure modes this organ is actually exposed to:

* **observation rate red-low.** A control point at AUTHORITY whose outcomes
  have stopped being observed is not learning — it is accumulating confident
  ignorance while continuing to decide. That is the single most dangerous
  state the organ can be in, and it is silent unless something watches for it.
* **overconfidence yellow/red-high.** A head whose stated confidence drifts
  above its accuracy is lying about itself, and everything downstream sizes
  its caution by that number.
"""

from __future__ import annotations

import logging
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Ontogeny.Telemetry")

#: Channel ids. Reserved block 0x0501–0x050F; events 0x1101–0x110F.
CHANNEL_OBSERVATION_RATE = "ontogeny.observation_rate"
CHANNEL_EPISODES = "ontogeny.episodes"
CHANNEL_NOVELTY = "ontogeny.novelty"
CHANNEL_AUTHORITY_STAGE = "ontogeny.authority_rank"
CHANNEL_OVERCONFIDENCE = "ontogeny.overconfidence"
CHANNEL_WORLD_SURPRISE = "ontogeny.world_surprise"
CHANNEL_EVIDENCE_ROWS = "ontogeny.evidence_rows"

EVENT_STAGE_CHANGE = "ontogeny_stage_change"
EVENT_REVOKED = "ontogeny_authority_revoked"
EVENT_ERA = "ontogeny_state_era"

#: Channels that only exist once there is something real to put in them.
#:
#: Observation rate, calibration and world-model surprise are undefined until
#: outcomes have been swept, predictions scored and the model stepped — days,
#: on a fresh instance. A declared channel that has never been written reads as
#: stale, so declaring these at boot would make a perfectly healthy new
#: instance permanently in violation, and an alarm that always fires is an
#: alarm nobody reads. They are declared on first real value instead.
#:
#: The dangerous case this might seem to hide — a head that *holds authority*
#: while its outcomes stop being observed — is caught by
#: ``ontogeny.authority_implies_observation`` in invariants.py, which is the
#: right instrument for it. A telemetry channel is for trending; an invariant
#: is for alarming.
_DEFERRED_SPECS: dict[str, dict[str, Any]] = {}

_declared = False
_deferred_done: set[str] = set()


def declare() -> list[str]:
    """Declare the organ's channels and events. Idempotent."""
    global _declared
    if _declared:
        return []
    try:
        from core.fsw.telemetry_dictionary import ChannelType, EventSeverity, channel, event
    except ImportError as exc:
        record_degradation("ontogeny_telemetry", exc, severity="debug",
                           action="telemetry dictionary unavailable")
        return []

    names: list[str] = []
    for spec in (
        dict(
            identifier=0x0502, name=CHANNEL_EPISODES, type=ChannelType.INT, unit="count",
            description="episodes the organ has lived through this process",
            owner="core/ontogeny/service.py", group="ontogeny", stale_after_s=600.0,
        ),
        dict(
            identifier=0x0503, name=CHANNEL_NOVELTY, unit="fraction",
            description="distance of the current state from the centre of her lived distribution",
            owner="core/ontogeny/state.py", group="ontogeny",
            yellow_high=0.85, stale_after_s=600.0,
        ),
        dict(
            identifier=0x0504, name=CHANNEL_AUTHORITY_STAGE, type=ChannelType.INT, unit="rank",
            description="highest authority rank held by any learned head (0 observe … 3 authority)",
            owner="core/ontogeny/authority.py", group="ontogeny",
            enum_labels=("observe", "shadow", "advisory", "authority"), stale_after_s=600.0,
        ),
        dict(
            identifier=0x0507, name=CHANNEL_EVIDENCE_ROWS, type=ChannelType.INT, unit="count",
            description="episodes in the corpus carrying a real outcome label",
            owner="core/ontogeny/experience.py", group="ontogeny", stale_after_s=600.0,
        ),
    ):
        try:
            channel(**spec)
            names.append(str(spec["name"]))
        except (ValueError, TypeError, KeyError) as exc:
            record_degradation("ontogeny_telemetry", exc, severity="debug",
                               action=f"channel {spec.get('name')} not declared")

    for spec in (
        dict(
            identifier=0x1101, name=EVENT_STAGE_CHANGE, severity=EventSeverity.ACTIVITY_HI,
            format_string="{control_point} moved {previous} -> {stage} ({reason})",
            description="a learned head changed how much authority it holds",
            owner="core/ontogeny/authority.py",
        ),
        dict(
            identifier=0x1102, name=EVENT_REVOKED, severity=EventSeverity.WARNING_HI,
            format_string="{control_point} lost authority: {reason}",
            description="a head that was deciding is no longer trusted to",
            owner="core/ontogeny/authority.py",
        ),
        dict(
            identifier=0x1103, name=EVENT_ERA, severity=EventSeverity.ACTIVITY_HI,
            format_string="ontogenetic state entered era {era} ({reason})",
            description="the persistent state was reset because its input space changed",
            owner="core/ontogeny/state.py",
        ),
    ):
        try:
            event(**spec)
            names.append(str(spec["name"]))
        except (ValueError, TypeError, KeyError) as exc:
            record_degradation("ontogeny_telemetry", exc, severity="debug",
                               action=f"event {spec.get('name')} not declared")

    _DEFERRED_SPECS.update({
        CHANNEL_OBSERVATION_RATE: dict(
            identifier=0x0501, name=CHANNEL_OBSERVATION_RATE, unit="fraction",
            description="share of closed episodes whose outcome was actually observed",
            owner="core/ontogeny/resolution.py", group="ontogeny",
            yellow_low=0.25, red_low=0.05, stale_after_s=86400.0,
        ),
        CHANNEL_OVERCONFIDENCE: dict(
            identifier=0x0505, name=CHANNEL_OVERCONFIDENCE, unit="delta",
            description="mean stated confidence minus measured accuracy; positive is overconfident",
            owner="core/ontogeny/calibration.py", group="ontogeny",
            yellow_high=0.08, red_high=0.15, stale_after_s=86400.0,
        ),
        CHANNEL_WORLD_SURPRISE: dict(
            identifier=0x0506, name=CHANNEL_WORLD_SURPRISE, unit="nats",
            description="world-model prediction error, rolling mean",
            owner="core/world_model/learned_world_model.py", group="ontogeny",
            yellow_high=4.0, stale_after_s=3600.0,
        ),
    })
    _declared = True
    return names


def _declare_on_demand(name: str) -> bool:
    """Declare a conditional channel the first time it has a real value."""
    if name in _deferred_done:
        return True
    spec = _DEFERRED_SPECS.get(name)
    if spec is None:
        return False
    try:
        from core.fsw.telemetry_dictionary import channel

        channel(**spec)
        _deferred_done.add(name)
        return True
    except (ImportError, ValueError, TypeError, KeyError) as exc:
        record_degradation("ontogeny_telemetry", exc, severity="debug",
                           action=f"deferred channel {name} not declared")
        return False


def sample(report: dict[str, Any]) -> None:
    """Write the organ's current values to their channels."""
    if not _declared:
        return
    try:
        from core.fsw.telemetry_dictionary import write
    except ImportError:
        return

    def _put(name: str, value: Any) -> None:
        if value is None:
            return
        if name in _DEFERRED_SPECS and not _declare_on_demand(name):
            return
        try:
            write(name, value)
        except (ValueError, TypeError, KeyError) as exc:
            record_degradation("ontogeny_telemetry", exc, severity="debug",
                               action=f"channel {name} not written")

    _put(CHANNEL_EPISODES, int(report.get("episodes_seen") or 0))
    _put(CHANNEL_NOVELTY, float(report.get("novelty") or 0.0))
    resolution = report.get("resolution") or {}
    # Only report an observation rate once something has actually been swept.
    # A fresh process legitimately has none, and publishing 0.0 would trip the
    # red-low limit on every boot — an alarm that always fires is an alarm
    # nobody reads.
    if resolution.get("swept"):
        _put(CHANNEL_OBSERVATION_RATE, float(resolution.get("observation_rate") or 0.0))

    ranks = {"observe": 0, "shadow": 1, "advisory": 2, "authority": 3}
    stages = [
        ranks.get(str(detail.get("stage")), 0)
        for detail in (report.get("control_points") or {}).values()
    ]
    _put(CHANNEL_AUTHORITY_STAGE, max(stages) if stages else 0)

    calibration = report.get("calibration") or {}
    if calibration:
        _put(
            CHANNEL_OVERCONFIDENCE,
            max(float(rep.get("overconfidence") or 0.0) for rep in calibration.values()),
        )

    world = report.get("world_model") or {}
    _put(CHANNEL_WORLD_SURPRISE, world.get("mean_surprise"))

    evidence = sum(
        int((detail.get("corpus") or {}).get("evidence_rows") or 0)
        for detail in (report.get("control_points") or {}).values()
    )
    _put(CHANNEL_EVIDENCE_ROWS, evidence)


__all__ = [
    "CHANNEL_AUTHORITY_STAGE",
    "CHANNEL_EPISODES",
    "CHANNEL_EVIDENCE_ROWS",
    "CHANNEL_NOVELTY",
    "CHANNEL_OBSERVATION_RATE",
    "CHANNEL_OVERCONFIDENCE",
    "CHANNEL_WORLD_SURPRISE",
    "EVENT_ERA",
    "EVENT_REVOKED",
    "EVENT_STAGE_CHANGE",
    "declare",
    "sample",
]
