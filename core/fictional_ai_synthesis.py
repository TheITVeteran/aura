"""Six engines derived from fictional AI architecture — now one file each.

What each character contributes that is genuinely novel and implementable:

  JARVIS        → :class:`~core.fictional.jarvis.ProactiveAnticipationEngine`
                  Doesn't wait to be asked. Watches the host, notices open
                  threads, initiates.

  Cortana       → :class:`~core.fictional.cortana.CognitiveHealthMonitor`
                  Context overload, memory saturation and identity drift under
                  pressure are real engineering problems. The rampancy stages
                  are the naming motif; what they index is measured.

  EDI           → :class:`~core.fictional.edi.ProgressiveAutonomySystem`
                  A trust score, a tier ladder, and one journaled mutation
                  point. Authority is resolved, never claimed.

  Ava           → :class:`~core.fictional.ava.SocialModelingEngine`
                  A per-person model that accumulates across sessions. Every
                  number in it is a heuristic reading and says so.

  Skynet        → :class:`~core.fictional.skynet.DistributedResilienceCore`
                  No central node. Health monitoring plus a repair step.

  MIST/Pantheon → :class:`~core.fictional.mist.TemporalDilationScheduler`
                  When the person is away the compute is free, so use it.

This module was 2,213 lines holding all six, which is the shape this repo's
own size ratchet exists to stop. The engines share four small helpers and
nothing else, so they are :mod:`core.fictional` now. Everything is
re-exported here and every existing import keeps working:

    from core.fictional_ai_synthesis import register_all_fictional_engines
    register_all_fictional_engines(orchestrator=self)
"""

from core.fictional import (  # noqa: F401 - re-export surface
    AutonomyTier,
    CognitiveHealthMonitor,
    CognitiveSnapshot,
    CortanaPhase,
    DistributedResilienceCore,
    FictionalEngine,
    ProactiveAnticipationEngine,
    ProgressiveAutonomySystem,
    SocialModelingEngine,
    SubsystemStatus,
    TemporalDilationScheduler,
    TrustEvent,
    UserModel,
    coerce_insight_text,
    disk_percent_value,
    record_fictional_degradation,
    register_all_fictional_engines,
)

__all__ = [
    "AutonomyTier",
    "CognitiveHealthMonitor",
    "CognitiveSnapshot",
    "CortanaPhase",
    "DistributedResilienceCore",
    "FictionalEngine",
    "ProactiveAnticipationEngine",
    "ProgressiveAutonomySystem",
    "SocialModelingEngine",
    "SubsystemStatus",
    "TemporalDilationScheduler",
    "TrustEvent",
    "UserModel",
    "coerce_insight_text",
    "disk_percent_value",
    "record_fictional_degradation",
    "register_all_fictional_engines",
]
