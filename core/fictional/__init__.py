"""The six fictional-AI engines, one module each.

They were a single 2,213-line file with six unrelated classes in it, which
is the shape this repo's own size ratchet exists to stop. They share four
small helpers in :mod:`core.fictional.common` and nothing else.

:mod:`core.fictional_ai_synthesis` re-exports everything here, so every
existing import keeps working.
"""

from core.fictional.ava import SocialModelingEngine, UserModel
from core.fictional.common import (
    coerce_insight_text,
    disk_percent_value,
    record_fictional_degradation,
)
from core.fictional.cortana import (
    CognitiveHealthMonitor,
    CognitiveSnapshot,
    CortanaPhase,
)
from core.fictional.edi import AutonomyTier, ProgressiveAutonomySystem, TrustEvent
from core.fictional.jarvis import FictionalEngine, ProactiveAnticipationEngine
from core.fictional.mist import TemporalDilationScheduler
from core.fictional.registry import register_all_fictional_engines
from core.fictional.skynet import DistributedResilienceCore, SubsystemStatus

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
