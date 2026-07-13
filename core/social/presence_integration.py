"""core/social/presence_integration.py
───────────────────────────
The "Presence Patch" that wires the v30 components into the Orchestrator.
"""

import logging

from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service, register_runtime_service
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.PresenceIntegration")

def apply_presence_patch(orchestrator):
    """
    Wires OpinionEngine, ProactivePresence, and social/discourse systems
    into a running RobustOrchestrator.
    Called during _init_proactive_systems in boot.py.
    """
    logger.info("🔧 [PresencePatch] Applying Phase 30 communication hierarchy...")

    # 1. Initialize & Register OpinionEngine
    from core.epistemics.opinion_engine import OpinionEngine
    opinion_engine = OpinionEngine(orchestrator)
    register_runtime_service("opinion_engine", opinion_engine)
    orchestrator.opinion_engine = opinion_engine
    logger.info("✅ OpinionEngine registered.")

    # 2. Initialize & Register ProactivePresence
    from core.autonomy.proactive_presence import ProactivePresence
    presence = ProactivePresence(orchestrator)
    register_runtime_service("proactive_presence", presence)
    orchestrator.proactive_presence = presence
    logger.info("✅ ProactivePresence registered.")

    # 3. Start ProactivePresence background task
    get_task_tracker().create_task(presence.start())
    logger.info("🚀 ProactivePresence loop started.")

    # 4. Hook VAD to prevent interruption
    try:
        from core.senses.voice_engine import get_voice_engine
        voice_engine = get_voice_engine()
        voice_engine._on_vad_change = presence.mark_user_speaking
        logger.info("🎤 VAD pinned to ProactivePresence.")
    except (ImportError, AttributeError, RuntimeError) as e:
        record_degradation('presence_integration', e)
        logger.warning("Failed to hook VAD: %s", e)

    # 5. Canonical identity-scoped relational memory authority
    try:
        from core.social.relational_memory import get_relational_memory_authority

        relational_memory = get_relational_memory_authority()
        if not get_runtime_service("relational_memory", default=None):
            register_runtime_service(
                "relational_memory",
                relational_memory,
                required=False,
                owner="relational_memory",
                registered_by="presence_integration",
                failure_policy="degrade",
            )
        logger.info(
            "✅ RelationalMemoryAuthority registered (persistence=%s).",
            relational_memory.persistence_available,
        )
    except (ImportError, AttributeError, RuntimeError) as e:
        record_degradation('presence_integration', e)
        logger.warning("RelationalMemoryAuthority init failed: %s", e)
        relational_memory = None

    # 6. Shared Ground compatibility adapter
    try:
        from core.memory.shared_ground import get_shared_ground

        if relational_memory is None:
            raise RuntimeError("relational memory authority is unavailable")
        sg = get_shared_ground(authority=relational_memory)
        register_runtime_service("shared_ground", sg, required=False)
        logger.info("✅ SharedGround adapter registered (%d consented entries).", len(sg.entries))
    except (ImportError, AttributeError, RuntimeError) as e:
        record_degradation('presence_integration', e)
        logger.warning("SharedGround adapter init failed: %s", e)

    # 7. SocialMemory compatibility adapter
    try:
        # Only register if not already present (another boot path may have registered it)
        if not get_runtime_service("social_memory", default=None):
            from core.memory.social_memory import SocialMemory

            if relational_memory is None:
                raise RuntimeError("relational memory authority is unavailable")
            social_mem = SocialMemory(authority=relational_memory)
            register_runtime_service("social_memory", social_mem, required=False)
            logger.info("✅ SocialMemory adapter registered.")
    except (ImportError, AttributeError, RuntimeError) as e:
        record_degradation('presence_integration', e)
        logger.warning("SocialMemory init failed: %s", e)

    # 8. TheoryOfMind (user model: rapport, trust, emotional state)
    try:
        if not get_runtime_service("theory_of_mind", default=None):
            from core.consciousness.theory_of_mind import get_theory_of_mind
            ce = getattr(orchestrator, "cognitive_engine", None)
            tom = get_theory_of_mind(ce)
            register_runtime_service("theory_of_mind", tom)
            logger.info("✅ TheoryOfMind registered.")
    except (ImportError, AttributeError, RuntimeError) as e:
        record_degradation('presence_integration', e)
        logger.warning("TheoryOfMind init failed: %s", e)

    # 9. DiscourseTracker (topic threading, user emotional trend, conversation energy)
    try:
        from core.brain.discourse_tracker import DiscourseTracker
        ce = getattr(orchestrator, "cognitive_engine", None)
        discourse_tracker = DiscourseTracker(ce)
        register_runtime_service("discourse_tracker", discourse_tracker)
        logger.info("✅ DiscourseTracker registered.")
    except (ImportError, AttributeError, RuntimeError) as e:
        record_degradation('presence_integration', e)
        logger.warning("DiscourseTracker init failed: %s", e)

    return True
