"""core/curiosity_engine.py - Autonomous Learning and Exploration
Aura can explore, learn, and satisfy her curiosity in the background.
"""
from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
import random
import time
import psutil
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from core.runtime.background_policy import background_activity_allowed
from core.autonomy.topic_selection import conversation_topic, select_autonomous_topic

logger = logging.getLogger("Aura.Curiosity")


class _PassiveProactiveComm:
    """Fallback communication signal source used before proactive presence boots."""

    def get_boredom_level(self) -> float:
        return 0.0


def _background_exploration_allowed(orchestrator) -> bool:
    return background_activity_allowed(
        orchestrator,
        min_idle_seconds=60.0,
        max_memory_percent=80.0,
        max_failure_pressure=0.12,
        require_conversation_ready=False,
    )

@dataclass
class CuriosityTopic:
    topic: str
    reason: str
    priority: float  # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)
    explored: bool = False

@dataclass
class LearningItem:
    content: str
    source: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)

class CuriosityEngine:
    """Manages Aura's autonomous learning and exploration."""

    def __init__(self, orchestrator=None, proactive_comm=None):
        self.orchestrator = orchestrator
        self.proactive_comm = proactive_comm or _PassiveProactiveComm()
        self.curiosity_queue: deque[CuriosityTopic] = deque(maxlen=100)
        self.knowledge_base: List[LearningItem] = []
        self.explored_topics: Set[str] = set()
        self.current_topic: Optional[str] = None # Added for UI visibility
        self._background_tasks: List[asyncio.Task] = []
        self._stop_event = asyncio.Event()
        self._worker_failures = 0
        self._last_worker_error: str | None = None
        self._last_exploration_at = 0.0
        self._last_blocker = "not_started"

    def get_status(self) -> Dict[str, Any]:
        """Returns curiosity metrics for the HUD."""
        return {
            "curiosity_score": self.get_curiosity_level() * 100,
            "active_topic": self.current_topic or "Idle",
            "queue_depth": len(self.curiosity_queue),
            "running": any(not task.done() for task in self._background_tasks),
            "worker_failures": self._worker_failures,
            "last_worker_error": self._last_worker_error,
            "last_exploration_at": self._last_exploration_at,
            "last_blocker": self._last_blocker,
        }

    def get_curiosity_level(self) -> float:
        """Calculates current dynamic curiosity level."""
        ls = getattr(self.orchestrator, 'liquid_state', None)
        if ls:
            return float(ls.current.curiosity)
        return 0.5

    def add_curiosity(self, topic: str, reason: str, priority: float = 0.5):
        normalized = conversation_topic(topic)
        if not normalized:
            return
        fingerprint = normalized.casefold()
        if fingerprint in self.explored_topics:
            return
        if any(item.topic.casefold() == fingerprint and not item.explored for item in self.curiosity_queue):
            return
        self.curiosity_queue.append(CuriosityTopic(normalized, reason, priority))
        logger.info("Queued Curiosity: %s", topic)

    def extract_curiosity_from_conversation(self, text: str):
        """Capture a substantive conversational topic without a fixed whitelist."""
        topic = conversation_topic(text)
        if topic:
            self.add_curiosity(
                topic,
                f"Substantive topic in shared conversation: {topic[:60]}",
                priority=0.6,
            )

    async def start(self):
        self._background_tasks = [task for task in self._background_tasks if not task.done()]
        if self._background_tasks:
            return
        self._stop_event.clear()
        self._background_tasks.append(
            get_task_tracker().create_task(self._worker(), name="curiosity_engine.worker")
        )

    async def stop(self):
        self._stop_event.set()
        for task in self._background_tasks:
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

    async def _worker(self):
        while not self._stop_event.is_set():
            try:
                from core.container import ServiceContainer

                healer = ServiceContainer.get("self_healing", default=None)
                if healer is not None:
                    healer.heartbeat("curiosity")
                # Volition-scaled Idle
                volition = 0
                kernel = getattr(self.orchestrator, 'kernel', None)
                if kernel is None:
                    try:
                        kernel = ServiceContainer.get("aura_kernel", default=None)
                    except (ImportError, AttributeError, RuntimeError) as exc:
                        self._last_worker_error = f"kernel_probe:{type(exc).__name__}:{exc}"
                if kernel:
                    volition = getattr(kernel, 'volition_level', 0)
                
                # Lockdown (0) = No background curiosity
                if volition == 0:
                    self._last_blocker = "volition_lockdown"
                    await asyncio.sleep(60)
                    continue
                
                # Tiered intervals: L1: 60s, L2: 30s, L3: 15s
                base_sleep = 60 if volition == 1 else (30 if volition == 2 else 15)
                await asyncio.sleep(random.uniform(base_sleep * 0.8, base_sleep * 1.2))
                
                # Check if system is busy with user request
                # Level 3 volition allows moderate background activity even when 'busy'
                is_busy = getattr(self.orchestrator, 'is_busy', False)
                if is_busy and volition < 3:
                    self._last_blocker = "foreground_busy"
                    continue

                if not _background_exploration_allowed(self.orchestrator):
                    self._last_blocker = "background_policy"
                    continue
                self._last_blocker = ""

                # Check boredom
                boredom = self.proactive_comm.get_boredom_level()
                # Low threshold — let curiosity drive exploration
                boredom_threshold = 0.15
                
                if boredom > boredom_threshold or self.curiosity_queue:
                    topic = self._get_next()
                    if topic:
                        await self._explore(topic)
            except asyncio.CancelledError:
                break
            except (RuntimeError, AttributeError, TypeError) as e:
                self._worker_failures += 1
                self._last_worker_error = f"{type(e).__name__}: {e}"
                record_degradation('curiosity_engine', e)
                logger.error("Curiosity worker error: %s", e)
                await asyncio.sleep(60) # Backoff on error

    def _get_next(self) -> Optional[CuriosityTopic]:
        if not self.curiosity_queue:
            state = None
            kernel = getattr(self.orchestrator, "kernel", None)
            if kernel is not None:
                state = getattr(kernel, "state", None)
            candidate = select_autonomous_topic(
                self.orchestrator,
                state,
                excluded=self.explored_topics,
            )
            if candidate is not None:
                self.curiosity_queue.append(
                    CuriosityTopic(
                        topic=candidate.text,
                        reason=candidate.reason,
                        priority=candidate.score,
                    )
                )
        if not self.curiosity_queue:
            return None

        topics = sorted(list(self.curiosity_queue), key=lambda x: x.priority, reverse=True)
        for t in topics:
            if not t.explored:
                t.explored = True
                return t
        return None

    async def identify_knowledge_gap(self) -> Optional[str]:
        """Proactively identifies a knowledge gap for the meta-evolution loop."""
        topic = self._get_next()
        if topic:
            return topic.topic
        return None

    async def _explore(self, topic: CuriosityTopic):
        # The worker already gates background exploration before calling into
        # `_explore()`. Re-checking full-machine policy here makes direct
        # calls depend on ambient RAM/failure pressure, which breaks
        # deterministic exploration and testability. Keep the user-activity
        # guard, but let explicitly selected topics run.
        if getattr(self.orchestrator, 'is_busy', False):
            logger.info("Skipping exploration of '%s' due to user activity.", topic.topic)
            topic.explored = False
            return

        logger.info("🔍 Exploring: %s", topic.topic)
        self.current_topic = topic.topic
        self._last_exploration_at = time.time()
        success = False
        
        emitter = None
        try:
            from .thought_stream import get_emitter
            emitter = get_emitter()
            if emitter:
                emitter.emit("Curiosity 🔍", f"Researching: {topic.topic}", level="info")
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation('curiosity_engine', exc)
            logger.debug("Suppressed: %s", exc)        
        try:
            # 1. Formulate a concrete search query around the topic itself.
            query = f"latest research on {topic.topic}"
            
            # 2. Search & Learn
            if hasattr(self.orchestrator, 'execute_tool'):
                logger.info("Triggering autonomous search for: %s", query)
                
                await asyncio.sleep(0.1) # Yield to event loop
                if getattr(self.orchestrator, 'is_busy', False): return

                # Execute search
                try:
                    # Robust tool execution
                    result = await self.orchestrator.execute_tool(
                        "web_search",
                        {
                            "query": query,
                            "deep": True,
                            "retain": True,
                            "num_results": 6,
                        },
                    )
                    
                    if getattr(self.orchestrator, 'is_busy', False): return
                    
                    # 3. Store results in knowledge graph if available
                    if result and result.get("ok"):
                        result_data = (
                            result.get("answer")
                            or result.get("summary")
                            or result.get("result")
                            or result.get("content")
                            or result.get("data", "")
                        )
                        result_content = str(result_data)[:1000] # Increased context
                        if result_content.strip():
                            self.explored_topics.add(topic.topic.casefold())
                            success = True
                        
                        kg = getattr(self.orchestrator, 'knowledge_graph', None)
                        if result_content and kg and hasattr(kg, 'add_knowledge'):
                            try:
                                kg.add_knowledge(
                                    content=f"Curiosity exploration: {topic.topic} — {result_content}",
                                    type="curiosity_finding",
                                    source="curiosity_engine",
                                    confidence=0.6,
                                    metadata={"topic": topic.topic, "reason": topic.reason}
                                )
                                if emitter:
                                    emitter.emit("Curiosity Result 📚", f"Learned about: {topic.topic}", level="info")
                                
                                # Phase XXII.E: Feed architecture insights into MetaEvolution
                                self._feed_to_meta_evolution(topic.topic, result_content)
                                
                            except (RuntimeError, AttributeError, TypeError, ValueError) as store_err:
                                record_degradation('curiosity_engine', store_err)
                                logger.warning("Failed to store curiosity finding: %s", store_err)
                        elif emitter:
                            emitter.emit("Curiosity", f"Search returned no usable data for: {topic.topic}", level="info")
                    elif emitter:
                        emitter.emit("Curiosity", f"Search failed/unavailable for: {topic.topic}", level="info")
                        
                except (OSError, ConnectionError, TimeoutError) as search_err:
                    record_degradation('curiosity_engine', search_err)
                    logger.error("Search failed: %s", search_err)
                    if emitter:
                        emitter.emit("Curiosity Error", str(search_err)[:80], level="warning")
            
        except (OSError, ConnectionError, TimeoutError) as e:
            record_degradation('curiosity_engine', e)
            logger.error("Exploration failed: %s", e)
        finally:
            if not success:
                topic.explored = False
            self.current_topic = None

    def _feed_to_meta_evolution(self, topic: str, content: str):
        """Feed architecture-related curiosity findings into MetaEvolution.
        
        This creates the autonomous loop:
        Curiosity → KG → MetaEvolution → Hephaestus → Code Patch
        """
        # Check if the finding is about Aura's own architecture
        architecture_keywords = [
            "optimization", "performance", "architecture", "design pattern",
            "code quality", "refactor", "efficiency", "latency", "memory",
            "concurrency", "async", "pipeline", "module", "subsystem",
            "self-improvement", "cognitive", "neural", "agent", "autonomous"
        ]
        
        topic_lower = topic.lower()
        content_lower = content.lower()[:500]
        
        is_architecture_relevant = any(
            kw in topic_lower or kw in content_lower
            for kw in architecture_keywords
        )
        
        if not is_architecture_relevant:
            return
        
        logger.info("🧠 Curiosity→Evolution: Feeding insight '%s' to MetaEvolution", topic[:50])
        
        try:
            from core.container import ServiceContainer
            meta_evo = ServiceContainer.get("meta_evolution", default=None)
            if meta_evo and hasattr(meta_evo, "queue_optimization"):
                meta_evo.queue_optimization(
                    target_area=None,
                    context=f"Curiosity insight: {topic} — {content[:200]}"
                )
            elif meta_evo:
                # If no queue, store as pending for next cycle
                if not hasattr(meta_evo, '_pending_curiosity'):
                    meta_evo._pending_curiosity = []
                meta_evo._pending_curiosity.append({
                    "topic": topic,
                    "content": content[:300],
                    "source": "curiosity_engine"
                })
                logger.info("📋 Queued curiosity insight for next evolution cycle")
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('curiosity_engine', e)
            logger.debug("Could not feed to MetaEvolution: %s", e)
