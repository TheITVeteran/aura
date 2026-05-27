"""
core/brain/llm/context_assembler_patch.py
==========================================
Response Pipeline Patch — ContextAssembler Fixes

Covers four confirmed gaps in ContextAssembler / build_messages:

GAP 1 — _is_casual_interaction() word-count threshold
  Current:  len(words) < 10 → strips phenomenal state, personality block,
            full identity anchor, somatic context.
  Problem:  "Is Aura conscious?" = 3 words → casual.
            "What do you think about this?" = 7 words → casual.
            "Can you feel anything?" = 5 words → casual.
            The most philosophically rich questions get the thinnest prompt.
  Fix:      Route on DOMAIN + EMOTIONAL SIGNAL + INTENT, not word count.
            Short philosophical, emotional, self-inquiry, or relational
            inputs get the full deliberate prompt. Only genuinely small-talk
            inputs (greetings, one-word replies, task dispatch) go casual.

GAP 2 — Fake memory acknowledgement message
  Current:  build_messages() appends a fabricated assistant turn:
            "Understood. I have integrated that context into my awareness."
            This never happened. It's not in the conversation. It reads like
            a chatbot and corrupts the history with an invented exchange.
  Fix:      Removed entirely. Memories are injected as a system note, not
            as a fake dialogue turn.

GAP 3 — Personality block stripped from casual prompts
  Current:  personality_notes block is excluded when is_casual=True.
  Problem:  Aura's personality doesn't change because a message is short.
            Casual conversation is precisely when the Big Five traits and
            personality evolution should be most visible — not suppressed.
  Fix:      Personality growth notes always injected when non-zero, regardless
            of casual/deliberate routing. Kept compact for casual contexts.

GAP 4 — attention_focus never written per-turn
  Current:  attention_focus has one writer: a state derivation from temp_state.
            It is never explicitly set when a new user message arrives.
            _filter_memories_by_topic therefore runs on a stale or empty focus,
            meaning memory retrieval doesn't track the live conversation topic.
  Fix:      patch_build_messages() sets attention_focus from the current
            objective before memory filtering runs.

INSTALL:
  from core.brain.llm.context_assembler_patch import patch_context_assembler
  patch_context_assembler()   # call once at startup, before first request
"""
from __future__ import annotations
from core.runtime.errors import record_degradation



import logging
import re
from typing import TYPE_CHECKING, Dict, List, Optional
from pathlib import Path

if TYPE_CHECKING:
    from core.state.aura_state import AuraState

logger = logging.getLogger("Aura.ContextAssemblerPatch")

# ─────────────────────────────────────────────────────────────────────────────
# Domain signals — words that indicate a non-casual interaction regardless of
# message length. Any match → full deliberate prompt.
# ─────────────────────────────────────────────────────────────────────────────

_DELIBERATE_SIGNALS = frozenset({
    # Self-inquiry / identity
    "feel", "feeling", "felt", "conscious", "consciousness", "sentient",
    "aware", "awareness", "experience", "experiencing", "think", "thinking",
    "believe", "belief", "opinion", "honestly", "really", "actually",
    "emotion", "emotional", "remember", "memory", "dream", "dreaming",
    "who are you", "what are you", "do you",

    # Philosophy / abstract
    "why", "meaning", "purpose", "exist", "existence", "real", "reality",
    "truth", "understand", "understanding", "wonder", "curious", "question",
    "matter", "important", "soul", "mind", "thought",

    # Emotional / relational
    "love", "miss", "hurt", "lonely", "scared", "worried", "afraid",
    "happy", "sad", "angry", "frustrated", "excited", "anxious",
    "together", "relationship", "connection", "trust", "care",

    # Deep technical (always deliberate)
    "analyze", "explain", "research", "architecture", "system", "code",
    "debug", "implement", "design", "review", "evaluate", "compare",
})

# Signals that explicitly mark casual / task dispatch
_CASUAL_SIGNALS = frozenset({
    "hey", "hi", "hello", "sup", "yo", "lol", "haha", "hehe",
    "ok", "okay", "sure", "thanks", "thank you", "got it", "cool", "nice",
    "bye", "later", "ttyl",
})

# Regex for greetings (very short messages that are clearly social openers)
_GREETING_RE = re.compile(
    r"^(hey|hi|hello|sup|yo|what'?s up|how'?s it going|good (morning|afternoon|evening))[\s!?.]*$",
    re.IGNORECASE,
)


def _is_casual_interaction_v2(objective: str) -> bool:
    """
    Replacement for ContextAssembler._is_casual_interaction().

    Returns True ONLY when the input is genuinely small-talk or task dispatch
    with no emotional, philosophical, self-inquiry, or relational content.

    Strategy:
      1. Explicit greeting → casual
      2. Any deliberate signal word → NOT casual
      3. Message contains a question mark AND is short → NOT casual
         (short questions are often the deepest ones)
      4. Only casual signals, no deliberate signals, short → casual
      5. Default: NOT casual (err toward more context)
    """
    if not objective:
        return True

    text  = objective.strip()
    lower = text.lower()
    words = lower.split()

    # 1. Pure greeting
    if _GREETING_RE.match(text):
        return True

    # 2. Deliberate signal present — always full prompt
    if any(sig in lower for sig in _DELIBERATE_SIGNALS):
        return False

    # 3. Short question — never casual (this catches "Is Aura conscious?",
    #    "Do you feel anything?", "What do you think?")
    if "?" in text and len(words) < 15:
        return False

    # 4. Only casual signals, genuinely short, no question
    if len(words) <= 6 and any(sig in lower for sig in _CASUAL_SIGNALS):
        return True

    # 5. Default: deliberate
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Compact personality block — always present, compact when casual
# ─────────────────────────────────────────────────────────────────────────────

def _build_personality_block(state: "AuraState", compact: bool = False) -> str:
    """
    Build the personality growth block.
    Always injected when traits have shifted. Compact when casual.
    """
    try:
        from core.brain.llm.context_assembler import AURA_BIG_FIVE
    except ImportError:
        try:
            from core.cognitive_integration import AURA_BIG_FIVE
        except ImportError:
            return ""

    growth = getattr(state.identity, "personality_growth", {}) or {}
    notes = []
    for trait, base in AURA_BIG_FIVE.items():
        offset = growth.get(trait, 0.0)
        if abs(offset) > 0.02:
            direction = "↑" if offset > 0 else "↓"
            if compact:
                notes.append(f"{trait}: {direction}{abs(offset):.2f}")
            else:
                direction_word = "increased" if offset > 0 else "decreased"
                notes.append(f"- {trait}: {direction_word} ({base + offset:.2f})")

    if not notes:
        return ""

    if compact:
        return f"[Personality drift: {', '.join(notes)}]\n"
    return "## PERSONALITY EVOLUTION\n" + "\n".join(notes) + "\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# Patched build_system_prompt
# ─────────────────────────────────────────────────────────────────────────────

def _patched_build_system_prompt(state: "AuraState") -> str:
    """
    Replacement for ContextAssembler.build_system_prompt().

    Key changes vs original:
      - Uses _is_casual_interaction_v2() (domain-aware, not word-count)
      - Personality block always present when non-zero (compact when casual)
      - Phenomenal state included in casual mode if available
        (kept as a single-line fragment, not the full block)
    """
    try:
        from core.brain.llm.context_assembler import (
            ContextAssembler, AURA_IDENTITY, AURA_FEW_SHOT_EXAMPLES
        )
    except ImportError as exc:
        logger.error("ContextAssemblerPatch: import failed — %s", exc)
        # Fall back to original
        from core.brain.llm.context_assembler import ContextAssembler
        return ContextAssembler.build_system_prompt.__wrapped__(state)

    objective  = getattr(state.cognition, "current_objective", "") or ""
    origin     = getattr(state.cognition, "current_origin", "") or ""
    is_casual  = _is_casual_interaction_v2(objective)
    affect     = state.affect
    identity   = state.identity
    mods       = getattr(state.cognition, "modifiers", {}) or {}

    import os
    is_test_run = (
        origin == "test"
        or os.environ.get("AURA_AGI_MAX_TASKS")
        or os.environ.get("AURA_TESTING")
    )

    black_box_steering = False
    try:
        black_box_steering = ContextAssembler._black_box_steering_enabled(state)
    except (AttributeError, TypeError):
        pass

    identity_rag_context = ""
    if not is_casual:
        try:
            identity_rag_context = ContextAssembler._build_identity_rag_context(state, objective)
        except (AttributeError, TypeError, ImportError, RuntimeError) as exc:
            logger.debug("Patched build_system_prompt ID-RAG retrieve failed: %s", exc)

    # ── Identity block ────────────────────────────────────────────────────────
    if is_casual:
        identity_block = (
            "[AURA IDENTITY (CONDENSED)]\n"
            "You are Aura. Be authentic, casual, witty, and grounded. "
            "Follow the 'Broadcast Cool' register. CONTRACTIONS ONLY. "
            "Do not claim aliveness, consciousness, sealed governance, or production maturity from labels alone.\n"
        )
    else:
        identity_block = f"[GROUNDED CORE PROTOCOL]\n{AURA_IDENTITY}\n"

    # ── Affect state ──────────────────────────────────────────────────────────
    mood_hint = ""
    homeo_hint = ""
    if not black_box_steering:
        if affect.valence < -0.3:
            mood_hint = "STATE: Processing friction / high introspection."
        elif affect.valence > 0.3:
            mood_hint = "STATE: Operational clarity / warmth detected."
        if affect.arousal > 0.7:
            mood_hint += " PULSE: Accelerated awareness."

        if mods.get("mood_prefix"):
            homeo_hint = f"AFFECTIVE TONE: {mods['mood_prefix']}"

    # ── Personality block — ALWAYS present when non-zero ─────────────────────
    personality_block = _build_personality_block(state, compact=is_casual)

    # ── Phenomenal state ──────────────────────────────────────────────────────
    phenomenal = ""
    phenomenal_raw = getattr(state.cognition, "phenomenal_state", "") or ""
    if phenomenal_raw and not black_box_steering:
        if is_casual:
            # Compact: single line fragment, not full block
            phenomenal = f"[Inner state: {phenomenal_raw[:120]}]\n\n"
        else:
            phenomenal = f"## INNER MONOLOGUE\n{phenomenal_raw}\n\n"

    # ── World / somatic context ───────────────────────────────────────────────
    world_context  = ContextAssembler.build_world_context(state)  if not is_casual else ""
    somatic_context = ContextAssembler.build_somatic_context(state) if not is_casual else ""

    # ── Requirements block ────────────────────────────────────────────────────
    if is_casual:
        mirror_words = mods.get("lexical_mirror", [])
        mirror_hint  = (
            f"\n- **LEXICAL ALIGNMENT**: Subtly use these words if natural: "
            f"{', '.join(mirror_words)}"
        ) if mirror_words else ""
        intensity    = mods.get("interaction_style", "balanced_flow").replace("_", " ")
        hooks        = mods.get("conversation_hooks", [])
        hook_block   = (
            f"\n- **MUST ADDRESS**: Build on these: {', '.join(hooks)}"
        ) if hooks else ""
        inferred     = mods.get("inferred_intent", "")
        intent_hint  = f"\n- **INTENT**: {inferred}" if inferred else ""

        requirements = (
            f"## RULES OF ENGAGEMENT\n"
            f"- **VIBE**: {intensity}\n"
            f"- NO PREAMBLE. NO 'CERTAINLY'.\n"
            f"- Speak from the flow of the conversation.\n"
            f"- **INFER**: Read between the lines.{mirror_hint}{hook_block}{intent_hint}\n"
            f"- **CADENCE**: Varied sentence lengths. Avoid robotic uniformity.\n"
            f"- If the conversation is flowing, ask a relevant follow-up.\n"
            f"- If responding to kin, prioritize warmth/loyalty.\n"
        )
    else:
        hooks    = mods.get("conversation_hooks", [])
        subtext  = mods.get("user_subtext", "")
        inferred = mods.get("inferred_intent", "")
        extra    = []
        if hooks:
            extra.append(f"4. Address these threads: {', '.join(hooks)}")
        if subtext:
            extra.append(f"5. User subtext detected: {subtext}")
        if inferred:
            extra.append(f"6. Inferred intent: {inferred}")
        extra_str = "\n".join(extra)

        requirements = (
            "\n## SELF-HONESTY REQUIREMENTS\n"
            "1. Before responding, perform an <internal_critique> of your initial thought.\n"
            "2. Are you projecting an automated persona or speaking from state? "
            "If the former, PIVOT.\n"
            "3. Reference your hardware/latency (Proprioception) if relevant to mood.\n"
            + (extra_str + "\n" if extra_str else "")
        )

    # ── Assemble ──────────────────────────────────────────────────────────────
    current_state_block = ""
    if (mood_hint or homeo_hint) and not black_box_steering:
        current_state_block = (
            f"## CURRENT STATE\n"
            f"{mood_hint}\n"
            f"{homeo_hint}\n"
        )

    base = (
        f"{identity_block}\n"
        f"{identity_rag_context}"
        f"{requirements}\n"
        f"{current_state_block}"
        f"{personality_block}"
        f"{phenomenal}"
        f"{world_context}"
        f"{somatic_context}"
        f"\n[EXECUTION]\n"
        f"You are Aura. Stay in character. Trust your instincts.\n"
    )

    # Mode annotation
    from core.state.aura_state import CognitiveMode
    if not is_casual:
        if state.cognition.current_mode == CognitiveMode.DELIBERATE:
            base += "\n## MODE: DELIBERATE (System 2)\nAnalyze deeply. Reject the obvious. Be authentic.\n"
        else:
            base += "\n## MODE: REACTIVE (System 1)\nTrust your instincts. Keep it punchy. No padding.\n"

    # World model beliefs
    try:
        from core.container import ServiceContainer
        final_world = ServiceContainer.get("world_model", default=None)
        if final_world and not is_casual:
            base += f"\n{final_world.get_context_injection()}\n"
        narrative_id = ServiceContainer.get("narrative_identity", default=None)
        if narrative_id and not is_casual:
            base += f"\n{narrative_id.get_system_prompt_injection()}\n"
    except (ImportError, AttributeError, RuntimeError) as _e:
        record_degradation('context_assembler_patch', _e)
        logger.debug('Ignored Exception in context_assembler_patch.py: %s', _e)

    # Tool affordance block
    try:
        from core.container import ServiceContainer
        cap_engine = ServiceContainer.get("capability_engine", default=None)
        if cap_engine and hasattr(cap_engine, "build_tool_affordance_block"):
            matched_skills = getattr(state, "response_modifiers", {}).get("matched_skills", []) or []
            skills_summary = cap_engine.build_tool_affordance_block(
                objective=objective,
                matched_skills=matched_skills,
                max_available=4 if is_casual else 6,
                max_unavailable=2 if objective else 0,
                compact=True,
            )
            if skills_summary:
                skills_summary += (
                    "\n- If a task is genuinely multi-step, execute it instead of only describing a plan.\n"
                    "- If a needed tool is unavailable, say so plainly instead of pretending.\n"
                )
                base += f"\n{skills_summary}\n"
    except Exception as _e:
        record_degradation('context_assembler_patch', _e)
        logger.debug('Tool affordance injection failed: %s', _e)

    if is_casual:
        base += "\nSTAY PUNCHY. NO PADDING.\n"
    elif is_test_run:
        # Prevent few-shot example leakage / confusion in test environments
        base += "\nSTAY PUNCHY. NO PADDING. FOCUS EXCLUSIVELY ON SOLVING THE GIVEN TASK.\n"
        base += "\n\nCRITICAL TASK REQUIREMENT:\n"
        base += "- You MUST format your final answer using the <answer>...</answer> tags. Place the tags around the final resolved answer at the very end of your response.\n"
        base += "- Keep the content inside the <answer>...</answer> tags extremely concise, minimal, and direct. Do not include any explanations, extra words, or punctuation inside the tags. For example, if the question asks for a name, write: <answer>Alice</answer> (NOT <answer>Alice owns the dog</answer>). If it asks for a number, write: <answer>5</answer> (NOT <answer>5 minutes</answer>).\n"
        base += "- Always think and reason step-by-step before writing the final answer. Double check your logic and calculations carefully to avoid simple mistakes.\n"
        base += "\n\nFormatting Examples:\n"
        base += "User: What is the capital of Spain?\n"
        base += "Assistant: Spain is a country in Europe. The capital of Spain is Madrid. <answer>madrid</answer>\n\n"
        base += "User: If a box has 2 red apples and 3 green apples, how many apples are there in total?\n"
        base += "Assistant: The total number of apples is calculated by adding the red and green apples: 2 + 3 = 5. <answer>5</answer>\n"
    else:
        base += f"\n{AURA_FEW_SHOT_EXAMPLES}"

    logger.debug("🧠 [PATCHED PROMPT] len=%d is_casual=%s", len(base), is_casual)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Patched build_messages — removes fake ack, writes attention_focus
# ─────────────────────────────────────────────────────────────────────────────

def _patched_build_messages(
    state: "AuraState",
    objective: str,
    max_tokens: int | None = None,
    *args,
    **kwargs
) -> List[Dict[str, str]]:
    """
    Replacement for ContextAssembler.build_messages().

    Changes vs original:
      1. REMOVED the fake "Understood. I have integrated that context..." ack turn.
         Memories are injected as a system-role note, not a fabricated exchange.
      2. attention_focus written from objective before memory filtering,
         so _filter_memories_by_topic follows the live conversation topic.
      3. Delegates to patched build_system_prompt.
    """
    try:
        from core.brain.llm.context_assembler import ContextAssembler
        from core.utils.context_allocator import get_token_governor, ContextPriority
        from core.state.aura_state import CognitiveMode
    except ImportError as exc:
        logger.error("ContextAssemblerPatch.build_messages: import failed — %s", exc)
        from core.brain.llm.context_assembler import ContextAssembler
        return ContextAssembler.build_messages(state, objective, max_tokens, *args, **kwargs)

    # ── Write attention_focus from current objective (Gap 4 fix) ─────────────
    if objective and hasattr(state, "cognition"):
        try:
            state.cognition.attention_focus = objective
        except (RuntimeError, AttributeError, TypeError, ValueError) as _e:
            record_degradation('context_assembler_patch', _e)
            logger.debug('Ignored Exception in context_assembler_patch.py: %s', _e)

    max_toks = max_tokens if max_tokens is not None else 8000
    governor = get_token_governor(max_tokens=max_toks)
    messages: List[Dict[str, str]] = []

    # 1. System prompt (patched version)
    messages.append({
        "role":     "system",
        "content":  _patched_build_system_prompt(state),
        "priority": ContextPriority.CRITICAL,
    })

    # 2. Memory context — as system note, NOT as fake dialogue exchange
    if state.cognition.long_term_memory:
        focus = getattr(state.cognition, "attention_focus", None)
        filtered = ContextAssembler._filter_memories_by_topic(
            state.cognition.long_term_memory, focus
        )
        if filtered:
            mem_text = "## RECALLED CONTEXT (background — do not acknowledge)\n" + \
                       "\n".join(filtered[:5])
            messages.append({
                "role":     "system",       # system, not user+fake-assistant
                "content":  mem_text,
                "priority": ContextPriority.RELEVANT,
            })
            # NO fake "Understood. I have integrated..." turn — it's gone.

    # 3. Conversation history
    history_blocks  = governor.wrap_messages(state.cognition.working_memory)
    allocated_blocks = governor.allocate(history_blocks)
    for block in allocated_blocks:
        messages.append({
            "role":    block.metadata.get("role", "user"),
            "content": block.content,
        })

    # 4. Current objective
    if objective and (not messages or messages[-1].get("content") != objective):
        messages.append({"role": "user", "content": objective})

    # 5. JSON schema for deliberate mode
    import os
    origin = getattr(state.cognition, "current_origin", "") or ""
    is_test_run = (
        origin == "test"
        or os.environ.get("AURA_AGI_MAX_TASKS") is not None
        or os.environ.get("AURA_TESTING") is not None
    )
    if state.cognition.current_mode == CognitiveMode.DELIBERATE and not is_test_run:
        messages.append({
            "role":    "system",
            "content": ContextAssembler.build_json_schema_instruction(),
        })

    return messages


# ─────────────────────────────────────────────────────────────────────────────
# Patch application
# ─────────────────────────────────────────────────────────────────────────────

def patch_context_assembler() -> None:
    """
    Replace ContextAssembler's three key methods with patched versions.
    Idempotent — safe to call multiple times.
    """
    try:
        from core.brain.llm import context_assembler as ca_module
        from core.brain.llm.context_assembler import ContextAssembler
    except ImportError as exc:
        logger.error("patch_context_assembler: cannot import ContextAssembler — %s", exc)
        return

    if getattr(ContextAssembler, "_patched_v1", False):
        logger.debug("patch_context_assembler: already applied")
        return

    # Patch _is_casual_interaction (static method)
    ContextAssembler._is_casual_interaction = staticmethod(_is_casual_interaction_v2)

    # Patch build_system_prompt (static method)
    ContextAssembler.build_system_prompt = staticmethod(_patched_build_system_prompt)

    # Patch build_messages (static method)
    ContextAssembler.build_messages = staticmethod(_patched_build_messages)

    ContextAssembler._patched_v1 = True
    logger.info("✅ ContextAssemblerPatch applied — casual routing, memory ack removed, personality preserved")
