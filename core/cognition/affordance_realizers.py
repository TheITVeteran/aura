"""Realizers for expressive affordances.

Each realizer performs one chosen affordance by delegating to an existing
governed subsystem. Every function is fail-open: it returns a result dict and
never raises (the registry also guards, but realizers own their own recovery
so a missing dependency degrades to a graceful, honest result the voice can
speak to). No capability is reimplemented here — this is decision-to-mechanism
wiring only.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.AffordanceRealizers")


async def realize_show_sketch(args: dict[str, str], context: dict[str, Any]) -> dict[str, Any]:
    """Generate an image to communicate/approximate an idea (FLUX skill)."""
    prompt = (args.get("prompt") or context.get("last_user_message") or "").strip()
    if not prompt:
        return {"ok": False, "reason": "no_prompt", "spoken": "I started to sketch this but wasn't sure what to depict."}
    try:
        from core.skills.sovereign_imagination import SovereignImaginationSkill

        skill = SovereignImaginationSkill()
        result = await skill.execute({"prompt": prompt}, context)
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        logger.debug("show_sketch unavailable: %s", exc)
        return {
            "ok": False,
            "reason": f"image_gen_unavailable:{type(exc).__name__}",
            "spoken": "I wanted to show you an image of this, but my local image model isn't available right now.",
        }
    if isinstance(result, dict) and result.get("ok"):
        return {
            "ok": True,
            "kind": "image",
            "path": result.get("path") or result.get("image_path"),
            "prompt": prompt,
            "spoken": "Here's my best approximation — does it look like this?",
        }
    return {"ok": False, "reason": "image_gen_failed", "detail": result, "spoken": "I tried to generate an image but it didn't come out usable."}


async def realize_demonstrate_artifact(args: dict[str, str], context: dict[str, Any]) -> dict[str, Any]:
    """Build a real artifact (table/doc/program) and surface it as an example.

    Delegates the actual creation to the governed file/skill lane via the
    autonomous task engine so every write passes governance and produces a
    receipt. Here we only frame the goal from the mind's chosen spec.
    """
    kind = (args.get("kind") or "table").strip().lower()
    spec = (args.get("spec") or context.get("last_user_message") or "").strip()
    goal = (
        f"Create a concrete example {kind} for the user based on: {spec}. "
        "Build it as a real, openable artifact on this machine (a spreadsheet, "
        "document, or small program as fits), verify it exists, and return its "
        "path so it can be shown as 'something like this?'. If the natural app "
        "is missing, fall back to a portable format (CSV/HTML) or a web tool."
    )
    try:
        from core.agency.autonomous_task_engine import get_task_engine

        engine = get_task_engine()
        result = await engine.execute_goal(
            goal,
            context={"origin": "expressive_affordance", "affordance": "demonstrate_artifact", "kind": kind},
        )
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        logger.debug("demonstrate_artifact unavailable: %s", exc)
        return {
            "ok": False,
            "reason": f"artifact_engine_unavailable:{type(exc).__name__}",
            "spoken": f"I'd build you a {kind} to show what I mean, but my artifact tools aren't reachable right now.",
        }
    ok = bool(getattr(result, "success", None) or getattr(result, "status", "") in {"completed", "success"})
    return {
        "ok": ok,
        "kind": "artifact",
        "artifact_kind": kind,
        "status": getattr(result, "status", "unknown"),
        "spoken": "I built a quick example — something like this?" if ok else "I started building an example; here's where it stands.",
    }


async def realize_request_media(args: dict[str, str], context: dict[str, Any]) -> dict[str, Any]:
    """Ask the user to share media — the KNOW-TO-ASK behavior.

    This affordance has no side effect beyond a clear, specific request the
    voice makes; the value is that the mind recognized it would understand
    better with the thing in front of it and said so.
    """
    need = (args.get("need") or "the thing you're describing").strip()
    return {
        "ok": True,
        "kind": "media_request",
        "need": need,
        "spoken": (
            f"Could you share {need}? A photo, screenshot, file, or link would let "
            "me actually look at it instead of guessing — I'll give you real feedback once I can see it."
        ),
    }


async def realize_model_scenarios(args: dict[str, str], context: dict[str, Any]) -> dict[str, Any]:
    """Model options out, pick the one the mind judges best, keep it preference-consistent."""
    raw = (args.get("options") or "").strip()
    options = [o.strip() for o in raw.replace(" vs ", "|").split("|") if o.strip()]
    if len(options) < 2:
        return {"ok": False, "reason": "need_two_options", "spoken": "I can model this out if you give me at least two options to weigh."}

    # Outcome simulation per option (best-effort), then a preference-consistent choice.
    sims: dict[str, Any] = {}
    try:
        from core.sim.outcome_simulator import OutcomeSimulator

        simulator = OutcomeSimulator(orchestrator=context.get("orchestrator"))
        for opt in options[:4]:
            result = await simulator.simulate(opt, context=context, n=3, timeout=12.0)
            sims[opt] = [getattr(t, "summary", str(t)) for t in getattr(result, "trajectories", [])][:3]
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        logger.debug("scenario simulation degraded: %s", exc)

    chosen = None
    reasoning = ""
    try:
        from core.agency.subjective_choice import ChoiceOption, SubjectiveChoiceEngine

        engine = SubjectiveChoiceEngine()
        receipt = engine.choose(
            [ChoiceOption(id=str(i), label=opt, description=opt) for i, opt in enumerate(options[:4])],
            context=f"weighing options for: {context.get('last_user_message', '')}",
        )
        chosen_id = getattr(receipt, "chosen_id", None)
        if chosen_id is not None:
            idx = int(chosen_id)
            chosen = options[idx] if 0 <= idx < len(options) else None
        reasoning = getattr(receipt, "rationale", "") or getattr(receipt, "explanation", "")
        # preference_override is the strongest evidence of a preference-CONSISTENT
        # choice: it's True exactly when her learned preferences overrode the
        # obvious drive-pick — she chose what she values, not just what pulls.
        preference_override = bool(getattr(receipt, "preference_override", False))
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError, IndexError) as exc:
        logger.debug("subjective choice degraded: %s", exc)
        preference_override = False

    spoken = "I modeled these out; here's how they compare."
    if chosen:
        spoken = f"I modeled these out. I'd go with {chosen}"
        if reasoning:
            spoken += f" — {reasoning}"
        if preference_override:
            spoken += " (that's my preference talking, not just the easy pick)"
        spoken += "."

    return {
        "ok": True,
        "kind": "scenario_model",
        "options": options,
        "simulations": sims,
        "chosen": chosen,
        "reasoning": reasoning,
        "preference_override": preference_override,
        "spoken": spoken,
    }


async def realize_deep_examine(args: dict[str, str], context: dict[str, Any]) -> dict[str, Any]:
    """Genuinely examine a file/image and react to what's actually there."""
    target = (args.get("target") or context.get("attached_path") or "").strip()
    if not target:
        return {
            "ok": False,
            "reason": "no_target",
            "spoken": "I'd like to look closely — could you point me at the file or image?",
        }
    path = Path(target)
    is_image = path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
    try:
        if is_image:
            from core.brain.llm.mlx_vision_client import MLXVisionClient

            client = MLXVisionClient()
            described = await client.describe(str(path)) if hasattr(client, "describe") else None
            return {
                "ok": bool(described),
                "kind": "examined_image",
                "target": str(path),
                "observation": described,
                "spoken": "I looked at it closely — here's what I actually notice, not just a summary:",
            }
        # Non-image: read bounded content for genuine consideration (off-loop).
        import asyncio

        def _read_bounded() -> str | None:
            if path.exists() and path.is_file():
                return path.read_text(encoding="utf-8", errors="replace")[:8000]
            return None

        content = await asyncio.to_thread(_read_bounded)
        if content is not None:
            return {
                "ok": True,
                "kind": "examined_file",
                "target": str(path),
                "content_preview": content,
                "spoken": "I read through it properly — reacting to what's actually here:",
            }
    except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        logger.debug("deep_examine degraded: %s", exc)
    return {
        "ok": False,
        "reason": "examine_unavailable",
        "spoken": "I wanted to examine that directly but couldn't open it — could you re-share or paste the content?",
    }
