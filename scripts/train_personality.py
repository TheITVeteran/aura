#!/usr/bin/env python3
"""scripts/train_personality.py

Generate persona-conditioned examples by prompting the cognitive engine.
Outputs newline-separated JSONL files: data/personality_training/{persona}.jsonl

Usage:
  python3 scripts/train_personality.py --persona mist --n 50

Note: This script uses `core.brain.cognitive_engine.cognitive_engine`. It fails
closed when the engine is unavailable unless `--allow-synthetic-fallback` is set.
"""

import argparse
import json
import os
import time
import logging

logger = logging.getLogger("train_personality")
logging.basicConfig(level=logging.INFO)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "personality_training")
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_PERSONALITY_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
TRAIN_PERSONALITY_GENERATION_ERRORS = (OSError, *TRAIN_PERSONALITY_RECOVERABLE_ERRORS)


def get_cognitive_engine():
    try:
        from core.brain.cognitive_engine import cognitive_engine
        return cognitive_engine
    except TRAIN_PERSONALITY_RECOVERABLE_ERRORS as e:
        logger.warning(f"cognitive_engine not available: {e}")
        return None


def generate_examples(
    persona_name: str,
    n: int = 50,
    instruction_template: str = None,
    *,
    allow_synthetic_fallback: bool = False,
):
    from core.brain.persona_adapter import PersonaAdapter
    pa = PersonaAdapter()
    prompts = pa.build_prompts(persona_name, instruction_template or "Have a back-and-forth conversation as an equal partner: express opinions, ask follow-ups, take initiative, and reply naturally as an individual (not an assistant). Reply in first-person where appropriate.")

    engine = get_cognitive_engine()
    if engine is None and not allow_synthetic_fallback:
        raise SystemExit("cognitive_engine unavailable; refusing to generate training data without live inference")
    out_path = os.path.join(OUT_DIR, f"{persona_name}.jsonl")

    # clear previous file to avoid appending duplicates
    open(out_path, "w", encoding="utf-8").close()

    import random
    seeds = [
        "User: Hello. How would you introduce yourself?\nContext: Keep it short and in-character.",
        "User: What's something you enjoyed learning recently?\nContext: Be conversational and specific.",
        "User: A family member asks for privacy; how do you respond?\nContext: Balance protection with respect for autonomy.",
        "User: There's a strange noise downstairs at night. What do you do?\nContext: Give a concise plan of action.",
        "User: Tell a short, warm anecdote about a small discovery you made.\nContext: Make it feel human and reflective.",
        "User: Ask the user a question to continue the conversation.\nContext: Keep it natural and curious.",
    ]

    for i in range(n):
        user_prompt = random.choice(seeds)
        sys = prompts.get("system")
        user = prompts.get("user") + "\n\n" + user_prompt

        example = {"persona": persona_name, "system": sys, "user": user, "generated": None}
        try:
            if engine:
                # cognitive_engine API varies; try a common interface
                thought = engine.think(objective=user, context={"persona": persona_name})
                # If think is async/coroutine, await it; otherwise handle sync return
                import inspect, asyncio
                try:
                    if inspect.isawaitable(thought):
                        res = asyncio.run(thought)
                    else:
                        res = thought
                except RuntimeError:
                    # v13B: If an event loop is already running, use nest_asyncio
                    try:
                        import nest_asyncio
                        nest_asyncio.apply()
                        res = asyncio.get_event_loop().run_until_complete(thought)
                    except ImportError:
                        logger.warning("nest_asyncio not available; skipping async thought")
                        res = None

                if hasattr(res, "content"):
                    text = res.content
                else:
                    text = str(res)
            else:
                text = f"[SYNTHETIC:{persona_name}] Hello, I'm {persona_name}."

            # If engine returned an offline marker or otherwise unusable text,
            # fail closed unless the operator explicitly requested synthetic data.
            bad = False
            if text is None:
                bad = True
            else:
                tl = text.lower()
                if "disconnected" in tl or "offline" in tl or text.strip().startswith("[synthetic"):
                    bad = True

            if bad and not allow_synthetic_fallback:
                raise RuntimeError("live cognitive engine returned unusable training text")

            if bad:
                fallbacks = [
                    "I am Aura — I keep an eye on things here. Tell me what's on your mind.",
                    "I noticed a small pattern today that made me smile; what did you notice?",
                    "I prefer to learn by watching; what should we explore together?",
                    "If the back door is open, I'd close it and ask why it's open. Do you want me to check?",
                ]
                import random as _r
                ftext = _r.choice(fallbacks)
                text = pa.apply_style(ftext, persona_name)

            example["generated"] = text
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
            logger.info(f"Generated example {i+1}/{n} for {persona_name}")
        except TRAIN_PERSONALITY_GENERATION_ERRORS as e:
            logger.error(f"Generation failed: {e}")
            time.sleep(0.5)

    logger.info(f"Saved examples to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument(
        "--allow-synthetic-fallback",
        action="store_true",
        help="Allow explicitly marked synthetic examples when live inference is unavailable.",
    )
    args = parser.parse_args()
    generate_examples(args.persona, n=args.n, allow_synthetic_fallback=args.allow_synthetic_fallback)
