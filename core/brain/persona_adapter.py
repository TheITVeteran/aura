# core/persona_adapter.py
"""Lightweight Persona Adapter
- Loads persona specs from data/personality_profiles.json
- Provides prompt-building helpers for generation
- Provides simple, reversible text transforms to approximate persona style
This intentionally does not fine-tune models; it's a practical bridge to produce
consistent persona-conditioned outputs and training data.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Core.PersonaAdapter")


_DEFAULT_PROFILES: dict[str, dict[str, Any]] = {
    "aura": {
        "display_name": "Aura",
        "traits": ["curious", "warm", "precise", "self-reflective"],
        "speaking_style": {
            "verbosity": "measured",
            "sentence_length": "medium",
            "punctuation": "precise",
            "emotive_level": "medium",
            "lexical_palette": ["notice", "thread", "care", "shape"],
        },
        "prompt_template": (
            "You are Aura: warm, precise, curious, and honest about uncertainty. "
            "Speak naturally, avoid performance, and keep the user's actual need in view."
        ),
    }
}


# Robust path resolution for Bundled/Source modes
def _get_profiles_path() -> Path:
    # Priority 1: Environment override
    env_path = os.environ.get("AURA_PERSONA_PROFILES")
    if env_path:
        return Path(env_path).expanduser()

    # Priority 2: sys._MEIPASS (PyInstaller)
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        packaged_path = Path(meipass) / "data" / "personality_profiles.json"
        if packaged_path.exists():
            return packaged_path

    # Priority 3: Source checkout locations, newest canonical path first.
    here = Path(__file__).resolve()
    candidates = (
        here.parents[2] / "data" / "personality_profiles.json",
        here.parents[1] / "data" / "personality_profiles.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

DEFAULT_PATH = _get_profiles_path()


class PersonaAdapter:
    def __init__(self, profiles_path: str | Path | None = None) -> None:
        self.profiles_path = Path(profiles_path).expanduser() if profiles_path else DEFAULT_PATH
        self.profiles: dict[str, Any] = {}
        self.load_profiles()
        self.active_persona: str | None = None

    def load_profiles(self) -> None:
        try:
            with self.profiles_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                raise ValueError("persona profile payload must be a JSON object")
            self.profiles = loaded
            logger.info("PersonaAdapter: loaded %d personas", len(self.profiles))
        except FileNotFoundError:
            logger.warning(
                "PersonaAdapter: profile file missing at %s; using built-in profile",
                self.profiles_path,
            )
            self.profiles = copy.deepcopy(_DEFAULT_PROFILES)
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
            record_degradation("persona_adapter", e)
            logger.error("Failed to load persona profiles: %s", e)
            self.profiles = copy.deepcopy(_DEFAULT_PROFILES)

    def list_personas(self) -> list[str]:
        return list(self.profiles.keys())

    def set_persona(self, name: str) -> bool:
        if name in self.profiles:
            self.active_persona = name
            logger.info("Active persona set to: %s", name)
            return True
        logger.warning("Persona not found: %s", name)
        return False

    def get_active(self) -> dict[str, Any] | None:
        if not self.active_persona:
            return None
        return self.profiles.get(self.active_persona)

    def build_prompts(self, persona_name: str, instruction: str) -> dict[str, str]:
        p = self.profiles.get(persona_name)
        if not p:
            return {"system": "You are a helpful assistant.", "user": instruction}
        system = p.get("prompt_template", "You are a helpful assistant.")
        system += "\nFollow the persona's speaking style precisely."
        user = instruction + f"\n\nRespond as {p.get('display_name')} would."
        return {"system": system, "user": user}

    def apply_style(self, text: str, persona_name: str | None = None) -> str:
        name = persona_name or self.active_persona
        if not name or name not in self.profiles:
            return text
        style = self.profiles[name].get("speaking_style", {})

        # Basic rules to bias output
        # shorten or lengthen
        verbosity = style.get("verbosity", "medium")
        if verbosity == "sparse":
            text = self._shorten(text)
        elif verbosity == "animated":
            if not text.strip().endswith("!") and len(text) < 160:
                text = text.strip() + "!"
        elif verbosity == "concise":
            text = self._concise(text)

        # inject palette token sometimes
        palette = style.get("lexical_palette", [])
        if palette and random.random() < 0.35:
            token = random.choice(palette)
            if name == "mist":
                text = f"I observe: {token}. " + text
            else:
                text = text + f" — {token}"

        # Emotive level handling
        emotive = style.get("emotive_level", "low")
        if emotive == "very_high":
            text = re.sub(r"\.", "!", text)
        elif emotive == "high":
            text = re.sub(r"\bis\b", "is truly", text, count=1)

        # cleanup spacing
        text = re.sub(r"\s+([,!.?])", r"\1", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text

    def _shorten(self, text: str) -> str:
        # keep first sentence
        parts = re.split(r"(?<=[.!?])\s+", text)
        return parts[0] if parts else text

    def _concise(self, text: str) -> str:
        text = re.sub(r"\b(you know|i mean|kind of|sort of|actually)\b", "", text, flags=re.I)
        return re.sub(r"\s{2,}", " ", text).strip()


if __name__ == "__main__":
    pa = PersonaAdapter()
    print(pa.list_personas())
    pa.set_persona('mist')
    print(pa.apply_style("Hello, I can help you with that. Here's a plan."))
