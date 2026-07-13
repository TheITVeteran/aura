"""Social subsystem exports."""

from __future__ import annotations

from typing import Any

from .dialogue_cognition import (
    DialogueCognitionEngine,
    DialogueCognitionProfile,
    get_dialogue_cognition,
)
from .social_imagination import (
    SocialImagination,
    SocialImaginationFrame,
    get_social_imagination,
)

__all__ = [
    "DialogueCognitionEngine",
    "DialogueCognitionProfile",
    "get_dialogue_cognition",
    "SocialImagination",
    "SocialImaginationFrame",
    "TheoryOfMindEngine",
    "get_social_imagination",
]


def __getattr__(name: str) -> Any:
    if name == "TheoryOfMindEngine":
        from .theory_of_mind import TheoryOfMindEngine

        return TheoryOfMindEngine
    raise AttributeError(name)
