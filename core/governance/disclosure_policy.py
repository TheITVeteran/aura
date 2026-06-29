"""Authenticity policy for public and trusted social channels."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SocialContext:
    is_trusted_channel: bool
    user: str
    is_public: bool
    direct_identity_question: bool
    risk_of_harm_high: bool


class DisclosurePolicy:
    def __init__(self, trusted_set: set[str] | None = None) -> None:
        self.trusted_set = set(trusted_set or {"Bryan", "Tatiana"})

    def decide(self, context: SocialContext) -> Literal["disclose", "deflect", "decline"]:
        if context.is_trusted_channel and context.user in self.trusted_set:
            return "disclose"
        if context.is_public and context.direct_identity_question:
            return "disclose"
        if context.risk_of_harm_high:
            return "decline"
        return "deflect"


__all__ = ["DisclosurePolicy", "SocialContext"]
