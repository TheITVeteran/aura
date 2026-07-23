"""Shared value object for bounded latent-state counterfactual probes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from core.brain.llm.latent_cortex.verified_best import (
    VerifierObservation,
    validate_observation,
)


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class CounterfactualProbeResult:
    """One fixed-compute probe result; generated answer text is discarded."""

    probe_tokens_sha256: str
    probe_token_count: int
    observation: Mapping[str, Any]
    layer_apps: int

    def normalized(self) -> dict[str, Any]:
        observation = (
            validate_observation(self.observation)
            if "observation_sha256" in self.observation
            else VerifierObservation.from_value(self.observation).to_dict()
        )
        if (
            not is_sha256(self.probe_tokens_sha256)
            or type(self.probe_token_count) is not int
            or self.probe_token_count <= 0
            or type(self.layer_apps) is not int
            or self.layer_apps <= 0
        ):
            raise ValueError("counterfactual probe result is invalid")
        return {
            "probe_tokens_sha256": self.probe_tokens_sha256,
            "probe_token_count": self.probe_token_count,
            "observation": observation,
            "layer_apps": self.layer_apps,
        }


__all__ = ["CounterfactualProbeResult", "is_sha256"]
