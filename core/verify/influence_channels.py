"""core/verify/influence_channels.py

Canonical ids for the channels through which a faculty can reach the output.

Channel ids are a contract, the same way telemetry ids are: a verdict recorded
against ``live_mind.steering_alpha`` is only comparable across boots and across
releases if that string means the same thing every time. Never reuse an id for
a different channel; retire it and mint a new one.

The split that matters is DIRECT versus TEXT_MEDIATED. A faculty whose only
actuator is a sentence in the system prompt is doing prompt engineering however
much machinery sits behind the sentence — the model is being told about the
mind rather than driven by it. Both kinds are measurable and both are listed
here, but they are not the same claim and the receipt keeps them apart.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "LIVE_MIND_GENERATION_CONTROLS",
    "LIVE_MIND_STEERING_ALPHA",
    "LIVE_MIND_RECURRENT_LOOPS",
    "LIVE_MIND_CONTEXT_BLOCK",
    "AFFECT_GENERATION_CONTROLS",
    "SPIKING_SAMPLING_BIAS",
    "IMAGINATION_SAMPLING_BIAS",
    "BICAMERAL_SAMPLING_BIAS",
    "QUALIA_RICHNESS",
    "DIRECT_ACTUATION_CHANNELS",
    "TEXT_MEDIATED_CHANNELS",
    "ALL_CHANNELS",
]

# --- Direct actuation: numbers that reach the sampler or the latent loop ----

#: Temperature and top-p derived from the live mind snapshot.
LIVE_MIND_GENERATION_CONTROLS: Final = "live_mind.generation_controls"

#: Steering coefficient handed to the Recursive Latent Cortex.
LIVE_MIND_STEERING_ALPHA: Final = "live_mind.steering_alpha"

#: How many recurrent passes the latent cortex runs for this turn.
LIVE_MIND_RECURRENT_LOOPS: Final = "live_mind.recurrent_loops"

#: Affective control_effects → temperature/token-budget/repetition modulation.
AFFECT_GENERATION_CONTROLS: Final = "affect.generation_controls"

#: Spiking active-inference sampling bias.
SPIKING_SAMPLING_BIAS: Final = "spiking.sampling_bias"

#: Imagination workspace sampling bias.
IMAGINATION_SAMPLING_BIAS: Final = "imagination.sampling_bias"

#: Bicameral advisory sampling bias.
BICAMERAL_SAMPLING_BIAS: Final = "bicameral.sampling_bias"

# --- Text mediated: state serialized into the prompt -----------------------

#: The [LIVE MIND CONTEXT] block: mind snapshot, substrate, governance, lane,
#: serialized as JSON into the system prompt, followed by a sentence asserting
#: that it is "causal grounding for the reply". Whether it is, is what the
#: measurement decides.
LIVE_MIND_CONTEXT_BLOCK: Final = "live_mind.context_block"

#: Phenomenal richness as it reaches any consumer downstream.
QUALIA_RICHNESS: Final = "qualia.richness"


DIRECT_ACTUATION_CHANNELS: Final = frozenset(
    {
        LIVE_MIND_GENERATION_CONTROLS,
        LIVE_MIND_STEERING_ALPHA,
        LIVE_MIND_RECURRENT_LOOPS,
        AFFECT_GENERATION_CONTROLS,
        SPIKING_SAMPLING_BIAS,
        IMAGINATION_SAMPLING_BIAS,
        BICAMERAL_SAMPLING_BIAS,
    }
)

TEXT_MEDIATED_CHANNELS: Final = frozenset(
    {
        LIVE_MIND_CONTEXT_BLOCK,
        QUALIA_RICHNESS,
    }
)

ALL_CHANNELS: Final = DIRECT_ACTUATION_CHANNELS | TEXT_MEDIATED_CHANNELS
