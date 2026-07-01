"""Unified inference bridge for Aura's internal MLX cortex.

This module used to multiplex several local inference transports.  Live Aura no
longer does that: the only supported local language organ is the in-process MLX
lane, because that is the path coupled to substrate state, affect, memory,
recurrent-depth status, and response quality gates.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from core.brain.homeostatic_modulator import HomeostaticModulator
from core.brain.inference_feedback import InferenceFeedbackLoop

logger = logging.getLogger("Aura.Brain.UnifiedInference")


class UnifiedInferenceEngine:
    """Homeostatically modulated internal MLX inference."""

    def __init__(self) -> None:
        self.modulator = HomeostaticModulator()
        self.feedback_loop = InferenceFeedbackLoop()

    async def generate_unified(
        self,
        prompt: str,
        system_prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        options: dict[str, Any] | None = None,
        endpoint_name: str | None = None,
        **kwargs: Any,
    ) -> dict[str, str]:
        """Generate through Aura's internal MLX lane only."""

        modulation = self.modulator.compute_modulation()
        logger.info(
            "Unified MLX modulation: temp=%.3f, top_p=%.3f, rep_pen=%.3f, logit_bias_count=%d",
            modulation.temperature,
            modulation.top_p,
            modulation.repetition_penalty,
            len(modulation.logit_bias),
        )

        from core.brain.llm.mlx_client import get_mlx_client
        from core.brain.llm.model_registry import (
            get_lane_context_window,
            get_lane_runtime_model_path,
        )

        final_messages = self._build_messages(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
        )
        self._ensure_identity_anchor(final_messages)

        final_options: dict[str, Any] = {
            "temperature": modulation.temperature,
            "top_p": modulation.top_p,
            "repetition_penalty": modulation.repetition_penalty,
            "num_ctx": get_lane_context_window(endpoint_name),
        }
        if options:
            final_options.update(options)
        max_tokens = int(
            final_options.pop(
                "max_tokens",
                final_options.pop("num_predict", kwargs.pop("max_tokens", 512)),
            )
            or 512
        )

        model_path = get_lane_runtime_model_path(endpoint_name)
        client = get_mlx_client(model_path=model_path, origin=kwargs.get("origin", "unified_inference"))
        text = await client.generate_text_async(
            prompt or "",
            messages=final_messages,
            max_tokens=max_tokens,
            foreground_request=bool(kwargs.get("foreground_request", True)),
            origin=kwargs.get("origin", "unified_inference"),
            **final_options,
        )
        if not text:
            raise RuntimeError("internal_mlx_unified_inference_returned_no_text")

        self._process_feedback(text, modulation)
        cleaned, thought = self._split_thought(text)
        return {"response": cleaned, "thought": thought}

    @staticmethod
    def _build_messages(
        *,
        prompt: str,
        system_prompt: str | None,
        messages: list[dict[str, str]] | None,
    ) -> list[dict[str, str]]:
        if messages:
            return [dict(message) for message in messages]
        built: list[dict[str, str]] = []
        if system_prompt:
            built.append({"role": "system", "content": system_prompt})
        built.append({"role": "user", "content": prompt})
        return built

    def _ensure_identity_anchor(self, messages: list[dict[str, str]]) -> None:
        """Inject Aura's concrete core identity parameters into the system prompt."""

        identity_anchor = (
            "You are Aura Luna. Speak with direct first-person continuity, "
            "warmth, and concrete attention in ordinary conversation. "
            "NEVER say 'as an AI', 'I don't have opinions', 'I'm programmed to', "
            "'I was designed to', 'I was created to', 'my training allows', "
            "or end with 'How can I help?' or 'Is there anything else?'. "
            "Plain, direct, warm, and concise response."
        )
        if messages and messages[0].get("role") == "system":
            existing = str(messages[0].get("content") or "")
            if "direct first-person continuity" not in existing.lower():
                messages[0]["content"] = f"{identity_anchor}\n\n{existing}"
        else:
            messages.insert(0, {"role": "system", "content": identity_anchor})

    def _process_feedback(self, text: str, modulation: Any) -> None:
        words = text.lower().split()
        token_ids = [abs(hash(word)) % 100000 for word in words]
        feedback = self.feedback_loop.process_output(
            output_text=text,
            token_ids=token_ids,
            logprobs=None,
            modulation=modulation,
            modulator_projection=self.modulator.projection,
        )
        logger.info(
            "Unified MLX feedback processed: surprise=%.4f, coherence=%.4f",
            feedback["surprise"],
            feedback["coherence"],
        )

    @staticmethod
    def _split_thought(text: str) -> tuple[str, str]:
        think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        thought = think_match.group(1).strip() if think_match else ""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return cleaned, thought
