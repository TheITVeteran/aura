"""Unified inference bridge for Aura's internal MLX cortex.

This module used to multiplex several local inference transports.  Live Aura no
longer does that: the only supported local language organ is the in-process MLX
lane, because that is the path coupled to substrate state, affect, memory,
recurrent-depth status, and response quality gates.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from typing import Any

from core.brain.homeostatic_modulator import HomeostaticModulator
from core.brain.inference_feedback import InferenceFeedbackLoop
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Brain.UnifiedInference")

#: Arguments generate_text_async receives EXPLICITLY. A caller option carrying
#: any of these names would be splatted alongside the explicit value and raise
#: "multiple values for argument" from inside the client.
_RESERVED_CALL_ARGS = frozenset({
    "messages", "max_tokens", "foreground_request", "origin", "prompt",
})

#: Bound on how many lexical surrogate ids the fallback will emit, so a very
#: long generation cannot hand the feedback loop an unbounded vector.
_MAX_FALLBACK_TOKENS = 2048


def _request_deadline_s() -> float:
    """Wall-clock ceiling for one unified generation."""
    try:
        value = float(os.getenv("AURA_UNIFIED_INFERENCE_TIMEOUT_S", "300") or 300)
    except (TypeError, ValueError):
        value = 300.0
    return min(3600.0, max(5.0, value))


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
            # Caller options are merged then splatted as **final_options AFTER
            # explicit messages/max_tokens/foreground_request/origin arguments.
            # A caller passing any of those names produced a
            # "multiple values for argument" TypeError from deep inside the
            # client — a request-shape error surfacing as a crash. Reserved
            # names are dropped here, loudly, rather than allowed to collide.
            colliding = _RESERVED_CALL_ARGS & set(options)
            if colliding:
                logger.warning(
                    "Unified inference: ignoring caller option(s) %s — they are "
                    "passed explicitly and would collide at the call boundary.",
                    ", ".join(sorted(colliding)),
                )
            final_options.update(
                {k: v for k, v in options.items() if k not in _RESERVED_CALL_ARGS}
            )
        max_tokens = int(
            final_options.pop(
                "max_tokens",
                final_options.pop("num_predict", kwargs.pop("max_tokens", 512)),
            )
            or 512
        )

        model_path = get_lane_runtime_model_path(endpoint_name)
        client = get_mlx_client(model_path=model_path, origin=kwargs.get("origin", "unified_inference"))
        # max_tokens was the ONLY bound: no wall-clock deadline, so a wedged
        # or pathologically slow generation held the lane indefinitely with no
        # cancellation point. The ceiling is generous and env-overridable — it
        # exists to catch a hang, not to cut off legitimately long work.
        try:
            text = await asyncio.wait_for(
                client.generate_text_async(
                    prompt or "",
                    messages=final_messages,
                    max_tokens=max_tokens,
                    foreground_request=bool(kwargs.get("foreground_request", True)),
                    origin=kwargs.get("origin", "unified_inference"),
                    **final_options,
                ),
                timeout=_request_deadline_s(),
            )
        except TimeoutError as exc:
            record_degradation(
                "unified_inference", exc, severity="warning",
                action=f"abandoned a generation that exceeded {_request_deadline_s():.0f}s",
            )
            raise RuntimeError("internal_mlx_unified_inference_deadline_exceeded") from exc
        if not text:
            raise RuntimeError("internal_mlx_unified_inference_returned_no_text")

        intero_getter = getattr(client, "get_last_interoception", None)
        interoception = intero_getter() if callable(intero_getter) else None
        self._process_feedback(text, modulation, interoception=interoception)
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

    def _process_feedback(
        self, text: str, modulation: Any, interoception: dict[str, Any] | None = None
    ) -> None:
        """Feed the homeostatic loop from the generation that actually happened.

        When the worker shipped an interoception trace for this exact text
        (fingerprint-verified), the loop runs on REAL sampled-token ids and
        log-probabilities from the substrate. The word-hash fallback survives
        only for interoception-less runs (tap disabled, degraded worker).
        """
        token_ids: list[int] | None = None
        logprobs: list[float] | None = None
        intero = interoception or {}
        if intero:
            try:
                from core.being.thought_interoception import text_fingerprint

                if intero.get("_text_fingerprint") == text_fingerprint(text):
                    ids = [int(i) for i in (intero.get("token_ids_sample") or [])]
                    lps = [float(v) for v in (intero.get("logprob_sample") or [])]
                    if ids and lps:
                        token_ids = ids
                        logprobs = lps
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                logger.debug("Interoception pairing unavailable for feedback: %s", exc)

        if token_ids is None:
            # Legacy heuristic path. These are NOT tokenizer ids and never were:
            # Python's hash() is salted per process, so the same word produced a
            # different "token id" on every run and none of them corresponded to
            # anything in the vocabulary. Feeding those into the homeostatic loop
            # as generation substrate was feeding it noise that merely looked
            # like measurement.
            #
            # The fallback is kept — the loop still needs lexical structure when
            # no trace exists — but it is made DETERMINISTIC (stable across
            # processes and runs, so the same text yields the same signal) and
            # bounded, and the caller is told plainly that this is not substrate
            # evidence.
            words = text.lower().split()[:_MAX_FALLBACK_TOKENS]
            token_ids = [
                int.from_bytes(
                    hashlib.blake2b(w.encode("utf-8", "ignore"), digest_size=4).digest(),
                    "big",
                ) % 100000
                for w in words
            ]
            if words:
                logger.debug(
                    "Unified inference feedback running on LEXICAL surrogate ids "
                    "(no interoception trace): %d words.", len(words)
                )

        feedback = self.feedback_loop.process_output(
            output_text=text,
            token_ids=token_ids,
            logprobs=logprobs,
            modulation=modulation,
            modulator_projection=self.modulator.projection,
            # With a real trace, the thought-interoception organ has already fed
            # the homeostatic engines for this generation — only train here.
            feed_engines=logprobs is None,
        )
        logger.info(
            "Unified MLX feedback processed: surprise=%.4f, coherence=%.4f, grounded=%s",
            feedback["surprise"],
            feedback["coherence"],
            "real_logprobs" if logprobs else "lexical_fallback",
        )

    @staticmethod
    def _split_thought(text: str) -> tuple[str, str]:
        think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        thought = think_match.group(1).strip() if think_match else ""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return cleaned, thought
