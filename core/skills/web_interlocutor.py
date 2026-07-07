"""Visible web interlocutor skill.

This skill exposes the generic browser-dialogue loop through Aura's normal
CapabilityEngine route. It is for conversations with web AI/chat surfaces or
other text-entry web pages where Aura must visibly send messages, wait for
responses, learn from the exchange, and persist a governed memory.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from core.capabilities.web_interlocutor import WebInterlocutorSession, get_web_interlocutor_job_manager
from core.container import ServiceContainer
from core.skills.base_skill import BaseSkill


class WebInterlocutorParams(BaseModel):
    mode: str = Field("run", description="run|start_background|status|cancel")
    objective: str = Field("", description="What Aura is trying to learn or discuss.")
    url: str = Field("", description="Optional visible web chat URL to open or attach to.")
    opening_message: str = Field("", description="Optional first message. If omitted Aura derives one.")
    max_turns: int = Field(3, description="Number of Aura->interlocutor turns, 1-20.")
    wait_timeout_s: float = Field(45.0, description="Seconds to wait for each visible reply.")
    persist_memory: bool = Field(True, description="Write learned summary through MemoryWriteGateway.")
    job_id: str = Field("", description="Background job id for status/cancel.")

    @field_validator("mode")
    @classmethod
    def _valid_mode(cls, value: str) -> str:
        mode = str(value or "run").strip().lower()
        if mode not in {"run", "start_background", "status", "cancel"}:
            raise ValueError("mode must be run, start_background, status, or cancel")
        return mode

    @field_validator("max_turns")
    @classmethod
    def _bound_turns(cls, value: int) -> int:
        return max(1, min(int(value or 1), 20))


class WebInterlocutorSkill(BaseSkill):
    name = "web_interlocutor"
    description = (
        "Hold a visible, governed conversation with another web AI/chat page in the user's browser, "
        "wait for replies, summarize what was learned, and store it through memory."
    )
    input_model = WebInterlocutorParams
    metabolic_cost = 2
    effect_scope = "foreground_browser_dialogue"
    timeout_seconds = 420.0

    async def execute(self, params: Any, context: dict[str, Any]) -> dict[str, Any]:
        if isinstance(params, WebInterlocutorParams):
            parsed = params
        else:
            parsed = WebInterlocutorParams(**dict(params or {}))
        if parsed.mode == "status":
            return get_web_interlocutor_job_manager().status(parsed.job_id)
        if parsed.mode == "cancel":
            return get_web_interlocutor_job_manager().cancel(parsed.job_id)
        brain = (context or {}).get("brain") or ServiceContainer.get("cognitive_engine", default=None)
        if parsed.mode == "start_background":
            return get_web_interlocutor_job_manager().start(
                objective=parsed.objective,
                url=parsed.url,
                opening_message=parsed.opening_message,
                max_turns=parsed.max_turns,
                wait_timeout_s=parsed.wait_timeout_s,
                persist_memory=parsed.persist_memory,
                context={**(context or {}), "brain": brain, "background": True},
            )
        session = WebInterlocutorSession(cognitive_engine=brain)
        result = await session.run(
            objective=parsed.objective,
            url=parsed.url,
            opening_message=parsed.opening_message,
            max_turns=parsed.max_turns,
            wait_timeout_s=parsed.wait_timeout_s,
            persist_memory=parsed.persist_memory,
            context=context or {},
        )
        payload = result.to_dict()
        payload["summary"] = (
            f"Completed {len(result.turns)} visible web interlocutor turns and stored learned memory "
            f"{result.memory_record_id or 'not requested'}."
            if result.ok
            else f"Web interlocutor session failed: {result.status}"
        )
        return payload
