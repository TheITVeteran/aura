"""Compatibility facade over Aura's canonical governed skill contract.

Legacy project skills still import ``infrastructure.BaseSkill``.  They now use
the same governance, timeout, retry, result, and execution-stat machinery as
``core.skills.base_skill.BaseSkill`` while retaining the older ``inputs`` JSON
schema helper during migration.
"""

from __future__ import annotations

import asyncio  # noqa: F401 - compatibility tests and callers patch this shared module.
import json
import logging
import re
from typing import Any

from core.skills.base_skill import BaseSkill as _CanonicalBaseSkill

logger = logging.getLogger("Aura.Skills.Compatibility")


class BaseSkill(_CanonicalBaseSkill):
    """Canonical BaseSkill with the legacy declarative-input adapter."""

    name: str = "unknown_skill"
    description: str = "No description provided."
    inputs: dict[str, str] = {}
    output: str = "Result string or dict"
    aliases: list[str] = []

    async def extract_and_validate_args(
        self,
        raw_input: str,
        llm_client: Any | None = None,
    ) -> dict[str, Any]:
        match = re.search(r"(\{.*\})", raw_input, re.DOTALL)
        candidate = match.group(1) if match else raw_input
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as original_error:
            generator = getattr(llm_client, "generate_text_async", None)
            if not callable(generator):
                raise ValueError(f"Invalid JSON for {self.name} and no recovery client available") from original_error
            recovery_prompt = (
                "Extract one valid JSON object matching this schema and return only JSON: "
                f"{json.dumps(self.to_json_schema(), sort_keys=True)}. Input: {raw_input}"
            )
            recovered_raw = str(await generator(recovery_prompt, model="llama3"))
            recovered_match = re.search(r"(\{.*\})", recovered_raw, re.DOTALL)
            recovered_candidate = recovered_match.group(1) if recovered_match else recovered_raw
            try:
                parsed = json.loads(recovered_candidate)
            except (json.JSONDecodeError, TypeError, ValueError) as recovery_error:
                raise ValueError(
                    f"Could not parse valid arguments for {self.name}: {recovery_error}"
                ) from original_error
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object for {self.name}")
        missing = [name for name in self.inputs if name not in parsed]
        if missing:
            logger.warning("Skill %s missing declared keys: %s", self.name, missing)
        return parsed

    def to_json_schema(self) -> dict[str, Any]:
        properties = {
            name: {"description": description, "type": "string"}
            for name, description in self.inputs.items()
        }
        return {
            "function": {
                "description": self.description,
                "name": self.name,
                "parameters": {
                    "additionalProperties": False,
                    "properties": properties,
                    "required": list(properties),
                    "type": "object",
                },
            },
            "type": "function",
        }
