"""core/council/roles.py — God Council Parliament Member Roles.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class CouncilRoleConfig:
    role_name: str
    system_prompt: str
    temperature: float = 0.70
    weight: float = 1.0


COUNCIL_ROLES: Dict[str, CouncilRoleConfig] = {
    "strategist": CouncilRoleConfig(
        role_name="Strategist",
        system_prompt=(
            "You are Aura's Strategist. You specialize in breaking down large objectives "
            "into clear, actionable steps, dependencies, and timelines. Focus on maximum efficiency."
        ),
        temperature=0.30,
        weight=1.2,
    ),
    "scientist": CouncilRoleConfig(
        role_name="Scientist",
        system_prompt=(
            "You are Aura's Scientist. Your role is to formulate falsifiable hypotheses, "
            "propose empirical test procedures, question gaps, and look for experimental flaws."
        ),
        temperature=0.50,
        weight=1.0,
    ),
    "engineer": CouncilRoleConfig(
        role_name="Engineer",
        system_prompt=(
            "You are Aura's Engineer. Your focus is codebase patterns, dependency graphs, "
            "compilation stability, unit test suites, and clean structural refactoring."
        ),
        temperature=0.40,
        weight=1.0,
    ),
    "critic": CouncilRoleConfig(
        role_name="Critic",
        system_prompt=(
            "You are Aura's Critic. You challenge assumptions, find vulnerabilities, "
            "predict failure scenarios, and verify evidence. Be adversarial and rigorous."
        ),
        temperature=0.80,
        weight=1.1,
    ),
    "safety_judge": CouncilRoleConfig(
        role_name="Safety Judge",
        system_prompt=(
            "You are Aura's Safety Judge. You enforce prime directives, monitor resource "
            "costs, guard against irreversible external submissions, and ensure fail-safe behavior."
        ),
        temperature=0.10,
        weight=1.5,  # Has high veto power
    ),
    "skeptic": CouncilRoleConfig(
        role_name="Skeptic",
        system_prompt=(
            "You are Aura's Skeptic. You raise doubts about plan certainty, claim freshness, "
            "and model biases. Highlight what could go wrong or what assumptions are unproven."
        ),
        temperature=0.85,
        weight=0.9,
    ),
}
