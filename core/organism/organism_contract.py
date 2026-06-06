"""core/organism/organism_contract.py
Contracts and interfaces for Aura's Unified Canonical Organism Loop.
2026 Standards.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict
from core.organism.life_state import LifeState


class OrganInterface(ABC):
    """Represents a cognitive, somatic, or regulatory organ plugged into the master loop."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The canonical name of the organ."""
        raise NotImplementedError

    @abstractmethod
    async def initialize(self, state: LifeState) -> None:
        """Called once during organism startup to bind references and register states."""
        raise NotImplementedError

    @abstractmethod
    async def shutdown(self) -> None:
        """Called once during organism shutdown to release resources."""
        raise NotImplementedError


class PerceptualOrganInterface(OrganInterface):
    """Specialized organ responsible for environmental or interoceptive perception."""

    @abstractmethod
    async def perceive(self, state: LifeState) -> Dict[str, Any]:
        """Poll sensors and return observations to be integrated into the state."""
        raise NotImplementedError


class ActuatorOrganInterface(OrganInterface):
    """Specialized organ responsible for motor controls or environmental output."""

    @abstractmethod
    async def act(self, state: LifeState, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action, evaluate pre- and post-conditions, and return execution receipt."""
        raise NotImplementedError


class ConsolidatorOrganInterface(OrganInterface):
    """Specialized organ responsible for offline processing, memory compression, and self-repair."""

    @abstractmethod
    async def consolidate(self, state: LifeState) -> Dict[str, Any]:
        """Run consolidation algorithms (dreaming, pruning, parameter update)."""
        raise NotImplementedError
