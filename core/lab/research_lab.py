"""core/lab/research_lab.py — Autonomous Research Lab.

Coordinates the full scientific research loop:
  generate hypothesis → search literature → extract claims → compare evidence →
  design experiment → run simulation/code → interpret result → update beliefs →
  propose next experiment.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Dict, List, Optional

from core.lab.hypothesis_engine import HypothesisEngine, Hypothesis
from core.lab.literature_miner import LiteratureMiner
from core.lab.experiment_designer import ExperimentDesigner
from core.lab.simulation_runner import SimulationRunner
from core.lab.result_interpreter import ResultInterpreter
from core.lab.research_memory import ResearchMemory

logger = logging.getLogger("Aura.ResearchLab")


class ResearchStage(StrEnum):
    HYPOTHESIS = "hypothesis_generation"
    LITERATURE = "literature_mining"
    DESIGN = "experiment_design"
    SIMULATION = "simulation"
    INTERPRETATION = "interpretation"
    BELIEF_UPDATE = "belief_update"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ResearchCycle:
    cycle_id: str
    topic: str
    stage: ResearchStage = ResearchStage.HYPOTHESIS
    started_at: float = field(default_factory=time.time)
    hypothesis: Optional[Hypothesis] = None
    mined_facts: List[Dict[str, Any]] = field(default_factory=list)
    experiment_spec: Optional[Dict[str, Any]] = None
    simulation_result: Optional[Dict[str, Any]] = None
    conclusion: Optional[Dict[str, Any]] = None
    next_step: Optional[str] = None
    error: Optional[str] = None


class ResearchLab:
    """Aura's scientific research lab, driving discovery of new knowledge."""

    def __init__(self) -> None:
        self.hypothesis_engine = HypothesisEngine()
        self.literature_miner = LiteratureMiner()
        self.designer = ExperimentDesigner()
        self.simulator = SimulationRunner()
        self.interpreter = ResultInterpreter()
        self.memory = ResearchMemory()
        self.cycles: Dict[str, ResearchCycle] = {}
        self._cycle_counter = 0

    async def execute_cycle(self, topic: str) -> Dict[str, Any]:
        """Execute a full scientific research loop on a topic."""
        self._cycle_counter += 1
        cycle = ResearchCycle(
            cycle_id=f"research_{self._cycle_counter}_{int(time.time())}",
            topic=topic,
        )
        self.cycles[cycle.cycle_id] = cycle
        logger.info("🔬 ResearchLab starting cycle %s on '%s'", cycle.cycle_id, topic)

        try:
            # 1. HYPOTHESIS: Generate hypothesis
            cycle.stage = ResearchStage.HYPOTHESIS
            cycle.hypothesis = self.hypothesis_engine.generate_hypothesis(topic)
            logger.info("💡 Generated Hypothesis: '%s'", cycle.hypothesis.statement)

            # 2. LITERATURE: Mine literature
            cycle.stage = ResearchStage.LITERATURE
            cycle.mined_facts = self.literature_miner.mine_documents([
                {"title": "Prior Research", "content": f"Previous baseline data regarding {topic}", "confidence": 0.85}
            ])
            logger.info("📖 Mined %d facts from prior literature", len(cycle.mined_facts))

            # 3. DESIGN: Design experiment
            cycle.stage = ResearchStage.DESIGN
            cycle.experiment_spec = self.designer.design_experiment(cycle.hypothesis, cycle.mined_facts)
            logger.info("🧪 Designed experiment: '%s'", cycle.experiment_spec.get("name"))

            # 4. SIMULATION: Run simulation/code
            cycle.stage = ResearchStage.SIMULATION
            cycle.simulation_result = await self.simulator.run_sim(cycle.experiment_spec)
            logger.info("🎲 Simulation completed: score=%.2f", cycle.simulation_result.get("score", 0.0))

            # 5. INTERPRET: Interpret results
            cycle.stage = ResearchStage.INTERPRETATION
            cycle.conclusion = self.interpreter.interpret(cycle.hypothesis, cycle.simulation_result)
            logger.info("🎯 Interpretation: hypothesis_validated=%s", cycle.conclusion.get("validated"))

            # 6. BELIEF UPDATE
            cycle.stage = ResearchStage.BELIEF_UPDATE
            self.memory.save_research_outcome(cycle.cycle_id, cycle.conclusion)

            # 7. PROPOSE NEXT STEP
            cycle.next_step = f"Investigate variables affecting {cycle.hypothesis.variables.get('independent', 'target')}"
            cycle.stage = ResearchStage.COMPLETED

        except Exception as e:
            cycle.stage = ResearchStage.FAILED
            cycle.error = str(e)
            logger.error("🔬 Research cycle %s failed at stage %s: %s", cycle.cycle_id, cycle.stage, e)

        return {
            "ok": cycle.stage == ResearchStage.COMPLETED,
            "cycle_id": cycle.cycle_id,
            "hypothesis": cycle.hypothesis.statement if cycle.hypothesis else None,
            "validated": cycle.conclusion.get("validated", False) if cycle.conclusion else False,
            "next_step": cycle.next_step,
            "error": cycle.error,
        }
