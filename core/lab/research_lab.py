"""core/lab/research_lab.py — Autonomous Research Lab Orchestrator.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from core.lab.hypothesis_engine import HypothesisEngine
from core.lab.experiment_designer import ExperimentDesigner
from core.lab.simulation_runner import SimulationRunner
from core.lab.result_interpreter import ResultInterpreter
from core.lab.research_memory import ResearchMemo, ResearchMemoryStore

logger = logging.getLogger("Aura.ResearchLab")


class ResearchLab:
    """Coordinates hypothesis generation, literature mining, and experiment runner."""

    def __init__(self) -> None:
        self.memory_store = ResearchMemoryStore()

    async def run_research(self, topic: str) -> Dict[str, Any]:
        """Runs the entire research loop: hypothesis -> design -> run -> verify -> save."""
        logger.info("🔬 Autonomous Research Lab convenes for topic: '%s'", topic)

        # 1. Formulate Hypothesis
        hyp = await HypothesisEngine.formulate_hypothesis(topic)

        # 2. Design Protocol
        protocol = ExperimentDesigner.design_protocol(hyp.hypothesis_id, hyp.statement)

        # 3. Run Simulation/Experiment
        raw_results = await SimulationRunner.execute_protocol(protocol)

        # 4. Interpret Results
        interpretation = ResultInterpreter.interpret(raw_results, hyp.verification_metric)

        # 5. Compile Research Memo & Save
        validated = interpretation.get("hypothesis_validated", False)
        summary = (
            f"Research on {topic} was executed under protocol {protocol.protocol_id}. "
            f"The hypothesis '{hyp.statement}' was {'VALIDATED' if validated else 'REJECTED'} "
            f"based on {interpretation.get('interpretation_details')}."
        )
        memo = ResearchMemo(
            memo_id=f"memo_{hyp.hypothesis_id}",
            topic=topic,
            hypothesis_statement=hyp.statement,
            validated=validated,
            data_points=raw_results,
            summary_prose=summary,
        )
        self.memory_store.save_memo(memo)

        return {
            "ok": True,
            "topic": topic,
            "hypothesis_id": hyp.hypothesis_id,
            "statement": hyp.statement,
            "validated": validated,
            "memo_id": memo.memo_id,
            "summary": summary,
        }


# Singleton
_lab_instance: ResearchLab | None = None


def get_research_lab() -> ResearchLab:
    global _lab_instance
    if _lab_instance is None:
        _lab_instance = ResearchLab()
    return _lab_instance
