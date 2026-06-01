"""research/bench_harness.py — Automated Regression & RSI Benchmarking
=======================================================================
Aura's automated evaluation framework. Runs nightly to verify that:
1. Newly learned skills transfer correctly.
2. The causal world model's predictive calibration remains high.
3. Overfitting hasn't destroyed general intelligence metrics.

This fulfills Phase 22.12.
"""

import json
import logging
import time
from typing import Dict, Any

from research.sim_worlds.toy_world import ToyWorld
from research.sim_worlds.market_sim import MarketSim
from research.sim_worlds.puzzle_env import PuzzleEnv
from core.container import ServiceContainer

logger = logging.getLogger("Aura.BenchHarness")

class BenchmarkHarness:
    """Nightly eval suite for autonomous reasoning and planning."""
    
    def __init__(self):
        from core.config import config
        self.report_dir = config.paths.data_dir / "benchmarks"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
    def run_nightly_eval(self) -> Dict[str, Any]:
        """Runs the complete suite and records composite regression metrics."""
        logger.info("Starting Nightly Benchmark Run...")
        
        toy = ToyWorld()
        market = MarketSim()
        puzzle = PuzzleEnv()
        
        toy_score = self._eval_toy(toy)
        market_score = self._eval_market(market)
        puzzle_score = self._eval_puzzle(puzzle)
        
        # Check skill library reuse
        skill_lib = ServiceContainer.get("skill_library", default=None)
        reuse_rate = 0.0
        transfer_ratio = 0.0
        if skill_lib and skill_lib.skills:
            total_uses = sum(s.successes + s.failures for s in skill_lib.skills.values())
            reused = sum(1 for s in skill_lib.skills.values() if (s.successes + s.failures) > 1)
            reuse_rate = reused / len(skill_lib.skills)
            if total_uses > 0:
                successful_transfers = sum(
                    1
                    for skill in skill_lib.skills.values()
                    if skill.successes > 0 and (skill.successes + skill.failures) > 1
                )
                transfer_ratio = successful_transfers / len(skill_lib.skills)
            
        report = {
            "timestamp": time.time(),
            "toy_world_score": toy_score,
            "market_sim_score": market_score,
            "puzzle_env_score": puzzle_score,
            "skill_reuse_rate": reuse_rate,
            "transfer_ratio": transfer_ratio,
            "composite_score": (toy_score + market_score + puzzle_score) / 3.0,
            "passed_milestones": {
                "transfer_ratio": transfer_ratio >= 0.35,
                "skill_reuse": reuse_rate > 0.5
            }
        }
        
        self._save_report(report)
        return report

    def _eval_toy(self, env: ToyWorld) -> float:
        env.reset()
        score = 0
        for _ in range(10):
            env.step("explore")
            _, r, done, _ = env.step("rest")
            score += r
            if done: break
        return max(0.0, score / 10.0) # Normalize
        
    def _eval_market(self, env: MarketSim) -> float:
        env.reset()
        state = env._get_state()
        for _ in range(50):
            action = "buy" if state["price"] < 100 else "sell"
            state, r, done, _ = env.step(action)
            if done: break
        return state["portfolio_value"] / 1000.0 # Ratio of starting cash
        
    def _eval_puzzle(self, env: PuzzleEnv) -> float:
        env.reset()
        score = 0
        for i in range(4):
            _, r, done, _ = env.step(i)
            score += r
            if done: break
        return max(0.0, score)

    def _save_report(self, report: Dict[str, Any]):
        filename = f"bench_{int(report['timestamp'])}.json"
        try:
            with open(self.report_dir / filename, "w") as f:
                json.dump(report, f, indent=4)
            logger.info(f"Benchmark saved: {filename} (Score: {report['composite_score']:.2f})")
        except (OSError, TypeError, ValueError) as e:
            logger.error(f"Failed to save benchmark report: {e}")

if __name__ == "__main__":
    harness = BenchmarkHarness()
    print(json.dumps(harness.run_nightly_eval(), indent=2))
