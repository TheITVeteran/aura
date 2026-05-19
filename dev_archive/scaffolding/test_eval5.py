import tempfile
from pathlib import Path

from core.learning.autonomous_rsi import AutonomousSuccessorEngine

with tempfile.TemporaryDirectory() as tmp:
    res = AutonomousSuccessorEngine(tmp, seed=4401, tasks_per_generation=40).run(generations=1)
    for r in res.artifacts:
        with open(Path(r.directory) / "solver.py") as f:
            print(f"--- {r.generation_id} SOURCE ---")
            print(f.read())
