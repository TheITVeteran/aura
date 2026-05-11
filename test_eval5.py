import sys
sys.path.append("/Users/bryan/.aura/live-source")
from core.learning.autonomous_rsi import AutonomousSuccessorEngine
import tempfile
with tempfile.TemporaryDirectory() as tmp:
    res = AutonomousSuccessorEngine(tmp, seed=4401, tasks_per_generation=40).run(generations=1)
    for r in res.artifacts:
        with open(r.directory + "/solver.py", "r") as f:
            print(f"--- {r.generation_id} SOURCE ---")
            print(f.read())
