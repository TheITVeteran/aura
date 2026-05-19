import tempfile

from core.learning.autonomous_rsi import AutonomousSuccessorEngine

with tempfile.TemporaryDirectory() as tmp:
    res = AutonomousSuccessorEngine(tmp, seed=4401, tasks_per_generation=40).run(generations=4)
    print("VERDICT:", res.verdict.verdict)
    for r in res.records:
        print(f"{r.generation_id}: promoted={r.promoted} after_score={r.after_score} hidden={r.hidden_eval_score}")
