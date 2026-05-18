from core.learning.autonomous_rsi import generate_solver_source, solve_with_generated_code
from core.learning.successor_lab import Task

source = generate_solver_source({"gcd", "mod"}, generation_id="Aura-G1")
task = Task(task_id="t1", kind="gcd", metadata={"a": 10, "b": 5})
res = solve_with_generated_code(task, source)
print("SOURCE:\n", source)
print("RES:", res)
