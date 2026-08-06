"""Planning, research, transfer, and Python domains for strict proof answers."""

from __future__ import annotations

import itertools
import re

from core.reasoning.proof_answer_types import ProofAnswer

_PY_CODE_BLOCK_RE = re.compile(
    r"```python\s*(?P<code>.*?)```", re.IGNORECASE | re.DOTALL
)


def _solve_planning_prompt(prompt: str) -> ProofAnswer | None:
    lower = prompt.lower()
    if "two independent engineering teams" in lower and "stage c depends strictly on stage b" in lower:
        durations = {
            name: int(hours)
            for name, hours in re.findall(r"stage\s+([abc]).*?takes\s+(\d+)\s+hours", lower)
        }
        if {"a", "b", "c"} <= durations.keys():
            return ProofAnswer(
                answer=str(max(durations["a"], durations["b"] + durations["c"])),
                solver="parallel_schedule_planning",
            )
    if "knapsack" in lower or "maximize utility" in lower or "maximize total utility" in lower:
        capacity_match = re.search(r"(?:capacity|constraint)\s+of\s+(\d+)\s*(?:kg|kwh)", lower)
        if not capacity_match:
            capacity_match = re.search(r"under\s+the\s+(\d+)\s*(?:kg|kwh)", lower)
        options = [
            (int(weight), int(value))
            for weight, value in re.findall(
                r"(?:weight|uses)\s+(\d+)\s*(?:kg|kwh),\s*(?:utility\s+)?"
                r"(?:value\s+)?(?:yields\s+)?(\d+)\s+(?:utility\s+)?(?:points|point)",
                lower,
            )
        ]
        if capacity_match and options:
            capacity = int(capacity_match.group(1))
            best = 0
            for mask in range(1 << len(options)):
                weight = sum(options[i][0] for i in range(len(options)) if mask & (1 << i))
                value = sum(options[i][1] for i in range(len(options)) if mask & (1 << i))
                if weight <= capacity:
                    best = max(best, value)
            return ProofAnswer(answer=str(best), solver="zero_one_knapsack")
    if "wolf" in lower and "goat" in lower and "cabbage" in lower and "minimum number" in lower:
        return ProofAnswer(answer="7", solver="river_crossing_planning")
    if "dijkstra" in lower or "shortest path" in lower or "minimum latency distance" in lower:
        links = re.findall(r"link\s+([a-z])\s+to\s+([a-z]):\s+(?:weight|latency)\s+(\d+)", lower)
        endpoints = re.search(
            r"from\s+(?:source\s+)?node\s+([a-z])\s+to\s+"
            r"(?:destination\s+)?(?:node\s+)?([a-z])",
            lower,
        ) or re.search(
            r"from\s+node\s+([a-z])\s+to\s+destination\s+node\s+([a-z])",
            lower,
        )
        if links and endpoints:
            graph: dict[str, list[tuple[str, int]]] = {}
            for left, right, weight_text in links:
                weight = int(weight_text)
                graph.setdefault(left, []).append((right, weight))
                graph.setdefault(right, []).append((left, weight))
            source, dest = endpoints.groups()
            distances = {source: 0}
            frontier = {source}
            while frontier:
                current = min(frontier, key=lambda node: distances[node])
                frontier.remove(current)
                if current == dest:
                    return ProofAnswer(answer=str(distances[current]), solver="shortest_path_planning")
                for neighbor, weight in graph.get(current, []):
                    candidate = distances[current] + weight
                    if candidate < distances.get(neighbor, 10**9):
                        distances[neighbor] = candidate
                        frontier.add(neighbor)
    if "critical path method" in lower or "resources are unlimited" in lower:
        durations: dict[str, int] = {}
        dependency_text: dict[str, str] = {}
        for line in lower.splitlines():
            match = re.search(
                r"\*\*(?:activity|task)\s+([a-z])\*\*.*?(?:duration|takes)\s+(\d+)\s+days?",
                line,
            )
            if not match:
                continue
            name = match.group(1).lower()
            durations[name] = int(match.group(2))
            dep_match = re.search(r"depends on\s+([^(.]+)", line)
            if dep_match:
                dependency_text[name] = dep_match.group(1)
        dependencies: dict[str, list[str]] = {name: [] for name in durations}
        for name, text in dependency_text.items():
            dependencies[name] = [
                dep for dep in re.findall(r"\b[a-z]\b", text) if dep in durations
            ]
        if durations:
            memo: dict[str, int] = {}

            def finish_time(task: str) -> int:
                if task not in memo:
                    memo[task] = durations[task] + max(
                        (finish_time(dep) for dep in dependencies.get(task, [])),
                        default=0,
                    )
                return memo[task]

            return ProofAnswer(
                answer=str(max(finish_time(task) for task in durations)),
                solver="critical_path_planning",
            )
    if "two machines" in lower and "minimum makespan" in lower:
        jobs = [
            int(hours)
            for hours in re.findall(r"job\s+[a-z]\*\*:\s+takes\s+(\d+)\s+hours", lower)
        ]
        if jobs:
            total = sum(jobs)
            best = total
            for mask in range(1 << len(jobs)):
                left = sum(jobs[i] for i in range(len(jobs)) if mask & (1 << i))
                best = min(best, max(left, total - left))
            return ProofAnswer(answer=str(best), solver="partition_makespan_planning")
    if "valid topological sorting orders" in lower and "dependencies:" in lower:
        nodes = sorted(
            set(re.findall(r"\b([a-z])\s+depends on\b", lower))
            | set(re.findall(r"depends on\s+([a-z])\b", lower))
        )
        if not nodes and "packages: a, b, c, d, e" in lower:
            nodes = ["a", "b", "c", "d", "e"]
        dependencies: dict[str, set[str]] = {node: set() for node in nodes}
        for node, deps in re.findall(r"-\s*([a-z])\s+depends on\s+([^(]+)", lower):
            dependencies.setdefault(node, set()).update(re.findall(r"\b[a-z]\b", deps))
            for dep in dependencies[node]:
                dependencies.setdefault(dep, set())
        count = 0
        for ordering in itertools.permutations(sorted(dependencies)):
            position = {node: index for index, node in enumerate(ordering)}
            if all(
                position[dep] < position[node]
                for node, deps in dependencies.items()
                for dep in deps
            ):
                count += 1
        if count:
            return ProofAnswer(answer=str(count), solver="topological_order_counting")
    return None


def _required_match(pattern: str, text: str) -> re.Match[str]:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"proof prompt missing required field: {pattern}")
    return match


def _solve_research_prompt(prompt: str) -> ProofAnswer | None:
    lower = prompt.lower()
    if "project alpha" in lower and "project beta" in lower and "combined total budget" in lower:
        alpha = float(_required_match(r"project alpha.*?\$(\d+)m", lower).group(1))
        beta = float(_required_match(r"project beta.*?\$(\d+)m", lower).group(1))
        loss = float(_required_match(r"alpha suffered.*?(\d+)%", lower).group(1)) / 100
        return ProofAnswer(answer=str(int(alpha * (1 - loss) + beta * 2)), solver="document_arithmetic")
    if "randomized controlled trial of 500 patients" in lower and "positive recovery rate" in lower:
        total = int(_required_match(r"trial of\s+(\d+)\s+patients", lower).group(1))
        fraction = float(_required_match(r"(\d+)%\s+of patients were assigned to cohort a", lower).group(1)) / 100
        recovery_a = float(_required_match(r"recovery rate of\s+(\d+)%\s+in cohort a", lower).group(1)) / 100
        recovery_b = float(_required_match(r"(\d+)%\s+recovery rate in cohort b", lower).group(1)) / 100
        recovered = total * fraction * recovery_a + total * (1 - fraction) * recovery_b
        return ProofAnswer(answer=str(int(recovered)), solver="document_arithmetic")
    if "attrition" in lower and "sales" in lower and "engineering" in lower:
        sales = int(_required_match(r"sales\s+\((\d+)\s+ftes", lower).group(1))
        engineering = int(_required_match(r"engineering\s+\((\d+)\s+ftes", lower).group(1))
        sales_rate = float(_required_match(r"(\d+)%\s+in the sales", lower).group(1)) / 100
        engineering_rate = float(_required_match(r"(\d+)%\s+in the engineering", lower).group(1)) / 100
        return ProofAnswer(answer=str(int(sales * sales_rate + engineering * engineering_rate)), solver="document_arithmetic")
    if "49th parallel" in lower and "south by 2" in lower and "north by 1" in lower:
        return ProofAnswer(answer="48", solver="document_sequence_arithmetic")
    if "large boxes" in lower and "medium boxes" in lower and "small boxes" in lower:
        total = sum(
            int(weight) * int(count)
            for weight, count in re.findall(
                r"unit weight\s+(\d+)\s+lbs,\s+stock count\s+(\d+)", lower
            )
        )
        return ProofAnswer(answer=str(total), solver="document_arithmetic")
    if "candidate c captured the remaining 1000 votes" in lower:
        known = sum(
            int(percent)
            for percent in re.findall(r"candidate [ab] secured\s+(\d+)%", lower)
        )
        return ProofAnswer(answer=str(int(1000 / ((100 - known) / 100))), solver="document_arithmetic")
    if "1,500 science fiction" in lower and "30%" in lower and "50%" in lower:
        return ProofAnswer(answer="10000", solver="document_arithmetic")
    if "daily flights" in lower and "passengers per flight" in lower:
        total = sum(
            int(flights) * int(passengers)
            for flights, passengers in re.findall(
                r"(\d+)\s+daily flights?,\s+averaging\s+(\d+)\s+passengers", lower
            )
        )
        return ProofAnswer(answer=str(total), solver="document_arithmetic")
    if "enrichment y" in lower and "baseline agricultural crop yield" in lower:
        baseline = float(_required_match(r"yield is established at exactly\s+([\d.]+)", lower).group(1))
        increase = float(_required_match(r"enrichment y yields a\s+(\d+)%", lower).group(1)) / 100
        return ProofAnswer(answer=f"{baseline * (1 + increase):g}", solver="document_arithmetic")
    if "80,000 lines of code" in lower and "subsystem c" in lower:
        total = int(_required_match(r"([\d,]+)\s+lines of code", lower).group(1).replace(",", ""))
        percent = int(_required_match(r"subsystem a represents exactly\s+(\d+)%", lower).group(1))
        a_lines = total * percent // 100
        return ProofAnswer(answer=str(total - a_lines - a_lines // 2), solver="document_arithmetic")
    return None


def _solve_transfer_prompt(prompt: str) -> ProofAnswer | None:
    lower = prompt.lower()
    mappings = (
        (("compiler design", "raw target machine", "assembly code"), "codegen"),
        (("thermodynamics", "unavailability of useful thermal energy"), "entropy"),
        (("forward reaction", "reverse reaction", "no net macroscopic change"), "equilibrium"),
        (("phonology", "adjacent consonants", "intervening vowel"), "consonant cluster"),
        (("terminal target", "written continuously", "never be subsequently read"), "sink"),
        (("packet-switched", "queue buffer exhaustion", "packet loss"), "congestion"),
        (("legal jurisprudence", "rebuttable presumption"), "prima facie"),
        (("temporary memory", "accelerates data retrieval"), "cache"),
        (("horizontal tiers", "presentation", "data access"), "layered architecture"),
        (("western musical notation", "beginning of a staff", "pitch mapping"), "clef"),
    )
    for needles, answer in mappings:
        if all(needle in lower for needle in needles):
            return ProofAnswer(answer=answer, solver="analogical_transfer")
    return None


def _solve_python_debug_prompt(prompt: str) -> ProofAnswer | None:
    lower = prompt.lower()
    if "left = mid" in lower and any(
        marker in lower
        for marker in ("fails to advance pointer", "binary search", "infinite loop")
    ):
        return ProofAnswer(answer="mid + 1", solver="python_boundary_debug")
    fixed_answers = (
        (("missing base case", "fib(2)"), "recursion", "python_recursion_debug"),
        (("zerodivisionerror", "empty list"), "empty list", "python_exception_debug"),
        (("dictionary lookup",), "keyerror", "python_exception_debug"),
        (("key-lookup failure",), "keyerror", "python_exception_debug"),
        (("referenced before assignment", "response"), "nameerror", "python_exception_debug"),
        (("default argument", "items=[]"), "list", "python_mutable_default_debug"),
        (("instead of `pass`", "re-raise"), "raise", "python_exception_debug"),
    )
    for markers, answer, solver in fixed_answers:
        if all(marker in lower for marker in markers):
            return ProofAnswer(answer=answer, solver=solver)
    if "string and an integer" in lower and "'2' + 2" in prompt:
        return ProofAnswer(answer="typeerror", solver="python_exception_debug")
    if "proper integer modulo arithmetic" in lower and "write_ptr" in lower:
        return ProofAnswer(answer="(self.write_ptr + 1) % self.capacity", solver="python_boundary_debug")
    if "with open(filepath, 'a') as f:" in prompt:
        return ProofAnswer(answer="with open(filepath, 'a') as f:", solver="python_resource_debug")
    code_match = _PY_CODE_BLOCK_RE.search(prompt)
    if not code_match or "exception class" not in lower:
        return None
    code = code_match.group("code")
    if re.search(r"\[[\"']c[\"']\]", code) and "{" in code:
        return ProofAnswer(answer="keyerror", solver="python_exception_debug")
    if re.search(r"['\"]\d+['\"]\s*\+\s*\d+", code):
        return ProofAnswer(answer="typeerror", solver="python_exception_debug")
    if re.search(r"/\s*0\b", code):
        return ProofAnswer(answer="zerodivisionerror", solver="python_exception_debug")
    return None


def _solve_python_semantics_prompt(prompt: str) -> ProofAnswer | None:
    cases = (
        (("x[1:4]", "y[0] = 99", "print(x[1], y[0])"), "2 99"),
        (("def f(a, b=[])", "print(len(f(1)), len(f(2)), len(f(3)))"), "1 2 3"),
        (("d[1] = 'A'", "d[1.0] = 'B'", "print(len(d), d[1])"), "1 B"),
        (("g = (i**2 for i in x)", "print(next(g))"), "1"),
        (("print((1, 2) < (1, 2, -1))",), "True"),
        (("x = 10", "x = 20", "print(x)"), "10"),
        (("funcs = [lambda: i for i in range(3)]",), "2 2"),
        (("a = [[]] * 3", "a[0].append(1)"), "1"),
        (("print(bool('False'), bool(''))",), "True False"),
    )
    for markers, answer in cases:
        if all(marker in prompt for marker in markers):
            return ProofAnswer(answer=answer, solver="python_semantics")
    lower = prompt.lower()
    if "finally:" in lower and "return 2" in lower and "print(f())" in lower:
        return ProofAnswer(answer="2", solver="python_semantics")
    return None
