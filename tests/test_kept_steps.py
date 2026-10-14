"""A kept kind is a step: a program in her sandbox asks it for the answer to a sub-problem."""

from __future__ import annotations

import json
import time

import pytest

from core.learning.procedures_from_solved_examples import (
    ProcedureBook,
    Signature,
    SolvedExample,
    dumps,
    induce,
    kept_steps_from,
    run_procedure,
)
from core.sandbox.untrusted_python import available_boundary

pytestmark = pytest.mark.skipif(not available_boundary(), reason="no OS sandbox on this host")

PERCENT = ("def solve(problem):\n    a, b = problem.split('Calculate ')[1].rstrip('.').split('*')\n"
           "    value = float(a) * float(b)\n    return str(int(value)) if value == int(value) else str(value)\n")
# A discount is a percent of the price taken from the price: one kept kind, then a subtraction.
DISCOUNT = ("def solve(problem):\n    import re\n"
            "    price, rate = re.findall(r'[0-9.]+', problem)[:2]\n"
            "    off = float(kept_answer('percent', f'Calculate {float(rate) / 100}*{price}.'))\n"
            "    value = float(price) - off\n    return str(int(value)) if value == int(value) else str(value)\n")
LOOP = "def solve(problem):\n    return kept_answer('loop', problem)\n"


def _book_file(tmp_path, procedures: dict[str, list[str]]):
    book = ProcedureBook({k: Signature({k: 1.0}, 0.99) for k in procedures}, procedures,
                         {k: 40 for k in procedures},
                         examples={"percent": [{"problem": "Calculate 0.2*50.", "answer": "10"}]})
    path = tmp_path / "book.json"
    path.write_text(dumps(book.to_json()), encoding="utf-8")
    return path


def test_a_procedure_answers_through_another_kept_kind(tmp_path) -> None:
    steps = kept_steps_from(_book_file(tmp_path, {"percent": [PERCENT]}))
    assert steps is not None and "'percent'" in steps.listing and "Calculate 0.2*50." in steps.listing
    assert run_procedure(DISCOUNT, "A coat costs 80 dollars with 25 percent off.", steps=steps) == ("60", "")
    returned, error = run_procedure(DISCOUNT, "A coat costs 80 dollars with 25 percent off.")
    assert returned is None and "kept_answer" in error


def test_a_kind_that_asks_for_its_own_answer_is_refused(tmp_path) -> None:
    steps = kept_steps_from(_book_file(tmp_path, {"loop": [LOOP]}))
    returned, error = run_procedure("def solve(problem):\n    return kept_answer('loop', problem)\n", "x",
                                    steps=steps)
    assert returned is None and "asks for its own answer" in error


def test_her_book_runs_a_kind_built_on_another(tmp_path, monkeypatch) -> None:
    from core.learning import procedures_from_solved_examples as module

    monkeypatch.setattr("core.utils.paths.aura_data_dir", lambda: tmp_path)
    module._kept_cache.clear()
    path = tmp_path / module.KEPT_BOOK_RELATIVE
    path.parent.mkdir(parents=True)
    path.write_text(_book_file(tmp_path, {"percent": [PERCENT], "discount": [DISCOUNT]}).read_text())
    book = module.kept_procedure_book()
    assert book is not None and book.steps is not None and module.her_kept_steps() is book.steps
    receipt = book.answer_restated("discount", "A lamp costs 40 dollars with 10 percent off.")
    assert receipt["answer"] == "36", receipt
    module._kept_cache.clear()


def test_induction_shows_the_kept_kinds_and_keeps_a_procedure_that_calls_one(tmp_path) -> None:
    steps = kept_steps_from(_book_file(tmp_path, {"percent": [PERCENT]}))
    examples = [SolvedExample(f"d{i}", f"A hat costs {20 + i} dollars with 50 percent off.",
                              str((20 + i) / 2).removesuffix(".0")) for i in range(12)]
    asked: list[str] = []

    def propose(requests: list[str]) -> list[str]:
        asked.extend(requests)
        return [f"```python\n{DISCOUNT}```"] * len(requests)

    families = induce({"discount": examples}, propose, shots=3, lines=1, rounds=1, steps=steps)
    assert "kept_answer" in asked[0] and "'percent'" in asked[0]
    assert families["discount"].admitted


@pytest.mark.asyncio
async def test_her_reasoning_organ_calls_a_kept_kind_from_its_program(tmp_path) -> None:
    from core.brain.executable_reasoning import derive_executable_candidate
    from core.brain.symbolic_sandbox import SymbolicSandbox

    steps = kept_steps_from(_book_file(tmp_path, {"percent": [PERCENT]}))
    prompts: list[str] = []

    async def generate(prompt: str, temperature: float) -> str:
        prompts.append(prompt)
        return "```python\nprint(kept_answer('percent', 'Calculate 0.15*340.'))\n```"

    result = await derive_executable_candidate(
        objective="What is 15% of 340?", task_type="math", generate=generate, sandbox=SymbolicSandbox(),
        deadline=time.monotonic() + 60, explicitly_enabled=True, kept_steps=steps)
    assert "kept_answer" in prompts[0] and "'percent'" in prompts[0]
    assert result.succeeded and str(result.candidate).strip() == "51"
    assert result.receipt["attempts"][-1]["kept_steps_called"] is True


def test_a_book_with_no_procedure_offers_no_steps(tmp_path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(json.dumps(ProcedureBook({}, {}).to_json()), encoding="utf-8")
    assert kept_steps_from(path) is None
