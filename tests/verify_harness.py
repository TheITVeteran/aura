"""Verify Evaluation Harness."""
import asyncio
import logging
from pathlib import Path

from core.self_modification.evaluation_harness import EvaluationHarness
from core.self_modification.code_repair import SandboxTester, CodeFix


class DeterministicBrain:
    async def think(self, prompt, **_kwargs):
        class Thought:
            content = (
                "import sys\n"
                "sys.path.insert(0, '.')\n"
                "import test_target\n"
                "assert test_target.VALUE == 1\n"
            )
        return Thought()


def test_harness(tmp_path: Path):
    asyncio.run(_run_harness_check(tmp_path))


async def _run_harness_check(tmp_path: Path):
    logging.getLogger(__name__).info("Testing Evaluation Harness")
    brain = DeterministicBrain()
    tester = SandboxTester(code_base_path=str(tmp_path))
    harness = EvaluationHarness(brain, tester, code_base_path=str(tmp_path))

    target_path = tmp_path / "test_target.py"
    target_path.write_text("VALUE = 0\n", encoding="utf-8")
    
    fix = CodeFix(
        target_file="test_target.py",
        target_line=1,
        original_code="VALUE = 0",
        fixed_code="VALUE = 1",
        explanation="Update the constant so the generated probe passes.",
        hypothesis="The probe should fail on VALUE=0 and pass after VALUE=1.",
        confidence="high"
    )

    probe = await harness.create_weakness_probe("test_target.py", {"bug": "constant too low"})
    assert probe is not None
    assert "assert test_target.VALUE == 1" in probe

    valid, message = await harness.evaluate_fix(fix, {"bug": "constant too low"})
    assert valid, message

if __name__ == "__main__":
    asyncio.run(_run_harness_check(Path.cwd()))
