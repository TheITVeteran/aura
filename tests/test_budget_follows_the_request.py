"""93 seconds of finished research, reported as "Completed 0/0 steps".

Measured 2026-07-29. desktop_task declared a flat `timeout_seconds = 180.0`,
and the same research objective measured 98s, 100s, 156s, 161s and 176s across
one evening. The budget therefore sat inside its own spread and the outcome was
a coin flip. That night it lost:

    Task desktop_task timed out.  (187,388ms)
    ...it did not complete: Operation took too long. Completed 0/0 steps.

while the log alongside it read:

    Deep research complete: 1 loops, 1 queries, 3 sources, 93.5s

The research had succeeded. The governor cancelled the coroutine and the reply
told Bryan nothing had happened.

Reading is the cost, and the request says how much reading there is — so the
budget follows the request, the same way the source count does.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def Skill():  # noqa: N802 - reads as the class it provides
    from core.skills.desktop_task import DesktopTaskSkill

    return DesktopTaskSkill


THE_OBJECTIVE = (
    "Create a folder called Orca Demo in my Documents folder. Then find 3 "
    "recent articles about orcas online, read them, and write a synthesis "
    "with your own opinion into a PDF saved inside that Orca Demo folder."
)


class TestTheBudgetIsSizedToTheWork:
    def test_the_measured_objective_gets_more_than_it_took(self, Skill):
        """The slowest observed run of this exact objective was 176s, and one
        exceeded 187s."""
        budget = Skill.timeout_for({"objective": THE_OBJECTIVE})
        assert budget > 187.0, budget

    def test_more_sources_means_more_time(self, Skill):
        two = Skill.timeout_for({"objective": "find 2 articles about orcas and write a PDF"})
        four = Skill.timeout_for({"objective": "find 4 articles about orcas and write a PDF"})
        assert four > two

    def test_a_task_with_no_reading_keeps_the_floor(self, Skill):
        """A wallpaper change should not inherit a research budget."""
        budget = Skill.timeout_for(
            {"objective": "Find a picture of a rock online and set it as my wallpaper."}
        )
        assert budget == Skill.timeout_seconds

    def test_a_missing_objective_keeps_the_floor(self, Skill):
        assert Skill.timeout_for({}) == Skill.timeout_seconds
        assert Skill.timeout_for(None) == Skill.timeout_seconds

    def test_the_floor_is_never_lowered(self, Skill):
        """A skill sizing a request may ask for more time, never less — the
        declared number is a floor, not a suggestion."""
        for objective in ("", "make a folder", THE_OBJECTIVE):
            assert Skill.timeout_for({"objective": objective}) >= Skill.timeout_seconds


class TestTheEngineAsksTheSkill:
    def test_a_skill_that_can_size_a_request_is_consulted(self):
        import inspect

        from core.capability_engine import CapabilityEngine

        source = inspect.getsource(CapabilityEngine)
        assert 'getattr(skill_instance, "timeout_for", None)' in source

    def test_a_skill_that_declines_keeps_its_declared_number(self):
        """Nothing is required to implement timeout_for."""
        import inspect

        from core.capability_engine import CapabilityEngine

        source = inspect.getsource(CapabilityEngine)
        assert "if requested_budget > timeout_budget:" in source, (
            "a smaller request must not shrink the declared floor"
        )
