from __future__ import annotations

import pytest

from scripts import fetch_models


def test_reasoning_solver_alias_overrides_solver_lane():
    plan = fetch_models._plan_with_reasoning_solver("r1-qwen32b")
    assert plan is not None
    assert plan["solver"] == "DeepSeek-R1-Distill-Qwen-32B-4bit"


def test_reasoning_solver_alias_supports_qwq():
    plan = fetch_models._plan_with_reasoning_solver("qwq32b")
    assert plan is not None
    assert plan["solver"] == "QwQ-32B-4bit"


def test_reasoning_solver_unknown_alias_fails_closed():
    with pytest.raises(SystemExit):
        fetch_models._plan_with_reasoning_solver("not-a-model")
