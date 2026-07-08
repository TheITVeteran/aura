"""Behavior tests for the idle self-play flywheel.

The soundness-critical contracts under test with the REAL machinery:
battery generation, exact grading, and the canonical preference store all
run for real — only the LLM route and the idle gate are faked. If these
pass, a practice burst produces sound DPO contrast pairs and nothing else.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.learning.heldout_battery import BatterySpec, generate_battery
from core.learning.selfplay_flywheel import (
    EVAL_SEED_FLOOR,
    SEED_SPAN,
    SelfPlayFlywheel,
    reset_selfplay_flywheel_for_test,
)
from core.learning.verifiable_preference_harness import VerifiablePreferenceHarness

pytestmark = pytest.mark.unit


class ContrastRouter:
    """Answers each task correctly once, then wrongly — a guaranteed contrast."""

    def __init__(self):
        self.calls: list[dict] = []
        self._by_prompt: dict[str, int] = {}
        self._answers: dict[str, str] = {}

    def prime(self, tasks) -> None:
        for task in tasks:
            self._answers[task.prompt] = task.answer

    async def think(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        nth = self._by_prompt.get(prompt, 0)
        self._by_prompt[prompt] = nth + 1
        truth = self._answers.get(prompt, "0")
        return f"Answer: {truth}" if nth == 0 else "Answer: definitely-wrong-999983"


@pytest.fixture
def flywheel(tmp_path, monkeypatch, service_container):
    reset_selfplay_flywheel_for_test()
    store = tmp_path / "prefs.jsonl"
    harness = VerifiablePreferenceHarness(store_path=store)
    monkeypatch.setattr(
        "core.learning.verifiable_preference_harness.get_verifiable_preference_harness",
        lambda: harness,
    )
    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_allowed",
        lambda *_a, **_k: True,
    )
    state_path = tmp_path / "flywheel_state.json"
    monkeypatch.setattr(SelfPlayFlywheel, "_state_path", lambda self: state_path)

    fw = SelfPlayFlywheel()
    fw.burst_tasks = 3
    fw.attempts_per_task = 2
    fw._flywheel_test_ctx = {  # handles for tests, not runtime
        "store": store,
        "harness": harness,
        "state_path": state_path,
        "container": service_container,
    }
    yield fw
    reset_selfplay_flywheel_for_test()


def _register_router(service_container, router) -> None:
    service_container.register_instance("llm_router", router, required=False)


async def test_burst_produces_sound_contrast_pairs(flywheel):
    ctx = flywheel._flywheel_test_ctx
    router = ContrastRouter()
    router.prime(generate_battery(BatterySpec(seed=3, size=flywheel.burst_tasks)))
    _register_router(ctx["container"], router)

    stats = await flywheel._burst()

    assert stats["attempts"] == flywheel.burst_tasks * flywheel.attempts_per_task
    assert stats["correct"] == flywheel.burst_tasks  # one win per task by design
    assert stats["pairs"] >= flywheel.burst_tasks  # every task had win+loss contrast
    rows = [json.loads(l) for l in ctx["store"].read_text().splitlines() if l.strip()]
    assert len(rows) == stats["pairs"]
    for row in rows:
        assert row["chosen"] != row["rejected"]
        assert row["chosen"].startswith("Answer:")
    # every route call was explicitly background with a named origin
    assert all(c["is_background"] is True for c in router.calls)
    assert all(c["origin"] == "selfplay_flywheel" for c in router.calls)


async def test_burst_state_persists_and_cursor_advances(flywheel):
    ctx = flywheel._flywheel_test_ctx
    router = ContrastRouter()
    router.prime(generate_battery(BatterySpec(seed=3, size=flywheel.burst_tasks)))
    _register_router(ctx["container"], router)

    await flywheel._burst()
    state = json.loads(ctx["state_path"].read_text())
    assert state["seed_cursor"] == 1
    assert state["bursts"] == 1
    assert state["total_pairs"] >= flywheel.burst_tasks
    assert 0.0 <= state["correct_rate_ema"] <= 1.0


def test_seed_rotation_never_touches_eval_floor():
    for cursor in (0, 1, 996, 997, 998, 5000, 10_000_000):
        seed = 3 + (cursor % SEED_SPAN)
        assert 3 <= seed < EVAL_SEED_FLOOR


async def test_burst_yields_mid_flight_when_conversation_starts(flywheel, monkeypatch):
    ctx = flywheel._flywheel_test_ctx
    router = ContrastRouter()
    router.prime(generate_battery(BatterySpec(seed=3, size=flywheel.burst_tasks)))
    _register_router(ctx["container"], router)

    gate = {"calls": 0}

    def flip_after_two(*_a, **_k):
        gate["calls"] += 1
        return gate["calls"] <= 2

    monkeypatch.setattr(flywheel, "_still_allowed", flip_after_two)
    stats = await flywheel._burst()
    assert stats["aborted"] is True
    assert stats["attempts"] == 2  # stopped as soon as the gate flipped


async def test_empty_responses_are_not_training_signal(flywheel):
    ctx = flywheel._flywheel_test_ctx

    class SilentRouter:
        async def think(self, prompt: str, **kwargs):
            return "   "

    _register_router(ctx["container"], SilentRouter())
    stats = await flywheel._burst()
    assert stats["attempts"] == 0
    assert stats["pairs"] == 0
    assert not ctx["store"].exists() or not ctx["store"].read_text().strip()


async def test_no_router_skips_without_crashing(flywheel):
    stats = await flywheel._burst()
    assert stats == {"skipped": "no_llm_router"}


async def test_disabled_by_env(monkeypatch, service_container):
    reset_selfplay_flywheel_for_test()
    monkeypatch.setenv("AURA_SELFPLAY_FLYWHEEL", "0")
    fw = SelfPlayFlywheel()
    await fw.start()
    assert fw._task is None and fw._active is False
    reset_selfplay_flywheel_for_test()


async def test_stop_cancels_cleanly(flywheel, monkeypatch):
    monkeypatch.setenv("AURA_SELFPLAY_FLYWHEEL", "1")
    await flywheel.start()
    assert flywheel._task is not None
    await flywheel.stop()
    assert flywheel._task is None and flywheel._active is False


async def test_correct_rate_ema_tracks_across_bursts(flywheel):
    ctx = flywheel._flywheel_test_ctx
    router = ContrastRouter()
    # prime both burst seeds (cursor 0 → seed 3, cursor 1 → seed 4)
    for seed in (3, 4):
        router.prime(generate_battery(BatterySpec(seed=seed, size=flywheel.burst_tasks)))
    _register_router(ctx["container"], router)

    await flywheel._burst()
    first = json.loads(ctx["state_path"].read_text())["correct_rate_ema"]
    assert first == 0.5  # 1 win of 2 attempts per task
    await flywheel._burst()
    second = json.loads(ctx["state_path"].read_text())["correct_rate_ema"]
    assert second == pytest.approx(0.8 * first + 0.2 * 0.5)


def test_status_reports_persisted_truth(flywheel):
    status = flywheel.get_status()
    assert status["service"] == "selfplay_flywheel"
    assert status["bursts"] == 0


# ── harvest tool discipline ───────────────────────────────────────────────────

def test_harvest_refuses_eval_floor_seeds():
    from types import SimpleNamespace

    from tools.selfplay_harvest import harvest

    args = SimpleNamespace(
        model="unused", store="unused", tasks=4, attempts=2,
        seed_start=EVAL_SEED_FLOOR, temp=0.8, max_tokens=64,
    )
    with pytest.raises(SystemExit, match="sealed held-out"):
        harvest(args)
