"""The sampler that says where her time goes: what it counts, on a stack it is shown."""
from __future__ import annotations

import asyncio
import sys
import time

import pytest

from tools.where_her_time_goes import WhereHerTimeGoes, _hers

pytestmark = pytest.mark.unit


def test_a_function_of_hers_is_named_by_its_folder_and_anyone_else_s_is_not():
    from core.runtime import browser_calls_read_no_files

    assert _hers(browser_calls_read_no_files.install.__code__) == "core/runtime/browser_calls_read_no_files.py:install"
    assert _hers(asyncio.run.__code__) == ""


def test_a_stretch_with_the_loop_kept_busy_is_listed_with_who_kept_it_and_a_short_one_is_not():
    watch = WhereHerTimeGoes()

    def busy_at() -> None:
        watch._look(sys._getframe())

    busy_at()
    watch._busy_since = time.perf_counter() - 0.5              # as if it had been busy for half a second
    idle = type("Frame", (), {})()
    idle.f_code = type("Code", (), {"co_filename": "/lib/python3.12/selectors.py", "co_name": "select"})()
    idle.f_back = None
    watch._look(idle)
    assert len(watch.kept_busy) == 1 and watch.kept_busy[0][0] >= 0.5
    busy_at()
    watch._look(idle)
    assert len(watch.kept_busy) == 1                            # a moment's work is not a stretch
    assert "THE LOOP KEPT BUSY" in watch.said() and "2 looks" not in watch.said().splitlines()[0]
