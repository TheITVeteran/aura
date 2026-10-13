"""The words that tell the model who to be are counted, and the count only goes down.

Bryan, 13 October 2026: get rid of any part that is prompting and build it as
real systems engineering. tools/where_she_is_still_prompted.py is the map.
"""
from __future__ import annotations

import json

from tools.where_she_is_still_prompted import BASELINE, prompting_in, the_map


def test_a_string_that_instructs_the_model_is_counted_and_content_is_not():
    source = '''
PROMPT = "You are Aura Luna. You speak in short, direct sentences. Never say you can help with that, ever."
WHAT_WAS_SAID = "The limpet came back to its scar on the rock after grazing through the whole of the night tide."
LABEL = "You are here"

def f():
    """You are reading a docstring. Never count a docstring, however long it runs on for."""
'''
    found = prompting_in(source)
    assert [size for _line, size in found] == [len("You are Aura Luna. You speak in short, direct sentences. Never say you can help with that, ever.")]


def test_no_file_instructs_the_model_in_more_words_than_the_baseline_holds():
    held = json.loads(BASELINE.read_text())["files"]
    grown = {
        name: (row["characters"], held.get(name, {}).get("characters", 0))
        for name, row in the_map().items()
        if row["characters"] > held.get(name, {}).get("characters", 0)
    }
    assert not grown, f"more prompting than the baseline holds: {grown}"
