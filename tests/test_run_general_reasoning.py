"""The general reasoning harness runs every stage on a fake decoder and scores three arms on the same requests."""

from __future__ import annotations

import json
import sys
import types

import pytest

from core.sandbox.untrusted_python import available_boundary

pytestmark = pytest.mark.skipif(not available_boundary(), reason="no OS sandbox on this host")

TASKS = [
    {"id": "mmlu_pro:1", "request": "Q1 options", "truth": "A", "group": "math"},
    {"id": "mmlu_pro:2", "request": "Q2 options", "truth": "A", "group": "math"},
    {"id": "mmlu_pro:3", "request": "Q3 options", "truth": "B", "group": "law"},
    {"id": "mmlu_pro:4", "request": "Q4 seats", "truth": "C", "group": "law"},
]


def fake_decode(model, tokenizer, conversations, *, max_tokens, width, on_record, thinking=True, sampler=None,
                seed=None):
    for index, conversation in enumerate(conversations):
        text = conversation[-1]["content"]
        given = conversation[0]["role"] == "system"
        if text.startswith("Two answers") and "Q4 seats" in text:
            reply = "FIRST: Ann sits first\nSECOND: Cy sits first\nQUESTION: Who sits first?"
        elif text.startswith("Two answers"):
            reply = "FIRST: the sum is 3\nSECOND: the sum is 4\nQUESTION: What is 1 + 2?"
        elif text.startswith("Problem:"):
            reply = "```python\nprint('SOLUTIONS: 1')\nprint('ANSWER: (C)')\n```"
        elif "TASK:" in text:
            reply = "```python\nprint(1 + 2)\n```"
        elif text.startswith("Passages:"):
            reply = "NEITHER"
        elif text == "Q1 options":
            reply = "the answer is (A)"
        elif text == "Q4 seats":
            reply = "the answer is (C)" if given or sampler is not None else "the answer is (A)"
        elif text == "Q2 options":
            reply = "the answer is (A)" if given else ("the answer is (B)" if sampler is None else "the answer is (A)")
        else:
            reply = "the answer is (B)" if sampler is None else "the answer is (C)"
        on_record(index, {"public_text": reply, "generated_tokens": 100, "termination": "stop"})


def test_every_stage_runs_and_the_arms_are_scored_on_the_same_requests(tmp_path, monkeypatch) -> None:
    from tools import run_general_reasoning as harness

    monkeypatch.setitem(harness.LOADERS, "mmlu_pro", lambda part: TASKS)
    monkeypatch.setattr("tools.g12_batched.decode_stream", fake_decode)
    monkeypatch.setitem(sys.modules, "mlx_lm", types.SimpleNamespace(load=lambda path: (None, types.SimpleNamespace(
        encode=lambda text, add_special_tokens=False: list(range(len(text.split())))))))
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", types.SimpleNamespace(make_sampler=lambda **kw: object()))

    class Lane:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr("core.runtime.model_lane_control.standalone_model_lane", Lane)
    monkeypatch.setattr("core.knowledge.corpus_grounding.corpus_grounding_for",
                        lambda q: types.SimpleNamespace(passages=()))
    monkeypatch.setattr("core.knowledge.passages_about.passages_about", lambda q, **k: [])
    model = tmp_path / "model"
    model.mkdir()
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["run_general_reasoning.py", "--benchmark", "mmlu_pro", "--part", "development",
                                      "--count", "4", "--seed", "5", "--output", str(out), "--model", str(model)])
    assert harness.main() == 0
    report = json.loads((out / "summary.json").read_text())
    assert report["requests"] == 4
    mechanism = {k: json.loads((out / "rows" / "mechanism" / f"mmlu_pro_{k}.json").read_text()) for k in (1, 2, 3, 4)}
    # Q4: greedy A, sampled C; no numbers to compute, the engine's unique solution is C, the sampled side.
    assert mechanism[4]["outcome"] == "settled:solved as constraints by her engine" and mechanism[4]["answer"] == "C"
    assert mechanism[1]["outcome"] == "agreed" and mechanism[1]["answer_text_correct"]
    # Q2: greedy B, sampled A; the crux's numbers are settled in the sandbox (1 + 2 = 3) and her last answer is A.
    assert mechanism[2]["outcome"] == "settled:computed in the sandbox" and mechanism[2]["answer"] == "A"
    final = json.loads((out / "rows" / "final" / "mmlu_pro_2.json").read_text())
    assert final["public_text"] == "the answer is (A)"
    assert report["accuracy"]["ordinary"]["correct"] == 2 and report["accuracy"]["mechanism_a"]["correct"] == 4
    assert report["mechanism_a_vs_ordinary"]["mechanism_a_only"] == 2
    # The run is resumed, not repeated: a second call decodes nothing.
    calls: list[int] = []
    monkeypatch.setattr("tools.g12_batched.decode_stream", lambda *a, **k: calls.append(1))
    assert harness.main() == 0 and not calls
