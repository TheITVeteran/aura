from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import tools.run_semantic_output_transfer_canary as canary


def test_generate_uses_canonical_sequence_decode_not_stream_text(monkeypatch) -> None:
    responses = (
        SimpleNamespace(text="not-a-prefix", token=1, finish_reason=None),
        SimpleNamespace(text="not-a-suffix", token=2, finish_reason=None),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm",
        SimpleNamespace(stream_generate=lambda *_args, **_kwargs: iter(responses)),
    )

    class Tokenizer:
        @staticmethod
        def decode(tokens):
            return "FINAL_ANSWER: {\"value\":1}" if tokens == [1, 2] else "FINAL_ANSWER: {"

    text, tokens = canary._generate(object(), Tokenizer(), [9], 8)
    assert tokens == [1, 2]
    assert text == 'FINAL_ANSWER: {"value":1}'


def test_main_holds_non_evicting_lane_for_complete_model_lifetime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    out = tmp_path / "report.json"
    args = SimpleNamespace(model=str(model), out=out)
    observed: list[tuple[str, object]] = []

    monkeypatch.setattr(
        canary,
        "_parser",
        lambda: SimpleNamespace(parse_args=lambda: args),
    )

    @contextmanager
    def lane(**kwargs):
        observed.append(("enter", kwargs))
        yield
        observed.append(("exit", None))

    monkeypatch.setattr(canary, "standalone_model_lane", lane)
    monkeypatch.setattr(
        canary,
        "_run_admitted",
        lambda parsed, resolved: observed.append(("run", (parsed, resolved))) or 0,
    )

    assert canary.main() == 0
    assert [name for name, _payload in observed] == ["enter", "run", "exit"]
    kwargs = observed[0][1]
    assert isinstance(kwargs, dict)
    assert kwargs["allow_owner_eviction"] is False
    assert kwargs["preemptible"] is False
    assert observed[1][1] == (args, model.resolve())
