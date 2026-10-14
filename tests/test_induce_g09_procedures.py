"""G09's induction resumes from its logged proposals instead of decoding them again."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.induce_g09_procedures import logged_proposer


def _decoder(seen: list[list[str]], stop_after: int | None = None):
    def decode(requests: list[str], on_record) -> None:
        seen.append(list(requests))
        for position, request in enumerate(requests):
            if stop_after is not None and position == stop_after:
                raise KeyboardInterrupt
            on_record(position, {"public_text": f"code for {request}", "termination": "stop",
                                 "generated_tokens": 3, "seconds": 0.1})
    return decode


def test_a_reset_costs_only_the_proposals_in_flight(tmp_path: Path) -> None:
    requests = ["a", "b", "c", "d"]
    first: list[list[str]] = []
    with pytest.raises(KeyboardInterrupt):
        logged_proposer(tmp_path, _decoder(first, stop_after=2), thinking=True)(requests, 0)
    again: list[list[str]] = []
    texts = logged_proposer(tmp_path, _decoder(again), thinking=True)(requests, 0)
    assert again == [["c", "d"]]
    assert texts == [f"code for {r}" for r in requests]
    # Proposals logged in one thinking mode are not reused in the other.
    closed: list[list[str]] = []
    logged_proposer(tmp_path, _decoder(closed), thinking=False)(requests, 0)
    assert closed == [requests]


BENCH = Path("~/.aura/benchmarks").expanduser()


@pytest.mark.skipif(not (BENCH / "bbh-src/bbh/word_sorting.json").exists()
                    or not (BENCH / "g09/dm-math/mathematics_dataset-v1.0/interpolate/numbers__gcd.txt").exists(),
                    reason="the BBH and DeepMind mathematics files are not on this host")
def test_every_dm_module_and_bbh_task_splits_into_disjoint_halves_graded_by_its_own_matcher() -> None:
    from tools.induce_g09_procedures import grader_for, problems

    for kind in ("bbh_word_sorting", "bbh_date_understanding", "dm_numbers__gcd"):
        development, test = problems(kind, "development"), problems(kind, "test")
        assert development and test and not {r["key"] for r in development} & {r["key"] for r in test}
        assert abs(len(development) - len(test)) <= 1
        row = development[0]
        assert grader_for(kind)(str(row["truth"]), row["truth"])
    option = next(r for r in problems("bbh_date_understanding", "development") if r["truth"].startswith("("))
    assert grader_for("bbh_date_understanding")(option["truth"].strip("()"), option["truth"])
