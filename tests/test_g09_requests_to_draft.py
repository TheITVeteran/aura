"""Attempt 3's rule: a domain's draws are divided evenly among its kinds that keep a procedure."""

from __future__ import annotations

from tools.g09_requests_to_draft import spread


def test_a_total_is_divided_evenly_with_the_remainder_by_seed() -> None:
    shares = spread(300, ["c", "a", "b", "d", "e", "f", "g"], 20261014)
    assert sum(shares.values()) == 300 and set(shares.values()) == {42, 43}
    assert sum(v == 43 for v in shares.values()) == 300 % 7
    assert shares == spread(300, ["g", "f", "e", "d", "c", "b", "a"], 20261014)
    assert spread(300, [], 1) == {} and spread(300, ["only"], 1) == {"only": 300}
