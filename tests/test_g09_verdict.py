"""G09 closes only when the primary, breadth and no-harm criteria all hold."""

from __future__ import annotations

from tools.g09_verdict import verdict


def line(gained: int, lost: int, p: float) -> dict:
    return {"gained": gained, "lost": lost, "two_sided_p": p}


def test_breadth_counts_held_out_domains_and_harm_counts_every_domain() -> None:
    every = {"pooled": line(30, 2, 1e-6), "domains": {"math": line(12, 0, 0.0005), "code": line(9, 1, 0.02),
                                                      "planning": line(9, 1, 0.02), "knowledge": line(0, 0, 1.0)}}
    held = {"pooled": line(28, 1, 1e-6), "domains": {"math": line(11, 0, 0.001), "code": line(8, 1, 0.04),
                                                     "planning": line(3, 0, 0.25)}}
    result = verdict(every, held, alpha=0.05, breadth_need=2, breadth_among=["code", "planning", "math"],
                     harm_alpha=0.05)
    assert result["closes"] and result["breadth"]["domains_with_gain"] == ["code", "math"]
    stricter = verdict(every, held, alpha=0.0167, breadth_need=2, breadth_among=["code", "planning", "math"],
                       harm_alpha=0.05)
    assert not stricter["closes"] and stricter["breadth"]["domains_with_gain"] == ["math"]
    every["domains"]["knowledge"] = line(0, 7, 0.0156)
    assert not verdict(every, held, alpha=0.05, breadth_need=2, breadth_among=["code", "planning", "math"],
                       harm_alpha=0.05)["no_harm"]["holds"]
