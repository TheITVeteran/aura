"""Tests for the inner-light battery + activity source.

The decisive test: a conscious-like activity matrix occupies all four axes while
NO control does — and the two strongest controls (time-shuffle, phase-randomise)
each reach exactly 3/4, proving the four measures are not redundant yet neither
reproduces the whole signature.
"""
from __future__ import annotations

import time

import numpy as np

from core.consciousness.inner_light import activity as act
from core.consciousness.inner_light import battery as bat


def _pink(T, rng, beta=1.4):
    w = rng.standard_normal(T)
    F = np.fft.rfft(w)
    f = np.arange(len(F), dtype=float)
    f[0] = 1.0
    x = np.fft.irfft(F / f ** (beta / 2.0), n=T)
    return (x - x.mean()) / x.std()


def _rich(seed=0, nmod=3, per=2, T=2000, amp=4.0, dur=2, rate=22, noise=0.35, frac=0.7):
    rng = np.random.default_rng(seed)
    n = nmod * per
    lats = [_pink(T, rng) for _ in range(nmod)]
    chans = [lats[m] + noise * rng.standard_normal(T) for m in range(nmod) for _ in range(per)]
    M = np.stack(chans)
    k = max(1, int(round(frac * n)))
    for t in rng.choice(T - dur, size=T // rate, replace=False):
        who = rng.choice(n, size=k, replace=False)
        M[np.ix_(who, np.arange(t, t + dur))] += amp
    return M


# ── the decisive discrimination ──────────────────────────────────────────────

def test_conscious_like_matrix_is_the_only_one_at_four_of_four():
    res = bat.run_on_matrix(_rich())
    assert res.real_axes == 4
    assert res.verdict == "signature_present"
    assert res.discriminating is True
    assert res.best_control_axes < 4          # no control reproduces the whole signature
    assert "NOT a claim that Aura is conscious" in res.caveat


def test_strongest_controls_reach_three_not_four():
    res = bat.run_on_matrix(_rich())
    # time-shuffle keeps everything but temporal structure → loses only criticality
    ts = res.controls["time_shuffle"]
    assert ts["axes"] == 3
    assert ts["membership"]["criticality"] is False
    # phase-randomise keeps the spectrum but destroys non-linearity → loses only ignition
    pr = res.controls["phase_randomize"]
    assert pr["axes"] == 3
    assert pr["membership"]["ignition"] is False


def test_white_noise_is_not_conscious_like():
    from core.consciousness.inner_light import controls as c
    res = bat.run_on_matrix(c.white_noise((6, 2000)))
    assert res.real_axes < 4
    assert res.verdict in {"signature_absent", "signature_partial"}
    assert res.real_membership["integrated_complexity"] is False


def test_ordered_is_not_conscious_like():
    from core.consciousness.inner_light import controls as c
    res = bat.run_on_matrix(c.ordered((6, 2000), period=8))
    assert res.real_axes < 4
    assert res.real_membership["differentiation"] is False


def test_insufficient_matrix_is_honest():
    res = bat.run_on_matrix(np.zeros((1, 4)))
    assert res.verdict == "insufficient_data"
    assert res.score == 0.0


# ── activity source ──────────────────────────────────────────────────────────

def test_build_activity_matrix_from_events():
    rng = np.random.default_rng(0)
    now = time.time()
    channels = ["affect", "memory", "will", "world_model"]
    events = []
    for i in range(400):
        events.append((now + i * 0.1, channels[i % len(channels)]))
    sample = act.build_activity_matrix(events, n_bins=64)
    assert sample.sufficient
    assert sample.matrix.shape[0] == 4
    assert sample.matrix.shape[1] == 64
    assert sample.n_events == 400


def test_activity_insufficient_cases():
    now = time.time()
    too_few_channels = [(now + i, "solver") for i in range(100)]
    s1 = act.build_activity_matrix(too_few_channels)
    assert not s1.sufficient and "channel" in s1.reason

    too_few_events = [(now + i, c) for i, c in enumerate(["a", "b", "c"])]
    s2 = act.build_activity_matrix(too_few_events)
    assert not s2.sufficient


def test_run_live_insufficient_when_bus_empty():
    from core.runtime.consequence_bus import ConsequenceBus
    ConsequenceBus.reset()
    res = bat.run_live(bus=ConsequenceBus.get())
    assert res.verdict == "insufficient_data"
    assert "insufficient live activity" in res.caveat
    ConsequenceBus.reset()
