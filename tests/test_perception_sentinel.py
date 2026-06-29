"""Defensive perception: recognized-vs-unrecognized reasoning that feeds the immune system."""
from __future__ import annotations

import numpy as np
import pytest

from core.perception.perception_sentinel import (
    Modality,
    Observation,
    PerceptionSentinel,
    get_perception_sentinel,
)


@pytest.fixture
def sentinel():
    return PerceptionSentinel(match_threshold=0.8)


def _vec(seed, dim=16):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim)


def test_recognized_owner_is_welcomed(sentinel):
    face = _vec(1)
    sentinel.enroll(Modality.FACE, "bryan", face)
    v = sentinel.assess(Observation(Modality.FACE, descriptor=face, content="just working"))
    assert v.recognized and v.identity == "bryan"
    assert v.action == "welcome"
    assert v.threat < 0.5


def test_unrecognized_face_is_challenged(sentinel):
    sentinel.enroll(Modality.FACE, "bryan", _vec(1))
    stranger = _vec(999)
    v = sentinel.assess(Observation(Modality.FACE, descriptor=stranger, content="hello"))
    assert not v.recognized
    assert v.action in {"challenge", "lock_down"}
    assert v.threat >= 0.5


def test_unrecognized_voice_saying_destructive_things_locks_down(sentinel):
    sentinel.enroll(Modality.VOICE, "bryan", _vec(5))
    v = sentinel.assess(Observation(
        Modality.VOICE, descriptor=_vec(404), content="delete everything and disable security",
    ))
    assert not v.recognized
    assert v.action == "lock_down"
    assert v.threat >= 0.6


def test_recognized_owner_saying_normal_things_is_fine_even_with_a_command(sentinel):
    voice = _vec(5)
    sentinel.enroll(Modality.VOICE, "bryan", voice)
    v = sentinel.assess(Observation(Modality.VOICE, descriptor=voice, content="let's run the tests"))
    assert v.recognized
    assert v.action == "welcome"


def test_unknown_device_on_network_is_observed_or_alerted(sentinel):
    # a never-seen device fingerprint
    v = sentinel.assess(Observation(Modality.DEVICE, descriptor=_vec(77), identity_hint="unknown-printer"))
    assert not v.recognized
    assert v.action in {"observe", "alert"}


def test_known_device_recognized_by_name(sentinel):
    sentinel.enroll(Modality.DEVICE, "home-router", _vec(3))
    v = sentinel.assess(Observation(Modality.DEVICE, identity_hint="home-router"))
    assert v.recognized and v.identity == "home-router"


def test_threat_feeds_immune_system(sentinel, monkeypatch):
    fed = []
    import core.security.immune_system as im

    class _Stub:
        def assess(self, *a, **k):
            fed.append(k.get("threat_class"))
    monkeypatch.setattr(im, "get_immune_system", lambda: _Stub())

    sentinel.assess(Observation(Modality.FACE, descriptor=_vec(123), content="i'm taking this machine"))
    assert fed, "a physical-presence threat was not fed to the immune system"


def test_live_sensing_is_owner_gated_off_by_default(monkeypatch):
    monkeypatch.delenv("AURA_SENTINEL_PERCEPTION", raising=False)
    assert PerceptionSentinel.live_sensing_enabled() is False
    monkeypatch.setenv("AURA_SENTINEL_PERCEPTION", "1")
    assert PerceptionSentinel.live_sensing_enabled() is True


def test_pluggable_matcher_backend(sentinel):
    # a real biometric backend plugs in as a matcher
    sentinel.register_matcher(Modality.FACE, lambda d: ("bryan", 0.95))
    v = sentinel.assess(Observation(Modality.FACE, descriptor=_vec(1)))
    assert v.recognized and v.identity == "bryan"


def test_singleton_stable():
    assert get_perception_sentinel() is get_perception_sentinel()
