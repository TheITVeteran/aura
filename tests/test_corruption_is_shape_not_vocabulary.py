"""Unknown is not corrupt.

`contains_corrupted_language` is a FATAL check: `_sanitize_telemetry_leakage`
returns None on it and the entire reply is thrown away, in every mode. It was
backed by /usr/share/dict/words — a word list with no modern technical
vocabulary — so a correct technical answer was destroyed:

    Your repo config has a stale webhook and the auth middleware is misordered.

Measured 2026-07-27. Meanwhile the actual steering-collapse output that this
check exists to stop passed cleanly, because every word in "Do product of
multiple exponent term simplify reflexion" IS in the dictionary.

So the check was failing in both directions at once, and the reason is that a
missing word means the list is old, not that the model broke. Corruption is a
property of how a token is BUILT — vowels, letter runs — which is why the
evidence is now shape rather than membership.

Both halves of this corpus matter: loosening it until nothing is corrupt
would pass the first class and fail the second.
"""

import pytest

from core.phases.dialogue_policy import contains_corrupted_language

pytestmark = pytest.mark.unit

REAL_ANSWERS = [
    "Your repo config has a stale webhook and the auth middleware is misordered.",
    "Postgres, Redis, and Kafka handle that.",
    "The GRPO run uses LoRA adapters on the MLX backend.",
    "Kubernetes reschedules the pod automatically when the node drains.",
    "I would containerize the microservice and put it behind a load balancer.",
    "I use async batching in the runtime to keep latency low.",
    "Honestly? I think preference is the right word for it.",
    "It's 1:24 AM, and I know that from my internal clock.",
]

CORRUPTED = [
    "asdkfj qwerty zxcvbn plorp",
    "xublcate ingediate evocer brolen",
    "The systm is wrking thlought the brolen evocer pathway",
]


@pytest.mark.parametrize("text", REAL_ANSWERS)
def test_a_real_answer_is_never_destroyed(text: str):
    assert not contains_corrupted_language(text), (
        "a usable answer would be annihilated by a fatal check"
    )


@pytest.mark.parametrize("text", CORRUPTED)
def test_genuine_corruption_is_still_caught(text: str):
    assert contains_corrupted_language(text)


def test_shape_beats_vocabulary():
    """The distinction the fix rests on, stated directly."""
    from core.phases.dialogue_policy import _looks_like_a_word

    for unknown_but_real in ("webhook", "kubernetes", "misordered", "middleware"):
        assert _looks_like_a_word(unknown_but_real)
    for unknown_and_garbage in ("asdkfj", "zxcvbn", "hjklzxcv"):
        assert not _looks_like_a_word(unknown_and_garbage)
