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


# ── The verdict may not depend on the host ────────────────────────────────
#
# Shape became the evidence, but the dictionary stayed in the arithmetic: the
# final two branches counted tokens "not in this host's word list". So the
# identical reply was corrupt on a machine without /usr/share/dict/words and
# clean on one with it, and a FATAL gate moved with the operating system
# rather than with the text. Found by a reviewer running the integration slice
# on Linux, where "mostly" was convicted.

#: Every one of these is an ordinary English word whose consonant tail made the
#: old terminal-wall rule call it malformed, because that rule alone counted
#: `y` as a consonant.
ADVERBS_THE_SHAPE_RULE_USED_TO_EAT = (
    "mostly", "exactly", "strictly", "firmly", "warmly", "friendly", "directly",
    "softly", "swiftly", "ghastly", "costly", "lastly", "vastly", "justly",
    "quickly", "monthly", "nightly", "rightly", "tightly", "lightly",
    "slightly", "brightly", "promptly", "abruptly", "correctly", "perfectly",
    "instantly", "constantly", "currently", "recently", "apparently",
)


@pytest.mark.parametrize("word", ADVERBS_THE_SHAPE_RULE_USED_TO_EAT)
def test_an_ordinary_adverb_is_not_malformed(word: str):
    from core.phases.dialogue_policy import _looks_like_a_word

    assert _looks_like_a_word(word), f"{word!r} is a word"


def test_a_reply_full_of_adverbs_survives():
    text = (
        "Mostly it works exactly as documented, and the retry currently fires "
        "promptly, so the lane recently recovered correctly and is running "
        "constantly again."
    )
    assert not contains_corrupted_language(text)


def test_the_verdict_does_not_depend_on_a_host_word_list(monkeypatch, tmp_path):
    """No filesystem read may change what this fatal gate decides."""
    import core.phases.dialogue_policy as mod

    assert not hasattr(mod, "_word_list"), (
        "a host word list is back in a gate that destroys answers"
    )

    def _no_filesystem(*args, **kwargs):
        raise AssertionError("corruption verdicts must not read the filesystem")

    monkeypatch.setattr(mod, "open", _no_filesystem, raising=False)
    for text in REAL_ANSWERS + list(CORRUPTED):
        contains_corrupted_language(text)


def test_the_false_positive_rate_on_real_english_stays_tiny():
    """The rule is fatal, so its error rate on real words is a bound, not a hope.

    /usr/share/dict/words is used here as a CORPUS to measure against — never
    as an oracle the runtime consults. The test skips where it is absent
    precisely because the runtime no longer needs it.
    """
    from pathlib import Path

    from core.phases.dialogue_policy import _looks_like_a_word

    corpus = Path("/usr/share/dict/words")
    if not corpus.exists():
        pytest.skip("no local word corpus to measure against")
    words = [
        w
        for w in (
            line.strip().lower()
            for line in corpus.read_text(errors="ignore").splitlines()
        )
        if w.isalpha() and 4 <= len(w) <= 24
    ]
    assert len(words) > 50_000, "corpus too small to bound anything"
    rejected = [w for w in words if not _looks_like_a_word(w)]
    rate = len(rejected) / len(words)
    # Was 1.96% before the `y` fix. The bound is deliberately close to the
    # measured 0.14% so a regression cannot hide inside a generous margin.
    assert rate < 0.005, f"{rate:.4%} of real English words called malformed"
