"""Ratchet: skill trigger patterns must never dispatch tools from plain talk.

Class discovered live (July 8): image_gen's "paint (?:me )?(?:an? )?" had an
all-optional tail — a memory request mentioning "the paint color I chose"
dispatched the diffusion model mid-conversation and crashed CRITICAL. The
audit then found four siblings: noun-phrase "the search for ..." dispatched
web_search, "the news about ... made me sad" dispatched web_search, a bare
"clock" mention dispatched the clock skill via the generic name layer, and
the intent normalizer rewrote "really" → "recall" so every casual sentence
containing "really" routed toward memory_ops.

Contract, both directions:
  * BENIGN — ordinary conversation (feelings, small talk, mentions of
    tool-adjacent nouns in non-imperative frames) dispatches NOTHING;
  * POSITIVE — real requests keep dispatching, so tightening can never
    quietly lobotomize the tool surface.

Growing BENIGN is always welcome; removing from it needs the same scrutiny
as loosening a security gate. Reminiscence ("I remember my first bike") is
deliberately NOT in BENIGN: autobiographical memory statements routing to
memory_ops is intended behavior with its own downstream carve-outs.
"""
from __future__ import annotations

import pytest

from core.capability_engine import CapabilityEngine
from core.utils.intent_normalization import normalize_memory_intent_text

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def engine():
    return CapabilityEngine()


BENIGN_CONVERSATION = [
    # the live-caught shapes
    "Keep this in mind for later: the paint color I chose for the bedroom is sage.",
    "the search for a new apartment is exhausting",
    "time really flies when things get busy",
    "the news about the local library closing made me sad",
    "the clock in the kitchen is five minutes fast",
    # common-word skill names in ordinary frames
    "we should speak more often about this stuff",
    "I love to listen to music while cooking",
    "her personality really shines in groups",
    "my curiosity got the better of me yesterday",
    # tool-adjacent nouns/verbs in non-imperative frames
    "I finally cleaned my desk and it feels like a new room",
    "my sister said she might visit next weekend",
    "I've been feeling a bit overwhelmed at work lately",
    "that movie we talked about was better than I expected",
    "I need to look after my health more",
    "we should talk about the trip budget sometime",
    "honestly the weather has been beautiful all week",
    "I read an interesting article about coral reefs yesterday",
    "my code review went well today",
    "let me think about what to cook tonight",
    "I wrote a letter to my grandmother",
    "she plays the violin beautifully",
    "what do you think about getting a dog",
    "I watched the sunrise this morning and it was stunning",
    "my friend runs a small bakery downtown",
    "the file cabinet in the office is a mess",
    "I listened to a podcast about deep sea creatures",
    "do you ever wonder what makes a good friendship",
    "the mail arrived late again today",
    "I took some photos of the garden this afternoon",
    "my brother is learning to code in his free time",
    "I have a call with my accountant tomorrow",
    "that song has been stuck in my head all day",
    "I finally finished reading that novel",
    "my screen time has been way too high lately",
    "I browsed a bookstore for an hour and bought nothing",
    "we planted tomatoes in the garden over the weekend",
    "the terminal at the airport was completely packed",
    "she downloaded way too many recipe apps",
    "the window in my bedroom lets in great light",
    "he executed the plan flawlessly at work",
    "I'd love to visit Japan in the spring",
]

POSITIVE_REQUESTS = [
    ("search the web for apple silicon benchmarks", "web_search"),
    ("look up the boiling point of ethanol", "web_search"),
    ("what's the latest news on the election", "web_search"),
    ("generate an image of a lighthouse at dusk", "image_gen"),
    ("draw me a dragon curled around a teapot", "image_gen"),
    ("use clock to check the time", "clock"),
    ("speak this aloud: hello world", "speak"),
    ("remember that my dentist appointment is on Friday", "memory_ops"),
]


class TestBenignConversationDispatchesNothing:
    @pytest.mark.parametrize("sentence", BENIGN_CONVERSATION)
    def test_no_dispatch(self, engine, sentence):
        matched = engine.detect_intent(sentence)
        assert matched == [], (
            f"benign conversation dispatched {matched}: {sentence!r} — a trigger "
            "pattern is over-broad; tighten the pattern, do not shrink this corpus"
        )


class TestRealRequestsStillDispatch:
    @pytest.mark.parametrize("sentence,skill", POSITIVE_REQUESTS)
    def test_dispatches(self, engine, sentence, skill):
        matched = engine.detect_intent(sentence)
        assert skill in matched, (
            f"{sentence!r} no longer dispatches {skill} (got {matched}) — "
            "a tightening went too far"
        )


class TestNormalizerRewritesTyposNotWords:
    def test_really_is_never_recall(self):
        assert "recall" not in normalize_memory_intent_text(
            "time really flies when things get busy"
        )

    @pytest.mark.parametrize("typo,target", [
        ("remeber", "remember"),
        ("rememebr", "remember"),
        ("recal", "recall"),
        ("reacll", "recall"),
    ])
    def test_typos_still_normalize(self, typo, target):
        assert target in normalize_memory_intent_text(f"can you {typo} this for me")

    @pytest.mark.parametrize("word", [
        "recently", "regard", "regards", "reveal", "reload", "related",
        "remind", "reminds", "remained", "records", "recorded",
    ])
    def test_ordinary_r_words_untouched(self, word):
        assert word in normalize_memory_intent_text(f"she {word} the garden plan")
