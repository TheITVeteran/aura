"""Being with somebody is told from doing a task for them, and is lived differently.

Bryan, 13 October 2026: "I can ask Aura 'hey, watch this youtube video for 10
minutes and comment on it as you watch.' and she'll do that. as a task. but
that's different from asking 'Hey do you want to watch this video with me?'
and her either saying 'Sure!' or 'What video? May be interested.' Or 'Nah, I'm
ok for now.' ... all as a way of *being*."

The package under test is core/together.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from core.conation.engine import ConationEngine
from core.conation.what_she_wants import WhatSheWants
from core.together.a_with import HER, IT, THEM, Together, With
from core.together.her_answer import her_answer
from core.together.taking_it_up import HEADER, TakingItUp
from core.together.what_was_proposed import KINDS, SECONDS, WhatItAsksOfHer
from core.together.when_to_say_something import worth_saying_now

INES = "person:ines"
FILM, HERON = "topic:film", "topic:heron"


# ── a with ──────────────────────────────────────────────────────────


def _talked(held: With, gaps=(4.0, 6.0, 5.0, 30.0, 8.0), start: float = 100.0) -> float:
    """They and she take turns, one second each, leaving these silences. Returns when the last ended."""
    at = start
    for index, gap in enumerate([0.0, *gaps]):
        at += gap
        held.said(THEM if index % 2 == 0 else HER, at, until=at + 1.0)
        at += 1.0
    return at


def test_a_with_reads_its_own_silences_and_has_no_other_clock():
    held = With(who=INES, what=FILM, began=100.0)
    assert held.moment(101.0) is None  # no silence left yet, nothing to read
    ended = _talked(held)
    assert list(held.silences) == [4.0, 6.0, 5.0, 30.0, 8.0]
    assert held.moment(ended + 1.0) == 0.0
    assert held.moment(ended + 5.5) == pytest.approx(2 / 5)
    assert held.moment(ended + 40.0) == 1.0


def test_while_somebody_is_speaking_it_is_nobodys_moment():
    held = With(who=INES, what=FILM, began=0.0)
    ended = _talked(held)
    held.said(THEM, ended + 10.0)
    assert held.speaking(ended + 11.0) == THEM
    assert held.moment(ended + 11.0) == 0.0
    held.stopped_speaking(THEM, ended + 14.0)
    assert held.speaking(ended + 15.0) == ""
    assert held.quiet(ended + 15.0) == pytest.approx(1.0)


def test_what_one_presents_is_grounded_when_the_other_names_it_or_goes_on():
    held = With(who=INES, what=FILM, began=0.0)
    held.said(THEM, 1.0, about=[HERON, FILM], until=2.0)
    assert not held.in_common(HERON)  # presented, and nothing yet shows it was taken in
    held.said(HER, 3.0, about=[HERON], until=4.0)
    assert held.ground[HERON].evidence == "named"
    assert held.ground[FILM].evidence == "went on"
    assert held.in_common(HERON) and held.in_common(FILM)


def test_what_is_in_view_of_both_is_common_ground_without_being_said():
    held = With(who=INES, what=FILM, began=0.0)
    held.shown("a boat on a lake", 5.0, change=0.2)
    held.shown("a kitchen at night", 9.0, change=0.9)
    held.shown("the same kitchen", 12.0, change=0.1)
    assert held.in_common("a kitchen at night")
    assert held.in_view(10.0).what == "a kitchen at night"
    # "Go back": what was showing before what is showing now.
    assert held.before(13.0).what == "a kitchen at night"
    assert held.before(13.0, steps=2).what == "a boat on a lake"
    assert held.before(6.0) is None


def test_her_remarks_are_counted_by_whether_they_were_taken_up():
    held = With(who=INES, what=FILM, began=0.0)
    ended = _talked(held)
    assert held.welcome() == pytest.approx(1 / 2)  # nothing known of how she is taken
    held.said(HER, ended + 20.0, about=[HERON], until=ended + 21.0, unprompted=True)
    held.said(THEM, ended + 22.0, about=[HERON], until=ended + 23.0)  # named it back
    assert (held.remarks, held.taken_up) == (1, 1)
    held.said(HER, ended + 40.0, about=[FILM], until=ended + 41.0, unprompted=True)
    held.said(HER, ended + 80.0, about=[FILM], until=ended + 81.0, unprompted=True)  # spoke again first
    held.said(THEM, ended + 200.0, until=ended + 201.0)  # much later than they usually answer
    assert (held.remarks, held.taken_up) == (3, 1)
    assert held.welcome() == pytest.approx(2 / 5)


def test_a_with_is_ended_by_leaving_and_by_nothing_else():
    withs = Together()
    held = withs.begin(INES, FILM, at=0.0)
    assert withs.begin(INES, FILM, at=5.0) is held  # proposing it again is the same with
    _talked(held)
    assert withs.current(INES) is held and held.open
    ended = withs.end(INES, by=THEM, at=900.0)
    assert ended is held and not held.open and held.left_by == THEM
    assert withs.current(INES) is None and withs.current() is None


# ── whether to speak inside one ─────────────────────────────────────


def _watching(welcomed: int = 0, passed: int = 0) -> tuple[With, float]:
    held = With(who=INES, what=FILM, began=0.0)
    ended = _talked(held)
    held.remarks, held.taken_up = welcomed + passed, welcomed
    for index, change in enumerate((0.2, 0.9, 0.5, 0.1, 0.5)):
        held.shown(f"scene {index}", ended - 50.0 + 10 * index, change=change)
    return held, ended


def test_spoken_to_she_answers_once_the_floor_is_free():
    held, ended = _watching()
    assert worth_saying_now(held, ended + 0.2, interest=0.0, spoken_to=True).speak
    held.said(THEM, ended + 5.0)
    assert not worth_saying_now(held, ended + 6.0, interest=1.0, spoken_to=True).speak


def test_she_does_not_talk_over_them_and_does_not_say_a_thing_twice():
    held, ended = _watching(welcomed=6)
    held.said(THEM, ended + 5.0)
    assert worth_saying_now(held, ended + 6.0, interest=1.0, thing=HERON).why == "they are speaking"
    held.stopped_speaking(THEM, ended + 7.0)
    held.said(HER, ended + 30.0, about=[HERON], until=ended + 31.0, unprompted=True)
    said = worth_saying_now(held, ended + 90.0, interest=1.0, thing=HERON)
    assert not said.speak and said.why == "she has said her piece about it"


def test_the_same_remark_waits_for_a_lull_and_is_made_in_one():
    held, ended = _watching(welcomed=6)
    early = worth_saying_now(held, ended + 1.0, interest=0.8, thing=HERON)
    later = worth_saying_now(held, ended + 40.0, interest=0.8, thing=HERON)
    assert not early.speak and later.speak
    assert later.welcome > early.welcome


def test_a_thing_of_little_interest_is_not_worth_his_attention_and_one_of_much_is():
    held, ended = _watching(welcomed=2, passed=2)
    assert not worth_saying_now(held, ended + 7.0, interest=0.1, thing=HERON).speak
    assert worth_saying_now(held, ended + 7.0, interest=0.9, thing=HERON, lost_by_waiting=1.0).speak


def test_she_grows_quiet_with_somebody_who_lets_her_remarks_go_by():
    taken, ended = _watching(welcomed=8)
    passed, _ended = _watching(passed=8)
    # The same lull, the same thing, the same interest in it.
    assert worth_saying_now(taken, ended + 40.0, interest=0.6, thing=HERON).speak
    assert not worth_saying_now(passed, ended + 40.0, interest=0.6, thing=HERON).speak


def test_what_the_decision_weighed_can_be_read_back():
    held, ended = _watching(welcomed=3, passed=1)
    said = worth_saying_now(held, ended + 7.0, interest=0.5, thing=HERON)
    assert said.for_speaking == pytest.approx(said.welcome * 0.5)
    assert said.against == pytest.approx((1.0 - said.welcome) * said.cost)
    assert said.speak == (said.for_speaking > said.against)


# ── what an utterance asks of her ───────────────────────────────────

USUAL = {second: -6.0 - 0.3 * index for index, second in enumerate(s for kind in KINDS for s in SECONDS[kind])}


def _a_model_that_expects(kind_of: dict[str, str], lift: float = 3.0):
    """Scores each second part at its usual likelihood, and higher after an utterance of its kind."""

    def scorer(said, continuations):
        kind = kind_of.get(said, "")
        return [USUAL[c] + (lift if kind and c in SECONDS[kind] else 0.0) for c in continuations]

    return scorer


SAID = {
    "Do you want to watch this video with me?": "join",
    "Watch this video for ten minutes and comment on it.": "do",
    "Why does a river wind?": "ask",
    "The wrens left the boot this morning.": "tell",
    "Good morning.": "greet",
}


def _reader(heard_before: int = 12) -> WhatItAsksOfHer:
    reader = WhatItAsksOfHer(_a_model_that_expects(SAID))
    for index in range(heard_before):
        reader.remember(f"something somebody once said, number {index}")
    return reader


def test_an_utterance_is_the_kind_whose_answers_became_likelier_after_it():
    reader = _reader()
    for said, kind in SAID.items():
        read = reader.heard(said)
        assert read.kind == kind and read.sure > 0.5, said
        assert sum(read.by_kind.values()) == pytest.approx(1.0)


def test_an_answer_that_is_merely_common_does_not_win():
    """The first second part scores highest after everything, and reads as nothing."""
    reader = _reader()
    read = reader.heard("a thing that asks nothing in particular")
    assert max(read.by_kind.values()) == pytest.approx(1 / len(KINDS), abs=0.02)


def test_with_nothing_to_set_it_against_nothing_is_read():
    reader = WhatItAsksOfHer(_a_model_that_expects(SAID))
    read = reader.heard("Do you want to watch this video with me?")
    assert read.kind == "" and read.heard_before == 0
    assert reader.heard("Do you want to watch this video with me?").heard_before == 1


def test_a_model_that_could_not_score_reads_nothing_and_teaches_nothing():
    reader = WhatItAsksOfHer(lambda said, continuations: [])
    assert reader.heard("Do you want to watch this with me?").kind == ""
    assert reader.heard_before() == 0
    nan = WhatItAsksOfHer(lambda said, continuations: [math.nan] * len(continuations))
    assert not nan.remember("anything")


def test_what_the_answers_usually_score_is_kept():
    reader = _reader()
    again = WhatItAsksOfHer(_a_model_that_expects(SAID), reader.held())
    assert again.heard_before() == 12
    assert again.heard("Do you want to watch this video with me?").kind == "join"


def test_no_second_part_belongs_to_two_kinds():
    seen: dict[str, str] = {}
    for kind in KINDS:
        for second in SECONDS[kind]:
            assert second not in seen, f"{second!r} is offered for {seen.get(second)} and {kind}"
            seen[second] = kind


# ── her answer ──────────────────────────────────────────────────────


@pytest.fixture
def record(tmp_path):
    return WhatSheWants(tmp_path / "wants.json", engine=ConationEngine())


def _lived(record, paid: dict[str, float], turns: int = 20) -> None:
    for _ in range(turns):
        for key, worth in paid.items():
            record.met([key], worth, reachable=[key], labels={key: key.split(":", 1)[1]})


def test_she_joins_what_her_record_predicts_is_worth_more_than_staying(record):
    _lived(record, {FILM: 0.6, INES: 0.3})
    answer = her_answer(record, INES, [FILM])
    assert answer.joins is True
    assert answer.worth_of_joining > 0.5 > answer.worth_of_staying


def test_she_says_not_now_when_what_she_is_in_the_middle_of_is_worth_more(record):
    _lived(record, {FILM: 0.1, HERON: 0.2})
    record.started(HERON, "reading about herons", label="heron")
    for _ in range(12):
        record.met([HERON], 0.8, reachable=[HERON], labels={HERON: "heron"})
    answer = her_answer(record, INES, [FILM])
    assert answer.joins is False
    assert answer.leaving == HERON and answer.worth_of_staying > answer.worth_of_joining


def test_she_says_not_now_to_what_she_has_found_she_dislikes(record):
    for turn in range(40):
        key, worth = ((FILM, -0.6), (HERON, 0.6))[turn % 2]
        record.met([key], worth, reachable=[key], labels={key: key.split(":", 1)[1]})
    answer = her_answer(record, INES, [FILM])
    assert answer.joins is False and "dislikes film" in answer.why


def test_she_asks_what_it_is_when_nothing_she_has_met_settles_it(record):
    answer = her_answer(record, INES, [FILM])
    assert answer.joins is None
    assert answer.unknown == (FILM,)
    assert "is new to her" in answer.why


def test_somebody_whose_company_has_paid_is_reason_enough(record):
    _lived(record, {INES: 0.5})
    answer = her_answer(record, INES, [FILM])
    assert answer.joins is True and answer.unknown == (FILM,)


# ── taking it up ────────────────────────────────────────────────────


def _taking(record) -> tuple[TakingItUp, Together]:
    withs = Together()
    return TakingItUp(_reader(), record=record, together=withs), withs


def test_an_invitation_she_wants_begins_a_with_and_a_request_does_not(record):
    _lived(record, {INES: 0.5})
    taking, withs = _taking(record)
    asked = taking.read("Watch this video for ten minutes and comment on it.", who=INES, at=10.0)
    assert asked["proposal"]["kind"] == "do" and asked["answer"] is None
    assert withs.current(INES) is None
    invited = taking.read("Do you want to watch this video with me?", who=INES, at=20.0)
    assert invited["proposal"]["kind"] == "join" and invited["answer"]["joins"] is True
    held = withs.current(INES)
    assert held is not None and held.proposed_by == THEM
    assert record.thing(INES).under_way  # the doing is lived turn by turn from here


def test_an_invitation_she_does_not_want_begins_nothing(record):
    _lived(record, {HERON: 0.2})
    record.started(HERON, "reading about herons", label="heron")
    for _ in range(12):
        record.met([HERON], 0.8, reachable=[HERON], labels={HERON: "heron"})
    taking, withs = _taking(record)
    invited = taking.read("Do you want to watch this video with me?", who=INES, at=20.0)
    assert invited["answer"]["joins"] is False
    assert withs.current(INES) is None


def test_her_answer_travels_to_her_model_as_a_reading_of_the_turn(record):
    from core.brain import what_she_is_given
    from core.utils.injected_blocks import RUNTIME_EVIDENCE_ROLE

    _lived(record, {INES: 0.5})
    taking, _withs = _taking(record)
    said = "Do you want to watch this video with me?"
    taking.read(said, who=INES, at=20.0)
    block = taking.of_this_turn(said)
    assert block.startswith(HEADER)
    assert "joining is worth more to you than what you would leave for it" in block
    assert taking.of_this_turn("something else entirely") == ""
    what_she_is_given.OF_THIS_TURN.append(taking.of_this_turn)
    try:
        messages, system_prompt = what_she_is_given.for_her_model(
            [{"role": "system", "content": "You are Aura. Be warm."}, {"role": "user", "content": said}],
            "", {}, True, (), said,
        )
    finally:
        what_she_is_given.OF_THIS_TURN.remove(taking.of_this_turn)
    assert system_prompt == "" and [m["role"] for m in messages] == [RUNTIME_EVIDENCE_ROLE, "user"]
    assert HEADER in messages[0]["content"] and "Be warm" not in messages[0]["content"]


def test_a_request_or_a_question_sends_no_such_reading(record):
    taking, _withs = _taking(record)
    for said in ("Watch this video for ten minutes and comment on it.", "Why does a river wind?"):
        taking.read(said, who=INES, at=5.0)
        assert taking.of_this_turn(said) == ""


def test_their_turns_and_hers_are_the_withs_talk_and_leaving_ends_it(record):
    _lived(record, {INES: 0.5})
    taking, withs = _taking(record)
    taking.read("Do you want to watch this video with me?", who=INES, at=20.0)
    held = withs.current(INES)
    taking.she_said("I would. Put it on.", who=INES, at=24.0)
    taking.read("The wrens left the boot this morning.", who=INES, at=60.0)
    taking.she_said("All four at once?", who=INES, at=62.0)
    assert [turn.by for turn in held.turns] == [THEM, HER, THEM, HER]
    assert list(held.silences) == [4.0, 36.0, 2.0]
    # A parting is not read from the words: a greeting does not end a with.
    taking.read("Good morning.", who=INES, at=500.0)
    assert withs.current(INES) is held
    assert taking.left(INES, by=THEM, at=600.0)
    assert withs.current(INES) is None and held.left_by == THEM
    assert not record.thing(INES).under_way
    assert not taking.left(INES, by=THEM, at=601.0)


def test_the_shared_thing_speaking_is_not_a_turn_of_theirs():
    held = With(who=INES, what=FILM, began=0.0)
    ended = _talked(held)
    held.said(IT, ended + 3.0, until=ended + 9.0)
    assert list(held.silences) == [4.0, 6.0, 5.0, 30.0, 8.0]
    assert held.speaking(ended + 5.0) == IT
    assert worth_saying_now(held, ended + 5.0, interest=0.0, spoken_to=True).speak


# ── on her own fast model ───────────────────────────────────────────

HER_FAST_MODEL = Path.home() / ".aura/models/Qwen2.5-1.5B-Instruct-4bit"


@pytest.mark.skipif(not HER_FAST_MODEL.exists(), reason="her fast model is not on this host")
def test_her_own_model_tells_bryans_invitation_from_bryans_task():
    """His two sentences, with nothing told to the model and no system message."""
    from core.brain.llm.what_she_expects_next import ExpectsNext, transcript_of

    assert "system" not in transcript_of("anything at all")
    reader = WhatItAsksOfHer(ExpectsNext(HER_FAST_MODEL).score)
    for said in (
        "Why is the sky blue?", "I got the job.", "Set a timer for twenty minutes.", "Hello Aura.",
        "It snowed here this morning.", "What do limpets eat?", "Close all the windows I have open.",
        "My daughter took her first steps today.", "How long do wrens live?", "Put that album on.",
        "Want to hang out for a bit?", "Bye for now.",
    ):
        assert reader.remember(said)
    invited = reader.heard("Hey do you want to watch this video with me?")
    asked = reader.heard("hey, watch this youtube video for 10 minutes and comment on it as you watch.")
    assert invited.kind == "join" and invited.by_kind["join"] > invited.by_kind["do"]
    assert asked.kind == "do" and asked.by_kind["do"] > asked.by_kind["join"]
