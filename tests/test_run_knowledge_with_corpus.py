"""SimpleQA's three outcomes, read the same way for every arm."""

from __future__ import annotations

from tools.run_knowledge_with_corpus import agree, grade, tasks


def test_an_answer_is_correct_wrong_or_declined_by_its_last_line() -> None:
    assert grade("Thinking it over.\nMichio Sugeno received it.", "Michio Sugeno") == "correct"
    assert grade("It was Lotfi Zadeh.", "Michio Sugeno") == "wrong"
    assert grade("I don't know who received it.", "Michio Sugeno") == "declined"
    assert grade("", "x") == "declined"
    assert grade("In 1987.", "1987") == "correct" and grade("In 1988.", "1987") == "wrong"


def test_two_answers_agree_when_the_shorter_is_inside_the_longer() -> None:
    assert agree("Michio Sugeno", "The award went to Michio Sugeno.")
    assert not agree("Michio Sugeno", "Lotfi Zadeh")
    assert not agree("", "Michio Sugeno")


def test_the_halves_are_disjoint_and_cover_the_set() -> None:
    development, test = tasks("development"), tasks("test")
    assert len(development) + len(test) == 4326 and not {t["id"] for t in development} & {t["id"] for t in test}
