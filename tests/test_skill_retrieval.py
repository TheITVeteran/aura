"""Finding a skill by what it does, not by whether someone wrote the regex."""

from __future__ import annotations

import pytest

from core.skills.skill_retrieval import (
    LexicalIndex,
    SkillDocument,
    SkillRetriever,
    get_skill_retriever,
)

CATALOG = [
    SkillDocument("web_search", "Search the internet for information", "catalog"),
    SkillDocument("read_csv_file", "Load a comma separated values file into rows", "catalog"),
    SkillDocument("summarize_text", "Produce a short summary of a long document", "catalog"),
    SkillDocument("open_app", "Launch a desktop application by name", "catalog"),
]


def _retriever(documents=CATALOG) -> SkillRetriever:
    r = SkillRetriever()
    r.register_provider("catalog", lambda: list(documents))
    return r


def _names(hits) -> list[str]:
    return [h.name for h in hits]


# ── what the trigger patterns could not do ───────────────────────────────


def test_a_different_spelling_still_finds_the_skill():
    assert _names(_retriever().retrieve("summarise this article", k=1)) == ["summarize_text"]


def test_a_word_inside_a_compound_name_is_found():
    assert _names(_retriever().retrieve("load a csv", k=1)) == ["read_csv_file"]


def test_word_order_does_not_matter():
    a = _names(_retriever().retrieve("desktop application launch", k=1))
    b = _names(_retriever().retrieve("launch desktop application", k=1))
    assert a == b == ["open_app"]


def test_a_description_match_works_without_the_name():
    assert _names(_retriever().retrieve("search the internet", k=1)) == ["web_search"]


# ── an empty result has to be possible ───────────────────────────────────


def test_an_unrelated_query_returns_nothing():
    """Without a floor, "top 3" always returns three, whatever was asked."""
    assert _retriever().retrieve("zzzz qqqq xxxx", k=3) == []


def test_an_empty_query_returns_nothing():
    assert _retriever().retrieve("", k=3) == []
    assert _retriever().retrieve("   ", k=3) == []


def test_k_of_zero_returns_nothing():
    assert _retriever().retrieve("search the internet", k=0) == []


def test_an_empty_corpus_returns_nothing():
    r = SkillRetriever()
    r.register_provider("none", lambda: [])
    assert r.retrieve("anything", k=3) == []


# ── ranking ──────────────────────────────────────────────────────────────


def test_results_are_ordered_best_first():
    hits = _retriever().retrieve("summary of a long document", k=4)
    assert hits[0].name == "summarize_text"
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))


def test_k_bounds_the_result():
    assert len(_retriever().retrieve("search internet file document", k=2)) <= 2


def test_ties_break_by_name_so_ranking_is_deterministic():
    documents = [
        SkillDocument("alpha", "identical description here", "catalog"),
        SkillDocument("bravo", "identical description here", "catalog"),
    ]
    first = _names(_retriever(documents).retrieve("identical description here", k=2))
    second = _names(_retriever(documents).retrieve("identical description here", k=2))
    assert first == second


# ── providers ────────────────────────────────────────────────────────────


def test_the_index_follows_a_changing_corpus():
    documents = list(CATALOG)
    r = SkillRetriever()
    r.register_provider("catalog", lambda: list(documents))
    assert r.corpus_size() == 4

    documents.append(SkillDocument("send_email", "Compose and send an email message", "catalog"))
    assert r.corpus_size() == 5
    assert _names(r.retrieve("compose an email", k=1)) == ["send_email"]


def test_an_earlier_provider_wins_a_name_clash():
    """A learned macro must not shadow the shipped skill it was named after."""
    r = SkillRetriever()
    r.register_provider("catalog", lambda: [SkillDocument("clock", "shipped", "catalog")])
    r.register_provider("macros", lambda: [SkillDocument("clock", "learned", "macro")])
    assert _retriever_source(r, "clock") == "catalog"


def _retriever_source(r: SkillRetriever, name: str) -> str:
    hits = [h for h in r.retrieve("shipped learned clock", k=5) if h.name == name]
    return hits[0].source if hits else ""


def test_a_failing_provider_does_not_take_the_others_down():
    def broken():
        raise RuntimeError("provider exploded")

    r = _retriever()
    r.register_provider("broken", broken)
    assert _names(r.retrieve("search the internet", k=1)) == ["web_search"]


def test_a_provider_can_be_removed():
    r = _retriever()
    r.unregister_provider("catalog")
    assert r.corpus_size() == 0


def test_documents_without_a_name_are_ignored():
    r = SkillRetriever()
    r.register_provider("mixed", lambda: [SkillDocument("", "no name", "catalog"), *CATALOG])
    assert r.corpus_size() == len(CATALOG)


# ── the semantic backend ─────────────────────────────────────────────────


def _fake_encoder(vectors: dict[str, list[float]]):
    def encode(texts):
        return [vectors.get(t, vectors.get("__default__", [0.0, 0.0, 1.0])) for t in texts]

    return encode


def test_an_installed_encoder_is_actually_used():
    documents = [
        SkillDocument("alpha", "aaa", "catalog"),
        SkillDocument("bravo", "bbb", "catalog"),
    ]
    r = _retriever(documents)
    r.install_encoder(
        _fake_encoder(
            {
                "alpha aaa": [1.0, 0.0],
                "bravo bbb": [0.0, 1.0],
                "find me bravo": [0.0, 1.0],
                "__default__": [0.0, 1.0],
            }
        )
    )
    hits = r.retrieve("find me bravo", k=1)
    assert _names(hits) == ["bravo"]
    assert hits[0].backend == "semantic"
    assert r.report()["backend"] == "semantic"


def test_a_broken_encoder_degrades_to_lexical_instead_of_raising():
    def exploding(_texts):
        raise RuntimeError("encoder died")

    r = _retriever()
    r.install_encoder(exploding)
    hits = r.retrieve("search the internet", k=1)
    assert _names(hits) == ["web_search"]
    assert hits[0].backend == "lexical"
    assert r.report()["backend"] == "lexical", "a dead encoder is dropped, not retried forever"


def test_an_encoder_returning_the_wrong_count_is_dropped():
    r = _retriever()
    r.install_encoder(lambda texts: [[1.0, 0.0]])
    assert _names(r.retrieve("search the internet", k=1)) == ["web_search"]
    assert r.report()["backend"] == "lexical"


def test_semantic_results_below_zero_similarity_are_not_offered():
    documents = [SkillDocument("alpha", "aaa", "catalog")]
    r = _retriever(documents)
    r.install_encoder(
        _fake_encoder({"alpha aaa": [1.0, 0.0], "opposite": [-1.0, 0.0], "__default__": [-1.0, 0.0]})
    )
    assert r.retrieve("opposite", k=3) == []


def test_clearing_the_encoder_returns_to_lexical():
    r = _retriever()
    r.install_encoder(_fake_encoder({"__default__": [1.0, 0.0]}))
    r.install_encoder(None)
    assert r.report()["backend"] == "lexical"


# ── the lexical index directly ───────────────────────────────────────────


def test_the_index_rebuilds_cleanly_from_empty():
    index = LexicalIndex()
    assert index.search("anything", 3) == []
    index.rebuild(CATALOG)
    assert index.search("search the internet", 1)
    index.rebuild([])
    assert index.search("search the internet", 1) == []


def test_a_query_of_pure_punctuation_scores_nothing():
    assert LexicalIndex(CATALOG).search("!!! ???", 3) == []


# ── the singleton ────────────────────────────────────────────────────────


def test_the_singleton_is_stable():
    assert get_skill_retriever() is get_skill_retriever()


# ── learned macros are visible now ───────────────────────────────────────


@pytest.fixture
def library(tmp_path, monkeypatch):
    from core.agency.skill_library import SkillLibrary

    lib = SkillLibrary()
    monkeypatch.setattr(lib, "data_path", tmp_path / "skills.json")
    lib.skills = {}
    lib.learn_skill(
        "deploy_site",
        "Build the static site and upload it to the host",
        ["target"],
        [{"tool_name": "shell", "arguments": {"cmd": "build"}}],
    )
    lib.learn_skill(
        "tidy_downloads",
        "Sort files in the downloads folder into dated subfolders",
        ["folder"],
        [{"tool_name": "file_operation", "arguments": {"path": "{{folder}}"}}],
    )
    return lib


def test_a_learned_macro_can_be_found_by_what_it_does(library):
    found = library.retrieve("upload the built site", k=1)
    assert [s.name for s in found] == ["deploy_site"]


def test_the_prompt_is_retrieved_rather_than_dumped(library):
    text = library.get_available_skills_prompt("sort my downloads folder", k=1)
    assert "tidy_downloads" in text
    assert "deploy_site" not in text, "a growing library must not become a growing prompt"


def test_an_irrelevant_objective_offers_no_macros(library):
    assert library.get_available_skills_prompt("qqqq zzzz xxxx") == ""


def test_without_an_objective_everything_reliable_is_listed(library):
    text = library.get_available_skills_prompt()
    assert "deploy_site" in text and "tidy_downloads" in text


def test_an_unreliable_macro_is_not_offered(library):
    skill = library.skills["deploy_site"]
    skill.successes, skill.failures = 0, 10
    assert "deploy_site" not in library.get_available_skills_prompt()
    # Other macros may still surface — character trigrams bridge "upload" and
    # "downloads", which is the cost of matching morphology. What must not
    # happen is the macro that fails nine times in ten being offered for the
    # objective it is named after.
    assert "deploy_site" not in [s.name for s in library.retrieve("upload the built site", k=3)]


def test_an_empty_library_offers_nothing(tmp_path, monkeypatch):
    from core.agency.skill_library import SkillLibrary

    lib = SkillLibrary()
    monkeypatch.setattr(lib, "data_path", tmp_path / "skills.json")
    lib.skills = {}
    assert lib.get_available_skills_prompt("anything") == ""
    assert lib.retrieve("anything") == []
