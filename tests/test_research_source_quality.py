"""An ad redirect is not an article, and a nav bar is not reporting.

Measured live. Asked for "3 recent articles about AI", the document she wrote
cited these as her three sources:

  1. AI fundamentals | OpenAI — https://openai.com/academy/what-is-ai/
  2. OpenAI | Research & Deployment — https://openai.com/
  3. Claude Ai - Amazing AI Assistant — https://duckduckgo.com/y.js?ad_domain=
     ai%2Dpro.org&ad_provider=bingv7aa&ad_type=txad&click_metadata=...

The third is a DuckDuckGo AD REDIRECT, and its 600-character tracking URL was
printed into the document as a citation. The second is a product homepage.

And what she wrote as the synthesis was the site's navigation bar:

  "Taken together, the reporting points to this: AI fundamentals | OpenAI Skip
   to main content Research Products Business Developers Company Foundation
   (opens in a new window) Log in Try ChatGPT (opens in a new window)..."
"""

from __future__ import annotations

from core.skills.desktop_task import DesktopTaskSkill


def test_the_live_ad_redirect_is_rejected():
    ad = (
        "https://duckduckgo.com/y.js?ad_domain=ai%2Dpro.org&ad_provider=bingv7aa"
        "&ad_type=txad&click_metadata=xyhqpbOr3aW5cxvxdz2Ep"
    )
    assert DesktopTaskSkill._is_article_url(ad) is False


def test_click_trackers_and_search_pages_are_rejected():
    for url in (
        "https://www.bing.com/aclick?ld=e8EOOiDV7bPONO3m0q4lz0wzVUCUxOUD",
        "https://www.google.com/search?q=AI",
        "https://googleadservices.com/pagead/aclk?sa=L",
    ):
        assert DesktopTaskSkill._is_article_url(url) is False, url


def test_bare_product_homepages_are_not_articles():
    for url in ("https://openai.com/", "https://chatgpt.com/", "https://example.com"):
        assert DesktopTaskSkill._is_article_url(url) is False, url


def test_real_articles_are_kept():
    for url in (
        "https://openai.com/academy/what-is-ai/",
        "https://www.nature.com/articles/d41586-026-01234-5",
        "https://www.reuters.com/technology/some-story-2026-04-10/",
    ):
        assert DesktopTaskSkill._is_article_url(url) is True, url


def test_navigation_furniture_is_stripped_from_the_article_text():
    raw = (
        "AI fundamentals | OpenAI Skip to main content Research Products Business "
        "Developers Company Foundation (opens in a new window) Log in Try ChatGPT "
        "(opens in a new window) OpenAI April 10, 2026 Artificial intelligence "
        "systems learn patterns from data rather than following explicit rules."
    )
    cleaned = DesktopTaskSkill._strip_page_chrome(raw)
    for chrome in ("Skip to main content", "opens in a new window", "Try ChatGPT", "Log in"):
        assert chrome not in cleaned, f"nav furniture survived: {chrome!r}"
    # The actual reporting must survive.
    assert "Artificial intelligence systems learn patterns from data" in cleaned


def test_stripping_never_empties_real_prose():
    prose = "Researchers reported a measurable gain on held-out tasks this quarter."
    assert DesktopTaskSkill._strip_page_chrome(prose) == prose


def test_a_research_document_opens_with_the_synthesis_not_template_filler():
    """Two bodies were written, the empty one first.

    `_document_body` never consulted the research synthesis, so a research
    objective fell through to the generic composer and the document opened with:

        "Notes on the requested subject: The requested subject is the focus of
         this note. The important part is to describe the subject clearly..."

    ...followed by the actual three-source synthesis.
    """
    objective = (
        "Open a Google tab and find 3 recent articles about AI, read them and form "
        "your own opinion. Then open Google Docs and write a synthesis of the three "
        "articles plus your opinion."
    )
    context = {
        "desktop_task_research_query": "AI",
        "desktop_task_research_synthesis": (
            "I read three pieces on AI this morning. In my view the useful "
            "through-line is that capability gains are now reported alongside "
            "evaluation caveats, which was not true a year ago."
        ),
        "desktop_task_research_sources": [
            {
                "title": "AI fundamentals",
                "url": "https://openai.com/academy/what-is-ai/",
                "snippet": "Artificial intelligence systems learn patterns from data.",
            }
        ],
    }

    body = DesktopTaskSkill._document_body(objective, context)
    assert not body.startswith("Notes on"), "template filler still leads the document"
    assert "is the focus of this note" not in body
    assert body.lstrip().startswith("I read three pieces"), (
        "her synthesis must be the document, not a preamble to it"
    )
    assert "openai.com/academy" in body, "sources must still be recorded"


def test_site_navigation_is_dropped_by_sentence_shape_not_a_phrase_list():
    """A phrase list cannot generalise; every site's nav has its own words.

    NASA's survived the phrase filter intact and was written into the document
    as what the reporting said:

      "Taken together, the reporting points to this: Ocean Warming - Earth
       Indicator - NASA Science Explore Search News & Events News & Events
       Recently Published Video Series on NASA+ Podcasts & Audio Blogs
       Newsletters Social Media Media Resources Mult…"

    What separates nav from reporting is grammar. Reporting runs long, carries
    lowercase function words, and ends in a full stop; navigation is a run of
    Title Case labels with almost no verbs and almost no periods.
    """
    live = (
        "Ocean Warming - Earth Indicator - NASA Science Explore Search News & "
        "Events News & Events Recently Published Video Series on NASA+ Podcasts "
        "& Audio Blogs Newsletters Social Media Media Resources Mult… The "
        "concentration of the 2023 warming in near-surface waters suggests that "
        "upper ocean stratification, possibly modulated by large-scale climate "
        "modes, may have played an important role in the observed heat uptake. "
        "Extreme Weather Questions (FAQ) Earth Indicators Carbon Dioxide Global "
        "Temperature Methane Arctic Sea Ice Minimum Extent Ice Sheets"
    )
    cleaned = DesktopTaskSkill._strip_page_chrome(live)

    for nav in ("Explore Search", "Podcasts", "Newsletters", "Arctic Sea Ice", "Carbon Dioxide"):
        assert nav not in cleaned, f"navigation survived: {nav!r}"
    assert "upper ocean stratification" in cleaned, "the actual finding was lost"
    assert "may have played an important role" in cleaned


def test_ordinary_prose_passes_through_untouched():
    prose = (
        "Researchers reported a measurable gain on held-out tasks this quarter, "
        "and the effect persisted after the controls were tightened."
    )
    assert DesktopTaskSkill._strip_page_chrome(prose) == prose


def test_a_page_with_no_sentences_degrades_instead_of_vanishing():
    """Selecting prose must never turn a thin source into nothing at all."""
    assert DesktopTaskSkill._strip_page_chrome("Short note.") == "Short note."


def test_asking_for_a_synthesis_enables_authoring_it():
    """The synthesis was never attempted, so of course it was never written.

    Model synthesis required an opt-in flag that NOTHING on the live path set,
    so every "read them and form your own opinion, then write a synthesis in
    your own words" fell to the deterministic composer, which concatenates
    source snippets. What came back was "Taken together, the reporting points to
    this: <snippet> <snippet>" — no takeaway, nothing learned.
    """
    demo = (
        "Open a Google tab and find 3 recent articles about ocean warming, read "
        "them and form your own opinion. Then open Google Docs, start a new "
        "document, and write a synthesis of the three articles plus your opinion "
        "in your own words."
    )
    assert DesktopTaskSkill._allow_research_model_synthesis({}, demo) is True, (
        "authoring the synthesis IS the request; refusing it cannot satisfy it"
    )

    for phrasing in (
        "summarize the three articles in a Google Doc",
        "read them and give me your assessment",
        "write it up in your own words",
        "what do you think about these articles?",
    ):
        assert DesktopTaskSkill._objective_requests_authored_synthesis(phrasing) is True, phrasing


def test_merely_collecting_sources_stays_on_the_cheap_path():
    """The guard exists so background work cannot quietly spend a second model."""
    collect = "Find 3 links about ocean warming and paste them into a doc."
    assert DesktopTaskSkill._allow_research_model_synthesis({}, collect) is False
    assert DesktopTaskSkill._allow_research_model_synthesis({}, "") is False


def test_the_explicit_opt_in_still_works():
    assert (
        DesktopTaskSkill._allow_research_model_synthesis(
            {"allow_desktop_task_model_synthesis": True},
            "Find 3 links and paste them.",
        )
        is True
    )


def test_she_authors_the_artifact_when_she_must_supply_the_words():
    """A note that describes what a note should contain is empty of content.

    Only self-summaries and research documents ever reached the model. "Write a
    note with three sentences about orcas" fell to the deterministic composer,
    and the entire body was:

        "Notes on the requested subject: The requested subject is the focus of
         this note. The important part is to describe the subject clearly,
         ground it in concrete details, and preserve enough context that the
         note is useful after the moment of writing has passed."

    Correctly created, correctly saved, and saying nothing about orcas.
    """
    needs = DesktopTaskSkill._objective_needs_authored_content

    for objective in (
        "Open the Notes app and write a new note with three sentences about orcas.",
        "Write a short summary of what you think about whales into a note",
        "Write me a paragraph about bioluminescence and save it",
    ):
        assert needs(objective) is True, objective


def test_an_objective_with_its_own_content_source_is_not_authored():
    """Writing something new would ignore what was actually asked."""
    needs = DesktopTaskSkill._objective_needs_authored_content

    for objective in (
        "Create a note from the clipboard.",
        "Save the clipboard into a file",
        'Write a note that says "hello world"',
        "Put the selection into a document",
    ):
        assert needs(objective) is False, objective
