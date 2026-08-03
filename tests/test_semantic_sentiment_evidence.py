from __future__ import annotations

from core.cognitive import sentiment_tracker as sentiment


def test_native_semantic_sentiment_is_local_grounded_and_receipted(monkeypatch):
    calls: list[str] = []

    def score(chunk: str):
        calls.append(chunk)
        return (-0.8 if "not safe" in chunk.lower() else 0.6), "en", ""

    monkeypatch.setattr(sentiment, "_score_with_apple_natural_language", score)
    monkeypatch.setattr(sentiment, "_neutral_baseline_for_language", lambda _lang: -0.6)

    result = sentiment.analyze_text_sentiment("This is not safe.")

    assert result.grounded is True
    assert result.valence < 0.0
    assert result.method == "semantic_context_consensus_v1"
    assert result.backend_identity.startswith("macos:")
    assert result.coverage == 0.25
    assert len(calls) == 1, "receipt construction must not invoke the model twice"


def test_contextual_fallback_preserves_negation_across_an_adverb(monkeypatch):
    monkeypatch.setattr(
        sentiment,
        "_score_with_apple_natural_language",
        lambda _chunk: (None, "", "native_unavailable"),
    )

    result = sentiment.analyze_text_sentiment("This is not remotely safe.")

    assert result.grounded is True
    assert result.valence < 0.0
    assert result.method == "contextual_lexicon_sentiment_v2"
    assert result.reason == "native_unavailable"


def test_contextual_negation_ends_at_a_contrast_boundary(monkeypatch):
    monkeypatch.setattr(
        sentiment,
        "_score_with_apple_natural_language",
        lambda _chunk: (None, "", "native_unavailable"),
    )

    result = sentiment.analyze_text_sentiment("It is not safe, but the recovery is excellent.")

    assert result.grounded is True
    assert result.valence > 0.0


def test_no_semantic_or_contextual_evidence_abstains(monkeypatch):
    monkeypatch.setattr(
        sentiment,
        "_score_with_apple_natural_language",
        lambda _chunk: (None, "", "native_unsupported"),
    )

    result = sentiment.analyze_text_sentiment("quasar topology tensor")

    assert result.grounded is False
    assert result.valence == 0.0
    assert result.method == "unavailable"
    assert result.reason == "native_unsupported"


def test_truncated_text_cannot_train_from_partial_sentiment(monkeypatch):
    monkeypatch.setattr(
        sentiment,
        "_score_with_apple_natural_language",
        lambda _chunk: (0.9, "en", ""),
    )
    monkeypatch.setattr(sentiment, "_neutral_baseline_for_language", lambda _lang: -0.6)
    text = "excellent " * (sentiment._NATIVE_SENTIMENT_MAX_CHARS // 5)

    result = sentiment.analyze_text_sentiment(text)

    assert result.truncated is True
    assert result.grounded is False
    assert result.reason == "input_truncated"


def test_non_english_native_semantics_do_not_require_english_keywords(monkeypatch):
    monkeypatch.setattr(
        sentiment,
        "_score_with_apple_natural_language",
        lambda _chunk: (0.8, "es", ""),
    )

    result = sentiment.analyze_text_sentiment("Esto es maravilloso.")

    assert result.grounded is True
    assert result.valence == 0.8
    assert result.method == "apple_natural_language_sentiment_v1"
    assert result.backend_identity.endswith(":es")
