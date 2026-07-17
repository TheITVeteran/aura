"""Deterministic product-quality contract for resident latent answers.

The worker receipt proves how an episode ran. This module separately proves
that the user-visible text is complete enough to count as an answer. It never
generates replacement prose and therefore cannot create a second model owner.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any

OUTPUT_QUALITY_SCHEMA = "aura.latent_output_quality.v1"

_WORD_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)\S", re.MULTILINE)
_SENTENCE_END_RE = re.compile(r"[.!?](?:[\"')\]]+)?(?=\s|$)")
_REQUEST_FACETS = {
    "compare": re.compile(r"\b(?:compare|contrast|difference|versus|vs\.?)\b", re.I),
    "select": re.compile(
        r"\b(?:choose|recommend|prefer|stronger|better|best\s+(?:design|option|architecture))\b",
        re.I,
    ),
    "verify": re.compile(r"\b(?:verify|test|prove|validate|certif(?:y|ication))\b", re.I),
    "explain": re.compile(r"\b(?:explain|why|how|caus(?:e|al|ality))\b", re.I),
    "enumerate": re.compile(r"\b(?:list|enumerate|steps|each)\b", re.I),
}
_ANSWER_FACETS = {
    "compare": re.compile(
        r"\b(?:whereas|while|unlike|versus|compared|by\s+contrast|in\s+contrast|early|late)\b",
        re.I,
    ),
    "select": re.compile(
        r"\b(?:choose|recommend|prefer|stronger|best|should\s+(?:use|choose|adopt)|the\s+winner)\b",
        re.I,
    ),
    "verify": re.compile(
        r"\b(?:verify|test|assert|inject|simulate|fault|cancel|timeout|restart|invariant|receipt)\w*\b",
        re.I,
    ),
    "explain": re.compile(
        r"\b(?:because|therefore|thus|so\s+that|leads?\s+to|prevents?|causes?|ensures?)\b",
        re.I,
    ),
    "enumerate": re.compile(r"\b(?:first|second|third|finally|steps?)\b", re.I),
}
_STOPWORDS = {
    "about", "after", "again", "against", "also", "among", "answer", "because",
    "before", "being", "both", "could", "design", "does", "each", "every", "explain",
    "from", "have", "into", "itself", "more", "most", "other", "should", "some", "stronger",
    "such", "than", "that", "their", "then", "there", "these", "they", "this", "through",
    "under", "using", "verify", "what", "when", "where", "which", "while", "with", "would",
}
_SUBJECT_TRAIL_RE = re.compile(
    r"\b(?:under|against|across|including)\s+([^?.;\n]{3,180})",
    re.I,
)
_SUBJECT_NOISE = {
    "and", "case", "cases", "condition", "conditions", "fault", "faults", "scenario",
    "scenarios", "the", "with",
}


def request_facets(objective: Any) -> list[str]:
    """Facets a request explicitly asks for (compare/select/verify/…).

    Public so allocation can shape the answer surface (token budget, decode
    discipline) with EXACTLY the same definition the quality gate will later
    judge the answer by — no drift between what is provisioned and what is
    demanded."""
    text = objective if isinstance(objective, str) else ""
    return [name for name, pattern in _REQUEST_FACETS.items() if pattern.search(text)]


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(text)]


def _concept(token: str) -> str:
    token = token.lower().replace("-", " ").strip()
    replacements = {
        "cancellation": "cancel",
        "cancelled": "cancel",
        "canceled": "cancel",
        "cancelling": "cancel",
        "timeouts": "timeout",
        "restarted": "restart",
        "restarts": "restart",
        "restarting": "restart",
    }
    if token in replacements:
        return replacements[token]
    if len(token) > 6 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 5 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def _max_blank_line_run(text: str) -> int:
    maximum = current = 0
    for line in text.splitlines():
        if line.strip():
            current = 0
        else:
            current += 1
            maximum = max(maximum, current)
    return maximum


def _terminal_complete(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    # An odd fence count means an unclosed code block ANYWHERE in the text —
    # the answer is structurally truncated no matter how its last line ends.
    if stripped.count("```") % 2 != 0:
        return False
    if stripped.endswith("```"):
        return True
    return stripped.endswith((".", "?", "!", ")", "]", "}"))


def _listed_subjects(objective: str) -> list[dict[str, Any]]:
    subjects: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for match in _SUBJECT_TRAIL_RE.finditer(objective):
        for raw_part in re.split(r",|\band\b", match.group(1), flags=re.I):
            raw_tokens = [
                token
                for token in _tokens(raw_part)
                if token not in _SUBJECT_NOISE
            ]
            keys: list[str] = []
            for token in raw_tokens:
                keys.extend(
                    _concept(part)
                    for part in token.replace("-", " ").split()
                    if part and part not in _SUBJECT_NOISE
                )
            key_tuple = tuple(dict.fromkeys(key for key in keys if len(key) >= 3))
            if not key_tuple or key_tuple in seen:
                continue
            seen.add(key_tuple)
            subjects.append(
                {
                    "label": " ".join(raw_tokens)[:80],
                    "keys": list(key_tuple),
                }
            )
            if len(subjects) >= 8:
                return subjects
    return subjects


def evaluate_latent_output(
    text: Any,
    *,
    generated_tokens: Any,
    termination: Any,
    objective: Any,
) -> dict[str, Any]:
    """Return a self-contained, hash-bound acceptance receipt."""

    rendered = text if isinstance(text, str) else ""
    objective_text = objective if isinstance(objective, str) else ""
    generated = generated_tokens if type(generated_tokens) is int else 0
    stop = termination if isinstance(termination, str) else ""
    words = _tokens(rendered)
    objective_words = _tokens(objective_text)
    normalized_nonempty_lines = [
        " ".join(_tokens(line))
        for line in rendered.splitlines()
        if line.strip()
    ]
    normalized_nonempty_lines = [line for line in normalized_nonempty_lines if line]
    list_items = len(_LIST_ITEM_RE.findall(rendered))
    code_fence_count = rendered.count("```")
    structured = bool(list_items >= 2 or (code_fence_count >= 2 and code_fence_count % 2 == 0))
    sentence_count = len(_SENTENCE_END_RE.findall(rendered))
    # Technical prose legitimately chains clauses with semicolons
    # ("cancellation revokes the token; timeouts trip the guard; …") — a
    # live 228-word answer satisfying every requested facet was rejected as
    # underdeveloped because only [.!?] counted as discourse boundaries.
    semicolon_clauses = rendered.count(";")
    discourse_units = max(sentence_count + semicolon_clauses, list_items)
    max_blank_lines = _max_blank_line_run(rendered)
    lexical_yield = len(words) / max(1, generated)

    trigrams = list(zip(words, words[1:], words[2:]))
    trigram_diversity = len(set(trigrams)) / max(1, len(trigrams))
    line_duplication_ratio = (
        1.0 - len(set(normalized_nonempty_lines)) / len(normalized_nonempty_lines)
        if normalized_nonempty_lines
        else 0.0
    )

    requested_facets = [
        name for name, pattern in _REQUEST_FACETS.items() if pattern.search(objective_text)
    ]
    satisfied_facets = [
        name
        for name in requested_facets
        if _ANSWER_FACETS[name].search(rendered)
        or (name == "enumerate" and list_items >= 2)
    ]
    missing_facets = sorted(set(requested_facets) - set(satisfied_facets))
    compound = len(requested_facets) >= 2

    objective_terms: list[str] = []
    for token in objective_words:
        concept = _concept(token)
        if len(concept) >= 4 and token not in _STOPWORDS and concept not in objective_terms:
            objective_terms.append(concept)
        if len(objective_terms) >= 32:
            break
    answer_concepts = {_concept(token) for token in words}
    matched_objective_terms = [
        term for term in objective_terms if term in answer_concepts
    ]
    objective_coverage = len(matched_objective_terms) / max(1, len(objective_terms))

    listed_subjects = _listed_subjects(objective_text)
    covered_subjects = [
        subject["label"]
        for subject in listed_subjects
        if all(key in answer_concepts for key in subject["keys"])
    ]
    listed_subject_coverage = len(covered_subjects) / max(1, len(listed_subjects))

    minimum_words = 5
    if compound:
        minimum_words = 28 if structured else 48
    elif generated >= 64:
        minimum_words = 16
    if stop == "token_limit" and generated >= 128 and not structured:
        minimum_words = max(minimum_words, math.ceil(generated * 0.25))

    reasons: list[str] = []
    if not rendered.strip():
        reasons.append("empty_output")
    if generated <= 0:
        reasons.append("invalid_generated_token_count")
    if len(words) < minimum_words:
        reasons.append("insufficient_lexical_content")
    if generated >= 64 and not structured and lexical_yield < 0.22:
        reasons.append("low_lexical_yield")
    if max_blank_lines > 2:
        reasons.append("excessive_blank_lines")
    if len(trigrams) >= 24 and trigram_diversity < 0.70:
        reasons.append("repetitive_language")
    if len(normalized_nonempty_lines) >= 4 and line_duplication_ratio > 0.35:
        reasons.append("repeated_lines")
    if compound and discourse_units < 3 and not structured:
        reasons.append("compound_answer_underdeveloped")
    if missing_facets:
        reasons.append("missing_requested_facets")
    if compound and (len(matched_objective_terms) < 2 or objective_coverage < 0.08):
        reasons.append("objective_disconnected")
    if len(listed_subjects) >= 2 and listed_subject_coverage < 0.60:
        reasons.append("listed_subjects_uncovered")
    terminal_complete = _terminal_complete(rendered)
    if rendered.strip() and not structured and not terminal_complete:
        reasons.append("terminal_fragment")

    return {
        "schema": OUTPUT_QUALITY_SCHEMA,
        "policy": "resident_latent_product_quality_v1",
        "passed": not reasons,
        "text_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "objective_sha256": hashlib.sha256(objective_text.encode("utf-8")).hexdigest(),
        "char_count": len(rendered),
        "word_count": len(words),
        "generated_token_count": generated,
        "termination": stop,
        "lexical_yield": round(lexical_yield, 6),
        "sentence_count": sentence_count,
        "list_item_count": list_items,
        "structured_output": structured,
        "max_blank_line_run": max_blank_lines,
        "trigram_diversity": round(trigram_diversity, 6),
        "line_duplication_ratio": round(line_duplication_ratio, 6),
        "terminal_complete": terminal_complete,
        "minimum_word_count": minimum_words,
        "compound_request": compound,
        "requested_facets": requested_facets,
        "satisfied_facets": satisfied_facets,
        "missing_facets": missing_facets,
        "objective_term_count": len(objective_terms),
        "matched_objective_terms": matched_objective_terms,
        "objective_term_coverage": round(objective_coverage, 6),
        "listed_subjects": [subject["label"] for subject in listed_subjects],
        "covered_listed_subjects": covered_subjects,
        "listed_subject_coverage": round(listed_subject_coverage, 6),
        "reasons": reasons,
    }


__all__ = ["OUTPUT_QUALITY_SCHEMA", "evaluate_latent_output", "request_facets"]
