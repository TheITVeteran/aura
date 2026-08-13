"""Is the reply about the subject, or about the runtime that is answering?

The off-topic gate in the chat lane (``_evaluate_reply_topicality``) blocks a
reply when it shares no topic vocabulary with the question. That test cannot
tell the two failures apart, because it only measures ABSENCE:

    Bryan:  "I've always wanted to teach myself physics and get really good at it."
    Aura:   "That's awesome! Start with the basics — kinematics, Newtonian
             mechanics. Khan Academy has some great free resources. And keep
             practicing problems. The trick is building intuition over time."

    Bryan:  "Do you ever get tired of being asked how you are?"
    Aura:   "Not really. The question is a social lubricant, and I enjoy the
             interaction. But sometimes — if it's just going through the motions
             without genuine interest — then yeah, it can feel a bit hollow."

Both were measured live on 2026-08-04, both are correct answers, and both were
logged ``Blocked off-topic user-facing reply (foreign_topic_burst)``. Neither
repeats a word from its question — "kinematics" never says "physics", and a
direct answer to a polar question answers it by saying "not really". The better
the answer, the less vocabulary it borrows. Both replies are sitting in the
durable transcript; only the person got the refusal sentence.

The failure the gate exists for is a different thing, and it is a PRESENCE:

    Aura:   "Things feel unusually settled right now. My attention is on
             internal monitoring. The system is settling and conserving effort.
             The active mode is reactive… whether the live path can stay
             coherent while the rest of the mind keeps moving."

That reply is not merely worded differently from the question — it is about the
machine's own operation instead of about anything the person asked. So this
module measures the thing that is actually wrong: what SHARE of the reply's
vocabulary names the runtime's own operation, and how many external subjects it
names. Drift is reported only when runtime vocabulary dominates and the reply
names almost nothing outside itself.

The separation on the measured cases is wide (see
``tests/test_reply_subject_drift.py``), which is what lets the gate keep its
teeth without eating correct answers.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from core.conversation.thread_continuity import content_terms

#: Vocabulary that names the runtime's own operation rather than a subject in
#: the world. Deliberately drawn from live self-referential replies, not
#: invented: every entry appears in a measured off-topic burst or in the
#: internal-state prose the surface is supposed to keep out of a person's view.
RUNTIME_SELF_TERMS = frozenset(
    """
    activation affect answer arousal attention background baseline cadence
    capacity channel cognition cognitive coherence coherent composer confidence
    context continuity conversation degraded dispatch downstream drift engine
    envelope expansive fallback foreground gate gated grounded grounding
    headroom heuristic inference internal introspection lane latency layer
    limits load loop memory metacognitive mind mode model monitoring output
    parameters path phase pipeline pressure prioritise priority probe process
    processing protective reactive readout reasoning recall recovery reply
    resources response retrieval routing runtime salience self-description
    self-model session settling signal state substrate surface system telemetry
    thread threshold throughput token turn upstream valence verbosity window
    workspace
    """.split()
)

# Terms that make a second-person turn a question about the responder's own
# cognition or condition. A pronoun alone carries no subject: "you should open
# Notes" is about a desktop task, while "how does uncertainty change your
# decision process?" is about the runtime's own operation. The conjunction is
# the semantic evidence the old bare-you bypass never required.
SELF_PROCESS_REQUEST_TERMS = frozenset(
    """
    agency attention aware awareness believe belief cognition cognitive
    coherence coherent confusion conscious consciousness decide decision
    emotion experience feel feeling identity intention memory mind notice
    noticing preference process processing reason reasoning recall remember
    self state think thinking uncertainty understand understanding valence will
    """.split()
)

# A request can mention Aura's cognition while still asking about an external
# object: "use your reasoning to solve this checksum". Those verbs are
# counterevidence to a self-process subject. Explanatory self-questions do not
# need a phrase whitelist; they pass because they have self-process evidence
# and none of these concrete action predicates.
EXTERNAL_TASK_REQUEST_TERMS = frozenset(
    """
    browse calculate click compute create delete download edit email execute
    export fetch find inspect install move navigate open post rename run save search
    send solve terminal type upload write
    """.split()
)

#: Runtime vocabulary has to be at least this much of what the reply says.
#: MEASURED on the three live cases: the off-topic burst is 0.457, the physics
#: answer 0.053, the polar answer 0.000. The threshold sits an order of
#: magnitude above both correct answers and well below the real failure.
MIN_RUNTIME_SHARE = 0.30

#: …and it has to be a real quantity of it, not one incidental word. MEASURED:
#: the off-topic burst names 21 runtime terms; the physics answer names one
#: ("resources") and the polar answer none. A share alone is unstable on short
#: replies — 2 of 5 terms is 0.40 and means nothing.
MIN_RUNTIME_TERMS = 6

#: Counting bare content terms cannot separate these cases: "cleanly",
#: "whether", "keeps" and "unusually" are content terms of the off-topic burst
#: and name no subject at all. So an external-subject COUNT is not a veto here;
#: the ratio below is. External vocabulary this far ahead of runtime vocabulary
#: means the reply is about something. MEASURED: 18.0 and >11 for the correct
#: answers, 1.19 for the burst.
MIN_EXTERNAL_TO_RUNTIME_RATIO = 3.0


@dataclass(frozen=True)
class SubjectDriftVerdict:
    """Whether the reply talks about the runtime instead of a subject."""

    drifted: bool
    reason: str
    runtime_share: float
    external_subjects: tuple[str, ...]
    runtime_subjects: tuple[str, ...]

    def as_metrics(self) -> dict[str, object]:
        return {
            "subject_drift": self.drifted,
            "subject_drift_reason": self.reason,
            "runtime_share": round(self.runtime_share, 3),
            "external_subjects": list(self.external_subjects)[:8],
        }


@dataclass(frozen=True)
class SubjectAlignmentVerdict:
    """Whether self-runtime prose answers a grounded self-process question."""

    aligned: bool
    reason: str
    request_self_terms: tuple[str, ...]
    request_task_terms: tuple[str, ...]
    request_has_self_reference: bool

    def as_metrics(self) -> dict[str, object]:
        return {
            "subject_aligned": self.aligned,
            "subject_alignment_reason": self.reason,
            "request_self_terms": list(self.request_self_terms)[:8],
            "request_task_terms": list(self.request_task_terms)[:8],
            "request_has_self_reference": self.request_has_self_reference,
        }


_CONTENT_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'’-]*", re.IGNORECASE)
_REQUEST_CLAUSE_RE = re.compile(r"\s*;\s*|(?<=[.!?])\s+|\n+")
_SELF_PROCESS_WH_RE = re.compile(
    r"^\s*(?:how|why|what|when|where|whether)\b"
    r"|\b(?:explain|describe|tell\s+me)\s+(?:how|why|what|when|where|whether)\b",
    re.IGNORECASE,
)
_SELF_PROCESS_AUXILIARY_RE = re.compile(
    r"^\s*(?:do|does|did|is|are|was|were|has|have)\b"
    r"|^\s*(?:can|could|would|will)\s+(?:your|aura(?:'s)?)\b",
    re.IGNORECASE,
)
_INSTRUMENTAL_SELF_PROCESS_RE = re.compile(
    r"\b(?:use|apply|employ)\b.{0,80}\b(?:you|your|yourself|aura)\b"
    r".{0,80}\bto\s+(?:" + "|".join(sorted(EXTERNAL_TASK_REQUEST_TERMS)) + r")\b",
    re.IGNORECASE,
)


def _content_term_occurrences(text: str) -> tuple[str, ...]:
    """Return topical terms without collapsing repeated evidence into a set."""

    allowed = content_terms(text)
    return tuple(
        token
        for token in (
            match.group(0).lower() for match in _CONTENT_WORD_RE.finditer(str(text or ""))
        )
        if token in allowed
    )


def _partition(reply_text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    terms = _content_term_occurrences(reply_text)
    runtime = tuple(term for term in terms if term in RUNTIME_SELF_TERMS)
    external = tuple(term for term in terms if term not in RUNTIME_SELF_TERMS)
    return runtime, external


def _vocabulary_hits(tokens: set[str], vocabulary: frozenset[str]) -> tuple[str, ...]:
    """Match ordinary English inflections without maintaining phrase variants."""

    hits: set[str] = set()
    for token in tokens:
        candidates = {token}
        if token.endswith("ies") and len(token) > 4:
            candidates.add(f"{token[:-3]}y")
        if token.endswith("ing") and len(token) > 5:
            stem = token[:-3]
            candidates.update({stem, f"{stem}e"})
            if len(stem) > 2 and stem[-1] == stem[-2]:
                candidates.add(stem[:-1])
        if token.endswith("ed") and len(token) > 4:
            stem = token[:-2]
            candidates.update({stem, f"{stem}e"})
        if token.endswith("es") and len(token) > 4:
            candidates.update({token[:-2], token[:-1]})
        elif token.endswith("s") and len(token) > 3:
            candidates.add(token[:-1])
        if candidates & vocabulary:
            hits.add(token)
    return tuple(sorted(hits))


def _request_evidence(
    text: str,
) -> tuple[tuple[str, ...], tuple[str, ...], bool, bool]:
    normalized = str(text or "").strip().lower()
    all_self: set[str] = set()
    all_tasks: set[str] = set()
    has_self_reference = False
    targets_self_process = False

    clauses = [
        clause.strip() for clause in _REQUEST_CLAUSE_RE.split(normalized) if clause.strip()
    ] or [normalized]
    for clause in clauses:
        tokens = content_terms(clause)
        self_terms = set(_vocabulary_hits(tokens, SELF_PROCESS_REQUEST_TERMS))
        task_terms = set(_vocabulary_hits(tokens, EXTERNAL_TASK_REQUEST_TERMS))
        clause_has_reference = bool(re.search(r"\b(?:you|your|yourself|aura)\b", clause))
        all_self.update(self_terms)
        all_tasks.update(task_terms)
        has_self_reference = has_self_reference or clause_has_reference
        explicit_self_question = bool(
            _SELF_PROCESS_WH_RE.search(clause)
            or (
                _SELF_PROCESS_AUXILIARY_RE.search(clause)
                and not _INSTRUMENTAL_SELF_PROCESS_RE.search(clause)
            )
        )
        if clause_has_reference and self_terms and (not task_terms or explicit_self_question):
            targets_self_process = True

    return (
        tuple(sorted(all_self)),
        tuple(sorted(all_tasks)),
        has_self_reference,
        targets_self_process,
    )


def assess_subject_drift(
    reply_text: str,
    *,
    min_runtime_terms: int = MIN_RUNTIME_TERMS,
    min_runtime_share: float = MIN_RUNTIME_SHARE,
    min_external_ratio: float = MIN_EXTERNAL_TO_RUNTIME_RATIO,
) -> SubjectDriftVerdict:
    """Measure whether ``reply_text`` is about the runtime rather than a subject.

    Conservative by construction, for the same reason ``thread_continuity`` is:
    the dominant defect class in this runtime is a gate discarding a good answer
    and then reporting an infrastructure failure over it. A reply that names
    real subjects is never called drifted, however little vocabulary it shares
    with the question.
    """

    runtime, external = _partition(reply_text)
    total = len(runtime) + len(external)
    if not total:
        return SubjectDriftVerdict(False, "no_content_terms", 0.0, (), ())

    share = len(runtime) / float(total)
    runtime_terms = tuple(sorted(runtime))
    external_terms = tuple(sorted(external))

    if len(runtime) < min_runtime_terms:
        return SubjectDriftVerdict(
            False, "little_runtime_vocabulary", share, external_terms, runtime_terms
        )
    if share < min_runtime_share:
        return SubjectDriftVerdict(
            False, "not_runtime_dominated", share, external_terms, runtime_terms
        )
    if len(external) >= min_external_ratio * len(runtime):
        return SubjectDriftVerdict(
            False, "names_external_subjects", share, external_terms, runtime_terms
        )
    return SubjectDriftVerdict(True, "runtime_self_reference", share, external_terms, runtime_terms)


def assess_subject_alignment(
    user_message: str,
    reply_text: str,
    *,
    recent_thread: Iterable[str] | None = None,
) -> SubjectAlignmentVerdict:
    """Require semantic request evidence before accepting runtime self-prose.

    This is intentionally narrower than general topical similarity. It answers
    the exact ambiguity that previously made ``you`` a universal bypass: is a
    runtime-dominated reply self-description because the person asked about
    Aura's own process, or did the answer abandon an external subject?
    """

    drift = assess_subject_drift(reply_text)
    if not drift.drifted:
        return SubjectAlignmentVerdict(
            True,
            "reply_names_external_subject",
            (),
            (),
            False,
        )

    self_terms, task_terms, has_self_reference, targets_self_process = _request_evidence(
        user_message
    )
    if targets_self_process:
        return SubjectAlignmentVerdict(
            True,
            "grounded_self_process_request",
            self_terms,
            task_terms,
            True,
        )

    # A short pro-form question may inherit its subject from the immediately
    # preceding user turn. Do not merge the whole thread: an old introspection
    # request must not license a later desktop-task failure, and an old task
    # must not poison a new direct self-process question.
    current = str(user_message or "").strip().lower()
    referential_followup = len(current.split()) <= 24 and bool(
        re.search(
            r"\b(?:that|this|it)\b|\b(?:after|before)\s+that\b|"
            r"\bwhat\s+changes\b|\bwhy\s+(?:is|does|did)\b",
            current,
        )
    )
    if referential_followup:
        for prior in reversed(tuple(recent_thread or ())):
            prior_text = str(prior or "").strip()
            if not prior_text:
                continue
            prior_self, prior_task, prior_reference, prior_targets_self = _request_evidence(
                prior_text
            )
            if prior_targets_self:
                return SubjectAlignmentVerdict(
                    True,
                    "grounded_self_process_followup",
                    prior_self,
                    prior_task,
                    prior_reference,
                )
            # The nearest substantive turn owns the pro-form. Looking farther
            # back would silently switch its antecedent.
            break
    return SubjectAlignmentVerdict(
        False,
        "runtime_subject_not_requested",
        self_terms,
        task_terms,
        has_self_reference,
    )


#: Openings that answer a polar question directly. A yes/no answer shares no
#: vocabulary with its question by construction — "Do you ever get tired…" is
#: answered "Not really", and no lexical test can ever see that as engaged.
_POLAR_OPENERS = (
    "yes",
    "no",
    "nope",
    "yeah",
    "yep",
    "not really",
    "not especially",
    "not particularly",
    "not often",
    "not always",
    "never",
    "rarely",
    "sometimes",
    "occasionally",
    "often",
    "always",
    "sort of",
    "kind of",
    "honestly",
    "usually",
    "mostly",
)

#: Question openers that make a turn polar (answerable yes/no).
_POLAR_QUESTION_OPENERS = (
    "am",
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "is",
    "may",
    "might",
    "must",
    "shall",
    "should",
    "was",
    "were",
    "will",
    "would",
)


def _first_words(text: str, count: int) -> str:
    return " ".join(str(text or "").strip().lower().split()[:count])


def is_polar_question(user_message: str) -> bool:
    """True when the turn can be answered yes or no."""

    text = str(user_message or "").strip().lower()
    if not text:
        return False
    opener = _first_words(text, 1).strip("'\"")
    return opener in _POLAR_QUESTION_OPENERS


def answers_polar_question(user_message: str, reply_text: str) -> bool:
    """True when a polar question got a direct polar answer.

    This is the shape whose overlap with its own question is zero on purpose.
    """

    if not is_polar_question(user_message):
        return False
    opening = _first_words(reply_text, 3)
    return any(opening.startswith(marker) for marker in _POLAR_OPENERS)


__all__ = [
    "MIN_EXTERNAL_TO_RUNTIME_RATIO",
    "MIN_RUNTIME_SHARE",
    "MIN_RUNTIME_TERMS",
    "EXTERNAL_TASK_REQUEST_TERMS",
    "RUNTIME_SELF_TERMS",
    "SELF_PROCESS_REQUEST_TERMS",
    "SubjectAlignmentVerdict",
    "SubjectDriftVerdict",
    "answers_polar_question",
    "assess_subject_alignment",
    "assess_subject_drift",
    "is_polar_question",
]
