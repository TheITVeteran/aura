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


def _partition(reply_text: str) -> tuple[set[str], set[str]]:
    terms = content_terms(reply_text)
    runtime = {term for term in terms if term in RUNTIME_SELF_TERMS}
    return runtime, terms - runtime


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
    return SubjectDriftVerdict(
        True, "runtime_self_reference", share, external_terms, runtime_terms
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
    "RUNTIME_SELF_TERMS",
    "SubjectDriftVerdict",
    "answers_polar_question",
    "assess_subject_drift",
    "is_polar_question",
]
