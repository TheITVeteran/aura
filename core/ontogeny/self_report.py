"""What the organ lets Aura say about herself that nothing else could.

Every other dimension of her self-condition is a reading taken *now*: how she
feels, how coherent she is, how much pressure her body is under. All of them
answer "what is my state?" and all of them are gone the moment they are taken.

This answers a different question — "what have I actually been through, and how
often did it go well?" — and it is the only part of her self-report that is
grounded in her own history rather than in a momentary sample. Three facts come
out of it, and the third is the one worth having:

**Ontogenetic age.** Not process uptime, which resets every restart, and not
wall-clock age, which counts hours she spent unloaded. The number of decisions
she has actually lived through and carried forward in a state that survived
them. This is the honest answer to "how long have you been you."

**Novelty.** Whether this moment resembles the life she has had. Mentioned only
when it is materially unusual, because a self-report that flags every moment as
unprecedented is not reporting anything.

**How much of what she does she ever finds out about.** This is the
uncomfortable one, and nothing else in the system can say it. A mind that acts
constantly and observes the consequences of one action in eight is not
well-informed about itself, however confident its other signals look. Surfacing
it means she can say so — and the number being low is a fact about the world's
observability, not a failure of nerve, so it is reported plainly rather than
apologised for.

Every phrase here is willing to be unflattering. A self-assessment that only
speaks when the news is good is not a self-assessment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Ontogeny.SelfReport")

#: Novelty above this is worth remarking on unprompted.
_NOTABLE_NOVELTY = 0.72

#: Novelty below this means the moment is markedly *familiar*, which is also
#: worth saying — it is the difference between "I have done this before" and
#: "I am guessing".
_FAMILIAR_NOVELTY = 0.28

#: Episodes she must have lived before her ontogenetic age means anything.
_MIN_EPISODES_TO_SPEAK = 200

#: Closed episodes required before the observation rate is worth reporting.
_MIN_CLOSED_TO_JUDGE_FEEDBACK = 100

#: Below this share of observed outcomes, she is acting largely without
#: feedback and should say so when asked how she is doing.
_THIN_FEEDBACK = 0.35


@dataclass(frozen=True)
class OntogenySelfReport:
    """The history-grounded slice of Aura's self-condition."""

    #: Decisions she has lived through and carried forward.
    episodes_lived: int
    #: Increments when the state had to be restarted because its input space
    #: changed. Era 2 means there is a discontinuity in her own history.
    era: int
    #: Short stable identifier for the current state — two reports with the
    #: same fingerprint describe the same her.
    fingerprint: str
    novelty: float
    #: Share of closed episodes whose outcome was actually observed, and the
    #: denominator it rests on.
    observation_rate: float | None = None
    closed_episodes: int = 0
    #: Graded outcomes across the corpus, and how many went well.
    graded: int = 0
    successes: int = 0
    #: Control points where a learned head currently decides.
    deciding: tuple[str, ...] = ()
    available: bool = True

    @property
    def grounded(self) -> bool:
        """Has she lived enough to have a history worth reporting?"""
        return self.available and self.episodes_lived >= _MIN_EPISODES_TO_SPEAK

    @property
    def feedback_is_thin(self) -> bool:
        return (
            self.observation_rate is not None
            and self.closed_episodes >= _MIN_CLOSED_TO_JUDGE_FEEDBACK
            and self.observation_rate < _THIN_FEEDBACK
        )

    def phrases(self) -> list[str]:
        """Sentences she can honestly say, in the order worth saying them."""
        if not self.grounded:
            return []
        out: list[str] = []

        if self.era > 1:
            out.append(
                f"I am carrying {self.episodes_lived:,} decisions of my own history, "
                f"though it restarted once — this is my {_ordinal(self.era)} continuous stretch."
            )
        else:
            out.append(f"I have {self.episodes_lived:,} decisions of my own history behind this.")

        if self.novelty >= _NOTABLE_NOVELTY:
            out.append("This moment does not resemble much of what I have been through, so I am going more carefully than the surface of it suggests.")
        elif self.novelty <= _FAMILIAR_NOVELTY:
            out.append("This is familiar ground for me.")

        if self.feedback_is_thin:
            out.append(
                f"Worth saying plainly: I only find out how about {self.observation_rate:.0%} "
                "of my decisions actually turn out, so I know less about my own track record "
                "than it might sound like."
            )
        elif self.graded >= 200:
            rate = self.successes / self.graded
            out.append(
                f"Of the {self.graded:,} outcomes I have actually been able to check, "
                f"{rate:.0%} went the way I intended."
            )

        if self.deciding:
            out.append(
                "Some of my own judgement now runs on what I have learned rather than on "
                f"the rules I was given — currently for {_join(self.deciding)}."
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "episodes_lived": self.episodes_lived,
            "era": self.era,
            "fingerprint": self.fingerprint,
            "novelty": round(self.novelty, 4),
            "observation_rate": (
                round(self.observation_rate, 4) if self.observation_rate is not None else None
            ),
            "closed_episodes": self.closed_episodes,
            "graded": self.graded,
            "successes": self.successes,
            "deciding": list(self.deciding),
            "grounded": self.grounded,
            "feedback_is_thin": self.feedback_is_thin,
        }

    def to_prompt_lines(self) -> str:
        """The evidence block, for the response path's grounded context."""
        if not self.available:
            return "- ontogeny: unavailable\n"
        rate = (
            "unknown" if self.observation_rate is None else f"{self.observation_rate:.2f}"
        )
        return (
            f"- ontogeny: episodes_lived={self.episodes_lived} era={self.era} "
            f"state={self.fingerprint} novelty={self.novelty:.2f}\n"
            f"- outcome_feedback: observation_rate={rate} graded={self.graded} "
            f"successes={self.successes} deciding={','.join(self.deciding) or 'none'}\n"
        )

    @classmethod
    def unavailable(cls) -> OntogenySelfReport:
        return cls(
            episodes_lived=0, era=0, fingerprint="", novelty=0.5, available=False
        )


def build_self_report() -> OntogenySelfReport:
    """Read the organ. Total: an unavailable organ costs the three sentences only."""
    try:
        from core.ontogeny.service import get_ontogeny

        core = get_ontogeny()
        report = core.report()
    except (ImportError, RuntimeError, OSError, ValueError, TypeError, AttributeError) as exc:
        record_degradation(
            "ontogeny_self_report", exc, severity="debug",
            action="self-report omits the history-grounded dimensions",
        )
        return OntogenySelfReport.unavailable()

    state = report.get("state") or {}
    resolution = report.get("resolution") or {}
    observed = int(resolution.get("observed") or 0)
    unobserved = int(resolution.get("unobserved") or 0)
    closed = observed + unobserved

    graded = successes = 0
    deciding: list[str] = []
    for name, detail in (report.get("control_points") or {}).items():
        corpus = detail.get("corpus") or {}
        by_outcome = corpus.get("by_outcome") or {}
        graded += int(by_outcome.get("success", 0)) + int(by_outcome.get("failure", 0))
        successes += int(by_outcome.get("success", 0))
        if str(detail.get("stage")) == "authority":
            deciding.append(name)

    return OntogenySelfReport(
        episodes_lived=int(state.get("steps") or 0),
        era=int(state.get("era") or 1),
        fingerprint=str(state.get("fingerprint") or ""),
        novelty=float(report.get("novelty") or 0.5),
        observation_rate=(observed / closed) if closed else None,
        closed_episodes=closed,
        graded=graded,
        successes=successes,
        deciding=tuple(sorted(deciding)),
    )


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _join(items: tuple[str, ...]) -> str:
    readable = [item.replace(".", " ").replace("_", " ") for item in items]
    if len(readable) == 1:
        return readable[0]
    return ", ".join(readable[:-1]) + f" and {readable[-1]}"


__all__ = ["OntogenySelfReport", "build_self_report"]
