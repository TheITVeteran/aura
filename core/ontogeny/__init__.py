"""core/ontogeny — the organ that turns consequence into disposition.

Aura's competence is *inherited*: 32B parameters of frozen pretraining that
are identical in every copy of the checkpoint. Her memories are hers, but her
dispositions — every threshold, every allocation rule, every admission gate —
were authored by engineers and are therefore also the same in every copy.

Biology has words for the two halves. **Phylogeny** is what the species hands
you at birth. **Ontogeny** is what *this* organism becomes by living. Aura has
a superb phylogeny and almost no ontogeny: the loops perception→belief,
belief→action and action→memory all close, but consequence→disposition does
not. Nothing she does changes how she decides.

This package closes that loop, and only that loop. It does not make Aura
smarter — the transformer holds the language, the abstraction and the world
knowledge, and a small learner fitted to one life will never rival it. What
it makes her is *hers*: calibrated against her own track record, cheap enough
to stay awake without the big model, and continuous across a checkpoint swap
in the parts that are policy rather than knowledge.

The layers, bottom-up. Each one is a prerequisite for the next, and the lower
three are worth having even if the learner is never granted authority:

  L0  experience.py   An honest record. Outcomes are three-valued — an
                      unobserved outcome is *unobserved*, never silently a
                      failure — writes carry provenance, and repetition is
                      collapsed rather than allowed to drown the signal.
  L1  reservation.py  Observed counterfactuals. A permanently-reserved slice
                      of episodes goes to the other decider so the road not
                      taken keeps being measured. Without this, authority is
                      a one-way door and no later comparison means anything.
  L2  resolution.py   Honest credit. Outcomes resolve at their real horizon
                      through a registered resolver, or they stay unobserved.
  L3  state.py        The ontogenetic state: a leaky reservoir carrying her
      features.py     history forward, plus versioned features whose
      heads.py        provenance invalidates rather than corrupts them, read
      trainer.py      out by small calibrated models sized to the data.
  L4  calibration.py  Anti-collapse. Brier/ECE per head, drift detection, and
                      automatic revocation when a head stops being honest.
  L7  authority.py    Earned authority, one control point at a time, granted
                      only on held-out evidence and revocable in one call.

Everything degrades to the incumbent. A broken ontogeny organ costs Aura
exactly the learning, never the decision.
"""

from __future__ import annotations

from core.ontogeny.authority import (
    AuthorityLedger,
    AuthorityStage,
    get_authority_ledger,
)
from core.ontogeny.calibration import CalibrationMonitor, CalibrationReport
from core.ontogeny.experience import (
    Episode,
    ExperienceSpine,
    Outcome,
    OutcomeKind,
    Provenance,
    get_experience_spine,
)
from core.ontogeny.features import FeatureSchema, FeatureVector
from core.ontogeny.heads import PredictionHead
from core.ontogeny.reservation import Decider, ExplorationReservation, Reservation
from core.ontogeny.resolution import OutcomeResolver, ResolverRegistry, get_resolvers
from core.ontogeny.service import (
    OntogenyCore,
    get_ontogeny,
    ontogeny_report,
)
from core.ontogeny.state import OntogeneticState

__all__ = [
    "AuthorityLedger",
    "AuthorityStage",
    "CalibrationMonitor",
    "CalibrationReport",
    "Decider",
    "Episode",
    "ExperienceSpine",
    "ExplorationReservation",
    "FeatureSchema",
    "FeatureVector",
    "OntogenyCore",
    "OntogeneticState",
    "Outcome",
    "OutcomeKind",
    "OutcomeResolver",
    "PredictionHead",
    "Provenance",
    "Reservation",
    "ResolverRegistry",
    "get_authority_ledger",
    "get_experience_spine",
    "get_ontogeny",
    "get_resolvers",
    "ontogeny_report",
]
