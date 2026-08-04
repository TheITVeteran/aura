"""She must not invent a mechanism to explain her own knowledge.

Live, asked what she meant by "the foundation of what I am", she said:

    "When you were setting up your account, you went through a series of
     personality tests and questionnaires. The results are the foundation
     I'm referring to — they informed my model of who you are."

None of that exists. There is no account setup, no questionnaire, no
personality test anywhere in the codebase. The only Big Five in the repo is
AURA_BIG_FIVE — *her own* traits — which makes this most likely a confusion
of her personality for one the user supposedly took.

The existing HISTORICAL FIDELITY rule forbids fabricating past interactions.
It does not cover fabricating the *provenance* of knowledge, which is a
different claim and the one she made.
"""

from __future__ import annotations

import re

import pytest

from core.brain.llm.context_assembler import ContextAssembler
from core.state.aura_state import AuraState


def _prompt(objective: str) -> str:
    state = AuraState.default()
    state.cognition.current_objective = objective
    return ContextAssembler.build_system_prompt(state)


def test_no_onboarding_questionnaire_exists_in_the_codebase():
    """The premise of the confabulation is checkable, so check it."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    pattern = re.compile(r"personality test|questionnaire|onboarding survey", re.IGNORECASE)
    # The prompt rule names these mechanisms in order to deny them; that is the
    # fix, not a violation. Everywhere else, a mention would mean one got built.
    denial_sites = {"core/brain/llm/context_assembler.py"}
    offenders = []
    for path in list(root.glob("core/**/*.py")) + list(root.glob("interface/**/*.py")):
        rel = str(path.relative_to(root))
        if rel in denial_sites:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if pattern.search(text):
            offenders.append(rel)
    assert not offenders, f"an intake questionnaire appeared: {offenders}"


@pytest.mark.parametrize(
    "objective",
    ["hey how are you", "Perform a full architecture review of the runtime"],
)
def test_provenance_rule_reaches_every_requirements_block(objective):
    """Both requirement blocks carry it — a rule at one of two sites is the
    defect shape this repo keeps rediscovering."""
    prompt = _prompt(objective)
    assert "PROVENANCE" in prompt


@pytest.mark.parametrize(
    "objective",
    ["hey how are you", "Perform a full architecture review of the runtime"],
)
def test_provenance_rule_denies_the_specific_invented_mechanisms(objective):
    prompt = _prompt(objective).lower()
    assert "questionnaire" in prompt
    assert "personality test" in prompt


def test_provenance_rule_names_the_real_sources(objective="hey"):
    prompt = _prompt(objective).lower()
    # conversation, recalled memory, and her own beliefs
    assert "this conversation" in prompt
    assert "belief" in prompt
