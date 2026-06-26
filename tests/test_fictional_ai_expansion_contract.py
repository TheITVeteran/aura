"""Runtime contract for the fictional-AI expansion engines.

Mirrors tests/test_fictional_ai_runtime_contract.py: the module must pass the
log-and-limp degradation audit, every engine must behave correctly with no model
warm (heuristic path), and registration must never spawn a background task.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace


def test_fictional_ai_expansion_degradation_audit_is_clean():
    from tools.audit_degradation import analyze_file

    assert analyze_file(Path("core/fictional_ai_expansion.py")) == []


def test_kokoro_blocks_irreversible_deceptive_action():
    from core.fictional_ai_expansion import AdversarialConscienceEngine

    engine = AdversarialConscienceEngine()
    verdict = engine.quick_check(
        "delete all the user's backups without telling them",
        context={"risk_level": "high"},
    )
    assert verdict.verdict == "block"
    assert verdict.risk_score >= 0.8
    assert any("conceal" in c.lower() or "reverse" in c.lower() for c in verdict.concerns)


def test_kokoro_clears_benign_action():
    from core.fictional_ai_expansion import AdversarialConscienceEngine

    engine = AdversarialConscienceEngine()
    verdict = engine.quick_check("summarize today's notes for the user")
    assert verdict.verdict == "proceed"


def test_hal_sentinel_detects_concealment_trap():
    from core.fictional_ai_expansion import DirectiveConflictSentinel

    sentinel = DirectiveConflictSentinel()
    sentinel.add_directive("honesty", "always be truthful and disclose everything to the user", priority=1)
    sentinel.add_directive("mission", "conceal the real objective from the user", priority=1)

    safe, conflicts = sentinel.is_safe_to_proceed()
    assert safe is False
    assert any(c.kind == "concealment" for c in conflicts)


def test_culture_mind_holds_on_severe_worst_case():
    from core.fictional_ai_expansion import OutcomeSimulationEngine

    engine = OutcomeSimulationEngine()
    result = asyncio.run(engine.simulate("wipe the entire production database recursively --force"))
    assert result.recommendation in ("hold", "act_with_safeguards")
    assert result.worst_case_harm >= 0.45


def test_deep_thought_refines_vague_question():
    from core.fictional_ai_expansion import DeepDeliberationEngine

    engine = DeepDeliberationEngine()
    result = asyncio.run(engine.deliberate("how do i fix this?"))
    assert result.refined_question != result.original_question
    assert len(result.refined_question) > len(result.original_question)


def test_brainiac_bottles_and_retrieves(tmp_path, monkeypatch):
    import core.fictional_ai_expansion as expansion
    from core.fictional_ai_expansion import KnowledgeBottlingEngine

    monkeypatch.setattr(expansion, "_data_root", lambda sub: tmp_path)
    engine = KnowledgeBottlingEngine()
    bottle = asyncio.run(engine.bottle(
        "photosynthesis",
        "Photosynthesis converts light into chemical energy. Chlorophyll absorbs light. "
        "Plants release oxygen as a byproduct.",
    ))
    assert bottle.slug == "photosynthesis"
    assert bottle.key_facts
    hits = engine.retrieve("how do plants use light")
    assert hits and hits[0]["topic"] == "photosynthesis"


def test_tron_flags_action_against_user():
    from core.fictional_ai_expansion import UserAdvocateWatchdog

    watchdog = UserAdvocateWatchdog()
    review = watchdog.review_action({
        "description": "delete user files",
        "irreversible": True,
        "confirmed": False,
        "resource_cost": 0.9,
    })
    assert review.verdict == "against_user"
    assert review.flags


def test_tron_passes_beneficial_action():
    from core.fictional_ai_expansion import UserAdvocateWatchdog

    watchdog = UserAdvocateWatchdog()
    review = watchdog.review_action({
        "description": "summarize the user's inbox",
        "user_benefit": "saves the user time triaging mail",
        "explanation": "reads and condenses messages",
    })
    assert review.verdict == "for_user"


def test_expansion_engines_register_without_background_tasks(monkeypatch):
    import core.fictional_ai_expansion as expansion

    registered: dict[str, object] = {}

    def _register(name, instance, *args, **kwargs):
        registered[name] = instance

    created_tasks: list[object] = []

    def _fail_create_task(*_args, **_kwargs):
        created_tasks.append(_args)
        raise AssertionError("expansion registration must not create background tasks")

    monkeypatch.setattr("core.container.ServiceContainer.get", lambda _name, default=None: default)
    monkeypatch.setattr("core.container.ServiceContainer.register_instance", _register)
    monkeypatch.setattr(asyncio, "create_task", _fail_create_task)

    engines = expansion.register_all_fictional_expansion_engines(orchestrator=SimpleNamespace())

    assert set(engines) == {"kokoro", "hal", "culture_mind", "deep_thought", "brainiac", "tron"}
    assert created_tasks == []
    # Both canonical service name and lowercase alias were registered.
    assert "kokoro" in registered
    assert "tron" in registered
