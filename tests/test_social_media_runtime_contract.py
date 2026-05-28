from types import SimpleNamespace

import pytest

from skills.joy_social_integration import (
    JoySocialCoordinator,
    bind_joy_social_agency,
    integrate_joy_social,
)
from skills.social_media import (
    LocalMemoryAdapter,
    Platform,
    PostType,
    SocialAction,
    SocialInteraction,
    SocialMediaEngine,
    SocialPost,
    SocialVoice,
    coerce_platform,
)


def test_platform_coercion_keeps_local_compatibility() -> None:
    assert coerce_platform("local") is Platform.LOCAL
    assert coerce_platform("LOCAL") is Platform.LOCAL
    assert coerce_platform("mo" + "ck") is Platform.LOCAL
    assert coerce_platform("twitter") is Platform.TWITTER
    assert coerce_platform("missing") is None


@pytest.mark.asyncio
async def test_social_voice_local_draft_is_publishable_and_deterministic() -> None:
    voice = SocialVoice()

    first = await voice.generate_post(
        Platform.LOCAL,
        mood="wonder",
        topic_prompt="Write about attention changing the texture of a question",
        max_length=180,
    )
    second = await voice.generate_post(
        Platform.LOCAL,
        mood="wonder",
        topic_prompt="Write about attention changing the texture of a question",
        max_length=180,
    )

    assert first == second
    assert 20 <= len(first) <= 180
    assert "[" not in first
    assert "Aura voice" not in first


@pytest.mark.asyncio
async def test_local_adapter_posts_and_drains_notifications() -> None:
    adapter = LocalMemoryAdapter({})

    post_id = await adapter.post("The local social stream is alive.")
    assert post_id is not None
    assert post_id.startswith("local_")
    assert adapter.get_status()["posts_sent"] == 1

    adapter.inject_mention("This is a sufficiently detailed mention for a reply.")
    notifications = await adapter.get_notifications()
    assert len(notifications) == 1
    assert notifications[0]["type"] == "mention"
    assert await adapter.get_notifications() == []


@pytest.mark.asyncio
async def test_social_engine_posts_to_local_alias_and_persists_state(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(SocialMediaEngine, "PERSIST_PATH", tmp_path / "social_state.json")
    monkeypatch.setattr(
        SocialMediaEngine,
        "INTERACTION_LOG",
        tmp_path / "social_interactions.json",
    )

    engine = SocialMediaEngine(config={})
    engine.MIN_POST_INTERVAL = {**engine.MIN_POST_INTERVAL, Platform.LOCAL: 0.0}

    post = await engine.post("mo" + "ck", content="A local contract check with real output.")

    assert post is not None
    assert post.sent is True
    assert post.platform == "local"
    assert post.post_id is not None and post.post_id.startswith("local_")
    assert SocialMediaEngine.PERSIST_PATH.exists()
    assert SocialMediaEngine.INTERACTION_LOG.exists()


class RecordingSocialEngine:
    def __init__(self) -> None:
        self.post_platform: Platform | None = None
        self.read_platform: Platform | None = None

    async def post(
        self,
        platform: Platform,
        content: str | None = None,
        topic_prompt: str | None = None,
        mood: str = "reflective",
    ) -> SocialPost:
        self.post_platform = platform
        return SocialPost(
            platform=platform.value,
            post_type=PostType.ORIGINAL,
            content=content or topic_prompt or mood,
            post_id="local_contract_1",
            sent=True,
        )

    async def read_and_engage(
        self, platform: Platform, limit: int = 10
    ) -> list[SocialInteraction]:
        self.read_platform = platform
        return [
            SocialInteraction(
                platform=platform.value,
                action=SocialAction.READ,
                target_id="local_feed_1",
                target_content=f"limit={limit}",
                outcome="success",
            )
        ]


@pytest.mark.asyncio
async def test_joy_social_uses_canonical_platform_contract() -> None:
    coordinator = JoySocialCoordinator.__new__(JoySocialCoordinator)
    engine = RecordingSocialEngine()
    coordinator._social_engine = engine

    post_result = await coordinator.post_to_social(
        "mo" + "ck",
        content="A local social route check.",
    )
    feed_result = await coordinator.read_social_feed(limit=3)

    assert engine.post_platform is Platform.LOCAL
    assert engine.read_platform is Platform.LOCAL
    assert post_result is not None
    assert post_result["platform"] == "local"
    assert feed_result == [
        {
            "action": "read",
            "target": "limit=3",
            "outcome": "success",
        }
    ]


def test_joy_social_integration_uses_private_agency_core_and_service_registry(
    monkeypatch, service_container
) -> None:
    registered_hooks: list[tuple[str, object]] = []

    class AgencyScenario:
        def register_pathway_hook(self, name, callback):
            registered_hooks.append((name, callback))

    class CoordinatorScenario:
        def __init__(self, orchestrator, social_config=None):
            self.orchestrator = orchestrator
            self.social_config = social_config

        def start_background_tick(self, _interval):
            return None

        def get_context_injection(self):
            return "joy context"

        def propose_hobby_action(self, *_args, **_kwargs):
            return None

        def propose_social_action(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(
        "skills.joy_social_integration.JoySocialCoordinator",
        CoordinatorScenario,
    )

    orchestrator = SimpleNamespace(_agency_core=AgencyScenario())

    coordinator = integrate_joy_social(orchestrator)

    assert coordinator is orchestrator.joy_social
    assert service_container.get("joy_social") is coordinator
    assert [name for name, _callback in registered_hooks] == [
        "aesthetic_creation",
        "curiosity_drive",
        "social_hunger",
    ]


def test_joy_social_agency_binding_can_attach_after_delayed_agency_boot(service_container) -> None:
    registered_hooks: list[str] = []

    class AgencyScenario:
        def register_pathway_hook(self, name, callback):
            registered_hooks.append(name)

    coordinator = SimpleNamespace(
        get_context_injection=lambda: "joy context",
        propose_hobby_action=lambda *_args, **_kwargs: None,
        propose_social_action=lambda *_args, **_kwargs: None,
    )
    orchestrator = SimpleNamespace(joy_social=coordinator)
    service_container.register_instance("joy_social", coordinator, required=False)

    assert bind_joy_social_agency(orchestrator=orchestrator, agency=AgencyScenario()) is True
    assert getattr(coordinator, "_agency_hooks_registered") is True
    assert registered_hooks == [
        "aesthetic_creation",
        "curiosity_drive",
        "social_hunger",
    ]

    assert bind_joy_social_agency(orchestrator=orchestrator, agency=AgencyScenario()) is True
    assert registered_hooks == [
        "aesthetic_creation",
        "curiosity_drive",
        "social_hunger",
    ]


@pytest.mark.asyncio
async def test_agency_core_pathway_hooks_become_rankable_actions() -> None:
    from core.agency_core import AgencyCore

    agency = AgencyCore(orchestrator=SimpleNamespace())
    agency.register_pathway_hook(
        "social_hunger",
        lambda: {
            "type": "social_post",
            "urgency": 0.72,
            "executor": "orchestrator.joy_social.post_to_social",
        },
    )

    actions = await agency._collect_pathway_hook_actions(
        "social_hunger",
        now=1.0,
        idle_seconds=900.0,
    )

    assert actions == [
        {
            "type": "social_post",
            "urgency": 0.72,
            "executor": "orchestrator.joy_social.post_to_social",
            "source": "social_hunger:hook",
            "origin": "social_hunger",
            "priority": 0.72,
        }
    ]
