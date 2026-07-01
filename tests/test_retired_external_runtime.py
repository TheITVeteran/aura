import pytest

from core.brain.llm.retired_external_runtime import (
    ExternalLocalRuntimeRetiredError,
    RetiredExternalRuntimeClient,
    get_retired_external_runtime_client,
)


def test_retired_external_runtime_factory_raises():
    with pytest.raises(ExternalLocalRuntimeRetiredError, match="external_local_runtime_retired"):
        get_retired_external_runtime_client(model_path="/models/old-runtime")


def test_retired_external_runtime_spawn_raises():
    client = RetiredExternalRuntimeClient("/models/old-runtime")

    with pytest.raises(ExternalLocalRuntimeRetiredError, match="external_local_runtime_retired"):
        client._spawn_server_blocking()


@pytest.mark.asyncio
async def test_retired_external_runtime_generation_raises():
    client = RetiredExternalRuntimeClient("/models/old-runtime")

    with pytest.raises(ExternalLocalRuntimeRetiredError, match="external_local_runtime_retired"):
        await client.generate_text_async("hello")


def test_retired_external_runtime_is_never_conversation_ready():
    status = RetiredExternalRuntimeClient("/models/old-runtime").get_lane_status()

    assert status["state"] == "retired"
    assert status["conversation_ready"] is False
    assert "external_local_runtime_retired" in status["readiness_blockers"]
