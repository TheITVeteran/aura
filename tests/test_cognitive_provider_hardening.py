import asyncio

from core.providers.cognitive_provider import (
    ProxyLLMRouter,
    is_apple_silicon_host,
    register_cognitive_services,
)


class _Container:
    def __init__(self):
        self.factories = {}

    def register(self, name, factory, **kwargs):
        self.factories[name] = factory

    def has(self, name):
        return name in self.factories

    def get(self, name, default=None):
        factory = self.factories.get(name)
        if factory is None:
            return default
        return factory()


def test_proxy_router_is_explicit_degraded_router():
    router = ProxyLLMRouter()

    assert router.route("chat") is router
    assert asyncio.run(router.think("hello")) == "LLM unavailable in GUI Proxy Mode"
    assert asyncio.run(router.generate("hello")) == "LLM unavailable in GUI Proxy Mode"
    assert router.get_status()["status"] == "proxy"


def test_proxy_registration_uses_degraded_router_without_model_stack():
    container = _Container()

    register_cognitive_services(container, is_proxy=True)

    router = container.factories["llm_router"]()
    assert isinstance(router, ProxyLLMRouter)
    assert asyncio.run(router.think("hello")) == "LLM unavailable in GUI Proxy Mode"


def test_apple_silicon_probe_uses_reported_machine_first():
    calls = []

    def reader(name: str):
        calls.append(name)
        return 0

    assert is_apple_silicon_host(system_name="Darwin", machine_name="arm64", sysctl_reader=reader) is True
    assert calls == []


def test_apple_silicon_probe_checks_native_capability_for_translated_runtime():
    calls = []

    def reader(name: str):
        calls.append(name)
        return 1

    assert is_apple_silicon_host(system_name="Darwin", machine_name="x86_64", sysctl_reader=reader) is True
    assert calls == ["hw.optional.arm64"]


def test_apple_silicon_probe_handles_native_probe_failure():
    calls = []

    def reader(name: str):
        calls.append(name)
        raise OSError("sysctl unavailable")

    assert is_apple_silicon_host(system_name="Darwin", machine_name="x86_64", sysctl_reader=reader) is False
    assert calls == ["hw.optional.arm64"]


def test_apple_silicon_probe_rejects_non_darwin_hosts():
    assert is_apple_silicon_host(system_name="Linux", machine_name="arm64", sysctl_reader=lambda name: 1) is False
