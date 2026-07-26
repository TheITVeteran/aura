import os
import threading
import time

import pytest

import core.security.zenith_secrets as secrets


class FakeKeychain:
    def __init__(self):
        self.values = {}
        self.reads = []
        self.writes = []

    def get_password(self, service: str, account: str) -> str | None:
        self.reads.append((service, account))
        return self.values.get((service, account))

    def set_password(self, service: str, account: str, password: str) -> bool:
        self.writes.append((service, account, password))
        self.values[(service, account)] = password
        return True


def test_environment_secret_wins_without_touching_keychain(monkeypatch):
    fake = FakeKeychain()
    monkeypatch.setattr(secrets, "_KEYCHAIN_BACKEND", fake)
    monkeypatch.setenv("AURA_TEST_SECRET", "from-env")

    assert secrets.get_secret("AURA_TEST_SECRET") == "from-env"
    assert fake.reads == []


def test_keychain_backend_reads_and_writes_without_subprocess(monkeypatch):
    fake = FakeKeychain()
    monkeypatch.setattr(secrets, "_KEYCHAIN_BACKEND", fake)
    monkeypatch.delenv("AURA_TEST_SECRET", raising=False)

    secrets.set_secret("AURA_TEST_SECRET", "from-keychain")

    assert os.environ.get("AURA_TEST_SECRET") is None
    assert fake.writes == [(secrets._KEYCHAIN_SERVICE, "AURA_TEST_SECRET", "from-keychain")]
    assert secrets.get_secret("AURA_TEST_SECRET") == "from-keychain"
    assert fake.reads == [(secrets._KEYCHAIN_SERVICE, "AURA_TEST_SECRET")]


def test_keychain_unavailable_falls_back_to_environment(monkeypatch):
    monkeypatch.setattr(secrets, "_KEYCHAIN_BACKEND", secrets._KEYCHAIN_UNAVAILABLE)
    monkeypatch.delenv("AURA_TEST_SECRET", raising=False)

    secrets.set_secret("AURA_TEST_SECRET", "runtime-only")

    assert os.environ["AURA_TEST_SECRET"] == "runtime-only"


def test_strict_keychain_backend_never_falls_back_to_environment(monkeypatch):
    monkeypatch.setattr(secrets, "_KEYCHAIN_BACKEND", secrets._KEYCHAIN_UNAVAILABLE)
    monkeypatch.setenv("AURA_TEST_SECRET", "environment-is-not-custody")

    with pytest.raises(secrets.KeychainUnavailableError, match="strict macOS Keychain"):
        secrets.require_keychain_backend()


def test_native_keychain_is_never_opened_by_an_ordinary_test(monkeypatch):
    created = []

    class ForbiddenNativeKeychain:
        def __init__(self):
            created.append(True)
            raise AssertionError("ordinary tests must not touch the operator Keychain")

    monkeypatch.setattr(secrets, "_KEYCHAIN_BACKEND", None)
    monkeypatch.setattr(secrets, "_SecurityFrameworkKeychain", ForbiddenNativeKeychain)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_zenith_secrets.py::hermetic")
    monkeypatch.delenv("AURA_ALLOW_NATIVE_KEYCHAIN_IN_TESTS", raising=False)
    monkeypatch.delenv("AURA_HERMETIC_TEST_SECRET", raising=False)

    assert secrets.get_secret("AURA_HERMETIC_TEST_SECRET") is None
    assert created == []


def test_native_keychain_timeout_is_single_flight_and_does_not_hold_shutdown():
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class BlockingNativeKeychain:
        calls = 0

        def get_password(self, _service, _account):
            self.calls += 1
            started.set()
            try:
                release.wait(2.0)
                return "late-secret"
            finally:
                finished.set()

        def set_password(self, _service, _account, _password):
            raise AssertionError("write was not requested")

    native = BlockingNativeKeychain()
    backend = secrets._BoundedKeychainBackend(lambda: native, timeout_s=0.05)

    began = time.monotonic()
    try:
        with pytest.raises(secrets.KeychainUnavailableError, match="timed out"):
            backend.get_password("service", "account")
        assert time.monotonic() - began < 0.5
        assert started.is_set()

        with pytest.raises(secrets.KeychainUnavailableError, match="still in progress"):
            backend.get_password("service", "second-account")
        assert native.calls == 1
    finally:
        release.set()

    assert finished.wait(1.0)


def test_native_keychain_serializes_brief_concurrent_boot_reads():
    first_started = threading.Event()
    release_first = threading.Event()
    call_lock = threading.Lock()

    class BrieflyBlockingNativeKeychain:
        def __init__(self):
            self.calls = 0

        def get_password(self, _service, account):
            with call_lock:
                self.calls += 1
                call_number = self.calls
            if call_number == 1:
                first_started.set()
                release_first.wait(1.0)
            return f"secret:{account}"

        def set_password(self, _service, _account, _password):
            raise AssertionError("write was not requested")

    native = BrieflyBlockingNativeKeychain()
    backend = secrets._BoundedKeychainBackend(lambda: native, timeout_s=0.25)
    first_result = []

    first = threading.Thread(
        target=lambda: first_result.append(
            backend.get_password("service", "first-account")
        )
    )
    first.start()
    assert first_started.wait(0.5)
    timer = threading.Timer(0.03, release_first.set)
    timer.start()
    second = backend.get_password("service", "second-account")
    first.join(timeout=1.0)
    timer.cancel()

    assert not first.is_alive()
    assert first_result == ["secret:first-account"]
    assert second == "secret:second-account"
    assert native.calls == 2


def test_zenith_secrets_source_has_no_subprocess_invocation():
    source = secrets.Path(secrets.__file__).read_text(encoding="utf-8")

    assert "subprocess." not in source
