import os

import core.zenith_secrets as secrets


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


def test_zenith_secrets_source_has_no_subprocess_invocation():
    source = secrets.Path(secrets.__file__).read_text(encoding="utf-8")

    assert "subprocess." not in source
