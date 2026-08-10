"""The duplex route owns a microphone lease for exactly the socket lifetime."""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

from core.voice.microphone_authority import MicrophoneDenial, MicrophoneLease


class _Socket:
    client = SimpleNamespace(host="127.0.0.1")
    scope = {"headers": ()}
    query_params = {"mode": "focused"}

    def __init__(self) -> None:
        self.closed = asyncio.Event()
        self.text: list[str] = []
        self.close_code = 0
        self.close_reason = ""

    async def accept(self) -> None:
        return None

    async def receive(self):
        await self.closed.wait()
        return {"type": "websocket.disconnect"}

    async def send_text(self, value: str) -> None:
        self.text.append(value)

    async def send_bytes(self, _value: bytes) -> None:
        return None

    async def close(self, *, code: int, reason: str) -> None:
        self.close_code = code
        self.close_reason = reason
        self.closed.set()


def _local_auth(monkeypatch) -> None:
    from interface import auth

    monkeypatch.setattr(
        auth,
        "request_has_allowed_local_browser_origin",
        lambda _request: True,
    )
    monkeypatch.setattr(auth, "device_for_request", lambda _request: None)


def test_microphone_denial_happens_before_voice_models_are_constructed(monkeypatch):
    async def exercise() -> None:
        from interface.routes import voice_duplex

        _local_auth(monkeypatch)
        monkeypatch.setattr(voice_duplex, "_voice_input_permitted", lambda: True)
        monkeypatch.setattr(voice_duplex, "_voice_output_permitted", lambda: True)

        class Authority:
            def acquire(self, *_args, **_kwargs):
                return MicrophoneDenial(
                    "device_busy",
                    "another focused session owns the host microphone",
                    "end the other session",
                )

        monkeypatch.setattr(
            voice_duplex,
            "get_microphone_authority",
            lambda: Authority(),
        )
        monkeypatch.setattr(
            voice_duplex,
            "get_voice_model_runtime",
            lambda _config: (_ for _ in ()).throw(
                AssertionError("models constructed before microphone admission")
            ),
        )
        socket = _Socket()

        await voice_duplex.voice_duplex_endpoint(socket)

        assert socket.close_code == 4004
        payload = json.loads(socket.text[-1])
        assert payload["status"] == "microphone_device_busy"
        assert voice_duplex._SESSION_RESERVATIONS == {}

    asyncio.run(exercise())


def test_runtime_input_revocation_closes_an_open_duplex_session(monkeypatch):
    async def exercise() -> None:
        from interface.routes import voice_duplex

        _local_auth(monkeypatch)
        input_allowed = True
        released: list[str] = []
        lease = MicrophoneLease(
            lease_id="lease-1",
            holder="duplex",
            principal="owner:127.0.0.1",
            source="browser_duplex",
            group="host_microphone",
            mode="focused",
            session_id="session",
            generation=1,
            acquired_at=time.monotonic(),
            last_seen=time.monotonic(),
            preemptible=False,
        )

        class Authority:
            def acquire(self, *_args, **_kwargs):
                return lease

            def validate(self, value):
                return (value.active, "active" if value.active else value.revoked_reason)

            def heartbeat(self, _value):
                return True

            def release(self, value, *, reason):
                value._released = True
                value._revoked_reason = reason
                released.append(reason)
                return True

        authority = Authority()
        monkeypatch.setattr(
            voice_duplex,
            "get_microphone_authority",
            lambda: authority,
        )
        monkeypatch.setattr(
            voice_duplex,
            "get_audio_ingress_broker",
            lambda: SimpleNamespace(admit=lambda *_args: True),
        )
        monkeypatch.setattr(
            voice_duplex,
            "_voice_input_permitted",
            lambda: input_allowed,
        )
        monkeypatch.setattr(voice_duplex, "_voice_output_permitted", lambda: True)

        class Runtime:
            tts = object()

            @staticmethod
            def new_asr():
                return object()

        monkeypatch.setattr(voice_duplex, "get_voice_model_runtime", lambda _cfg: Runtime())

        class Session:
            def __init__(self, **_kwargs):
                return None

            async def start(self):
                nonlocal input_allowed
                input_allowed = False

            async def close(self):
                return None

            def status(self):
                return {"state": "idle"}

        monkeypatch.setattr(voice_duplex, "DuplexVoiceSession", Session)
        socket = _Socket()

        await asyncio.wait_for(voice_duplex.voice_duplex_endpoint(socket), timeout=2.0)

        assert socket.close_code == 4004
        assert socket.close_reason == "Voice disabled"
        assert any("voice_disabled_in_settings" in value for value in socket.text)
        assert "duplex_session_closed" in released
        assert voice_duplex._SESSION_RESERVATIONS == {}

    asyncio.run(exercise())
