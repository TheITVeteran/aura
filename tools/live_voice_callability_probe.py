#!/usr/bin/env python3
"""Probe Aura's live voice callability path on a running server.

The probe verifies the same pathway a desktop user relies on:

* UI bootstrap honestly advertises server-side capture and STT availability.
* `/api/privacy/microphone` starts the real listener instead of only flipping UI state.
* Bootstrap reflects the active listener.
* The listener can be disabled and reports inactive afterward.

It does not fake a spoken wake word; that requires physical audio input. The
unit suite covers transcript fanout/wake-word routing, while this probe checks
the live dependency/device side that usually breaks on real machines.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class VoiceProbeStep:
    name: str
    ok: bool
    detail: str
    elapsed_s: float
    data: dict[str, Any] = field(default_factory=dict)


class LiveVoiceCallabilityProbe:
    def __init__(self, base_url: str, *, timeout_s: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.steps: list[VoiceProbeStep] = []
        self.headers: dict[str, str] = {}
        token = os.environ.get("AURA_API_TOKEN", "").strip()
        if token:
            self.headers["X-Api-Token"] = token

    async def run(self) -> int:
        async with httpx.AsyncClient(timeout=self.timeout_s, headers=self.headers) as client:
            self.client = client
            try:
                await self._step("voice_bootstrap_ready", self._voice_bootstrap_ready)
                await self._step("microphone_enable_starts_listener", self._microphone_enable_starts_listener)
                await self._step("microphone_disable_stops_listener", self._microphone_disable_stops_listener)
            finally:
                await self._best_effort_disable()

        self._print_summary()
        return 0 if all(step.ok for step in self.steps) else 1

    async def _step(self, name: str, fn) -> None:
        start = time.monotonic()
        try:
            detail, data = await fn()
            self.steps.append(VoiceProbeStep(name, True, detail, time.monotonic() - start, data or {}))
        except (AssertionError, httpx.HTTPError, OSError, RuntimeError, TypeError, ValueError) as exc:
            self.steps.append(VoiceProbeStep(name, False, f"{type(exc).__name__}: {exc}", time.monotonic() - start))

    async def _get(self, path: str) -> dict[str, Any]:
        response = await self.client.get(f"{self.base_url}{path}")
        response.raise_for_status()
        return response.json()

    async def _post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        response = await self.client.post(f"{self.base_url}{path}", json=payload or {})
        response.raise_for_status()
        return response.json()

    async def _voice_payload(self) -> dict[str, Any]:
        bootstrap = await self._get("/api/ui/bootstrap")
        voice = bootstrap.get("voice")
        if not isinstance(voice, dict):
            raise AssertionError(f"bootstrap missing voice payload: {bootstrap.keys()}")
        return voice

    async def _voice_bootstrap_ready(self) -> tuple[str, dict[str, Any]]:
        voice = await self._voice_payload()
        required_true = ("available", "streaming_available", "server_capture", "capture_available", "stt_available")
        missing = [key for key in required_true if not bool(voice.get(key))]
        if missing:
            raise AssertionError(f"voice bootstrap missing live prerequisites {missing}: {voice}")
        if "listening" not in voice:
            raise AssertionError(f"voice bootstrap does not expose listener state: {voice}")
        return "voice bootstrap advertises capture, STT, streaming, and listener state", voice

    async def _microphone_enable_starts_listener(self) -> tuple[str, dict[str, Any]]:
        result = await self._post("/api/privacy/microphone", {"enabled": True})
        if not result.get("ok"):
            raise AssertionError(f"microphone enable failed: {result}")
        if not result.get("enabled") or not result.get("microphone_enabled"):
            raise AssertionError(f"microphone enable did not keep input enabled: {result}")
        if not result.get("listening"):
            raise AssertionError(f"microphone enable did not start listener: {result}")
        voice = await self._voice_payload()
        if not voice.get("listening"):
            raise AssertionError(f"bootstrap did not reflect active listener after enable: {voice}")
        if not voice.get("stt_initialized"):
            raise AssertionError(f"listener active but STT not initialized: {voice}")
        return "microphone enable started the live STT listener and bootstrap reflected it", {
            "privacy_result": result,
            "voice": voice,
        }

    async def _microphone_disable_stops_listener(self) -> tuple[str, dict[str, Any]]:
        result = await self._post("/api/privacy/microphone", {"enabled": False})
        if not result.get("ok"):
            raise AssertionError(f"microphone disable failed: {result}")
        if result.get("enabled") or result.get("microphone_enabled") or result.get("listening"):
            raise AssertionError(f"microphone disable did not clear input/listener state: {result}")
        voice = await self._voice_payload()
        if voice.get("microphone_enabled") or voice.get("listening"):
            raise AssertionError(f"bootstrap did not reflect inactive listener after disable: {voice}")
        return "microphone disable stopped the listener and bootstrap reflected it", {
            "privacy_result": result,
            "voice": voice,
        }

    async def _best_effort_disable(self) -> None:
        try:
            await self._post("/api/privacy/microphone", {"enabled": False})
        except (httpx.HTTPError, OSError, RuntimeError, TypeError, ValueError):
            return

    def _print_summary(self) -> None:
        print("\nLIVE VOICE CALLABILITY PROBE SUMMARY")
        print("=" * 72)
        for step in self.steps:
            mark = "PASS" if step.ok else "FAIL"
            print(f"[{mark}] {step.name} ({step.elapsed_s:.1f}s): {step.detail}")
            if not step.ok:
                continue
            print(json.dumps(step.data, indent=2, sort_keys=True)[:1600])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    return asyncio.run(LiveVoiceCallabilityProbe(args.base_url, timeout_s=args.timeout).run())


if __name__ == "__main__":
    raise SystemExit(main())
