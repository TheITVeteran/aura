"""Wake-word → canonical conversation lane unification.

Voice is a surface, not a separate runtime: a spoken command must enter
the same governed /api/chat lane the desktop UI uses. These tests pin:

1. Single-utterance capture: "Hey Aura, open my notes" arrives as ONE
   transcript chunk; the command portion must be seeded immediately
   (the transcript dedup means the chunk is never seen again).
2. Dispatch goes to the loopback /api/chat surface through the network
   gateway with the voice surface header — not to MissionState.
3. Barge-in stays live: execution runs as a background task the
   detection loop can cancel on "stop"/"cancel".
4. Lane failures degrade without crashing the detector.
"""
from __future__ import annotations

import asyncio
import json
import unittest
from unittest import mock

from core.voice.wake_word import WakeState, WakeWordDetector


def _no_services(name, default=None):
    return default


class _GatewayRecorder:
    """Stands in for the network gateway; records the request it receives."""

    def __init__(self, *, status_code=200, payload=None, ok=True, error=None):
        self.calls = []
        self._status_code = status_code
        self._payload = payload if payload is not None else {"response": "Done."}
        self._ok = ok
        self._error = error

    async def request_async(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        result = {
            "status_code": self._status_code,
            "headers": {},
            "content": json.dumps(self._payload).encode("utf-8"),
            "ok": self._ok,
        }
        if self._error:
            result["error"] = self._error
        return result


class SingleUtteranceCaptureTest(unittest.TestCase):
    def test_command_in_wake_chunk_is_seeded(self):
        detector = WakeWordDetector()
        with mock.patch(
            "core.voice.wake_word.ServiceContainer.get",
            staticmethod(_no_services),
        ):
            asyncio.run(detector._check_wake_word("Hey Aura, open my notes app"))
        self.assertEqual(detector.state, WakeState.LISTENING)
        self.assertEqual(detector._accumulated_transcript, "open my notes app")

    def test_bare_wake_word_seeds_empty_command(self):
        detector = WakeWordDetector()
        with mock.patch(
            "core.voice.wake_word.ServiceContainer.get",
            staticmethod(_no_services),
        ):
            asyncio.run(detector._check_wake_word("hey aura"))
        self.assertEqual(detector.state, WakeState.LISTENING)
        self.assertEqual(detector._accumulated_transcript, "")

    def test_single_utterance_dispatches_after_silence(self):
        """Wake chunk with command + silence → the command is dispatched."""
        detector = WakeWordDetector()
        detector.SILENCE_TIMEOUT_S = 0.0  # silence elapses immediately
        dispatched = []

        async def fake_process(command):
            dispatched.append(command)
            detector.state = WakeState.IDLE

        detector._process_command = fake_process

        async def scenario():
            with mock.patch(
                "core.voice.wake_word.ServiceContainer.get",
                staticmethod(_no_services),
            ):
                await detector._check_wake_word("Hey Aura, write a journal entry")
                await detector._accumulate_command("")

        asyncio.run(scenario())
        self.assertEqual(dispatched, ["write a journal entry"])


class ConversationLaneDispatchTest(unittest.TestCase):
    def _dispatch(self, detector, gateway, command="open my notes"):
        with mock.patch(
            "core.runtime.network_gateway.get_network_gateway",
            return_value=gateway,
        ), mock.patch.dict("os.environ", {"AURA_SERVER_PORT": "8123"}):
            return asyncio.run(detector._dispatch_to_conversation_lane(command))

    def test_dispatch_posts_to_loopback_chat_lane(self):
        detector = WakeWordDetector()
        gateway = _GatewayRecorder(payload={"response": "Notes is open."})
        ok, reply = self._dispatch(detector, gateway)

        self.assertTrue(ok)
        self.assertEqual(reply, "Notes is open.")
        self.assertEqual(len(gateway.calls), 1)
        call = gateway.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["url"], "http://127.0.0.1:8123/api/chat")
        self.assertEqual(call["headers"]["X-Aura-Surface"], "voice")
        body = json.loads(call["data"].decode("utf-8"))
        self.assertEqual(body["message"], "open my notes")
        self.assertEqual(body["session_id"], "voice-wake")
        self.assertEqual(call["source"], "wake_word:conversation_lane")

    def test_dispatch_failure_is_reported_not_raised(self):
        detector = WakeWordDetector()
        gateway = _GatewayRecorder(status_code=0, ok=False, error="connection refused")
        ok, reply = self._dispatch(detector, gateway)
        self.assertFalse(ok)
        self.assertIn("conversation lane dispatch failed", reply)

    def test_dispatch_unreadable_payload_is_reported(self):
        detector = WakeWordDetector()
        gateway = _GatewayRecorder()

        async def bad_payload(method, url, **kwargs):
            return {"status_code": 200, "ok": True, "content": b"\xff not json"}

        gateway.request_async = bad_payload
        ok, reply = self._dispatch(detector, gateway)
        self.assertFalse(ok)
        self.assertIn("unreadable payload", reply)

    def test_process_command_does_not_touch_mission_state(self):
        """The MissionState parallel path is gone — only the chat lane runs."""
        detector = WakeWordDetector()
        service_names = []

        def recording_get(name, default=None):
            service_names.append(name)
            return default

        async def scenario():
            with mock.patch(
                "core.voice.wake_word.ServiceContainer.get",
                staticmethod(recording_get),
            ), mock.patch.object(
                detector,
                "_dispatch_to_conversation_lane",
                mock.AsyncMock(return_value=(True, "done")),
            ):
                await detector._process_command("open my notes")
                await asyncio.wait_for(detector._dispatch_task, timeout=5.0)

        asyncio.run(scenario())
        self.assertNotIn("mission_state", service_names)
        self.assertNotIn("initiative_synthesizer", service_names)
        self.assertEqual(detector.state, WakeState.IDLE)


class BargeInTest(unittest.TestCase):
    def test_interrupt_cancels_inflight_dispatch(self):
        detector = WakeWordDetector()
        started = asyncio.Event()
        cancelled = []

        async def slow_dispatch(command):
            started.set()
            try:
                await asyncio.sleep(30.0)
            except asyncio.CancelledError:
                cancelled.append(command)
                raise
            return True, "never"

        async def scenario():
            with mock.patch(
                "core.voice.wake_word.ServiceContainer.get",
                staticmethod(_no_services),
            ), mock.patch.object(
                detector, "_dispatch_to_conversation_lane", slow_dispatch
            ):
                await detector._process_command("open my notes")
                await asyncio.wait_for(started.wait(), timeout=5.0)
                self.assertEqual(detector.state, WakeState.EXECUTING)
                await detector._handle_interrupt()
                with self.assertRaises(asyncio.CancelledError):
                    await detector._dispatch_task

        asyncio.run(scenario())
        self.assertEqual(cancelled, ["open my notes"])
        self.assertEqual(detector.state, WakeState.IDLE)

    def test_stop_cancels_inflight_dispatch(self):
        detector = WakeWordDetector()
        started = asyncio.Event()

        async def slow_dispatch(command):
            started.set()
            await asyncio.sleep(30.0)
            return True, "never"

        async def scenario():
            with mock.patch(
                "core.voice.wake_word.ServiceContainer.get",
                staticmethod(_no_services),
            ), mock.patch.object(
                detector, "_dispatch_to_conversation_lane", slow_dispatch
            ):
                await detector._process_command("open my notes")
                await asyncio.wait_for(started.wait(), timeout=5.0)
                await detector.stop()

        asyncio.run(scenario())
        self.assertTrue(detector._dispatch_task.cancelled())


class SpokenReplyTest(unittest.TestCase):
    def test_reply_spoken_through_voice_engine(self):
        detector = WakeWordDetector()
        spoken = []

        class Voice:
            async def speak(self, text):
                spoken.append(text)

        def services(name, default=None):
            if name == "voice_engine":
                return Voice()
            return default

        async def scenario():
            with mock.patch(
                "core.voice.wake_word.ServiceContainer.get",
                staticmethod(services),
            ):
                await detector._speak_reply("Notes is open.")

        asyncio.run(scenario())
        self.assertEqual(spoken, ["Notes is open."])

    def test_long_reply_is_bounded_at_sentence_boundary(self):
        detector = WakeWordDetector()
        spoken = []

        class Voice:
            async def speak(self, text):
                spoken.append(text)

        def services(name, default=None):
            if name == "voice_engine":
                return Voice()
            return default

        long_reply = ("I did the first step. " * 40).strip()  # ~880 chars

        async def scenario():
            with mock.patch(
                "core.voice.wake_word.ServiceContainer.get",
                staticmethod(services),
            ):
                await detector._speak_reply(long_reply)

        asyncio.run(scenario())
        self.assertEqual(len(spoken), 1)
        self.assertLessEqual(len(spoken[0]), detector.SPOKEN_REPLY_CHAR_BUDGET)
        self.assertTrue(spoken[0].endswith("."))

    def test_missing_voice_engine_degrades_silently(self):
        detector = WakeWordDetector()
        with mock.patch(
            "core.voice.wake_word.ServiceContainer.get",
            staticmethod(_no_services),
        ):
            asyncio.run(detector._speak_reply("Hello."))  # must not raise


if __name__ == "__main__":
    unittest.main()
