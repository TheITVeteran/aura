import asyncio
import contextlib
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from core.brain.llm.mlx_client import MLXLocalClient
from core.brain.llm.mlx_vision_client import MLXVisionClient
from core.brain.llm.mlx_worker import (
    IPCWriterThread,
    _apply_surface_generation_controls,
    _build_operator_evidence_prompt,
    _merge_stop_sequences,
    _operator_evidence_fragment_incomplete,
    _prompt_cache_entry_budget_for_model,
    _restore_surface_generation_controls,
    _should_emit_generation_progress,
    _trim_complete_operator_evidence,
)
from core.utils.deadlines import get_deadline

TMP_ROOT = Path(tempfile.gettempdir())
QWEN32_MODEL = str(TMP_ROOT / "Qwen2.5-32B-Instruct-8bit")
TEST_MODEL = str(TMP_ROOT / "test-model")


class TestMLXClientResilience(unittest.IsolatedAsyncioTestCase):
    class _FakeQueue:
        def __init__(self):
            self.closed = False
            self.joined = False

        def empty(self):
            return True

        def close(self):
            self.closed = True

        def join_thread(self):
            self.joined = True

    def _attach_local_ipc_queues(self, client):
        client._req_q = queue.Queue()
        client._res_q = queue.Queue()

    def test_close_releases_worker_and_ipc_queues(self):
        client = MLXLocalClient(model_path=TEST_MODEL)
        req_q = self._FakeQueue()
        res_q = self._FakeQueue()
        proc = MagicMock()
        proc.is_alive.return_value = True
        client._req_q = req_q
        client._res_q = res_q
        client._process = proc
        client._init_done = True

        client.close()

        proc.kill.assert_called_once()
        proc.join.assert_called_once_with(timeout=2.0)
        self.assertTrue(req_q.closed)
        self.assertTrue(req_q.joined)
        self.assertTrue(res_q.closed)
        self.assertTrue(res_q.joined)
        self.assertIsNone(client._req_q)
        self.assertIsNone(client._res_q)
        self.assertFalse(client._init_done)
        self.assertEqual(client.get_lane_status()["state"], "closed")

    def test_client_constructor_defers_ipc_queue_allocation(self):
        client = MLXLocalClient(model_path=TEST_MODEL)

        self.assertIsNone(client._req_q)
        self.assertIsNone(client._res_q)
        self.assertFalse(hasattr(client._substrate_mem, "get_lock"))
        self.assertFalse(hasattr(client._steering_active, "get_lock"))

    def test_replace_ipc_queues_closes_previous_queues_before_recreation(self):
        client = MLXLocalClient(model_path=TEST_MODEL)
        old_req_q = self._FakeQueue()
        old_res_q = self._FakeQueue()
        new_req_q = self._FakeQueue()
        new_res_q = self._FakeQueue()
        client._req_q = old_req_q
        client._res_q = old_res_q
        client._mp_context = MagicMock()
        client._mp_context.Queue.side_effect = [new_req_q, new_res_q]

        client._replace_ipc_queues()

        self.assertTrue(old_req_q.closed)
        self.assertTrue(old_req_q.joined)
        self.assertTrue(old_res_q.closed)
        self.assertTrue(old_res_q.joined)
        self.assertIs(client._req_q, new_req_q)
        self.assertIs(client._res_q, new_res_q)
        self.assertFalse(client._closed)

    def test_vision_client_releases_worker_and_ipc_queues(self):
        client = MLXVisionClient(model_path=TEST_MODEL)
        req_q = self._FakeQueue()
        res_q = self._FakeQueue()
        proc = MagicMock()
        proc.is_alive.return_value = False
        client._req_q = req_q
        client._res_q = res_q
        client._process = proc
        client._init_done = True

        client.stop()

        proc.join.assert_called_with(timeout=3.0)
        self.assertTrue(req_q.closed)
        self.assertTrue(req_q.joined)
        self.assertTrue(res_q.closed)
        self.assertTrue(res_q.joined)
        self.assertIsNone(client._req_q)
        self.assertIsNone(client._res_q)
        self.assertFalse(client._init_done)

    async def test_worker_stop_sequences_are_role_boundary_safe(self):
        stops = _merge_stop_sequences(["Assistant:", "Aura:", "user:", "\nHuman:"])

        self.assertNotIn("Assistant:", stops)
        self.assertNotIn("Aura:", stops)
        self.assertNotIn("user:", stops)
        self.assertIn("\nAssistant:", stops)
        self.assertIn("\nuser:", stops)
        self.assertIn("\nHuman:", stops)

    async def test_operator_evidence_contract_prompt_is_complete_and_bounded(self):
        prompt, prefix = _build_operator_evidence_prompt(
            [
                {"role": "system", "content": "Return one paragraph."},
                {
                    "role": "user",
                    "content": (
                        "What objective, governed tool use, receipt, trace, stop condition, "
                        "and personhood boundary should Aura use?"
                    ),
                },
            ],
            "",
        )

        self.assertIn("bounded software-operator evidence lane", prompt)
        self.assertIn("objective", prompt)
        self.assertIn("personhood boundary", prompt)
        self.assertEqual(
            prefix,
            "Operationally, Aura should set an objective, use governed tool actions, "
            "keep each receipt and trace, stop when blocked or unsafe, and treat the "
            "result as evidence of bounded software operation rather than personhood proof. ",
        )
        self.assertFalse(
            _operator_evidence_fragment_incomplete(
                "Aura should pursue a bounded objective, use governed tool calls with a "
                "receipt and trace, stop when governance or evidence fails, and treat "
                "that as operational evidence rather than proof of literal personhood."
            )
        )
        self.assertTrue(
            _operator_evidence_fragment_incomplete(
                "I feel like a person who chooses things in a shining field."
            )
        )
        self.assertTrue(
            _operator_evidence_fragment_incomplete(
                "Aura should pursue a bounded objective, use governed tool calls with a "
                "receipt and trace, stop when governance or evidence fails, and treat "
                "that as operational evidence rather than proof of literal personhood. "
                "That's one paragraph as requested."
            )
        )

    async def test_operator_evidence_trims_clipped_tail_to_complete_sentences(self):
        clipped = (
            "Operationally, Aura should set an objective, use governed tool actions, "
            "keep each receipt and trace, stop when blocked or unsafe, and treat the "
            "result as evidence of bounded software operation rather than personhood proof. "
            "Receipts and traces show tool use was governed. Stopping when blocked or "
            "unsafe shows boundedness. The result is evidence of software ope"
        )

        trimmed = _trim_complete_operator_evidence(clipped)

        self.assertEqual(
            trimmed,
            "Operationally, Aura should set an objective, use governed tool actions, "
            "keep each receipt and trace, stop when blocked or unsafe, and treat the "
            "result as evidence of bounded software operation rather than personhood proof. "
            "Receipts and traces show tool use was governed. Stopping when blocked or "
            "unsafe shows boundedness.",
        )
        self.assertFalse(_operator_evidence_fragment_incomplete(trimmed))

    async def test_surface_generation_controls_clamp_steering_and_recurrent_depth(self):
        class _Hook:
            _alpha = 5.0

        class _Engine:
            _alpha = 5.0
            _surface_alpha_override = None

            def __init__(self):
                self._hooks = [_Hook()]

            def set_surface_alpha_override(self, alpha):
                self._surface_alpha_override = alpha
                if alpha is not None:
                    for hook in self._hooks:
                        hook._alpha = min(hook._alpha, alpha)

        class _Inner:
            _recurrent_depth_config = {"n_loops": 2}

        class _Model:
            model = _Inner()

        engine = _Engine()
        state = _apply_surface_generation_controls(
            engine,
            _Model(),
            {"clean_user_surface_contract": True},
        )

        self.assertLessEqual(engine._surface_alpha_override, 0.35)
        self.assertLessEqual(engine._hooks[0]._alpha, 0.35)
        self.assertEqual(_Model.model._recurrent_depth_runtime_loops, 1)

        _restore_surface_generation_controls(state)

        self.assertIsNone(engine._surface_alpha_override)
        self.assertFalse(hasattr(_Model.model, "_recurrent_depth_runtime_loops"))

    async def test_foreground_request_lock_timeout_is_bounded_for_live_chat(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)

        self.assertEqual(
            client._request_lock_timeout(deadline=None, foreground_request=True),
            12.0,
        )

    async def test_foreground_request_lock_timeout_preempts_wedged_holder(self):
        from core.brain.llm.mlx_client import _new_shared_future

        client = MLXLocalClient(model_path=QWEN32_MODEL)
        stuck = _new_shared_future()
        client._request_lock.acquire()
        client._request_lock_owner_label = "Cortex"
        client._request_lock_acquired_at = time.time() - 2.0
        client._current_gen_future = stuck
        client._last_heartbeat = 0.0

        try:
            with patch.object(client, "_request_lock_timeout", return_value=0.05), patch.object(
                client, "_first_token_sla", return_value=0.01
            ):
                acquired = await client._acquire_request_lock(
                    owner_label="live_chat",
                    deadline=None,
                    foreground_request=True,
                )

            self.assertFalse(acquired)
            self.assertEqual(
                client._deferred_reboot_reason,
                "foreground_preemption_wedged_holder",
            )
            self.assertTrue(stuck.done())
        finally:
            client._current_gen_future = None
            with contextlib.suppress(RuntimeError):
                client._request_lock.release()

    async def test_worker_sanitizer_finishes_current_request_for_caller_recovery(self):
        worker_source = await asyncio.to_thread(
            Path("core/brain/llm/mlx_worker.py").read_text,
            encoding="utf-8",
        )
        marker = "Hallucination detected by sanitizer."
        start = worker_source.index(marker)
        end = worker_source.index('ipc_writer.put({', start)
        sanitizer_block = worker_source[start:end]

        self.assertIn("Returning empty text for caller-side recovery.", sanitizer_block)
        self.assertNotIn("continue", sanitizer_block)

    async def test_heavy_model_hotswap_reboots_other_heavy_client_before_spawn(self):
        import core.brain.llm.mlx_client as mlx_module

        primary_path = "/models/32B"
        deep_path = "/models/72B"

        primary = MLXLocalClient(model_path=primary_path)
        solver = MLXLocalClient(model_path=deep_path)

        primary_proc = MagicMock()
        primary_proc.is_alive.return_value = True
        primary._process = primary_proc
        primary._init_done = True
        primary.reboot_worker = AsyncMock()

        solver_proc = MagicMock()
        solver_proc.is_alive.return_value = True

        async def _spawn_solver():
            solver._init_future.set_result({"status": "ok", "action": "init"})
            return solver_proc

        old_clients = dict(mlx_module._CLIENTS)
        old_last_heavy = mlx_module._GLOBAL_LAST_HEAVY_MODEL
        old_last_swap = mlx_module._GLOBAL_LAST_SWAP_TIME
        mlx_module._CLIENTS = {
            primary_path: primary,
            deep_path: solver,
        }
        mlx_module._GLOBAL_LAST_HEAVY_MODEL = ""
        mlx_module._GLOBAL_LAST_SWAP_TIME = 0.0

        try:
            with patch("core.brain.llm.model_registry.ACTIVE_MODEL", "Qwen2.5-32B-Instruct-8bit"), \
                 patch("core.brain.llm.model_registry.DEEP_MODEL", "Qwen2.5-72B-Instruct-4bit"), \
                 patch("core.brain.llm.model_registry.get_model_path", side_effect=lambda name=None: primary_path if "32B" in str(name) or name is None else deep_path), \
                 patch("core.brain.llm.mlx_client.os.path.realpath", side_effect=lambda path: path), \
                 patch.object(solver, "_spawn_worker", side_effect=_spawn_solver):
                await solver._ensure_worker_alive()
        finally:
            mlx_module._CLIENTS = old_clients
            mlx_module._GLOBAL_LAST_HEAVY_MODEL = old_last_heavy
            mlx_module._GLOBAL_LAST_SWAP_TIME = old_last_swap

        primary.reboot_worker.assert_awaited_once()
        self.assertTrue(solver._init_done)

    async def test_ensure_worker_sets_init_future_before_spawn(self):
        client = MLXLocalClient(model_path=TEST_MODEL)

        async def spawn_side_effect():
            self.assertIsNotNone(client._init_future)
            self.assertFalse(client._init_future.done())
            client._init_future.set_result({"status": "ok", "action": "init"})
            proc = MagicMock()
            proc.is_alive.return_value = True
            return proc

        with patch.object(client, "_spawn_worker", side_effect=spawn_side_effect):
            await client._ensure_worker_alive()

        self.assertTrue(client._init_done)
        self.assertTrue(client.is_alive())

    async def test_ensure_worker_reuses_existing_handshake_future(self):
        client = MLXLocalClient(model_path=TEST_MODEL)

        live_process = MagicMock()
        live_process.is_alive.return_value = True
        client._process = live_process
        client._init_done = False
        client._init_future = AsyncMock()
        # Replace the async mock with a real Future to match runtime behavior.
        import asyncio
        real_future = asyncio.get_running_loop().create_future()
        real_future.set_result({"status": "ok", "action": "init"})
        client._init_future = real_future

        with patch.object(client, "_spawn_worker", new=AsyncMock()) as spawn_mock:
            await client._ensure_worker_alive()

        spawn_mock.assert_not_awaited()
        live_process.kill.assert_not_called()
        self.assertTrue(client._init_done)

    async def test_ensure_worker_reuses_cross_loop_handshake_future(self):
        client = MLXLocalClient(model_path=TEST_MODEL)

        live_process = MagicMock()
        live_process.is_alive.return_value = True
        client._process = live_process
        client._init_done = False

        holder = {}
        ready = threading.Event()

        def _loop_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            future = loop.create_future()
            holder["future"] = future
            ready.set()

            async def _complete():
                await asyncio.sleep(0.05)
                future.set_result({"status": "ok", "action": "init"})
                await asyncio.sleep(0.05)

            loop.run_until_complete(_complete())
            loop.close()

        thread = threading.Thread(target=_loop_thread, name="mlx-cross-loop-init", daemon=True)
        thread.start()
        ready.wait(timeout=1.0)
        client._init_future = holder["future"]

        try:
            with patch.object(client, "_spawn_worker", new=AsyncMock()) as spawn_mock:
                await client._ensure_worker_alive()
        finally:
            thread.join(timeout=1.0)

        spawn_mock.assert_not_awaited()
        self.assertTrue(client._init_done)
        live_process.kill.assert_not_called()

    async def test_cancelled_generation_preserves_healthy_worker(self):
        client = MLXLocalClient(model_path=TEST_MODEL)
        proc = MagicMock()
        proc.is_alive.return_value = True
        client._process = proc
        client._init_done = True
        self._attach_local_ipc_queues(client)
        client._set_lane_state("ready")
        client._last_heartbeat = client._last_progress_at = client._last_ready_at = 10_000.0

        async def _cancelled(*args, **kwargs):
            raise asyncio.CancelledError

        with patch.object(client, "_ensure_worker_alive", new=AsyncMock(return_value=True)):
            with patch.object(client, "_wait_for_generation_result", side_effect=_cancelled):
                with patch.object(client, "reboot_worker", new=AsyncMock()) as reboot_mock:
                    with patch("time.time", return_value=10_001.0):
                        with self.assertRaises(asyncio.CancelledError):
                            await client._generate_inner("hello", foreground_request=True)

        reboot_mock.assert_not_awaited()

    async def test_expected_cancelled_generation_does_not_mark_worker_unhealthy(self):
        client = MLXLocalClient(model_path=TEST_MODEL)
        self._attach_local_ipc_queues(client)
        client._expected_cancel_reason = "yield_to_Qwen2.5-72B-Instruct-4bit"
        client._expected_cancel_budget = 1
        client._expected_cancel_recorded_at = 10_000.0

        async def _cancelled(*args, **kwargs):
            raise asyncio.CancelledError

        with patch.object(client, "_ensure_worker_alive", new=AsyncMock(return_value=True)):
            with patch.object(client, "_wait_for_generation_result", side_effect=_cancelled):
                with patch("time.time", return_value=10_001.0):
                    with self.assertRaises(asyncio.CancelledError):
                        await client._generate_inner("hello", foreground_request=True)

        self.assertIsNone(client._deferred_reboot_reason)
        self.assertEqual(client._expected_cancel_budget, 0)

    async def test_generate_times_out_waiting_for_foreground_owner(self):
        import core.brain.llm.mlx_client as mlx_module

        client = MLXLocalClient(model_path=QWEN32_MODEL)
        old_owner = mlx_module._FOREGROUND_OWNER_NAME
        old_owned_at = mlx_module._FOREGROUND_OWNER_ACQUIRED_AT
        mlx_module._FOREGROUND_OWNER_NAME = "warmup:cortex"
        mlx_module._FOREGROUND_OWNER_ACQUIRED_AT = time.time()

        try:
            with patch.object(client, "_acquire_request_lock", new=AsyncMock(return_value=True)):
                with patch.object(client, "_generate_inner", new=AsyncMock()) as inner:
                    with patch("core.brain.llm.mlx_client._foreground_owner_wait_budget", return_value=0.0):
                        result = await client.generate(
                            "hello",
                            foreground_request=True,
                            owner_label="test",
                            deadline=get_deadline(30.0),
                        )
        finally:
            mlx_module._FOREGROUND_OWNER_NAME = old_owner
            mlx_module._FOREGROUND_OWNER_ACQUIRED_AT = old_owned_at

        self.assertIsNone(result)
        inner.assert_not_awaited()

    async def test_generate_clears_stale_foreground_owner_and_continues(self):
        import core.brain.llm.mlx_client as mlx_module

        client = MLXLocalClient(model_path=QWEN32_MODEL)
        old_owner = mlx_module._FOREGROUND_OWNER_NAME
        old_owned_at = mlx_module._FOREGROUND_OWNER_ACQUIRED_AT
        mlx_module._FOREGROUND_OWNER_NAME = "warmup:cortex"
        mlx_module._FOREGROUND_OWNER_ACQUIRED_AT = time.time() - 120.0

        try:
            with patch.object(client, "_acquire_request_lock", new=AsyncMock(return_value=True)):
                with patch.object(client, "_generate_inner", new=AsyncMock(return_value="ok")) as inner:
                    result = await client.generate(
                        "hello",
                        foreground_request=True,
                        owner_label="test",
                        deadline=get_deadline(30.0),
                    )
        finally:
            mlx_module._FOREGROUND_OWNER_NAME = old_owner
            mlx_module._FOREGROUND_OWNER_ACQUIRED_AT = old_owned_at

        self.assertEqual(result, "ok")
        inner.assert_awaited_once()

    async def test_foreground_generate_reserves_owner_before_request_lock(self):
        import core.brain.llm.mlx_client as mlx_module

        client = MLXLocalClient(model_path=QWEN32_MODEL)
        old_owner = mlx_module._FOREGROUND_OWNER_NAME
        old_owned_at = mlx_module._FOREGROUND_OWNER_ACQUIRED_AT
        observed_owner = []

        async def _acquire(*_args, **_kwargs):
            observed_owner.append(mlx_module._FOREGROUND_OWNER_NAME)
            return True

        try:
            mlx_module._FOREGROUND_OWNER_NAME = None
            mlx_module._FOREGROUND_OWNER_ACQUIRED_AT = 0.0
            with patch.object(client, "_acquire_request_lock", side_effect=_acquire):
                with patch.object(client, "_generate_inner", new=AsyncMock(return_value="ok")):
                    result = await client.generate(
                        "hello",
                        foreground_request=True,
                        owner_label="live_user",
                        deadline=get_deadline(30.0),
                    )
        finally:
            mlx_module._FOREGROUND_OWNER_NAME = old_owner
            mlx_module._FOREGROUND_OWNER_ACQUIRED_AT = old_owned_at

        self.assertEqual(result, "ok")
        self.assertEqual(observed_owner, ["live_user"])

    async def test_reboot_worker_clears_matching_warmup_owner(self):
        import core.brain.llm.mlx_client as mlx_module

        client = MLXLocalClient(model_path=QWEN32_MODEL)
        old_owner = mlx_module._FOREGROUND_OWNER_NAME
        old_owned_at = mlx_module._FOREGROUND_OWNER_ACQUIRED_AT
        mlx_module._FOREGROUND_OWNER_NAME = "warmup:Qwen2.5-32B-Instruct-8bit"
        mlx_module._FOREGROUND_OWNER_ACQUIRED_AT = time.time() - 20.0

        try:
            await client.reboot_worker(reason="yield_to_solver", mark_failed=False)
            self.assertIsNone(mlx_module._FOREGROUND_OWNER_NAME)
            self.assertEqual(mlx_module._FOREGROUND_OWNER_ACQUIRED_AT, 0.0)
        finally:
            mlx_module._FOREGROUND_OWNER_NAME = old_owner
            mlx_module._FOREGROUND_OWNER_ACQUIRED_AT = old_owned_at


    async def test_primary_lane_generate_requires_explicit_foreground_request(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)

        with patch.object(client, "_generate_inner", new=AsyncMock(return_value="ok")) as inner:
            result = await client.generate("hello")

        self.assertEqual(result, "ok")
        self.assertFalse(inner.await_args.kwargs["foreground_request"])
        self.assertFalse(inner.await_args.kwargs["request_is_background"])

    async def test_generate_suppresses_stale_unlock_in_finally(self):
        client = MLXLocalClient(model_path=TEST_MODEL)
        fake_lock = MagicMock()
        fake_lock.acquire.return_value = True
        fake_lock.release.side_effect = RuntimeError("release unlocked lock")
        client._request_lock = fake_lock

        with patch.object(client, "_generate_inner", new=AsyncMock(return_value="ok")):
            result = await client.generate("hello")

        self.assertEqual(result, "ok")
        fake_lock.release.assert_called()

    async def test_generate_soft_times_out_init_budget_without_killing_worker(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        proc = MagicMock()
        proc.is_alive.return_value = True
        client._process = proc
        client._init_done = False
        client._set_lane_state("handshaking")
        client._init_future = asyncio.get_running_loop().create_future()

        result = await client._generate_inner(
            "hello",
            foreground_request=True,
            owner_label="test",
            deadline=get_deadline(0.5),
        )

        self.assertIsNone(result)
        proc.kill.assert_not_called()
        self.assertIs(client._process, proc)
        self.assertFalse(client._init_future.done())
        self.assertEqual(client._lane_state, "recovering")

    async def test_listener_routes_init_error_without_action_to_init_future(self):
        client = MLXLocalClient(model_path=TEST_MODEL)
        self._attach_local_ipc_queues(client)
        client._init_future = asyncio.get_running_loop().create_future()

        listener = asyncio.create_task(client._response_listener_loop())
        try:
            client._res_q.put({"status": "error", "message": "Init failed: boom"})
            result = await asyncio.wait_for(asyncio.shield(client._init_future), timeout=2.0)
        finally:
            listener.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await listener

        self.assertEqual(result["action"], "init")
        self.assertEqual(result["message"], "Init failed: boom")

    async def test_generation_waiter_flags_first_token_sla_breach(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        proc = MagicMock()
        proc.is_alive.return_value = True
        client._process = proc
        client._init_done = True
        client._set_lane_state("ready")
        req_id = "req-1"
        future = asyncio.get_running_loop().create_future()
        client._pending_generations[req_id] = future
        client._current_request_id = req_id
        client._current_request_started_at = 100.0
        client._last_generation_completed_at = 1.0
        client._current_request_prompt_chars = 0
        deadline = get_deadline(None)

        with patch("core.brain.llm.mlx_client.asyncio.wait_for", side_effect=asyncio.TimeoutError):
            with patch(
                "core.brain.llm.mlx_client.time.time",
                return_value=100.0 + client._first_token_sla(foreground_request=True) + 1.0,
            ):
                result = await client._wait_for_generation_result(
                    req_id,
                    future,
                    deadline,
                    foreground_request=True,
                )

        self.assertIsNone(result)
        self.assertEqual(client._deferred_reboot_reason, "first_token_sla_exceeded")

    async def test_long_prompt_extends_first_token_sla_for_heavy_lane(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        cold_sla = client._first_token_sla(foreground_request=True)

        client._last_generation_completed_at = 1.0
        client._current_request_prompt_chars = 24_740

        warm_long_prompt_sla = client._first_token_sla(foreground_request=True)

        self.assertGreater(warm_long_prompt_sla, 22.0)
        self.assertGreater(warm_long_prompt_sla, cold_sla)

    async def test_generation_waiter_flags_token_progress_stall(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        proc = MagicMock()
        proc.is_alive.return_value = True
        client._process = proc
        client._init_done = True
        client._set_lane_state("ready")
        req_id = "req-2"
        future = asyncio.get_running_loop().create_future()
        client._pending_generations[req_id] = future
        client._current_request_id = req_id
        client._current_request_started_at = 100.0
        client._current_first_token_at = 105.0
        client._last_token_progress_at = 105.0

        with patch("core.brain.llm.mlx_client.asyncio.wait_for", side_effect=asyncio.TimeoutError):
            with patch("core.brain.llm.mlx_client.time.time", return_value=105.0 + client._token_stall_after() + 1.0):
                result = await client._wait_for_generation_result(
                    req_id,
                    future,
                    get_deadline(30.0),
                    foreground_request=True,
                )

        self.assertIsNone(result)
        self.assertEqual(client._deferred_reboot_reason, "token_progress_stalled")

    async def test_generation_waiter_recycles_fresh_heartbeat_token_stall(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        proc = MagicMock()
        proc.is_alive.return_value = True
        client._process = proc
        client._init_done = True
        client._set_lane_state("ready")
        req_id = "req-fresh-stall"
        future = asyncio.get_running_loop().create_future()
        client._pending_generations[req_id] = future
        client._current_request_id = req_id
        client._current_request_started_at = 100.0
        client._current_first_token_at = 105.0
        client._last_token_progress_at = 105.0
        now = 105.0 + client._token_stall_after(foreground_request=True) + 1.0
        client._last_heartbeat = now - 0.5

        with patch("core.brain.llm.mlx_client.asyncio.wait_for", side_effect=asyncio.TimeoutError):
            with patch("core.brain.llm.mlx_client.time.time", return_value=now):
                result = await client._wait_for_generation_result(
                    req_id,
                    future,
                    get_deadline(30.0),
                    foreground_request=True,
                )

        self.assertIsNone(result)
        self.assertEqual(client._deferred_reboot_reason, "recoverable_token_progress_stalled")
        self.assertTrue(future.cancelled())

    async def test_listener_drops_late_generation_for_previous_request(self):
        import core.brain.llm.mlx_client as mlx_module

        client = MLXLocalClient(model_path=QWEN32_MODEL)
        self._attach_local_ipc_queues(client)
        current_future = mlx_module._new_shared_future()
        client._current_gen_future = current_future
        client._current_request_id = "new-req"

        listener = asyncio.create_task(client._response_listener_loop())
        try:
            client._res_q.put({
                "status": "ok",
                "action": "generate",
                "id": "old-req",
                "text": "late stale answer",
            })
            await asyncio.sleep(0.25)
        finally:
            listener.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await listener

        self.assertFalse(current_future.done())

    async def test_warmup_precompile_accepts_empty_text_as_successful_compile(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        client._warmup_in_flight = True
        client._process = MagicMock()
        client._process.is_alive.return_value = True
        client._init_done = True

        with patch.object(client, "_generate_inner", new=AsyncMock(return_value="")):
            await client._run_warmup_precompile(
                request_is_background=False,
                foreground_request=True,
                owner_name="warmup:test",
                warmup_timeout=1.0,
            )

        self.assertEqual(client.get_lane_status()["state"], "ready")

    async def test_foreground_empty_generation_marks_recoverable_reboot(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        client._process = MagicMock()
        client._process.is_alive.return_value = True
        client._init_done = True
        self._attach_local_ipc_queues(client)
        client._set_lane_state("ready")

        with patch.object(client, "_ensure_worker_alive", new=AsyncMock(return_value=True)):
            with patch.object(
                client,
                "_wait_for_generation_result",
                new=AsyncMock(return_value={"status": "ok", "text": ""}),
            ):
                result = await client._generate_inner(
                    "hello",
                    _retry=False,
                    foreground_request=True,
                    owner_label="test",
                    deadline=get_deadline(30.0),
                )

        self.assertIsNone(result)
        self.assertEqual(client._deferred_reboot_reason, "recoverable_empty_generation")

    async def test_generate_reboots_recoverable_empty_generation_without_failed_lane(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)

        async def _empty_then_request_reboot(*args, **kwargs):
            client._deferred_reboot_reason = "recoverable_empty_generation"
            return None

        with patch.object(client, "_generate_inner", new=AsyncMock(side_effect=_empty_then_request_reboot)):
            with patch.object(client, "reboot_worker", new=AsyncMock()) as reboot_mock:
                result = await client.generate("hello", foreground_request=True, owner_label="test")

        self.assertIsNone(result)
        reboot_mock.assert_awaited_once_with(reason="empty_generation", mark_failed=False)

    async def test_supervision_status_reports_recycle_candidate(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        proc = MagicMock()
        proc.is_alive.return_value = True
        client._process = proc
        client._init_done = True
        client._process_started_at = 100.0
        client._last_generation_completed_at = 600.0

        with patch("core.brain.llm.mlx_client.time.time", return_value=2000.0):
            status = client.get_supervision_status()
            recyclable = client.should_recycle_for_fragmentation(
                max_uptime_s=900.0,
                min_idle_s=300.0,
            )

        self.assertTrue(status["alive"])
        self.assertAlmostEqual(status["process_uptime_s"], 1900.0, places=3)
        self.assertAlmostEqual(status["idle_for_s"], 1400.0, places=3)
        self.assertTrue(recyclable)

    async def test_heavy_model_swap_respects_cooldown_window(self):
        import core.brain.llm.mlx_client as mlx_module

        primary_path = "/models/32B"
        deep_path = "/models/72B"
        solver = MLXLocalClient(model_path=deep_path)

        solver_proc = MagicMock()
        solver_proc.is_alive.return_value = True

        async def _spawn_solver():
            solver._init_future.set_result({"status": "ok", "action": "init"})
            return solver_proc

        old_last_heavy = mlx_module._GLOBAL_LAST_HEAVY_MODEL
        old_last_swap = mlx_module._GLOBAL_LAST_SWAP_TIME
        mlx_module._GLOBAL_LAST_HEAVY_MODEL = primary_path
        mlx_module._GLOBAL_LAST_SWAP_TIME = 100.0

        try:
            with patch("core.brain.llm.model_registry.ACTIVE_MODEL", "Qwen2.5-32B-Instruct-8bit"), \
                 patch("core.brain.llm.model_registry.DEEP_MODEL", "Qwen2.5-72B-Instruct-4bit"), \
                 patch("core.brain.llm.model_registry.get_model_path", side_effect=lambda name=None: primary_path if "32B" in str(name) or name is None else deep_path), \
                 patch("core.brain.llm.mlx_client.os.path.realpath", side_effect=lambda path: path), \
                 patch("core.brain.llm.mlx_client.time.time", return_value=105.0), \
                 patch("core.brain.llm.mlx_client.asyncio.sleep", new_callable=AsyncMock) as sleep_mock, \
                 patch.object(solver, "_spawn_worker", side_effect=_spawn_solver):
                await solver._ensure_worker_alive()
        finally:
            mlx_module._GLOBAL_LAST_HEAVY_MODEL = old_last_heavy
            mlx_module._GLOBAL_LAST_SWAP_TIME = old_last_swap

        sleep_mock.assert_any_await(7.0)
        self.assertTrue(solver._init_done)

    async def test_heavy_model_swap_can_bypass_cooldown_for_fast_restore(self):
        import core.brain.llm.mlx_client as mlx_module

        primary_path = "/models/32B"
        deep_path = "/models/72B"
        primary = MLXLocalClient(model_path=primary_path)

        primary_proc = MagicMock()
        primary_proc.is_alive.return_value = True

        async def _spawn_primary():
            primary._init_future.set_result({"status": "ok", "action": "init"})
            return primary_proc

        old_last_heavy = mlx_module._GLOBAL_LAST_HEAVY_MODEL
        old_last_swap = mlx_module._GLOBAL_LAST_SWAP_TIME
        mlx_module._GLOBAL_LAST_HEAVY_MODEL = deep_path
        mlx_module._GLOBAL_LAST_SWAP_TIME = 100.0

        try:
            with patch("core.brain.llm.model_registry.ACTIVE_MODEL", "Qwen2.5-32B-Instruct-8bit"), \
                 patch("core.brain.llm.model_registry.DEEP_MODEL", "Qwen2.5-72B-Instruct-4bit"), \
                 patch("core.brain.llm.model_registry.get_model_path", side_effect=lambda name=None: primary_path if "32B" in str(name) or name is None else deep_path), \
                 patch("core.brain.llm.mlx_client.os.path.realpath", side_effect=lambda path: path), \
                 patch("core.brain.llm.mlx_client.time.time", return_value=105.0), \
                 patch("core.brain.llm.mlx_client.asyncio.sleep", new_callable=AsyncMock) as sleep_mock, \
                 patch.object(primary, "_spawn_worker", side_effect=_spawn_primary):
                await primary._ensure_worker_alive(skip_swap_cooldown=True)
        finally:
            mlx_module._GLOBAL_LAST_HEAVY_MODEL = old_last_heavy
            mlx_module._GLOBAL_LAST_SWAP_TIME = old_last_swap

        sleep_mock.assert_not_awaited()
        self.assertTrue(primary._init_done)


class TestIPCWriterThread(unittest.TestCase):
    def test_essential_messages_bypass_full_buffer(self):
        mp_queue = MagicMock()
        writer = IPCWriterThread(mp_queue)
        writer.local_queue = queue.Queue(maxsize=1)
        writer.local_queue.put({"status": "heartbeat"})

        item = {"status": "ok", "action": "generate", "text": "hello"}
        writer.put(item)

        mp_queue.put.assert_called_once_with(item, block=True, timeout=5.0)

    def test_heartbeat_is_dropped_when_buffer_full(self):
        mp_queue = MagicMock()
        writer = IPCWriterThread(mp_queue)
        writer.local_queue = queue.Queue(maxsize=1)
        writer.local_queue.put({"status": "heartbeat"})

        writer.put({"status": "heartbeat", "timestamp": 1.0})

        mp_queue.put.assert_not_called()


class TestMLXWorkerProgress(unittest.TestCase):
    def test_prompt_cache_budget_disables_deep_solver_retention(self):
        self.assertEqual(_prompt_cache_entry_budget_for_model("/models/Qwen2.5-72B-Instruct-4bit"), 0)
        self.assertEqual(_prompt_cache_entry_budget_for_model("/models/Qwen2.5-32B-Instruct-8bit"), 2)

    def test_generation_progress_emits_on_first_token(self):
        self.assertTrue(
            _should_emit_generation_progress(
                1,
                last_emit_at=100.0,
                now=100.2,
            )
        )

    def test_generation_progress_emits_on_time_gap_before_token_modulus(self):
        self.assertTrue(
            _should_emit_generation_progress(
                3,
                last_emit_at=100.0,
                now=101.7,
            )
        )

    def test_generation_progress_stays_quiet_when_recent_and_off_cycle(self):
        self.assertFalse(
            _should_emit_generation_progress(
                3,
                last_emit_at=100.0,
                now=100.4,
            )
        )


class TestMLXRuntimeProbeFailure(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_probe_failure_marks_lane_failed_without_spawn_loop(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)

        with patch.object(
            client,
            "_spawn_worker",
            side_effect=RuntimeError("mlx_runtime_probe_failed:metal_device_enumeration_crash"),
        ) as spawn_mock:
            alive = await client._ensure_worker_alive()

        self.assertFalse(alive)
        spawn_mock.assert_awaited_once()
        self.assertEqual(client.get_lane_status()["state"], "failed")
        self.assertEqual(
            client.get_lane_status()["last_error"],
            "mlx_runtime_unavailable:metal_device_enumeration_crash",
        )

    async def test_runtime_probe_recovery_clears_failed_lane_and_backoff(self):
        client = MLXLocalClient(model_path=QWEN32_MODEL)
        client.note_lane_failed("mlx_runtime_unavailable:metal_device_enumeration_crash")
        client._spawn_backoff_until = time.time() + 120.0
        client._consecutive_spawn_failures = 3

        proc = MagicMock()
        proc.is_alive.return_value = True

        async def _spawn():
            client._init_future.set_result({"status": "ok", "action": "init"})
            return proc

        with patch("core.brain.llm.mlx_client._probe_mlx_runtime", return_value=(True, "mlx_runtime_ok")):
            with patch.object(client, "_spawn_worker", side_effect=_spawn) as spawn_mock:
                alive = await client._ensure_worker_alive()

        self.assertTrue(alive)
        spawn_mock.assert_awaited_once()
        self.assertEqual(client.get_lane_status()["state"], "ready")
        self.assertEqual(client.get_lane_status()["last_error"], "")
        self.assertEqual(client._consecutive_spawn_failures, 0)
        self.assertEqual(client._spawn_backoff_until, 0.0)


def test_probe_reuses_fresh_positive_disk_cache(monkeypatch):
    import core.brain.llm.mlx_client as mlx_module

    monkeypatch.setattr(mlx_module.time, "time", lambda: 1000.0)
    monkeypatch.setattr(mlx_module, "_load_probe_cache_from_disk", lambda: (True, "mlx_runtime_ok", 950.0))
    monkeypatch.setattr(
        mlx_module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("probe should not run")),
    )
    monkeypatch.setattr(
        mlx_module,
        "_MLX_RUNTIME_PROBE",
        {"ok": None, "detail": "", "checked_at": 0.0},
    )

    ok, detail = mlx_module._probe_mlx_runtime(force=False)

    assert ok is True
    assert detail == "mlx_runtime_ok"


def test_probe_does_not_trust_stale_negative_disk_cache(monkeypatch):
    import core.brain.llm.mlx_client as mlx_module

    class _Completed:
        returncode = 0
        stdout = "mlx_runtime_ok\n"
        stderr = ""

    calls = []

    monkeypatch.setattr(mlx_module.time, "time", lambda: 1000.0)
    monkeypatch.setattr(
        mlx_module,
        "_load_probe_cache_from_disk",
        lambda: (False, "metal_device_enumeration_crash", 900.0),
    )
    monkeypatch.setattr(
        mlx_module.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)) or _Completed(),
    )
    monkeypatch.setattr(
        mlx_module,
        "_MLX_RUNTIME_PROBE",
        {"ok": None, "detail": "", "checked_at": 0.0},
    )
    monkeypatch.setattr(mlx_module, "_store_probe_cache_to_disk", lambda ok, detail: None)

    ok, detail = mlx_module._probe_mlx_runtime(force=False)

    assert ok is True
    assert detail == "mlx_runtime_ok"
    assert calls
