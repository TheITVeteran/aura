import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.conversation import demo_support


def test_extract_background_diagnostic_target_accepts_realistic_async_phrasing():
    target = demo_support.extract_background_diagnostic_target(
        "Aura, inspect interface/static/shell/src/App.jsx in the background and post the result here when you're done."
    )

    assert target == "interface/static/shell/src/App.jsx"


def test_resolve_target_path_rejects_absolute_paths_outside_repo(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    inside = repo_root / "inside.py"
    outside = tmp_path / "outside.py"
    inside.write_text("print('inside')\n", encoding="utf-8")
    outside.write_text("print('outside')\n", encoding="utf-8")

    assert demo_support._resolve_target_path(str(inside), repo_root=repo_root) == inside
    assert demo_support._resolve_target_path(str(outside), repo_root=repo_root) is None


def test_default_diagnostic_root_is_repository_not_core_directory():
    expected = Path(demo_support.__file__).resolve().parents[2]

    assert demo_support._repo_root() == expected
    assert (demo_support._repo_root() / "pyproject.toml").is_file()
    assert (demo_support._repo_root() / "core" / "conversation" / "demo_support.py").is_file()
    assert demo_support._resolve_target_path("pyproject.toml") == expected / "pyproject.toml"


def test_background_ack_discloses_static_evidence_boundary():
    ack = demo_support.build_background_diagnostic_ack("core/example.py")

    assert "statically inspect" in ack
    assert "verified runtime behavior" in ack
    assert "trace its core function" not in ack


@pytest.mark.asyncio
async def test_surface_activity_marks_user_requested_delivery_as_authorized():
    captured = {}

    class _FakeOutputGate:
        async def emit(self, content, origin="system", target="primary", metadata=None, **_kwargs):
            captured["content"] = content
            captured["origin"] = origin
            captured["target"] = target
            captured["metadata"] = dict(metadata or {})

    orch = SimpleNamespace(output_gate=_FakeOutputGate())

    await demo_support._surface_activity(orch, "Finished the background check.")

    assert captured["content"] == "Finished the background check."
    assert captured["origin"] == "assistant"
    assert captured["origin"] is demo_support._DiagnosticOutputOrigin.ASSISTANT
    assert captured["target"] == "primary"
    assert captured["metadata"]["requested_by_user"] is True
    assert captured["metadata"]["executive_authority"] is True
    assert captured["metadata"]["content_origin"] == "assistant_generated_background_result"
    assert captured["metadata"]["activity_kind"] == "static_source_inspection"


@pytest.mark.asyncio
async def test_surface_activity_fallback_preserves_assistant_origin():
    captured = {}

    class _FallbackOrchestrator:
        async def emit_spontaneous_message(self, content, **kwargs):
            captured["content"] = content
            captured.update(kwargs)

    await demo_support._surface_activity(
        _FallbackOrchestrator(),
        "Static inspection finished.",
    )

    assert captured["content"] == "Static inspection finished."
    assert captured["origin"] == "assistant"
    assert captured["origin"] is demo_support._DiagnosticOutputOrigin.ASSISTANT
    assert captured["modality"] == "chat"
    assert captured["metadata"]["content_origin"] == "assistant_generated_background_result"
    assert captured["origin"] != "user"


@pytest.mark.asyncio
async def test_run_background_file_diagnostic_records_failures_honestly(monkeypatch):
    recorded = {}

    async def _fake_record_recent_activity(_orch, payload):
        recorded["payload"] = payload

    async def _fake_surface_activity(_orch, summary):
        recorded["summary"] = summary

    monkeypatch.setattr(demo_support, "_record_recent_activity", _fake_record_recent_activity)
    monkeypatch.setattr(demo_support, "_surface_activity", _fake_surface_activity)
    monkeypatch.setattr(
        demo_support,
        "_resolve_target_path",
        lambda *_args, **_kwargs: Path(tempfile.gettempdir()) / "example.py",
    )
    monkeypatch.setattr(
        demo_support, "_summarize_target", lambda _path: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    await demo_support.run_background_file_diagnostic("example.py", SimpleNamespace())

    assert recorded["payload"]["ok"] is False
    assert recorded["payload"]["activity_kind"] == "static_source_inspection"
    assert recorded["payload"]["evidence_scope"] == "source_text_syntax_and_declarations"
    assert recorded["payload"]["code_executed"] is False
    assert recorded["payload"]["behavior_verified"] is False
    assert "RuntimeError" in recorded["summary"]


@pytest.mark.asyncio
async def test_recent_activity_reply_ignores_stale_persisted_state(monkeypatch):
    monkeypatch.setattr(
        demo_support,
        "_load_last_activity",
        lambda: {
            "target_name": "old_demo.py",
            "summary": "I finished the background diagnostic on `old_demo.py`. It did something useful.",
            "completed_at": 1.0,
        },
    )

    reply = await demo_support.maybe_build_recent_activity_reply(
        "What were you doing right before this session started?",
        SimpleNamespace(),
    )

    assert reply is None


@pytest.mark.asyncio
async def test_recent_activity_reply_ignores_stale_live_state(monkeypatch):
    monkeypatch.setattr(demo_support, "_load_last_activity", lambda: None)

    orch = SimpleNamespace(
        _demo_last_background_activity={
            "target_name": "old_demo.py",
            "summary": "I finished the background diagnostic on `old_demo.py`. It did something useful.",
            "completed_at": 1.0,
        }
    )

    reply = await demo_support.maybe_build_recent_activity_reply(
        "What were you doing right before this session started?",
        orch,
    )

    assert reply is None


@pytest.mark.asyncio
async def test_recent_activity_reply_uses_natural_just_working_on_preamble(monkeypatch):
    monkeypatch.setattr(
        demo_support,
        "_load_last_activity",
        lambda: {
            "target_name": "chat.py",
            "summary": "I finished the background diagnostic on `chat.py`. It traces the chat surface.",
            "completed_at": __import__("time").time(),
        },
    )

    reply = await demo_support.maybe_build_recent_activity_reply(
        "What were you just working on?",
        SimpleNamespace(),
    )

    assert reply is not None
    assert reply.startswith("I was just working on `chat.py`.")


@pytest.mark.asyncio
async def test_recent_activity_reply_preserves_static_inspection_boundary(monkeypatch):
    monkeypatch.setattr(
        demo_support,
        "_load_last_activity",
        lambda: {
            "target_name": "worker.py",
            "summary": (
                "I completed a static source inspection of `worker.py` without executing it or "
                "verifying runtime behavior. Its declared structure is: the `Worker` class."
            ),
            "completed_at": __import__("time").time(),
        },
    )

    reply = await demo_support.maybe_build_recent_activity_reply(
        "What were you doing right before this session started?",
        SimpleNamespace(),
    )

    assert reply is not None
    assert reply.startswith(
        "Right before this session, I was running a static source inspection of `worker.py`."
    )
    assert "Its declared structure is" in reply
    assert "diagnostic" not in reply
    assert "tracing its core function" not in reply


def test_python_summary_reports_real_parse_failures():
    summary = demo_support._python_summary(
        Path("broken.py"),
        "def nope(:\n    pass\n",
    )

    assert "static parse check" in summary
    assert "without executing it" in summary
    assert "doesn't currently parse as Python" in summary
    assert "line 1" in summary


def test_python_summary_reports_structure_without_claiming_diagnostic_completion():
    summary = demo_support._python_summary(
        Path("worker.py"),
        '"""Runs bounded work."""\n\nclass Worker:\n    def run(self):\n        return True\n',
    )

    assert "static source inspection" in summary
    assert "without executing it or verifying runtime behavior" in summary
    assert "Worker" in summary
    assert "finished the background diagnostic" not in summary
    assert "core function looks like" not in summary


def test_generic_summary_reports_text_evidence_without_behavior_claim():
    summary = demo_support._generic_summary(Path("settings.toml"), "enabled = true\n")

    assert "static text inspection" in summary
    assert "without executing or behaviorally validating it" in summary
    assert "enabled = true" in summary
    assert "finished the background diagnostic" not in summary
