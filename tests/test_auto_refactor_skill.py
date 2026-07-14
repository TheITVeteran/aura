from __future__ import annotations

import asyncio
import logging
import time

import pytest

from core.skills.auto_refactor import AutoRefactorParams, AutoRefactorSkill


@pytest.mark.asyncio
async def test_auto_refactor_scan_runs_off_event_loop(monkeypatch, tmp_path):
    skill = AutoRefactorSkill(root_path=tmp_path)
    observed = {}
    issue = {
        "file": "core/example.py",
        "line": 12,
        "rule_id": "PY-LONG-FUNCTION",
        "severity": "warning",
        "type": "complexity",
        "confidence": 0.96,
        "message": "Function 'demo' spans 88 lines.",
        "remediation": "Extract stages.",
    }

    async def fake_to_thread(fn, *args, **kwargs):
        observed["fn"] = fn
        observed["args"] = args
        observed["kwargs"] = kwargs
        return {
            "target": str(tmp_path),
            "display_target": ".",
            "issues": [issue],
            "coverage": {
                "coverage_complete": True,
                "files_examined_this_batch": 1,
            },
            "scan_errors": [],
        }

    monkeypatch.setattr("core.skills.auto_refactor.asyncio.to_thread", fake_to_thread)
    monkeypatch.setattr(skill, "_publish_proposals", lambda _issues: 1)

    result = await skill.execute(
        AutoRefactorParams(path=".", max_files=17, time_budget_s=1.5),
        context={},
    )

    assert observed["fn"] == skill._scan_codebase
    assert observed["args"] == (".",)
    assert observed["kwargs"] == {"max_files": 17, "time_budget_s": 1.5}
    assert result["ok"] is True
    assert result["status"] == "completed"
    assert result["issues_found"] == 1
    assert result["top_issues"][0]["file"] == "core/example.py"
    assert result["proposals_published"] == 1


def test_auto_refactor_reports_syntax_errors_without_warning(tmp_path, caplog):
    skill = AutoRefactorSkill(root_path=tmp_path)
    (tmp_path / "broken.py").write_text(
        "I'm reaching for an answer that feels honest, not just quick.\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="Skills.AutoRefactor"):
        report = skill._scan_codebase(".", max_files=10, time_budget_s=1.0)

    assert report["coverage"]["coverage_complete"] is True
    assert report["issues"][0]["rule_id"] == "PY-SYNTAX-ERROR"
    assert not caplog.records


def test_auto_refactor_finds_high_confidence_correctness_and_latency_rules(tmp_path):
    skill = AutoRefactorSkill(root_path=tmp_path)
    candidate_source = """
import time

async def blocked(values=[]):
    # DEFERRED_TAG: remove the blocking operation
    time.sleep(1)
    try:
        return values
    except Exception:
        return []
""".lstrip().replace("DEFERRED_TAG", "TO" "DO")
    (tmp_path / "candidate.py").write_text(
        candidate_source,
        encoding="utf-8",
    )

    report = skill._scan_codebase(".", max_files=10, time_budget_s=1.0)
    rules = {issue["rule_id"] for issue in report["issues"]}

    assert report["coverage"]["coverage_complete"] is True
    assert {
        "PY-ASYNC-BLOCKING-SLEEP",
        "PY-BROAD-EXCEPTION",
        "PY-DEFERRED-WORK",
        "PY-MUTABLE-DEFAULT",
    } <= rules


def test_auto_refactor_reports_invalid_encoding_and_broad_exception_tuple(tmp_path):
    skill = AutoRefactorSkill(root_path=tmp_path)
    (tmp_path / "invalid_encoding.py").write_bytes(
        b"# coding: missing-codec\nvalue = 1\n"
    )
    (tmp_path / "broad_tuple.py").write_text(
        "try:\n    value = 1\nexcept (ValueError, Exception):\n    value = 2\n",
        encoding="utf-8",
    )

    report = skill._scan_codebase(".", max_files=10, time_budget_s=1.0)
    rules = {issue["rule_id"] for issue in report["issues"]}

    assert report["coverage"]["coverage_complete"] is True
    assert {"PY-SOURCE-DECODE", "PY-BROAD-EXCEPTION"} <= rules


def test_auto_refactor_completes_incrementally_and_reuses_fingerprints(tmp_path):
    skill = AutoRefactorSkill(root_path=tmp_path)
    for index in range(7):
        (tmp_path / f"module_{index}.py").write_text(
            f"def value_{index}():\n    return {index}\n",
            encoding="utf-8",
        )

    reports = []
    for _ in range(5):
        report = skill._scan_codebase(".", max_files=2, time_budget_s=1.0)
        reports.append(report)
        if report["coverage"]["coverage_complete"]:
            break

    assert reports[0]["coverage"]["coverage_complete"] is False
    assert reports[-1]["coverage"]["coverage_complete"] is True
    assert reports[-1]["coverage"]["candidate_files_discovered"] == 7
    assert sum(item["coverage"]["files_examined_this_batch"] for item in reports) == 7

    (tmp_path / "module_3.py").write_text(
        "def value_3(values=[]):\n    return values\n",
        encoding="utf-8",
    )
    cached = skill._scan_codebase(".", max_files=20, time_budget_s=1.0)

    assert cached["coverage"]["coverage_complete"] is True
    assert cached["coverage"]["files_parsed_this_batch"] == 1
    assert cached["coverage"]["cache_hits_this_batch"] == 6
    assert [issue["rule_id"] for issue in cached["issues"]] == ["PY-MUTABLE-DEFAULT"]


def test_auto_refactor_honors_wall_clock_budget(tmp_path, monkeypatch):
    skill = AutoRefactorSkill(root_path=tmp_path)
    for index in range(20):
        (tmp_path / f"module_{index}.py").write_text("value = 1\n", encoding="utf-8")
    original = skill._scanner._analyze_python_file

    def slow_analyze(*args, **kwargs):
        time.sleep(0.02)
        return original(*args, **kwargs)

    monkeypatch.setattr(skill._scanner, "_analyze_python_file", slow_analyze)
    started = time.monotonic()
    report = skill._scan_codebase(".", max_files=100, time_budget_s=0.25)
    elapsed = time.monotonic() - started

    assert elapsed < 0.4
    assert report["coverage"]["coverage_complete"] is False
    assert report["coverage"]["deadline_reached"] is True
    assert report["coverage"]["files_examined_this_batch"] < 20


def test_auto_refactor_rejects_repository_escape(tmp_path):
    skill = AutoRefactorSkill(root_path=tmp_path)

    with pytest.raises(ValueError, match="inside the Aura repository"):
        skill._scan_codebase("../outside", max_files=1, time_budget_s=0.25)


@pytest.mark.asyncio
async def test_auto_refactor_concurrent_scan_is_deferred(monkeypatch, tmp_path):
    skill = AutoRefactorSkill(root_path=tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked_to_thread(_fn, *_args, **_kwargs):
        started.set()
        await release.wait()
        return {
            "target": str(tmp_path),
            "display_target": ".",
            "issues": [],
            "coverage": {
                "coverage_complete": True,
                "files_examined_this_batch": 0,
            },
            "scan_errors": [],
        }

    monkeypatch.setattr("core.skills.auto_refactor.asyncio.to_thread", blocked_to_thread)
    monkeypatch.setattr(skill, "_publish_proposals", lambda _issues: 0)
    first = asyncio.create_task(skill.execute(AutoRefactorParams(), context={}))
    await started.wait()

    second = await skill.execute(AutoRefactorParams(), context={})
    assert second["status"] == "deferred"
    assert second["reason"] == "scan_in_progress"

    release.set()
    assert (await first)["status"] == "completed"


@pytest.mark.asyncio
async def test_auto_refactor_broad_test_scope_is_truthfully_deferred(tmp_path):
    skill = AutoRefactorSkill(root_path=tmp_path)

    result = await skill._request_targeted_validation(tmp_path)

    assert result["status"] == "deferred"
    assert result["reason"] == "targeted_test_scope_required"

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    helper = tests_dir / "helpers.py"
    helper.write_text("VALUE = 1\n", encoding="utf-8")
    helper_result = await skill._request_targeted_validation(helper)
    assert helper_result["status"] == "deferred"
    assert helper_result["reason"] == "targeted_test_scope_required"


@pytest.mark.asyncio
async def test_auto_refactor_validation_uses_canonical_harness_job(
    monkeypatch,
    tmp_path,
):
    import core.skills.auto_refactor as module

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    target = tests_dir / "test_candidate.py"
    target.write_text("def test_candidate():\n    assert True\n", encoding="utf-8")
    observed = {}

    class Result:
        passed = True
        checks = {"pytest": True, "source_immutable": True}
        errors = []
        duration_s = 0.25

    class Harness:
        def __init__(self, root):
            observed["root"] = root

        async def run(self, changed_files, **kwargs):
            observed["changed_files"] = changed_files
            observed["kwargs"] = kwargs
            return Result()

    class Tracker:
        def __init__(self):
            self.tasks = []

        def create_task(self, coroutine, *, name):
            observed["task_name"] = name
            task = asyncio.create_task(coroutine)
            self.tasks.append(task)
            return task

    tracker = Tracker()
    monkeypatch.setattr(module, "SafeModificationHarness", Harness)
    monkeypatch.setattr(module, "get_task_tracker", lambda: tracker)
    skill = AutoRefactorSkill(root_path=tmp_path)

    queued = await skill._request_targeted_validation(target)
    await tracker.tasks[0]
    completed = await skill._request_targeted_validation(target)

    assert queued["status"] == "queued"
    assert completed["status"] == "passed"
    assert completed["owner"] == "safe_modification_harness"
    assert observed["changed_files"] == ["tests/test_candidate.py"]
    assert observed["kwargs"]["extra_test_targets"] == ["tests/test_candidate.py"]
    assert observed["kwargs"]["require_distributed_sandbox"] is False
    assert observed["task_name"].startswith("auto_refactor.validation.")
    assert len(tracker.tasks) == 1


@pytest.mark.asyncio
async def test_auto_refactor_validation_supports_bounded_test_subtree(
    monkeypatch,
    tmp_path,
):
    import core.skills.auto_refactor as module

    subtree = tmp_path / "tests" / "unit"
    subtree.mkdir(parents=True)
    (subtree / "conftest.py").write_text("VALUE = 1\n", encoding="utf-8")
    (subtree / "test_alpha.py").write_text(
        "def test_alpha():\n    assert True\n",
        encoding="utf-8",
    )
    observed = {}

    class Result:
        passed = True
        checks = {"pytest": True, "source_immutable": True}
        errors = []
        duration_s = 0.1

    class Harness:
        def __init__(self, _root):
            self.root = _root

        async def run(self, changed_files, **kwargs):
            observed["changed_files"] = changed_files
            observed["kwargs"] = kwargs
            return Result()

    class Tracker:
        def create_task(self, coroutine, *, name):
            observed["task_name"] = name
            observed["task"] = asyncio.create_task(coroutine)
            return observed["task"]

    monkeypatch.setattr(module, "SafeModificationHarness", Harness)
    monkeypatch.setattr(module, "get_task_tracker", Tracker)
    skill = AutoRefactorSkill(root_path=tmp_path)

    queued = await skill._request_targeted_validation(subtree)
    await observed["task"]
    completed = await skill._request_targeted_validation(subtree)

    assert queued["status"] == "queued"
    assert queued["source_file_count"] == 2
    assert completed["status"] == "passed"
    assert observed["changed_files"] == [
        "tests/unit/conftest.py",
        "tests/unit/test_alpha.py",
    ]
    assert observed["kwargs"]["extra_test_targets"] == ["tests/unit"]
    assert set(observed["kwargs"]["patch_content"]) == set(observed["changed_files"])


@pytest.mark.asyncio
async def test_auto_refactor_empty_test_subtree_is_deferred(tmp_path):
    subtree = tmp_path / "tests" / "empty"
    subtree.mkdir(parents=True)
    skill = AutoRefactorSkill(root_path=tmp_path)

    result = await skill._request_targeted_validation(subtree)

    assert result["status"] == "deferred"
    assert result["reason"] == "no_python_tests_in_scope"
