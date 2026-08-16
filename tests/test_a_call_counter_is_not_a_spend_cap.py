"""CP126 ``core/brain/llm/gemini_adapter.py`` — fifteen findings, four critical.

The four criticals are each a promise the code did not keep. The header
said free-tier limits prevent charges while the defaults shipped 2,000 and
10,000 requests a day and no spend cap of any kind. The class docstring
said restarts do not lose the count, and the adapter constructed the
limiter with no state path. A billable POST carrying the person's private
turn was marked ``read_only=True``. And nothing screened that turn before
it crossed to Google.

Each test names the promise and checks the code keeps it.
"""

from __future__ import annotations

import json

import pytest

from core.brain.llm.gemini_adapter import (
    DAILY_CALL_BUDGET,
    DailyRateLimiter,
    GeminiAdapter,
    GeminiProviderUnavailableError,
    _classify_gemini_failure,
    _is_daily_quota_exhaustion,
    _sse_text_chunks,
)
from core.runtime.errors import FallbackClassification


@pytest.fixture()
def limiter(tmp_path) -> DailyRateLimiter:
    return DailyRateLimiter(state_path=str(tmp_path / "usage.json"))


def _adapter(limiter: DailyRateLimiter) -> GeminiAdapter:
    return GeminiAdapter(api_key="test-key", rate_limiter=limiter)


# ── 74e032d2: a call counter is not a spend cap, and says so ────────────────


def test_the_header_no_longer_promises_charge_protection():
    import pathlib

    header = pathlib.Path("core/brain/llm/gemini_adapter.py").read_text()[:3000]
    assert "**These limits do not prevent charges.**" in header, (
        "the module still promises charge protection it does not provide"
    )
    assert "DAILY_CALL_BUDGET" in header, (
        "the one number that bounds spend is not documented where the claim was"
    )


def test_an_account_level_budget_binds_across_every_model(limiter):
    """Per-model limits shape throughput. The account is billed as one."""
    for index in range(DAILY_CALL_BUDGET):
        limiter._counts[f"model-{index % 7}"] = limiter._counts.get(f"model-{index % 7}", 0)
    limiter._counts["gemini-3.5-flash"] = DAILY_CALL_BUDGET
    assert limiter.can_call("gemini-2.0-flash") is False, (
        "a second model kept spending after the account budget was gone"
    )


def test_spend_is_reported_as_an_estimate(limiter):
    limiter._counts["gemini-3.5-flash"] = 100
    usage = limiter.get_usage()
    assert usage["account"]["spend_is_an_estimate"] is True
    assert usage["account"]["billing_plan_checked"] is False
    assert usage["account"]["estimated_spend_usd"] >= 0.0
    assert usage["account"]["calls_today"] == 100


# ── 01b0579a: the daily count survives a restart ────────────────────────────


def test_the_limiter_persists_without_being_asked(tmp_path, monkeypatch):
    """The docstring promised it; only an explicit path delivered it."""
    import core.runtime.state_ownership as ownership

    monkeypatch.setattr(ownership, "state_root", lambda: tmp_path)
    fresh = DailyRateLimiter()
    assert fresh._state_path, (
        "a restart resets the daily counter and can spend the provider's "
        "quota twice in one day"
    )
    assert str(tmp_path) in str(fresh._state_path)


def test_counts_survive_a_reconstruction(tmp_path):
    path = str(tmp_path / "usage.json")
    first = DailyRateLimiter(state_path=path)
    first.record_call("gemini-3.5-flash")
    first.record_call("gemini-3.5-flash")

    second = DailyRateLimiter(state_path=path)
    assert second._counts["gemini-3.5-flash"] == 2


# ── 36da4662: a billable POST is not read-only ──────────────────────────────


def test_the_generative_post_is_declared_effectful():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(GeminiAdapter._post_json).lstrip())
    read_only = [
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword)
        and node.arg == "read_only"
        and isinstance(node.value, ast.Constant)
    ]
    assert read_only == [False], (
        "the request that sends private content, spends quota and may incur a "
        f"charge is declared read_only={read_only}, which routes it past the "
        "effect governance that exists for writes"
    )


# ── 6343148b: nothing crosses to Google unscreened ──────────────────────────


def test_every_text_part_passes_the_egress_boundary(limiter, monkeypatch):
    import core.security.egress_privacy as egress

    seen: list[str] = []

    def _filter(text, *, provider):
        seen.append(str(text))
        return type("R", (), {"allowed": True, "text": f"[screened]{text}"})()

    monkeypatch.setattr(egress, "filter_model_prompt", _filter)
    adapter = _adapter(limiter)
    screened = adapter._screen_payload_for_egress(
        {
            "contents": [{"role": "user", "parts": [{"text": "my private turn"}]}],
            "systemInstruction": {"parts": [{"text": "the instructions"}]},
        }
    )

    assert "my private turn" in seen and "the instructions" in seen
    assert screened["contents"][0]["parts"][0]["text"].startswith("[screened]")


def test_a_refused_prompt_sends_nothing(limiter, monkeypatch):
    import core.security.egress_privacy as egress

    monkeypatch.setattr(
        egress,
        "filter_model_prompt",
        lambda text, *, provider: type("R", (), {"allowed": False, "reason": "credential"})(),
    )
    adapter = _adapter(limiter)
    assert adapter._screen_payload_for_egress(
        {"contents": [{"parts": [{"text": "sk-secret"}]}]}
    ) is None


def test_inline_media_is_refused_rather_than_waved_through(limiter):
    adapter = _adapter(limiter)
    assert adapter._screen_payload_for_egress(
        {"contents": [{"parts": [{"inlineData": {"data": "AAAA", "mimeType": "image/png"}}]}]}
    ) is None, (
        "a redaction pass cannot read base64 media, so passing it means the "
        "boundary inspected the text and waved the image"
    )


# ── a49001f5: the billing day follows the real Pacific rule ─────────────────


def test_the_billing_day_uses_the_timezone_database(limiter):
    import datetime as dt
    from zoneinfo import ZoneInfo

    expected = (
        dt.datetime.now(dt.UTC).astimezone(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
    )
    assert limiter._today() == expected, (
        "a fixed UTC-8 puts the reset an hour off the provider's boundary for "
        "eight months of the year"
    )


# ── bb409736: a local reset says what it cannot do ──────────────────────────


def test_a_manual_reset_clears_the_cluster_backoff(limiter):
    limiter.mark_cluster_429(retry_after=600.0)
    assert limiter.is_backed_off("gemini-3.5-flash") is True

    limiter.reset_manual()
    assert limiter.is_backed_off("gemini-3.5-flash") is False, (
        "the reset claimed all backoffs and left the cluster one, so counts "
        "read zero while every model still refused"
    )


def test_a_manual_reset_does_not_claim_provider_reconciliation(limiter):
    report = limiter.reset_manual()
    assert report["provider_quota_reconciled"] is False
    assert "quota" in report["warning"]


# ── f30df281: configuration cannot disable the protection ───────────────────


def test_a_hostile_threshold_cannot_block_or_disable_admission(limiter, monkeypatch):
    for value in ("-5", "1e9", "not-a-number", "nan"):
        monkeypatch.setenv("AURA_GEMINI_BACKGROUND_THRESHOLD", value)
        limiter._counts.clear()
        # Neither an unconditional refusal nor an unconditional pass.
        assert limiter.can_call("gemini-3.5-flash", is_background=False) is True
        limiter._counts["gemini-3.5-flash"] = 10**9
        assert limiter.can_call("gemini-3.5-flash", is_background=False) is False


# ── 7fb09702: priority is read ──────────────────────────────────────────────


def test_priority_changes_background_admission(limiter):
    # Past the cold-start grace window, which defers every background call
    # for its own reason.
    limiter._boot_time -= limiter.COLD_START_GRACE_S + 1.0
    limit = limiter.DEFAULT_LIMITS["gemini-3.5-flash"]
    # Above the 30% background reserve and below the account budget, so the
    # only thing that can decide this call is the priority.
    used = int(limit * 0.35)
    assert used < DAILY_CALL_BUDGET, "pick a count the account budget allows"
    limiter._counts["gemini-3.5-flash"] = used

    low = limiter.can_call("gemini-3.5-flash", is_background=True, priority=0.0)
    high = limiter.can_call("gemini-3.5-flash", is_background=True, priority=1.0)
    assert low is False and high is True, (
        "an urgent background call and an idle one were admitted identically; "
        "the priority argument was accepted, documented and never read"
    )


# ── e4b432c5: a quota dimension is parsed, not grepped for ──────────────────


def test_a_per_minute_quota_error_is_not_daily_exhaustion():
    per_minute = json.dumps(
        {"error": {"details": [{"violations": [{"quotaId": "GenerateRequestsPerMinutePerProject"}]}]}}
    )
    assert _is_daily_quota_exhaustion(per_minute) is False, (
        "a per-minute burst disabled the model for the rest of the local day"
    )


def test_a_per_day_quota_error_is_daily_exhaustion():
    per_day = json.dumps(
        {"error": {"details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProject"}]}]}}
    )
    assert _is_daily_quota_exhaustion(per_day) is True


def test_the_bare_word_quota_is_not_a_dimension():
    assert _is_daily_quota_exhaustion("you have exceeded your quota") is False


# ── 405d222c: protocol frames are not model text ────────────────────────────


def test_sse_frames_are_parsed_into_text():
    frame = 'data: {"candidates":[{"content":{"parts":[{"text":"hello"}]}}]}'
    assert _sse_text_chunks(frame) == ["hello"]


def test_protocol_noise_yields_nothing():
    assert _sse_text_chunks("") == []
    assert _sse_text_chunks(": keep-alive") == []
    assert _sse_text_chunks("data: [DONE]") == []
    assert _sse_text_chunks("data: not json") == []


def test_a_blocked_finish_is_raised_rather_than_read_as_the_end():
    blocked = 'data: {"candidates":[{"finishReason":"SAFETY","content":{"parts":[]}}]}'
    with pytest.raises(GeminiProviderUnavailableError) as caught:
        _sse_text_chunks(blocked)
    assert "safety" in str(caught.value)


def test_an_ordinary_stop_is_not_an_error():
    stop = 'data: {"candidates":[{"finishReason":"STOP","content":{"parts":[{"text":"done"}]}}]}'
    assert _sse_text_chunks(stop) == ["done"]


# ── fc2f4358: the streaming interface says whether it streamed ──────────────


def test_the_adapter_reports_that_production_does_not_stream(limiter):
    adapter = _adapter(limiter)
    assert adapter.streams_incrementally() is False, (
        "the advertised streaming interface awaits a full call and yields one "
        "value; a caller that needs early tokens has to be able to find out"
    )


# ── f3fe949e: the system prompt keeps its authority ─────────────────────────


def test_a_system_only_request_keeps_its_instruction_in_place():
    import ast
    import inspect

    source = inspect.getsource(GeminiAdapter)
    assert "Move system_prompt to user content" not in source
    # The demotion was: build a user part FROM sys_prompt, then clear the
    # instruction. Initialising the local to None before it is filled is
    # ordinary; clearing it after a sys_prompt exists is the defect.
    assert "Don\'t double-send" not in source
    tree = ast.parse(source.lstrip())
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        body = ast.dump(ast.Module(body=node.body, type_ignores=[]))
        if "sys_prompt" in body and "system_instruction" in body and "Constant(value=None)" in body:
            raise AssertionError(
                f"line {node.lineno}: the system instruction is still cleared "
                "where a sys_prompt is moved into a user part, so the person's "
                "text can contest it"
            )


# ── 70dea45b: provider bodies are summarised, not echoed ────────────────────


def test_an_error_summary_drops_the_body(limiter):
    adapter = _adapter(limiter)
    body = json.dumps(
        {
            "error": {
                "code": 400,
                "status": "INVALID_ARGUMENT",
                "message": "the user asked about their medical results",
                "details": [{"reason": "API_KEY_INVALID"}],
            }
        }
    )
    summary = adapter._safe_error_summary(body)
    assert "medical results" not in summary, (
        "the submitted content travelled into the log and the exception text"
    )
    assert "INVALID_ARGUMENT" in summary


def test_an_unparseable_body_is_reported_by_length_only(limiter):
    adapter = _adapter(limiter)
    summary = adapter._safe_error_summary("account 12345 quota project secret-name")
    assert "secret-name" not in summary
    assert "unparsed_provider_error" in summary


# ── 7df05293: failures are classified by kind ───────────────────────────────


def test_a_credential_failure_is_not_a_safe_fallback():
    classification, receipt = _classify_gemini_failure(
        RuntimeError("api key invalid"), "fell back to local"
    )
    assert classification is FallbackClassification.SILENT_LOSS_OF_CAPABILITY
    assert receipt is True


def test_a_privacy_refusal_needs_a_receipt_without_failing_closed():
    """The boundary WORKED, so this is not a bypass — and it is not routine.

    Classifying an honoured refusal as GOVERNANCE_BYPASS made the runtime's
    fail-closed policy raise on the correct outcome: content did not leave
    and the answer came from the local lane. What it needs is a receipt, so
    an operator sees the cloud lane went away and why.
    """
    classification, receipt = _classify_gemini_failure(
        PermissionError("egress refused"), "answered locally"
    )
    assert classification is FallbackClassification.SILENT_LOSS_OF_CAPABILITY
    assert receipt is True


def test_content_leaving_unscreened_would_still_be_a_bypass():
    """No path can produce this today. The classifier is ready if one does."""
    classification, receipt = _classify_gemini_failure(
        RuntimeError("sent unscreened"), "logged it"
    )
    assert classification is FallbackClassification.GOVERNANCE_BYPASS
    assert receipt is True


def test_an_ordinary_timeout_is_still_a_safe_fallback():
    classification, receipt = _classify_gemini_failure(
        TimeoutError("slow"), "retried on the local lane"
    )
    assert classification is FallbackClassification.SAFE_FALLBACK
    assert receipt is False
