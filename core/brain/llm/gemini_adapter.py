"""core/brain/llm/gemini_adapter.py
Frontier LLM Adapter for Google Gemini API.

Provides PRIMARY-tier reasoning via Gemini, with a local daily and
per-minute call budget and automatic fallback to on-device models when
that budget is spent.

**These limits do not prevent charges.** This header used to promise
"free tier rate limits to prevent charges" and quote 50 RPD for Pro and
1500 for Flash, while the code shipped 2,000 and 10,000 and labelled them
paid-tier baselines (CP126 ``74e032d2``). Nothing here checks a billing
plan, estimates a token price, reads a dollar budget, calls a billing API
or sets a provider-side quota. What it has is a call counter, and a call
counter on an account with billing enabled limits requests, not money.

:data:`DAILY_CALL_BUDGET` is the one number that stands between this
adapter and an unbounded bill. It is enforced across every model, not per
model, because the account is billed as one account. Set
``AURA_GEMINI_DAILY_CALL_BUDGET`` to change it; set it to 0 to refuse
every cloud call.

Estimated spend is reported from a per-call token estimate and a
published per-token price. It is an ESTIMATE and says so wherever it
appears; the provider is the only authority on what was actually charged.
"""
import asyncio
import json
import logging
import math
import os
import re
import threading
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

import httpx

from core.resilience.factory import circuit_breaker
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.network_gateway import get_network_gateway
from core.utils.exceptions import capture_and_log
from core.runtime.flags import FlagKind as _FlagKind, declare as _declare_flag
from core.runtime.flags import env_str

# Declared flags (migrated from raw os.environ reads so the knobs are
# inventoried and reportable). STRING kind with the original literal
# default keeps read semantics byte-identical to os.environ.get.
_FLAG_GEMINI_CHAT_MODEL = _declare_flag(
    "AURA_GEMINI_CHAT_MODEL",
    kind=_FlagKind.STRING,
    default="gemini-3.5-flash",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_GEMINI_DEEP_MODEL = _declare_flag(
    "AURA_GEMINI_DEEP_MODEL",
    kind=_FlagKind.STRING,
    default="gemini-3.5-flash",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_GEMINI_THINKING_MODEL = _declare_flag(
    "AURA_GEMINI_THINKING_MODEL",
    kind=_FlagKind.STRING,
    default="gemini-3.5-pro",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)


logger = logging.getLogger("Brain.Gemini")


def _clamp_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return max(low, min(high, result))


def _clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(low, min(high, result))


def _env_int(name: str, default: int, *, low: int, high: int, description: str) -> int:
    """A configured limit, clamped where it is read.

    These were parsed at import with a bare ``int(...)``: a non-numeric
    value raised during module import and took the whole adapter with it,
    and a negative or enormous one silently disabled the protection or
    blocked every call (CP126 ``f30df281``).
    """
    return _clamp_int(
        env_str(name, default=default, description=description, owner=_FLAG_OWNER),
        default,
        low,
        high,
    )


def _env_float(name: str, default: float, *, low: float, high: float, description: str) -> float:
    return _clamp_float(
        env_str(name, default=default, description=description, owner=_FLAG_OWNER),
        default,
        low,
        high,
    )


_FLAG_OWNER = "core.brain.llm.gemini_adapter"

#: Calls per day across EVERY Gemini model. The per-model limits below are
#: throughput shaping; this is the account-level ceiling, and it is the only
#: thing here that bounds spend at all (CP126 ``74e032d2``).
DAILY_CALL_BUDGET = _clamp_int(
    os.environ.get("AURA_GEMINI_DAILY_CALL_BUDGET", 4000), 4000, 0, 1_000_000
)
#: Published price per million input+output tokens, for the ESTIMATE only.
#: The provider is the authority on what was charged.
ESTIMATED_USD_PER_MTOK = _clamp_float(
    os.environ.get("AURA_GEMINI_USD_PER_MTOK", 0.30), 0.30, 0.0, 1000.0
)
#: Tokens assumed per call when the response carries no usage count.
ESTIMATED_TOKENS_PER_CALL = _clamp_int(
    os.environ.get("AURA_GEMINI_TOKENS_PER_CALL", 2000), 2000, 1, 1_000_000
)


GEMINI_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    httpx.HTTPError,
)


#: Failure kinds that are NOT a safe fallback. A credential failure, a
#: privacy refusal, a quota exhaustion and a billing surprise all need an
#: operator and evidence; hard-coding every one of them as SAFE_FALLBACK
#: with receipt_required=False understated exactly the incidents that
#: warrant action (CP126 ``7df05293``).
#: Content that left WITHOUT screening would be a governance bypass. No path
#: here can produce one: every request is screened or refused. The marker
#: exists so that if a future path ever does, it is classified as what it is
#: rather than as a fallback.
_GOVERNANCE_MARKERS = ("unscreened", "bypassed_egress", "governance_bypass")
#: Credentials, quota, spend and a refused egress all mean the cloud lane is
#: gone. Falling back to local is correct AND the operator needs to know,
#: which is what a receipt is for. Hard-coding every failure as
#: SAFE_FALLBACK with receipt_required=False understated exactly the
#: incidents that warrant action (CP126 ``7df05293``).
_CAPABILITY_LOSS_MARKERS = (
    "auth",
    "api_key",
    "api key",
    "permission",
    "credential",
    "quota",
    "billing",
    "budget",
    "exhaust",
    "egress",
    "privacy",
    "consent",
)


def _classify_gemini_failure(
    error: BaseException, action: str
) -> tuple[FallbackClassification, bool]:
    """Which kind of failure this is, and whether it needs a receipt."""
    text = f"{type(error).__name__} {error} {action}".lower()
    if any(marker in text for marker in _GOVERNANCE_MARKERS):
        return FallbackClassification.GOVERNANCE_BYPASS, True
    if any(marker in text for marker in _CAPABILITY_LOSS_MARKERS):
        return FallbackClassification.SILENT_LOSS_OF_CAPABILITY, True
    return FallbackClassification.SAFE_FALLBACK, False


def _record_gemini_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    classification, receipt_required = _classify_gemini_failure(error, action)
    record_degradation(
        "gemini_adapter",
        error,
        severity=severity,
        action=action,
        classification=classification,
        receipt_required=receipt_required,
        extra=extra,
    )


#: Quota dimensions Google reports that mean "spent for the whole day".
#: Anything else — a per-minute burst, a concurrency cap, a regional or
#: token limit — is a wait, not an exhaustion.
_DAILY_QUOTA_MARKERS = (
    "perday",
    "per_day",
    "per day",
    "requestsperday",
    "dailylimit",
    "daily limit",
)


def _first_reason(error: dict[str, Any]) -> str:
    details = error.get("details")
    if isinstance(details, list):
        for entry in details:
            if isinstance(entry, dict) and entry.get("reason"):
                return str(entry["reason"])[:60]
    return str(error.get("message", ""))[:0] or ""


def _is_daily_quota_exhaustion(text: str) -> bool:
    """Whether a 429 body actually says the DAY's quota is gone.

    Prefers the structured quota metric Google returns; falls back to the
    documented per-day markers. A bare "quota" is not one of them.
    """
    lowered = str(text or "").lower()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        details = (parsed.get("error") or {}).get("details")
        if isinstance(details, list):
            for entry in details:
                if not isinstance(entry, dict):
                    continue
                for violation in entry.get("violations") or []:
                    if not isinstance(violation, dict):
                        continue
                    metric = str(violation.get("quotaId") or violation.get("quotaMetric") or "").lower()
                    if any(marker in metric for marker in _DAILY_QUOTA_MARKERS):
                        return True
                    if metric:
                        # A named dimension that is NOT daily settles it.
                        return False
    return any(marker in lowered for marker in _DAILY_QUOTA_MARKERS)


def _sse_text_chunks(line: str) -> list[str]:
    """Extract generated text from one server-sent-events line.

    ``aiter_lines`` output was yielded verbatim, so a caller expecting model
    text received ``data: {...}`` protocol frames, keep-alive blanks and the
    done marker, and had to guess which was which (CP126 ``405d222c``).
    Blocked or truncated finishes were invisible for the same reason.
    """
    raw = str(line or "").strip()
    if not raw or raw.startswith(":"):
        return []
    if raw.startswith("data:"):
        raw = raw[5:].strip()
    if not raw or raw == "[DONE]":
        return []
    try:
        event = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    if not isinstance(event, dict):
        return []

    chunks: list[str] = []
    for candidate in event.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        finish = str(candidate.get("finishReason") or "").upper()
        if finish and finish not in {"STOP", "MAX_TOKENS"}:
            # SAFETY, RECITATION and the rest are outcomes the caller has to
            # be able to see. Yielding nothing made them look like the end.
            raise GeminiProviderUnavailableError(f"gemini_finish_{finish.lower()}")
        for part in (candidate.get("content") or {}).get("parts") or []:
            if isinstance(part, dict) and part.get("text"):
                chunks.append(str(part["text"]))
    return chunks


class GeminiProviderUnavailableError(RuntimeError):
    """Non-crash provider/configuration failure; local lanes should continue."""


GeminiProviderUnavailable = GeminiProviderUnavailableError


class DailyRateLimiter:
    """Tracks per-model daily usage and enforces free-tier limits.
    
    Resets at midnight Pacific Time (Google's reset schedule).
    Saves state to disk so restarts don't lose the count.
    """
    
    DEFAULT_LIMITS = {
        "gemini-pro": _env_int("AURA_GEMINI_RPD_PRO", 2000, low=0, high=1_000_000, description="gemini rpd pro"),
        "gemini-2.5-flash": _env_int("AURA_GEMINI_RPD_DEEP", 2000, low=0, high=1_000_000, description="gemini rpd deep"),
        "gemini-flash-latest": _env_int("AURA_GEMINI_RPD_FLASH", 10000, low=0, high=1_000_000, description="gemini rpd flash"),
        "gemini-2.0-flash": _env_int("AURA_GEMINI_RPD_FLASH", 10000, low=0, high=1_000_000, description="gemini rpd flash"),
        "gemini-2.5-pro": _env_int("AURA_GEMINI_RPD_THINKING", 2000, low=0, high=1_000_000, description="gemini rpd thinking"),
        "gemini-3.5-flash": _env_int("AURA_GEMINI_RPD_FLASH", 10000, low=0, high=1_000_000, description="gemini rpd flash"),
        "gemini-3.5-pro": _env_int("AURA_GEMINI_RPD_THINKING", 2000, low=0, high=1_000_000, description="gemini rpd thinking"),
    }
    
    # Per-minute limits (High-performance baseline for paid tiers)
    RPM_LIMITS = {
        "gemini-pro": _env_int("AURA_GEMINI_RPM_PRO", 50, low=0, high=1_000_000, description="gemini rpm pro"),
        "gemini-2.5-flash": _env_int("AURA_GEMINI_RPM_DEEP", 50, low=0, high=1_000_000, description="gemini rpm deep"),
        "gemini-flash-latest": _env_int("AURA_GEMINI_RPM_FLASH", 500, low=0, high=1_000_000, description="gemini rpm flash"),
        "gemini-2.0-flash": _env_int("AURA_GEMINI_RPM_FLASH", 500, low=0, high=1_000_000, description="gemini rpm flash"),
        "gemini-2.5-pro": _env_int("AURA_GEMINI_RPM_THINKING", 50, low=0, high=1_000_000, description="gemini rpm thinking"),
        "gemini-3.5-flash": _env_int("AURA_GEMINI_RPM_FLASH", 500, low=0, high=1_000_000, description="gemini rpm flash"),
        "gemini-3.5-pro": _env_int("AURA_GEMINI_RPM_THINKING", 50, low=0, high=1_000_000, description="gemini rpm thinking"),
    }
    
    def __init__(self, state_path: str | None = None):
        # Persist by DEFAULT. Disk persistence existed only when a caller
        # supplied a path, and GeminiAdapter constructed this with none — so
        # a restart reset the daily counters and could spend the provider's
        # quota (and the account's money) twice in one day, while the class
        # docstring said restarts do not lose the count (CP126 ``01b0579a``).
        if state_path is None:
            try:
                from core.runtime.state_ownership import state_root

                from core.runtime.file_write_gateway import get_file_write_gateway

                default_path = state_root() / "data" / "gemini" / "rate_limiter.json"
                # Through the gateway rather than a raw mkdir: this is a
                # durable state directory, and the gateway is where those are
                # governed.
                get_file_write_gateway().ensure_directory(
                    default_path.parent, source="brain.llm.gemini_adapter.rate_limiter"
                )
                state_path = str(default_path)
            except (ImportError, AttributeError, OSError, RuntimeError) as exc:
                _record_gemini_degradation(
                    exc,
                    action="ran the Gemini rate limiter in memory only; a restart will reset the daily count",
                    severity="warning",
                )
        self._counts: dict[str, int] = defaultdict(int)
        self._reset_date: str = self._today()
        self._state_path = state_path
        self._minute_timestamps: dict[str, deque] = defaultdict(deque)
        self._backoff_until: dict[str, float] = {}  # model -> timestamp
        self._cluster_backoff_until: float = 0.0     # All Gemini models
        self._boot_time: float = time.monotonic()
        self.COLD_START_GRACE_S: float = 90.0  # 90s boot grace window
        # Guards the check-and-increment sequence: can_call read the counters
        # and record_call incremented them separately, so N concurrent
        # requests could all pass the check and then all consume, blowing the
        # quota (and, on a paid tier, the spend cap).
        self._lock = threading.RLock()
        self._load_state()

    def try_reserve(self, model: str, is_background: bool = False,
                    priority: float = 0.5) -> bool:
        """Atomically admit AND reserve one call's quota.

        Reserving up front (rather than can_call now, record_call after the
        response) makes admission race-free and also counts the attempt
        against quota immediately — the provider counts failed, 4xx/5xx,
        cancelled, and partial requests too, so recording only completed
        successes let those slip the local counter and overrun the real
        quota.
        """
        with self._lock:
            if not self.can_call(model, is_background=is_background, priority=priority):
                return False
            self.record_call(model)
            return True
    
    def _today(self) -> str:
        """Current date on Google's billing day boundary.

        A fixed UTC-8 was an approximation that is simply wrong for eight
        months of the year: under daylight saving the local counter reset
        an hour away from the provider's boundary, so calls made in that
        hour were counted against the wrong day (CP126 ``a49001f5``).
        """
        import datetime as dt

        try:
            from zoneinfo import ZoneInfo

            pacific = datetime.now(dt.UTC).astimezone(ZoneInfo("America/Los_Angeles"))
        except (ImportError, KeyError, ValueError) as exc:
            # No tz database on this host. The fixed offset is wrong under
            # DST, so say so rather than let the approximation pass as the
            # provider's boundary.
            _record_gemini_degradation(
                exc,
                action="fell back to a fixed UTC-8 billing day; the reset can be an hour off under daylight saving",
                severity="info",
            )
            pacific = datetime.now(dt.UTC).astimezone(dt.timezone(dt.timedelta(hours=-8)))
        return pacific.strftime("%Y-%m-%d")
    
    def _load_state(self):
        """Load daily counts from disk if available."""
        if self._state_path:
            try:
                import json
                from pathlib import Path
                p = Path(self._state_path)
                if p.exists():
                    data = json.loads(p.read_text())
                    if data.get("date") == self._today():
                        self._counts = defaultdict(int, data.get("counts", {}))
                        self._reset_date = data["date"]
                        logger.info("📊 Loaded Gemini usage: %s", dict(self._counts))
                    else:
                        logger.info("📊 New day — resetting Gemini usage counters")
            except GEMINI_RECOVERABLE_ERRORS as e:
                _record_gemini_degradation(
                    e,
                    action="started Gemini rate limiter with empty in-memory counters after state load failed",
                    extra={"state_path": self._state_path},
                )
                logger.debug("Failed to load rate limiter state: %s", e)
    
    def _save_state(self):
        """Persist daily counts to disk."""
        if self._state_path:
            try:
                from pathlib import Path
                state_path = Path(self._state_path)
                atomic_write_text(state_path, json.dumps({
                    "date": self._reset_date,
                    "counts": dict(self._counts),
                }))
            except GEMINI_RECOVERABLE_ERRORS as e:
                _record_gemini_degradation(
                    e,
                    action="kept Gemini rate limiter counters in memory after durable state save failed",
                    extra={"state_path": self._state_path},
                )
                capture_and_log(e, {'module': __name__})
    
    def _maybe_reset(self):
        """Reset counters if it's a new day."""
        today = self._today()
        if today != self._reset_date:
            logger.info("📊 Daily reset: Gemini quotas refreshed (%s → %s)", 
                       self._reset_date, today)
            self._counts.clear()
            self._reset_date = today
            self._save_state()
    
    def _check_rpm(self, model: str) -> bool:
        """Check if we're within per-minute rate limit."""
        now = time.monotonic()
        rpm_limit = self.RPM_LIMITS.get(model, 3)
        
        # Clean old timestamps with O(1) amortized popleft
        dq = self._minute_timestamps[model]
        while dq and now - dq[0] >= 60:
            dq.popleft()
        
        return len(dq) < rpm_limit
    
    def _record_rpm(self, model: str):
        """Record a call for RPM tracking."""
        self._minute_timestamps[model].append(time.monotonic())
    
    def mark_429(self, model: str, retry_after: float = 60.0):
        """Mark a 429 backoff for this model."""
        self._backoff_until[model] = time.monotonic() + retry_after
        logger.warning("🚫 Gemini %s: 429 backoff for %.0fs", model, retry_after)
    
    def is_backed_off(self, model: str) -> bool:
        """Check if this model is in a 429 backoff period."""
        now = time.monotonic()
        
        # Check cluster-level backoff first
        if now < self._cluster_backoff_until:
            return True
            
        until = self._backoff_until.get(model, 0)
        if now < until:
            return True
        return False

    def mark_cluster_429(self, retry_after: float = 5.0):
        """Mark a brief cooldown for ALL Gemini models to prevent project quota thumping."""
        self._cluster_backoff_until = time.monotonic() + retry_after
        logger.info("🛡️ Gemini Cluster: Entering %.0fs project-level backoff", retry_after)
    
    def is_cold_start(self) -> bool:
        """True during the first 90s after boot — protect against startup RPM storms."""
        return (time.monotonic() - self._boot_time) < self.COLD_START_GRACE_S

    def account_calls_today(self) -> int:
        """Calls across EVERY model today. The account is billed as one."""
        return sum(self._counts.values())

    def estimated_spend_usd(self) -> float:
        """An ESTIMATE, from a call count and a published price. Not a bill."""
        tokens = self.account_calls_today() * ESTIMATED_TOKENS_PER_CALL
        return round((tokens / 1_000_000.0) * ESTIMATED_USD_PER_MTOK, 4)

    def can_call(self, model: str, is_background: bool = False, priority: float = 0.5) -> bool:
        """Check if we have remaining quota for this model.

        ``priority`` is read. It was accepted, documented as lowering
        background admission, and never looked at, so an urgent background
        call and an idle one were admitted identically (CP126
        ``7fb09702``).
        """
        self._maybe_reset()

        # The account-level ceiling comes first, because per-model limits
        # shape throughput and this is the only line that bounds spend.
        if self.account_calls_today() >= DAILY_CALL_BUDGET:
            logger.warning(
                "🚫 Gemini: daily account budget of %d calls is spent (est. $%.2f)",
                DAILY_CALL_BUDGET,
                self.estimated_spend_usd(),
            )
            return False

        priority = _clamp_float(priority, 0.5, 0.0, 1.0)
        
        # Block non-urgent background calls during boot grace window
        if is_background and self.is_cold_start():
            logger.debug("⏳ Cold-start grace: deferring background Gemini call for %s", model)
            return False
            
        # Check 429 backoff first
        if self.is_backed_off(model):
            return False
        
        # Check RPM
        if not self._check_rpm(model):
            logger.debug("⏳ Gemini %s: RPM limit reached, waiting...", model)
            return False
        
        # Check daily limit
        limit = self.DEFAULT_LIMITS.get(model, 80)
        
        # [Pipeline Hardening] Conservative background limit: Stop using Gemini for background tasks
        # once we hit the preservation threshold (default 30%), preserving 70% for the User.
        # Clamped: a negative threshold blocked every background call and a
        # value above one disabled the preservation entirely.
        preservation_threshold = _env_float(
            "AURA_GEMINI_BACKGROUND_THRESHOLD", 0.3, low=0.0, high=1.0,
            description="gemini background threshold",
        )
        # Priority widens the background reserve for work that says it
        # matters. At priority 1.0 a background call may use the same share
        # a foreground one would; at 0.0 it gets the documented floor.
        effective_threshold = preservation_threshold + (1.0 - preservation_threshold) * priority
        if is_background and self._counts[model] > (limit * effective_threshold):
            logger.debug(
                "📉 Preserving Gemini %s quota: background call diverted "
                "(threshold %.2f at priority %.2f).",
                model,
                effective_threshold,
                priority,
            )
            return False

        return self._counts[model] < limit

    def reset_manual(self) -> dict[str, Any]:
        """Clear the LOCAL counters and backoffs. Returns what it could not do.

        It claimed to reset all backoffs and left ``_cluster_backoff_until``
        untouched, so the local state contradicted itself: daily counts at
        zero and the whole cluster still refusing (CP126 ``bb409736``).

        It also cannot reconcile with the provider. Clearing a local counter
        that reached the real quota re-enables calls that will 429, and on a
        billed account those attempts still cost. The refusal to pretend
        otherwise is the return value.
        """
        with self._lock:
            self._counts.clear()
            self._backoff_until.clear()
            self._cluster_backoff_until = 0.0
            self._reset_date = self._today()
            self._save_state()
        logger.warning(
            "📊 Gemini rate limits manually RESET locally; the provider's quota "
            "is unchanged and unknown."
        )
        return {
            "local_counters_cleared": True,
            "model_backoffs_cleared": True,
            "cluster_backoff_cleared": True,
            "provider_quota_reconciled": False,
            "warning": (
                "the provider's own quota was not consulted; calls admitted "
                "after this reset may 429 and still count against the account"
            ),
        }
    
    def record_call(self, model: str):
        """Record one API call (attempt) against quota."""
        with self._lock:
            self._maybe_reset()
            self._counts[model] += 1
            self._record_rpm(model)
            remaining = self.DEFAULT_LIMITS.get(model, 80) - self._counts[model]
            if remaining <= 10:
                logger.warning("⚠️ Gemini %s: %d calls remaining today", model, remaining)
            self._save_state()
    
    def get_usage(self) -> dict:
        """Return current usage stats, per model and per ACCOUNT."""
        self._maybe_reset()
        per_model = {
            model: {
                "used": self._counts.get(model, 0),
                "limit": limit,
                "remaining": limit - self._counts.get(model, 0),
            }
            for model, limit in self.DEFAULT_LIMITS.items()
        }
        return {
            "models": per_model,
            "account": {
                "calls_today": self.account_calls_today(),
                "daily_call_budget": DAILY_CALL_BUDGET,
                "estimated_spend_usd": self.estimated_spend_usd(),
                # Said out loud wherever the number appears. Nothing here
                # reads a billing plan or a billing API.
                "spend_is_an_estimate": True,
                "billing_plan_checked": False,
            },
            **per_model,
        }


class GeminiAdapter:
    """Adapter for Google Gemini API — slots into IntelligentLLMRouter as PRIMARY tier.
    
    Provides both streaming (generate_text_stream_async) and non-streaming (call)
    interfaces so the router's race_think_stream and think_stream both work.
    """
    
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    
    # Gemini 3.5 family: Flash for speed and deep, Pro for thinking/advanced reasoning
    CHAT_MODEL = _FLAG_GEMINI_CHAT_MODEL.value()
    DEEP_MODEL = _FLAG_GEMINI_DEEP_MODEL.value()
    THINKING_MODEL = _FLAG_GEMINI_THINKING_MODEL.value()
    
    def __init__(self, api_key: str, model: str = None, 
                 rate_limiter: DailyRateLimiter | None = None,
                 timeout: float = 120.0):
        self.api_key = api_key
        self.model = model or self.CHAT_MODEL
        self.timeout = timeout
        self.rate_limiter = rate_limiter or DailyRateLimiter()
        self._disabled_until: float = 0.0
        self._disabled_reason: str = ""
        # Last call's telemetry, so the string-only compatibility methods
        # (generate/generate_text_async) do not ERASE a provider failure into
        # an indistinguishable empty string — the router reads this to tell a
        # failed call apart from a genuinely empty answer.
        self._last_generation_metadata: dict[str, Any] = {}
        #: Whether the last generate_text_stream_async actually streamed.
        #: In production it does not: there is no incremental network
        #: stream, and a caller that needs one has to be able to find out.
        self._last_stream_was_incremental: bool = False
        logger.info("✨ GeminiAdapter initialized: model=%s", self.model)

    def streams_incrementally(self) -> bool:
        """Whether the last stream call produced tokens as they arrived."""
        return self._last_stream_was_incremental

    def get_last_generation_metadata(self) -> dict[str, Any]:
        """Telemetry from the most recent call, including {'error': ...} on
        failure. Consulted by the router when a string result is empty."""
        return dict(self._last_generation_metadata)

    def _reserve_quota(self, is_background: bool, priority: float = 0.5) -> bool:
        """Atomically admit and count one call. Prefers the limiter's
        race-free ``try_reserve``; falls back to the older
        can_call/record_call pair for limiters that predate it."""
        limiter = self.rate_limiter
        reserve = getattr(limiter, "try_reserve", None)
        if callable(reserve):
            return bool(
                reserve(self.model, is_background=is_background, priority=priority)
            )
        if not limiter.can_call(
            self.model, is_background=is_background, priority=priority
        ):
            return False
        limiter.record_call(self.model)
        return True

    def is_available(self) -> bool:
        return time.monotonic() >= self._disabled_until

    def availability_reason(self) -> str:
        if self.is_available():
            return ""
        return self._disabled_reason or "gemini_provider_disabled"

    def _mark_provider_unavailable(self, reason: str, cooldown_s: float = 3600.0) -> None:
        self._disabled_reason = str(reason or "gemini_provider_unavailable")
        self._disabled_until = time.monotonic() + max(60.0, float(cooldown_s or 3600.0))
    
    async def close(self):
        return None

    def _get_client(self) -> Any | None:
        """Optional test/diagnostic client seam; production uses NetworkGateway."""
        return None

    def _screen_payload_for_egress(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Inspect every text part before it crosses to Google, or send none.

        Prompt, system instruction, message history, arbitrary parts and
        inlineData were posted to the external endpoint with no sensitivity
        classification, no redaction and no per-request cloud opt-in (CP126
        ``6343148b``). The request goes through NetworkGateway, which
        inspects a body it can parse — but the decision to send a person's
        turn to a third party is not a transport decision, and it was made
        nowhere.

        Returns None to mean "do not send this", which the callers already
        treat as a failed cloud leg and answer locally.

        `inlineData` is refused rather than screened. It is base64 media, a
        redaction pass cannot read it, and passing it through would mean the
        boundary inspected the text and waved the image.
        """
        try:
            from core.security.egress_privacy import filter_model_prompt
        except (ImportError, AttributeError) as exc:
            _record_gemini_degradation(
                exc,
                action="refused the Gemini request rather than send an unscreened prompt",
                severity="warning",
            )
            return None

        screened = json.loads(json.dumps(payload))
        blocks: list[str] = []

        def _screen_parts(parts: Any) -> bool:
            if not isinstance(parts, list):
                return True
            for part in parts:
                if not isinstance(part, dict):
                    continue
                if part.get("inlineData"):
                    blocks.append("inline_media_cannot_be_screened")
                    return False
                text = part.get("text")
                if not text:
                    continue
                result = filter_model_prompt(str(text), provider="gemini")
                if not getattr(result, "allowed", False):
                    blocks.append(str(getattr(result, "reason", "refused")))
                    return False
                part["text"] = getattr(result, "text", text)
            return True

        for content in screened.get("contents") or []:
            if isinstance(content, dict) and not _screen_parts(content.get("parts")):
                break
        instruction = screened.get("systemInstruction")
        if not blocks and isinstance(instruction, dict):
            _screen_parts(instruction.get("parts"))

        if blocks:
            _record_gemini_degradation(
                PermissionError(f"egress refused: {blocks[0]}"),
                action="answered locally rather than send content the boundary refused",
                severity="warning",
                extra={"model": self.model},
            )
            return None
        return screened

    def _auth_headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """API key travels in the x-goog-api-key HEADER, never the URL.

        A key in the query string leaks into gateway, proxy, exception,
        tracing, and access logs — a far wider exposure surface than a
        header, which logging sinks routinely redact.
        """
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        if extra:
            headers.update(extra)
        return headers

    async def _post_json(self, url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any], str]:
        """POST to the provider. Effectful, bounded, and cancellable-ish.

        ``read_only=True`` was a misclassification with teeth: this request
        transmits the person's private content to a third party, consumes
        quota, may incur a charge, and creates provider-side logs and
        retention. Marking it read-only let it past the effect governance
        and audit policy that exists for writes (CP126 ``36da4662``).

        The work runs in a worker thread, so cancelling this coroutine does
        NOT stop the HTTP request: the bytes still go, the quota is still
        spent, and the provider still logs it (CP126 ``4d0e2c27``). A thread
        cannot be killed in Python, so instead of pretending, cancellation
        is recorded — a caller that failed over needs to know a paid request
        is still in flight behind it.
        """
        screened = self._screen_payload_for_egress(payload)
        if screened is None:
            return 0, {}, "egress_refused"
        payload = screened
        request = asyncio.to_thread(
            get_network_gateway().request,
            "POST",
            url,
            headers=self._auth_headers(),
            data=json.dumps(payload),
            timeout=self.timeout,
            source=f"llm_provider:gemini:{self.model}",
            read_only=False,
        )
        try:
            response = await request
        except asyncio.CancelledError:
            _record_gemini_degradation(
                RuntimeError("gemini request cancelled while in flight"),
                action=(
                    "the worker thread cannot be stopped, so the request completes "
                    "and still counts against quota and spend"
                ),
                severity="warning",
                extra={"model": self.model},
            )
            raise
        body = response.get("content") or b""
        text = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else str(body)
        data: dict[str, Any] = {}
        if text.strip():
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    data = parsed
            except json.JSONDecodeError:
                data = {}
        return int(response.get("status_code") or 0), data, text
    
    async def _handle_error(self, response: httpx.Response):
        """Standardized error handling for Gemini API."""
        error_body = await response.aread()
        await self._handle_error_payload(response.status_code, error_body)

    @staticmethod
    def _safe_error_summary(text: str) -> str:
        """What may be logged from a provider error body.

        Up to 500 characters of the raw body reached the logs and the
        exception text, and compatibility paths propagated it. Those bodies
        carry account identifiers, project numbers, quota dimensions, safety
        verdicts and — on a content error — the submitted content itself
        (CP126 ``70dea45b``). The structured fields are what an operator
        needs; the prose is what leaks.
        """
        try:
            parsed = json.loads(text)
            error = parsed.get("error") if isinstance(parsed, dict) else None
            if isinstance(error, dict):
                return (
                    f"status={error.get('status', '')} "
                    f"code={error.get('code', '')} "
                    f"reason={_first_reason(error)}"
                ).strip()
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            pass
        return f"unparsed_provider_error:{len(text)}_chars"

    async def _handle_error_payload(self, status_code: int, error_body: bytes | str):
        """Standardized error handling for Gemini API gateway responses."""
        if isinstance(error_body, str):
            error_body = error_body.encode("utf-8", errors="replace")
        text = error_body.decode('utf-8', errors='replace')
        lowered = text.lower()
        
        if status_code == 429:
            retry_after = self._parse_retry_after(error_body)
            # A 429 whose body mentions "quota" anywhere used to set the
            # local daily counter to its maximum, disabling the model for
            # the rest of the day. Per-minute, per-region, per-token,
            # concurrency and per-project limits all say "quota" and none of
            # them is daily exhaustion (CP126 ``e4b432c5``).
            if _is_daily_quota_exhaustion(text):
                logger.error("🚫 Gemini %s: DAILY QUOTA EXHAUSTED", self.model)
                self.rate_limiter._counts[self.model] = self.rate_limiter.DEFAULT_LIMITS.get(self.model, 80)
                self.rate_limiter._save_state()
            else:
                self.rate_limiter.mark_429(self.model, retry_after)
                # Phase 39: Protect project quota by pulsing a cluster-wide backoff
                self.rate_limiter.mark_cluster_429(min(retry_after, 5.0))
            
            msg = f"🚫 Gemini {self.model}: 429 rate limited, backoff {retry_after:.0f}s"
            logger.warning(msg)
            raise GeminiProviderUnavailable(msg)
        elif status_code in {401, 403} or any(
            marker in lowered
            for marker in ("permission_denied", "api key", "leaked", "api_key_invalid")
        ):
            reason = f"Gemini {self.model}: provider_auth_failed_http_{status_code}"
            self._mark_provider_unavailable(reason, cooldown_s=24 * 60 * 60)
            logger.warning("%s", reason)
            raise GeminiProviderUnavailable(reason)
        else:
            # Structured fields only. 500 characters of raw body reached the
            # log AND the exception text that compatibility paths propagate,
            # and a content error carries the submitted content.
            msg = f"Gemini API error {status_code}: {self._safe_error_summary(text)}"
            logger.warning(msg)
            raise GeminiProviderUnavailable(msg)

    def _parse_retry_after(self, error_body: bytes) -> float:
        """Extract retry-after duration from a 429 error response."""
        try:
            text = error_body.decode('utf-8', errors='replace')
            match = re.search(r'retry in (\d+\.?\d*)', text)
            if match:
                return float(match.group(1))
        except GEMINI_RECOVERABLE_ERRORS as e:
            _record_gemini_degradation(
                e,
                action="used default Gemini retry-after backoff after parse failed",
                severity="debug",
            )
            capture_and_log(e, {'module': __name__})
        return 60.0  # Default 60s backoff
    
    @circuit_breaker(service_name="gemini-api")
    async def generate_text_stream_async(
        self, prompt: str, 
        system_prompt: str | None = None,
        cancel_event=None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream-compatible Gemini path routed through the canonical network gateway."""
        is_background = kwargs.get("is_background", False)
        if not self.is_available():
            raise GeminiProviderUnavailable(self.availability_reason())
        # Atomic reserve: admits AND counts the attempt in one step.
        if not self._reserve_quota(is_background):
            msg = f"🚫 Gemini {self.model} local rate limited"
            logger.warning(msg)
            raise GeminiProviderUnavailable(msg)

        if cancel_event and cancel_event.is_set():
            return

        try:
            injected_client = self._get_client()
            if injected_client is not None and hasattr(injected_client, "stream"):
                # Key in the auth header, not the URL (see _auth_headers).
                url = f"{self.BASE_URL}/models/{self.model}:streamGenerateContent?alt=sse"
                stream_payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
                if system_prompt:
                    stream_payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
                async with injected_client.stream(
                    "POST",
                    url,
                    json=stream_payload,
                    headers=self._auth_headers(),
                    timeout=self.timeout,
                ) as response:
                    if getattr(response, "status_code", 200) != 200:
                        await self._handle_error(response)
                    if hasattr(response, "aiter_lines"):
                        async for line in response.aiter_lines():
                            if cancel_event and cancel_event.is_set():
                                return
                            for chunk in _sse_text_chunks(line):
                                yield chunk
                    # Already counted at reserve time; do not double-count.
                    self._last_stream_was_incremental = True
                    return
            # No incremental network stream exists on this path. `call` runs
            # to completion and one full string is yielded, so there are no
            # early tokens and cancelling mid-answer stops nothing (CP126
            # ``fc2f4358``). The interface is kept because the router's
            # race_think_stream needs it; the claim is not.
            self._last_stream_was_incremental = False
            ok, text, metadata = await self.call(prompt, system_prompt=system_prompt, **kwargs)
            if not ok:
                raise GeminiProviderUnavailable(str(metadata.get("error") or "gemini_stream_call_failed"))
            if cancel_event and cancel_event.is_set():
                return
            yield text
        except GeminiProviderUnavailable as e:
            logger.warning("Gemini stream unavailable: %s", e)
            raise
        except httpx.TimeoutException as e:
            self._mark_provider_unavailable("gemini_stream_timeout", cooldown_s=300.0)
            _record_gemini_degradation(
                e,
                action="raised provider-unavailable signal so router can fail over after Gemini stream timeout",
                extra={"model": self.model},
            )
            logger.error("Gemini stream timeout: %s", e)
            raise GeminiProviderUnavailable(str(e)) from e
        except GEMINI_RECOVERABLE_ERRORS as e:
            _record_gemini_degradation(
                e,
                action="raised provider-unavailable signal so router can fail over after Gemini stream error",
                extra={"model": self.model},
            )
            logger.error("Gemini stream error: %s", e)
            raise GeminiProviderUnavailable(str(e)) from e

    @circuit_breaker(service_name="gemini-api")
    async def call(
        self, prompt: str, **kwargs
    ) -> tuple[bool, str, dict[str, Any]]:
        """Non-streaming call — compatible with LLMRouter's _think_internal fallback."""
        is_background = kwargs.get("is_background", False)
        if not self.is_available():
            return False, "", {"error": self.availability_reason()}
        # Atomic reserve: admits AND counts the attempt in one step (the
        # provider counts failed/cancelled requests too).
        if not self._reserve_quota(is_background):
            logger.warning("🚫 Gemini %s rate limited", self.model)
            return False, "", {"error": "Rate limit reached"}

        contents = []
        system_instruction = None
        
        sys_prompt = kwargs.get("system_prompt") or kwargs.get("system", "")
        if sys_prompt:
            system_instruction = {"parts": [{"text": sys_prompt}]}
        
        parts = kwargs.get("parts")
        if not parts:
            # Handle standard 'messages' format if provided
            messages = kwargs.get("messages")
            if messages and isinstance(messages, list):
                # Map to Gemini-style contents
                for msg in messages:
                    role = str(msg.get("role", "user") or "user").strip().lower()
                    content = msg.get("content", "")
                    if role == "system":
                        # Multiple system messages accumulate rather than the
                        # last silently overwriting earlier governance text.
                        if system_instruction is None:
                            system_instruction = {"parts": [{"text": content}]}
                        else:
                            system_instruction["parts"].append({"text": content})
                    elif role in {"user", "model", "assistant"}:
                        contents.append({
                            "role": "user" if role == "user" else "model",
                            "parts": [{"text": content}]
                        })
                    else:
                        # tool/function/developer/malformed/attacker-chosen
                        # roles are NOT silently promoted to model authority —
                        # they carry no authority and are dropped with a note.
                        _record_gemini_degradation(
                            ValueError(f"unrecognized_message_role:{role[:32]}"),
                            action="dropped a message with an unrecognized role from the Gemini request",
                            severity="warning",
                            extra={"model": self.model},
                        )
                
                # If we have contents, we skip the prompt-based construction below
                if contents:
                    parts = True # sentinel to skip next block
            
        if not parts:
            # Guard: prompt can be empty when user input is in system_prompt
            if prompt and prompt.strip():
                parts = [{"text": prompt}]
            elif sys_prompt:
                # The system prompt used to be MOVED into a user part and
                # systemInstruction cleared, so governance text arrived with
                # the same authority as anything the person typed and became
                # contestable by it (CP126 ``f3fe949e``). It keeps its place;
                # the user turn gets a minimal placeholder so the request is
                # valid.
                parts = [{"text": "Respond according to your instructions."}]
            else:
                return False, "", {"error": "No content to send to Gemini"}
        
        if contents:
            # Already built via messages
            pass  # no-op: intentional
        else:
            # Final guard: filter out any parts with empty/None text unless they have inlineData
            parts = [p for p in parts] if isinstance(parts, list) else []
            parts = [p for p in parts if p.get("text") or p.get("inlineData")]
            if not parts:
                return False, "", {"error": "All parts were empty after filtering"}
                
            contents.append({
                "role": "user",
                "parts": parts
            })
        
        # Validate generation params to finite, in-range values: an invalid
        # temperature or max_tokens forwarded verbatim can be rejected by the
        # provider (a wasted, quota-counted call) or produce undefined
        # sampling behavior.
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": _clamp_float(kwargs.get("temperature"), 0.8, 0.0, 2.0),
                "maxOutputTokens": _clamp_int(kwargs.get("max_tokens"), 2048, 1, 32768),
                "topP": 0.95,
            },
        }
        
        # [v11.0 HARDENING] Native JSON Mode & Schema Enforcement
        schema = kwargs.get("schema")
        if schema:
            payload["generationConfig"]["responseMimeType"] = "application/json"
            if isinstance(schema, dict):
                payload["generationConfig"]["responseSchema"] = schema
            logger.info("🎯 Gemini: JSON Mode ACTIVE with schema enforcement.")

        if system_instruction:
            payload["systemInstruction"] = system_instruction
        
        # Key is in the auth header (see _auth_headers), not the URL.
        url = f"{self.BASE_URL}/models/{self.model}:generateContent"
        
        metadata = {
            "model": self.model,
            "endpoint": "gemini_frontier",
            "latency_ms": 0,
        }
        
        t0 = time.monotonic()
        
        try:
            injected_client = self._get_client()
            if injected_client is not None and hasattr(injected_client, "post"):
                response = await injected_client.post(url, json=payload, timeout=self.timeout)
                metadata["latency_ms"] = int((time.monotonic() - t0) * 1000)
                if response.status_code != 200:
                    try:
                        await self._handle_error(response)
                    except GeminiProviderUnavailable as e:
                        return False, "", {"error": str(e)}
                    except GEMINI_RECOVERABLE_ERRORS as e:
                        _record_gemini_degradation(
                            e,
                            action="returned failed Gemini call result after provider error handler failed",
                            extra={"model": self.model, "status_code": response.status_code},
                        )
                        return False, "", {"error": str(e)}
                try:
                    data = response.json()
                except (json.JSONDecodeError, ValueError, AttributeError) as e:
                    return False, "", {"error": str(e)}
            else:
                status_code, data, body_text = await self._post_json(url, payload)
                metadata["latency_ms"] = int((time.monotonic() - t0) * 1000)

                if status_code != 200:
                    try:
                        await self._handle_error_payload(status_code, body_text)
                    except GeminiProviderUnavailable as e:
                        return False, "", {"error": str(e)}
                    except GEMINI_RECOVERABLE_ERRORS as e:
                        _record_gemini_degradation(
                            e,
                            action="returned failed Gemini call result after provider error handler failed",
                            extra={"model": self.model, "status_code": status_code},
                        )
                        return False, "", {"error": str(e)}

            # Already counted at reserve time; do not double-count.
            candidates = data.get("candidates", [])
            if not candidates:
                return False, "", {"error": "No candidates in response"}
            
            # Surface the candidate's terminal status instead of blindly
            # concatenating parts: a SAFETY/RECITATION block or a MAX_TOKENS
            # truncation is a FAILURE, not a clean answer, and a prompt-level
            # block (promptFeedback.blockReason) means no valid content at all.
            prompt_block = str(
                data.get("promptFeedback", {}).get("blockReason", "") or ""
            )
            if prompt_block:
                return False, "", {"error": f"gemini_prompt_blocked:{prompt_block}"}
            candidate = candidates[0]
            finish_reason = str(candidate.get("finishReason", "") or "")
            metadata["finish_reason"] = finish_reason
            if finish_reason in {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT"}:
                return False, "", {"error": f"gemini_finish_{finish_reason.lower()}"}

            parts = candidate.get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)

            # PROVIDER RECEIPT (CP126 inference-gate 8ff3084b /
            # llm_health_router 3bc237f4). Provenance was previously asserted
            # from the ROUTER's own endpoint record, so a misregistered,
            # proxied or deceptive client was described identically to a real
            # one. These fields come from the PROVIDER's response body, not
            # from our configuration: responseId is generated server-side and
            # modelVersion is the model that actually answered.
            response_id = str(data.get("responseId") or "").strip()
            model_version = str(data.get("modelVersion") or "").strip()
            if response_id or model_version:
                metadata["provider_receipt"] = {
                    "provider": "gemini",
                    "response_id": response_id,
                    "model_version": model_version,
                    "requested_model": str(kwargs.get("model") or self.model or ""),
                    "source": "provider_response_body",
                }
                # The model that ANSWERED must be the one we asked for.
                requested = str(kwargs.get("model") or self.model or "")
                if model_version and requested and not model_version.startswith(requested.split("-latest")[0]):
                    metadata["provider_receipt"]["model_version_mismatch"] = True

            # Extract token usage
            usage = data.get("usageMetadata", {})
            metadata["tokens_used"] = usage.get("totalTokenCount", 0)
            metadata["prompt_tokens"] = usage.get("promptTokenCount", 0)
            metadata["completion_tokens"] = usage.get("candidatesTokenCount", 0)
            metadata["truncated"] = finish_reason == "MAX_TOKENS"

            if not text:
                return False, "", {"error": "Empty response from Gemini"}

            # If a JSON schema was requested, VERIFY the output actually parses
            # as JSON rather than trusting the provider to have enforced it —
            # provider omission or an unsupported schema silently returns prose
            # that downstream JSON consumers then choke on.
            requested_schema = kwargs.get("schema")
            if requested_schema:
                try:
                    json.loads(text)
                    metadata["schema_verified"] = True
                except (json.JSONDecodeError, TypeError, ValueError):
                    metadata["schema_verified"] = False
                    return False, "", {"error": "gemini_schema_not_satisfied"}
            
            # [Phase 18] Record Metabolic Cost
            try:
                from core.ops.metabolic_monitor import get_cost_tracker
                get_cost_tracker().record_operation(
                    op_type="gemini_call",
                    tokens=metadata.get("tokens_used", 0),
                    duration_s=(time.monotonic() - t0),
                    model_tier="PRIMARY" if self.model == self.DEEP_MODEL else "SECONDARY"
                )
            except GEMINI_RECOVERABLE_ERRORS as _e:
                _record_gemini_degradation(
                    _e,
                    action="returned Gemini call result after metabolic cost recording failed",
                    severity="debug",
                    extra={"model": self.model, "tokens_used": metadata.get("tokens_used", 0)},
                )
                logger.debug('Ignored Exception in gemini_adapter.py: %s', _e)
            
            return True, text, metadata
            
        except TimeoutError:
            metadata["latency_ms"] = int((time.monotonic() - t0) * 1000)
            return False, "", {"error": f"Timeout after {self.timeout}s"}
        except GEMINI_RECOVERABLE_ERRORS as e:
            _record_gemini_degradation(
                e,
                action="returned failed Gemini call result after recoverable client error",
                extra={"model": self.model},
            )
            metadata["latency_ms"] = int((time.monotonic() - t0) * 1000)
            return False, "", {"error": str(e)}

    async def generate_text_async(
        self, prompt: str, 
        system_prompt: str | None = None,
        model: str | None = None,
        **kwargs
    ) -> str:
        """Compatibility method for code paths that call generate_text_async.

        The per-call ``model`` argument is honored (it was previously accepted
        and ignored, so a caller believed a requested model served while
        ``self.model`` was always used).
        """
        if system_prompt:
            kwargs["system_prompt"] = system_prompt
        return self._string_result(await self._call_with_model(prompt, model, kwargs))

    async def generate(
        self, prompt: str,
        system_prompt: str = "",
        **kwargs
    ) -> str:
        """Compatibility method for LLM router's generate()."""
        if system_prompt:
            kwargs["system_prompt"] = system_prompt
        return self._string_result(
            await self._call_with_model(prompt, kwargs.pop("model", None), kwargs)
        )

    async def _call_with_model(
        self, prompt: str, model: str | None, kwargs: dict[str, Any]
    ) -> tuple[bool, str, dict[str, Any]]:
        """Run one call, optionally against a requested model, stashing its
        telemetry so a failure is discoverable through the string API."""
        if model and model != self.model:
            original = self.model
            self.model = model
            try:
                result = await self.call(prompt, **kwargs)
            finally:
                self.model = original
        else:
            result = await self.call(prompt, **kwargs)
        success, text, meta = result
        provider_receipt = (meta or {}).get("provider_receipt")
        self._last_generation_metadata = {
            "provider": "gemini",
            "model": model or self.model,
            "ok": bool(success),
            **({"provider_receipt": dict(provider_receipt)} if isinstance(provider_receipt, dict) else {}),
            **({} if success else {"error": str((meta or {}).get("error", "gemini_call_failed"))}),
        }
        return result

    @staticmethod
    def _string_result(result: tuple[bool, str, dict[str, Any]]) -> str:
        success, text, _meta = result
        return text if success else ""

    async def think(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        **kwargs
    ) -> str | None:
        """
        Unified interface for non-chat callers.
        Returns the simplified string result from Gemini.
        """
        # Distinguish between prompt string and message list
        if not prompt and "messages" in kwargs:
            msgs = kwargs.get("messages", [])
            if msgs and isinstance(msgs, list):
                prompt = msgs[-1].get("content", "")
                if not system_prompt:
                    system_prompt = next((m["content"] for m in msgs if m["role"] == "system"), None)

        if not prompt:
            return None

        # Call generate with the prompt
        text = await self.generate(prompt, system_prompt=system_prompt, **kwargs)
        return text if text and text.strip() else None

    async def unload_models(self):
        """No-op for API models — nothing to unload."""
        return None
    
    async def generate_stream(self, prompt: str, system_prompt: str = None, **kwargs):
        """Alias for generate_text_stream_async — matches the interface expected by LLMRouter.think_stream."""
        async for chunk in self.generate_text_stream_async(prompt, system_prompt=system_prompt, **kwargs):
            yield chunk

    def get_usage_stats(self) -> dict:
        """Return human-readable usage stats for the UI."""
        return self.rate_limiter.get_usage()
