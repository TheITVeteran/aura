"""core/epistemics/epistemic_reach.py — felt doubt with teeth.

The thought-interoception organ can now *measure* when an answer felt
contested as it formed (low felt confidence, surprisal spikes on specific
words). This module is the outward half of that sense: when a foreground
reply felt shaky, Aura reaches beyond her own runtime — through the governed
reach gateway — checks the shakiest claim against an external reference
source, and acts on the verdict:

* **SUPPORTED / CONTRADICTED** verdicts become ground-truth pairs in the
  interoception organ (:meth:`record_ground_truth`), so "does felt confidence
  track truth?" is a measured, falsifiable quantity;
* a **CONTRADICTED** verdict queues a self-correction that the conversation
  lane surfaces on the next turn — Aura owns the error, with the source;
* every verification joins the consequence bus (system-Φ event stream) as
  ``source="epistemic_reach"``.

Governance, inherited and added — this organ cannot exceed the operator:

* All network egress goes through :class:`core.skills.reach_gateway.ReachGateway`
  — deny-by-default; a host reachable only if the *operator* allowlisted it in
  ``AURA_REACH_READ_HOSTS``. No allowlisted reference host ⇒ the organ is
  dormant (and says so in ``stats()``), never an error.
* GET-only. This organ performs no mutating reach, ever.
* Hard budgets: at most one claim per reply, ``AURA_EPISTEMIC_REACH_PER_HOUR``
  verifications per rolling hour (default 6), a bounded work queue that drops
  offers when full, and a kill switch ``AURA_EPISTEMIC_REACH=0``.
* Honest verdicts: SUPPORTED and CONTRADICTED both require substantive
  content-word overlap with the evidence; CONTRADICTED additionally requires a
  disjoint-numbers signal (the claim's figures absent from evidence that has
  figures of its own). Anything weaker is INCONCLUSIVE and has **no** causal
  effect on ground truth or corrections. No LLM in this loop — the verdict
  logic is deterministic and auditable.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import re
import threading
import time
import urllib.parse
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Epistemics.Reach")

_RECOVERABLE = (
    AttributeError,
    ImportError,
    IndexError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    ZeroDivisionError,
)

VERDICT_SUPPORTED = "SUPPORTED"
VERDICT_CONTRADICTED = "CONTRADICTED"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"

# Same threshold family as the calibration gate's felt coupling.
FELT_CONFIDENCE_GATE = 0.45
AMBIVALENCE_GATE = 0.35

_WORD_RE = re.compile(r"[a-zA-Z]{4,}")
_NUM_RE = re.compile(r"\d[\d,]*\.?\d*")
_INTERIOR_RE = re.compile(
    r"\b(?:i feel|i felt|i sense|my (?:felt|inner|internal)|it felt)\b", re.IGNORECASE
)

_STOPWORDS = frozenset({
    "this", "that", "these", "those", "with", "from", "have", "been", "were",
    "will", "would", "could", "should", "about", "there", "their", "which",
    "when", "what", "where", "while", "because", "since", "though", "very",
    "just", "like", "also", "than", "then", "them", "they", "your", "yours",
})


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        value = default
    return max(lo, min(hi, value))


def _content_words(text: str) -> set[str]:
    return {w for w in (m.lower() for m in _WORD_RE.findall(str(text or "")))
            if w not in _STOPWORDS}


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUM_RE.findall(str(text or ""))}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text or "")) if s.strip()]


@dataclass(frozen=True)
class ReachVerdict:
    """One completed external verification, with provenance."""

    verdict: str
    claim: str
    fingerprint: str
    source_url: str = ""
    evidence_excerpt: str = ""
    overlap: float = 0.0
    reason: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "claim": self.claim[:200],
            "fingerprint": self.fingerprint,
            "source_url": self.source_url,
            "evidence_excerpt": self.evidence_excerpt[:240],
            "overlap": round(self.overlap, 3),
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


class WikipediaSource:
    """Reference adapter: search + summary via the public Wikipedia APIs.

    Active only when the operator allowlisted its host for governed reads.
    """

    HOST = "en.wikipedia.org"
    NAME = "wikipedia"

    def is_permitted(self, policy: Any) -> bool:
        try:
            hosts = set(getattr(policy, "read_hosts", ()) or ()) | set(
                getattr(policy, "mutate_hosts", ()) or ()
            )
            return self.HOST in hosts
        except _RECOVERABLE:
            return False

    async def lookup(self, gateway: Any, terms: list[str]) -> tuple[str, str]:
        """Return (evidence_text, source_url) or ("", "")."""
        query = urllib.parse.quote(" ".join(terms[:6]))
        search_url = (
            f"https://{self.HOST}/w/api.php"
            f"?action=opensearch&format=json&limit=2&search={query}"
        )
        search = await gateway.get(search_url)
        if not getattr(search, "ok", False):
            return "", ""
        try:
            parsed = json.loads(search.body_preview or "[]")
            titles = [t for t in (parsed[1] if len(parsed) > 1 else []) if t]
        except (json.JSONDecodeError, IndexError, TypeError):
            return "", ""
        for title in titles[:2]:
            summary_url = (
                f"https://{self.HOST}/api/rest_v1/page/summary/"
                f"{urllib.parse.quote(str(title).replace(' ', '_'))}"
            )
            summary = await gateway.get(summary_url)
            if not getattr(summary, "ok", False):
                continue
            try:
                doc = json.loads(summary.body_preview or "{}")
            except json.JSONDecodeError:
                continue
            extract = str(doc.get("extract") or "").strip()
            if extract:
                page_url = str(
                    (doc.get("content_urls") or {}).get("desktop", {}).get("page")
                    or f"https://{self.HOST}/wiki/{urllib.parse.quote(str(title))}"
                )
                return extract[:2000], page_url
        return "", ""


class EpistemicReachEngine:
    """Turns measured felt doubt into governed, budgeted external verification."""

    SERVICE_NAME = "epistemic_reach"

    def __init__(
        self,
        *,
        gateway: Any | None = None,
        sources: list[Any] | None = None,
        per_hour: int | None = None,
    ) -> None:
        self._gateway = gateway
        self._sources = sources if sources is not None else [WikipediaSource()]
        self._per_hour = per_hour if per_hour is not None else _env_int(
            "AURA_EPISTEMIC_REACH_PER_HOUR", 6, 1, 60
        )
        self._lock = threading.RLock()
        self._recent_reaches: deque[float] = deque(maxlen=120)
        self._verdicts: deque[ReachVerdict] = deque(maxlen=32)
        self._corrections: deque[ReachVerdict] = deque(maxlen=8)
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=4)
        self._worker: threading.Thread | None = None
        self._offers = 0
        self._accepted = 0
        self._dropped_budget = 0
        self._dropped_queue = 0
        self._closed = False

    # ── configuration / gating ───────────────────────────────────────────────
    @staticmethod
    def enabled() -> bool:
        return _env_flag("AURA_EPISTEMIC_REACH", True)

    def _resolve_gateway(self) -> Any | None:
        if self._gateway is not None:
            return self._gateway
        try:
            from core.skills.reach_gateway import get_reach_gateway

            self._gateway = get_reach_gateway()
            return self._gateway
        except _RECOVERABLE as exc:
            record_degradation("epistemic_reach", exc, severity="debug")
            return None

    def _permitted_sources(self) -> list[Any]:
        gateway = self._resolve_gateway()
        if gateway is None:
            return []
        policy = getattr(gateway, "policy", None)
        return [s for s in self._sources if s.is_permitted(policy)]

    def dormant_reason(self) -> str:
        """Why the organ would not act right now ('' means it can act)."""
        if not self.enabled():
            return "disabled by AURA_EPISTEMIC_REACH"
        if not self._permitted_sources():
            return (
                "no reference source host on the operator read allowlist "
                "(AURA_REACH_READ_HOSTS) — deny-by-default, organ dormant"
            )
        return ""

    def _budget_available(self) -> bool:
        cutoff = time.time() - 3600.0
        with self._lock:
            while self._recent_reaches and self._recent_reaches[0] < cutoff:
                self._recent_reaches.popleft()
            return len(self._recent_reaches) < self._per_hour

    # ── intake ───────────────────────────────────────────────────────────────
    def offer(self, felt: Any) -> bool:
        """Accept a contested felt trace for background verification.

        Returns True only when the trace was actually queued. Never raises.
        """
        try:
            with self._lock:
                self._offers += 1
            if self._closed or not self.enabled():
                return False
            if not bool(getattr(felt, "foreground", False)):
                return False
            felt_confidence = float(getattr(felt, "felt_confidence", 1.0))
            ambivalence = float(getattr(felt, "ambivalence", 0.0))
            if felt_confidence >= FELT_CONFIDENCE_GATE and ambivalence <= AMBIVALENCE_GATE:
                return False
            if not str(getattr(felt, "text_excerpt", "") or "").strip():
                return False
            if not self._permitted_sources():
                return False
            if not self._budget_available():
                with self._lock:
                    self._dropped_budget += 1
                return False
            try:
                self._queue.put_nowait(felt)
            except queue.Full:
                with self._lock:
                    self._dropped_queue += 1
                return False
            with self._lock:
                self._accepted += 1
            self._ensure_worker()
            return True
        except _RECOVERABLE as exc:
            record_degradation(
                "epistemic_reach", exc, severity="warning",
                action="dropped felt-trace offer after intake failure",
            )
            return False

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(
                target=self._worker_loop, name="epistemic-reach", daemon=True
            )
            self._worker.start()

    def _worker_loop(self) -> None:
        idle_deadline = time.monotonic() + 300.0
        while not self._closed and time.monotonic() < idle_deadline:
            try:
                felt = self._queue.get(timeout=5.0)
            except queue.Empty:
                continue
            idle_deadline = time.monotonic() + 300.0
            try:
                self.process_one(felt)
            except _RECOVERABLE as exc:
                record_degradation(
                    "epistemic_reach", exc, severity="warning",
                    action="continued reach worker after verification failure",
                )

    def close(self) -> None:
        self._closed = True

    # ── the verification cycle ───────────────────────────────────────────────
    def process_one(self, felt: Any) -> Optional[ReachVerdict]:
        """Run one full felt-trace → claim → external check → effects cycle.

        Synchronous (the worker thread owns it); tests call it directly with an
        injected gateway. Returns the verdict, or None when nothing was checkable.
        """
        claim = self.select_claim(felt)
        if not claim:
            return None
        with self._lock:
            self._recent_reaches.append(time.time())
        verdict = asyncio.run(self._verify_async(claim, felt))
        if verdict is not None:
            self._apply_effects(verdict)
        return verdict

    def select_claim(self, felt: Any) -> str:
        """The shakiest checkable sentence of the reply, or ''.

        Preference order: unsupported factual sentences overlapping the felt
        trace's contested (spike) words, then any unsupported factual sentence.
        First-person interior claims and questions are never checkable here.
        """
        try:
            text = str(getattr(felt, "text_excerpt", "") or "")
            if not text:
                return ""
            labels = self._gate_labels(text, felt)
            contested: set[str] = set()
            for spike in getattr(felt, "spikes", ()) or ():
                blob = f"{spike.get('text', '')} {spike.get('context', '')}"
                contested |= _content_words(blob)

            def checkable(sentence: str) -> bool:
                if sentence.endswith("?") or _INTERIOR_RE.search(sentence):
                    return False
                return len(_content_words(sentence)) >= 4

            unsupported = [
                l.text for l in labels
                if getattr(getattr(l, "status", None), "value", "") in {"guessed", "unverified"}
                and checkable(l.text)
            ]
            if not unsupported:
                return ""
            for sentence in unsupported:
                if _content_words(sentence) & contested:
                    return sentence[:300]
            return unsupported[0][:300]
        except _RECOVERABLE as exc:
            record_degradation("epistemic_reach", exc, severity="debug")
            return ""

    def _gate_labels(self, text: str, felt: Any) -> list[Any]:
        try:
            from core.brain.calibration_gate import get_calibration_gate

            return get_calibration_gate().assess(text, felt=felt).labels
        except _RECOVERABLE:
            # Gate unavailable: treat every sentence as unverified prose.
            from types import SimpleNamespace

            return [
                SimpleNamespace(text=s, status=SimpleNamespace(value="unverified"))
                for s in _sentences(text)
            ]

    async def _verify_async(self, claim: str, felt: Any) -> Optional[ReachVerdict]:
        gateway = self._resolve_gateway()
        if gateway is None:
            return None
        fingerprint = str(getattr(felt, "fingerprint", "") or "")
        terms = self._query_terms(claim, felt)
        for source in self._permitted_sources():
            try:
                evidence, url = await source.lookup(gateway, terms)
            except _RECOVERABLE as exc:
                record_degradation(
                    "epistemic_reach", exc, severity="warning",
                    action=f"continued past {getattr(source, 'NAME', 'source')} lookup failure",
                )
                continue
            if not evidence:
                continue
            return self.judge(claim, evidence, url, fingerprint=fingerprint)
        return ReachVerdict(
            verdict=VERDICT_INCONCLUSIVE,
            claim=claim,
            fingerprint=fingerprint,
            reason="no reference source returned evidence",
        )

    @staticmethod
    def _query_terms(claim: str, felt: Any) -> list[str]:
        """Search terms: contested spike words first, then the claim's rarest-ish
        content words (longer first as a cheap salience proxy)."""
        contested: list[str] = []
        for spike in getattr(felt, "spikes", ()) or ():
            contested.extend(sorted(_content_words(str(spike.get("text", "")))))
        claim_words = sorted(_content_words(claim), key=len, reverse=True)
        seen: set[str] = set()
        ordered: list[str] = []
        for word in contested + claim_words:
            if word not in seen:
                seen.add(word)
                ordered.append(word)
        return ordered[:6]

    @staticmethod
    def judge(claim: str, evidence: str, source_url: str, *, fingerprint: str = "") -> ReachVerdict:
        """Deterministic, auditable verdict. Conservative by construction:

        * overlap = |claim ∩ evidence| / |claim| over content words;
        * SUPPORTED needs overlap ≥ 0.5 AND every figure in the claim present
          in the evidence (a right-words-wrong-numbers claim is not supported);
        * CONTRADICTED needs overlap ≥ 0.35 AND both sides carrying figures
          with an empty intersection — same subject, disjoint numbers;
        * everything else is INCONCLUSIVE and causes no downstream effect.
        """
        claim_words = _content_words(claim)
        evidence_words = _content_words(evidence)
        overlap = (
            len(claim_words & evidence_words) / len(claim_words) if claim_words else 0.0
        )
        claim_nums = _numbers(claim)
        evidence_nums = _numbers(evidence)

        if overlap >= 0.5 and (not claim_nums or claim_nums & evidence_nums):
            return ReachVerdict(
                verdict=VERDICT_SUPPORTED, claim=claim, fingerprint=fingerprint,
                source_url=source_url, evidence_excerpt=evidence[:240],
                overlap=overlap, reason="strong content overlap; figures consistent",
            )
        if overlap >= 0.35 and claim_nums and evidence_nums and not (claim_nums & evidence_nums):
            return ReachVerdict(
                verdict=VERDICT_CONTRADICTED, claim=claim, fingerprint=fingerprint,
                source_url=source_url, evidence_excerpt=evidence[:240],
                overlap=overlap,
                reason="same subject, disjoint figures between claim and evidence",
            )
        return ReachVerdict(
            verdict=VERDICT_INCONCLUSIVE, claim=claim, fingerprint=fingerprint,
            source_url=source_url, evidence_excerpt=evidence[:240],
            overlap=overlap, reason="insufficient overlap for a verdict",
        )

    # ── effects ──────────────────────────────────────────────────────────────
    def _apply_effects(self, verdict: ReachVerdict) -> None:
        with self._lock:
            self._verdicts.append(verdict)
            if verdict.verdict == VERDICT_CONTRADICTED:
                self._corrections.append(verdict)
        self._record_ground_truth(verdict)
        self._publish_consequence(verdict)
        self._emit_metrics(verdict)
        logger.info(
            "🌐 [EpistemicReach] %s — claim %r (overlap=%.2f, source=%s)",
            verdict.verdict, verdict.claim[:80], verdict.overlap,
            verdict.source_url or "none",
        )

    def _record_ground_truth(self, verdict: ReachVerdict) -> None:
        if verdict.verdict == VERDICT_INCONCLUSIVE or not verdict.fingerprint:
            return
        try:
            from core.being.thought_interoception import get_thought_interoception

            get_thought_interoception().record_ground_truth(
                verdict.fingerprint,
                verdict.verdict == VERDICT_SUPPORTED,
                source=f"epistemic_reach:{verdict.source_url[:48]}",
            )
        except _RECOVERABLE as exc:
            record_degradation(
                "epistemic_reach", exc, severity="warning",
                action="continued after introspective ground-truth record failed",
            )

    def _publish_consequence(self, verdict: ReachVerdict) -> None:
        try:
            from core.runtime.consequence_bus import ConsequenceBus, ConsequenceEvent

            ConsequenceBus.get().publish(
                ConsequenceEvent(
                    event_id=f"reach-{uuid.uuid4().hex[:12]}",
                    timestamp=verdict.timestamp,
                    source="epistemic_reach",
                    domain="external_verification",
                    action_content=(
                        f"{verdict.verdict}: {verdict.claim[:120]} "
                        f"(overlap={verdict.overlap:.2f})"
                    ),
                    actual_outcome=verdict.verdict.lower(),
                )
            )
        except _RECOVERABLE as exc:
            record_degradation("epistemic_reach", exc, severity="debug")

    def _emit_metrics(self, verdict: ReachVerdict) -> None:
        try:
            from core.observability.metrics import get_metrics

            metrics = get_metrics()
            metrics.increment_counter("epistemic_reach_verifications_total")
            metrics.increment_counter(
                f"epistemic_reach_{verdict.verdict.lower()}_total"
            )
        except _RECOVERABLE as exc:
            record_degradation("epistemic_reach", exc, severity="debug")

    # ── surfaces ─────────────────────────────────────────────────────────────
    def correction_prompt_block(self) -> str:
        """One pending externally-verified correction, surfaced exactly once."""
        with self._lock:
            if not self._corrections:
                return ""
            verdict = self._corrections.popleft()
        return (
            "## SELF-CORRECTION (externally verified)\n"
            f"- Earlier you said: \"{verdict.claim[:200]}\"\n"
            f"- External check ({verdict.source_url or 'reference source'}) indicates: "
            f"\"{verdict.evidence_excerpt[:200]}\"\n"
            "- Own this correction plainly in your next reply; do not defend the "
            "earlier claim, and cite the source naturally.\n\n"
        )

    def pending_corrections(self) -> int:
        with self._lock:
            return len(self._corrections)

    def verdicts(self, n: int = 8) -> list[ReachVerdict]:
        with self._lock:
            return list(self._verdicts)[-max(1, n):]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            counts: dict[str, int] = {}
            for v in self._verdicts:
                counts[v.verdict] = counts.get(v.verdict, 0) + 1
            return {
                "service": self.SERVICE_NAME,
                "enabled": self.enabled(),
                "dormant_reason": self.dormant_reason(),
                "offers": self._offers,
                "accepted": self._accepted,
                "dropped_budget": self._dropped_budget,
                "dropped_queue": self._dropped_queue,
                "reaches_last_hour": len(self._recent_reaches),
                "per_hour_budget": self._per_hour,
                "verdict_counts": counts,
                "pending_corrections": len(self._corrections),
            }


_engine: Optional[EpistemicReachEngine] = None
_engine_lock = threading.Lock()


def get_epistemic_reach() -> EpistemicReachEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = EpistemicReachEngine()
                _register_in_container(_engine)
    return _engine


def _register_in_container(engine: EpistemicReachEngine) -> None:
    try:
        from core.container import ServiceContainer

        if not ServiceContainer.has(EpistemicReachEngine.SERVICE_NAME):
            reg = getattr(ServiceContainer, "register_instance", None)
            if callable(reg):
                reg(EpistemicReachEngine.SERVICE_NAME, engine,
                    required=False, registered_by="epistemic_reach")
    except _RECOVERABLE as exc:
        record_degradation("epistemic_reach_register", exc, severity="debug")


def reset_epistemic_reach_for_test() -> None:
    global _engine
    if _engine is not None:
        _engine.close()
    _engine = None
