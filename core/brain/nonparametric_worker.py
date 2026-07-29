"""KV-cached foreground wiring for non-parametric memory — the latency-correct form.

The validation-grade processor in ``nonparametric_generation`` recomputes a full forward
over the running tokens every step just to recover the hidden-state query key. That is
O(n²) and would make the 32B foreground take minutes — which is exactly why it was kept out
of the live response path.

This module closes that gap. The model's *normal* generation forward (the one
``stream_generate`` already runs with a KV cache, O(1) per token) computes the hidden state
we need. We capture it as a side effect of that forward instead of recomputing:

  * ``HiddenStateTap`` wraps ``model.model`` (the inner transformer that returns hidden
    states) with a transparent proxy that records the last-token hidden each call. Calling
    the model during generation therefore fills ``tap.last_key`` for free.
  * ``make_tapped_nonparametric_processor`` is a standard ``(tokens, logits) -> logits``
    mlx_lm logits-processor that reads ``tap.last_key`` — no extra forward — and interpolates
    the non-parametric recall into the logits. If the tap has nothing (structure mismatch,
    first call), it returns the logits unchanged: fail-open, and crucially **never** falls
    back to the O(n²) recompute on the foreground path.

``cached_generate_with_memory`` is the standalone O(n) reference loop (real KV cache, one
forward per token) used to validate the mechanism end-to-end without the worker.

Everything is fail-open: a tap that can't install, a model whose head can't be found, or any
per-token error leaves generation exactly as it would have been without memory.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
import pathlib
import time
from typing import Any

import numpy as np

from core.brain.nonparametric_generation import normalize
from core.runtime.errors import record_degradation

logger = logging.getLogger("Brain.NonParametricWorker")


def foreground_enabled() -> bool:
    """Whether the foreground non-parametric memory path is switched on.

    Default ON since the July end-to-end proof (tools/nonparametric_proof.py):
    one-shot recall of session-random facts verified on the real model with
    the anisotropy-corrected gate, and an unrelated control generation
    byte-identical with the datastore loaded. Every layer stays fail-open
    (empty store = no processor; below-gate similarity = untouched logits).
    Kill switch: AURA_NONPARAMETRIC_FOREGROUND=0.
    """
    from core.runtime.flags import FlagKind, declare

    return bool(
        declare(
            "AURA_NONPARAMETRIC_FOREGROUND",
            kind=FlagKind.BOOL,
            default=True,
            description="Foreground non-parametric recall blend (proven by tools/nonparametric_proof.py)",
            owner="core.brain.nonparametric_worker",
        ).value()
    )


_STRUCTURAL_OUTPUT_CONTRACT_KINDS = frozenset(
    {
        "exact_reply",
        "list_count",
        "paragraph_count",
        "sentence_count",
        "word_count",
    }
)


def foreground_memory_admitted_for_job(job: Any) -> bool:
    """Keep token-level recall from competing with structural decoding.

    A structural contract is a decoder constraint, not a retrieval request.
    The non-parametric store may still participate when the caller explicitly
    declares that memory grounding is required; otherwise Aura's ordinary
    model and response-contract machinery own the turn.
    """

    if not isinstance(job, dict):
        return True
    contract = job.get("requested_output_contract")
    kind = (
        str(contract.get("kind") or "").strip().lower()
        if isinstance(contract, dict)
        else ""
    )
    if kind not in _STRUCTURAL_OUTPUT_CONTRACT_KINDS:
        return True
    return bool(
        job.get("requires_memory_grounding")
        or job.get("memory_state_contract")
        or job.get("grounded_recall_contract")
    )


# Last foreground-recall outcome, so a request can report whether memory was
# installed, deliberately skipped, empty, or failed to build. Every path
# returned None before, which made those indistinguishable (CP126 29374cf0).
_RECALL_OUTCOME: dict[str, Any] = {"status": "not_attempted", "detail": ""}


def _set_recall_outcome(status: str, detail: str = "") -> None:
    _RECALL_OUTCOME["status"] = status
    _RECALL_OUTCOME["detail"] = detail


def last_recall_outcome() -> dict[str, Any]:
    """What the most recent foreground-recall build actually did.

    status is one of: not_attempted, disabled, not_admitted, unavailable,
    empty, installed, failed.
    """
    return dict(_RECALL_OUTCOME)


# A datastore has to be able to answer before it is allowed to speak.
#
# LIVE DEFECT, 2026-07-26. The resident 32B served fluent, grammatical,
# meaning-free replies to ordinary questions — and kept serving them with
# substrate steering clamped to 0.01 and recurrent depth off, which is what
# ruled both of those out as the cause:
#
#   "Define S as extracting for Draw [w] from colored ([I:E]): (card frequency
#    in the bag - matching cards already know to counted)"
#
# The live datastore (~/.aura/data/runtime/nonparametric_memory_5120, built
# 2026-07-13) holds 1,689 hidden-state keys of which 1,677 — 99.3% — carry no
# decoded token text at all. Their token_ids are the most ordinary tokens in
# the vocabulary: space, digits, "the", "is", "to". The remaining 12 store an
# entire ANSWER as a single "token", which is not what a per-token kNN store
# holds either.
#
# Blending THAT into the model's top-64 logits, at a weight that reaches 0.87,
# is a recipe for text that is grammatically shaped and says nothing — which
# is precisely the failure. Recall is an enhancement; a store that cannot
# support recall must decline, and the module's stated contract is already to
# fail open to normal generation.
_MIN_USABLE_ENTRY_FRACTION = 0.5
_MIN_USABLE_ENTRIES = 32


#: Refusals already reported, so a permanently-unusable store is named once
#: per process rather than once per turn. The 2026-07-13 store produced 591
#: identical warnings in a single session — the guard was working and the log
#: was the only thing that suffered.
_REPORTED_UNUSABLE: set[str] = set()


def _quarantine_unusable_datastore(memory: Any, reason: str) -> str:
    """Move a provably-unusable store aside so a good one can be built.

    Renamed, never deleted: the files are evidence of how the store went wrong
    and they are the user's data. What matters is that they stop being loaded,
    because the guard below is permanent — nothing else retires the store, so
    without this the faculty is dark for every future session too.
    """
    raw_path = str(getattr(memory, "_path", "") or "")
    if not raw_path:
        return ""
    base = pathlib.Path(raw_path).expanduser()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    moved: list[str] = []
    for suffix in (".keys.npy", ".meta.json"):
        source = base.with_name(base.name + suffix)
        if not source.exists():
            continue
        target = source.with_name(f"{source.name}.unusable-{stamp}")
        try:
            source.rename(target)
            moved.append(target.name)
        except OSError as exc:
            logger.warning("Could not quarantine %s: %s", source, exc)
            return ""
    if not moved:
        return ""
    logger.warning(
        "Non-parametric memory: quarantined an unusable datastore (%s). "
        "Renamed to %s; a fresh store will build from this session onward.",
        reason,
        ", ".join(moved),
    )
    return ", ".join(moved)


def _unusable_datastore_reason(memory: Any) -> str:
    """Why this datastore may not steer live generation, or "" if it may."""
    try:
        tokens = list(getattr(memory, "_tokens", None) or [])
    except (AttributeError, TypeError, ValueError):
        return ""
    total = len(tokens)
    if total == 0:
        return ""
    usable = sum(1 for token in tokens if str(token or "").strip())
    reason = ""
    if usable < _MIN_USABLE_ENTRIES:
        reason = (
            f"only {usable} of {total} entries carry a recallable token "
            f"(need at least {_MIN_USABLE_ENTRIES})"
        )
    elif usable < total * _MIN_USABLE_ENTRY_FRACTION:
        reason = (
            f"{total - usable} of {total} entries carry no recallable token "
            f"({100.0 * usable / total:.1f}% usable)"
        )
    if not reason:
        return ""
    # SAY IT ONCE, AND THEN DO SOMETHING ABOUT IT.
    #
    # This condition cannot improve on its own — the store on disk is what it
    # is — so repeating the verdict every turn is noise and leaving the store
    # in place keeps the faculty dark forever. Quarantine it once and let a
    # fresh one accumulate.
    key = f"{str(getattr(memory, '_path', '') or '?')}:{total}:{usable}"
    if key not in _REPORTED_UNUSABLE:
        _REPORTED_UNUSABLE.add(key)
        _quarantine_unusable_datastore(memory, reason)
    return reason


def maybe_build_foreground(
    model: Any,
    *,
    job: Any = None,
) -> tuple[HiddenStateTap, Callable[[Any, Any], Any]] | None:
    """Build (tap, processor) for the live worker iff foreground memory is on and non-empty.

    Returns None when disabled, when there is no datastore, or when the datastore is empty —
    so the live path pays nothing (no tap, no processor) unless there is genuinely something
    to recall. Fully fail-open: any error returns None and the worker generates normally.
    """
    if not foreground_enabled():
        _set_recall_outcome("disabled", "foreground recall is switched off")
        return None
    if not foreground_memory_admitted_for_job(job):
        _set_recall_outcome("not_admitted", "this job did not admit foreground recall")
        return None
    try:
        dim = int(getattr(getattr(model, "args", None), "hidden_size", 0) or 0)
        if dim <= 0:
            _set_recall_outcome("unavailable", "model exposes no hidden_size")
            return None
        from core.brain.nonparametric_memory import get_nonparametric_memory
        memory = get_nonparametric_memory(dim)
        if memory is None:
            _set_recall_outcome("unavailable", "no datastore")
            return None
        if len(memory) == 0:
            _set_recall_outcome("empty", "datastore holds no entries")
            return None
        unusable = _unusable_datastore_reason(memory)
        if unusable:
            _set_recall_outcome("not_admitted", unusable)
            logger.warning(
                "🧠 [WORKER] Foreground non-parametric memory REFUSED: %s. "
                "Generating from the model alone.",
                unusable,
            )
            return None
        tap = HiddenStateTap(model)
        proc = make_tapped_nonparametric_processor(tap, memory)
        logger.info("🧠 [WORKER] Foreground non-parametric memory ACTIVE (%d entries, dim=%d).",
                    len(memory), dim)
        _set_recall_outcome("installed", f"{len(memory)} entries at dim {dim}")
        return tap, proc
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        # CP126 29374cf0. Failing open is RIGHT here — recall is an
        # enhancement, and generating normally without it is the correct
        # degradation. What was missing is that the caller could not tell
        # installed from skipped from broken: every path returned None, and
        # a genuine build error was filed at debug alongside routine
        # "nothing to recall".
        #
        # The outcome is now recorded so a request can say which happened,
        # and a real failure is a warning rather than debug noise.
        _set_recall_outcome("failed", f"{type(exc).__name__}: {exc}")
        record_degradation(
            "nonparametric_foreground_build",
            exc,
            severity="warning",
            action="foreground non-parametric memory not installed; normal generation",
        )
        return None


# ── head detection: hidden states → logits, model-agnostic ──────────────────

def _logits_from_hidden(model: Any, hidden: Any) -> Any:
    """Project hidden states to vocab logits using the model's own (possibly tied) head."""
    args = getattr(model, "args", None)
    if getattr(args, "tie_word_embeddings", False):
        inner = getattr(model, "model", None)
        embed = getattr(inner, "embed_tokens", None)
        if embed is not None and hasattr(embed, "as_linear"):
            return embed.as_linear(hidden)
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None:
        return lm_head(hidden)
    # Last resort: a tied embedding without the flag set.
    inner = getattr(model, "model", None)
    embed = getattr(inner, "embed_tokens", None)
    if embed is not None and hasattr(embed, "as_linear"):
        return embed.as_linear(hidden)
    raise AttributeError("could not locate the model's output head")


# ── the tap: capture the hidden the generation forward already computes ──────

class _TappedInner:
    """Transparent proxy around model.model that records the last-token hidden per call."""

    def __init__(self, inner: Any, tap: HiddenStateTap) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_tap", tap)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = self._inner(*args, **kwargs)
        try:
            # out is hidden states [B, T, H]; record the last position of the last call.
            self._tap.last_key = normalize(np.array(out[0, -1], dtype=np.float32))
        except (IndexError, ValueError, TypeError):
            self._tap.last_key = None
        return out

    # Delegate everything else so the proxy is indistinguishable from the real module.
    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_inner"), name)


class HiddenStateTap:
    """Installs a recording proxy around ``model.model`` for the span of a generation.

    Use as a context manager. If the swap can't be done (unexpected structure, mlx setattr
    quirk), ``active`` stays False and the tap simply records nothing — the caller's
    processor then no-ops, so generation is unaffected.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self._real_inner: Any = None
        self.active = False
        self.last_key: np.ndarray | None = None

    def __enter__(self) -> HiddenStateTap:
        try:
            inner = getattr(self._model, "model", None)
            if inner is None or not callable(inner):
                return self
            self._real_inner = inner
            self._model.model = _TappedInner(inner, self)
            self.active = True
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("nonparametric_worker_tap", exc, severity="debug",
                               action="hidden-state tap disabled; foreground memory inert")
            self.active = False
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._real_inner is not None:
            try:
                self._model.model = self._real_inner
            except (AttributeError, RuntimeError, TypeError, ValueError) as e:
                record_degradation("nonparametric_worker_tap", e, severity="warning",
                                   action="failed to restore model.model after tap")
        self._real_inner = None
        self.active = False


# ── the foreground processor: reads the tap, no recompute ───────────────────

def make_tapped_nonparametric_processor(
    tap: HiddenStateTap,
    memory: Any,
    *,
    k: int = 4,
    temperature: float = 0.1,
    phi: float | None = 0.5,
    free_energy: float | None = 0.7,
    min_cos: float = 0.55,
    # A retrieved neighbour may INFORM the next token; it may not choose it.
    # base_lam was 0.75 and the free-energy term below multiplies by up to
    # 1.16, so a single neighbour could take 87% of the next-token
    # distribution away from the model — against ~0.25 in the kNN-LM
    # literature, and tuned there on held-out data rather than asserted.
    # At 0.87 the datastore is not augmenting generation, it is performing it.
    base_lam: float = 0.25,
    max_lam: float = 0.35,
) -> Callable[[Any, Any], Any]:
    """O(1)-per-token non-parametric logits-processor driven by the hidden-state tap."""
    import mlx.core as mx

    state = {"last_fired_index": -1}

    def _proc(tokens: Any, logits: Any) -> Any:
        key = tap.last_key
        if key is None:
            return logits  # tap inactive / no hidden yet → leave logits untouched (fail-open)
        try:
            neighbors = memory.query(key, k=k)
            if not neighbors:
                return logits
            # Anisotropy-corrected gate (see Neighbor.similarity): raw cosine
            # cannot separate unrelated prompts on real hidden states.
            sim = float(getattr(neighbors[0], "similarity", -1.0))
            gate = float(getattr(memory, "min_similarity", lambda: min_cos)())
            nearest_index = int(getattr(neighbors[0], "index", -1))
            if sim < gate:
                return logits
            # Anti-stutter: the same nearest entry twice in a row means the
            # recalled chain ended and its tail is re-firing.
            if nearest_index == state["last_fired_index"]:
                return logits
            state["last_fired_index"] = nearest_index
            fe = 0.5 if free_energy is None else float(free_energy)
            lam = base_lam * ((sim - gate) / max(1e-6, 1.0 - gate)) * (0.6 + 0.8 * fe)
            lam = min(float(max_lam), max(0.0, float(lam)))
            lg = np.array(logits, dtype=np.float32).reshape(-1)
            ktop = min(64, lg.shape[0])
            idx = np.argpartition(lg, -ktop)[-ktop:]
            sub = lg[idx] - lg[idx].max()
            ex = np.exp(sub)
            ex /= ex.sum()
            lm_probs = {int(t): float(p) for t, p in zip(idx, ex, strict=True)}
            blended = memory.interpolate(
                lm_probs, key, k=k, temperature=temperature, phi=phi,
                free_energy=free_energy, lam_override=min(lam, 0.9),
            )
            out = lg.copy()
            import math as _m

            for t, p in blended.items():
                out[int(t)] = _m.log(max(p, 1e-12))
            return mx.array(out).reshape(logits.shape)
        except (RuntimeError, ValueError, TypeError, AttributeError, IndexError) as exc:
            record_degradation("nonparametric_tapped_processor", exc)
            return logits

    return _proc


# ── standalone O(n) reference loop (validation without the worker) ──────────

def cached_generate_with_memory(
    model: Any,
    tokenizer: Any,
    prompt: str,
    memory: Any,
    *,
    max_tokens: int = 40,
    k: int = 4,
    # kNN softmax temperature over UNIT-key L2 distances: exact match d=0
    # must dominate an unrelated entry at d≈0.35, so the scale is ~0.1 —
    # the old 2.0 made the kNN distribution nearly uniform across entries
    # (measured: cross-fact digit leakage corrupted recall).
    temperature: float = 0.1,
    phi: float | None = 0.5,
    free_energy: float | None = 0.7,
    use_memory: bool = True,
    min_cos: float = 0.55,
    base_lam: float = 0.75,
) -> str:
    """Greedy generation with a real KV cache: one incremental forward per token.

    The hidden key for step t is read from the *same* cached forward that produces the
    step-t logits, so adding non-parametric recall costs a datastore query, not a forward.
    This is the latency-correct shape the foreground tap mirrors. Fail-open per token.
    """
    import mlx.core as mx

    try:
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(model)
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("nonparametric_cached_generate", exc,
                           action="KV cache unavailable; aborting cached generation")
        return ""

    ids = list(tokenizer.encode(prompt))
    eos = getattr(tokenizer, "eos_token_id", None)
    out_ids: list[int] = []
    last_fired_index = -1  # anti-stutter: an entry may not fire twice in a row

    # Prefill the cache with the full prompt, then decode one token at a time.
    cursor = mx.array([ids])
    for _step in range(max(1, int(max_tokens))):
        try:
            hidden = model.model(cursor, cache=cache)           # cached forward (incremental)
            logits = _logits_from_hidden(model, hidden)
            key = normalize(np.array(hidden[0, -1], dtype=np.float32))
            lg = np.array(logits[0, -1], dtype=np.float32)
        except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
            record_degradation("nonparametric_cached_generate", exc)
            break

        next_id = int(np.argmax(lg))
        if use_memory:
            try:
                neighbors = memory.query(key, k=k)
                if neighbors:
                    # Anisotropy-corrected gate (see Neighbor.similarity).
                    sim = float(getattr(neighbors[0], "similarity", -1.0))
                    gate = float(getattr(memory, "min_similarity", lambda: min_cos)())
                    nearest_index = int(getattr(neighbors[0], "index", -1))
                    # Anti-stutter: a chain walks DIFFERENT entries each
                    # step; the same nearest entry twice in a row means the
                    # chain ended and the stale tail is re-firing.
                    if sim >= gate and nearest_index != last_fired_index:
                        fe = 0.5 if free_energy is None else float(free_energy)
                        lam = base_lam * max(0.0, (sim - gate) / max(1e-6, 1.0 - gate)) * (0.6 + 0.8 * fe)
                        from core.brain.nonparametric_generation import _topk_probs
                        blended = memory.interpolate(
                            _topk_probs(lg), key, k=k, temperature=temperature, phi=phi,
                            free_energy=free_energy, lam_override=min(lam, 0.9),
                        )
                        memory_choice = int(max(blended, key=blended.get))
                        if memory_choice != next_id:
                            last_fired_index = nearest_index
                        next_id = memory_choice
            except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
                record_degradation("nonparametric_cached_generate_interp", exc)

        out_ids.append(next_id)
        if eos is not None and next_id == eos:
            break
        cursor = mx.array([[next_id]])   # only the new token next step — O(1) forward

    return tokenizer.decode(out_ids).strip()
