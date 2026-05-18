# core/rl_glue.py
import hashlib
import logging
from typing import Any

import numpy as np

logger = logging.getLogger("Kernel.RL_Glue")

OBSERVATION_DIM = 128
ACTION_CONTEXT_DIM = 64
DRIVE_NAMES = ("energy", "curiosity", "social", "competence", "uptime_value")
DRIVE_DEFAULTS = {
    "energy": 1.0,
    "curiosity": 0.5,
    "social": 0.5,
    "competence": 0.5,
    "uptime_value": 0.5,
}
_EMBEDDING_RECOVERABLE_ERRORS = (
    AttributeError,
    TypeError,
    ValueError,
    RuntimeError,
    OSError,
)


class RLInterface:
    """Connects the cognitive architecture to the RL environment."""

    def __init__(self, memory_nexus):
        self.memory = memory_nexus

    def get_state_vector(self) -> np.ndarray:
        """Converts current Agent State -> Vector (Observation).
        Uses the last few episodic logs + drive states.
        """
        action_vector = self._action_context_vector()
        drive_vector = self._drive_vector()

        full_vec = np.concatenate([action_vector, drive_vector])

        # Pad/truncate to the 128-dim observation shape defined in rl_env.py.
        if len(full_vec) < OBSERVATION_DIM:
            full_vec = np.pad(full_vec, (0, OBSERVATION_DIM - len(full_vec)))

        return full_vec[:OBSERVATION_DIM].astype(np.float32)

    def _action_context_vector(self) -> np.ndarray:
        text = self._latest_memory_text()
        embedded = self._embedded_context_vector(text)
        if embedded is not None:
            return _project_vector(embedded, ACTION_CONTEXT_DIM)
        return _hash_context_vector(text, ACTION_CONTEXT_DIM)

    def _embedded_context_vector(self, text: str) -> np.ndarray | None:
        for embed in _embedding_callables(self.memory):
            try:
                vector = embed(text)
            except _EMBEDDING_RECOVERABLE_ERRORS as exc:
                logger.debug("RL embedding source skipped: %s", exc)
                continue
            coerced = _coerce_vector(vector)
            if coerced is not None:
                return coerced
        return None

    def _latest_memory_text(self) -> str:
        if self.memory is None:
            return "no-memory-context"
        for method_name in ("latest_action", "latest_event", "get_recent_summary", "snapshot"):
            method = getattr(self.memory, method_name, None)
            if callable(method):
                value = method()
                if value:
                    return str(value)
        return str(self.memory)

    def _drive_vector(self) -> np.ndarray:
        drives = _drive_mapping(self.memory)
        values: list[float] = []
        for name in DRIVE_NAMES:
            raw = _read_drive_value(drives, name, DRIVE_DEFAULTS[name])
            values.append(max(-1.0, min(1.0, raw)))
        return np.array(values, dtype=np.float32)

    def calculate_reward(self, result: dict) -> float:
        """Determines reward based on tool execution success."""
        if result.get("ok"):
            return 1.0
        return -0.1


def _embedding_callables(memory: Any):
    seen: set[int] = set()
    for obj in _embedding_objects(memory):
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        for attr in ("embed", "embed_sync"):
            method = getattr(obj, attr, None)
            if callable(method):
                yield method

        embed_fn = getattr(obj, "_embed_fn", None)
        if callable(embed_fn):

            def chroma_embed(text: str, fn=embed_fn):
                result = fn([text])
                return result[0] if result else None

            yield chroma_embed


def _embedding_objects(memory: Any):
    if memory is None:
        return
    yield memory
    for attr in (
        "vector",
        "vector_memory",
        "_vector_memory",
        "memory_vector",
        "embedder",
        "embedding_engine",
    ):
        child = getattr(memory, attr, None)
        if child is not None:
            yield child
            nested_embedder = getattr(child, "embedder", None)
            if nested_embedder is not None:
                yield nested_embedder


def _coerce_vector(vector: Any) -> np.ndarray | None:
    if vector is None:
        return None
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim > 1:
        array = array[0]
    array = array.reshape(-1)
    if array.size == 0:
        return None
    return np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=-1.0)


def _project_vector(vector: Any, size: int) -> np.ndarray:
    array = _coerce_vector(vector)
    if array is None:
        return np.zeros(size, dtype=np.float32)
    if len(array) > size:
        array = np.array([chunk.mean() for chunk in np.array_split(array, size)], dtype=np.float32)
    elif len(array) < size:
        array = np.pad(array, (0, size - len(array)))

    norm = float(np.linalg.norm(array))
    if norm > 1.0:
        array = array / norm
    return np.clip(array, -1.0, 1.0).astype(np.float32)


def _hash_context_vector(text: str, size: int) -> np.ndarray:
    normalized = " ".join(str(text).lower().split())
    if not normalized:
        normalized = "no-memory-context"
    vec = np.zeros(size, dtype=np.float32)
    grams = [normalized] if len(normalized) <= 3 else [normalized[i : i + 3] for i in range(len(normalized) - 2)]
    for gram in grams:
        digest = hashlib.blake2b(gram.encode("utf-8", errors="replace"), digest_size=8).digest()
        value = int.from_bytes(digest[:8], "little")
        index = value % size
        sign = 1.0 if (value >> 11) & 1 else -1.0
        vec[index] += sign

    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec = vec / norm
    return vec.astype(np.float32)


def _drive_mapping(memory: Any) -> Any:
    if memory is None:
        return None
    for owner in (memory, getattr(memory, "engine", None)):
        getter = getattr(owner, "get_drive_vector", None)
        if callable(getter):
            try:
                values = getter()
            except _EMBEDDING_RECOVERABLE_ERRORS as exc:
                logger.debug("RL drive vector source skipped: %s", exc)
                continue
            if values:
                return values
    return getattr(memory, "drives", None)


def _read_drive_value(drives: Any, name: str, neutral: float) -> float:
    if drives is None:
        return neutral
    if isinstance(drives, dict):
        value = drives.get(name, neutral)
    else:
        value = getattr(drives, name, neutral)
    try:
        if hasattr(value, "level") and hasattr(value, "capacity"):
            capacity = float(value.capacity or 1.0)
            raw = float(value.level or 0.0) / capacity
        elif hasattr(value, "urgency"):
            raw = float(value.urgency or 0.0)
        else:
            raw = float(value)
    except (TypeError, ValueError):
        return neutral
    if raw > 1.0 and raw <= 100.0:
        raw = raw / 100.0
    return raw
