"""Who owns a piece of state, and which runtime is allowed to touch it.

Aura accumulated persistent stores, registries, singletons and learning
systems faster than it accumulated boundaries between them. The result was
spooky action at a distance — behaviour changing because another checkout
ran, bugs that vanished when reproduced alone.

The two live cases this exists to make impossible:

* Terminal-grid TESTS read and wrote the live user-global learning
  directory. Learned risk from live state reached 1.0 and vetoed a benign
  move; test episodes were written back into the organism's real learning
  memory. A test run permanently changed what Aura believed.
* A fixture left a fail-closed Mycelium instance registered after its test
  finished. Dozens of later tests failed only inside the full suite,
  because they inherited global state nobody had declared they owned.

Three ideas, and the third is the one that does the work:

1. **Identity.** Every runtime has an immutable ``runtime_instance_id``.
   Every persistent record names the runtime and model that produced it,
   so a store can always answer "who wrote this".
2. **Roots.** All state lives under one explicitly resolved state root.
   There is no second way to find it and no ambient default buried in a
   subsystem.
3. **Structural separation.** Live, test, bench and dev roots are
   different directories, and a non-live runtime that reaches for the live
   root does not get a warning — it raises. A boundary that merely logs is
   a boundary that has already been crossed.

The profile is DERIVED, not declared by the code being tested. A test
cannot opt itself back into the live root by setting a flag, because the
tests that did this damage did not know they were doing it.
"""
from __future__ import annotations

import enum
import os
import platform
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "RuntimeProfile",
    "StateOwnershipViolation",
    "runtime_instance_id",
    "runtime_profile",
    "state_root",
    "live_state_root",
    "is_live_state_path",
    "assert_state_path_allowed",
    "stamp_record",
    "runtime_identity",
]


class StateOwnershipViolation(RuntimeError):
    """A runtime reached for state it does not own.

    Deliberately not a warning. The failures this prevents were silent:
    nothing raised, a test wrote into live learning memory, and the damage
    was found weeks later in behaviour rather than in a log.
    """


class RuntimeProfile(str, enum.Enum):
    """Which kind of runtime this process is. Derived, never self-declared."""

    LIVE = "live"
    TEST = "test"
    BENCH = "bench"
    DEV = "dev"

    @property
    def may_touch_live_state(self) -> bool:
        return self is RuntimeProfile.LIVE


#: The real user-global root. Never used directly outside this module —
#: everything goes through ``state_root()`` so the profile can intercept.
_LIVE_ROOT_NAME = ".aura"

_LOCK = threading.RLock()
_INSTANCE_ID: str | None = None
_PROFILE: RuntimeProfile | None = None
_ROOT: Path | None = None
_ROOT_KEY: tuple[str, str] | None = None
_STARTED_AT = time.time()


def _home() -> Path:
    override = os.environ.get("AURA_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home().expanduser().resolve()


#: The home this process started with, captured before anything can redirect
#: it. THE reference point: "live state" means the real user's ~/.aura, and a
#: test that repoints HOME must not be able to move what that phrase denotes
#: — otherwise the guard would happily let a run write to whatever it had
#: just declared to be its own live root.
_ORIGINAL_HOME: Path = _home()


def live_state_root() -> Path:
    """The REAL instance's state root. Fixed for the life of the process.

    Deliberately not recomputed from the current ``HOME``: this is the thing
    being protected, and a protected resource whose identity follows a
    mutable environment variable is not protected.
    """
    return _ORIGINAL_HOME / _LIVE_ROOT_NAME


def _derive_profile() -> RuntimeProfile:
    """Work out what this process is from evidence, not from a declaration.

    Order matters. A bench run inside pytest is a BENCH run — it has its
    own root and its own numbers, and folding it into the test root would
    let one overwrite the other.
    """
    if os.environ.get("AURA_BENCH_RUN") or os.environ.get("AURA_BENCHMARK"):
        return RuntimeProfile.BENCH
    # pytest sets PYTEST_CURRENT_TEST for the duration of every test. It is
    # set by the runner, so code under test cannot clear it to win back the
    # live root — which is exactly the move that would reintroduce the bug.
    if (
        os.environ.get("PYTEST_CURRENT_TEST")
        or os.environ.get("AURA_TESTING")
        or os.environ.get("AURA_PROOF_RUN")
        or "PYTEST_VERSION" in os.environ
    ):
        return RuntimeProfile.TEST
    if os.environ.get("AURA_DEV_RUNTIME"):
        return RuntimeProfile.DEV
    return RuntimeProfile.LIVE


def runtime_profile() -> RuntimeProfile:
    """This process's profile, resolved once.

    Cached deliberately: a profile that could change mid-process would let
    a runtime write half its state to one root and half to another, which
    is worse than either root being wrong.
    """
    global _PROFILE
    with _LOCK:
        if _PROFILE is None:
            _PROFILE = _derive_profile()
        return _PROFILE


def runtime_instance_id() -> str:
    """Immutable identity for this runtime, stable for the process lifetime.

    Carries the profile in the id itself, so a record found in the wrong
    place names its own origin without a lookup.
    """
    global _INSTANCE_ID
    with _LOCK:
        if _INSTANCE_ID is None:
            _INSTANCE_ID = f"{runtime_profile().value}-{uuid.uuid4().hex[:16]}"
        return _INSTANCE_ID


def state_root() -> Path:
    """THE state root. The only way to find where state lives.

    Resolution order:

    1. ``AURA_STATE_ROOT`` — an explicit injection, honoured for every
       profile. This is how a harness gives a run its own world.
    2. The profile's default. LIVE gets the real root; TEST, BENCH and DEV
       get siblings that are structurally distinct directories, not
       subdirectories of the live root — a subdirectory is one ``..`` away
       from the thing it was supposed to be separate from.
    3. When ``HOME`` has been redirected away from the home this process
       started with — the standard way a test isolates itself — then
       ``$HOME/.aura`` is ALREADY not the live instance's state, and it is
       used as-is. Appending a ``-test`` suffix there would send the run to
       a directory its own fixture never created, which is a broken test
       rather than a safer one.

    The cache is keyed on the inputs rather than set once, so a test that
    repoints ``HOME`` per-test gets the root it just arranged. A live
    runtime never changes either input, so it resolves once and stays.
    """
    global _ROOT, _ROOT_KEY
    with _LOCK:
        injected = os.environ.get("AURA_STATE_ROOT") or ""
        key = (injected, str(_home()))
        if _ROOT is not None and _ROOT_KEY == key:
            return _ROOT
        if injected:
            root = Path(injected).expanduser().resolve()
        else:
            profile = runtime_profile()
            current = _home() / _LIVE_ROOT_NAME
            if profile is RuntimeProfile.LIVE or current != live_state_root():
                # Either this IS the live runtime, or HOME has been moved
                # and `current` is already somewhere private.
                root = current
            else:
                root = current.with_name(f"{current.name}-{profile.value}")
        _ROOT, _ROOT_KEY = root, key
        return _ROOT


def is_live_state_path(path: Path | str) -> bool:
    """Whether a path lands inside the real instance's state.

    Resolved, not string-compared: ``~/.aura-test/../.aura/data`` is the
    live root, and a check that missed that would be decorative. Uses
    ``strict=False`` so a path that does not exist yet — the interesting
    case, since writes create things — still resolves.
    """
    try:
        candidate = Path(path).expanduser().resolve(strict=False)
        live = live_state_root()
    except (OSError, RuntimeError, ValueError):
        return False
    if candidate == live:
        return True
    return live in candidate.parents


def assert_state_path_allowed(path: Path | str, *, source: str = "unknown") -> None:
    """Refuse a write into live state from a runtime that does not own it.

    The enforcement point. Called from the file-write gateway, so every
    consequential write in the codebase inherits it without each subsystem
    remembering to ask.

    A LIVE runtime is unaffected. Everything else raises on contact with
    the live root, which is what turns "tests wrote into Aura's real
    learning memory" from a thing that happened into a thing that cannot.
    """
    profile = runtime_profile()
    if profile.may_touch_live_state:
        return
    if not is_live_state_path(path):
        return
    if os.environ.get("AURA_ALLOW_LIVE_STATE_WRITE") == "1":
        # A deliberate, named escape for the rare tool that really does
        # maintain the live instance. Env-only: nothing in the codebase can
        # set it for itself mid-run without an operator having done so.
        return
    raise StateOwnershipViolation(
        f"{profile.value} runtime tried to write live instance state at {path} "
        f"(source={source}). Live state belongs to the live runtime. Use "
        f"AURA_STATE_ROOT to give this run its own root, or set "
        f"AURA_ALLOW_LIVE_STATE_WRITE=1 if maintaining the live instance is "
        f"genuinely the intent."
    )


def runtime_identity() -> dict[str, Any]:
    """Who this runtime is, for stamping onto anything it persists."""
    return {
        "runtime_instance_id": runtime_instance_id(),
        "runtime_profile": runtime_profile().value,
        "state_root": str(state_root()),
        "started_at": _STARTED_AT,
        "pid": os.getpid(),
        "host": platform.node(),
    }


def stamp_record(
    payload: Mapping[str, Any], *, model_identity: str | None = None
) -> dict[str, Any]:
    """Name the runtime (and model) that produced a persistent record.

    A record that does not say who wrote it cannot be attributed after the
    fact, and unattributable state is how a test's episodes ended up
    indistinguishable from the organism's real ones.

    Non-mutating: returns a new mapping. Never overwrites an existing
    stamp, so a record forwarded between runtimes keeps its true origin.
    """
    stamped = dict(payload or {})
    if "_runtime" not in stamped:
        identity = runtime_identity()
        if model_identity:
            identity["model_identity"] = str(model_identity)
        stamped["_runtime"] = identity
    return stamped


def reset_for_testing() -> None:
    """Clear the cached resolution. For tests OF this module only.

    Named loudly on purpose: anything else calling this is defeating the
    caching that keeps a runtime's state in one place.
    """
    global _INSTANCE_ID, _PROFILE, _ROOT
    with _LOCK:
        _INSTANCE_ID = None
        _PROFILE = None
        _ROOT = None
