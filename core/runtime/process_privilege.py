"""core/runtime/process_privilege.py — what each kind of child process is allowed to be.

Clean-room adoption of Chromium's process-role security model. No Chromium code
is used (its `third_party` tree is a licensing minefield and none of it is
needed): what is adopted is the discipline of writing the privilege boundary
down as a **matrix that the code enforces**, rather than as case-by-case
decisions spread across every spawn site.

Aura already isolates processes — the MLX worker runs apart, subprocesses go
through a gateway, the browser is separate. What it did not have is a single
statement of what each ROLE may do, so the boundaries were re-decided at each
call site and could drift apart without anything noticing.

The roles are ordered by how much they should be trusted with, and the ordering
is the point: a process that parses hostile input should not also hold
authority. Web content, PDFs, generated code and third-party skills are all
things Aura will process on someone else's behalf, and each is a place where
"it only reads the file" quietly becomes "it can reach the network".

What this module is NOT: a sandbox. It does not apply seccomp, entitlements, or
namespaces — those are OS mechanisms Aura's gateway layer would have to grow.
It is the *declaration and admission* half: every spawn declares its role, the
matrix says what that role may do, and a spawn asking for more than its role
allows is refused with the specific privilege named. Enforcing a declared
boundary is worth having even before the kernel enforces it too, because it
makes the boundary reviewable and makes violations loud.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

__all__ = [
    "Privilege",
    "PrivilegeDecision",
    "ProcessRole",
    "ROLE_PRIVILEGES",
    "check_spawn",
    "role_for_source",
]


class ProcessRole(IntEnum):
    """Ordered by trust. Lower is less trusted.

    Ordering matters because it makes "this role is at least as constrained as
    that one" expressible, and because reviewing a matrix sorted by trust makes
    an over-privileged role visually obvious.
    """

    #: Parses one hostile artifact. Should hold nothing.
    DOCUMENT_DECODER = 0
    #: Renders or scrapes web content. Untrusted input by definition.
    WEB_CONTENT = 1
    #: Runs generated or third-party code.
    UNTRUSTED_CODE = 2
    #: Executes one capability under a scoped token.
    TOOL_RUNNER = 3
    #: Loads and runs model weights. Trusted with compute, not with authority.
    MODEL_WORKER = 4
    #: Collects diagnostics. Separate so a crash handler survives the crash.
    CRASH_HANDLER = 5
    #: Policy, scheduling, governance. Parses nothing hostile.
    COORDINATOR = 6


class Privilege(StrEnum):
    """Capabilities a child process may hold."""

    NETWORK = "network"
    FILESYSTEM_READ = "filesystem_read"
    FILESYSTEM_WRITE = "filesystem_write"
    #: May ask the Will for authority to act. Distinct from doing the acting.
    REQUEST_AUTHORITY = "request_authority"
    #: May spawn further processes. Almost always the wrong answer.
    SPAWN_CHILDREN = "spawn_children"
    #: May read model weights from disk.
    MODEL_WEIGHTS = "model_weights"
    #: May reach the user directly (notifications, speech, UI).
    USER_SURFACE = "user_surface"
    #: May read secrets, tokens, credentials.
    SECRETS = "secrets"


#: The matrix. Read it top to bottom: privilege accumulates with trust, and
#: anything that parses hostile input holds almost nothing.
ROLE_PRIVILEGES: dict[ProcessRole, frozenset[Privilege]] = {
    # One file in, structured data out. No network — a decoder that can phone
    # home turns a malicious PDF into an exfiltration channel.
    ProcessRole.DOCUMENT_DECODER: frozenset({Privilege.FILESYSTEM_READ}),
    # Web content needs the network by definition; it must never hold
    # authority, secrets, or the ability to spawn.
    ProcessRole.WEB_CONTENT: frozenset({Privilege.NETWORK}),
    # Generated code gets a scratch area and nothing else. Notably NO network:
    # code Aura wrote herself is still code nobody reviewed.
    ProcessRole.UNTRUSTED_CODE: frozenset({
        Privilege.FILESYSTEM_READ, Privilege.FILESYSTEM_WRITE,
    }),
    # A tool runner acts, so it may request authority — but it holds one
    # capability's scope, not the ability to spawn more processes.
    ProcessRole.TOOL_RUNNER: frozenset({
        Privilege.NETWORK, Privilege.FILESYSTEM_READ, Privilege.FILESYSTEM_WRITE,
        Privilege.REQUEST_AUTHORITY, Privilege.USER_SURFACE,
    }),
    # The model worker is trusted with a great deal of COMPUTE and with the
    # weights, and with nothing else. It does not act, so it needs no authority
    # and no network: a model that can reach the network is a model that can be
    # made to exfiltrate its context.
    ProcessRole.MODEL_WORKER: frozenset({
        Privilege.FILESYSTEM_READ, Privilege.FILESYSTEM_WRITE,
        Privilege.MODEL_WEIGHTS,
    }),
    # Diagnostics only, and deliberately able to write where it can be read
    # after the fact. No network: a crash report is the user's to send.
    ProcessRole.CRASH_HANDLER: frozenset({
        Privilege.FILESYSTEM_READ, Privilege.FILESYSTEM_WRITE,
    }),
    # The coordinator decides. It parses nothing hostile, which is what earns
    # it this list.
    ProcessRole.COORDINATOR: frozenset({
        Privilege.NETWORK, Privilege.FILESYSTEM_READ, Privilege.FILESYSTEM_WRITE,
        Privilege.REQUEST_AUTHORITY, Privilege.SPAWN_CHILDREN,
        Privilege.MODEL_WEIGHTS, Privilege.USER_SURFACE, Privilege.SECRETS,
    }),
}


#: Substrings that identify a spawn's role from the `source` every gateway
#: caller already passes. Ordered most-specific first, because "browser" also
#: appears inside "browser_screenshot_decoder".
_SOURCE_ROLE_HINTS: tuple[tuple[str, ProcessRole], ...] = (
    ("crash", ProcessRole.CRASH_HANDLER),
    ("faulthandler", ProcessRole.CRASH_HANDLER),
    ("decoder", ProcessRole.DOCUMENT_DECODER),
    ("pdf", ProcessRole.DOCUMENT_DECODER),
    ("ocr", ProcessRole.DOCUMENT_DECODER),
    ("browser", ProcessRole.WEB_CONTENT),
    ("phantom", ProcessRole.WEB_CONTENT),
    ("playwright", ProcessRole.WEB_CONTENT),
    ("scrape", ProcessRole.WEB_CONTENT),
    ("fetch", ProcessRole.WEB_CONTENT),
    ("sandbox", ProcessRole.UNTRUSTED_CODE),
    ("generated_code", ProcessRole.UNTRUSTED_CODE),
    ("self_code", ProcessRole.UNTRUSTED_CODE),
    ("mlx", ProcessRole.MODEL_WORKER),
    ("model", ProcessRole.MODEL_WORKER),
    ("lora", ProcessRole.MODEL_WORKER),
    ("train", ProcessRole.MODEL_WORKER),
    ("external_action", ProcessRole.TOOL_RUNNER),
    ("skill", ProcessRole.TOOL_RUNNER),
    ("capability", ProcessRole.TOOL_RUNNER),
    ("tool", ProcessRole.TOOL_RUNNER),
)


def role_for_source(source: str) -> ProcessRole | None:
    """Infer a spawn's role from the source label callers already pass.

    Returns None when nothing matches. That is deliberate: an unrecognised
    source must not be silently assigned a role, because guessing wrong here
    means either a spurious refusal or an unearned privilege. An explicit role
    is always better, and the caller is told to declare one.
    """
    text = str(source or "").strip().lower()
    if not text:
        return None
    for hint, role in _SOURCE_ROLE_HINTS:
        if hint in text:
            return role
    return None


@dataclass(frozen=True)
class PrivilegeDecision:
    """Whether a spawn may hold what it asked for."""

    allowed: bool
    role: ProcessRole | None
    source: str
    requested: frozenset[Privilege] = field(default_factory=frozenset)
    denied: frozenset[Privilege] = field(default_factory=frozenset)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "role": self.role.name.lower() if self.role is not None else None,
            "source": self.source,
            "requested": sorted(p.value for p in self.requested),
            "denied": sorted(p.value for p in self.denied),
            "reason": self.reason,
        }


def check_spawn(
    source: str,
    requested: set[Privilege] | frozenset[Privilege] | None = None,
    *,
    role: ProcessRole | None = None,
) -> PrivilegeDecision:
    """Decide whether a process in this role may hold these privileges.

    An unknown role is NOT a refusal. This matrix is young and the source
    vocabulary is large; refusing every unrecognised spawn would break the
    runtime and get the check disabled, which is worse than an incomplete
    matrix. Unknown roles are reported so the vocabulary can be completed from
    real traffic rather than guessed at.
    """
    requested = frozenset(requested or ())
    resolved = role if role is not None else role_for_source(source)

    if resolved is None:
        return PrivilegeDecision(
            allowed=True, role=None, source=source, requested=requested,
            reason="role not recognised; declare one to have this enforced",
        )

    granted = ROLE_PRIVILEGES.get(resolved, frozenset())
    denied = requested - granted
    if not denied:
        return PrivilegeDecision(
            allowed=True, role=resolved, source=source, requested=requested,
            reason=f"{resolved.name.lower()} may hold all requested privileges",
        )
    return PrivilegeDecision(
        allowed=False, role=resolved, source=source, requested=requested,
        denied=denied,
        reason=(
            f"{resolved.name.lower()} may not hold "
            + ", ".join(sorted(p.value for p in denied))
        ),
    )
