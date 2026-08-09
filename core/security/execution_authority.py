"""One gate in front of every surface that runs a caller-supplied program.

Aura has general reach: `sovereign_terminal` runs arbitrary shell, `mcp_client`
spawns arbitrary MCP servers and calls arbitrary tools on them. That reach is
the point — a terminal is access to everything the machine's software can do,
and MCP widens it to everything a connector exposes. It is also the single
most consequential capability in the system, so it is the one that must be
hardest to use without asking.

It was the easiest. Before this module:

  * `sovereign_terminal._run_command` ran anything the caller passed through
    `subprocess_gateway`, gated only by a substring denylist. It never asked
    the Will. Zero `authorize_*` calls in the file.
  * `mcp_client` spawned a caller-supplied `server_command` via `stdio_client`,
    which does not go through `subprocess_gateway` at all, and declared
    `requires_approval = False`.
  * `capability_engine._Shell` — the one that DID ask the Will — was confined
    to a fixed allowlist.

So the governed shell was narrow and the general shells were ungoverned:
exactly backwards. Anything the allowlist refused was reachable by asking for
the same command through the terminal skill instead.

The rule this module enforces is not "which commands are allowed". A denylist
of dangerous-looking strings is a lexical gate deciding a semantic question,
and it loses to `$'\\x72\\x6d'` or to a program name nobody thought of. The
rule is: **the decision belongs to the Will, and a surface that cannot reach
the Will does not execute.** Fail closed. The denylists that remain are
defence-in-depth behind that decision, never the decision itself.

`authorize_execution` is deliberately the only export that runs anything's
gate, so `test_general_execution_surfaces_are_governed.py` can assert that
every process-spawning surface calls it — the next one added inherits the
gate instead of rediscovering the hole.
"""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Security.ExecutionAuthority")

# The tool name the Will sees. One name for all general execution, so a
# standing directive like "never run anything that touches my keychain" is
# written once and covers the terminal, MCP, and whatever comes next.
EXECUTION_TOOL_NAME = "general_execution"

# Kinds of execution, for the receipt. Not a permission tier — the Will
# decides; this only tells it what shape of thing is being asked for.
KIND_SHELL = "shell"
KIND_OPEN = "open_target"
KIND_MCP_SERVER = "mcp_server"
KIND_MCP_TOOL = "mcp_tool"

_KINDS = frozenset({KIND_SHELL, KIND_OPEN, KIND_MCP_SERVER, KIND_MCP_TOOL})

# Priority floor. General execution is never a low-priority background
# nicety; if it is worth spawning a process for, it is worth the Will's
# attention.
_PRIORITY = 0.8


@dataclass(frozen=True)
class ExecutionVerdict:
    """What the Will said, plus what the caller needs to honour it."""

    approved: bool
    reason: str
    kind: str
    descriptor: str
    token_id: str | None = None
    intent_id: str | None = None
    standing_token: str | None = None
    outcome: str = "denied"
    constraints: dict[str, Any] = field(default_factory=dict)

    def as_error(self) -> dict[str, Any]:
        """The refusal in the shape every skill already returns."""
        return {
            "ok": False,
            "error": self.reason,
            "governance": {
                "outcome": self.outcome,
                "kind": self.kind,
                "authorized": False,
            },
        }

    def receipt(self) -> dict[str, Any]:
        """What actually authorized this, for the result envelope.

        A caller that ran something should be able to say which decision let
        it. An approval with no recoverable receipt is indistinguishable from
        no approval at all once the log rotates.
        """
        return {
            "outcome": self.outcome,
            "kind": self.kind,
            "authorized": bool(self.approved),
            "token_id": self.token_id,
        }


def describe_command(command: Any) -> str:
    """A stable, readable descriptor for the thing being run.

    Used for the Will's receipt and for logs. Not a security boundary — the
    full arguments go to the gateway as structured data, not as this string.
    """
    if isinstance(command, str):
        return command.strip()
    if isinstance(command, (list, tuple)):
        try:
            return shlex.join(str(part) for part in command)
        except (TypeError, ValueError):
            return " ".join(str(part) for part in command)
    return str(command)


async def authorize_execution(
    kind: str,
    command: Any,
    *,
    source: str,
    cwd: str | None = None,
    extra: dict[str, Any] | None = None,
) -> ExecutionVerdict:
    """Ask the Will before running a caller-supplied program.

    Returns a denied verdict rather than raising, because every call site is
    a skill that must turn a refusal into an answer rather than a traceback.

    Fails CLOSED. If the gateway cannot be imported, cannot be constructed,
    or raises, the answer is no. The alternative — running because the thing
    that would have said no was unavailable — is the exact defect this
    codebase keeps finding: the absence of a check reported as a passed
    check.
    """
    if kind not in _KINDS:
        raise ValueError(f"unknown execution kind: {kind!r}")

    descriptor = describe_command(command)
    if not descriptor:
        return ExecutionVerdict(
            approved=False,
            reason="Execution requires a command.",
            kind=kind,
            descriptor="",
            outcome="invalid",
        )

    args: dict[str, Any] = {
        "kind": kind,
        "command": descriptor,
    }
    if isinstance(command, (list, tuple)):
        args["argv"] = [str(part) for part in command]
    if cwd:
        args["cwd"] = str(cwd)
    if extra:
        args.update(extra)

    try:
        from core.executive.authority_gateway import get_authority_gateway

        gateway = get_authority_gateway()
        decision = await gateway.authorize_tool_execution(
            EXECUTION_TOOL_NAME,
            args,
            source=source,
            priority=_PRIORITY,
            # Never derived from a risk label. `is_critical` is an
            # unconditional pass on the canonical path, so inferring it from
            # how dangerous something looks makes the most dangerous things
            # the ones that skip the veto. See CLAIMS_NOT_SUPPORTED.md #10.
            is_critical=False,
            context={"execution_kind": kind},
        )
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation(
            "execution_authority",
            exc,
            severity="critical",
            action=f"refused {kind} execution because the authority gateway was unavailable",
            extra={"kind": kind, "source": source, "command": descriptor[:240]},
        )
        return ExecutionVerdict(
            approved=False,
            reason=f"Authority unavailable for {kind} execution: {exc}",
            kind=kind,
            descriptor=descriptor,
            outcome="authority_unavailable",
        )

    if not getattr(decision, "approved", False):
        reason = getattr(decision, "reason", "") or "refused by the Will"
        logger.info(
            "Execution refused (%s) for %s: %s", kind, source, reason
        )
        return ExecutionVerdict(
            approved=False,
            reason=f"Authority refused this {kind} execution: {reason}",
            kind=kind,
            descriptor=descriptor,
            outcome=str(getattr(decision, "outcome", "denied") or "denied"),
            constraints=dict(getattr(decision, "constraints", {}) or {}),
        )

    token_id = getattr(decision, "capability_token_id", None)
    intent_id = getattr(decision, "executive_intent_id", None)
    standing_token = getattr(decision, "standing_authority_token", None)

    # An approval nobody authenticated is a caller's claim, not a grant.
    #
    # `verify_tool_access` is NOT sufficient here: its own docstring says it
    # proves only that some token naming this tool exists in this process,
    # and that anyone who can import the capability system can mint one. For
    # the most consequential sink in the codebase, the thing to check is the
    # Ed25519-signed capability, which the verifier cannot forge because it
    # holds only the public key.
    signed = getattr(decision, "signed_capability", None)
    if signed is None:
        # Fail closed. A mint failure here means the Will approved but the
        # grant could not be signed — running anyway would be executing on an
        # unauthenticated approval, which is the shape of every bypass this
        # codebase has found.
        return ExecutionVerdict(
            approved=False,
            reason=(
                f"Authority approved this {kind} execution but issued no signed "
                "capability; refusing to execute on an unauthenticated grant."
            ),
            kind=kind,
            descriptor=descriptor,
            outcome="capability_unsigned",
        )

    try:
        from core.governance.capability_chain import get_capability_verifier

        result = get_capability_verifier().verify(
            signed,
            expected_action_digest=None,  # bound at the gateway, checked here for provenance
            consume=True,
        )
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation(
            "execution_authority",
            exc,
            severity="critical",
            action=f"refused {kind} execution because capability verification raised",
            extra={"kind": kind, "source": source},
        )
        return ExecutionVerdict(
            approved=False,
            reason=f"Capability verification failed for {kind} execution: {exc}",
            kind=kind,
            descriptor=descriptor,
            outcome="capability_unverifiable",
        )

    if not getattr(result, "ok", False):
        detail = getattr(result, "detail", "") or "signature did not verify"
        logger.warning(
            "Execution refused (%s) for %s: capability rejected — %s",
            kind,
            source,
            detail,
        )
        return ExecutionVerdict(
            approved=False,
            reason=f"Capability rejected for {kind} execution: {detail}",
            kind=kind,
            descriptor=descriptor,
            outcome="capability_rejected",
        )

    return ExecutionVerdict(
        approved=True,
        reason=getattr(decision, "reason", "") or "approved",
        kind=kind,
        descriptor=descriptor,
        token_id=token_id,
        intent_id=intent_id,
        standing_token=standing_token,
        outcome=str(getattr(decision, "outcome", "approved") or "approved"),
        constraints=dict(getattr(decision, "constraints", {}) or {}),
    )


def release_execution(
    verdict: ExecutionVerdict,
    *,
    source: str,
    success: bool = True,
    error: str = "",
) -> dict[str, Any] | None:
    """Close the intent, token, and standing lease once the program finished.

    A capability token that outlives the thing it authorized is a live grant
    nobody is tracking, and an executive intent that never completes leaves
    the Will believing an action is still in flight.

    The receipt is returned rather than discarded. `finalize_tool_execution`
    reports exactly which of intent / token / lease failed to close, and the
    gateway's own comment records that every call site used to throw that
    away — which is why an unreconciled grant could never be closed by
    anyone, because nobody knew it was open.
    """
    if not verdict.approved:
        return None
    if not (verdict.intent_id or verdict.token_id or verdict.standing_token):
        return None
    try:
        from core.executive.authority_gateway import get_authority_gateway

        receipt = get_authority_gateway().finalize_tool_execution(
            executive_intent_id=verdict.intent_id,
            capability_token_id=verdict.token_id,
            standing_authority_token=verdict.standing_token,
            success=bool(success),
            error=str(error or ""),
        )
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation(
            "execution_authority",
            exc,
            severity="warning",
            action="left an execution grant open after the program finished",
            extra={"kind": verdict.kind, "source": source, "token": verdict.token_id},
        )
        return None

    if isinstance(receipt, dict) and not receipt.get("closed", True):
        record_degradation(
            "execution_authority",
            RuntimeError(f"execution grant did not close: {receipt.get('errors')}"),
            severity="warning",
            action="recorded an unreconciled execution grant",
            extra={"kind": verdict.kind, "source": source, "receipt": receipt},
        )
    return receipt


__all__ = [
    "EXECUTION_TOOL_NAME",
    "KIND_MCP_SERVER",
    "KIND_MCP_TOOL",
    "KIND_OPEN",
    "KIND_SHELL",
    "ExecutionVerdict",
    "authorize_execution",
    "describe_command",
    "release_execution",
]
