"""The stable reference for "self" across restarts and evolutions.

It was not stable. This returned

    state.identity.name + "-" + state.state_id[:8]

and ``state_id`` is a fresh uuid4 on every derived state — so the anchor moved
several times a second during ordinary cognition and changed completely on
restart. Renaming her changed it too. The one object whose entire job is to say
"this is the same entity" was a function of the most transient field in the
state, and it fell back to the literal string "Aura-Transient" whenever the
repository was not up, which is exactly when an anchor is worth having.

The anchor is now the fingerprint of a durable Ed25519 key
(core/identity/entity_key.py) that persists under the state root and is
independent of every state version. State lineage was already strong; this is
what it was missing to be signed to anything.

The name is still reported, because a human-readable anchor is more useful than
a bare digest — but it is reported ALONGSIDE the key fingerprint rather than as
part of it, so renaming her does not change who she is.
"""

import logging

from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service

logger = logging.getLogger("Aura.IdentityAnchor")


class IdentityAnchor:
    """Identity continuity for Aura, anchored to a key rather than to a state."""

    def __init__(self):
        self._aura_id = None
        logger.info("IdentityAnchor initialized.")

    def get_identity(self) -> str:
        """The persistent entity id.

        Derived from the durable entity key, so it is the same value on the
        first tick after boot and on the ten-thousandth, and the same value
        after a restart. It does not consult AuraState at all — the previous
        implementation's dependency on the state repository is what made it
        transient in the first place.
        """

        if self._aura_id:
            return self._aura_id

        try:
            from core.identity.entity_key import entity_identity

            self._aura_id = entity_identity().entity_id
            return self._aura_id
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError) as e:
            record_degradation(
                "identity_anchor",
                e,
                severity="critical",
                action=(
                    "could not resolve the durable entity key; reported an "
                    "explicitly unanchored identity rather than inventing one"
                ),
            )
            logger.error("Failed to resolve the durable entity key: %s", e)

        # Not a fallback identity — a statement that there is none. Returning a
        # plausible-looking id here would be the "absence of a check reported
        # as a passed check" failure, in the one place it matters most.
        return "Aura-Unanchored"

    def display_name(self) -> str:
        """The name she goes by, which is not the same question as who she is."""

        try:
            repo = get_runtime_service("state_repo", default=None)
            state = getattr(repo, "_current", None) if repo else None
            name = str(getattr(getattr(state, "identity", None), "name", "") or "").strip()
            return name or "Aura"
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("identity_anchor", e)
            return "Aura"

    def report(self) -> dict:
        """Anchor, name and chain position, kept apart on purpose."""

        try:
            from core.identity.entity_key import entity_identity

            identity = entity_identity()
            return {
                "entity_id": identity.entity_id,
                "display_name": self.display_name(),
                "chain_head": identity.chain_head,
                "anchored": identity.entity_id != "",
                "scheme": identity.report()["scheme"],
            }
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError) as e:
            record_degradation("identity_anchor", e, severity="error")
            return {
                "entity_id": "Aura-Unanchored",
                "display_name": self.display_name(),
                "chain_head": "",
                "anchored": False,
            }

    def __repr__(self):
        return f"<IdentityAnchor(id='{self.get_identity()}')>"
