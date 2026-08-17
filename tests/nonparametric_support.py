"""One named identity for datastore entries a test writes.

`NonParametricMemory.add` requires an `EntryProvenance`. It used to default
to `unattributed`/`UNVERIFIED`/`anonymous`, and the live ingest path took
that default for every entry it ever wrote — which is why nothing the
resident worker learned could be revoked by source or erased by principal.
The default is gone, so tests name themselves like every other writer.
"""

from __future__ import annotations

from core.brain.nonparametric_identity import EntryProvenance, TrustLevel

#: The principal a test's own entries belong to.
TEST_PRINCIPAL = "test_principal"


def entry_provenance(
    *,
    principal: str = TEST_PRINCIPAL,
    source_id: str = "test_suite",
    trust: str = TrustLevel.VERIFIED,
    verifier: str = "tests",
    evidence_id: str = "",
) -> EntryProvenance:
    """Provenance for an entry a test writes on its own behalf."""
    return EntryProvenance(
        source_id=source_id,
        principal=principal,
        trust=trust,
        verifier=verifier,
        evidence_id=evidence_id,
    )
