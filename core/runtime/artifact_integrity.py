"""core/runtime/artifact_integrity.py — nothing becomes running code unverified.

Clean-room adoption of the verify-before-install discipline from The Update
Framework and SLSA. No TUF or Sigstore code is used, and no key infrastructure
is introduced: Aura has no self-updater today, so a full TUF client would be
speculative machinery for a path that does not exist.

What DOES exist is the problem TUF's first principle addresses. Several live
paths promote a file into executable Python — emergency rollback, code repair,
kernel refinement, sandbox promotion — and a file that becomes running code is
the highest-consequence write in the system. This module is the shared gate for
that moment, extracted from the one place it was already proven
(``ImmuneSystem.initiate_rollback``) so every other promotion path can use the
same checks instead of reinventing weaker ones.

Four checks, in increasing cost order so the cheap disqualifiers run first:

1. **Containment** — the source resolves *inside* the directory it is supposed
   to come from. Compared by resolved path components, because
   ``str(p).startswith(str(base))`` accepts any sibling whose name merely begins
   with the base: ``data/backups_evil`` passes a ``data/backups`` prefix test.
2. **Regular file** — resolved strictly and confirmed a real file, so a symlink
   swapped in after the check cannot redirect the read.
3. **Digest** — a manifest must exist and match. Its ABSENCE is a refusal, not
   a warning: unsigned content must not become running code merely because
   nobody supplied a signature.
4. **Parses** — the content is valid Python for the module it will replace.
   Restoring a syntactically broken file leaves the target unimportable,
   bricking the very thing the promotion was meant to rescue.

What this deliberately is NOT: a signature system. A sha256 manifest proves the
bytes are the bytes that were recorded, not that a trusted party recorded them.
Real signing needs a key hierarchy and rotation policy, and claiming otherwise
here would be exactly the overclaim this module exists to prevent. The verdict
says which guarantee it is giving.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__ = [
    "ArtifactVerdict",
    "IntegrityFailure",
    "IntegrityLevel",
    "verify_artifact",
]


class IntegrityFailure(StrEnum):
    """Why an artifact was refused. Each maps to one check."""

    UNRESOLVABLE = "unresolvable"
    OUTSIDE_ROOT = "outside_root"
    NOT_REGULAR_FILE = "not_regular_file"
    MANIFEST_MISSING = "manifest_missing"
    MANIFEST_UNREADABLE = "manifest_unreadable"
    DIGEST_MISMATCH = "digest_mismatch"
    NOT_VALID_PYTHON = "not_valid_python"


class IntegrityLevel(StrEnum):
    """What guarantee a passing verdict actually carries.

    Named explicitly so a caller cannot mistake a content digest for a
    signature. This module can currently only reach DIGEST.
    """

    #: Bytes match a recorded digest. Proves integrity, NOT authorship.
    DIGEST = "digest"
    #: Bytes match a digest signed by a trusted key. Not implemented.
    SIGNED = "signed"


@dataclass(frozen=True)
class ArtifactVerdict:
    """Whether an artifact may become running code, and what was proven."""

    ok: bool
    path: Path | None
    level: IntegrityLevel | None = None
    failure: IntegrityFailure | None = None
    detail: str = ""
    digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "path": str(self.path) if self.path else "",
            "level": self.level.value if self.level else None,
            "failure": self.failure.value if self.failure else None,
            "detail": self.detail,
            "digest": self.digest,
        }

    def __str__(self) -> str:
        if self.ok:
            return f"verified({self.level.value if self.level else '?'})"
        return f"refused({self.failure.value if self.failure else '?'}: {self.detail})"


def verify_artifact(
    source: str | Path,
    *,
    within: str | Path,
    require_python: bool = True,
    manifest_suffix: str = ".sha256",
) -> ArtifactVerdict:
    """Decide whether a file may be promoted into running code.

    Pure and synchronous; callers on an event loop should run it in a thread
    (its reads are blocking syscalls). Never raises — a verification routine
    that can itself explode is not a gate.
    """
    root_path = Path(within)
    try:
        root = root_path.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        # ValueError is not hypothetical: an embedded null byte in a path
        # raises it from resolve(), and a gate that can itself explode on
        # hostile input is not a gate.
        return ArtifactVerdict(
            ok=False, path=None, failure=IntegrityFailure.UNRESOLVABLE,
            detail=f"root {root_path} unresolvable: {exc}",
        )

    try:
        # strict=True so a missing path fails here rather than being papered
        # over by a later existence check on a half-resolved path.
        artifact = Path(source).resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        return ArtifactVerdict(
            ok=False, path=None, failure=IntegrityFailure.UNRESOLVABLE,
            detail=f"{source} unresolvable: {exc}",
        )

    # 1. Containment, by path components rather than string prefix.
    if artifact != root and root not in artifact.parents:
        return ArtifactVerdict(
            ok=False, path=artifact, failure=IntegrityFailure.OUTSIDE_ROOT,
            detail=f"{artifact} is not inside {root}",
        )

    # 2. A regular file. resolve(strict=True) already followed any link; this
    #    rejects directories, sockets, and devices.
    try:
        if not artifact.is_file():
            return ArtifactVerdict(
                ok=False, path=artifact, failure=IntegrityFailure.NOT_REGULAR_FILE,
                detail=f"{artifact} is not a regular file",
            )
    except (OSError, ValueError) as exc:
        return ArtifactVerdict(
            ok=False, path=artifact, failure=IntegrityFailure.NOT_REGULAR_FILE,
            detail=str(exc),
        )

    # 3. Digest. A missing manifest is a REFUSAL.
    manifest = artifact.with_suffix(artifact.suffix + manifest_suffix)
    try:
        if not manifest.is_file():
            return ArtifactVerdict(
                ok=False, path=artifact, failure=IntegrityFailure.MANIFEST_MISSING,
                detail=(
                    f"no integrity manifest at {manifest}; unsigned content "
                    "cannot become running code"
                ),
            )
        raw = manifest.read_text(encoding="utf-8").split()
        if not raw:
            return ArtifactVerdict(
                ok=False, path=artifact, failure=IntegrityFailure.MANIFEST_UNREADABLE,
                detail=f"{manifest} is empty",
            )
        expected = raw[0].strip().lower()
        actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
    except (OSError, UnicodeDecodeError, IndexError, ValueError) as exc:
        return ArtifactVerdict(
            ok=False, path=artifact, failure=IntegrityFailure.MANIFEST_UNREADABLE,
            detail=str(exc),
        )
    if not expected or expected != actual:
        return ArtifactVerdict(
            ok=False, path=artifact, failure=IntegrityFailure.DIGEST_MISMATCH,
            detail=f"expected {expected or '<empty>'}, got {actual}",
            digest=actual,
        )

    # 4. It has to parse, or promoting it bricks the target.
    if require_python:
        try:
            ast.parse(artifact.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            return ArtifactVerdict(
                ok=False, path=artifact, failure=IntegrityFailure.NOT_VALID_PYTHON,
                detail=f"{exc}; promoting it would leave the target unimportable",
                digest=actual,
            )

    return ArtifactVerdict(
        ok=True, path=artifact, level=IntegrityLevel.DIGEST, digest=actual
    )
