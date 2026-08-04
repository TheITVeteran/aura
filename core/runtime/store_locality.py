"""Where a WAL store may live, and what actually limits it to one machine.

The review's limitation: "Transactional state commits via SQLite WAL enforce
single-process execution locks, limiting distributed multi-node scaling."

MEASURED 2026-08-04, and the first half is not true. Four separate processes
writing 200 rows each to one WAL-mode database:

    elapsed 0.16s
    p0 200, p1 200, p2 200, p3 200   —  800 rows, zero OperationalError

WAL does not impose a single-PROCESS lock. It serialises writers against each
other and lets readers run concurrently with a writer; any number of processes
on the same host may participate. ``tests/test_wal_store_locality.py`` runs
that measurement rather than restating the claim.

What WAL really requires is a shared-memory index file (``-shm``) that every
participant mmaps. That is a same-machine mechanism. On a network filesystem
the mmap is not coherent between hosts, SQLite's own documentation says WAL
does not work there, and the failure mode is not an error — it is silent
database corruption.

So the constraint is SINGLE HOST, not single process, and the dangerous case is
the one nothing was checking: a store placed on NFS/SMB/AFP because a data
directory was pointed at a network share. This module measures the filesystem
under a store and refuses WAL where WAL is unsafe.

That is the honest boundary. It is not multi-node inference, and nothing here
pretends to be — it is the difference between "we are single-node" as an
assumption and as an enforced, measured property.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.StoreLocality")

#: Filesystem types where SQLite's WAL shared-memory index is not coherent.
#: Naming them explicitly is deliberate: an unknown type is reported as unknown
#: rather than assumed safe or assumed hostile.
NETWORK_FILESYSTEMS = frozenset(
    {
        "nfs",
        "nfs4",
        "smbfs",
        "cifs",
        "afpfs",
        "webdav",
        "ftp",
        "sshfs",
        "fuse.sshfs",
        "fuse.s3fs",
        "fuse.gcsfuse",
        "9p",
        "glusterfs",
        "ceph",
        "lustre",
    }
)

#: Types known to be local block or memory filesystems.
LOCAL_FILESYSTEMS = frozenset(
    {
        "apfs",
        "hfs",
        "ext2",
        "ext3",
        "ext4",
        "xfs",
        "btrfs",
        "zfs",
        "tmpfs",
        "ramfs",
        "overlay",
        "devtmpfs",
        "f2fs",
        "ntfs",
        "exfat",
        "vfat",
        "msdos",
    }
)


@dataclass(frozen=True)
class StoreLocality:
    """What filesystem a store sits on, and whether WAL is safe there."""

    path: str
    fstype: str
    is_network: bool
    is_known: bool

    @property
    def wal_is_safe(self) -> bool:
        """WAL needs a coherent shared-memory index; networks do not give one."""
        return not self.is_network

    def as_metrics(self) -> dict[str, object]:
        return {
            "store_path": self.path,
            "fstype": self.fstype,
            "is_network_filesystem": self.is_network,
            "fstype_recognised": self.is_known,
            "wal_is_safe": self.wal_is_safe,
        }


def _fstype_darwin(path: Path) -> str:
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["/sbin/mount"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    best_len, best_type = -1, ""
    for line in out.splitlines():
        # "/dev/disk3s5 on / (apfs, local, journaled)"
        if " on " not in line or "(" not in line:
            continue
        mount_point = line.split(" on ", 1)[1].split(" (", 1)[0]
        fstype = line.rsplit("(", 1)[1].split(",", 1)[0].strip().rstrip(")")
        try:
            resolved = str(path)
            if resolved.startswith(mount_point) and len(mount_point) > best_len:
                best_len, best_type = len(mount_point), fstype
        except (TypeError, ValueError):
            continue
    return best_type.lower()


def _fstype_linux(path: Path) -> str:
    try:
        import psutil

        best_len, best_type = -1, ""
        for part in psutil.disk_partitions(all=True):
            if str(path).startswith(part.mountpoint) and len(part.mountpoint) > best_len:
                best_len, best_type = len(part.mountpoint), part.fstype
        return str(best_type).lower()
    except (ImportError, OSError, AttributeError):
        return ""


def describe_store(path: str | os.PathLike[str]) -> StoreLocality:
    """Measure the filesystem under ``path`` — the file need not exist yet."""
    # Absolute, or the mount-point prefix match below silently compares a
    # relative path against "/" and reports the wrong filesystem.
    try:
        resolved = Path(path).expanduser().absolute()
    except (OSError, RuntimeError, ValueError):
        resolved = Path(path)
    probe = resolved
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent

    fstype = ""
    try:
        if platform.system() == "Darwin":
            fstype = _fstype_darwin(probe)
        else:
            fstype = _fstype_linux(probe)
    except (OSError, RuntimeError, ValueError) as exc:
        record_degradation("store_locality", exc, severity="warning")

    fstype = (fstype or "").lower()
    is_network = fstype in NETWORK_FILESYSTEMS or fstype.startswith("fuse.")
    is_known = bool(fstype) and (fstype in NETWORK_FILESYSTEMS or fstype in LOCAL_FILESYSTEMS)
    return StoreLocality(
        path=str(resolved), fstype=fstype or "unknown", is_network=is_network, is_known=is_known
    )


def assert_wal_safe(path: str | os.PathLike[str], *, subsystem: str) -> StoreLocality:
    """Refuse to open a WAL store where WAL cannot be coherent.

    Fails CLOSED for a known network filesystem, because the failure mode there
    is silent corruption rather than an error, and a corrupted ledger is worse
    than an unavailable one. An UNRECOGNISED filesystem is recorded and allowed:
    refusing everything this module has not heard of would take the runtime
    down on any ordinary host with an unusual mount.
    """
    locality = describe_store(path)
    if locality.is_network:
        raise RuntimeError(
            f"{subsystem}: refusing a WAL store on {locality.fstype!r} "
            f"({locality.path}). SQLite's WAL index is shared memory and is not "
            "coherent across hosts; the failure mode is silent corruption."
        )
    if not locality.is_known:
        record_degradation(
            subsystem,
            RuntimeError(f"unrecognised filesystem {locality.fstype!r} under a WAL store"),
            severity="warning",
            action="allowed the store; WAL safety on this filesystem is unverified",
            extra=locality.as_metrics(),
        )
    return locality


__all__ = [
    "LOCAL_FILESYSTEMS",
    "NETWORK_FILESYSTEMS",
    "StoreLocality",
    "assert_wal_safe",
    "describe_store",
]
