"""Simulation dataset registry for Aura research and evaluation loops.

The registry mirrors the useful integration shape of large scientific
simulation collections without depending on any specific upstream package or
dataset license: metadata in, deterministic manifests out, optional local
record streaming when a dataset has already been staged by the operator.
"""
from __future__ import annotations

import builtins
import csv
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation


@dataclass(frozen=True)
class SimulationDataset:
    name: str
    domain: str
    description: str = ""
    splits: tuple[str, ...] = ("train", "val", "test")
    size_gb: float | None = None
    local_path: str | None = None
    remote_uri: str | None = None
    license: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulationDataset:
        return cls(
            name=str(data["name"]),
            domain=str(data.get("domain", "unknown")),
            description=str(data.get("description", "")),
            splits=tuple(str(split) for split in data.get("splits", ("train", "val", "test"))),
            size_gb=float(data["size_gb"]) if data.get("size_gb") is not None else None,
            local_path=str(data["local_path"]) if data.get("local_path") else None,
            remote_uri=str(data["remote_uri"]) if data.get("remote_uri") else None,
            license=str(data["license"]) if data.get("license") else None,
            tags=tuple(str(tag) for tag in data.get("tags", ())),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class SimulationShard:
    dataset: str
    split: str
    uri: str
    estimated_size_gb: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SimulationWellRegistry:
    """Catalog and plan simulation datasets without forcing huge downloads."""

    schema_version = 1

    def __init__(self, manifest_path: str | Path | None = None) -> None:
        if manifest_path is None:
            try:
                from core.config import config

                manifest_path = config.paths.data_dir / "simulation_well" / "manifest.json"
            except (ImportError, AttributeError, RuntimeError):
                manifest_path = Path.home() / ".aura" / "data" / "simulation_well" / "manifest.json"
        self.manifest_path = Path(manifest_path)
        self._datasets: dict[str, SimulationDataset] = {}
        self._load_error: str | None = None
        self._load()

    def register(self, dataset: SimulationDataset) -> None:
        self._datasets[dataset.name] = dataset
        self._persist()

    def register_many(self, datasets: Iterable[SimulationDataset]) -> None:
        for dataset in datasets:
            self._datasets[dataset.name] = dataset
        self._persist()

    def get(self, name: str) -> SimulationDataset:
        try:
            return self._datasets[name]
        except KeyError as exc:
            raise KeyError(f"unknown simulation dataset: {name}") from exc

    def list(
        self,
        *,
        domain: str | None = None,
        tag: str | None = None,
        local_only: bool = False,
    ) -> builtins.list[SimulationDataset]:
        datasets = list(self._datasets.values())
        if domain:
            datasets = [dataset for dataset in datasets if dataset.domain == domain]
        if tag:
            datasets = [dataset for dataset in datasets if tag in dataset.tags]
        if local_only:
            datasets = [dataset for dataset in datasets if dataset.local_path]
        return sorted(datasets, key=lambda dataset: dataset.name)

    def plan_shards(
        self,
        names: Iterable[str],
        *,
        split: str = "train",
        max_total_gb: float | None = None,
        prefer_local: bool = True,
    ) -> builtins.list[SimulationShard]:
        shards: list[SimulationShard] = []
        total = 0.0
        for name in names:
            dataset = self.get(name)
            if split not in dataset.splits:
                raise ValueError(f"{dataset.name} has no split {split!r}")
            size = dataset.size_gb
            if max_total_gb is not None and size is not None and total + size > max_total_gb:
                continue
            uri = self._split_uri(dataset, split, prefer_local=prefer_local)
            shards.append(
                SimulationShard(
                    dataset=dataset.name,
                    split=split,
                    uri=uri,
                    estimated_size_gb=size,
                    metadata={"domain": dataset.domain, "tags": list(dataset.tags)},
                )
            )
            if size is not None:
                total += size
        return shards

    def stream_records(self, name: str, *, split: str = "train", limit: int | None = None) -> Iterator[dict[str, Any]]:
        dataset = self.get(name)
        if not dataset.local_path:
            raise FileNotFoundError(f"{name} is not staged locally")
        base = Path(dataset.local_path)
        candidates = [
            base / f"{split}.jsonl",
            base / f"{split}.json",
            base / f"{split}.csv",
            base if base.is_file() else base / "data.jsonl",
        ]
        source = next((path for path in candidates if path.exists()), None)
        if source is None:
            raise FileNotFoundError(f"no readable local split for {name}:{split}")

        count = 0
        suffix = source.suffix.lower()
        if suffix == ".jsonl":
            with source.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if limit is not None and count >= limit:
                        break
                    if line.strip():
                        count += 1
                        yield json.loads(line)
        elif suffix == ".json":
            data = json.loads(source.read_text(encoding="utf-8"))
            records = data if isinstance(data, list) else data.get("records", [])
            for record in records:
                if limit is not None and count >= limit:
                    break
                count += 1
                yield dict(record)
        elif suffix == ".csv":
            with source.open("r", encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    if limit is not None and count >= limit:
                        break
                    count += 1
                    yield dict(row)
        else:
            raise ValueError(f"unsupported local split format: {source}")

    def export_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "datasets": [asdict(dataset) for dataset in self.list()],
        }

    def _split_uri(self, dataset: SimulationDataset, split: str, *, prefer_local: bool) -> str:
        if prefer_local and dataset.local_path:
            return str(Path(dataset.local_path) / f"{split}")
        if dataset.remote_uri:
            return f"{dataset.remote_uri.rstrip('/')}/{split}"
        if dataset.local_path:
            return str(Path(dataset.local_path) / f"{split}")
        raise FileNotFoundError(f"{dataset.name} has no local_path or remote_uri")

    def _load(self) -> None:
        if not self.manifest_path.exists():
            return
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if int(data.get("schema_version", 0)) != self.schema_version:
                raise ValueError("unsupported simulation well manifest schema")
            self._datasets = {
                dataset.name: dataset
                for dataset in (
                    SimulationDataset.from_dict(raw)
                    for raw in data.get("datasets", [])
                )
            }
        except (
            OSError,
            ConnectionError,
            TimeoutError,
            ValueError,
            TypeError,
            KeyError,
        ) as exc:
            self._datasets = {}
            self._load_error = str(exc)
            record_degradation("simulation_well", exc)

    def _persist(self) -> None:
        atomic_write_text(
            self.manifest_path,
            json.dumps(self.export_manifest(), indent=2, sort_keys=True),
        )


def default_simulation_well(manifest_path: str | Path | None = None) -> SimulationWellRegistry:
    registry = SimulationWellRegistry(manifest_path)
    if registry._load_error is not None or registry.list():
        return registry
    registry.register_many(
        [
            SimulationDataset(
                name="active_matter",
                domain="biophysics",
                description="Particle and field dynamics for active matter style experiments.",
                tags=("spatiotemporal", "physics", "benchmark"),
                remote_uri="hf://datasets/polymathic-ai/active_matter",
                license="external-metadata-only",
            ),
            SimulationDataset(
                name="fluid_dynamics",
                domain="fluids",
                description="Fluid surrogate-modeling placeholder for local staged datasets.",
                tags=("pde", "surrogate", "benchmark"),
                license="operator-supplied",
            ),
        ]
    )
    return registry


__all__ = [
    "SimulationDataset",
    "SimulationShard",
    "SimulationWellRegistry",
    "default_simulation_well",
]
