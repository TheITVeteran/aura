"""SPARK-002 primary-literature dossier: versioned, typed, hash-bindable.

Every Spark mechanism traces to primary literature.  This registry is the
single versioned bibliography: each entry names the primary paper, the
mechanism family it grounds, which SPARK items rely on it, its claim
status — ``replicated`` (independently reproduced across groups),
``reported`` (credible primary result, limited independent replication),
or ``proposal`` (design/position work) — and its declared license.
``validate_literature`` fails closed on duplicate ids, malformed arXiv
identifiers, dangling SPARK references, or missing mechanism coverage,
and produces the registry digest the SPARK-072 methods package binds.

PDF byte hashes are deliberately NOT recorded here: the sealed methods
package assembles and hashes the exact source files at packaging time
against these immutable arXiv version identifiers, and per-PDF license
text is re-verified there.  The dossier's job is that no mechanism rests
on a placeholder citation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Never

from core.brain.llm.latent_cortex.epistemic_state import canonical_sha256

LITERATURE_SCHEMA = "aura.latent_cortex.spark_literature.v1"
LITERATURE_VERSION = "2026.07.23.1"

CLAIM_STATUSES = ("replicated", "reported", "proposal")

# Every mechanism family the Spark blueprints rely on must be grounded.
REQUIRED_MECHANISMS = (
    "self_consistency",
    "process_reward",
    "verifier_training",
    "process_vs_outcome",
    "self_taught_reasoning",
    "latent_thought",
    "recurrent_depth",
    "adaptive_computation",
    "tree_search",
    "bandit_tree_policy",
    "self_correction_limits",
    "mistake_location",
    "rl_self_correction",
    "sycophancy",
    "iterative_refinement",
    "test_time_compute",
    "policy_optimization",
    "verifiable_rewards",
    "fast_weights",
    "low_rank_adaptation",
)

_ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


class LiteratureError(ValueError):
    """Stable fail-closed literature-dossier error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise LiteratureError(code)


@dataclass(frozen=True, slots=True)
class LiteratureEntry:
    """One primary source and what in Spark stands on it."""

    entry_id: str
    mechanism: str
    title: str
    authors: str
    year: int
    venue: str
    arxiv_id: str
    claim_status: str
    license_declared: str
    supports: str
    spark_items: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.entry_id, str)
            or not re.fullmatch(r"[a-z0-9_]{3,60}", self.entry_id or "")
            or self.mechanism not in REQUIRED_MECHANISMS
            or not isinstance(self.title, str)
            or len(self.title) < 8
            or not isinstance(self.authors, str)
            or len(self.authors) < 4
            or isinstance(self.year, bool)
            or not isinstance(self.year, int)
            or not 1990 <= self.year <= 2026
            or not isinstance(self.venue, str)
            or not self.venue
            or not isinstance(self.arxiv_id, str)
            or not (self.arxiv_id == "" or _ARXIV_ID.fullmatch(self.arxiv_id))
            or self.claim_status not in CLAIM_STATUSES
            or not isinstance(self.license_declared, str)
            or len(self.license_declared) < 5
            or not isinstance(self.supports, str)
            or len(self.supports) < 20
            or not isinstance(self.spark_items, tuple)
            or not self.spark_items
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or not 1 <= item <= 72
                for item in self.spark_items
            )
        ):
            _fail("literature_entry_invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "mechanism": self.mechanism,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "arxiv_id": self.arxiv_id,
            "claim_status": self.claim_status,
            "license_declared": self.license_declared,
            "supports": self.supports,
            "spark_items": list(self.spark_items),
        }


_ARXIV_LICENSE = "arXiv non-exclusive license (re-verify per PDF at packaging)"

ENTRIES: tuple[LiteratureEntry, ...] = (
    LiteratureEntry(
        entry_id="wang_2022_self_consistency",
        mechanism="self_consistency",
        title="Self-Consistency Improves Chain of Thought Reasoning in Language Models",
        authors="Wang, Wei, Schuurmans, Le, Chi, Narang, Chowdhery, Zhou",
        year=2022,
        venue="ICLR 2023",
        arxiv_id="2203.11171",
        claim_status="replicated",
        license_declared=_ARXIV_LICENSE,
        supports=(
            "Independent parallel sampling with agreement-based selection "
            "grounds fresh-context branch isolation and vote weighting."
        ),
        spark_items=(14, 15, 16, 17),
    ),
    LiteratureEntry(
        entry_id="lightman_2023_lets_verify",
        mechanism="process_reward",
        title="Let's Verify Step by Step",
        authors="Lightman, Kosaraju, Burda, Edwards, Baker, Lee, Leike, Schulman, Sutskever, Cobbe",
        year=2023,
        venue="ICLR 2024",
        arxiv_id="2305.20050",
        claim_status="replicated",
        license_declared=_ARXIV_LICENSE,
        supports=(
            "Step-level process reward models outperform outcome-only "
            "grading; grounds the process verifier and step scoring."
        ),
        spark_items=(39, 41, 46),
    ),
    LiteratureEntry(
        entry_id="cobbe_2021_verifiers",
        mechanism="verifier_training",
        title="Training Verifiers to Solve Math Word Problems",
        authors="Cobbe, Kosaraju, Bavarian, Chen, Jun, Kaiser, et al.",
        year=2021,
        venue="arXiv",
        arxiv_id="2110.14168",
        claim_status="replicated",
        license_declared=_ARXIV_LICENSE,
        supports=(
            "Trained verifiers ranking sampled solutions ground the "
            "verifier-mesh design and candidate reranking."
        ),
        spark_items=(39, 40, 42),
    ),
    LiteratureEntry(
        entry_id="uesato_2022_process_outcome",
        mechanism="process_vs_outcome",
        title="Solving math word problems with process- and outcome-based feedback",
        authors="Uesato, Kushman, Kumar, Song, Siegel, Wang, Creswell, Irving, Higgins",
        year=2022,
        venue="arXiv (DeepMind)",
        arxiv_id="2211.14275",
        claim_status="replicated",
        license_declared=_ARXIV_LICENSE,
        supports=(
            "Process supervision reduces reasoning errors at matched final "
            "accuracy; grounds preferring process-graded acceptance."
        ),
        spark_items=(41, 60),
    ),
    LiteratureEntry(
        entry_id="zelikman_2022_star",
        mechanism="self_taught_reasoning",
        title="STaR: Bootstrapping Reasoning With Reasoning",
        authors="Zelikman, Wu, Mu, Goodman",
        year=2022,
        venue="NeurIPS 2022",
        arxiv_id="2203.14465",
        claim_status="replicated",
        license_declared=_ARXIV_LICENSE,
        supports=(
            "Generate-filter-finetune loops on verified traces ground the "
            "verified STaR flywheel and replay buffer."
        ),
        spark_items=(58, 63),
    ),
    LiteratureEntry(
        entry_id="zelikman_2024_quiet_star",
        mechanism="latent_thought",
        title="Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking",
        authors="Zelikman, Harik, Shao, Jayasiri, Haber, Goodman",
        year=2024,
        venue="arXiv",
        arxiv_id="2403.09629",
        claim_status="reported",
        license_declared=_ARXIV_LICENSE,
        supports=(
            "Internal token-level thought before emission grounds silent "
            "latent deliberation ahead of the answer channel."
        ),
        spark_items=(23, 61),
    ),
    LiteratureEntry(
        entry_id="hao_2024_coconut",
        mechanism="latent_thought",
        title="Training Large Language Models to Reason in a Continuous Latent Space",
        authors="Hao, Sukhbaatar, Su, Li, Hu, Weston, Tian",
        year=2024,
        venue="COLM 2025",
        arxiv_id="2412.06769",
        claim_status="reported",
        license_declared=_ARXIV_LICENSE,
        supports=(
            "Feeding hidden states back as input embeddings grounds the "
            "continuous latent workspace instead of token-only thought."
        ),
        spark_items=(23, 24, 38),
    ),
    LiteratureEntry(
        entry_id="geiping_2025_recurrent_depth",
        mechanism="recurrent_depth",
        title="Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach",
        authors="Geiping, McLeish, Jain, Kirchenbauer, Singh, Bartoldson, Kailkhura, Bhatele, Goldstein",
        year=2025,
        venue="arXiv",
        arxiv_id="2502.05171",
        claim_status="reported",
        license_declared=_ARXIV_LICENSE,
        supports=(
            "Shared-parameter looped cores whose effective depth grows with "
            "iterations ground the looped middle-layer recurrence."
        ),
        spark_items=(24, 26, 62),
    ),
    LiteratureEntry(
        entry_id="dehghani_2018_universal_transformers",
        mechanism="recurrent_depth",
        title="Universal Transformers",
        authors="Dehghani, Gouws, Vinyals, Uszkoreit, Kaiser",
        year=2018,
        venue="ICLR 2019",
        arxiv_id="1807.03819",
        claim_status="replicated",
        license_declared=_ARXIV_LICENSE,
        supports=(
            "Weight-tied recurrent application of a transformer block with "
            "dynamic halting is the primary ancestor of looped depth."
        ),
        spark_items=(24, 26),
    ),
    LiteratureEntry(
        entry_id="graves_2016_act",
        mechanism="adaptive_computation",
        title="Adaptive Computation Time for Recurrent Neural Networks",
        authors="Graves",
        year=2016,
        venue="arXiv (DeepMind)",
        arxiv_id="1603.08983",
        claim_status="replicated",
        license_declared=_ARXIV_LICENSE,
        supports=(
            "Learned halting under a compute penalty grounds the stop and "
            "convergence gates and depth-by-difficulty scaling."
        ),
        spark_items=(26, 52, 53),
    ),
    LiteratureEntry(
        entry_id="yao_2023_tree_of_thoughts",
        mechanism="tree_search",
        title="Tree of Thoughts: Deliberate Problem Solving with Large Language Models",
        authors="Yao, Yu, Zhao, Shafran, Griffiths, Cao, Narasimhan",
        year=2023,
        venue="NeurIPS 2023",
        arxiv_id="2305.10601",
        claim_status="replicated",
        license_declared=_ARXIV_LICENSE,
        supports=(
            "Explicit search over intermediate reasoning states with "
            "backtracking grounds the latent tree/forest search."
        ),
        spark_items=(35, 38, 49),
    ),
    LiteratureEntry(
        entry_id="kocsis_2006_uct",
        mechanism="bandit_tree_policy",
        title="Bandit based Monte-Carlo Planning",
        authors="Kocsis, Szepesvári",
        year=2006,
        venue="ECML 2006",
        arxiv_id="",
        claim_status="replicated",
        license_declared="Springer LNCS (verify at packaging)",
        supports=(
            "UCT confidence-bound tree policies ground exploration versus "
            "exploitation over partial reasoning states."
        ),
        spark_items=(38, 51),
    ),
    LiteratureEntry(
        entry_id="huang_2023_cannot_self_correct",
        mechanism="self_correction_limits",
        title="Large Language Models Cannot Self-Correct Reasoning Yet",
        authors="Huang, Chen, Mishra, Zheng, Yu, Song, Zhou",
        year=2023,
        venue="ICLR 2024",
        arxiv_id="2310.01798",
        claim_status="replicated",
        license_declared=_ARXIV_LICENSE,
        supports=(
            "Intrinsic self-correction without oracle feedback degrades "
            "accuracy; the negative result motivating the whole program."
        ),
        spark_items=(1, 3, 50),
    ),
    LiteratureEntry(
        entry_id="tyen_2023_mistake_location",
        mechanism="mistake_location",
        title="LLMs cannot find reasoning errors, but can correct them given the error location",
        authors="Tyen, Mansoor, Cărbune, Chen, Mak",
        year=2023,
        venue="ACL Findings 2024",
        arxiv_id="2311.08516",
        claim_status="reported",
        license_declared=_ARXIV_LICENSE,
        supports=(
            "Detection, not correction, is the bottleneck; grounds the "
            "dedicated trained mistake locator."
        ),
        spark_items=(29, 47),
    ),
    LiteratureEntry(
        entry_id="kumar_2024_score",
        mechanism="rl_self_correction",
        title="Training Language Models to Self-Correct via Reinforcement Learning",
        authors="Kumar, Zhuang, Agarwal, Su, Co-Reyes, et al.",
        year=2024,
        venue="ICLR 2025",
        arxiv_id="2409.12917",
        claim_status="reported",
        license_declared=_ARXIV_LICENSE,
        supports=(
            "Multi-turn RL that rewards measured improvement between "
            "attempts grounds the delta-reward revision objective."
        ),
        spark_items=(60, 61),
    ),
    LiteratureEntry(
        entry_id="sharma_2023_sycophancy",
        mechanism="sycophancy",
        title="Towards Understanding Sycophancy in Language Models",
        authors="Sharma, Tong, Korbak, Duvenaud, et al.",
        year=2023,
        venue="ICLR 2024",
        arxiv_id="2310.13548",
        claim_status="replicated",
        license_declared=_ARXIV_LICENSE,
        supports=(
            "Preference-trained deference to doubt framing grounds blind, "
            "role-separated, decoy-balanced review."
        ),
        spark_items=(18, 19),
    ),
    LiteratureEntry(
        entry_id="madaan_2023_self_refine",
        mechanism="iterative_refinement",
        title="Self-Refine: Iterative Refinement with Self-Feedback",
        authors="Madaan, Tandon, Gupta, Hallinan, et al.",
        year=2023,
        venue="NeurIPS 2023",
        arxiv_id="2303.17651",
        claim_status="reported",
        license_declared=_ARXIV_LICENSE,
        supports=(
            "The strongest same-context refinement baseline the Spark "
            "design must beat, not emulate."
        ),
        spark_items=(1, 3),
    ),
    LiteratureEntry(
        entry_id="snell_2024_test_time_compute",
        mechanism="test_time_compute",
        title="Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters",
        authors="Snell, Lee, Xu, Kumar",
        year=2024,
        venue="arXiv",
        arxiv_id="2408.03314",
        claim_status="reported",
        license_declared=_ARXIV_LICENSE,
        supports=(
            "Compute-optimal allocation across sampling, search, and "
            "revision grounds adaptive breadth/depth/tool routing."
        ),
        spark_items=(51, 52),
    ),
    LiteratureEntry(
        entry_id="shao_2024_grpo",
        mechanism="policy_optimization",
        title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
        authors="Shao, Wang, Zhu, Xu, Song, Bi, et al.",
        year=2024,
        venue="arXiv (DeepSeek)",
        arxiv_id="2402.03300",
        claim_status="replicated",
        license_declared=_ARXIV_LICENSE,
        supports=(
            "Group-relative policy optimization is the exact training "
            "algorithm of the resident recurrent-GRPO campaigns."
        ),
        spark_items=(59, 60, 69),
    ),
    LiteratureEntry(
        entry_id="lambert_2024_tulu3_rlvr",
        mechanism="verifiable_rewards",
        title="Tulu 3: Pushing Frontiers in Open Language Model Post-Training",
        authors="Lambert, Morrison, Pyatkin, Huang, Ivison, et al.",
        year=2024,
        venue="arXiv (AI2)",
        arxiv_id="2411.15124",
        claim_status="reported",
        license_declared=_ARXIV_LICENSE,
        supports=(
            "Reinforcement learning against verifiable rewards grounds the "
            "verifier-gated training admission discipline."
        ),
        spark_items=(60, 69),
    ),
    LiteratureEntry(
        entry_id="ba_2016_fast_weights",
        mechanism="fast_weights",
        title="Using Fast Weights to Attend to the Recent Past",
        authors="Ba, Hinton, Mnih, Leibo, Ionescu",
        year=2016,
        venue="NeurIPS 2016",
        arxiv_id="1610.06258",
        claim_status="replicated",
        license_declared=_ARXIV_LICENSE,
        supports=(
            "Short-lived weight adaptation layered over slow weights "
            "grounds query-scoped fast-weight learning with erasure."
        ),
        spark_items=(55, 56),
    ),
    LiteratureEntry(
        entry_id="hu_2021_lora",
        mechanism="low_rank_adaptation",
        title="LoRA: Low-Rank Adaptation of Large Language Models",
        authors="Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen",
        year=2021,
        venue="ICLR 2022",
        arxiv_id="2106.09685",
        claim_status="replicated",
        license_declared=_ARXIV_LICENSE,
        supports=(
            "Low-rank adapters are the mechanical substrate of every "
            "treatment delta the campaigns train, freeze, and verify."
        ),
        spark_items=(55, 64, 69),
    ),
)


def validate_literature() -> dict[str, Any]:
    """Fail closed unless the dossier is complete, unique, and well-formed."""

    entry_ids: set[str] = set()
    arxiv_ids: set[str] = set()
    mechanisms: set[str] = set()
    for entry in ENTRIES:
        if entry.entry_id in entry_ids:
            _fail("literature_duplicate_entry")
        entry_ids.add(entry.entry_id)
        if entry.arxiv_id:
            if entry.arxiv_id in arxiv_ids:
                _fail("literature_duplicate_arxiv_id")
            arxiv_ids.add(entry.arxiv_id)
        mechanisms.add(entry.mechanism)
    if mechanisms != set(REQUIRED_MECHANISMS):
        _fail("literature_mechanism_coverage_incomplete")
    body = {
        "schema": LITERATURE_SCHEMA,
        "version": LITERATURE_VERSION,
        "entry_count": len(ENTRIES),
        "entries": [entry.to_dict() for entry in ENTRIES],
    }
    return {**body, "registry_sha256": canonical_sha256(body)}


def render_literature_markdown() -> str:
    """Deterministic human-readable dossier; the committed doc must match."""

    receipt = validate_literature()
    lines = [
        "# Spark primary-literature dossier",
        "",
        f"Version `{LITERATURE_VERSION}` — registry digest "
        f"`{receipt['registry_sha256']}`.",
        "",
        "Generated by `tools/render_spark_literature.py` from",
        "`core/brain/llm/latent_cortex/literature.py`; edit the registry,",
        "never this file. Claim status: **replicated** = independently",
        "reproduced across groups; **reported** = credible primary result",
        "with limited independent replication; **proposal** = design or",
        "position work. PDF byte hashes and per-PDF license text are bound",
        "at methods-package assembly (SPARK-072) against the immutable",
        "identifiers recorded here.",
        "",
    ]
    for entry in ENTRIES:
        identifier = (
            f"arXiv:{entry.arxiv_id}" if entry.arxiv_id else entry.venue
        )
        lines.extend(
            [
                f"## {entry.title}",
                "",
                f"- **Entry**: `{entry.entry_id}` ({entry.mechanism})",
                f"- **Authors**: {entry.authors}",
                f"- **Venue**: {entry.venue} ({entry.year}); {identifier}",
                f"- **Claim status**: {entry.claim_status}",
                f"- **License (declared)**: {entry.license_declared}",
                f"- **Grounds**: {entry.supports}",
                "- **SPARK items**: "
                + ", ".join(f"SPARK-{item:03d}" for item in entry.spark_items),
                "",
            ]
        )
    return "\n".join(lines)


__all__ = [
    "CLAIM_STATUSES",
    "ENTRIES",
    "LITERATURE_SCHEMA",
    "LITERATURE_VERSION",
    "REQUIRED_MECHANISMS",
    "LiteratureEntry",
    "LiteratureError",
    "render_literature_markdown",
    "validate_literature",
]
