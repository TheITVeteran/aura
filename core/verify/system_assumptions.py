"""The assumptions Aura's guarantees actually rest on.

seL4 ships a page called *What the Proofs Assume*. This is Aura's. Every gate
in this repository is a conditional statement, and the conditions were until
now unwritten — which means a green suite read as unconditional.

Nothing here is a to-do list in disguise. OUTSIDE_THE_SYSTEM entries are not
debt and will never be discharged; they are the edge of what this process can
establish about itself, and naming the edge is the point. UNDISCHARGED entries
are debt, and each says what paying it would take.

Import this module to populate the registry. It is data, deliberately: putting
the assumptions in one readable file is most of the value, because the audience
is a person deciding how much to believe.
"""

from __future__ import annotations

from core.verify.assumptions import AssumptionStatus as S
from core.verify.assumptions import assume

# ---------------------------------------------------------------------------
# Underneath us. seL4's compiler/hardware category — never discharged.
# ---------------------------------------------------------------------------

assume(
    "host.fsync_reaches_stable_storage",
    scope="persistence",
    owner="core/runtime/atomic_writer.py",
    status=S.OUTSIDE_THE_SYSTEM,
    statement=(
        "os.fsync() on the write path pushes data all the way to stable storage "
        "before returning."
    ),
    breaks=(
        "Durability of every governed write, including mind-state backups and the "
        "hash-chained ledgers. A power loss could lose acknowledged writes, and a "
        "restored backup could be silently short of what was committed."
    ),
    note=(
        "REPAIRED for the lane that needs it, and still true for the rest. macOS "
        "fsync() hands the write to the drive and returns without flushing the "
        "drive's own cache; fcntl(fd, F_FULLFSYNC) is what actually flushes it. "
        "atomic_write_bytes/text/json now take power_safe=True, which requests "
        "F_FULLFSYNC on both the file and its parent directory and falls back to "
        "plain fsync (once, remembered) on filesystems that refuse it. "
        "The identity ledger — commitments, preference history, identity snapshots "
        "— uses it. "
        "It is not the default because the cost was measured rather than guessed: "
        "40 writes of 4KB on this host gave 0.214ms median for fsync against "
        "8.006ms for F_FULLFSYNC, roughly 37x. Paying that on every write would "
        "trade a silent correctness gap for a loud liveness one, and an on-loop "
        "fsync here once froze the live event loop for twenty minutes. Worth "
        "noting the tails ran the other way: fsync's worst case was 27.1ms against "
        "F_FULLFSYNC's 11.1ms. "
        "So the residual assumption is narrow and real: every write that has NOT "
        "opted in survives process death but not power loss."
    ),
)

assume(
    "host.wall_clock_moves_forward",
    scope="memory",
    owner="core/cognition/actr_activation.py",
    status=S.OUTSIDE_THE_SYSTEM,
    statement=(
        "time.time() is non-decreasing across a session, apart from bounded NTP "
        "adjustment."
    ),
    breaks=(
        "Base-level activation, which is a function of elapsed time. A backward "
        "jump makes memories appear to be from the future; they clamp to the "
        "minimum age and become maximally active, so recall would prefer exactly "
        "the wrong traces."
    ),
    note=(
        "Not discharged because the clock is the OS's. The exposure is bounded "
        "rather than eliminated: ages are clamped at _MIN_AGE_S so a skewed "
        "timestamp degrades ranking instead of producing infinite activation. "
        "Discharging it properly means storing a monotonic counter alongside every "
        "wall-clock timestamp in the episodic store, which is a schema change."
    ),
)

assume(
    "host.memory_is_not_silently_corrupted",
    scope="runtime",
    owner="core/runtime",
    status=S.OUTSIDE_THE_SYSTEM,
    statement="RAM does not flip bits underneath the process without detection.",
    breaks=(
        "Every in-memory invariant, and the hash chains that would otherwise catch "
        "tampering — a bit flip after hashing is indistinguishable from correct data."
    ),
    note=(
        "The host is consumer Apple Silicon without ECC. Flight software answers "
        "this with memory scrubbing and EDAC; there is no equivalent available from "
        "userspace here. Persisted state is hash-chained, so corruption that "
        "survives to disk is detectable on read even though it is not preventable."
    ),
)

assume(
    "substrate.numeric_determinism",
    scope="cognition",
    owner="core/brain/llm",
    status=S.OUTSIDE_THE_SYSTEM,
    statement=(
        "The same weights, prompt and seed produce the same logits across runs on "
        "this machine."
    ),
    breaks=(
        "Every A/B and lesion measurement. A non-deterministic substrate makes the "
        "control arm differ from the treatment arm for reasons unrelated to the "
        "manipulation, which inflates apparent effects."
    ),
    note=(
        "MLX kernel scheduling and fused-op selection are outside this process. "
        "Mitigated rather than discharged: measurements requiring a verdict run a "
        "null arm through the identical path, so substrate noise appears in both "
        "arms instead of being attributed to the treatment."
    ),
)

# ---------------------------------------------------------------------------
# Debt. Checkable here, unchecked so far.
# ---------------------------------------------------------------------------

assume(
    "evaluation.no_external_adjudication",
    scope="claims",
    owner="artifacts/current/agi_live/RETRACTION.json",
    status=S.UNDISCHARGED,
    statement=(
        "Every battery, baseline and scoring rule in this repository was authored "
        "and administered by this project."
    ),
    breaks=(
        "The external validity of every capability number. A self-authored battery "
        "cannot rule out a handicap that favours the system under test — which has "
        "already happened once here and was caught only from the inside."
    ),
    note=(
        "RETRACTION.json names the replacement requirement itself: a battery "
        "authored or administered by someone other than this project. That "
        "requirement is still unmet, so this is the assumption behind every "
        "capability claim. Discharging it needs a third party, not more code."
    ),
)

assume(
    "memory.retrieval_curve_is_fitted",
    scope="memory",
    owner="core/cognition/actr_activation.py",
    status=S.DISCHARGED,
    statement=(
        "The retrieval threshold tau and noise s are fit to Aura's own measured "
        "recall, not carried from published defaults."
    ),
    breaks=(
        "Any claim that activation predicts WHICH memories come back. Unfitted "
        "parameters would make that a statement about ACT-R rather than about Aura."
    ),
    discharged_by="tests/test_actr_fit.py::test_fitted_parameters_are_distinct_from_the_published_defaults",
    note=(
        "tau=-0.4666, s=2.0 by maximum likelihood over 6,000 samples; Brier skill "
        "0.154 over base rate. Reproduce with tools/fit_actr_retrieval.py."
    ),
)

assume(
    "memory.latency_equation_does_not_transfer",
    scope="memory",
    owner="core/cognition/actr_activation.py",
    status=S.DISCHARGED,
    statement=(
        "No absolute retrieval latency is predicted from activation, because "
        "measured latency does not depend on activation here."
    ),
    breaks=(
        "Nothing, as stated — but the inverse would break a great deal. F is a "
        "pure multiplicative scale and would absorb any timing at all, so a fitted "
        "F over an absent relationship would be a confident number with no "
        "mechanism under it."
    ),
    discharged_by="tests/test_actr_fit.py::test_the_latency_null_still_holds_on_the_live_ranking_path",
    note=(
        "Measured r^2 = 0.000037 regressing ln(T) on -A over 6,000 samples. "
        "T = F*e^-A earns its shape in ACT-R because retrieval is a race between "
        "activations; Aura's recall is a ranked scan whose cost tracks candidate "
        "count and store behaviour. The fitting tool refuses to emit an F below "
        "r^2=0.10, and the bound test fails if retrieval ever becomes "
        "activation-driven — at which point F becomes worth fitting."
    ),
)

assume(
    "workspace.bidders_are_cooperative",
    scope="consciousness",
    owner="core/consciousness/global_workspace.py",
    status=S.UNDISCHARGED,
    statement=(
        "Subsystems submit bids reflecting their genuine salience rather than "
        "inflating priority to win the broadcast."
    ),
    breaks=(
        "The competition. Adaptation bounds how long any one source can hold the "
        "workspace, but nothing bounds what a source may claim its priority is, so "
        "a miscalibrated producer can crowd the field indefinitely."
    ),
    note=(
        "Discharging it means calibrating each producer's priority distribution "
        "against downstream outcome value and rejecting bids that are out of "
        "distribution for their source. The per-source data needed for that is "
        "already recorded in the broadcast history."
    ),
)

# ---------------------------------------------------------------------------
# Discharged. Each names a checker, and the checker is verified to exist.
# ---------------------------------------------------------------------------

assume(
    "architecture.no_upward_imports",
    scope="runtime",
    owner="tools/check_layering.py",
    status=S.DISCHARGED,
    statement=(
        "core/runtime and core/observability do not import cognition or agency, so "
        "a check can still load when the thing it checks is broken."
    ),
    breaks=(
        "The ability to diagnose a broken subsystem, which is the only moment "
        "diagnosis matters."
    ),
    discharged_by="make layering",
)

assume(
    "persistence.no_sync_fsync_inside_async",
    scope="persistence",
    owner="core/runtime/file_write_gateway.py",
    status=S.DISCHARGED,
    statement="No new synchronous file write happens inside an async def.",
    breaks=(
        "Event-loop liveness. An on-loop fsync in this codebase once froze the live "
        "runtime for twenty minutes."
    ),
    discharged_by="tests/test_async_write_lane_ratchet.py",
)

assume(
    "workspace.competition_actually_rotates",
    scope="consciousness",
    owner="core/consciousness/global_workspace.py",
    status=S.DISCHARGED,
    statement=(
        "In steady state the broadcast rotates across every competitive source, "
        "and bid strength still decides who wins more often."
    ),
    breaks=(
        "The global workspace as an architecture. Without rotation it is a sort; "
        "without order-preservation it is a coin flip."
    ),
    discharged_by=(
        "tests/test_global_workspace_competes.py::"
        "test_a_two_point_gap_does_not_buy_a_monopoly"
    ),
)

assume(
    "memory.activation_is_scale_free",
    scope="memory",
    owner="core/cognition/actr_activation.py",
    status=S.DISCHARGED,
    statement=(
        "Memory activation depends only on elapsed time, never on absolute "
        "wall-clock position."
    ),
    breaks=(
        "Recall ranking, silently and progressively. The scorer this replaced was "
        "keyed to a hardcoded epoch and had decayed to a constant across the whole "
        "operating range before anyone noticed."
    ),
    discharged_by="tests/test_actr_activation.py::test_recency_is_scale_free",
)
