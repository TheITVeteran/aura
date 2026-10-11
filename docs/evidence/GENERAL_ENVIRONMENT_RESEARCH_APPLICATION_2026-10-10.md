# General environment research application, 10 October 2026

Audited revision: `8f2879ec2`. This record separates existing code, research
principles used in that code, and proposed work. Presentation qualification
remains **unproved**. A current code path and a synthetic test do not establish
completed gameplay in the ordinary app.

The implementation has no archived-game solution, named winning layout, private
game-state reader, or trained policy added by these checkpoints. The mechanisms
operate on observed controls, geometry, delivered input, required effects and
execution owners. That scope permits transfer; it does not prove performance
across random games.

## Research principles present in code

| Primary source | Applied mechanism | Code and validation boundary |
| --- | --- | --- |
| [BehaviorTree.CPP sequences](https://www.behaviortree.dev/docs/nodes-library/SequenceNode/) | Retain confirmed requirements while checking current prerequisites between inputs. Reopening a selector need not erase its earlier completion. | `core/cognition/reading_the_rules.py`, `core/skills/screen_step_evidence.py`; observed-step and choice-binding regressions. This is Aura's procedure implementation. |
| [BehaviorTree.CPP asynchronous actions](https://www.behaviortree.dev/docs/guides/asynchronous_nodes/) and [AutoGen cancellation source](https://microsoft.github.io/autogen/stable/_modules/autogen_core/_cancellation_token.html) | Bind running work to its owner, cancel pending work, release input during cleanup and detach listeners. | `core/runtime/what_stops_it.py`, executor/pursuit scopes, inference gate/router and `core/brain/llm/mlx_local_client.py`. Two ordinary UI Stop cases acknowledged owned generation cancellation on 10 October; natural completion needs separate lifetime closure. |
| [Real-time action chunk execution](https://arxiv.org/abs/2506.07339) | Distinguish committed input from future computation; bind physical evidence to the delivery interval. | `core/perception/observed_transfer.py`, `core/skills/screen_pursuit_acting.py`. Transfer receipts reject stale captures, unrelated animation and existing occupancy. The learned flow policy was not transplanted. |
| [OpenCV template matching](https://docs.opencv.org/4.13.0/d4/dc6/tutorial_py_template_matching.html) | Search fresh pixels for a remembered appearance and reject ambiguous matches. | `core/perception/visual_anchor.py`, `_where_the_way_on_was`; `tests/test_visual_anchor.py`. Bounded local movement and uniform fading are covered. Arbitrary motion, enabledness and global identity are not covered. |
| [AppAgent](https://arxiv.org/abs/2312.13771) | Keep the observed result of an interface action for later navigation. | Existing `core/agency/where_things_lead.py` records screens, action edges and introduced/removed labels. It is a measured navigation memory; AppAgent's model-generated documentation pipeline was not installed. |

The evidence records for `288ba8ea8`, `64aefec64`, `e7dabe356` and `8f2879ec2`
contain the focused checks, ordinary Stop observations and their remaining limits.
Those records are retained without replacing their earlier verdicts.

## Existing architecture and gaps

The repository already has typed preconditions, declared effects, bounded
backward dependency search and observed execution in
`core/cognition/procedure.py`, `procedure_planning.py` and
`procedure_execution.py`. The screen path does not currently compile its
observed interface dependencies into that planner. Its local navigation uses
`WhereThingsLead.toward`, categorical selector relations and procedure ordering;
`a_plan_to_an_end.py` still asks the model for plan proposals. Typed search must
receive grounded step identities and current facts before it can replace that
proposal path. A signature describes a possible effect; executing and observing
the effect must discharge the obligation.

The immediate missing pieces at this revision are:

1. **Grounded manipulation candidates.** Source discovery still uses labels
   and detected pressable shapes. Choice binding filters out position-only
   shape labels. A newly exposed unlabeled object therefore needs measured
   identity and extent before it can become a selected source. Positive tests
   must generate and deliver the correct transfer through the normal observer,
   rather than supplying source coordinates directly.
2. **Action-local response attribution.** A changed screen can include animation
   elsewhere. Navigation/selection learning needs evidence from the acted-on
   region and its delivery window, with unrelated changes excluded. Tests need
   background animation, delayed response, unchanged controls and actual local
   response in several interface domains.
3. **Completed-operation lifetime closure.** Browser run cleanup resets its
   context and closes input observers, but outstanding guide questions can
   still finish and publish. A child owner, generation stamp, bounded teardown
   and post-await publication checks are required for natural completion,
   Stop, exceptions and reuse of a guide. Closing one run must preserve its
   parent and any sibling operation.

These are architectural requirements. The concurrent implementation work is not
counted as completed by this snapshot.

## Investigated systems not transplanted

[ByteTrack](https://arxiv.org/abs/2110.06864) retains low-confidence detections
through association. [BoT-SORT](https://arxiv.org/abs/2206.14651) combines motion,
appearance and camera-motion compensation. Both depend on suitable measured
detections and association evidence. Aura already has motion/appearance tracking
in `core/perception/what_moves_in_the_picture.py`; neither research tracker was
installed or benchmarked here. More permissive association must preserve unknown
identity when several objects remain plausible.

[ADEPT](https://www.mit.edu/~k2smith/publication/adept/) represents objects and
compares observations with physical expectations. Aura's existing picture
tracker retains temporarily occluded objects and predicts their paths, but this
audit does not establish ADEPT's inference model, human developmental mechanisms
or general physical reasoning in Aura. Current visual anchors verify local
persistence; hidden-object predictions do not authorize coordinates without
current evidence.

[DreamerV3](https://arxiv.org/abs/2301.04104) and
[TD-MPC2](https://arxiv.org/abs/2310.16828) learn world models for imagined or
optimized control. Aura's current measured transition and rehearsal code is not
either algorithm. Deploying these systems would require compatible observations,
actions, reward/termination evidence, training data, resource limits and transfer
evaluation. They were not trained, installed or credited as capability here.

General prerequisites and action effects can also support platformers, inventory
use, open-world tasks and fictional threat response. No mastery of those domains
has been established by this work. The remaining acceptance evidence includes
completed ordinary app gameplay and the random three-game request.

## Follow-on lifetime implementation

The audit's lifetime gap has an implementation under review in this checkout.
`core/cognition/guidance_lifetime.py` gives each operation a child execution
owner and a publication generation. `_one_run` enters it before constructing
the guide and closes it on normal completion, Stop and exceptions. Questions
join the owner before priority admission and remain owned through retries and
generation. Plan, counsel, code, place and avatar results check the generation
before publishing. Pending request markers retire by task identity, so an old
question cannot block a new run or clear the new run's marker. Shared kind
questions have separate queues and callbacks for each owner.

[Python's task documentation](https://docs.python.org/3.12/library/asyncio-task.html)
describes cancellation as a request delivered at suspension points and the
need for cleanup. This implementation also rejects late results from observers
that catch cancellation. Teardown invalidates publication first, then waits
at most 0.5 seconds for owned tasks; an observer still releasing its resources
has no authority to publish. The parent and sibling owners remain open.

Focused lifetime, named-place, question-priority, kind-knowledge, plan and place
reading checks passed **91 tests** in 11.19 seconds with the shared Python and
isolated test logs. This measures these code paths. Repository gates, full-suite
validation, ordinary app completion and random-game performance remain separate
requirements. The proposed typed screen-planner adapter remains unbuilt pending
grounded interface dependency inputs.

## Measured observation and response milestone

The follow-on implementation grounds anonymous source candidates in exact
captured pixels, full measured extents and container relationships. It verifies
the source again before adding it to the live action inventory. A changed
candidate in another panel cannot become a selector's value without a measured
group relation. Candidate appearance permits an interaction trial; it does not
establish a name or draggability.

[Visual Grounding for User Interfaces](https://aclanthology.org/2024.naacl-industry.9/)
uses visual organization to assist grounding. The adopted mechanism is explicit
geometric grouping. [PDDLStream](https://github.com/caelan/pddlstream) separates
symbolic requirements from black-box geometric sampling, and
[GIFT](https://www.roboticsproceedings.org/rss17/p060.html) learns affordances from
interaction. The applicable boundary here is executable: measured candidates
enter bounded actions, while fresh transfer receipts establish what those actions
did. No external trained model or planner implementation was installed.

The response stream now owns input from dispatch start through completion,
preserves capture timestamps, excludes prior owned responses from autonomous
background and keeps incomplete intervals unknown. A fresh same-surface capture
and exactly one new input window bind the response to its caller. Visual
responses can satisfy a change expectation when OCR remains identical; they
cannot satisfy a missing semantic result. Unknown measurements no longer teach
failed actions, physical nonresponse, world dynamics or action-conditioned
reward. Completed guidance loses publication authority and releases its pending
markers without stopping sibling work.

Validation before the final cropped-view and color-response checks: 325 focused
tests passed in 23.93 seconds; an additional 43-test set including the actual
object-to-body pipeline and two-panel counterfactuals passed in 6.69 seconds.
These sets overlap and are not an additive count. The three-surface pipeline
uses real screenshot decoding, candidate discovery, action inventory, normal
dispatch and independent placement verification with a fixture body. It is
offline evidence, not ordinary app proof.

Smoke passed 165 tests with one skip in 57.34 seconds. Lint, compile, layering
and writing checks passed. Governance lint still reports 101 existing baseline
regressions; an AST comparison of every changed production file found zero
changed effect buckets. No governance baseline was refreshed. Final checks and
checkpoint identity will be recorded after the remaining two response fixes.

The next demonstrated integration gaps are action refinement and visual state
identity. A physical drag can choose a currently revealed object and place it
in one input, but the procedure path still demands two independent actions.
Same-word palettes also collapse into one navigation state. Both require
grounded bindings and observed effects; merely routing them through the existing
typed planner would leave those missing facts unresolved. Presentation
qualification and the random-game request remain unproved.

Final milestone checks passed after the cropped-view and color-channel fixes:
334 focused tests in 33.05 seconds, the actual cropped-page settlement regression
separately, and smoke at 165 passed / one skipped in 59.72 seconds. Lint, compile,
layering and writing passed again. The response watcher now survives the exact
crop used for input; equal-brightness color changes are measured per channel.
No ordinary live run was started during this research implementation milestone.
