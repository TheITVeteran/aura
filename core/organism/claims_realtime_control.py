"""Register measured counter presence and control evidence."""
from typing import Any


def install_realtime_control_claims(suite: Any) -> None:
    from core.agency.a_body_seen_then import _body_seen_then_invariant
    from core.agency.a_response_worth_measuring_again import _provisional_response_invariant
    from core.agency.body_trials_keep_the_world import _body_trial_phase_invariant
    from core.agency.causal_identification import (
        _stale_and_ambiguous_identity_witnesses_remain_unknown,
    )
    from core.agency.controls_measured_on_one_thing import (
        _own_control_witness_invariant,
        _resolved_motion_invariant,
        _trajectory_control_invariant,
    )
    from core.agency.controls_ready_for_the_motion_model import _model_readiness_invariant
    from core.agency.countering_motion_as_it_appears import _counter_trial_measurement_invariant
    from core.agency.deliberate_action import _unmeasured_response_invariant
    from core.agency.giving_a_trial_time import _control_trial_time_invariant
    from core.agency.how_a_game_works import _known_body_controls_invariant
    from core.agency.keeping_a_trial_observable import _observed_viability_invariant
    from core.agency.using_a_control_on_a_thing import _object_use_target_invariant
    from core.agency.what_meeting_things_does import _absent_counters_are_not_current
    from core.agency.where_clicks_pay import _click_attribution_respects_stretch_boundaries
    from core.agency.where_things_lead import _unknown_selector_invariant
    from core.agency.which_one_answers_to_her import (
        _ambiguous_controls_remain_unknown,
        _emissions_require_distinct_input_receipts,
        _fresh_control_experiment_forgets_old_rejections,
    )
    from core.agency.which_way_she_faces import _turn_observation_invariant
    from core.agency.words_read_with_their_picture import _word_picture_start_invariant
    from core.cognition.action_refinement import _bound_refinement_invariant
    from core.cognition.guidance_lifetime import _completed_guidance_invariant
    from core.cognition.procedure_binding import _choice_binding_invariant
    from core.cognition.reading_the_rules import _procedure_completion_is_measured
    from core.organism.model_validation import (
        Claim,
        Evidence,
        Observation,
        ValidationTest,
        boolean_score,
    )
    from core.perception.a_small_thing_in_a_joined_patch import _small_joined_thing_invariant
    from core.perception.drawn_objects import _object_candidates_invariant
    from core.perception.eyes_of_their_own import _child_capture_custody_invariant
    from core.perception.foreground_appearance import _foreground_mask_invariant
    from core.perception.observed_transfer import _transfer_receipts_require_evidence
    from core.perception.parts_when_things_touch import _joined_parts_invariant
    from core.perception.relations_between_things import _relations_require_unique_finite_entities
    from core.perception.the_drawing_as_objects import _scene_requires_a_matching_complete_frame
    from core.perception.visual_anchor import _remembered_labels_invariant
    from core.perception.visual_probe_cache import _visual_probe_miss_invariant
    from core.perception.watching_it_happen import _response_window_keeps_its_evidence
    from core.perception.where_the_words_point import _named_places_invariant
    from core.perception.word_labels_seen_again import _visible_word_labels_invariant
    from core.perception.words_kept_with_their_pixels import _kept_word_pixels_invariant
    from core.runtime.action_executor import _terminal_effect_flags_invariant
    from core.runtime.executors import _interactive_and_receipt_workers_are_separate
    from core.runtime.what_she_learned import (
        _indexed_knowledge_invariant,
        _retained_structure_invariant,
    )
    from core.runtime.what_stops_it import _stop_listener_lifetime_invariant
    from core.skills.current_words_before_play import _current_play_words_invariant
    from core.skills.screen_step_evidence import (
        _construction_readiness_invariant,
        _creation_completion_invariant,
        _response_dispatch_invariant,
    )
    from core.skills.sovereign_browser_drawing import (
        _current_play_veto_invariant,
        _stalled_making_invariant,
    )
    from core.skills.sovereign_browser_picking import _picked_outcomes_invariant

    for name, owner, fixture, probe, statement in (
        *_visual_motion_claims(),
        ("pixel_control_requires_resolved_displacement", "core/agency/controls_measured_on_one_thing.py",
         "tests/test_which_thing_answers_to_her.py", lambda: _resolved_motion_invariant() == (),
         "Pixel-derived control evidence requires observed displacement beyond the tracker resolution before assigning a new body identity."),
        ("entity_control_cannot_borrow_kind_evidence", "core/agency/controls_measured_on_one_thing.py",
         "tests/test_a_kind_cannot_lend_a_controls_identity.py", lambda: _own_control_witness_invariant() == (),
         "A kind's control prior cannot replace an entity's missing input trials or measured exclusions."),
        ("returning_paths_keep_their_entity_trials", "core/agency/controls_measured_on_one_thing.py",
         "tests/test_entity_press_trajectories.py", lambda: _trajectory_control_invariant() == (),
         "Returning trajectories require repeated contrasting input receipts on the same observed entity and experiment."),
        ("controls_are_ready_for_the_motion_model", "core/agency/controls_ready_for_the_motion_model.py",
         "tests/test_control_readiness_matches_the_motion_model.py", lambda: _model_readiness_invariant() == (),
         "A control remains unknown until the active motion model has measured its response."),
        ("control_trials_preserve_observation_time", "core/agency/keeping_a_trial_observable.py",
         "tests/test_a_control_trial_preserves_its_view.py", lambda: _observed_viability_invariant() == (),
         "Measured acceleration can interrupt an unknown control trial when waiting and braking would leave the observed field."),
        ("object_use_requires_positive_target_evidence", "core/agency/using_a_control_on_a_thing.py",
         "tests/test_a_control_is_used_on_what_it_is_for.py", lambda: _object_use_target_invariant() == (),
         "Unknown action text cannot authorize using a key on an unnamed object without a stated target or measured effect."),
        ("remembered_labels_keep_current_word_pixels", "core/perception/word_labels_seen_again.py",
         "tests/test_visible_word_labels.py", lambda: _visible_word_labels_invariant() == (),
         "A retained label requires a unique fresh match to its measured word pixels within the same view and bounded lifetime."),
        ("provisional_responses_do_not_establish_identity", "core/agency/a_response_worth_measuring_again.py",
         "tests/test_a_cold_control_loop_preserves_time_to_measure.py", lambda: _provisional_response_invariant() == (),
         "A fresh pulse and neutral interval can select a bounded independent repeat without granting body identity; missing or competing measurements abstain."),
        ("visual_avatar_guess_keeps_its_observed_track", "core/agency/a_body_seen_then.py",
         "tests/test_a_body_guess_keeps_its_observed_track.py", lambda: _body_seen_then_invariant() == (),
         "A visual body guess follows its captured observed track; absence or changed appearance retires it and causal identity takes precedence."),
        ("an_empty_reading_still_measures_the_initial_screen", "core/agency/words_read_with_their_picture.py",
         "tests/test_an_empty_reading_still_saw_the_world.py", lambda: _word_picture_start_invariant() == (),
         "A completed empty word reading establishes the initial picture; pending or unavailable readings remain unmeasured."),
        ("counter_trials_wait_for_measured_neutral_motion", "core/agency/countering_motion_as_it_appears.py",
         "tests/test_counter_trials_wait_for_measured_motion.py", lambda: _counter_trial_measurement_invariant() == (),
         "Initial counter trials wait for fresh settled unheld motion, preserve ordinary exploration and terminate within their finite input and observation budget."),
        ("joined_foreground_keeps_a_distinct_small_thing", "core/perception/a_small_thing_in_a_joined_patch.py",
         "tests/test_a_small_thing_survives_a_joined_foreground.py", lambda: _small_joined_thing_invariant() == (),
         "Joined foreground is separated before an area rejection; floor correction leaves visibly distinct minority pixels outside the backdrop."),
        ("body_trials_preserve_their_interaction_phase", "core/agency/body_trials_keep_the_world.py",
         "tests/test_body_trials_leave_interface_controls_available.py", lambda: _body_trial_phase_invariant() == (),
         "Known body trials exclude explicit resets and text-entry controls while preserving unknown inputs and interface availability."),
        ("kept_words_need_their_captured_pixels", "core/perception/words_kept_with_their_pixels.py",
         "tests/test_old_menu_words_keep_their_pixels.py", lambda: _kept_word_pixels_invariant() == (),
         "Retained word readings can be contradicted by measured appearance changes; absent pixels and changed view geometry stay unknown."),
        ("turns_require_settled_observations", "core/agency/which_way_she_faces.py",
         "tests/test_a_turn_is_not_learned_from_unseen_motion.py", lambda: _turn_observation_invariant() == (),
         "Stationary turning samples require settled observations; missing pictures cannot establish that a press moved nothing."),
        ("current_play_words_require_a_measured_frame", "core/skills/current_words_before_play.py",
         "tests/test_current_words_precede_play_interpretation.py", lambda: _current_play_words_invariant() == (),
         "Fast play handoff reads use current measured words; empty, old and future frames cannot remove a menu."),
        ("control_trials_keep_their_measurement_time", "core/agency/giving_a_trial_time.py",
         "tests/test_a_control_trial_gets_time_to_answer.py", lambda: _control_trial_time_invariant() == (),
         "A control trial retains its finite measurement interval while input and observation identity remain unchanged."),
        ("joined_parts_keep_measured_pixels", "core/perception/parts_when_things_touch.py",
         "tests/test_touching_things_keep_their_parts.py", lambda: _joined_parts_invariant() == (),
         "Touching foreground parts retain measured pixels and separate only where recent appearance and motion distinguish them."),
        ("known_body_controls_set_trial_priority", "core/agency/how_a_game_works.py",
         "tests/test_known_body_controls_set_trial_priority.py", lambda: _known_body_controls_invariant() == (),
         "Known body inputs set control trial priority before incidental interface input guesses; unknown roles preserve uncertainty."),
        ("foreground_masks_require_measured_support", "core/perception/foreground_appearance.py",
         "tests/test_transfer_identity_survives_background_changes.py", lambda: _foreground_mask_invariant() == (),
         "Appearance masks exclude only a measured flat exterior while retaining bounded foreground support and enclosed highlights."),
        ("creation_completion_requires_current_work", "core/skills/screen_step_evidence.py",
         "tests/test_creation_completion_needs_current_work.py", lambda: _creation_completion_invariant() == (),
         "Construction-based making completion requires independently reobserved placed work; navigation and historical activity cannot establish it."),
        ("visual_probe_misses_require_current_pixels", "core/perception/visual_probe_cache.py",
         "tests/test_visual_probe_misses_require_current_pixels.py", lambda: _visual_probe_miss_invariant() == (),
         "A failed visual probe suppresses repeated work only for matching measured pixels and interpretation inputs within its lifetime."),
        ("stalled_making_is_not_completion", "core/skills/sovereign_browser_drawing.py",
         "tests/test_making_timeout_does_not_make_an_artifact.py", lambda: _stalled_making_invariant() == (),
         "An unchanged making task's timeout ends its attempt without establishing a produced artifact."),
        ("receipts_keep_terminal_effect_flags", "core/runtime/action_executor.py",
         "tests/test_external_completion_keeps_its_terminal_outcome.py", lambda: _terminal_effect_flags_invariant() == (),
         "Post-action receipt flags preserve the durable transaction's effect and reconciliation outcome."),
        ("relations_require_unique_finite_entities", "core/perception/relations_between_things.py",
         "tests/test_spatial_relations_require_unique_current_entities.py", lambda: _relations_require_unique_finite_entities() == (),
         "Spatial completion evidence abstains when object names are ambiguous or their geometry is unmeasured."),
        ("picked_objectives_require_each_outcome", "core/skills/sovereign_browser_picking.py",
         "tests/test_picked_objectives_require_each_outcome.py", lambda: _picked_outcomes_invariant() == (),
         "A collection's completion requires each requested objective rather than each pursuit merely ending."),
        ("current_play_vetoes_predicted_completion", "core/skills/sovereign_browser_drawing.py",
         "tests/test_play_completion_requires_terminal_evidence.py", lambda: _current_play_veto_invariant() == (),
         "Current turn and ongoing-play evidence vetoes an inferred win or loss."),
        ("refined_obligations_require_their_bound_transfer", "core/cognition/action_refinement.py",
         "tests/test_source_choice_and_placement_share_one_owned_action.py", lambda: _bound_refinement_invariant() == (),
         "A combined choice and placement requires the exact source-use transfer bound before delivery."),
        ("unknown_selector_views_do_not_create_navigation_identity", "core/agency/where_things_lead.py",
         "tests/test_selector_views_keep_visual_navigation_states.py", lambda: _unknown_selector_invariant() == (),
         "Unmeasured selector contents cannot create a navigation state or an edge between states."),
        ("response_streams_keep_owned_capture_windows", "core/perception/watching_it_happen.py",
         "tests/test_action_response_windows.py", _response_window_keeps_its_evidence,
         "Response streams retain delivery intervals and capture timestamps, excluding prior owned changes from background."),
        ("responses_belong_to_one_delivered_dispatch", "core/skills/screen_step_evidence.py",
         "tests/test_visual_responses_keep_their_own_dispatch.py", lambda: _response_dispatch_invariant() == (),
         "Visual responses belong to exactly one delivered input and a fresh observation of the bound surface."),
        ("unmeasured_responses_are_not_graded", "core/agency/deliberate_action.py",
         "tests/test_visual_responses_keep_their_own_dispatch.py", lambda: _unmeasured_response_invariant() == (),
         "Unmeasured responses cannot resolve action episodes or teach success and failure consequences."),
        ("object_candidates_require_measured_extents", "core/perception/drawn_objects.py",
         "tests/test_drawn_objects_have_measured_extents.py", lambda: _object_candidates_invariant() == (),
         "Anonymous object candidates have current measured extents and appearance witnesses, without assumed affordances."),
        ("completed_guidance_cannot_publish", "core/cognition/guidance_lifetime.py",
         "tests/test_guidance_lifetime.py", lambda: _completed_guidance_invariant() == (),
         "Retired guidance operations reject delayed publication while preserving their parent execution owner."),
        ("remembered_labels_require_current_pixels", "core/perception/visual_anchor.py",
         "tests/test_visual_anchor.py", lambda: _remembered_labels_invariant() == (),
         "Remembered control labels require fresh current appearance in the same surface and capture context."),
        ("choices_require_current_observed_controls", "core/cognition/procedure_binding.py",
         "tests/test_procedure_bindings_follow_observed_controls.py", lambda: _choice_binding_invariant() == (),
         "Choice bindings require an observed selector effect and currently visible choices in its capture context."),
        ("construction_controls_require_current_work", "core/skills/screen_step_evidence.py",
         "tests/test_procedure_bindings_follow_observed_controls.py", lambda: _construction_readiness_invariant() == (),
         "Construction execution controls cannot use narrative or historical drag records as current readiness."),
        ("stop_listeners_have_scoped_lifetimes", "core/runtime/what_stops_it.py",
         "tests/test_stopped_execution_cannot_keep_acting.py", lambda: _stop_listener_lifetime_invariant() == (),
         "Cancellation listeners detach without stopping sibling work and retain the parent's stop direction."),
        ("procedures_require_owned_effects", "core/cognition/reading_the_rules.py",
         "tests/test_procedure_steps_need_their_own_evidence.py", _procedure_completion_is_measured,
         "Procedure completion requires the selected step's own observed effect and preserves unresolved prerequisites."),
        ("named_places_require_current_scene", "core/perception/where_the_words_point.py",
         "tests/test_named_places_keep_their_scene.py", lambda: _named_places_invariant() == (),
         "Named visual destinations belong to the current surface, episode, viewport and image revision."),
        ("transfers_require_measured_occupancy", "core/perception/observed_transfer.py",
         "tests/test_observed_transfer.py", lambda: _transfer_receipts_require_evidence() == (),
         "Transfer verification requires fresh measured pixels showing unique new occupancy at the bound destination."),
        ("child_pixels_keep_capture_custody", "core/perception/eyes_of_their_own.py",
         "tests/test_other_process_pixels_keep_their_capture.py", lambda: _child_capture_custody_invariant() == (),
         "Other-process screen readings preserve the exact final captured pixels and reject mismatched capture metadata."),
    ):
        suite.add_test(ValidationTest(
            name=name, description=statement, required_capability="",
            observation=Observation(name, True, fixture),
            predict=lambda _model, check=probe: check(),
            score=lambda value, observation, subject=name: boolean_score(value, expected=observation.value, subject=subject),
            owner=owner,
        ))
        suite.add_claim(Claim(
            statement=statement, test=name, owner=owner, asserted_in=owner,
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note="Offline causal and unknown-state regressions; ordinary live demo performance remains independently unproved.",
        ))

    name = "emissions_require_distinct_input_receipts"
    owner = "core/agency/which_one_answers_to_her.py"
    suite.add_test(ValidationTest(
        name=name, description="emissions require distinct, unambiguous delivered input trials",
        required_capability="", observation=Observation("distinct_input_effects", True,
                "tests/test_steering_and_triggering_use_separate_inputs.py"),
        predict=lambda _model: _emissions_require_distinct_input_receipts() == (),
        score=lambda value, observation, subject=name: boolean_score(value, expected=observation.value, subject=subject),
        owner=owner,
    ))
    suite.add_claim(Claim(
        statement="Emission learning accepts effects during input delivery and requires distinct unambiguous trials.",
        test=name, owner=owner, asserted_in=owner, evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note="Delivery windows, repeated births, ambiguity and failed fresh trials measured; live gameplay requires separate evidence.",
    ))

    name = "retained_structures_reject_damaged_closures"
    owner = "core/runtime/what_she_learned.py"
    suite.add_test(ValidationTest(
        name=name, description="damaged declared positional structures exclude their closure and preserve independent knowledge",
        required_capability="", observation=Observation("damaged_structures_excluded", True,
                "tests/test_retained_positional_structures_keep_their_shape.py"),
        predict=lambda _model: _retained_structure_invariant() == (),
        score=lambda value, observation, subject=name: boolean_score(value, expected=observation.value, subject=subject),
        owner=owner,
    ))
    suite.add_claim(Claim(
        statement="Declared retained positional structures are validated before reuse, excluding damaged closures while preserving independent knowledge.",
        test=name, owner=owner, asserted_in=owner, evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note="Fixed records, vectors, schema evolution and bounded roundtrips measured in independent synthetic domains; live use requires separate evidence.",
    ))

    name = "indexed_knowledge_keeps_relationships"
    owner = "core/runtime/what_she_learned.py"
    suite.add_test(ValidationTest(
        name=name, description="bounded retention preserves the subject and shape of referenced evidence",
        required_capability="", observation=Observation("indexed_relationships_preserved", True,
                "tests/test_indexed_knowledge_keeps_its_relationships.py"),
        predict=lambda _model: _indexed_knowledge_invariant() == (),
        score=lambda value, observation, subject=name: boolean_score(value, expected=observation.value, subject=subject),
        owner=owner,
    ))
    suite.add_claim(Claim(
        statement="Bounded indexed-state retention remaps retained references together and protects structural vectors.",
        test=name, owner=owner, asserted_in=owner, evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note="Compaction, roundtrip and invalid legacy references measured; broader schema coverage requires separate evidence.",
    ))

    name = "click_attribution_respects_stretch_boundaries"
    owner = "core/agency/where_clicks_pay.py"
    suite.add_test(ValidationTest(
        name=name, description="counter resets release pending click credit while retaining learned places",
        required_capability="", observation=Observation("click_credit_rebased", True,
                "tests/test_where_clicks_pay.py"),
        predict=lambda _model: _click_attribution_respects_stretch_boundaries(),
        score=lambda value, observation, subject=name: boolean_score(value, expected=observation.value, subject=subject),
        owner=owner,
    ))
    suite.add_claim(Claim(
        statement="A new stretch cannot credit a pending click with the preceding stretch's counter reset.",
        test=name, owner=owner, asserted_in=owner, evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note="Counter reset and retained learning measured; useful live control requires separate evidence.",
    ))

    name = "identity_requires_current_distinguishing_evidence"
    owner = "core/agency/causal_identification.py"
    suite.add_test(ValidationTest(
        name=name, description="stale experiments and equal responder evidence leave identity unknown",
        required_capability="", observation=Observation("causal_identity_unknown", True,
                "tests/test_causal_identification.py"),
        predict=lambda _model: _stale_and_ambiguous_identity_witnesses_remain_unknown(),
        score=lambda value, observation, subject=name: boolean_score(value, expected=observation.value, subject=subject),
        owner=owner,
    ))
    suite.add_claim(Claim(
        statement="Stale or indistinguishable causal witnesses cannot establish a control identity.",
        test=name, owner=owner, asserted_in=owner, evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note="Measured evidence epochs and ambiguity; broader semantic identification requires separate evaluation.",
    ))

    name = "interactive_and_receipt_workers_are_separate"
    owner = "core/runtime/executors.py"
    suite.add_test(ValidationTest(
        name=name, description="observation and durable receipt workers have separate pool ownership",
        required_capability="", observation=Observation("separate_workers", True,
                "tests/test_interactive_work_survives_shared_pool_saturation.py"),
        predict=lambda _model: _interactive_and_receipt_workers_are_separate(),
        score=lambda value, observation, subject=name: boolean_score(value, expected=observation.value, subject=subject),
        owner=owner,
    ))
    suite.add_claim(Claim(
        statement="Interactive perception and durable receipt work have separate workers from background work.",
        test=name, owner=owner, asserted_in=owner, evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note="Pool ownership and default-pool saturation measured; end-to-end latency needs separate live runs.",
    ))

    name = "current_counters_exclude_absent_history"
    owner = "core/agency/what_meeting_things_does.py"
    suite.add_test(ValidationTest(
        name=name, description="a counter absent from a new reading remains history without becoming current",
        required_capability="",
        observation=Observation("absent_counter_is_history_only", True,
                                "tests/test_a_new_attempt_measures_its_own_state.py"),
        predict=lambda _model: _absent_counters_are_not_current(),
        score=lambda value, observation, subject=name: boolean_score(value, expected=observation.value, subject=subject),
        owner=owner,
    ))
    suite.add_claim(Claim(
        statement="Current counter readings exclude labels absent from the latest observation.",
        test=name, owner=owner, asserted_in=owner, evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note="Measured counter history and presence; game wins and general environment coverage require separate runs.",
    ))
    name = "ambiguous_controls_remain_unknown"
    owner = "core/agency/which_one_answers_to_her.py"
    suite.add_test(ValidationTest(
        name=name, description="blocked and unsettled observations leave a key available for another trial",
        required_capability="",
        observation=Observation("ambiguous_control_is_unknown", True,
                                "tests/test_which_thing_answers_to_her.py"),
        predict=lambda _model: _ambiguous_controls_remain_unknown(),
        score=lambda value, observation, subject=name: boolean_score(value, expected=observation.value, subject=subject),
        owner=owner,
    ))
    suite.add_claim(Claim(
        statement="Blocked and unsettled motion cannot establish a key's control response.",
        test=name, owner=owner, asserted_in=owner, evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note="Measured ambiguous and free samples; actual task completion requires separate live evidence.",
    ))
    name = "fresh_control_experiment_releases_old_evidence"
    owner = "core/agency/which_one_answers_to_her.py"
    suite.add_test(ValidationTest(
        name=name, description="a fresh experiment discards previous rejections and response calibration",
        required_capability="", observation=Observation("control_evidence_released", True,
                "tests/test_exact_executor_connections.py"),
        predict=lambda _model: _fresh_control_experiment_forgets_old_rejections(),
        score=lambda value, observation, subject=name: boolean_score(value, expected=observation.value, subject=subject),
        owner=owner,
    ))
    suite.add_claim(Claim(
        statement="A fresh control experiment releases prior object rejections and response calibration.",
        test=name, owner=owner, asserted_in=owner, evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note="Evidence reset measured; actual reacquisition requires separate live runs.",
    ))
    name = "drawing_scene_requires_matching_frame"
    owner = "core/perception/the_drawing_as_objects.py"
    suite.add_test(ValidationTest(
        name=name, description="only complete, bounded scene descriptions of the captured frame are accepted",
        required_capability="", observation=Observation("matching_frame_required", True,
                "tests/test_the_drawing_can_describe_its_objects.py"),
        predict=lambda _model: _scene_requires_a_matching_complete_frame(),
        score=lambda value, observation, subject=name: boolean_score(value, expected=observation.value, subject=subject),
        owner=owner,
    ))
    suite.add_claim(Claim(
        statement="A structured drawing observation requires a complete description matching its captured pixels' dimensions.",
        test=name, owner=owner, asserted_in=owner, evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note="Frame and metadata acceptance measured; supported paint and actual control are measured in separate browser runs.",
    ))
    from core.intent.capability_selection import _artifact_lane_is_owned_by_the_declaration

    name = "artifact_lane_follows_declared_effect"
    owner = "core/intent/capability_selection.py"
    suite.add_test(ValidationTest(
        name=name, description="an artifact effect takes precedence over a generic desktop shortcut",
        required_capability="", observation=Observation("declared_artifact_owner", True,
                "tests/test_artifact_work_precedes_a_desktop_shortcut.py"),
        predict=lambda _model: _artifact_lane_is_owned_by_the_declaration(),
        score=lambda value, observation, subject=name: boolean_score(value, expected=observation.value, subject=subject),
        owner=owner,
    ))
    suite.add_claim(Claim(
        statement="A strongest artifact declaration defers generic desktop shortcuts to capability selection.",
        test=name, owner=owner, asserted_in=owner, evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note="Lane ownership measured; autonomous repair and use require separate live evidence.",
    ))


def _visual_motion_claims() -> tuple:
    """Measured foreground and point continuity claims."""
    from core.agency.arriving_by_her_pushes import _arrival_braking_invariant
    from core.agency.finding_her_another_way import _current_body_motion_invariant
    from core.agency.inputs_while_measuring_control import _single_input_interval_invariant
    from core.agency.time_from_picture_to_control import _picture_age_invariant
    from core.agency.what_a_press_does import _settled_trial_invariant, _unchanged_input_invariant
    from core.agency.what_follows_the_pointer import _pointer_body_continuity_invariant
    from core.agency.which_one_answers_to_her import _motion_segments_invariant
    from core.perception.a_reading_before_more_detail import _finished_reading_invariant
    from core.perception.activity_in_the_picture import _picture_activity_invariant
    from core.perception.boundary_support_for_translation import _visible_boundary_invariant
    from core.perception.frames_as_they_are_drawn import _frame_time_invariant
    from core.perception.points_that_keep_their_motion import _point_identity_invariant
    from core.perception.position_from_the_same_points import _point_path_invariant
    from core.perception.reading_again_after_loss import _reading_renewal_invariant
    from core.perception.the_whole_patch_around_a_visible_corner import _whole_patch_invariant
    from core.perception.translation_from_visible_support import _translation_support_invariant
    from core.perception.what_moves_in_the_picture import _missing_points_invariant
    from core.perception.what_the_points_on_a_thing_say import _point_meaning_invariant

    return (
        ("repeated_stream_frames_keep_their_original_time", "core/perception/frames_as_they_are_drawn.py",
         "tests/test_a_frame_keeps_its_capture_time.py", lambda: _frame_time_invariant() == (),
         "A consumed stream frame cannot become a fresh observation through silence; canvas and screenshot requests retain their start time through slow completion."),
        ("control_prediction_uses_the_current_picture_age", "core/agency/time_from_picture_to_control.py",
         "tests/test_control_uses_this_pictures_age.py", lambda: _picture_age_invariant() == (),
         "Prediction refreshes the current picture age before steering and keeps measured delivery and post-acknowledgement response separate."),
        ("finished_readings_survive_only_with_their_current_capture", "core/perception/a_reading_before_more_detail.py",
         "tests/test_a_finished_reading_survives_slow_detail.py", lambda: _finished_reading_invariant() == (),
         "Optional inspection timeouts can retain completed scoped readings only after a new capture agrees; incomplete, changed or refused captures remain unavailable."),
        ("arrivals_need_measured_braking_room", "core/agency/arriving_by_her_pushes.py",
         "tests/test_a_carried_body_arrives_before_it_stops.py", lambda: _arrival_braking_invariant() == (),
         "Carried arrival plans need measured braking pushes and finite goals; terminal closing speed is weighed against room before first contact."),
        ("point_paths_keep_their_original_visible_marks", "core/perception/position_from_the_same_points.py",
         "tests/test_a_point_path_keeps_its_original_marks.py", lambda: _point_path_invariant() == (),
         "A path measured from points retains original named marks, current owned pixels and coherent displacement; lost or replacement marks end the path."),
        ("secondary_inputs_end_independent_body_observations", "core/agency/inputs_while_measuring_control.py",
         "tests/test_inputs_while_measuring_control.py", lambda: _single_input_interval_invariant() == (),
         "Another delivered action ends the isolated control interval; measured prefixes remain intact and later mixed motion cannot extend them."),
        ("pointer_control_needs_one_continuing_body", "core/agency/what_follows_the_pointer.py",
         "tests/test_pointer_control_belongs_to_one_body.py", lambda: _pointer_body_continuity_invariant() == (),
         "Pointer-control sightings keep the same entity and birth; separate objects, capture gaps and unmeasured movement cannot share a trail."),
        ("visible_activity_does_not_need_certified_body_velocity", "core/perception/activity_in_the_picture.py",
         "tests/test_visible_activity_is_not_body_velocity.py", lambda: _picture_activity_invariant() == (),
         "Visible activity prevents a still-screen handoff without supplying body translation or control authority; capture gaps cannot establish continuous quiet."),
        ("old_control_trials_need_current_body_motion", "core/agency/which_one_answers_to_her.py",
         "tests/test_entity_press_trajectories.py", lambda: _current_body_motion_invariant() == (),
         "Historical control receipts cannot assign a new body identity while its current pixel movement is unavailable, contradicted or stale."),
        ("control_trials_start_on_delivered_input", "core/agency/playing_as_it_happens.py",
         "tests/test_control_trials_start_when_input_is_delivered.py", lambda: _unchanged_input_invariant() == (),
         "Choosing an unchanged held input preserves the physical intervention start and cannot manufacture an independent control trial."),
        ("settled_behavior_keeps_the_independent_observation", "core/agency/what_a_press_does.py",
         "tests/test_translation_needs_visible_support.py", lambda: _settled_trial_invariant() == (),
         "Settled behavior retains measured trial coverage until the delivered input interval ends; lack of response is observed rather than inferred."),
        ("plain_patch_motion_requires_visible_exterior_boundaries", "core/perception/boundary_support_for_translation.py",
         "tests/test_visible_boundary_support.py", lambda: _visible_boundary_invariant() == (),
         "Plain-patch translation requires observed corresponding exterior boundaries; an interior window on a uniform surface remains unmeasured."),
        ("control_motion_requires_agreeing_current_support", "core/perception/translation_from_visible_support.py",
         "tests/test_translation_needs_visible_support.py", lambda: _translation_support_invariant() == (),
         "Detected box movement requires agreeing current visible support before it teaches control responses or physics."),
        ("control_measurements_resume_on_new_observed_segments", "core/agency/which_one_answers_to_her.py",
         "tests/test_translation_needs_visible_support.py", lambda: _motion_segments_invariant() == (),
         "Missing motion starts a fresh measured segment of the held input; contradictory motion retires that trial."),
        ("lost_readings_are_refreshed_without_duplicate_requests", "core/perception/reading_again_after_loss.py",
         "tests/test_a_body_guess_keeps_its_observed_track.py", lambda: _reading_renewal_invariant() == (),
         "An unresolved reading can be refreshed after measured loss; unread frames, active requests and current ownership do not authorize another request."),
        ("missing_points_do_not_relax_object_identity", "core/perception/what_moves_in_the_picture.py",
         "tests/test_points_keep_their_observed_motion.py", lambda: _missing_points_invariant() == (),
         "Missing point support cannot relax measured shape and appearance continuity or lend an identity through a shared kind."),
        ("point_meaning_requires_current_owned_support", "core/perception/what_the_points_on_a_thing_say.py",
         "tests/test_the_points_on_a_thing_say_how_soon_it_arrives.py", lambda: _point_meaning_invariant() == (),
         "Point-derived approach and identity require current owned pixels; absent support and ambiguous or expired owners remain unknown."),
        ("visible_corner_keeps_its_whole_connected_patch", "core/perception/the_whole_patch_around_a_visible_corner.py",
         "tests/test_a_visible_corner_keeps_the_whole_patch.py", lambda: _whole_patch_invariant() == (),
         "A foreground corner can recover its bounded connected colour patch when a measured surrounding floor distinguishes it."),
        ("point_identity_needs_current_pixels", "core/perception/points_that_keep_their_motion.py",
         "tests/test_points_keep_their_observed_motion.py", lambda: _point_identity_invariant() == (),
         "Bounded point tracks retain measured identities across consecutive frames and retire them on occlusion, gaps or resets."),
    )
