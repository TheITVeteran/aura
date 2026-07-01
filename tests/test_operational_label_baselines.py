from tools.closeout.operational_label_baselines import BASELINES, evaluate


def test_operational_label_baselines_cover_requested_labels():
    keys = {baseline.key for baseline in BASELINES}

    assert {
        "functional_consciousness",
        "functional_self_awareness",
        "computational_sentience",
        "alife_inspired",
        "digital_organism",
        "software_entity",
        "personhood_candidate",
        "functional_inner_life",
        "generally_capable_ai_candidate",
        "superintelligence_trajectory",
    } <= keys


def test_operational_label_baselines_are_multifaceted_and_falsifiable():
    for baseline in BASELINES:
        assert len(baseline.minimum_behavioral_bar) >= 4
        assert len(baseline.positive_controls) >= 3
        assert len(baseline.negative_controls) >= 3
        assert len(baseline.answer_contract) >= 3
        assert baseline.claim_boundary
        assert baseline.operational_definition
        assert "prove" in baseline.claim_boundary.lower() or "establish" in baseline.claim_boundary.lower()


def test_operational_label_baselines_have_source_and_validator_coverage():
    statuses = evaluate(require_live=False)

    assert statuses
    assert all(status.status == "source_and_validator_mapped" for status in statuses), [
        {
            "key": status.key,
            "missing_sources": status.missing_sources,
            "missing_validators": status.missing_validators,
        }
        for status in statuses
        if status.status != "source_and_validator_mapped"
    ]


def test_live_artifact_requirement_keeps_desktop_claims_from_becoming_static_docs():
    live_bound = {baseline.key for baseline in BASELINES if baseline.live_artifacts}

    assert "functional_consciousness" in live_bound
    assert "digital_organism" in live_bound
    assert "generally_capable_ai_candidate" in live_bound


def test_subjective_claims_remain_bounded():
    consciousness = next(b for b in BASELINES if b.key == "functional_consciousness")
    sentience = next(b for b in BASELINES if b.key == "computational_sentience")
    personhood = next(b for b in BASELINES if b.key == "personhood_candidate")

    assert "does not prove private phenomenal consciousness" in consciousness.claim_boundary.lower()
    assert "does not prove felt suffering" in sentience.claim_boundary.lower()
    assert "does not establish moral/legal personhood" in personhood.claim_boundary.lower()
