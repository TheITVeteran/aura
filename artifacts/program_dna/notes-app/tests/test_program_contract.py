from src.program import ReconstructedProgram


def test_reconstructed_program_exposes_inferred_capabilities():
    program = ReconstructedProgram()
    assert program.capabilities()


def test_reconstructed_program_rejects_unknown_feature():
    program = ReconstructedProgram()
    try:
        program.execute('not_inferred')
    except ValueError:
        pass
    else:
        raise AssertionError('unknown features must fail closed')


def test_reconstructed_program_explains_evidence_trace():
    program = ReconstructedProgram()
    for capability in program.capabilities():
        receipt = program.execute(capability, {'probe': True})
        assert receipt['evidence_trace']
