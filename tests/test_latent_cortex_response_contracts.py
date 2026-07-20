from __future__ import annotations

import pytest

from core.brain.llm.latent_cortex.response_contracts import (
    ResponseContractError,
    parse_response_contract,
    validate_response_payload,
)


@pytest.mark.parametrize(
    ("contract", "payload"),
    [
        ('{"sequence":list[int],"checksum":int}', {"sequence": [2, 3], "checksum": 5}),
        (
            '{"choice":"H"|"not_H","posterior":"p/q","confidence_band":str}',
            {"choice": "H", "posterior": "9/11", "confidence_band": "high"},
        ),
        (
            '{"returns":list[{"state":list[[str,int]],"pressure":list[int]}],"time_complexity":"O(n^2)"}',
            {"returns": [{"state": [["a", 1]], "pressure": [1]}], "time_complexity": "O(n^2)"},
        ),
    ],
)
def test_current_contract_shapes_parse_and_validate(contract: str, payload: object) -> None:
    parsed = parse_response_contract(contract)

    assert validate_response_payload(payload, parsed) == {
        "valid": True,
        "errors": [],
        "error_count": 0,
    }


@pytest.mark.parametrize(
    ("contract", "payload", "error"),
    [
        ('{"count":int}', {"count": True}, "$.count:expected_int"),
        ('{"count":int}', {"count": 1, "extra": 2}, "$.extra:unexpected"),
        ('{"pair":[str,int]}', {"pair": ["x"]}, "$.pair:tuple_shape_mismatch"),
        ('{"choice":"H"|"not_H"}', {"choice": "maybe"}, "$.choice:literal_mismatch"),
    ],
)
def test_contract_validation_rejects_wrong_shape(
    contract: str, payload: object, error: str
) -> None:
    result = validate_response_payload(payload, contract)

    assert result["valid"] is False
    assert error in result["errors"]


@pytest.mark.parametrize(
    "contract",
    [
        '{"count":int,"count":int}',
        '{"count":float}',
        '{"count":int} trailing',
        "{count:int}",
    ],
)
def test_contract_parser_fails_closed(contract: str) -> None:
    with pytest.raises(ResponseContractError):
        parse_response_contract(contract)
