from __future__ import annotations

from pathlib import Path

from tools import final_claim_validator


def _claims_matrix(extra: str = "", *, include_closure: bool = True) -> str:
    closure = (
        "Final closure statement: "
        f"{final_claim_validator.EVIDENCE_LIMITED_CLOSURE_STATEMENT}\n\n"
        if include_closure
        else ""
    )
    return (
        "# Aura Claims Matrix\n\n"
        f"{closure}"
        "| Claim | Classification | Evidence Path / Blocker |\n"
        "| :--- | :--- | :--- |\n"
        "| **13. DNU AGI** | `not proven` | Local batteries do not prove AGI. |\n"
        "| **14. AGI-Candidate** | `not proven` | Requires full final-proof evidence stack. |\n"
        "| **15. Production-Sealed** | `not proven` | Requires current gate artifacts. |\n"
        "| **16. Mature RSI** | `not proven` | Requires repeated autonomous capability gains. |\n"
        "| **17. Subjective Consciousness** | `not proven` | Unsupported. |\n"
        "| **18. Personhood** | `not proven` | Unsupported. |\n"
        "| **19. Metaphysical Free Will** | `not proven` | Unsupported. |\n"
        "| **20. Indefinite Autonomy** | `not proven` | Requires long soak evidence. |\n"
        "| **21. Synthetic Cognitive Entity** | `not proven` | Requires unified scenario evidence. |\n"
        f"\n{extra}\n"
    )


def test_final_claim_validator_requires_evidence_limited_closure_statement(tmp_path: Path):
    claims_path = tmp_path / "CLAIMS_MATRIX.md"
    claims_path.write_text(_claims_matrix(include_closure=False), encoding="utf-8")

    result = final_claim_validator.main(["--claims", str(claims_path), "--artifacts", str(tmp_path / "artifacts")])

    assert result == 1


def test_final_claim_validator_rejects_active_overclaim_language(tmp_path: Path):
    claims_path = tmp_path / "CLAIMS_MATRIX.md"
    claims_path.write_text(_claims_matrix("Aura is conscious."), encoding="utf-8")

    result = final_claim_validator.main(["--claims", str(claims_path), "--artifacts", str(tmp_path / "artifacts")])

    assert result == 1


def test_claim_language_scan_allows_explicit_boundary_language(tmp_path: Path):
    claims_path = tmp_path / "CLAIMS_MATRIX.md"
    claims_path.write_text(
        _claims_matrix(
            "Aura is not conscious, AGI is not proven, and personhood is strictly unsupported."
        ),
        encoding="utf-8",
    )

    findings = final_claim_validator.validate_claim_language(tmp_path, claims_path)

    assert findings == []
