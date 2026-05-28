import asyncio


class HangingBaselineRouter:
    def __init__(self) -> None:
        self.calls = []
        self.force_abort_count = 0

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(10.0)
        return "late baseline response"

    def force_abort_active_generation(self, *, reason: str) -> bool:
        self.force_abort_count += 1
        return True


def test_dnu_baseline_timeout_preserves_shared_model_lane(monkeypatch):
    from tools.agi import run_dnu_agi_proof_battery as dnu

    router = HangingBaselineRouter()
    monkeypatch.setattr(dnu, "_baseline_timeout_seconds", lambda: 0.01)

    async def run() -> None:
        try:
            await dnu._generate_baseline_response(
                router,
                prompt="answer inside tags",
                system_prompt="baseline",
                purpose="raw_llm_baseline",
            )
        except TimeoutError:
            return
        raise AssertionError("baseline timeout was not surfaced")

    asyncio.run(run())

    assert router.force_abort_count == 0
    assert router.calls[0]["origin"] == "baseline"
    assert router.calls[0]["proof_primary_lane_required"] is True


def test_agency_baseline_timeout_preserves_shared_model_lane(monkeypatch):
    from tools.agency import run_agency_emergence_battery as agency

    router = HangingBaselineRouter()
    monkeypatch.setattr(agency, "_agency_baseline_timeout_seconds", lambda: 0.01)

    async def run() -> None:
        try:
            await agency._generate_agency_baseline_response(
                router,
                prompt="one sentence",
                system_prompt="baseline",
                purpose="agency_raw_llm_baseline",
            )
        except TimeoutError:
            return
        raise AssertionError("baseline timeout was not surfaced")

    asyncio.run(run())

    assert router.force_abort_count == 0
    assert router.calls[0]["origin"] == "baseline"
    assert router.calls[0]["proof_primary_lane_required"] is True
