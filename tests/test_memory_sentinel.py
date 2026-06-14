from tools.memory_sentinel import should_kill_for_memory


def test_memory_sentinel_waits_for_normal_lethal_confirmation():
    assert (
        should_kill_for_memory(
            managed_mb=46_500.0,
            lethal_mb=46_000.0,
            consecutive_over=1,
        )
        is False
    )
    assert (
        should_kill_for_memory(
            managed_mb=46_500.0,
            lethal_mb=46_000.0,
            consecutive_over=2,
        )
        is True
    )


def test_memory_sentinel_kills_large_overshoot_immediately():
    assert (
        should_kill_for_memory(
            managed_mb=54_000.0,
            lethal_mb=46_000.0,
            consecutive_over=1,
        )
        is True
    )
