"""core/organism/cycle_clock.py
Cycle clock and diurnal synchronization engine for Aura.
Tracks day/night transitions, sleep pressure, and tick frequency scaling.
"""
import time
from datetime import datetime


class CycleClock:
    """Manages cycle ticking, time perception, and diurnal sleep pressure calculations."""

    def __init__(self, tick_rate_hz: float = 1.0):
        self.tick_rate_hz = tick_rate_hz
        self.tick_interval_s = 1.0 / tick_rate_hz
        self.start_time = time.time()
        self.total_ticks = 0

    def get_time_of_day(self) -> float:
        """Returns fractional hour of current day (0.0 to 24.0)."""
        now = datetime.now()
        return now.hour + now.minute / 60.0 + now.second / 3600.0

    def is_night(self) -> bool:
        """Determine if it is night time (typically 22:00 to 06:00)."""
        hour = self.get_time_of_day()
        return hour >= 22.0 or hour <= 6.0

    def calculate_sleep_pressure(self, sleep_debt: float, hours_awake: float) -> float:
        """Calculate homeostatic sleep pressure from 0.0 to 1.0."""
        # Standard homeostatic model combining wake hours and cumulative debt
        pressure = (hours_awake / 16.0) * 0.7 + (sleep_debt / 24.0) * 0.3
        return min(max(pressure, 0.0), 1.0)

    def should_sleep(self, sleep_pressure: float) -> bool:
        """Heuristics to determine if organism should initiate consolidation sleep."""
        # Sleep pressure > 0.8 is a strong signal, or night-time combined with moderate pressure
        if sleep_pressure > 0.85:
            return True
        if self.is_night() and sleep_pressure > 0.6:
            return True
        return False

    def tick_sleep(self, dt: float) -> float:
        """Simulate time delta. Returns dynamic interval adjust based on cycle status."""
        self.total_ticks += 1
        # Dynamic rate scaling: slow down ticks at night or under high load to save power
        if self.is_night():
            return self.tick_interval_s * 2.0  # Conserve cycles
        return self.tick_interval_s
