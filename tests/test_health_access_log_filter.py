"""Health-poll requests must not flood the uvicorn access log.

One live launch log carried 30,298 identical 'GET /api/health/boot 503'
lines — a third of the entire file — burying real request traffic.
"""
from __future__ import annotations

import logging

import aura_main


def _access_record(path: str, status: int = 200) -> logging.LogRecord:
    # uvicorn access records: msg='%s - "%s %s HTTP/%s" %d',
    # args=(client, method, path, http_version, status)
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:1234", "GET", path, "1.1", status),
        exc_info=None,
    )


class TestHealthPollAccessLogFilter:
    def setup_method(self):
        self.filter = aura_main._HealthPollAccessLogFilter()

    def test_health_boot_polls_are_dropped(self):
        assert self.filter.filter(_access_record("/api/health/boot", 503)) is False
        assert self.filter.filter(_access_record("/api/health", 200)) is False
        assert self.filter.filter(_access_record("/metrics", 200)) is False

    def test_real_traffic_passes(self):
        assert self.filter.filter(_access_record("/api/chat", 200)) is True
        assert self.filter.filter(_access_record("/", 200)) is True
        assert self.filter.filter(_access_record("/static/aura.js", 200)) is True
        # Similar-but-different prefixes must not be swallowed.
        assert self.filter.filter(_access_record("/api/healthcheck-report", 200)) is True

    def test_malformed_records_pass_through(self):
        record = logging.LogRecord(
            name="uvicorn.access", level=logging.INFO, pathname=__file__,
            lineno=1, msg="plain message", args=(), exc_info=None,
        )
        assert self.filter.filter(record) is True

    def test_install_is_idempotent(self):
        access_logger = logging.getLogger("uvicorn.access")
        before = list(access_logger.filters)
        try:
            aura_main._install_health_access_log_filter()
            aura_main._install_health_access_log_filter()
            ours = [
                f for f in access_logger.filters
                if isinstance(f, aura_main._HealthPollAccessLogFilter)
            ]
            assert len(ours) == 1
        finally:
            access_logger.filters = before
