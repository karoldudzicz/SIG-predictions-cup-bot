"""Watchdog feeding and outage handling in main.py's loop. Fakes only - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import asyncio
import logging

import aiohttp
import pytest

import main
from exchange_api import ApiError, AuthError, RequestTimeoutError
from oracle_api import OracleTimeoutError
from test_cycle_resilience import SETTINGS
from test_startup import StartupExchange, _AsyncCM, harness  # noqa: F401 - pytest fixture


class _ScriptedStrategy:
    """run_cycle raises the next exception from `script` (None = success), counting calls."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    async def run_cycle(self):
        self.calls += 1
        outcome = self.script.pop(0) if self.script else None
        if outcome is not None:
            raise outcome
        return _Report()


class _Report:
    def __init__(self):
        self.__dict__.update({"ok": True})


@pytest.fixture
def fast(monkeypatch, tmp_path):
    notified: list[str] = []
    monkeypatch.setattr(main, "sd_notify", notified.append)
    monkeypatch.setattr(main, "failure_backoff", lambda n: 0.0)
    monkeypatch.setattr(main, "write_heartbeat", lambda *a, **k: None)
    return notified


def _timeout():
    return RequestTimeoutError("GET", "/portfolio/positions", 4, 17.0)


def test_watchdog_is_fed_after_a_failed_cycle(fast):
    notified = fast
    code = asyncio.run(main.Bot(SETTINGS, max_cycles=1)._loop(_ScriptedStrategy([_timeout()])))
    assert code == main.EXIT_OK
    assert "WATCHDOG=1" in notified


def test_watchdog_pings_during_a_long_backoff_sleep(fast, monkeypatch):
    notified = fast
    monkeypatch.setattr(main, "WATCHDOG_PING_S", 0.05)
    asyncio.run(main.Bot(SETTINGS)._idle(0.3))
    assert notified.count("WATCHDOG=1") >= 5  # ~every 0.05 s for 0.3 s


@pytest.mark.parametrize(
    "error",
    [
        _timeout(),
        OracleTimeoutError("kalshi", "GET", "/markets", 5, 30.0),
        ApiError(503, "SERVICE_UNAVAILABLE", "down"),
        ApiError(429, "RATE_LIMITED", "slow down"),
        aiohttp.ClientConnectionError("reset"),
    ],
    ids=["request-timeout", "oracle-timeout", "503", "429", "connection"],
)
def test_twelve_api_outage_cycles_keep_the_process_alive(fast, error):
    strategy = _ScriptedStrategy([error] * 12)
    code = asyncio.run(main.Bot(SETTINGS, max_cycles=12)._loop(strategy))
    assert code == main.EXIT_OK  # max_cycles reached, never EXIT_TOO_MANY_FAILURES
    assert strategy.calls == 12


def test_non_api_failures_still_escalate(fast):
    strategy = _ScriptedStrategy([RuntimeError("bug")] * 20)
    code = asyncio.run(main.Bot(SETTINGS, max_cycles=20)._loop(strategy))
    assert code == main.EXIT_TOO_MANY_FAILURES
    assert strategy.calls == main.MAX_CONSECUTIVE_FAILURES


def test_non_retryable_api_error_still_escalates(fast):
    strategy = _ScriptedStrategy([ApiError(400, "VALIDATION_ERROR", "bad")] * 20)
    assert asyncio.run(main.Bot(SETTINGS, max_cycles=20)._loop(strategy)) == main.EXIT_TOO_MANY_FAILURES


def test_auth_error_in_the_loop_still_exits_2(harness, monkeypatch):  # noqa: F811
    notified, install = harness
    install(StartupExchange())
    monkeypatch.setattr(main, "Strategy", lambda *a, **k: _ScriptedStrategy([AuthError(401, "API_KEY_REVOKED", "revoked")]))
    assert asyncio.run(main.Bot(SETTINGS).run()) == main.EXIT_NEEDS_HUMAN


def test_outage_alarm_fires_once_per_interval(caplog):
    alarm = main.OutageAlarm(after_s=900, repeat_s=300, now=0.0)
    with caplog.at_level(logging.ERROR, logger="main"):
        fired_at = [t for t in range(0, 2101, 60) if alarm.maybe_alert(float(t), "RequestTimeoutError(...)", 1)]
    assert fired_at == [900, 1200, 1500, 1800, 2100]  # first at 15 min, then every 5 min
    record = next(r for r in caplog.records if r.getMessage() == "No successful cycle")
    assert (record.minutes_without_success, record.last_error) == (15.0, "RequestTimeoutError(...)")

    alarm.record_success(2150.0)  # a success re-arms the full 15-minute wait
    assert not alarm.maybe_alert(2200.0, "x", 1)
    assert alarm.maybe_alert(2150.0 + 900, "x", 1)
