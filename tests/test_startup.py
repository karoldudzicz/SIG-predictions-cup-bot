"""Startup reconciliation vs. a slow SIG: retry essentials within a budget, never crash on
the non-essential account value, stop promptly on SIGTERM. Fakes only - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import asyncio
import logging
import time

import aiohttp
import pytest

import main
from exchange_api import ApiError, AuthError, RequestTimeoutError
from strategy import OrderJournal
from test_cycle_resilience import MARKET_MAP, SETTINGS


class StartupExchange:
    """positions: behaviours consumed per call - "timeout", "hang", "502", "conn", "auth" or "ok" (default)."""

    dry_run = True
    halted = False

    def __init__(self, positions=(), pnl="ok"):
        self.positions_plan = list(positions)
        self.pnl = pnl
        self.positions_calls = 0
        self.cancel_all_calls = 0

    def halt(self, reason):
        self.halted = True

    def get_stats(self):
        return {"get_retries": 0, "slowest_get_s": 0.0}

    async def cancel_all_orders(self, **kwargs):
        self.cancel_all_calls += 1
        return None

    async def get_positions(self):
        self.positions_calls += 1
        behaviour = self.positions_plan.pop(0) if self.positions_plan else "ok"
        if behaviour == "timeout":
            raise RequestTimeoutError("GET", "/portfolio/positions", 4, 17.0)
        if behaviour == "hang":
            await asyncio.sleep(60)
        if behaviour == "502":
            raise ApiError(502, "ORDER_STATUS_UNKNOWN", "bad gateway")
        if behaviour == "conn":
            raise aiohttp.ClientConnectionError("connection reset")
        if behaviour == "auth":
            raise AuthError(401, "INVALID_API_KEY", "Invalid API key.")
        return []

    async def get_pnl(self):
        if self.pnl == "timeout":
            raise RequestTimeoutError("GET", "/portfolio/pnl", 4, 17.0)
        return {"totalAccountValue": 100_000.0}


class _AsyncCM:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc_info):
        return False


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Patch main.py's collaborators; returns (sd_notify messages, install(exchange))."""
    notified: list[str] = []
    monkeypatch.setattr(main, "sd_notify", notified.append)
    monkeypatch.setattr(main, "failure_backoff", lambda attempt: 0.05)  # keep the tests fast
    monkeypatch.setattr(main, "PolymarketClient", lambda: _AsyncCM(object()))
    monkeypatch.setattr(main, "KalshiClient", lambda: _AsyncCM(object()))
    monkeypatch.setattr(main, "OrderJournal", lambda: OrderJournal(tmp_path / "j.jsonl"))
    monkeypatch.setattr(main.MarketMap, "load", classmethod(lambda cls: MARKET_MAP))

    def install(exchange):
        monkeypatch.setattr(main, "ExchangeClient", lambda settings: _AsyncCM(exchange))

    return notified, install


async def _no_loop(self, strategy):
    return main.EXIT_OK  # stop right after a successful startup


def test_pnl_timeout_at_startup_continues(harness, monkeypatch, caplog):
    notified, install = harness
    install(StartupExchange(pnl="timeout"))
    monkeypatch.setattr(main.Bot, "_loop", _no_loop)
    with caplog.at_level(logging.INFO, logger="main"):
        assert asyncio.run(main.Bot(SETTINGS).run()) == main.EXIT_OK
    assert "READY=1" in notified
    assert any(r.getMessage() == "Startup account value unavailable, continuing" for r in caplog.records)
    done = next(r for r in caplog.records if r.getMessage() == "Startup reconciliation done")
    assert done.account_value is None


def test_positions_timing_out_twice_then_succeeding_continues(harness, monkeypatch, caplog):
    notified, install = harness
    exchange = StartupExchange(positions=["timeout", "timeout"])
    install(exchange)
    monkeypatch.setattr(main.Bot, "_loop", _no_loop)
    with caplog.at_level(logging.WARNING, logger="main"):
        assert asyncio.run(main.Bot(SETTINGS).run()) == main.EXIT_OK
    assert exchange.positions_calls == 3
    retries = [r for r in caplog.records if r.getMessage() == "Startup request failed, retrying"]
    assert [(r.path, r.attempt) for r in retries] == [("/portfolio/positions", 1), ("/portfolio/positions", 2)]
    assert "READY=1" in notified
    assert notified[0].startswith("EXTEND_TIMEOUT_USEC=")  # systemd start timeout extended first


@pytest.mark.parametrize("positions", [["timeout"] * 100, ["hang"]], ids=["during-backoff", "during-request"])
def test_sigterm_during_startup_wait_exits_promptly(harness, monkeypatch, positions):
    notified, install = harness
    install(StartupExchange(positions=positions))
    monkeypatch.setattr(main, "failure_backoff", lambda attempt: 30.0)  # long wait to be interrupted

    async def scenario():
        bot = main.Bot(SETTINGS)
        asyncio.get_running_loop().call_later(0.2, bot.request_stop, "SIGTERM")
        return await bot.run()

    started = time.monotonic()
    assert asyncio.run(scenario()) == main.EXIT_OK
    assert time.monotonic() - started < 2.0
    assert "READY=1" not in notified


def test_startup_budget_exhausted_exits_non_zero(harness, monkeypatch, caplog):
    notified, install = harness
    exchange = StartupExchange(positions=["timeout"] * 1000)
    install(exchange)
    monkeypatch.setattr(main, "STARTUP_RETRY_BUDGET_S", 0.3)
    with caplog.at_level(logging.CRITICAL, logger="main"):
        code = asyncio.run(main.Bot(SETTINGS).run())
    assert code == main.EXIT_STARTUP_FAILED != 0
    assert exchange.positions_calls > 2  # it kept retrying until the budget was gone
    assert "READY=1" not in notified
    assert any(r.getMessage() == "Startup retry budget exhausted, exiting" for r in caplog.records)


def test_startup_survives_502_and_connection_error(harness, monkeypatch, caplog):
    notified, install = harness
    exchange = StartupExchange(positions=["502", "conn"])
    install(exchange)
    monkeypatch.setattr(main.Bot, "_loop", _no_loop)
    with caplog.at_level(logging.WARNING, logger="main"):
        assert asyncio.run(main.Bot(SETTINGS).run()) == main.EXIT_OK
    assert exchange.positions_calls == 3
    retries = [r for r in caplog.records if r.getMessage() == "Startup request failed, retrying"]
    assert [r.attempt for r in retries] == [1, 2]
    assert "502" in retries[0].error and "ClientConnectionError" in retries[1].error
    assert "READY=1" in notified


def test_auth_error_at_startup_exits_2_immediately(harness):
    notified, install = harness
    exchange = StartupExchange(positions=["auth"])
    install(exchange)
    assert asyncio.run(main.Bot(SETTINGS).run()) == main.EXIT_NEEDS_HUMAN
    assert exchange.positions_calls == 1  # no retry
    assert "READY=1" not in notified
