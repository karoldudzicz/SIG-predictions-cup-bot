"""Cycle behaviour when individual reads fail, and how main.py reports failures.

Fake exchange / oracle clients - no network. Run:  python -m pytest tests -v
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import pytest

import main
from config import Settings
from exchange_api import ApiError, RequestTimeoutError
from market_mapper import MarketMap, MarketMapping, Race
from oracle_api import OracleQuote
from strategy import OrderJournal, Strategy

SETTINGS = Settings(
    sig_api_key="test-key",
    sig_base_url="https://sig.invalid/api/v1",
    dry_run=True,
    log_level="DEBUG",
    log_dir=Path("."),
    tournament_slug="t",
    oracle_staleness_threshold_s=10.0,
    cycle_interval_s=15.0,
)
MARKET_MAP = MarketMap(
    {
        "1": MarketMapping("1", "1", "Will the Democratic Party win the Arizona Senate?", Race("senate", "AZ"), "D", "tok-d", None),
        "2": MarketMapping("2", "2", "Will the Republican Party win the Arizona Senate?", Race("senate", "AZ"), "R", None, None),
    }
)


class FakeExchange:
    """Every read succeeds unless named in `fail`; place_order counts calls (DRY_RUN: returns None)."""

    dry_run = True
    halted = False

    def __init__(self, fail: set[str] = frozenset(), book_errors: dict[str, Exception] | None = None):
        self.fail = set(fail)
        self.book_errors = book_errors or {}  # exchange_id -> exception raised by get_orderbook
        self.placed = 0
        self.placed_exchange_ids: list[str] = []
        self.stats = {"get_retries": 0, "slowest_get_s": 0.0}

    def reset_get_stats(self):
        self.stats = {"get_retries": 3, "slowest_get_s": 7.25}  # pretend this cycle hit the slow mode

    def get_stats(self):
        return dict(self.stats)

    def _maybe_fail(self, read: str) -> None:
        if read in self.fail:
            raise RequestTimeoutError("GET", f"/{read}", 4, 17.0)

    async def get_pnl(self):
        self._maybe_fail("portfolio/pnl")
        return {"totalAccountValue": 100_000.0}

    async def get_positions(self):
        self._maybe_fail("portfolio/positions")
        return []

    async def iter_orders(self, **filters):
        self._maybe_fail("orders")
        return
        yield  # async generator

    async def get_prices(self, exchange_ids, tournament_id=None):
        self.prices_tournament_id = tournament_id
        self._maybe_fail("exchanges/prices")
        return {eid: {"exchangeId": eid, "bestBid": 0.40, "bestAsk": 0.45, "receivedAt": time.time()} for eid in exchange_ids}

    async def get_tournament(self, slug):
        return {"startDate": "2026-10-01T16:00:00Z", "endDate": "2026-11-04T17:00:00Z"}

    async def iter_relationships(self):
        return
        yield

    async def get_leaderboard(self, slug, limit=50):
        self._maybe_fail("leaderboard")
        return {"myRank": 5, "total": 100}

    async def get_orderbook(self, exchange_id, depth=20):
        if exchange_id in self.book_errors:
            raise self.book_errors[exchange_id]
        return {"bids": [{"price": 0.40, "quantity": 1000}], "asks": [{"price": 0.45, "quantity": 1000}]}

    async def get_constraints(self, **kwargs):
        return {"data": [], "violationsCount": 0}

    def note_open_orders(self, orders):
        pass

    def read_pause_remaining(self):
        return 0.0

    async def place_order(self, order, key, prepaid=False):
        self.placed += 1
        self.placed_exchange_ids.append(order.exchange_id)
        return None  # DRY_RUN


class FakePolymarket:
    async def get_quotes(self, tokens):
        # Fair P(D wins) = 0.60 vs SIG ask 0.45 -> a buy is planned for every mapped token.
        return {t: OracleQuote("polymarket", t, 0.59, 0.61, 500.0, 500.0, None, time.time()) for t in ("tok-d", "tok-ga")}


class FakeKalshi:
    async def get_quotes(self, tickers):
        return {}


TWO_RACES = MarketMap(
    {
        **{m.sig_exchange_id: m for m in MARKET_MAP},
        "3": MarketMapping("3", "3", "Will the Democratic Party win the Georgia Senate?", Race("senate", "GA"), "D", "tok-ga", None),
    }
)


def _strategy(exchange: FakeExchange, tmp_path: Path, market_map: MarketMap = MARKET_MAP) -> Strategy:
    return Strategy(SETTINGS, exchange, FakePolymarket(), FakeKalshi(), market_map, OrderJournal(tmp_path / "j.jsonl"))


# ---------------------------------------------------------------------- non-essential reads


def test_leaderboard_failure_does_not_fail_the_cycle(tmp_path, caplog):
    exchange = FakeExchange(fail={"leaderboard"})
    with caplog.at_level(logging.WARNING, logger="strategy"):
        report = asyncio.run(_strategy(exchange, tmp_path).run_cycle())
    assert report.kelly_fraction == 0.25  # neutral default: base kelly_fraction
    assert exchange.placed == 1 and report.dry_run == 1  # the cycle ran to the end and traded
    warning = next(r for r in caplog.records if r.getMessage() == "Non-essential read failed, continuing")
    assert warning.read == "leaderboard"


def test_pnl_failure_uses_last_known_bankroll(tmp_path, caplog):
    exchange = FakeExchange()
    strategy = _strategy(exchange, tmp_path)
    asyncio.run(strategy.run_cycle())
    exchange.fail.add("portfolio/pnl")
    with caplog.at_level(logging.WARNING, logger="strategy"):
        report = asyncio.run(strategy.run_cycle())
    assert report.bankroll == 100_000.0
    assert any(getattr(r, "read", None) == "portfolio/pnl" for r in caplog.records)


def test_pnl_failure_without_any_known_bankroll_skips_the_cycle(tmp_path):
    exchange = FakeExchange(fail={"portfolio/pnl"})
    with pytest.raises(RequestTimeoutError):
        asyncio.run(_strategy(exchange, tmp_path).run_cycle())
    assert exchange.placed == 0


# ---------------------------------------------------------------------- essential reads


@pytest.mark.parametrize("read", ["portfolio/positions", "orders"])
def test_essential_read_failure_skips_the_cycle_without_orders(tmp_path, read):
    exchange = FakeExchange(fail={read})
    with pytest.raises(RequestTimeoutError):
        asyncio.run(_strategy(exchange, tmp_path).run_cycle())
    assert exchange.placed == 0


# ---------------------------------------------------------------------- main.py reporting


class _FailingStrategy:
    def __init__(self, behaviour):
        self._behaviour = behaviour

    async def run_cycle(self):
        await self._behaviour()


def _run_one_cycle(strategy) -> int:
    return asyncio.run(main.Bot(SETTINGS, max_cycles=1)._loop(strategy))


def test_request_timeout_is_reported_as_api_failure_not_cycle_timeout(caplog):
    async def request_times_out():
        raise RequestTimeoutError("GET", "/exchanges/prices", 4, 17.2)

    with caplog.at_level(logging.ERROR, logger="main"):
        _run_one_cycle(_FailingStrategy(request_times_out))
    messages = [r.getMessage() for r in caplog.records]
    assert "Cycle timed out" not in messages
    record = next(r for r in caplog.records if r.getMessage() == "Cycle failed: API request timed out")
    assert (record.method, record.path, record.attempts) == ("GET", "/exchanges/prices", 4)


def test_bare_timeout_error_inside_the_cycle_is_not_a_cycle_timeout(caplog):
    async def bare_timeout():
        raise TimeoutError

    with caplog.at_level(logging.ERROR, logger="main"):
        _run_one_cycle(_FailingStrategy(bare_timeout))
    messages = [r.getMessage() for r in caplog.records]
    assert "Cycle failed: API request timed out" in messages and "Cycle timed out" not in messages


def test_real_cycle_timeout_is_still_reported_as_such(monkeypatch, caplog):
    monkeypatch.setattr(main, "CYCLE_TIMEOUT_S", 0.1)

    async def hangs():
        await asyncio.sleep(5)

    with caplog.at_level(logging.ERROR, logger="main"):
        _run_one_cycle(_FailingStrategy(hangs))
    assert "Cycle timed out" in [r.getMessage() for r in caplog.records]


def test_failure_backoff_schedule():
    assert [main.failure_backoff(n) for n in range(1, 8)] == [5, 10, 20, 40, 60, 60, 60]


# ---------------------------------------------------------------------- one order book times out


def test_one_book_timeout_drops_only_its_order(tmp_path, caplog):
    control = FakeExchange()
    asyncio.run(_strategy(control, tmp_path / "control", TWO_RACES).run_cycle())
    assert sorted(control.placed_exchange_ids) == ["1", "3"]  # both orders are planned normally

    exchange = FakeExchange(book_errors={"1": RequestTimeoutError("GET", "/exchanges/1/orderbook", 4, 17.0)})
    with caplog.at_level(logging.WARNING, logger="strategy"):
        report = asyncio.run(_strategy(exchange, tmp_path, TWO_RACES).run_cycle())
    assert exchange.placed_exchange_ids == ["3"]  # the other market still trades
    assert report.skipped_no_book == 1
    skipped = next(r for r in caplog.records if r.getMessage() == "Order skipped: order book unavailable")
    assert (skipped.exchange_id, skipped.side) == ("1", "yes")


def test_non_timeout_book_error_still_fails_the_cycle(tmp_path):
    exchange = FakeExchange(book_errors={"1": ApiError(400, "VALIDATION_ERROR", "bad id")})
    with pytest.raises(ApiError):
        asyncio.run(_strategy(exchange, tmp_path, TWO_RACES).run_cycle())
    assert exchange.placed == 0  # nothing is sent from a cycle that failed during planning


# ---------------------------------------------------------------------- GET counters


def test_cycle_done_carries_get_counters(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="strategy"):
        report = asyncio.run(_strategy(FakeExchange(), tmp_path).run_cycle())
    assert (report.get_retries, report.slowest_get_s) == (3, 7.25)
    done = next(r for r in caplog.records if r.getMessage() == "Cycle done")
    assert (done.get_retries, done.slowest_get_s) == (3, 7.25)


def test_cycle_failed_line_carries_get_counters(caplog):
    async def fails():
        raise RuntimeError("boom")

    bot = main.Bot(SETTINGS, max_cycles=1)
    bot._exchange = FakeExchange()
    bot._exchange.reset_get_stats()
    with caplog.at_level(logging.ERROR, logger="main"):
        asyncio.run(bot._loop(_FailingStrategy(fails)))
    failed = next(r for r in caplog.records if r.getMessage() == "Cycle failed")
    assert (failed.get_retries, failed.slowest_get_s) == (3, 7.25)


def test_real_exchange_client_counts_retries_and_slowest_get(monkeypatch):
    import exchange_api
    from test_exchange_http import _client, _FakeSession

    monkeypatch.setattr(exchange_api, "SIG_GET_TIMEOUT_S", 0.2)
    client = _client(_FakeSession(delays=[30.0, 0.05, 0.01]))  # 1st GET: hang + retry; 2nd GET: fast
    client.reset_get_stats()
    asyncio.run(client.get_account())
    asyncio.run(client.get_account())
    stats = client.get_stats()
    assert stats["get_retries"] == 1
    assert 0.2 <= stats["slowest_get_s"] < 1.5
