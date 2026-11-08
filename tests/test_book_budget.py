"""Order-book read budget per cycle: top-N by expected profit, deferral without waiting,
recent-book cache. Fakes - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

import exchange_api
import strategy
from exchange_api import READ_BOOK, ReadDeferredError
from market_mapper import MarketMap, MarketMapping, Race
from oracle_api import OracleQuote
from strategy import OrderJournal, PlannedOrder, Strategy, book_read_order
from test_cycle_resilience import SETTINGS, TWO_RACES, FakeExchange, FakeKalshi, FakePolymarket

N = 30
MAP = MarketMap({
    str(i): MarketMapping(str(i), str(i), f"D seat {i}", Race("house", "ZZ", f"{i:02d}"), "D", f"tok-{i}", None) for i in range(1, N + 1)
})
FAIR = {str(i): 0.55 + 0.01 * i for i in range(1, N + 1)}  # ask 0.45 everywhere: edge grows with i


class Oracle:
    async def get_quotes(self, tokens):
        now = time.time()
        return {f"tok-{eid}": OracleQuote("polymarket", f"tok-{eid}", f - 0.005, f + 0.005, 500.0, 500.0, None, now) for eid, f in FAIR.items() if f"tok-{eid}" in tokens}


class BudgetExchange(FakeExchange):
    """FakeExchange (DRY_RUN) with a read bucket: a fixed per-cycle book budget."""

    def __init__(self, budget):
        super().__init__()
        self.budget = budget
        self.reads: list[str] = []
        self.token_waits: list[float | None] = []

    async def get_tournament(self, slug):
        return {"id": "tid", **(await super().get_tournament(slug))}

    def read_budget(self, read_class, window_s, reserve=0):
        assert read_class == READ_BOOK
        return self.budget

    async def get_orderbook(self, exchange_id, depth=20, token_wait_s=None):
        self.reads.append(exchange_id)
        self.token_waits.append(token_wait_s)
        book = await super().get_orderbook(exchange_id, depth)
        return {**book, "receivedAt": time.time()}


def _strategy(exchange, tmp_path, market_map=MAP, oracle=None):
    return Strategy(SETTINGS, exchange, oracle or Oracle(), FakeKalshi(), market_map, OrderJournal(tmp_path / "j.jsonl"))


# ---------------------------------------------------------------------- (a) + (b)


def test_thirty_planned_twelve_read_eighteen_deferred_without_waiting(tmp_path, caplog):
    exchange = BudgetExchange(budget=12)
    with caplog.at_level(logging.INFO):
        report = asyncio.run(_strategy(exchange, tmp_path).run_cycle())
    assert len(exchange.reads) == 12 and report.books_read == 12
    assert report.books_deferred == 18 and report.skipped_no_book == 0
    assert set(exchange.token_waits) == {strategy.BOOK_READ_WINDOW_S}  # never up to the GET deadline
    deferred_logs = [r for r in caplog.records if r.getMessage() == "Order deferred: read budget"]
    assert len(deferred_logs) == 18 and all(r.levelno == logging.INFO for r in deferred_logs)
    assert not [r for r in caplog.records if r.getMessage() in ("Request failed, giving up", "Request failed, retrying", "Order skipped: order book unavailable")]
    # (b) the books read are those of the best expected profit: here the 12 highest fairs.
    assert sorted(exchange.reads, key=int) == [str(i) for i in range(N - 11, N + 1)]


def test_read_order_reduces_then_deferred_then_expected_profit():
    fair = {"1": 0.60, "2": 0.70, "3": 0.80, "4": 0.50}
    margins = dict.fromkeys(fair, 0.01)
    orders = [
        PlannedOrder("increase", "1", "yes", 1000, 0.45, "edge"),  # EV 1000 * 0.14 = 140
        PlannedOrder("increase", "2", "yes", 500, 0.45, "edge"),  # EV 500 * 0.24 = 120
        PlannedOrder("increase", "3", "yes", 100, 0.45, "edge"),  # EV 100 * 0.34 = 34
        PlannedOrder("reduce", "4", "yes", 50, 0.49, "flip"),
    ]
    assert book_read_order(orders, fair, margins, set()) == ["4", "1", "2", "3"]
    assert book_read_order(orders, fair, margins, {"3"}) == ["4", "3", "1", "2"]  # deferred last cycle: next in line


# ---------------------------------------------------------------------- (c)


def test_next_cycle_reads_what_was_deferred(tmp_path):
    exchange = BudgetExchange(budget=12)
    taker = _strategy(exchange, tmp_path)
    asyncio.run(taker.run_cycle())
    first = set(exchange.reads)
    deferred = set(MAP_IDS) - first
    exchange.reads.clear()
    report = asyncio.run(taker.run_cycle())
    assert report.books_cache_hits == 12  # last cycle's books, < 10 s old
    assert len(exchange.reads) == 12 and set(exchange.reads) <= deferred
    assert report.books_deferred == 6


MAP_IDS = [str(i) for i in range(1, N + 1)]


# ---------------------------------------------------------------------- (d)


def test_cache_respects_its_age_and_the_quote_age(tmp_path, monkeypatch):
    exchange = BudgetExchange(budget=100)
    taker = _strategy(exchange, tmp_path, market_map=TWO_RACES, oracle=FakePolymarket())
    first = asyncio.run(taker.run_cycle())
    assert first.books_read == 2 and first.dry_run == 2

    # 9 s old: reused, and its age now counts in the quote age (here limited to 5 s).
    for book in taker._recent_books.values():
        book["receivedAt"] -= 9
    real = strategy.quote_too_old
    monkeypatch.setattr(strategy, "quote_too_old", lambda order: real(order, max_age_s=5.0))
    exchange.reads.clear()
    second = asyncio.run(taker.run_cycle())
    assert second.books_cache_hits == 2 and exchange.reads == []
    assert second.skipped_stale_quote == 2 and second.dry_run == 0

    # Older than BOOK_CACHE_MAX_AGE_S: read again, fresh, the orders go out.
    for book in taker._recent_books.values():
        book["receivedAt"] -= strategy.BOOK_CACHE_MAX_AGE_S
    third = asyncio.run(taker.run_cycle())
    assert third.books_cache_hits == 0 and sorted(exchange.reads) == ["1", "3"] and third.dry_run == 2


# ---------------------------------------------------------------------- (e)


def test_few_orders_behave_as_before(tmp_path):
    plain = FakeExchange()  # no read bucket: the old path
    budgeted = BudgetExchange(budget=50)
    before = asyncio.run(_strategy(plain, tmp_path / "a", market_map=TWO_RACES, oracle=FakePolymarket()).run_cycle())
    after = asyncio.run(_strategy(budgeted, tmp_path / "b", market_map=TWO_RACES, oracle=FakePolymarket()).run_cycle())
    for field in ("targets", "planned", "dry_run", "skipped_no_book", "skipped_stale_quote", "books_read"):
        assert getattr(before, field) == getattr(after, field), field
    assert after.books_deferred == 0 and plain.placed_exchange_ids == budgeted.placed_exchange_ids


# ---------------------------------------------------------------------- the client side


def _client_with_bucket(tokens, per_minute=80.0, burst=10):
    from test_exchange_http import _client, _FakeSession

    session = _FakeSession(delays=[0.0] * 3, body={"bids": [], "asks": []})
    client = _client(session, dry_run=False)
    client._read_bucket = exchange_api._WriteBucket(per_minute=per_minute, burst=burst)
    client._read_bucket._tokens = tokens
    return client, session


def test_read_budget_counts_tokens_now_plus_the_window_minus_the_reserve():
    client, _ = _client_with_bucket(tokens=5.0)  # 80/min = 1.33/s
    assert client.read_budget(READ_BOOK, 3.0, reserve=1) == 8  # floor(5 + 4) - 1
    client._start_read_pause(30.0, "/x")
    assert client.read_budget(READ_BOOK, 3.0) == 0  # paused: no book reads at all


def test_no_token_within_the_window_is_a_deferral_not_a_timeout(caplog):
    client, session = _client_with_bucket(tokens=0.0, per_minute=1e-3)
    started = time.monotonic()
    with caplog.at_level(logging.DEBUG, logger="exchange_api"), pytest.raises(ReadDeferredError):
        asyncio.run(client.get_orderbook("1", token_wait_s=0.05))
    assert time.monotonic() - started < 1.0 and session.calls == []
    assert not [r for r in caplog.records if r.getMessage().startswith("Request failed")]  # no retry, no give-up


def test_reserve_keeps_room_for_the_next_cycles_essential_reads(tmp_path):
    taker = _strategy(FakeExchange(), tmp_path)
    # 30 markets: positions + orders + pnl + 1 price page = 4 (DRY_RUN: no fills read), the
    # bucket refills 80/60 * 5 s = 6.7 tokens in the shortest gap -> no reserve needed.
    assert taker._book_read_reserve() == 0
    big = MarketMap({str(i): MarketMapping(str(i), str(i), "x", Race("house", "ZZ", f"{i:03d}"), "D", None, None) for i in range(1, 801)})
    assert _strategy(FakeExchange(), tmp_path / "b", market_map=big)._book_read_reserve() == 5  # 3 + 8 pages - 6.7
