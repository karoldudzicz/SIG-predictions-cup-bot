"""SIG read volume: bulk prices, order books only for orders that can be sent, fallback.
Fakes - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import asyncio

from risk import RiskLimits
from strategy import OrderJournal, Strategy
from test_cycle_resilience import SETTINGS, FakeExchange, FakeKalshi, FakePolymarket, TWO_RACES


class BookCountingExchange(FakeExchange):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.books: list[str] = []

    async def get_tournament(self, slug):
        return {"id": "tid-uuid", **(await super().get_tournament(slug))}

    async def get_orderbook(self, exchange_id, depth=20):
        self.books.append(exchange_id)
        return await super().get_orderbook(exchange_id, depth)


def _strategy(exchange, tmp_path, limits=RiskLimits()):
    return Strategy(SETTINGS, exchange, FakePolymarket(), FakeKalshi(), TWO_RACES, OrderJournal(tmp_path / "j.jsonl"), limits=limits)


def test_bulk_prices_are_scoped_to_the_tournament(tmp_path):
    exchange = BookCountingExchange()
    report = asyncio.run(_strategy(exchange, tmp_path).run_cycle())
    assert exchange.prices_tournament_id == "tid-uuid"
    assert report.sig_prices == "bulk" and report.sig_prices_age_s is not None
    assert sorted(exchange.books) == ["1", "3"]  # one book per order that can be sent, not per market
    assert report.books_read == 2


def test_no_book_is_read_for_an_order_without_room_under_the_caps(tmp_path):
    exchange = BookCountingExchange()
    no_race_room = RiskLimits(max_race_worst_loss_fraction=0.0)
    report = asyncio.run(_strategy(exchange, tmp_path, no_race_room).run_cycle())
    assert exchange.books == [] and report.books_read == 0
    assert report.skipped_by_caps_precheck == 2 and report.planned == 0


def test_bulk_failure_falls_back_to_books_for_markets_that_may_trade(tmp_path):
    exchange = BookCountingExchange()
    taker = _strategy(exchange, tmp_path)
    asyncio.run(taker.run_cycle())  # a good bulk read: remembered for the fallback
    exchange.books.clear()
    exchange.fail = {"exchanges/prices"}
    report = asyncio.run(taker.run_cycle())  # no exception: the cycle goes on
    assert report.sig_prices == "fallback"
    # Both priced markets showed an edge on the last bulk read: one book each, reused for the fit.
    assert sorted(exchange.books) == ["1", "3"] and report.books_read == 2
    assert report.planned == 2


def test_bulk_failure_without_history_reads_no_books(tmp_path):
    exchange = BookCountingExchange(fail={"exchanges/prices"})
    report = asyncio.run(_strategy(exchange, tmp_path).run_cycle())
    assert report.sig_prices == "fallback" and exchange.books == [] and report.planned == 0


def test_get_prices_sends_the_tournament_id_and_counts_missing_ids(caplog):
    import logging

    from test_exchange_http import _client, _FakeSession

    class ParamsSession(_FakeSession):
        def request(self, method, url, params=None, json=None, timeout=None):
            self.params = params
            return super().request(method, url, params=params, json=json, timeout=timeout)

    session = ParamsSession(delays=[0.0], body={"data": [{"exchangeId": "1", "bestBid": 0.4, "bestAsk": 0.45}], "missingIds": ["9", "8"]})
    client = _client(session)
    with caplog.at_level(logging.WARNING, logger="exchange_api"):
        prices = asyncio.run(client.get_prices(["1", "9", "8"], tournament_id="tid-uuid"))
    assert session.params == {"ids": "1,9,8", "tournamentId": "tid-uuid"}
    assert prices["1"]["receivedAt"] > 0
    warning = next(r for r in caplog.records if r.getMessage() == "Exchanges missing from price snapshot")
    assert warning.missing == 2


# ---------------------------------------------------------------------- read bucket


def _budget_client(per_minute, burst, tokens=None, **session_kwargs):
    import exchange_api
    from test_exchange_http import _client, _FakeSession

    session = _FakeSession(**session_kwargs)
    client = _client(session)
    client._read_bucket = exchange_api._WriteBucket(per_minute=per_minute, burst=burst)
    if tokens is not None:
        client._read_bucket._tokens = tokens
    return client, session


def test_reads_wait_for_tokens_instead_of_going_out():
    import time

    client, session = _budget_client(per_minute=600, burst=2, delays=[0.0] * 4)  # 10 tokens/s

    async def burst():
        await asyncio.gather(*(client.get_orderbook(str(i)) for i in range(4)))

    started = time.monotonic()
    asyncio.run(burst())
    assert len(session.calls) == 4
    assert time.monotonic() - started >= 0.18  # 2 from the burst, 2 more at 10/s


def test_essential_reads_go_before_waiting_order_books():
    client, session = _budget_client(per_minute=600, burst=10, tokens=0.0, delays=[0.0] * 4)

    async def scenario():
        books = [asyncio.create_task(client.get_orderbook(str(i))) for i in range(3)]
        await asyncio.sleep(0)
        positions = asyncio.create_task(client.get_positions())
        await asyncio.gather(*books, positions)

    session.body = {"positions": [], "bids": [], "asks": []}
    asyncio.run(scenario())
    paths = [path for _, path, _ in session.calls]
    assert paths[0] == "/portfolio/positions"  # asked last, served first


def test_token_wait_is_bounded_by_the_get_deadline(monkeypatch):
    import exchange_api
    import pytest

    monkeypatch.setattr(exchange_api, "SIG_GET_DEADLINE_S", 0.2)
    client, session = _budget_client(per_minute=1e-3, burst=1, tokens=0.0, delays=[])
    with pytest.raises(exchange_api.RequestTimeoutError):
        asyncio.run(client.get_leaderboard("t"))
    assert session.calls == []  # never sent
