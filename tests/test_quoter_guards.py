"""Quoter pre-pilot guards: minimum OTHER spread, never marketable, one-sided books. Fakes - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import replace

import pytest

import quoter as Q
from exchange_api import PRICE_TICK
from strategy import SigBook
from test_quoter import NOW, FakeExchange, _ctx, _open_orders, _placed


def _manager(exchange, tmp_path, **kwargs):
    return Q.QuoterManager(exchange, journal=Q.QuoterJournal(tmp_path / "quoter.jsonl"), clock=lambda: NOW, dry_run_mode="inherit", allowlist=None, **kwargs)


def _run(manager, fair, book, now=NOW):
    ctx = _ctx({"1": fair}, now=now, books={"1": book})
    return asyncio.run(manager.run_cycle(replace(ctx, open_orders=_open_orders(manager))))


# ---------------------------------------------------------------------- never marketable


def test_report_case_one_tick_spread_is_not_marketable():
    """Other 0.50 / 0.505, fair 0.55: before the guard the YES bid was 0.505 = their ask."""
    plan = Q.plan_quotes(0.55, 0.005, other_yes_bid=0.50, other_no_bid=round(1 - 0.505, 3), party="D", net_delta=0)
    assert plan.yes == 0.50  # lowered to a tick under their ask: joins their bid, never lifts the ask
    assert plan.no <= round(1 - 0.50 - PRICE_TICK, 3)  # NO bid a tick under their NO ask


def test_no_quote_is_ever_marketable():
    rng = random.Random(11)
    for _ in range(3000):
        bid = round(rng.randint(10, 180) * PRICE_TICK, 3)
        ask = round(bid + rng.randint(1, 12) * PRICE_TICK, 3)
        plan = Q.plan_quotes(rng.uniform(0.05, 0.95), rng.uniform(0.005, 0.04), bid, round(1 - ask, 3), rng.choice("DR"), rng.uniform(-299, 299))
        if plan.yes is not None:
            assert plan.yes <= ask - PRICE_TICK + 1e-9
        if plan.no is not None:
            assert plan.no <= (1 - bid) - PRICE_TICK + 1e-9


def test_spread_reasons():
    assert Q.spread_block_reason(0.50, 0.505, 3, False) == "other spread 1 ticks < 3"
    assert Q.spread_block_reason(0.50, 0.51, 3, False) == "other spread 2 ticks < 3"
    assert Q.spread_block_reason(0.50, 0.515, 3, False) is None
    assert Q.spread_block_reason(0.50, None, 3, False) == "one-sided book"
    assert Q.spread_block_reason(None, None, 3, True) is None


# ---------------------------------------------------------------------- manager


@pytest.mark.parametrize("ask", [0.505, 0.51])  # 1 and 2 ticks
def test_narrow_spreads_get_no_quotes(tmp_path, ask):
    exchange = FakeExchange()
    report = _run(_manager(exchange, tmp_path), 0.55, SigBook(0.50, ask))
    assert report.markets == 0 and _placed(exchange) == []


def test_three_ticks_quotes_both_sides_without_crossing(tmp_path):
    exchange = FakeExchange()
    manager = _manager(exchange, tmp_path)
    report = _run(manager, 0.55, SigBook(0.50, 0.515))
    quotes = {o.side: o.price for o in _placed(exchange)}
    assert report.markets == 1 and set(quotes) == {"yes", "no"}
    assert quotes["yes"] == 0.505 and quotes["yes"] < 0.515  # improves their bid, below their ask
    assert 1 - quotes["no"] > 0.50  # our implied YES ask stays above their bid


def test_one_sided_book_is_not_quoted_unless_allowed(tmp_path):
    exchange = FakeExchange()
    assert _run(_manager(exchange, tmp_path), 0.50, SigBook(0.45, None)).markets == 0
    assert _placed(exchange) == []
    allowed = FakeExchange()
    assert _run(_manager(allowed, tmp_path / "b", allow_one_sided=True), 0.50, SigBook(0.45, None)).markets == 1
    assert {o.side for o in _placed(allowed)} == {"yes", "no"}


def test_quotes_are_withdrawn_when_the_spread_narrows(tmp_path, caplog):
    exchange = FakeExchange()
    manager = _manager(exchange, tmp_path)
    _run(manager, 0.50, SigBook(0.45, 0.55))
    assert len(manager.orders) == 2
    with caplog.at_level(logging.INFO, logger="quoter"):
        report = _run(manager, 0.50, SigBook(0.50, 0.51), now=NOW + 15)  # others tighten to 2 ticks
    assert report.markets == 0 and manager.orders == {}
    assert exchange.cancel_alls == [{"exchange_id": "1", "tournament_id": "tid"}]
    withdrawn = next(r for r in caplog.records if r.getMessage() == "Quoter quotes withdrawn")
    assert (withdrawn.exchange_id, withdrawn.reason) == ("1", "other spread 2 ticks < 3")


def test_our_own_quote_does_not_narrow_the_spread_we_measure(tmp_path):
    """3-tick market, we improve the bid: the bulk top shows OUR 0.505 (1 tick to their ask),
    but the threshold and targets use the book without our orders - no withdraw, no requote."""
    exchange = FakeExchange()
    manager = _manager(exchange, tmp_path)
    _run(manager, 0.55, SigBook(0.50, 0.515))
    ours = {o.side: o.price for o in manager.orders.values()}
    exchange.own_orders = lambda eid: [("bid", ours["yes"], 100.0), ("ask", round(1 - ours["no"], 3), 100.0)]

    async def book(exchange_id, depth=20):  # the real book: ours + theirs
        return {"bids": [{"price": 0.505, "quantity": 100}, {"price": 0.50, "quantity": 300}],
                "asks": [{"price": 0.515, "quantity": 200}, {"price": round(1 - ours["no"], 3), "quantity": 100}]}

    exchange.get_orderbook = book
    report = _run(manager, 0.55, SigBook(0.505, 0.515), now=NOW + 15)
    assert report.markets == 1 and report.requotes == 0 and exchange.cancel_alls == []
    assert len(exchange.batches) == 1  # nothing re-posted
