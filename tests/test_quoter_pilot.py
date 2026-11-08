"""Quoter micro-pilot controls: market allowlist, QUOTER_DRY_RUN, summary logging. Fakes - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace

import pytest

import config
import quoter as Q
from test_quoter import NOW, FakeExchange, _ctx, _open_orders, _placed
from strategy import SigBook


def _manager(exchange, tmp_path, *, mode="inherit", allowlist=None, clock=lambda: NOW):
    return Q.QuoterManager(exchange, journal=Q.QuoterJournal(tmp_path / "quoter.jsonl"), clock=clock, dry_run_mode=mode, allowlist=allowlist)


# ---------------------------------------------------------------------- allowlist


def test_allowlist_limits_the_quoted_markets(tmp_path):
    exchange = FakeExchange()
    manager = _manager(exchange, tmp_path, allowlist=frozenset({"3"}))
    report = asyncio.run(manager.run_cycle(_ctx({"1": 0.50, "3": 0.50, "5": 0.50})))
    assert report.markets == 1 and {o.exchange_id for o in _placed(exchange)} == {"3"}
    assert manager._last_skips["not allowlisted"] == 2


def test_allowlisted_market_still_obeys_every_guard(tmp_path):
    exchange = FakeExchange()
    manager = _manager(exchange, tmp_path, allowlist=frozenset({"1"}))
    report = asyncio.run(manager.run_cycle(_ctx({"1": 0.97})))  # tail
    assert report.markets == 0 and _placed(exchange) == [] and report.blocked_tail == 1


def test_no_allowlist_means_any_eligible_market(tmp_path):
    exchange = FakeExchange()
    report = asyncio.run(_manager(exchange, tmp_path).run_cycle(_ctx({"1": 0.50, "3": 0.50})))
    assert report.markets == 2


def test_allowlist_parsing():
    config.CONFIG_WARNINGS.clear()
    parse = config._env_id_list_override
    assert parse("X", environ={}) is None and parse("X", environ={"X": "  "}) is None
    assert parse("X", environ={"X": "1065, 1077"}) == frozenset({"1065", "1077"})
    assert config.CONFIG_WARNINGS == []
    assert parse("X", environ={"X": "1065,abc"}) == frozenset({"1065"})
    assert parse("X", environ={"X": "abc"}) == frozenset()  # broken list matches NOTHING, never everything
    assert len(config.CONFIG_WARNINGS) == 2
    config.CONFIG_WARNINGS.clear()


# ---------------------------------------------------------------------- QUOTER_DRY_RUN


@pytest.mark.parametrize(
    ("global_dry_run", "mode", "virtual"),
    [(False, "inherit", False), (False, "false", False), (False, "true", True), (True, "inherit", True), (True, "false", True), (True, "true", True)],
)
def test_virtual_table(global_dry_run, mode, virtual):
    assert Q.quoter_is_virtual(global_dry_run, mode) is virtual


def test_quoter_dry_run_true_with_live_bot_sends_nothing(tmp_path):
    exchange = FakeExchange()  # dry_run = False: the taker trades live
    manager = _manager(exchange, tmp_path, mode="true")
    assert manager.virtual
    asyncio.run(manager.run_cycle(_ctx({"1": 0.50})))
    assert exchange.batches == [] and len(manager.orders) == 2
    assert all(oid.startswith(Q.VIRTUAL_PREFIX) for oid in manager.orders)
    # A requote cancels only virtually: no cancel-all reaches the exchange.
    books = {"1": SigBook(yes_bid=0.455, yes_ask=0.55)}
    report = asyncio.run(manager.run_cycle(_ctx({"1": 0.50}, now=NOW + 15, books=books)))
    assert report.requotes == 1 and exchange.cancel_alls == [] and exchange.batches == []
    # Virtual quotes do not use up the live taker's room, but do use the quoter's own budget.
    assert manager.book_cost() == 0.0 and manager.resting_race_legs(Q.MarketMap({})) == {}
    assert manager.book_cost(for_others=False) > 0


def test_global_dry_run_is_always_virtual(tmp_path):
    exchange = FakeExchange()
    exchange.dry_run = True
    manager = _manager(exchange, tmp_path, mode="false")
    asyncio.run(manager.run_cycle(_ctx({"1": 0.50})))
    assert manager.virtual and exchange.batches == [] and len(manager.orders) == 2
    assert manager.book_cost() > 0  # whole bot simulated: virtual quotes count for every book


def test_live_quoter_sends(tmp_path):
    exchange = FakeExchange()
    manager = _manager(exchange, tmp_path, mode="inherit")
    asyncio.run(manager.run_cycle(_ctx({"1": 0.50})))
    assert not manager.virtual and len(exchange.batches) == 1


# ---------------------------------------------------------------------- logging


def test_one_info_summary_per_minute_and_market_detail_on_debug(tmp_path, caplog):
    exchange = FakeExchange()
    manager = _manager(exchange, tmp_path, clock=lambda: NOW)
    with caplog.at_level(logging.DEBUG, logger="quoter"):
        asyncio.run(manager.run_cycle(_ctx({"1": 0.50, "3": 0.97})))
        asyncio.run(manager.run_cycle(replace(_ctx({"1": 0.50, "3": 0.97}, now=NOW + 30), open_orders=_open_orders(manager))))
        books = {"1": SigBook(yes_bid=0.455, yes_ask=0.55), "3": SigBook(0.45, 0.55)}
        asyncio.run(manager.run_cycle(replace(_ctx({"1": 0.50, "3": 0.97}, now=NOW + 61, books=books), open_orders=_open_orders(manager))))
    summaries = [r for r in caplog.records if r.getMessage() == "Quoter summary"]
    assert len(summaries) == 1 and summaries[0].levelno == logging.INFO
    summary = summaries[0]
    assert summary.markets == ["1"] and summary.active_quotes == 2 and summary.cycles == 3
    assert summary.requotes == 1 and summary.placed == 4 and summary.skipped_markets == {"tail": 1}
    per_market = [r for r in caplog.records if r.getMessage() in ("Quoter market", "Quoter market skipped")]
    assert per_market and all(r.levelno == logging.DEBUG for r in per_market)
    detail = next(r for r in per_market if r.getMessage() == "Quoter market")
    assert (detail.exchange_id, detail.bid_yes, detail.bid_no, detail.other_yes_bid, detail.other_yes_ask) == ("1", 0.455, 0.455, 0.45, 0.55)
    skipped = next(r for r in per_market if r.getMessage() == "Quoter market skipped")
    assert (skipped.exchange_id, skipped.reason) == ("3", "tail")
    # Nothing per market at INFO.
    assert not [r for r in caplog.records if r.levelno == logging.INFO and getattr(r, "exchange_id", None) and r.getMessage() != "Quoter jump guard"]
