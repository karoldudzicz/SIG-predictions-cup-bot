"""Fill price convention (fills.py), and every reader that uses it. No network.

Synthetic numbers shaped like the live case that settled the convention: a REST NO fill
whose shares * price equals the position's cost basis (here 4000 * 0.95 = 3800).

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

import ladder as L
import markout as M
import quoter as Q
from fills import fill_cost, fill_price_side, fill_price_yes, fill_shares
from strategy import Holding, SigBook

NO_FILL = {"id": 77, "orderId": 1, "exchangeId": "9001", "side": "no", "quantity": -4000, "price": 0.95, "filledAt": "2027-01-15T08:00:00Z"}


def test_rest_no_fill_is_a_no_price():
    assert fill_shares(NO_FILL) == 4000
    assert fill_price_side(NO_FILL) == 0.95
    assert fill_cost(NO_FILL) == pytest.approx(3800.0)  # = the position's cost basis
    assert fill_price_yes(NO_FILL) == pytest.approx(0.05)


def test_rest_yes_fill_is_not_converted():
    fill = {**NO_FILL, "side": "yes", "quantity": 300, "price": 0.62}
    assert fill_price_side(fill) == fill_price_yes(fill) == 0.62 and fill_cost(fill) == pytest.approx(186.0)


def test_realtime_convention_is_kept_apart():
    # Realtime (documented): YES-denominated - the same NO trade would arrive as 0.05.
    realtime = {**NO_FILL, "price": 0.05}
    assert fill_price_side(realtime, "realtime") == pytest.approx(0.95)
    assert fill_price_yes(realtime, "realtime") == 0.05


# ---------------------------------------------------------------------- ladder


def test_ladder_no_inventory_cost_and_exit_on_the_no_side(tmp_path):
    from test_ladder import FakeExchange, _ctx, _manager, _quote

    exchange = FakeExchange()
    # Fair YES 0.05 on AZ D -> the favourite is NO (fair NO 0.95); the refetch on the fill keeps it there.
    manager = _manager(exchange, tmp_path, pm_quotes={"tok-1": _quote(0.04, 0.06)})
    asyncio.run(manager.run_cycle(_ctx({"1": 0.05})))
    level0 = next(o for o in manager.orders.values() if o.level == 0)
    assert level0.side == "no"
    fill = {"id": 1, "orderId": int(level0.order_id), "exchangeId": "1", "side": "no", "quantity": -100, "price": level0.price, "filledAt": "2027-01-15T08:00:00Z"}
    exchange.fills_pages = [[fill]]
    manager._fills_since = 0
    holdings = {"1": Holding("no", 100, 95.0, 100 * level0.price)}
    books = {"1": SigBook(yes_bid=0.01, yes_ask=0.04)}  # NO bid 0.96, NO ask 0.99
    asyncio.run(manager.run_cycle(_ctx({"1": 0.05}, holdings=holdings, books=books, open_orders=[])))
    pos = manager.positions["1"]
    assert pos.side == "no" and pos.last_fill_price == level0.price  # a NO price, not 1 - it
    assert pos.cost == pytest.approx(100 * level0.price)
    exit_order = exchange.batches[-1][0]
    assert (exit_order.action, exit_order.side) == ("sell", "no")
    assert 0.95 <= exit_order.price <= 0.99  # around fair NO, never near 0.05
    journal = [json.loads(line) for line in (tmp_path / "ladder.jsonl").read_text(encoding="utf-8").splitlines()]
    assert next(e for e in journal if e["event"] == "fill")["price"] == level0.price


def test_ladder_exit_fill_pnl_in_no_terms(tmp_path):
    from test_ladder import FakeExchange, _manager

    manager = _manager(FakeExchange(), tmp_path)
    manager.positions["9001"] = L.LadderPosition("9001", "no", bought=4000, cost=3800.0)
    manager._exit_ids.add("9")
    report = L.LadderReport()
    manager._on_exit_fill({"id": 2, "orderId": 9, "exchangeId": "9001", "side": "no", "quantity": -4000, "price": 0.96}, report)
    assert report.rounds_closed == 1 and report.realized_pnl == pytest.approx(4000 * (0.96 - 0.95), abs=1e-3)


# ---------------------------------------------------------------------- quoter


def test_quoter_no_fill_inventory(tmp_path):
    from types import SimpleNamespace

    manager = Q.QuoterManager(SimpleNamespace(dry_run=False, halted=False), journal=Q.QuoterJournal(tmp_path / "q.jsonl"), clock=lambda: 0.0)
    manager._order_meta["1"] = Q.QuoterOrder("1", "9001", "no", 0.95, 4000, 0.0)
    ctx = Q.QuoterContext(now=1.0, bankroll=1e5, tournament_id="t", market_map=None, fair={}, quotes={}, books={},
                          holdings={}, open_orders=[], limits=None, max_quote_age_s=10.0, fills=[NO_FILL])
    report = Q.QuoterReport()
    manager._apply_fills(ctx, report)
    pos = manager.positions["9001"]
    assert pos.side == "no" and pos.shares == -4000
    assert pos.avg_yes == pytest.approx(0.05) and pos.open_cost == pytest.approx(3800.0)
    replayed = Q.QuoterJournal(tmp_path / "q.jsonl").replay()["9001"]
    assert replayed.open_cost == pytest.approx(3800.0)


# ---------------------------------------------------------------------- markout


def test_no_markout_in_one_space(tmp_path):
    tracker = M.MarkoutTracker(tmp_path / "m.jsonl", clock=lambda: 0.0)
    filled_at = M.unix_time(NO_FILL["filledAt"])
    tracker.record("taker", NO_FILL, "buy", fair_yes=0.05)
    fill_line = json.loads((tmp_path / "m.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert (fill_line["price_source"], fill_line["price_side"]) == ("rest", 0.95)
    assert fill_line["price_yes"] == pytest.approx(0.05) and fill_line["markout_at_fill"] == pytest.approx(0.0)
    # YES fair down to 0.04 = NO up to 0.96: +2 ticks for the NO buyer.
    written = tracker.update(filled_at + 61, {"9001": 0.04})
    assert written[0].ticks == pytest.approx(2.0)


def test_legacy_no_lines_are_corrected_on_read(tmp_path):
    # Written before the fix: no price_source, the NO price stored as price_yes.
    legacy_fill = {"event": "fill", "source": "taker", "exchange_id": "9001", "side": "no", "action": "buy", "price_yes": 0.95,
                   "quantity": 4000, "fill_id": 77, "order_id": 1, "filled_at": 0.0, "fair_at_fill": 0.05}
    legacy_markout = {"event": "markout", "source": "taker", "exchange_id": "9001", "fill_id": 77, "horizon_s": 60, "fair": 0.04, "markout_ticks": -182.0}
    assert M.legacy_price_yes(legacy_fill) == pytest.approx(0.05)
    assert M.legacy_price_yes({**legacy_fill, "price_source": "rest", "price_yes": 0.05}) == 0.05  # new lines untouched

    path = tmp_path / "m.jsonl"
    path.write_text(json.dumps(legacy_fill) + "\n", encoding="utf-8")
    restarted = M.MarkoutTracker(path, clock=lambda: 30.0)
    assert restarted.update(61.0, {"9001": 0.04})[0].ticks == pytest.approx(2.0)  # pending horizon, corrected price

    spec = importlib.util.spec_from_file_location("markout_report", Path(__file__).resolve().parent.parent / "scripts" / "markout_report.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    rows = report.summarize([json.dumps(legacy_fill), json.dumps(legacy_markout)])
    assert rows == [{"source": "taker", "horizon_s": 60, "n": 1, "mean_ticks": 2.0, "negative_share": 0.0, "no_fair": 0, "corrected": 1}]
