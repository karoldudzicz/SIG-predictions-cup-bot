"""Quoter micro-live pilot flags (all default off): join touch, skip held, persisted blocks, daily
loss cap, resting notional cap, virtual detail. Fakes - no network.

Run from the project root:  python -m pytest tests/test_quoter_pilot_flags.py -v
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import replace
from datetime import datetime, timezone

import pytest

import config
import quoter as Q
from markout import Markout
from strategy import Holding, SigBook
from test_quoter import NOW, FakeExchange, _ctx, _open_orders, _placed


def _manager(exchange, tmp_path, clock=lambda: NOW, **flags):
    return Q.QuoterManager(exchange, journal=Q.QuoterJournal(tmp_path / "quoter.jsonl"), clock=clock, **flags)


def _cycle(manager, ctx):
    return asyncio.run(manager.run_cycle(replace(ctx, open_orders=[] if manager.virtual else _open_orders(manager))))


def _journal(tmp_path):
    return [json.loads(line) for line in (tmp_path / "quoter.jsonl").read_text(encoding="utf-8").splitlines()]


def _iso(unix):
    return datetime.fromtimestamp(unix, tz=timezone.utc).isoformat(timespec="milliseconds")


def test_every_pilot_flag_defaults_off():
    assert (config.QUOTER_JOIN_TOUCH, config.QUOTER_SKIP_HELD, config.QUOTER_PERSIST_BLOCKS, config.QUOTER_VIRTUAL_DETAIL) == (False,) * 4
    assert config.QUOTER_DAILY_LOSS_CAP == 0.0 and config.QUOTER_MAX_RESTING_FRAC == 0.0


def test_pilot_flags_logged_only_when_set(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="quoter"):
        _manager(FakeExchange(), tmp_path)
        assert not [r for r in caplog.records if r.getMessage() == "Quoter pilot flags"]
        _manager(FakeExchange(), tmp_path, join_touch=True)
    flags = [r for r in caplog.records if r.getMessage() == "Quoter pilot flags"]
    assert len(flags) == 1 and flags[0].join_touch is True and flags[0].daily_loss_cap == 0.0


# ---------------------------------------------------------------------- 1. join the touch


def test_join_touch_bids_at_the_touch_capped_and_never_marketable():
    improve = Q.plan_quotes(0.50, 0.005, 0.40, 0.40, "D", 0.0)
    join = Q.plan_quotes(0.50, 0.005, 0.40, 0.40, "D", 0.0, join_touch=True)
    assert (improve.yes, improve.no) == (0.405, 0.405) and (join.yes, join.no) == (0.40, 0.40)
    capped = Q.plan_quotes(0.50, 0.005, 0.49, 0.49, "D", 0.0, join_touch=True)  # touch 0.49 above fair - edge 0.485
    assert (capped.yes, capped.no) == (0.485, 0.485)
    tight = Q.plan_quotes(0.50, 0.005, 0.30, 0.695, "D", 0.0, join_touch=True)  # YES touch 0.30 / 0.305: one tick
    assert tight.yes <= 1 - 0.695 - 0.005 + 1e-9 and tight.yes + tight.no <= 1 - 2 * 0.005 + 1e-9


def test_join_touch_in_a_cycle(tmp_path):
    exchange = FakeExchange()
    _cycle(_manager(exchange, tmp_path, join_touch=True), _ctx({"1": 0.50}))  # other book 0.45 / 0.55
    assert sorted((o.side, o.price) for o in _placed(exchange)) == [("no", 0.45), ("yes", 0.45)]


# ---------------------------------------------------------------------- 2. skip held


def test_skip_held_skips_carry_and_withdraws_but_not_quoter_inventory(tmp_path):
    exchange = FakeExchange()
    manager = _manager(exchange, tmp_path, skip_held=True)
    _cycle(manager, _ctx({"1": 0.50, "3": 0.50}))
    assert {o.exchange_id for o in manager.orders.values()} == {"1", "3"}
    report = _cycle(manager, _ctx({"1": 0.50, "3": 0.50}, now=NOW + 15, holdings={"3": Holding("no", 200, 100.0, 90.0)}))  # carry on 3
    assert report.blocked_held == 1 and {o.exchange_id for o in manager.orders.values()} == {"1"}
    assert exchange.cancel_alls == [{"exchange_id": "3", "tournament_id": "tid"}]  # scoped to the held market only
    assert manager._last_skips["held (not quoter)"] == 1
    yes_id = next(oid for oid, o in manager.orders.items() if o.side == "yes")
    fill = {"id": "f1", "orderId": yes_id, "exchangeId": "1", "side": "yes", "price": 0.455, "quantity": 60, "filledAt": "2027-01-15T08:00:20.000Z"}
    report = _cycle(manager, _ctx({"1": 0.50}, now=NOW + 30, holdings={"1": Holding("yes", 60, 27.0, 27.3)}, fills=[fill]))
    assert report.blocked_held == 0 and report.markets == 1  # the 60 YES are the quoter's own
    report = _cycle(manager, _ctx({"1": 0.50}, now=NOW + 45, holdings={"1": Holding("yes", 160, 72.0, 72.3)}))
    assert report.blocked_held == 1  # 100 more than the quoter bought: someone else's


# ---------------------------------------------------------------------- 3. persisted kill / jump blocks


def test_blocks_survive_a_restart_only_with_the_flag(tmp_path):
    lines = [{"ts": _iso(NOW - 100), "event": "kill", "exchange_id": "1", "block_s": 3600},
             {"ts": _iso(NOW - 700), "event": "jump", "exchange_id": "3", "block_s": 600},  # expired
             {"ts": _iso(NOW - 10), "event": "jump", "exchange_id": "5", "block_s": 600}]
    (tmp_path / "quoter.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    fair = {"1": 0.50, "3": 0.50, "5": 0.50}
    exchange = FakeExchange()
    persisted = _manager(exchange, tmp_path, persist_blocks=True)
    assert persisted._blocked_until == {"1": pytest.approx(NOW + 3500), "5": pytest.approx(NOW + 590)}
    _cycle(persisted, _ctx(fair))
    assert {o.exchange_id for o in _placed(exchange)} == {"3"}
    plain = FakeExchange()
    _cycle(_manager(plain, tmp_path), _ctx(fair))
    assert {o.exchange_id for o in _placed(plain)} == {"1", "3", "5"}


def test_a_kill_written_by_one_process_blocks_the_next(tmp_path):
    first = _manager(FakeExchange(), tmp_path, clock=time.time)
    now = time.time()
    _cycle(first, _ctx({"3": 0.50}, now=now))
    _cycle(first, _ctx({"3": 0.50}, now=now + 15, markouts=[Markout("quoter", "3", f"f{i}", "o", now, 300, 0.5, -3.0) for i in range(3)]))
    second = _manager(FakeExchange(), tmp_path, clock=time.time, persist_blocks=True)
    assert second._blocked_until["3"] > time.time() + 3000


# ---------------------------------------------------------------------- 4. daily loss cap


def _loss_setup(tmp_path, cap=5.0):
    exchange = FakeExchange()
    manager = _manager(exchange, tmp_path, daily_loss_cap=cap)
    _cycle(manager, _ctx({"1": 0.50, "3": 0.50}))  # baseline 0 for the day
    yes_id = next(oid for oid, o in manager.orders.items() if o.exchange_id == "1" and o.side == "yes")
    fill = {"id": "f1", "orderId": yes_id, "exchangeId": "1", "side": "yes", "price": 0.455, "quantity": 100, "filledAt": "2027-01-15T08:00:10.000Z"}
    books = {"1": SigBook(yes_bid=0.39, yes_ask=0.41), "3": SigBook(yes_bid=0.45, yes_ask=0.55)}  # mid 0.40: 100 x (0.40 - 0.455) = -5.5
    return exchange, manager, fill, books


def test_daily_loss_cap_withdraws_everything_and_survives_a_restart(tmp_path, caplog):
    exchange, manager, fill, books = _loss_setup(tmp_path)
    with caplog.at_level(logging.WARNING, logger="quoter"):
        report = _cycle(manager, _ctx({"1": 0.50, "3": 0.50}, now=NOW + 15, books=books, fills=[fill]))
    assert report.loss_blocked and report.markets == 0 and manager.orders == {}
    assert {c["exchange_id"] for c in exchange.cancel_alls} == {"1", "3"}  # scoped cancels: never the taker's orders
    assert any(r.getMessage().startswith("Quoter daily loss cap hit") for r in caplog.records)
    events = [e["event"] for e in _journal(tmp_path)]
    assert events.count("day_open") == 1 and events.count("daily_loss_block") == 1
    # Restart, same UTC day: still blocked, the baseline is not reset.
    restarted_exchange = FakeExchange()
    restarted = _manager(restarted_exchange, tmp_path, clock=lambda: NOW + 30, daily_loss_cap=5.0)
    report = _cycle(restarted, _ctx({"1": 0.50, "3": 0.50}, now=NOW + 30, books=books))
    assert report.loss_blocked and _placed(restarted_exchange) == [] and [e["event"] for e in _journal(tmp_path)].count("day_open") == 1
    # Next UTC day: a new baseline, quoting again.
    report = _cycle(restarted, _ctx({"1": 0.50, "3": 0.50}, now=NOW + 86_400, books=books))
    assert not report.loss_blocked and report.markets >= 1 and _placed(restarted_exchange)


def test_daily_loss_under_the_cap_keeps_quoting(tmp_path):
    exchange, manager, fill, books = _loss_setup(tmp_path, cap=6.0)
    report = _cycle(manager, _ctx({"1": 0.50, "3": 0.50}, now=NOW + 15, books=books, fills=[fill]))
    assert not report.loss_blocked and report.markets == 2


def test_quoter_pnl_keeps_realized_of_positions_closed_before_a_restart(tmp_path):
    lines = [{"ts": _iso(NOW - 50), "event": "fill", "exchange_id": "1", "side": "yes", "price": 0.40, "quantity": 100},
             {"ts": _iso(NOW - 40), "event": "fill", "exchange_id": "1", "side": "no", "price": 0.55, "quantity": 100}]  # = sell YES @ 0.45
    (tmp_path / "quoter.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    manager = _manager(FakeExchange(), tmp_path, daily_loss_cap=5.0)
    assert manager.positions == {} and manager._quoter_pnl(_ctx({})) == pytest.approx(5.0)


# ---------------------------------------------------------------------- 5. resting notional cap


def test_max_resting_frac_caps_total_resting_notional(tmp_path):
    exchange = FakeExchange()
    report = _cycle(_manager(exchange, tmp_path, max_resting_frac=0.001), _ctx({"1": 0.50, "3": 0.50}))  # 100 SUSQ; a side 100 x 0.455
    assert len(_placed(exchange)) == 2 and report.blocked_resting == 2
    assert sum(o.quantity * o.price for o in _placed(exchange)) <= 100.0


# ---------------------------------------------------------------------- 6. virtual detail


def test_virtual_cross_rules():
    book = SigBook(yes_bid=0.45, yes_ask=0.455, last=0.46)
    assert Q.virtual_cross("yes", 0.455, book, None) == 0.455  # ask at our bid
    assert Q.virtual_cross("no", 0.545, SigBook(yes_bid=0.455, yes_ask=0.55), None) == 0.455  # YES bid at our ask (1 - 0.545)
    assert Q.virtual_cross("yes", 0.44, SigBook(0.40, 0.50, last=0.43), 0.45) == 0.43  # a new last trade through our bid
    assert Q.virtual_cross("yes", 0.44, SigBook(0.40, 0.50, last=0.43), 0.43) is None  # the same last trade as at placement
    assert Q.virtual_cross("yes", 0.44, SigBook(0.40, 0.50, last=0.45), None) is None and Q.virtual_cross("yes", 0.44, None, None) is None


def test_virtual_detail_counts_logs_and_fills_optimistically(tmp_path, caplog):
    exchange = FakeExchange()
    manager = _manager(exchange, tmp_path, dry_run_mode="true", virtual_detail=True)
    with caplog.at_level(logging.INFO, logger="quoter"):
        report = _cycle(manager, _ctx({"1": 0.50}))
        assert report.virtual_placed == 2 and report.placed == 0 and exchange.batches == []
        quotes = [r for r in caplog.records if r.getMessage() == "Quoter virtual quote"]
        assert len(quotes) == 2 and {(r.side, r.price, r.other_yes_bid, r.other_yes_ask, r.spread_ticks) for r in quotes} == {
            ("yes", 0.455, 0.45, 0.55, 20), ("no", 0.455, 0.45, 0.55, 20)}
        assert all(r.fair == 0.5 and r.edge == 0.015 for r in quotes)
        report = _cycle(manager, _ctx({"1": 0.50}, now=NOW + 15, books={"1": SigBook(yes_bid=0.45, yes_ask=0.455)}))  # ask down to our YES bid
    assert report.virtual_fills == 1 and manager.positions == {}
    fills = [r for r in caplog.records if r.getMessage().startswith("Quoter virtual fill")]
    assert len(fills) == 1 and fills[0].side == "yes" and "OPTIMISTIC" in fills[0].getMessage()
    assert [e for e in _journal(tmp_path) if e["event"] == "virtual_fill"][0]["crossed_at_yes"] == 0.455
    assert "fill" not in [e["event"] for e in _journal(tmp_path)]  # never quoter inventory


def test_virtual_detail_is_silent_for_a_live_quoter(tmp_path, caplog):
    exchange = FakeExchange()
    with caplog.at_level(logging.INFO, logger="quoter"):
        report = _cycle(_manager(exchange, tmp_path, virtual_detail=True), _ctx({"1": 0.50}))
    assert report.placed == 2 and report.virtual_placed == 0
    assert not [r for r in caplog.records if r.getMessage() == "Quoter virtual quote"]
