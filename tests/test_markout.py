"""markout.py and scripts/markout_report.py with fakes - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import markout as M
from strategy import OrderJournal, OrderStatus, Strategy
from test_cycle_resilience import SETTINGS, FakeExchange, FakeKalshi, FakePolymarket, MARKET_MAP

T0 = 1_800_000_000.0


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _fill(fill_id, order_id=1, side="yes", price=0.50, quantity=10, at=T0, exchange_id="1"):
    return {"id": fill_id, "orderId": order_id, "exchangeId": exchange_id, "side": side, "price": price,
            "quantity": quantity if side == "yes" else -quantity, "filledAt": _iso(at)}


def _lines(path: Path, event: str) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if json.loads(line)["event"] == event]


def test_markout_sign():
    assert M.markout_ticks(M.long_yes_direction("yes", "buy"), 0.50, 0.52) == pytest.approx(4.0)
    # Bought NO at YES 0.50 (NO cost 0.50); YES fair up to 0.52 = against us.
    assert M.markout_ticks(M.long_yes_direction("no", "buy"), 0.50, 0.52) == pytest.approx(-4.0)
    assert M.markout_ticks(M.long_yes_direction("yes", "sell"), 0.50, 0.48) == pytest.approx(4.0)
    assert M.long_yes_direction("yes", None) is None and M.markout_ticks(None, 0.5, 0.5) is None


def test_horizons_are_written_when_due(tmp_path):
    path = tmp_path / "markout.jsonl"
    tracker = M.MarkoutTracker(path, clock=lambda: T0)
    tracker.record("quoter", _fill(5), "buy", 0.50)
    assert tracker.update(T0 + 30, {"1": 0.51}) == []  # nothing due yet
    first = tracker.update(T0 + 61, {"1": 0.51})
    assert [(m.horizon_s, m.ticks) for m in first] == [(60, pytest.approx(2.0))]
    assert tracker.update(T0 + 301, {}) == []  # no fresh fair: wait (within grace)
    late = tracker.update(T0 + 301 + M.MARKOUT_GRACE_S, {})
    assert [(m.horizon_s, m.fair, m.ticks) for m in late] == [(300, None, None)]  # given up: fair null
    fill = _lines(path, "fill")[0]
    assert (fill["source"], fill["price_yes"], fill["quantity"], fill["fair_at_fill"]) == ("quoter", 0.50, 10, 0.50)
    assert [m["horizon_s"] for m in _lines(path, "markout")] == [60, 300]


def test_pending_horizons_survive_a_restart(tmp_path):
    path = tmp_path / "markout.jsonl"
    tracker = M.MarkoutTracker(path, clock=lambda: T0)
    tracker.record("taker", _fill(5, side="no", price=0.40), "buy", 0.40)
    tracker.update(T0 + 61, {"1": 0.40})
    restarted = M.MarkoutTracker(path, clock=lambda: T0 + 100)
    assert restarted.pending == 1
    written = restarted.update(T0 + 1801, {"1": 0.45})
    # Bought NO at a NO price of 0.40 (YES 0.60); fair YES 0.45 = NO worth 0.55: +30 ticks.
    assert [(m.horizon_s, m.ticks) for m in written] == [(300, pytest.approx(30.0)), (1800, pytest.approx(30.0))]


class _FillsExchange:
    def __init__(self, pages, dry_run=False):
        self.pages, self.dry_run, self.calls = list(pages), dry_run, 0

    async def get_tournament_fills(self, slug, cursor=None, limit=200):
        self.calls += 1
        return self.pages.pop(0) if self.pages else {"data": [], "pagination": {"hasMore": False}}


def test_fill_feed_returns_new_fills_once_oldest_first():
    old, a, b = _fill(1, at=T0 - 10), _fill(2, at=T0 + 5), _fill(3, at=T0 + 9)
    exchange = _FillsExchange([
        {"data": [b, a, old], "pagination": {"hasMore": False}},
        {"data": [b, a], "pagination": {"hasMore": False}},
    ])
    feed = M.FillFeed(exchange, "t", clock=lambda: T0)
    assert [f["id"] for f in asyncio.run(feed.read())] == [2, 3]  # the pre-start fill is skipped
    assert asyncio.run(feed.read()) == []  # nothing new


def test_fill_feed_sends_nothing_in_dry_run():
    exchange = _FillsExchange([], dry_run=True)
    assert asyncio.run(M.FillFeed(exchange, "t").read()) == [] and exchange.calls == 0


class _StaticFeed:
    def __init__(self, fills):
        self.fills = fills

    async def read(self):
        fills, self.fills = self.fills, []
        return fills


def test_cycle_records_taker_fills_with_their_action(tmp_path):
    journal = OrderJournal(tmp_path / "j.jsonl")
    journal.record("k-1", OrderStatus.ACCEPTED, result={"orderId": 77, "action": "buy", "side": "yes"})
    tracker = M.MarkoutTracker(tmp_path / "m.jsonl")
    feed = _StaticFeed([_fill(9, order_id=77), _fill(10, order_id=999)])
    taker = Strategy(SETTINGS, FakeExchange(), FakePolymarket(), FakeKalshi(), MARKET_MAP, journal, markout=tracker, fill_feed=feed)
    report = asyncio.run(taker.run_cycle())
    assert report.fills_new == 2
    fills = _lines(tmp_path / "m.jsonl", "fill")
    assert [(f["source"], f["action"]) for f in fills] == [("taker", "buy"), ("taker", None)]  # 999: unknown order
    assert fills[0]["fair_at_fill"] == pytest.approx(0.60)  # this cycle's oracle fair for exchange 1


def test_markout_report(tmp_path):
    spec = importlib.util.spec_from_file_location("markout_report", Path(__file__).resolve().parent.parent / "scripts" / "markout_report.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    lines = [
        json.dumps({"event": "fill", "source": "quoter"}),
        json.dumps({"event": "markout", "source": "quoter", "exchange_id": "1", "horizon_s": 300, "markout_ticks": -3.0}),
        json.dumps({"event": "markout", "source": "quoter", "exchange_id": "1", "horizon_s": 300, "markout_ticks": 1.0}),
        json.dumps({"event": "markout", "source": "quoter", "exchange_id": "2", "horizon_s": 300, "markout_ticks": None}),
        json.dumps({"event": "markout", "source": "taker", "exchange_id": "1", "horizon_s": 60, "markout_ticks": 2.0}),
    ]
    rows = report.summarize(lines)
    assert rows == [
        {"source": "quoter", "horizon_s": 300, "n": 2, "mean_ticks": -1.0, "negative_share": 0.5, "no_fair": 1, "corrected": 0},
        {"source": "taker", "horizon_s": 60, "n": 1, "mean_ticks": 2.0, "negative_share": 0.0, "no_fair": 0, "corrected": 0},
    ]
    assert [r["exchange_id"] for r in report.summarize(lines, by_exchange=True)] == ["1", "2", "1"]


def test_ladder_uses_the_cycle_fills_instead_of_reading_again(tmp_path):
    from dataclasses import replace

    from test_ladder import FakeExchange as LadderExchange, _ctx, _manager

    exchange = LadderExchange()

    async def no_read(*args, **kwargs):
        raise AssertionError("ladder read fills although strategy.py already did")

    exchange.get_tournament_fills = no_read
    manager = _manager(exchange, tmp_path)
    asyncio.run(manager.run_cycle(replace(_ctx({"1": 0.92}), fills=[])))
