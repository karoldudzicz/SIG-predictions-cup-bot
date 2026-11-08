"""Deep ladder (ladder.py) with fakes - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path

import pytest

import ladder as L
from config import Settings
from exchange_api import ExchangeClient, PRICE_TICK
from market_mapper import MarketMap, MarketMapping, Race
from oracle_api import OracleQuote
from risk import RiskLimits, race_worst_case_loss
from strategy import Holding, PlannedOrder, SigBook, apply_exposure_caps, compute_fair_price
from test_exchange_http import _client, _FakeSession

BANKROLL = 100_000.0
NOW = 1_800_000_000.0
LIMITS = RiskLimits()


def _map(races: dict[str, dict[str, str]]) -> MarketMap:
    return MarketMap(
        {
            eid: MarketMapping(eid, eid, f"Will the {party} win the {st} Senate?", Race("senate", st), party, f"tok-{eid}", None)
            for st, parties in races.items()
            for party, eid in parties.items()
        }
    )


MAP = _map({"AZ": {"D": "1", "R": "2"}, "GA": {"D": "3", "R": "4"}, "NV": {"D": "5", "R": "6"}})


def _quote(bid, ask, source="polymarket", age=0.0):
    return OracleQuote(source, "ref", bid, ask, 100.0, 100.0, None, time.time() - age)


class FakeExchange:
    dry_run = False
    halted = False

    def __init__(self, fills=None):
        self.batches: list[list] = []
        self.cancel_alls: list[dict] = []
        self.cancels: list[str] = []
        self.fills_pages = fills or []
        self._next_id = 100

    async def place_batch(self, orders, key, write_rank=1):
        self.batches.append(list(orders))
        results = []
        for i, order in enumerate(orders):
            self._next_id += 1
            results.append({"index": i, "ok": True, "status": 200, "data": {"orderId": self._next_id, "remainingQuantity": order.quantity, "open": True}})
        return results

    async def cancel_all_orders(self, **scope):
        self.cancel_alls.append(scope)
        return {"cancelled": 1, "errors": []}

    async def cancel_order(self, order_id):
        self.cancels.append(order_id)
        return {"orderId": order_id}

    async def get_tournament_fills(self, slug, cursor=None, limit=200):
        data = self.fills_pages.pop(0) if self.fills_pages else []
        return {"data": data, "pagination": {"hasMore": False, "nextCursor": None}}


class FakeOracle:
    def __init__(self, quotes):
        self.quotes = quotes  # token -> OracleQuote

    async def get_quotes(self, tokens):
        return {t: self.quotes[t] for t in tokens if t in self.quotes}


def _manager(exchange, tmp_path, pm_quotes=None, holes=None):
    holes_path = tmp_path / "holes.csv"
    if holes:
        holes_path.write_text("exchange_id\n" + "".join(f"{eid}\n" for eid in holes), encoding="utf-8")
    return L.LadderManager(
        exchange, FakeOracle(pm_quotes or {}), FakeOracle({}), compute_fair_price,
        journal=L.LadderJournal(tmp_path / "ladder.jsonl"), holes_path=holes_path, clock=lambda: NOW - 1000,
    )


def _ctx(fair, holdings=None, books=None, open_orders=None, bankroll=BANKROLL, quotes=None, now=NOW):
    return L.LadderContext(
        now=now, bankroll=bankroll, tournament_id="tid", tournament_slug="t", market_map=MAP, fair=fair,
        quotes=quotes or {eid: (_quote(fair[eid] - 0.01, fair[eid] + 0.01), None) for eid in fair},
        books=books or {eid: SigBook(yes_bid=0.40, yes_ask=0.45) for eid in fair},
        holdings=holdings or {}, open_orders=open_orders or [], limits=LIMITS, max_quote_age_s=10.0,
    )


# ---------------------------------------------------------------------- levels and band


def test_level_prices_round_down_to_the_tick():
    levels = L.ladder_levels(0.9137, u=PRICE_TICK)
    assert [(lv.index, lv.price, lv.quantity) for lv in levels] == [(0, 0.84, 300), (1, 0.76, 500), (2, 0.61, 800), (3, 0.41, 1200)]
    assert sum(lv.price * lv.quantity for lv in levels) <= 2000


def test_levels_below_the_minimum_price_are_skipped():
    assert [lv.price for lv in L.ladder_levels(0.505, u=PRICE_TICK)] == [0.435, 0.355, 0.205, 0.005]
    assert [lv.price for lv in L.ladder_levels(0.50, u=PRICE_TICK)] == [0.43, 0.35, 0.2]  # 0.50 - 0.50 = 0 -> skipped


def test_cost_cap_trims_sizes():
    levels = L.ladder_levels(0.9137, u=PRICE_TICK, max_cost=500)
    assert [(lv.price, lv.quantity) for lv in levels] == [(0.84, 300), (0.76, 326)]


def test_favourite_side_only():
    assert L.favourite_side(0.70) == ("yes", 0.70)
    side, fair_side = L.favourite_side(0.30)
    assert side == "no" and fair_side == pytest.approx(0.70)


def test_nothing_closer_than_u_plus_min_extra():
    rng = random.Random(7)
    for _ in range(500):
        fair_side, u = rng.uniform(0.5, 0.99), rng.uniform(0.005, 0.2)
        for level in L.ladder_levels(fair_side, u):
            assert fair_side - level.price >= u + L.LADDER_MIN_EXTRA_DISTANCE - 1e-9
    # A wide band removes the shallow level outright.
    assert [lv.index for lv in L.ladder_levels(0.80, u=0.06)] == [1, 2, 3]


def test_uncertainty_band():
    pm, ks = _quote(0.60, 0.62), _quote(0.55, 0.65, source="kalshi")
    assert L.uncertainty_band(pm, ks, 10) == pytest.approx(0.01)  # tightest spread 0.02 / 2
    assert L.uncertainty_band(_quote(0.60, 0.62), _quote(0.68, 0.70, source="kalshi"), 10) == pytest.approx(0.04)  # |0.61-0.69|/2
    assert L.uncertainty_band(None, None, 10) == PRICE_TICK


# ---------------------------------------------------------------------- budgets


def test_hold_book_excludes_ladder_inventory(tmp_path):
    manager = _manager(FakeExchange(), tmp_path)
    manager.positions["1"] = L.LadderPosition("1", "yes", bought=1000, cost=450.0)
    held = {"1": Holding("yes", 1500, 750.0, 675.0), "3": Holding("no", 200, 100.0, 90.0)}
    hold = manager.hold_book(held)
    assert hold["1"].shares == 500 and hold["1"].cost_basis == pytest.approx(225.0)
    assert hold["3"] == held["3"]


def test_ladder_cost_does_not_use_the_hold_book_gross_cap():
    """29,900 of ladder inventory elsewhere: the taker's gross cap only sees the hold book."""
    ladder_holding = {"3": Holding("yes", 50_000, 29_900.0, 29_900.0)}
    order = [PlannedOrder("increase", "1", "yes", 1000, 0.45, "edge")]
    kwargs = dict(market_map=MAP, fair={"1": 0.60}, margins={"1": 0.01}, bankroll=BANKROLL, limits=LIMITS)
    hold_book = apply_exposure_caps(order, {}, race_holdings=ladder_holding, **kwargs)
    assert hold_book[0].quantity == 1000
    if_counted = apply_exposure_caps(order, ladder_holding, **kwargs)
    assert if_counted[0].quantity < 1000  # what would happen if ladder cost hit the gross cap


def test_ladder_budget_is_its_own_fraction(tmp_path):
    exchange = FakeExchange()
    manager = _manager(exchange, tmp_path)
    big_hold_book = {"6": Holding("no", 40_000, 20_000.0, 20_000.0)}  # taker book, other race
    asyncio.run(manager.run_cycle(_ctx({"1": 0.92, "3": 0.91}, holdings=big_hold_book, bankroll=4_000.0)))
    placed_cost = sum(o.price * o.quantity for batch in exchange.batches for o in batch)
    assert 0 < placed_cost <= L.LADDER_BOOK_FRAC * 4_000.0 + 1e-6  # 30% of bankroll, regardless of the hold book


# ---------------------------------------------------------------------- race cap


def test_resting_ladder_orders_count_in_the_race_cap(tmp_path):
    manager = _manager(FakeExchange(), tmp_path)
    asyncio.run(manager.run_cycle(_ctx({"1": 0.92})))  # resting D YES ladder in AZ
    legs = manager.resting_race_legs(MAP)
    assert race_worst_case_loss(["D", "R"], legs["senate:AZ"]) <= LIMITS.max_race_worst_loss_fraction * BANKROLL + 1e-6
    order = [PlannedOrder("increase", "1", "yes", 2000, 0.85, "edge")]
    kwargs = dict(market_map=MAP, fair={"1": 0.92}, margins={"1": 0.01}, bankroll=BANKROLL, limits=LIMITS)
    with_ladder = apply_exposure_caps(order, {}, extra_race_legs=legs, **kwargs)
    without = apply_exposure_caps(order, {}, **kwargs)
    assert sum(o.quantity for o in with_ladder) < sum(o.quantity for o in without)


# ---------------------------------------------------------------------- exits


def test_exit_classification_and_noise_price():
    assert L.classify_exit(0.80, 0.79, 0.65, u=0.02) == "noise"
    assert L.classify_exit(0.80, 0.70, 0.65, u=0.02) == "information"  # moved 0.10 > 0.15 / 2 toward the fill
    assert L.classify_exit(0.80, 0.76, 0.65, u=0.02) == "drift"
    assert L.noise_exit_price(0.80, 0.02, anchor=0.84, steps=0) == 0.84
    assert L.noise_exit_price(0.80, 0.02, anchor=0.84, steps=3) == 0.825
    assert L.noise_exit_price(0.80, 0.02, anchor=0.84, steps=50) == 0.78  # never below fair - u


def _filled_manager(tmp_path, fair_after, best_ask=0.90):
    """A manager with one resting D YES ladder order in AZ that then fills at 0.84."""
    exchange = FakeExchange()
    manager = _manager(exchange, tmp_path, pm_quotes={"tok-1": _quote(fair_after - 0.01, fair_after + 0.01)})
    asyncio.run(manager.run_cycle(_ctx({"1": 0.92})))
    level0 = next(o for o in manager.orders.values() if o.level == 0)
    exchange.fills_pages = [[{"id": 1, "orderId": int(level0.order_id), "exchangeId": "1", "price": level0.price, "quantity": 300, "side": "yes", "filledAt": "2027-01-15T08:00:00Z"}]]
    manager._fills_since = 0
    holdings = {"1": Holding("yes", 300, 252.0, 252.0)}
    books = {"1": SigBook(yes_bid=0.80, yes_ask=best_ask)}
    report = asyncio.run(manager.run_cycle(_ctx({"1": fair_after}, holdings=holdings, books=books, open_orders=[])))
    return exchange, manager, report


def test_noise_fill_gets_a_passive_exit_under_the_best_ask(tmp_path):
    exchange, manager, report = _filled_manager(tmp_path, fair_after=0.915, best_ask=0.95)
    assert report.fills == 1 and manager.positions["1"].mode == "noise"
    exit_order = exchange.batches[-1][0]
    # One tick under the best ask (0.95 - 0.005), above the floor fair - u = 0.915 - 0.01.
    assert (exit_order.action, exit_order.side, exit_order.quantity, exit_order.price) == ("sell", "yes", 300, 0.945)
    assert "1" not in manager._blocked_until


def test_noise_exit_never_below_fair_minus_u(tmp_path):
    exchange, manager, _ = _filled_manager(tmp_path, fair_after=0.915, best_ask=0.90)
    # Tick under the ask would be 0.895, but fair - u = 0.905 is the floor.
    assert exchange.batches[-1][0].price == 0.905


def test_noise_exit_steps_down_a_tick_every_ten_minutes(tmp_path):
    exchange, manager, _ = _filled_manager(tmp_path, fair_after=0.915, best_ask=0.95)
    exit_id = manager.positions["1"].exit_order_id
    later = NOW + 2 * L.LADDER_EXIT_STEP_S + 1  # two steps later
    open_orders = [{"id": int(exit_id), "exchangeId": "1", "quantity": 300}] + [
        {"id": int(o.order_id), "exchangeId": o.exchange_id, "quantity": o.quantity} for o in manager.orders.values()
    ]
    asyncio.run(manager.run_cycle(_ctx({"1": 0.915}, holdings={"1": Holding("yes", 300, 252.0, 252.0)}, books={"1": SigBook(0.80, 0.95)}, open_orders=open_orders, now=later)))
    assert exchange.cancels == [exit_id]  # old exit cancelled ...
    assert exchange.batches[-1][0].price == 0.935  # ... re-posted two ticks lower


def test_information_fill_sells_at_the_bid_and_blocks_the_market(tmp_path):
    exchange, manager, report = _filled_manager(tmp_path, fair_after=0.86)  # fell 0.06 > (0.92-0.84)/2
    assert manager.positions["1"].mode == "information"
    assert {"exchange_id": "1", "tournament_id": "tid"} in exchange.cancel_alls
    exit_order = exchange.batches[-1][0]
    assert (exit_order.action, exit_order.price) == ("sell", 0.80)  # best bid
    assert manager._blocked_until["1"] == pytest.approx(NOW + L.LADDER_INFO_BLOCK_S)


def test_stale_oracles_on_fill_hold_and_add_nothing(tmp_path):
    exchange = FakeExchange()
    manager = _manager(exchange, tmp_path)  # no oracle quotes at all on refetch
    asyncio.run(manager.run_cycle(_ctx({"1": 0.92})))
    level0 = next(o for o in manager.orders.values() if o.level == 0)
    exchange.fills_pages = [[{"id": 1, "orderId": int(level0.order_id), "exchangeId": "1", "price": level0.price, "quantity": 300, "side": "yes", "filledAt": "2027-01-15T08:00:00Z"}]]
    manager._fills_since = 0
    batches_before = len(exchange.batches)
    asyncio.run(manager.run_cycle(_ctx({"1": 0.92}, holdings={"1": Holding("yes", 300, 252.0, 252.0)}, open_orders=[], quotes={"1": (None, None)})))
    assert manager.positions["1"].mode == "hold"
    assert len(exchange.batches) == batches_before  # no exit, no new levels


# ---------------------------------------------------------------------- circuit breaker


def test_breaker_function():
    fills = [(NOW - 10, "senate:AZ"), (NOW - 20, "senate:GA"), (NOW - 30, "senate:NV")]
    assert L.breaker_tripped(fills, NOW)
    assert not L.breaker_tripped(fills[:2], NOW)
    assert not L.breaker_tripped([(NOW - 100, "senate:AZ")] + fills[1:], NOW)  # one outside 60 s


def test_fills_in_three_races_cancel_everything_and_pause(tmp_path):
    exchange = FakeExchange()
    manager = _manager(exchange, tmp_path)
    asyncio.run(manager.run_cycle(_ctx({"1": 0.92, "3": 0.91, "5": 0.93})))
    first_orders = {eid: next(o for o in manager.orders.values() if o.exchange_id == eid and o.level == 3) for eid in ("1", "3", "5")}
    exchange.fills_pages = [[
        {"id": i, "orderId": int(o.order_id), "exchangeId": eid, "price": o.price, "quantity": 10, "side": "yes", "filledAt": "2027-01-15T08:00:00Z"}
        for i, (eid, o) in enumerate(first_orders.items())
    ]]
    manager._fills_since = 0
    report = asyncio.run(manager.run_cycle(_ctx({"1": 0.92, "3": 0.91, "5": 0.93}, open_orders=[], now=NOW + 5)))
    assert {"tournament_id": "tid"} in exchange.cancel_alls
    assert report.paused and manager.orders == {}
    assert manager._paused_until == pytest.approx(NOW + 5 + L.LADDER_BREAKER_PAUSE_S)


# ---------------------------------------------------------------------- DRY_RUN


def test_dry_run_places_nothing(tmp_path):
    session = _FakeSession(delays=[])
    client = _client(session, dry_run=True)
    manager = _manager(client, tmp_path)
    report = asyncio.run(manager.run_cycle(_ctx({"1": 0.92, "3": 0.91})))
    assert session.calls == []  # not a single request: no reads, no orders, no cancels
    assert report.placed == 0
    journal = (tmp_path / "ladder.jsonl").read_text(encoding="utf-8")
    assert '"event": "intent"' in journal and '"event": "placement"' in journal  # virtual, for the dry-run log
    # Second cycle with the same fair: virtual orders are resting, nothing is re-quoted.
    lines = journal.count("\n")
    asyncio.run(manager.run_cycle(_ctx({"1": 0.92, "3": 0.91})))
    assert (tmp_path / "ladder.jsonl").read_text(encoding="utf-8").count("\n") == lines
    assert session.calls == []


# ---------------------------------------------------------------------- write bucket


def test_write_bucket_rate_and_cancel_priority():
    from exchange_api import _WriteBucket

    async def scenario():
        bucket = _WriteBucket(per_minute=600, burst=2)  # 10/s, 2 stored
        started = time.monotonic()
        await bucket.acquire()
        await bucket.acquire()
        assert time.monotonic() - started < 0.05  # the burst is immediate
        order: list[str] = []

        async def take(name, priority):
            await bucket.acquire(rank=0 if priority else 1)
            order.append(name)

        normal = asyncio.create_task(take("order", False))
        await asyncio.sleep(0)
        cancel = asyncio.create_task(take("cancel", True))
        await asyncio.gather(normal, cancel)
        assert order == ["cancel", "order"]  # the cancel arrived later but went first
        assert time.monotonic() - started >= 0.15  # two more tokens at 10/s

    asyncio.run(scenario())


def test_halted_cancels_skip_the_bucket():
    session = _FakeSession(delays=[0.0], body={"cancelled": 0, "errors": []})
    client = _client(session, dry_run=False)
    client._write_bucket._tokens = 0.0  # bucket empty
    client._write_bucket._rate = 1e-6  # and practically never refilling
    client.halt("test")

    async def scenario():
        return await asyncio.wait_for(client.cancel_all_orders(), timeout=1)

    assert asyncio.run(scenario()) == {"cancelled": 0, "errors": []}
