"""Quoter (quoter.py) with fakes - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import replace

import pytest

import quoter as Q
from exchange_api import PRICE_TICK, WRITE_RANK_QUOTER
from markout import Markout
from oracle_api import OracleQuote
from risk import Direction, RaceLeg, RiskLimits
from strategy import Holding, SigBook
from test_exchange_http import _client, _FakeSession
from test_ladder import MAP  # AZ: D "1", R "2"; GA: D "3", R "4"; NV: D "5", R "6"

BANKROLL = 100_000.0
NOW = 1_800_000_000.0
LIMITS = RiskLimits()
TOTAL_CAP = LIMITS.max_total_gross_exposure_fraction * BANKROLL
D_CAP = LIMITS.max_directional_fraction * BANKROLL


def _quote(mid, half_spread=0.005, age=0.0):
    return OracleQuote("polymarket", "ref", mid - half_spread, mid + half_spread, 100.0, 100.0, None, time.time() - age)


class FakeExchange:
    dry_run = False
    halted = False

    def __init__(self):
        self.batches: list[tuple[list, int]] = []
        self.cancel_alls: list[dict] = []
        self.book_reads: list[str] = []
        self._next_id = 500

    async def place_batch(self, orders, key, write_rank=1):
        self.batches.append((list(orders), write_rank))
        results = []
        for i, order in enumerate(orders):
            self._next_id += 1
            results.append({"index": i, "ok": True, "status": 200, "data": {"orderId": self._next_id, "remainingQuantity": order.quantity, "open": True}})
        return results

    async def cancel_all_orders(self, **scope):
        self.cancel_alls.append(scope)
        return {"cancelled": 2, "errors": []}

    def own_orders(self, exchange_id):
        return []

    async def get_orderbook(self, exchange_id, depth=20):
        self.book_reads.append(exchange_id)
        return {"bids": [], "asks": []}


def _manager(exchange, tmp_path):
    return Q.QuoterManager(exchange, journal=Q.QuoterJournal(tmp_path / "quoter.jsonl"), clock=lambda: NOW - 1000)


def _ctx(fair, now=NOW, books=None, holdings=None, **extra):
    return Q.QuoterContext(
        now=now, bankroll=BANKROLL, tournament_id="tid", market_map=MAP, fair=fair,
        quotes={eid: (_quote(f), None) for eid, f in fair.items()},
        books=books or {eid: SigBook(yes_bid=0.45, yes_ask=0.55) for eid in fair},
        holdings=holdings or {}, open_orders=[], limits=LIMITS, max_quote_age_s=10.0, **extra,
    )


def _placed(exchange):
    return [order for orders, _ in exchange.batches for order in orders]


def _open_orders(manager):
    """What the next open-order read would return for the quoter's resting quotes."""
    return [{"id": oid, "quantity": o.quantity} for oid, o in manager.orders.items()]


# ---------------------------------------------------------------------- prices


def test_quotes_improve_the_best_other_bid_but_keep_the_edge():
    plan = Q.plan_quotes(0.50, 0.005, other_yes_bid=0.45, other_no_bid=0.46, party="D", net_delta=0)
    assert (plan.yes, plan.no, plan.edge) == (0.455, 0.465, 0.015)  # best other + tick, below fair - edge
    tight = Q.plan_quotes(0.50, 0.005, other_yes_bid=0.49, other_no_bid=None, party="D", net_delta=0)
    assert (tight.yes, tight.no) == (0.485, 0.485)  # capped at fair - edge; no other NO bid -> fair - edge


def test_edge_is_never_below_u_plus_one_cent():
    plan = Q.plan_quotes(0.50, 0.03, None, None, "D", 0, min_edge=0.01)
    assert plan.edge == pytest.approx(0.04) and plan.yes == 0.46 and plan.no == 0.46


def test_skew_comes_off_the_inventory_increasing_side():
    # Long 200 D: on a D market YES adds D -> YES bid 2 * 0.005 lower; NO untouched.
    d_market = Q.plan_quotes(0.50, 0.005, None, None, "D", net_delta=200)
    assert (d_market.yes, d_market.no, d_market.skew) == (0.475, 0.485, 0.01)
    # Same race delta on the R market: NO on R adds D -> the NO bid is skewed.
    r_market = Q.plan_quotes(0.50, 0.005, None, None, "R", net_delta=200)
    assert (r_market.yes, r_market.no) == (0.485, 0.475)
    # Long R (delta < 0): on the D market NO adds R.
    assert Q.plan_quotes(0.50, 0.005, None, None, "D", net_delta=-100).no == 0.48


def test_race_delta_counts_d_yes_and_r_no_alike():
    holdings = {"1": Holding("yes", 100, 50.0), "2": Holding("no", 50, 25.0), "3": Holding("yes", 999, 1.0)}
    assert Q.race_delta(MAP, "senate:AZ", holdings) == 150  # D YES 100 + R NO 50; GA ignored
    assert Q.race_delta(MAP, "senate:AZ", {"1": Holding("no", 30, 9.0), "2": Holding("yes", 20, 9.0)}) == -50


def test_bid_sum_never_above_one_minus_two_ticks(monkeypatch):
    rng = random.Random(3)
    for _ in range(2000):
        fair = rng.uniform(0.05, 0.95)
        plan = Q.plan_quotes(
            fair, rng.uniform(0.005, 0.04), rng.choice([None, rng.uniform(0.0, 1.0)]), rng.choice([None, rng.uniform(0.0, 1.0)]),
            rng.choice("DR"), rng.uniform(-299, 299),
        )
        if plan.yes is not None and plan.no is not None:
            assert plan.yes + plan.no <= 1 - 2 * PRICE_TICK + 1e-9
    # With no edge at all the guard itself has to act.
    monkeypatch.setattr(Q, "QUOTER_EDGE_OVER_U", 0.0)
    plan = Q.plan_quotes(0.50, 0.0, None, None, "D", 0, min_edge=0.0)
    assert plan.yes + plan.no == pytest.approx(1 - 2 * PRICE_TICK)


def test_delta_cap_drops_only_the_increasing_side():
    plan = Q.plan_quotes(0.50, 0.005, None, None, "D", net_delta=250, size=100, delta_cap=300)
    assert plan.yes is None and plan.no is not None and plan.delta_blocked == ("yes",)
    assert Q.plan_quotes(0.50, 0.005, None, None, "D", net_delta=0, size=100, delta_cap=300).delta_blocked == ()


def test_inventory_nets_like_positions():
    pos = Q.QuoterPosition("1")
    pos.apply("yes", 100, 0.45)
    pos.apply("no", 40, 0.50)  # = sell 40 YES at 0.50
    assert pos.shares == 60 and pos.realized == pytest.approx(40 * 0.05)
    assert pos.open_cost == pytest.approx(60 * 0.45)
    pos.apply("no", 100, 0.50)  # flips to 40 NO at a NO cost of 0.50
    assert pos.side == "no" and pos.shares == -40 and pos.open_cost == pytest.approx(40 * 0.50)


# ---------------------------------------------------------------------- manager


def test_quotes_both_sides_in_one_batch_at_quoter_rank(tmp_path):
    exchange = FakeExchange()
    manager = _manager(exchange, tmp_path)
    report = asyncio.run(manager.run_cycle(_ctx({"1": 0.50, "3": 0.60})))
    placed = _placed(exchange)
    assert report.markets == 2 and report.placed == 4 and len(exchange.batches) == 1
    assert exchange.batches[0][1] == WRITE_RANK_QUOTER
    assert {(o.exchange_id, o.side, o.price, o.action) for o in placed} >= {("1", "yes", 0.455, "buy"), ("1", "no", 0.455, "buy")}
    assert all(o.quantity == Q.QUOTER_SIZE for o in placed)


def test_unchanged_targets_are_not_requoted_but_a_tick_or_expiry_is(tmp_path):
    exchange = FakeExchange()
    manager = _manager(exchange, tmp_path)
    asyncio.run(manager.run_cycle(_ctx({"1": 0.50})))
    open_orders = _open_orders(manager)
    asyncio.run(manager.run_cycle(replace(_ctx({"1": 0.50}, now=NOW + 15), open_orders=open_orders)))
    assert exchange.cancel_alls == [] and len(exchange.batches) == 1  # nothing moved: no write
    # Best other YES bid up one tick -> the YES target moves -> scoped cancel + re-post.
    books = {"1": SigBook(yes_bid=0.455, yes_ask=0.55)}
    report = asyncio.run(manager.run_cycle(replace(_ctx({"1": 0.50}, now=NOW + 30, books=books), open_orders=_open_orders(manager))))
    assert report.requotes == 1 and exchange.cancel_alls == [{"exchange_id": "1", "tournament_id": "tid"}]
    assert {(o.side, o.price) for o in exchange.batches[-1][0]} == {("yes", 0.46), ("no", 0.455)}
    # Near expiry: requoted although nothing moved.
    late = NOW + 30 + Q.QUOTER_TTL_S - Q.QUOTER_EXPIRY_REQUOTE_S + 1
    report = asyncio.run(manager.run_cycle(replace(_ctx({"1": 0.50}, now=late, books=books), open_orders=_open_orders(manager))))
    assert report.requotes == 1


def test_tail_guard_withdraws_quotes(tmp_path):
    exchange = FakeExchange()
    manager = _manager(exchange, tmp_path)
    asyncio.run(manager.run_cycle(_ctx({"1": 0.50})))
    # Outside the jump window, so this is the tail guard and not the jump guard.
    later = NOW + Q.QUOTER_JUMP_WINDOW_S + 15
    report = asyncio.run(manager.run_cycle(replace(_ctx({"1": 0.97}, now=later), open_orders=_open_orders(manager))))
    assert report.markets == 0 and report.blocked_tail == 1 and report.blocked_jump == 0
    assert exchange.cancel_alls == [{"exchange_id": "1", "tournament_id": "tid"}] and manager.orders == {}
    assert Q.in_tails(0.04) and Q.in_tails(0.96) and not Q.in_tails(0.05) and not Q.in_tails(0.5)


def test_jump_guard_cancels_and_blocks(tmp_path):
    exchange = FakeExchange()
    manager = _manager(exchange, tmp_path)
    asyncio.run(manager.run_cycle(_ctx({"1": 0.50})))
    report = asyncio.run(manager.run_cycle(replace(_ctx({"1": 0.53}, now=NOW + 15), open_orders=_open_orders(manager))))
    assert report.blocked_jump == 1 and report.markets == 0 and manager.orders == {}
    # Still blocked while the fair is calm again ...
    calm = asyncio.run(manager.run_cycle(_ctx({"1": 0.53}, now=NOW + 15 + Q.QUOTER_JUMP_BLOCK_S - 1)))
    assert calm.markets == 0
    # ... and quoted again after QUOTER_JUMP_BLOCK_S.
    assert asyncio.run(manager.run_cycle(_ctx({"1": 0.53}, now=NOW + 16 + Q.QUOTER_JUMP_BLOCK_S))).markets == 1
    assert not Q.jumped([(NOW - 40, 0.40)], NOW, 0.50)  # outside the window


def test_stale_or_uncertain_oracle_means_no_quotes(tmp_path):
    exchange = FakeExchange()
    manager = _manager(exchange, tmp_path)
    ctx = replace(_ctx({"1": 0.50, "3": 0.50}), quotes={"1": (_quote(0.50, half_spread=0.05), None), "3": (_quote(0.50), None)})
    report = asyncio.run(manager.run_cycle(ctx))
    assert {o.exchange_id for o in _placed(exchange)} == {"3"} and report.blocked_oracle == 1  # u = 0.05 > 0.04


def _bad_markouts(eid, ticks):
    return [Markout("quoter", eid, f"f{i}", 1, NOW, 300, 0.4, t) for i, t in enumerate(ticks)]


def test_three_adverse_markouts_kill_the_market(tmp_path, caplog):
    exchange = FakeExchange()
    manager = _manager(exchange, tmp_path)
    asyncio.run(manager.run_cycle(_ctx({"1": 0.50})))
    ctx = replace(_ctx({"1": 0.50}, now=NOW + 15, markouts=_bad_markouts("1", [-3, -2.5, -4])), open_orders=_open_orders(manager))
    report = asyncio.run(manager.run_cycle(ctx))
    assert report.killed == 1 and report.markets == 0 and manager.orders == {}
    assert any(r.getMessage() == "Quoter market killed: consecutive adverse markouts" and r.levelno == 30 for r in caplog.records)
    assert asyncio.run(manager.run_cycle(_ctx({"1": 0.50}, now=NOW + 15 + Q.QUOTER_KILL_BLOCK_S - 1))).markets == 0


def test_a_good_markout_resets_the_streak_and_other_sources_do_not_count(tmp_path):
    manager = _manager(FakeExchange(), tmp_path)
    markouts = _bad_markouts("1", [-3, -3, 0.0, -3, -3]) + [Markout("taker", "1", "t", 1, NOW, 300, 0.4, -9)] + [Markout("quoter", "1", "x", 1, NOW, 60, 0.4, -9)]
    report = asyncio.run(manager.run_cycle(_ctx({"1": 0.50}, markouts=markouts)))
    assert report.killed == 0 and report.markets == 1


def test_fills_build_quoter_inventory(tmp_path):
    exchange = FakeExchange()
    manager = _manager(exchange, tmp_path)
    asyncio.run(manager.run_cycle(_ctx({"1": 0.50})))
    yes_id = next(oid for oid, o in manager.orders.items() if o.side == "yes")
    fill = {"id": 1, "orderId": int(yes_id), "exchangeId": "1", "side": "yes", "price": 0.455, "quantity": 100, "filledAt": "2027-01-15T08:00:00.000Z"}
    report = asyncio.run(manager.run_cycle(replace(_ctx({"1": 0.50}, now=NOW + 15, fills=[fill]), open_orders=[])))
    assert report.fills == 1 and manager.positions["1"].shares == 100
    assert manager.order_action(yes_id) == "buy" and "1" in manager.markets_in_use()
    hold = manager.hold_book({"1": Holding("yes", 150, 75.0, 68.0)})
    assert hold["1"].shares == 50  # the taker sees only its own 50
    assert Q.QuoterJournal(tmp_path / "quoter.jsonl").replay()["1"].shares == 100  # survives a restart


# ---------------------------------------------------------------------- limits


def test_global_cap_blocks_new_quotes(tmp_path):
    exchange = FakeExchange()
    report = asyncio.run(_manager(exchange, tmp_path).run_cycle(_ctx({"1": 0.50}, other_gross=TOTAL_CAP)))
    assert _placed(exchange) == [] and report.blocked_global == 2


def test_directional_cap_skips_the_side_that_adds_d(tmp_path):
    exchange = FakeExchange()
    ctx = _ctx({"1": 0.50}, other_directional={Direction.PRO_DEMOCRAT: D_CAP - 10.0})
    report = asyncio.run(_manager(exchange, tmp_path).run_cycle(ctx))
    assert [(o.exchange_id, o.side) for o in _placed(exchange)] == [("1", "no")]  # NO on a D market is pro-R
    assert report.blocked_direction == 1


def test_own_budget_and_race_cap(tmp_path):
    exchange = FakeExchange()
    small = replace(_ctx({"1": 0.50}), bankroll=400.0)  # 10% = 40 < one 100-share quote at 0.455
    assert asyncio.run(_manager(exchange, tmp_path).run_cycle(small)).blocked_budget == 2
    # Race worst case already at the 2% cap from resting ladder D YES (not inventory, so the
    # delta cap stays out of it): the D-adding YES quote cannot fit, the NO quote can.
    ladder_legs = {"senate:AZ": [RaceLeg("D", "yes", 4_000, 2_000.0)]}
    exchange = FakeExchange()
    report = asyncio.run(_manager(exchange, tmp_path / "b").run_cycle(_ctx({"1": 0.50}, other_race_legs=ladder_legs)))
    assert report.blocked_race == 1 and [o.side for o in _placed(exchange)] == ["no"]


def test_ladder_markets_are_excluded(tmp_path):
    exchange = FakeExchange()
    report = asyncio.run(_manager(exchange, tmp_path).run_cycle(_ctx({"1": 0.50, "3": 0.50}, exclude_markets=frozenset({"1"}))))
    assert report.markets == 1 and {o.exchange_id for o in _placed(exchange)} == {"3"}


# ---------------------------------------------------------------------- DRY_RUN and the real client


def test_dry_run_quotes_virtually_without_any_request(tmp_path):
    session = _FakeSession(delays=[])
    client = _client(session, dry_run=True)
    manager = Q.QuoterManager(client, journal=Q.QuoterJournal(tmp_path / "quoter.jsonl"), clock=lambda: NOW - 1000)
    report = asyncio.run(manager.run_cycle(_ctx({"1": 0.50})))
    assert report.placed == 0 and len(manager.orders) == 2 and all(oid.startswith("dry-") for oid in manager.orders)
    books = {"1": SigBook(yes_bid=0.455, yes_ask=0.55)}  # our own virtual bid is not in the real book
    asyncio.run(manager.run_cycle(_ctx({"1": 0.50}, now=NOW + 15, books=books)))  # requote: virtual cancel + post
    assert session.calls == []
    assert '"event": "placement"' in (tmp_path / "quoter.jsonl").read_text(encoding="utf-8")


def test_book_is_read_only_when_our_order_is_at_the_top(tmp_path):
    exchange = FakeExchange()
    exchange.own_orders = lambda eid: [("bid", 0.455, 100.0)]
    manager = _manager(exchange, tmp_path)
    asyncio.run(manager.run_cycle(_ctx({"1": 0.50, "3": 0.50}, books={"1": SigBook(0.455, 0.55), "3": SigBook(0.45, 0.55)})))
    assert exchange.book_reads == ["1"]


# ---------------------------------------------------------------------- strategy wiring


def test_cycle_runs_taker_quoter_and_ladder_on_separate_markets(tmp_path, caplog):
    import logging

    from strategy import OrderJournal, Strategy
    from test_cycle_resilience import SETTINGS, FakeExchange as TakerExchange, FakeKalshi, FakePolymarket, TWO_RACES
    from test_exposure_caps import _RecordingLadder

    class Exchange(TakerExchange):
        async def get_tournament(self, slug):
            return {"id": "tid", **(await super().get_tournament(slug))}

        async def place_batch(self, orders, key, write_rank=1):
            return None  # DRY_RUN

        async def cancel_all_orders(self, **scope):
            return None

        def own_orders(self, exchange_id):
            return []

    class Ladder(_RecordingLadder):
        def markets_in_use(self):
            return frozenset({"1"})  # the ladder holds AZ D

    ladder = Ladder()
    exchange = Exchange()
    quoter = Q.QuoterManager(exchange, journal=Q.QuoterJournal(tmp_path / "quoter.jsonl"))
    taker = Strategy(SETTINGS, exchange, FakePolymarket(), FakeKalshi(), TWO_RACES, OrderJournal(tmp_path / "j.jsonl"), ladder=ladder, quoter=quoter)
    with caplog.at_level(logging.INFO, logger="strategy"):
        report = asyncio.run(taker.run_cycle())
    assert exchange.placed >= 1  # the taker still traded
    assert quoter.markets == ["3"]  # GA D: AZ D belongs to the ladder (AZ R has no fair price)
    assert report.quoter_markets == 1 and len(quoter.orders) == 2  # virtual YES + NO
    assert "3" in ladder.ctx.exclude_markets and ladder.ctx.other_gross > 0  # the ladder sees the quoter's book
    done = next(r for r in caplog.records if r.getMessage() == "Cycle done")
    assert done.quoter_markets == 1 and done.quoter_blocked == {}


def test_a_timed_out_book_skips_only_that_market(tmp_path):
    from exchange_api import RequestTimeoutError

    exchange = FakeExchange()
    exchange.own_orders = lambda eid: [("bid", 0.455, 100.0)] if eid == "1" else []

    async def timeout(exchange_id, depth=20):
        raise RequestTimeoutError("GET", f"/exchanges/{exchange_id}/orderbook", 4, 17.0)

    exchange.get_orderbook = timeout
    books = {"1": SigBook(0.455, 0.55), "3": SigBook(0.45, 0.55)}
    asyncio.run(_manager(exchange, tmp_path).run_cycle(_ctx({"1": 0.50, "3": 0.50}, books=books)))
    assert {o.exchange_id for o in _placed(exchange)} == {"3"}
