"""Global gross cap (taker + ladder) and the ladder's directional check. Fakes - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace

import ladder as L
from risk import Direction
from strategy import Holding, OrderJournal, PlannedOrder, Strategy, apply_exposure_caps
from test_cycle_resilience import SETTINGS, FakeExchange as TakerExchange, FakeKalshi, FakePolymarket, MARKET_MAP
from test_ladder import BANKROLL, LIMITS, MAP, FakeExchange, _ctx, _manager

TOTAL_CAP = LIMITS.max_total_gross_exposure_fraction * BANKROLL  # 40,000
D_CAP = LIMITS.max_directional_fraction * BANKROLL  # 15,000


def _placed(exchange: FakeExchange) -> list:
    return [order for batch in exchange.batches for order in batch]


# ---------------------------------------------------------------------- taker side


def test_taker_sizes_into_the_room_the_ladder_leaves():
    order = [PlannedOrder("increase", "1", "yes", 10_000, 0.45, "edge")]
    kwargs = dict(market_map=MAP, fair={"1": 0.60}, margins={"1": 0.01}, bankroll=BANKROLL, limits=LIMITS)

    capped = apply_exposure_caps(order, {}, books_gross=TOTAL_CAP - 1_000.0, **kwargs)
    assert capped[0].quantity == 2_222  # 1,000 of room / 0.45
    assert "total_gross" in capped[0].reason

    assert apply_exposure_caps(order, {}, books_gross=TOTAL_CAP, **kwargs) == []  # no room left at all


def test_global_cap_is_silent_without_ladder():
    """ladder_gross=0: the 40% cap is never tighter than the 30% hold-book cap, so sizes and
    reasons are exactly what they were before it existed."""
    order = [PlannedOrder("increase", "1", "yes", 200_000, 0.45, "edge")]
    kwargs = dict(market_map=MAP, fair={"1": 0.60}, margins={"1": 0.01}, bankroll=BANKROLL, limits=LIMITS)
    near_gross_cap = {"5": Holding("yes", 60_000, 29_500.0, 29_500.0)}  # hold book at 29.5%
    capped = apply_exposure_caps(order, near_gross_cap, **kwargs)
    assert "total_gross" not in capped[0].reason and "gross_exposure" in capped[0].reason


# ---------------------------------------------------------------------- ladder side


def test_ladder_adds_no_level_at_the_global_cap(tmp_path):
    exchange = FakeExchange()
    manager = _manager(exchange, tmp_path)
    report = asyncio.run(manager.run_cycle(replace(_ctx({"1": 0.92, "3": 0.91}), other_gross=TOTAL_CAP)))
    assert _placed(exchange) == []
    assert report.blocked_global > 0 and report.blocked_direction == 0


def test_ladder_fills_only_the_remaining_global_room(tmp_path):
    exchange = FakeExchange()
    manager = _manager(exchange, tmp_path)
    room = 600.0
    report = asyncio.run(manager.run_cycle(replace(_ctx({"1": 0.92, "4": 0.91}), other_gross=TOTAL_CAP - room)))
    placed_cost = sum(o.price * o.quantity for o in _placed(exchange))
    assert 0 < placed_cost <= room
    assert report.blocked_global > 0  # the deeper levels did not fit whole and were skipped


def test_ladder_does_not_add_D_when_taker_is_near_the_D_cap(tmp_path):
    exchange = FakeExchange()
    manager = _manager(exchange, tmp_path)
    # "1" = AZ D, YES favourite -> pro-D. "4" = GA R, YES favourite -> pro-R.
    ctx = replace(_ctx({"1": 0.92, "4": 0.91}), other_directional={Direction.PRO_DEMOCRAT: D_CAP - 100.0})
    report = asyncio.run(manager.run_cycle(ctx))
    placed = _placed(exchange)
    assert {o.exchange_id for o in placed} == {"4"}  # R still quoted, no D added
    assert report.blocked_direction > 0


def test_ladder_counts_its_own_D_inventory_in_the_directional_cap(tmp_path):
    exchange = FakeExchange()
    manager = _manager(exchange, tmp_path)
    # Ladder D inventory (GA D YES) 5,000 + taker D 10,000 = the 15,000 cap.
    manager.positions["3"] = L.LadderPosition("3", "yes", bought=10_000, cost=5_000.0, mode="noise")
    ctx = replace(_ctx({"1": 0.92}), other_directional={Direction.PRO_DEMOCRAT: 10_000.0})
    report = asyncio.run(manager.run_cycle(ctx))
    assert _placed(exchange) == []
    assert report.blocked_direction > 0
    assert manager.directional_cost(MAP) == {Direction.PRO_DEMOCRAT: 5_000.0}


# ---------------------------------------------------------------------- strategy wiring


class _RecordingLadder:
    """Stands in for LadderManager: records the context, reports fixed block counters."""

    def __init__(self):
        self.ctx = None

    def hold_book(self, holdings):
        return dict(holdings)

    def order_ids(self):
        return set()

    def resting_race_legs(self, market_map):
        return {}

    def book_cost(self):
        return 0.0

    def directional_cost(self, market_map):
        return {}

    def markets_in_use(self):
        return frozenset()

    def order_action(self, order_id):
        return None

    async def run_cycle(self, ctx):
        self.ctx = ctx
        return L.LadderReport(blocked_global=2, blocked_direction=1)


def test_cycle_done_reports_ladder_blocks_and_hands_over_the_taker_book(tmp_path, caplog):
    ladder = _RecordingLadder()
    taker = Strategy(SETTINGS, TakerExchange(), FakePolymarket(), FakeKalshi(), MARKET_MAP, OrderJournal(tmp_path / "j.jsonl"), ladder=ladder)
    with caplog.at_level(logging.INFO, logger="strategy"):
        report = asyncio.run(taker.run_cycle())
    assert report.dry_run >= 1  # the taker "sent" a D YES buy (DRY_RUN)
    assert ladder.ctx.other_gross > 0
    assert ladder.ctx.other_directional.get(Direction.PRO_DEMOCRAT, 0.0) == ladder.ctx.other_gross
    done = next(r for r in caplog.records if r.getMessage() == "Cycle done")
    assert (done.ladder_blocked_global, done.ladder_blocked_direction) == (2, 1)
