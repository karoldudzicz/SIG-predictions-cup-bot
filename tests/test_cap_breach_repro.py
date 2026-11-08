"""Investigation (read-only): how race exposure can sit above the 0.02 race worst-loss cap.

Reproduces, against the CURRENT production functions (nothing is patched), the mechanisms
that let senate:RI / house:NH-01 / governor:FL exceed the caps. Each test asserts what the
code does today; none proposes a fix.

Run from the project root:  python -m pytest tests/test_cap_breach_repro.py -v

Setup: starting equity 10,000 -> race cap 0.02 = 200. senate:RI is D (1076) / R (1077);
D YES and R NO pay in the same scenarios, so they are one bet in the race worst case.
"""

from __future__ import annotations

from risk import RaceLeg, RiskLimits, race_worst_case_loss
from market_mapper import MarketMap, MarketMapping, Race
from strategy import Holding, PlannedOrder, SigBook, apply_exposure_caps, plan_orders
from risk import SizedBet

EQUITY = 10_000.0
MAP = MarketMap({m.sig_exchange_id: m for m in [
    MarketMapping("1076", "m1076", "D RI", Race("senate", "RI"), "D", None, None),
    MarketMapping("1077", "m1077", "R RI", Race("senate", "RI"), "R", None, None),
]})
FAIR = {"1076": 0.99, "1077": 0.01}
MARGINS = {"1076": 0.0, "1077": 0.0}
LIMITS = RiskLimits()  # race cap 0.02, contract 0.02, cluster 0.03 (current defaults)


def _loss(holdings: dict[str, Holding], extra: list[RaceLeg] = ()) -> float:
    legs = [RaceLeg(MAP.get(eid).party, h.side, h.shares, h.cost_basis) for eid, h in holdings.items()] + list(extra)
    return race_worst_case_loss(["D", "R"], legs)


def _accepted(orders, holdings, bankroll=EQUITY, extra_race_legs=None):
    # log=True: the full gate as run on fitted orders (strategy.py:986). With log=False (the
    # per-order pre-check, strategy.py:983) accepted orders come back UNSHRUNK by design.
    return apply_exposure_caps(orders, holdings, MAP, FAIR, MARGINS, bankroll, LIMITS, extra_race_legs=extra_race_legs, log=True)


def test_h1_over_cap_positions_are_held_and_only_new_increases_are_blocked():
    """Positions built before the caps existed (0ec8ce1: contract 0.05, cluster 0.10, no race
    cap) stay: plan_orders never sells down to target (SELL_ABOVE_TARGET = False) and the race
    cap only blocks further increases. D YES 500 + R NO 500 at 0.95 / 0.95 ~ 0.095 of equity."""
    held = {"1076": Holding("yes", 500, 480.0, 475.0), "1077": Holding("no", 500, 480.0, 475.0)}
    assert _loss(held) / EQUITY == 0.095
    target = {"1076": SizedBet("1076", "yes", 0.95, 200, 190.0, 0.04, 0.5, "senate:RI")}
    books = {"1076": SigBook(0.94, 0.95), "1077": SigBook(0.04, 0.05)}
    assert plan_orders({"1076": 0.99}, target, held, books, set(), MAP) == []  # above target: held, not reduced
    more = PlannedOrder("increase", "1077", "no", 100, 0.95, "edge")
    assert _accepted([more], held) == []  # the same bet: blocked
    # A hedge (R YES) is blocked too: "none of the listed parties wins" is already at -450,
    # past the -200 cap, and R YES makes that scenario slightly worse - the race is frozen.
    hedge = PlannedOrder("increase", "1077", "yes", 100, 0.05, "edge")
    assert _accepted([hedge], held) == []


def test_h2_cap_scales_with_current_bankroll_not_starting_equity():
    """apply_exposure_caps uses `bankroll` = /portfolio/pnl totalAccountValue (strategy.py:595,
    853). With the account at 10,750 the race cap allows 0.0215 of the STARTING equity."""
    order = PlannedOrder("increase", "1076", "yes", 1000, 0.95, "edge")
    (accepted,) = _accepted([order], {}, bankroll=10_750.0)
    loss = accepted.quantity * 0.95
    assert accepted.quantity == 226 and round(loss / EQUITY, 4) == 0.0215


def test_h3_no_memory_of_fills_between_cycles():
    """Each cycle starts from /portfolio/positions only. If a fill is not yet in positions when
    the next cycle reads them (5 s minimum gap), the same full room is granted again."""
    order = PlannedOrder("increase", "1076", "yes", 1000, 0.95, "edge")
    (first,) = _accepted([order], {})
    (second,) = _accepted([order], {})  # stale positions: the first fill is not visible yet
    assert first.quantity == second.quantity == 210
    total_loss = (first.quantity + second.quantity) * 0.95
    assert round(total_loss / EQUITY, 4) == 0.0399  # ~2x the cap if both fill


def test_h4_taker_resting_order_on_the_sibling_exchange_is_not_counted():
    """A taker order resting on 1077 (R NO) only skips 1077 itself (strategy.py:971); the race
    legs come from positions plus RESTING LADDER / QUOTER orders (_resting_race_legs,
    strategy.py:1036-1044) - not resting taker orders. 1076 (D YES, the same bet) then gets
    the full room; if the 1077 remainder fills before its 60 s TTL, the race is ~2x the cap."""
    resting_taker_r_no = RaceLeg("R", "no", 210, 199.5)  # what the open order would add if filled
    order = PlannedOrder("increase", "1076", "yes", 1000, 0.95, "edge")
    (unaware,) = _accepted([order], {})  # what the cycle does today: the leg is not passed in
    aware = _accepted([order], {}, extra_race_legs={"senate:RI": [resting_taker_r_no]})  # 199.5 of 200 used: dropped
    assert unaware.quantity == 210 and aware == []
    assert round((unaware.quantity * 0.95 + 199.5) / EQUITY, 4) == 0.0399
