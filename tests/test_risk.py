"""Regression tests for race-level worst-case caps and price-aware sizing.

Run from the project root:  python -m pytest tests -v
Pure maths - no network, no orders.
"""

from __future__ import annotations

import time
from collections import Counter

import pytest

from config import MAX_RACE_WORST_LOSS_FRAC
from market_mapper import MarketMap, MarketMapping, Race
from risk import (
    Cluster,
    Opportunity,
    RaceLeg,
    RiskLimits,
    build_clusters,
    check_buy,
    max_shares_within_race_cap,
    race_scenario_pnl,
    race_worst_case_loss,
    size_cluster,
    size_portfolio,
)
from oracle_api import OracleQuote
from strategy import (
    REASON_MISSING,
    REASON_ONE_SIDED,
    REASON_STALE,
    REASON_WIDE,
    Holding,
    PlannedOrder,
    SigBook,
    apply_exposure_caps,
    compute_fair_price,
    plan_orders,
    quote_too_old,
)

BANKROLL = 100_000.0
RACE_CAP = MAX_RACE_WORST_LOSS_FRAC * BANKROLL
LIMITS = RiskLimits()

# A thin two-party race and three orders that are all the same bet - D YES, R NO, R NO
# win only if D wins (synthetic numbers, shaped like the day-1 case that motivated the cap).
RI = Race("senate", "XX")
RI_D, RI_R = "9101", "9102"
RI_ORDERS = [
    ("D", RI_D, "yes", 5000, 0.96),
    ("R", RI_R, "no", 5000, 0.95),
    ("R", RI_R, "no", 60, 0.97),
]


def _ri_market_map() -> MarketMap:
    return MarketMap(
        {
            RI_D: MarketMapping(RI_D, "m1", "Will the Democratic Party win the XX Senate?", RI, "D", None, None),
            RI_R: MarketMapping(RI_R, "m2", "Will the Republican Party win the XX Senate?", RI, "R", None, None),
        }
    )


def test_d_yes_and_r_no_are_the_same_bet():
    legs = [RaceLeg("D", "yes", 100, 96.5), RaceLeg("R", "no", 100, 95.0)]
    pnl = race_scenario_pnl(["D", "R"], legs)
    assert pnl["D"] == pytest.approx(200 - 191.5)  # both pay
    assert pnl["R"] == pytest.approx(-191.5)  # both lose together
    assert race_worst_case_loss(["D", "R"], legs) == pytest.approx(191.5)


def test_none_of_listed_scenario_is_counted():
    # NO on both D and R pays in the "nobody listed wins" scenario; YES on D loses there.
    legs = [RaceLeg("D", "yes", 100, 60.0)]
    assert race_scenario_pnl(["D", "R", "I"], legs)["none of the listed parties"] == pytest.approx(-60.0)


def test_same_bet_regression_sequential():
    """Feed the three orders one by one: the race worst case never exceeds the cap."""
    legs: list[RaceLeg] = []
    allowed = []
    for party, _, side, quantity, price in RI_ORDERS:
        q = max_shares_within_race_cap(["D", "R"], legs, party, side, price, quantity, RACE_CAP)
        allowed.append(q)
        if q:
            legs.append(RaceLeg(party, side, q, q * price))
        assert race_worst_case_loss(["D", "R"], legs) <= RACE_CAP + 1e-6
    assert allowed[0] < RI_ORDERS[0][3]  # first order shrunk to the cap
    assert allowed[1:] == [0, 0]  # the same bet again: nothing left under the cap


def test_same_bet_regression_same_cycle():
    """All three planned in ONE cycle: earlier accepted orders must count against later ones."""
    planned = [PlannedOrder("increase", eid, side, quantity, price, "test") for _, eid, side, quantity, price in RI_ORDERS]
    accepted = apply_exposure_caps(
        planned,
        holdings={},
        market_map=_ri_market_map(),
        fair={RI_D: 0.985, RI_R: 0.012},
        margins={RI_D: 0.01, RI_R: 0.01},
        bankroll=BANKROLL,
        limits=LIMITS,
    )
    party = {RI_D: "D", RI_R: "R"}
    legs = [RaceLeg(party[o.exchange_id], o.side, o.quantity, o.quantity * o.price) for o in accepted]
    assert race_worst_case_loss(["D", "R"], legs) <= RACE_CAP + 1e-6
    assert sum(o.quantity * o.price for o in accepted) <= RACE_CAP + 1e-6


def test_existing_positions_count_against_the_race_cap():
    """Holdings already breach the cap: a new same-direction order must be dropped."""
    holdings = {RI_D: Holding("yes", 5000, 4900.0, 4800.0), RI_R: Holding("no", 5000, 4900.0, 4750.0)}
    planned = [PlannedOrder("increase", RI_R, "no", 100, 0.95, "test")]
    accepted = apply_exposure_caps(planned, holdings, _ri_market_map(), {RI_R: 0.012}, {RI_R: 0.01}, BANKROLL, LIMITS)
    assert accepted == []


def test_high_price_rejected():
    accepted, edge, reason = check_buy(price=0.965, win_prob=0.985, margin=0.01, limits=LIMITS)
    assert not accepted
    assert edge == pytest.approx(0.01)
    # And end to end through sizing: no bet at all.
    opportunity = Opportunity("1", fair_prob=0.985, yes_price=0.965, no_price=None, margin=0.01)
    assert size_cluster(Cluster("t", frozenset({"1"}), exclusive=True), {"1": opportunity}, BANKROLL, LIMITS) == []


def test_high_price_needs_bigger_edge_even_with_clean_margin():
    # 0.95 with fair 0.985 and the minimum margin: shrunk edge 0.025 < 0.04 -> rejected.
    assert not check_buy(price=0.95, win_prob=0.985, margin=0.01, limits=LIMITS)[0]
    # 0.94 with fair 0.995: shrunk edge 0.045 >= 0.04 -> allowed above MAX_BUY_PRICE.
    assert check_buy(price=0.94, win_prob=0.995, margin=0.01, limits=LIMITS)[0]


def test_mid_price_accepted_and_capped_per_market():
    accepted, edge, _ = check_buy(price=0.76, win_prob=0.80, margin=0.01, limits=LIMITS)
    assert accepted
    assert edge == pytest.approx(0.03)
    opportunity = Opportunity("1", fair_prob=0.80, yes_price=0.76, no_price=None, margin=0.01)
    bets = size_cluster(Cluster("t", frozenset({"1"}), exclusive=True), {"1": opportunity}, BANKROLL, LIMITS)
    assert len(bets) == 1 and bets[0].side == "yes"
    assert bets[0].stake <= LIMITS.max_fraction_per_contract * BANKROLL + 1e-6


def test_no_trade_when_margin_eats_the_edge():
    assert not check_buy(price=0.79, win_prob=0.80, margin=0.01, limits=LIMITS)[0]


# ---------------------------------------------------------------------- held positions in size_portfolio


def _map(races: dict[str, dict[str, str]]) -> MarketMap:
    """races: state -> {party: exchange_id}; all Senate races."""
    return MarketMap(
        {
            eid: MarketMapping(eid, eid, f"Will the {party} win the {state} Senate?", Race("senate", state), party, None, None)
            for state, parties in races.items()
            for party, eid in parties.items()
        }
    )


def test_held_position_without_target_fills_the_cluster_cap():
    """Held R NO worth 3% (no target: no edge) -> a new I YES target in the same race gets 0."""
    market_map = _map({"NE": {"D": "968", "I": "969", "R": "970"}})
    clusters = build_clusters(market_map)
    opportunities = {
        "969": Opportunity("969", fair_prob=0.40, yes_price=0.30, no_price=None, margin=0.01),  # I: big edge
        "970": Opportunity("970", fair_prob=0.60, yes_price=0.62, no_price=0.41, margin=0.01),  # R: no edge either way
    }
    without_holdings = size_portfolio(clusters, opportunities, BANKROLL, market_map, LIMITS)
    assert [b.exchange_id for b in without_holdings] == ["969"]  # control: I YES is wanted

    held = {"970": ("no", 0.03 * BANKROLL)}
    assert size_portfolio(clusters, opportunities, BANKROLL, market_map, LIMITS, held) == []


def test_held_positions_without_targets_use_up_the_directional_cap():
    """D-lean holdings worth 14% across races with no targets -> a new D-lean target gets <= the last 1%."""
    market_map = _map(
        {
            "AZ": {"D": "1", "R": "2"},
            "GA": {"D": "3", "R": "4"},
            "MI": {"D": "5", "R": "6"},
            "NC": {"D": "7", "R": "8"},
        }
    )
    clusters = build_clusters(market_map)
    opportunities = {
        "1": Opportunity("1", fair_prob=0.50, yes_price=0.55, no_price=0.50, margin=0.01),  # held, priced, no edge
        "7": Opportunity("7", fair_prob=0.60, yes_price=0.50, no_price=None, margin=0.01),  # new D-lean edge
    }
    control = size_portfolio(clusters, opportunities, BANKROLL, market_map, LIMITS)
    assert control[0].stake > 0.01 * BANKROLL  # without holdings it would take more than 1%

    held = {
        "1": ("yes", 5_000.0),  # D YES in AZ - has a price, but no target
        "3": ("yes", 5_000.0),  # D YES in GA - no fair price this cycle
        "6": ("no", 4_000.0),  # R NO in MI - also D-lean, no fair price
    }
    bets = size_portfolio(clusters, opportunities, BANKROLL, market_map, LIMITS, held)
    assert [b.exchange_id for b in bets] == ["7"]
    assert 0 < bets[0].stake <= LIMITS.max_directional_fraction * BANKROLL - 14_000.0 + 1e-6


# ---------------------------------------------------------------------- SELL_ABOVE_TARGET


def _above_target_setup():
    """Held 6000 D YES (worth 3% of bankroll); the 2% per-market cap shrinks the target to ~4000."""
    market_map = _map({"AZ": {"D": "1", "R": "2"}})
    opportunities = {"1": Opportunity("1", fair_prob=0.60, yes_price=0.50, no_price=None, margin=0.01)}
    holdings = {"1": Holding("yes", 6000, market_value=3000.0, cost_basis=2700.0)}
    held = {eid: (h.side, h.market_value) for eid, h in holdings.items()}
    targets = {b.exchange_id: b for b in size_portfolio(build_clusters(market_map), opportunities, BANKROLL, market_map, LIMITS, held)}
    assert targets["1"].side == "yes" and targets["1"].quantity < holdings["1"].shares  # the cap really cut the target
    books = {"1": SigBook(yes_bid=0.49, yes_ask=0.50)}
    return market_map, targets, holdings, books


def test_no_sell_when_cap_reduces_target_below_holdings(caplog):
    market_map, targets, holdings, books = _above_target_setup()
    with caplog.at_level("INFO", logger="strategy"):
        planned = plan_orders({"1": 0.60}, targets, holdings, books, set(), market_map, sell_above_target=False)
    assert planned == []
    assert any(r.getMessage() == "Position above target, holding" and r.exchange_id == "1" for r in caplog.records)


def test_sell_above_target_true_keeps_old_behaviour():
    market_map, targets, holdings, books = _above_target_setup()
    planned = plan_orders({"1": 0.60}, targets, holdings, books, set(), market_map, sell_above_target=True)
    assert [(p.kind, p.side, p.quantity) for p in planned] == [("reduce", "yes", 6000 - targets["1"].quantity)]


def test_no_sell_when_cap_removes_target_entirely():
    """Target scaled to 0 -> 'no edge' branch; it only sells at bid >= fair, impossible while the edge exists."""
    holdings = {"1": Holding("yes", 6000, market_value=3000.0, cost_basis=2700.0)}
    books = {"1": SigBook(yes_bid=0.49, yes_ask=0.50)}
    assert plan_orders({"1": 0.60}, {}, holdings, books, set(), None, sell_above_target=False) == []


# ---------------------------------------------------------------------- no_fair_reasons split


def _quote(bid, ask, age_s=0.0, source="polymarket"):
    return OracleQuote(source, "ref", bid, ask, 100.0, 100.0, None, fetched_at=time.time() - age_s)


def test_reason_counter_splits_by_cause():
    cases = {
        "fresh": (_quote(0.40, 0.42), None),
        "missing": (None, None),
        "stale": (_quote(0.40, 0.42, age_s=60), None),
        "one-sided": (_quote(None, 0.42), None),
        "crossed": (_quote(0.45, 0.42), None),
        "wide": (_quote(0.30, 0.45), None),
        "stale beats missing": (None, _quote(0.40, 0.42, age_s=60, source="kalshi")),
        "wide beats one-sided": (_quote(None, 0.42), _quote(0.30, 0.45, source="kalshi")),
    }
    reasons = {name: compute_fair_price(pm, ks, max_age_s=10)[1] for name, (pm, ks) in cases.items()}
    assert reasons == {
        "fresh": "ok",
        "missing": REASON_MISSING,
        "stale": REASON_STALE,
        "one-sided": REASON_ONE_SIDED,
        "crossed": REASON_ONE_SIDED,
        "wide": REASON_WIDE,
        "stale beats missing": REASON_STALE,
        "wide beats one-sided": REASON_WIDE,
    }
    counts = Counter(r for r in reasons.values() if r != "ok")
    assert counts == {REASON_STALE: 2, REASON_ONE_SIDED: 2, REASON_WIDE: 2, REASON_MISSING: 1}


# ---------------------------------------------------------------------- order-time quote age guard


def test_fair_price_carries_the_oldest_quote_time():
    fresh, older = _quote(0.40, 0.42, age_s=1), _quote(0.41, 0.43, age_s=5, source="kalshi")
    price, _ = compute_fair_price(fresh, older, max_age_s=10)
    assert price.fetched_at == pytest.approx(older.fetched_at)


def test_order_skipped_when_quote_too_old(caplog):
    now = time.time()
    order = PlannedOrder("increase", "1", "yes", 100, 0.50, "edge", quote_fetched_at=now - 30)
    with caplog.at_level("INFO", logger="strategy"):
        assert quote_too_old(order, now=now, max_age_s=20)
    record = next(r for r in caplog.records if r.getMessage() == "Order skipped: quote too old")
    assert record.quote_age_s == pytest.approx(30, abs=0.01)


def test_order_sent_when_quote_fresh_and_skipped_when_undated():
    now = time.time()
    assert not quote_too_old(PlannedOrder("increase", "1", "yes", 100, 0.50, "e", quote_fetched_at=now - 5), now=now, max_age_s=20)
    assert quote_too_old(PlannedOrder("increase", "1", "yes", 100, 0.50, "e"), now=now, max_age_s=20)
