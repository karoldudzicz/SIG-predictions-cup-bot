"""TAKER_GROSS_FRAC and DIRECTIONAL_LIMIT_MODE: defaults unchanged, env validation, and the
production stress mode reproducing gross_cap_preview. Fakes - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from dataclasses import replace

import pytest

import config
import risk
from risk import Direction, Opportunity, RiskLimits, SizedBet, size_portfolio, stress_directional_cap
from strategy import apply_exposure_caps, holdings_from_positions, plan_orders


# ---------------------------------------------------------------------- defaults


def test_defaults_are_todays_limits():
    assert (config.TAKER_GROSS_FRAC, config.MAX_GROSS_EXPOSURE_FRAC, config.TOTAL_GROSS_EXPOSURE_FRAC) == (0.30, 0.30, 0.40)
    assert (config.DIRECTIONAL_LIMIT_MODE, config.DIRECTIONAL_STRESS_S, config.DIRECTIONAL_STRESS_FRAC) == ("nominal", 0.5, 0.04)
    limits = RiskLimits()
    assert (limits.max_total_fraction, limits.max_gross_exposure_fraction, limits.max_total_gross_exposure_fraction) == (0.30, 0.30, 0.40)
    assert (limits.max_directional_fraction, limits.directional_mode) == (0.15, "nominal")


# ---------------------------------------------------------------------- env


def _reload(monkeypatch, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    try:
        config.CONFIG_WARNINGS.clear()
        reloaded = importlib.reload(config)
        return {k: getattr(reloaded, k) for k in (
            "TAKER_GROSS_FRAC", "MAX_GROSS_EXPOSURE_FRAC", "TOTAL_GROSS_EXPOSURE_FRAC",
            "DIRECTIONAL_LIMIT_MODE", "DIRECTIONAL_STRESS_S", "DIRECTIONAL_STRESS_FRAC",
        )}, {w["variable"] for w in reloaded.CONFIG_WARNINGS}
    finally:
        for name in env:
            monkeypatch.delenv(name)
        importlib.reload(config)  # back to the clean defaults for the other tests


def test_overrides_are_applied(monkeypatch):
    values, warnings = _reload(monkeypatch, {
        "TAKER_GROSS_FRAC": "0.5", "DIRECTIONAL_LIMIT_MODE": "stress", "DIRECTIONAL_STRESS_S": "0.75", "DIRECTIONAL_STRESS_FRAC": "0.03",
    })
    assert values == {
        "TAKER_GROSS_FRAC": 0.5, "MAX_GROSS_EXPOSURE_FRAC": 0.5, "TOTAL_GROSS_EXPOSURE_FRAC": 0.5,  # global lifted to the taker's
        "DIRECTIONAL_LIMIT_MODE": "stress", "DIRECTIONAL_STRESS_S": 0.75, "DIRECTIONAL_STRESS_FRAC": 0.03,
    }
    assert warnings == set()


def test_total_gross_is_the_larger_of_the_two(monkeypatch):
    values, _ = _reload(monkeypatch, {"TAKER_GROSS_FRAC": "0.5", "TOTAL_GROSS_EXPOSURE_FRAC": "0.7"})
    assert values["TOTAL_GROSS_EXPOSURE_FRAC"] == 0.7
    values, _ = _reload(monkeypatch, {"TAKER_GROSS_FRAC": "0.6", "TOTAL_GROSS_EXPOSURE_FRAC": "0.45"})
    assert values["TOTAL_GROSS_EXPOSURE_FRAC"] == 0.6


@pytest.mark.parametrize(
    ("env", "variable"),
    [
        ({"TAKER_GROSS_FRAC": "0.05"}, "TAKER_GROSS_FRAC"),
        ({"TAKER_GROSS_FRAC": "0.9"}, "TAKER_GROSS_FRAC"),
        ({"TAKER_GROSS_FRAC": "lots"}, "TAKER_GROSS_FRAC"),
        ({"DIRECTIONAL_LIMIT_MODE": "aggressive"}, "DIRECTIONAL_LIMIT_MODE"),
        ({"DIRECTIONAL_STRESS_S": "3"}, "DIRECTIONAL_STRESS_S"),
        ({"DIRECTIONAL_STRESS_S": "0.01"}, "DIRECTIONAL_STRESS_S"),
        ({"DIRECTIONAL_STRESS_FRAC": "0.9"}, "DIRECTIONAL_STRESS_FRAC"),
        ({"DIRECTIONAL_STRESS_FRAC": "0"}, "DIRECTIONAL_STRESS_FRAC"),
    ],
)
def test_out_of_range_values_fall_back_to_the_defaults(monkeypatch, env, variable):
    values, warnings = _reload(monkeypatch, env)
    assert (values["TAKER_GROSS_FRAC"], values["TOTAL_GROSS_EXPOSURE_FRAC"], values["DIRECTIONAL_LIMIT_MODE"]) == (0.30, 0.40, "nominal")
    assert (values["DIRECTIONAL_STRESS_S"], values["DIRECTIONAL_STRESS_FRAC"]) == (0.5, 0.04)
    assert variable in warnings


# ---------------------------------------------------------------------- stress cap (unit)


def test_stress_cap_scales_a_direction_and_prices_unpriced_holdings_at_their_mark(caplog):
    from market_mapper import MarketMap, MarketMapping, Race

    market_map = MarketMap({
        "1": MarketMapping("1", "1", "D A", Race("senate", "AZ"), "D", None, None),
        "2": MarketMapping("2", "2", "D B", Race("senate", "GA"), "D", None, None),
    })
    bet = SizedBet("1", "yes", 0.50, 4_000, 2_000.0, 0.1, 0.2, "c", Direction.PRO_DEMOCRAT)
    opportunities = {"1": Opportunity("1", fair_prob=0.50, yes_price=0.50, no_price=0.52, margin=0.01)}
    holdings = {"2": ("yes", 600.0)}  # 1,000 shares marked at 0.60, no fair this cycle
    limits = replace(RiskLimits(), directional_mode="stress", directional_stress_s=0.5, directional_stress_frac=0.004)  # budget 400
    with caplog.at_level(logging.INFO, logger="risk"):
        scaled, info = stress_directional_cap([bet], opportunities, market_map, 100_000.0, limits, holdings, {"2": 1_000})
    held = 1_000 * risk.side_loss_per_share(0.60, "D", "yes", "R", 0.5)
    wanted = 4_000 * risk.side_loss_per_share(0.50, "D", "yes", "R", 0.5)
    assert info["D"]["held_uncovered_stress"] == pytest.approx(held, abs=0.01)
    assert info["D"]["scale"] == pytest.approx((400 - held) / wanted, abs=1e-4)
    assert scaled[0].quantity == int(4_000 * (400 - held) / wanted)
    record = next(r for r in caplog.records if r.getMessage() == "Directional cap binding (stress)")
    assert (record.positions, record.adverse_wave) == ("D-directional", "wave R (R wins)")


# ---------------------------------------------------------------------- production == preview


def test_taker_gross_half_with_stress_reproduces_gross_cap_preview():
    """Production sizing in stress mode with TAKER_GROSS_FRAC = 0.5 (as config would build
    RiskLimits) gives the same targets and orders as gross_cap_preview's 0.50 variant."""
    import test_gross_cap_preview as T

    G = T.G
    inp = T._inputs()
    holdings = holdings_from_positions(inp.positions)
    view = G._cycle_view(inp)
    rows = G.position_rows(inp, view, holdings)
    production = replace(view.limits, max_total_fraction=0.5, max_gross_exposure_fraction=0.5, max_total_gross_exposure_fraction=0.5,
                         directional_mode="stress", directional_stress_s=0.5, directional_stress_frac=0.04)
    opportunities = {
        eid: Opportunity(eid, fair_prob=p, yes_price=view.books[eid].buy_price("yes"), no_price=view.books[eid].buy_price("no"), margin=view.margins[eid])
        for eid, p in view.fair.items()
    }
    held = {eid: (h.side, h.market_value) for eid, h in holdings.items()}
    shares = {eid: h.shares for eid, h in holdings.items()}
    bot_targets = {b.exchange_id: b for b in size_portfolio(view.clusters, opportunities, inp.bankroll, inp.market_map, production, held, shares)}
    preview_targets = G._targets(inp, view, holdings, rows, "stress", G.variant_limits(view.limits, 0.5))
    assert bot_targets == preview_targets

    planned = [o for o in plan_orders(view.fair, bot_targets, holdings, view.books, view.skip, inp.market_map, sell_above_target=False) if o.kind == "increase"]
    unpaced = replace(production, max_new_notional_per_cycle_fraction=1e9)
    accepted = apply_exposure_caps(planned, holdings, inp.market_map, view.fair, view.margins, inp.bankroll, unpaced, log=False)
    variant = next(v for v in G.build_gross_preview(inp, (0.50,))[1]["variants"])
    assert {o.exchange_id for o in accepted} == {o["exchange_id"] for o in variant["unlocked_orders"]} == {"1000", "936"}
    assert {o.exchange_id: o.quantity for o in accepted} == {o["exchange_id"]: o["quantity"] for o in variant["unlocked_orders"]}


def test_nominal_mode_ignores_held_shares():
    """The default path: passing held_shares changes nothing."""
    import test_gross_cap_preview as T

    G = T.G
    inp = T._inputs()
    holdings = holdings_from_positions(inp.positions)
    view = G._cycle_view(inp)
    opportunities = {
        eid: Opportunity(eid, fair_prob=p, yes_price=view.books[eid].buy_price("yes"), no_price=view.books[eid].buy_price("no"), margin=view.margins[eid])
        for eid, p in view.fair.items()
    }
    held = {eid: (h.side, h.market_value) for eid, h in holdings.items()}
    loose = replace(view.limits, max_total_fraction=0.6, max_directional_fraction=0.5)  # let targets through
    without = size_portfolio(view.clusters, opportunities, inp.bankroll, inp.market_map, loose, held)
    with_shares = size_portfolio(view.clusters, opportunities, inp.bankroll, inp.market_map, loose, held, {eid: h.shares for eid, h in holdings.items()})
    assert without == with_shares and without  # same, non-empty


def test_strategy_passes_held_shares(tmp_path, monkeypatch):
    import strategy
    from test_cycle_resilience import SETTINGS, FakeExchange, FakeKalshi, FakePolymarket, MARKET_MAP

    seen = {}
    real = strategy.size_portfolio

    def spy(*args, **kwargs):
        seen["args"] = args
        return real(*args, **kwargs)

    monkeypatch.setattr(strategy, "size_portfolio", spy)
    taker = strategy.Strategy(SETTINGS, FakeExchange(), FakePolymarket(), FakeKalshi(), MARKET_MAP, strategy.OrderJournal(tmp_path / "j.jsonl"),
                              limits=replace(RiskLimits(), directional_mode="stress"))
    asyncio.run(taker.run_cycle())
    assert len(seen["args"]) == 7 and seen["args"][6] == {}  # held_shares passed (no positions in this fake)
