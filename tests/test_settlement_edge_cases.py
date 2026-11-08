"""Investigation (read-only): how the bot behaves near settlement (2026-11-04T17:00Z).

Documents CURRENT behaviour of the production functions - nothing is patched and no test
here proposes a fix. Each test names the code it pins down.

Run from the project root:  python -m pytest tests/test_settlement_edge_cases.py -v
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from config import Settings
from exchange_api import ExchangeClient
from market_mapper import MarketMap, MarketMapping, Race
from oracle_api import KalshiClient, OracleQuote
from risk import RiskLimits, SizedBet, check_buy, tournament_kelly_fraction
from strategy import Holding, OrderJournal, OrderStatus, PlannedOrder, SigBook, compute_fair_price, fit_to_book, plan_orders

MAP = MarketMap({"1": MarketMapping("1", "m1", "D RI", Race("senate", "RI"), "D", "tok", None)})


def _pm(bid, ask):
    return OracleQuote("polymarket", "tok", bid, ask, 100, 100, None, time.time())


# ---------------------------------------------------------------------- buys near 1


def test_no_buy_above_0_95_even_when_the_oracle_says_certain():
    """risk.check_buy: above MAX_BUY_PRICE (0.93) the shrunk edge must reach HIGH_PRICE_MIN_EDGE
    (0.04); with margin >= MIN_ORACLE_MARGIN (0.01) and fair <= 1, no price above 0.95 passes."""
    limits = RiskLimits()
    assert check_buy(0.95, 1.0, 0.01, limits)[0] is True  # the boundary: 1 - 0.01 - 0.95 = 0.04
    assert check_buy(0.96, 1.0, 0.01, limits)[0] is False
    assert check_buy(0.98, 0.999, 0.01, limits)[0] is False
    assert check_buy(0.999, 1.0, 0.0, limits)[0] is False  # even with no margin: edge 0.001


# ---------------------------------------------------------------------- oracle extremes


def test_kalshi_quote_at_0_99_1_00_is_one_sided_and_unusable():
    """oracle_api.KalshiClient.quote_from_market: an ask of 1 (or a bid of 0) is read as an empty
    side, so a nearly decided Kalshi market gives no fair price."""
    q = KalshiClient.quote_from_market({"ticker": "K", "yes_bid_dollars": "0.99", "yes_ask_dollars": "1.00"})
    assert q.best_ask is None and q.mid is None
    fair, reason = compute_fair_price(None, q, 10.0)
    assert fair is None and reason == "one-sided/crossed book"
    settled = KalshiClient.quote_from_market({"ticker": "K", "yes_bid_dollars": "0", "yes_ask_dollars": "1", "last_price_dollars": "1"})
    assert settled.best_bid is None and settled.best_ask is None  # last = 1 is ignored


def test_a_single_oracle_near_0_is_trusted_without_a_cross_check():
    """strategy.compute_fair_price: the 0.10 disagreement check needs two oracles; with one, any
    tight quote is the fair price - a glitchy 0.01/0.03 Polymarket book becomes fair 0.02."""
    fair, reason = compute_fair_price(_pm(0.01, 0.03), None, 10.0)
    assert reason == "ok" and abs(fair.prob - 0.02) < 1e-9


# ---------------------------------------------------------------------- sells of held positions


def test_no_edge_sale_happens_at_any_bid_at_or_above_a_near_zero_fair():
    """strategy.plan_orders, "no target" branch: a held side is sold when bid >= fair_side, so an
    oracle at ~0 (right or wrong) dumps the position at whatever the bid is."""
    held = {"1": Holding("yes", 1000, 20.0, 600.0)}
    (sale,) = plan_orders({"1": 0.01}, {}, held, {"1": SigBook(0.02, 0.05)}, set(), MAP)
    assert (sale.kind, sale.side, sale.quantity, sale.price) == ("reduce", "yes", 1000, 0.02)


def test_flip_sells_the_whole_position_at_the_bid_without_a_fair_check():
    """strategy.plan_orders, "flip" branch: a target on the other side closes the held side at
    the current bid with no comparison to fair (unlike the "no edge" branch)."""
    held = {"1": Holding("yes", 1000, 300.0, 600.0)}
    target = {"1": SizedBet("1", "no", 0.70, 100, 70.0, 0.05, 0.1, "senate:RI")}
    (flip,) = plan_orders({"1": 0.30}, target, held, {"1": SigBook(0.05, 0.30)}, set(), MAP)
    assert (flip.kind, flip.side, flip.quantity, flip.price, flip.reason) == ("reduce", "yes", 1000, 0.05, "flip: close opposite side")


def test_reduce_fit_takes_the_best_bid_level_without_a_price_floor():
    """strategy.fit_to_book: a reduce takes the best level only, with no check against fair."""
    order = PlannedOrder("reduce", "1", "yes", 1000, 0.05, "flip: close opposite side")
    fitted = fit_to_book(order, {"bids": [{"price": 0.05, "quantity": 300}], "asks": []}, 0.30, RiskLimits(), 0.01)
    assert (fitted.quantity, fitted.price) == (300, 0.05)


# ---------------------------------------------------------------------- rejected orders, dry run, endgame Kelly


def test_a_rejected_order_does_not_block_the_next_cycle(tmp_path: Path):
    """strategy.OrderJournal.recent_unknown blocks only UNKNOWN orders: a 4xx REJECTED order
    (e.g. a 403 on a closed market) is planned and sent again next cycle."""
    journal = OrderJournal(tmp_path / "journal.jsonl")
    payload = {"exchangeId": "1", "side": "yes", "action": "buy", "quantity": 10, "price": 0.5}
    journal.record("k1", OrderStatus.REJECTED, code="MARKET_CLOSED", payload=payload)
    assert journal.recent_unknown(300) == {}
    journal.record("k2", OrderStatus.UNKNOWN, payload=payload)
    assert ("1", "yes") in journal.recent_unknown(300)


def test_dry_run_also_skips_cancel_all():
    """exchange_api.ExchangeClient.cancel_all_orders returns None in DRY_RUN without sending:
    switching a live bot to DRY_RUN=true and restarting leaves its resting orders in place."""
    settings = Settings(sig_api_key="k", sig_base_url="https://sig.invalid/api/v1", dry_run=True, log_level="INFO", log_dir=Path("."),
                        tournament_slug="t", oracle_staleness_threshold_s=10.0, cycle_interval_s=15.0)
    client = ExchangeClient(settings)  # no session: a real request would raise
    assert asyncio.run(client.cancel_all_orders()) is None


def test_endgame_kelly_rises_to_1_5x_when_outside_the_target():
    """risk.tournament_kelly_fraction: over the last 20% of the window the fraction ramps to
    variance_modifier (max 1.5x below the top-10% target, min 0.5x at 1st place)."""
    assert tournament_kelly_fraction(0.25, 1.0, 0.5) == 0.25  # before the endgame
    assert tournament_kelly_fraction(0.25, 1.0, 0.0) == 0.375  # last place at the close: 1.5x
    assert tournament_kelly_fraction(0.25, 0.0, 0.0) == 0.125  # 1st place at the close: 0.5x
