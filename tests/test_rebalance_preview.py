"""scripts/rebalance_preview.py on mocked data - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import sys
import time
from pathlib import Path

import pytest

from edge_scan import SigReader
from market_mapper import MarketMap, MarketMapping, Race
from oracle_api import OracleQuote

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "rebalance_preview.py"
_spec = importlib.util.spec_from_file_location("rebalance_preview", SCRIPT)
R = importlib.util.module_from_spec(_spec)
sys.modules["rebalance_preview"] = R  # dataclasses look their module up while it executes
_spec.loader.exec_module(R)

MAP = MarketMap(
    {
        "1": MarketMapping("1", "1", "D AZ", Race("senate", "AZ"), "D", "tok-1", None),
        "2": MarketMapping("2", "2", "R AZ", Race("senate", "AZ"), "R", "tok-2", None),
        "3": MarketMapping("3", "3", "D GA", Race("senate", "GA"), "D", "tok-3", None),
        "4": MarketMapping("4", "4", "R GA", Race("senate", "GA"), "R", None, None),
        "5": MarketMapping("5", "5", "D NV", Race("senate", "NV"), "D", "tok-5", None),
        "6": MarketMapping("6", "6", "R NV", Race("senate", "NV"), "R", None, None),
        "7": MarketMapping("7", "7", "D OH", Race("senate", "OH"), "D", "tok-7", None),
        "8": MarketMapping("8", "8", "R OH", Race("senate", "OH"), "R", None, None),
    }
)
POSITIONS = [
    # AZ: D YES with no edge left, its cost fills the AZ race worst case (R wins: -2,400 > 2,000)
    # -> the R NO target (pro-D, same race) is blocked by the race cap.
    {"exchangeId": "1", "quantity": 4_000, "marketValue": 2_320.0, "costBasis": 2_400.0},
    # GA: D YES with an edge, far above its 2%-capped target.
    {"exchangeId": "3", "quantity": 10_000, "marketValue": 5_800.0, "costBasis": 5_000.0},
    {"exchangeId": "5", "quantity": 1_000, "marketValue": 490.0, "costBasis": 480.0},  # no edge, bid 1c under fair
    {"exchangeId": "7", "quantity": -500, "marketValue": 250.0, "costBasis": 240.0},  # NO; book has no YES ask
    {"exchangeId": "8", "quantity": 200, "marketValue": 80.0, "costBasis": 70.0},  # no oracle at all
]
PRICES = {
    "1": {"bestBid": 0.58, "bestAsk": 0.59},
    "2": {"bestBid": 0.42, "bestAsk": 0.43},  # R NO costs 0.58, fair NO 0.67
    "3": {"bestBid": 0.58, "bestAsk": 0.59},
    "5": {"bestBid": 0.49, "bestAsk": 0.51},
    "7": {"bestBid": 0.48, "bestAsk": None},
    "8": {"bestBid": 0.30, "bestAsk": 0.36},
}


def _q(token, mid):
    return OracleQuote("polymarket", token, mid - 0.005, mid + 0.005, 500.0, 500.0, None, time.time())


def _pm():
    """Fresh oracle quotes per test: quotes built at import go stale (10 s) before a long suite reaches these tests."""
    return {"tok-1": _q("tok-1", 0.60), "tok-2": _q("tok-2", 0.33), "tok-3": _q("tok-3", 0.65), "tok-5": _q("tok-5", 0.50), "tok-7": _q("tok-7", 0.50)}


def _inputs(**overrides):
    kwargs = dict(market_map=MAP, positions=POSITIONS, open_orders=[], bankroll=100_000.0, prices=PRICES, pm_quotes=_pm(), ks_quotes={})
    return R.PreviewInputs(**{**kwargs, **overrides})


def _rows():
    rows, summary = R.build_preview(_inputs())
    return {row.exchange_id: row for row in rows}, rows, summary


def test_position_above_its_capped_target():
    by_id, _, _ = _rows()
    row = by_id["3"]
    assert row.note == "above target" and 0 < row.target_shares < row.held_shares
    assert row.excess_shares == row.held_shares - row.target_shares
    assert row.sell_price == 0.58 and row.fair_minus_bid == pytest.approx(0.07)
    assert row.sell_cost_vs_fair == pytest.approx(0.07 * row.excess_shares, abs=0.01)
    assert row.freed_gross == pytest.approx(5_000.0 * row.excess_shares / 10_000, abs=0.01)
    assert row.freed_directional == pytest.approx(5_800.0 * row.excess_shares / 10_000, abs=0.01)
    assert row.freed_race_worst_case > 0 and row.direction == "D"
    assert row.sold_today_with_flag and row.sellable


def test_no_target_no_bid_and_unpriced_positions():
    by_id, _, summary = _rows()
    row = by_id["1"]
    assert row.target_shares == 0 and row.excess_shares == 4_000 and row.note.startswith("no target")
    assert row.fair_minus_bid == pytest.approx(0.02) and row.freed_gross == pytest.approx(2_400.0)
    assert row.freed_race_worst_case == pytest.approx(2_320.0)  # R wins: -2,400 -> -80
    assert not row.sold_today_with_flag  # the flag only covers positions above a target on their side
    assert by_id["5"].fair_minus_bid == pytest.approx(0.01)
    # NO position: its bid is 1 - YES ask, and there is no YES ask.
    assert by_id["7"].side == "no" and by_id["7"].sell_price is None and by_id["7"].unsellable_reason == "no bid"
    assert by_id["8"].target_shares is None and by_id["8"].unsellable_reason == "no fresh fair"
    assert dict(summary["unsellable"]) == {"7": "no bid", "8": "no fresh fair"}


def test_no_side_costs_are_in_no_terms():
    prices = {**PRICES, "7": {"bestBid": 0.48, "bestAsk": 0.50}}  # NO bid = 1 - 0.50 = 0.50, fair NO = 0.50
    rows, _ = R.build_preview(_inputs(prices=prices))
    row = next(r for r in rows if r.exchange_id == "7")
    assert row.sell_price == 0.50 and row.fair_side == pytest.approx(0.50) and row.fair_minus_bid == pytest.approx(0.0)


def test_ranking_and_cost_bands():
    by_id, rows, summary = _rows()
    ranked = [eid for eid, _ in summary["ranked_by_cost_per_freed_dollar"]]
    assert ranked == ["5", "1", "3"]  # 0.021, 0.033, 0.14 per freed dollar
    assert [r.exchange_id for r in rows[:3]] == ranked  # CSV / log order: cheapest first
    tight, loose = summary["scenarios"]["fair_minus_bid<=0.01"], summary["scenarios"]["fair_minus_bid<=0.02"]
    assert tight["positions"] == 1 and tight["freed_gross"] == pytest.approx(480.0)
    assert loose["positions"] == 2 and loose["freed_gross"] == pytest.approx(480.0 + 2_400.0)
    assert loose["sell_cost_vs_fair"] == pytest.approx(10.0 + 80.0)
    assert loose["freed_race_worst_case"] == pytest.approx(tight["freed_race_worst_case"] + 2_320.0)
    assert loose["freed_directional"] == {"D": pytest.approx(490.0 + 2_320.0)}


def test_blocked_targets_get_room_only_when_their_race_is_freed():
    _, _, summary = _rows()
    assert summary["blocked_targets_today"] == 1  # AZ R NO: the AZ race worst case is full
    assert summary["scenarios"]["fair_minus_bid<=0.01"]["blocked_targets_with_room"] == 0  # NV sale frees nothing in AZ
    assert summary["scenarios"]["fair_minus_bid<=0.02"]["blocked_targets_with_room"] == 1
    assert summary["scenarios"]["all_sellable"]["blocked_targets_accepted_in_one_cycle"] == 1


def test_open_orders_are_skipped_like_the_taker_does():
    _, summary = R.build_preview(_inputs(open_orders=[{"exchangeId": "2"}]))
    assert summary["blocked_targets_today"] == 0  # no increase planned on an exchange with an open order


def test_outputs(tmp_path):
    rows, summary = R.build_preview(_inputs())
    R.write_outputs(rows, summary, tmp_path / "rebalance_preview.csv")
    written = list(csv.DictReader((tmp_path / "rebalance_preview.csv").open(encoding="utf-8")))
    assert [r["exchange_id"] for r in written] == [r.exchange_id for r in rows]
    assert set(written[0]) == set(R.CSV_FIELDS)
    assert (tmp_path / "rebalance_preview.json").exists()


# ---------------------------------------------------------------------- reads


class FakeClient:
    """GET only, routed by path."""

    def __init__(self):
        self.calls = []

    async def _request(self, method, path, params=None, json_body=None):
        assert method == "GET" and json_body is None, f"rebalance_preview must only GET, got {method} {path}"
        self.calls.append((path, dict(params or {})))
        page = {"data": [], "pagination": {"hasMore": False}}
        if path == "/tournaments/t":
            return {"id": "tid", "startDate": "2026-10-01T16:00:00Z", "endDate": "2026-11-04T17:00:00Z"}
        if path == "/portfolio/pnl":
            return {"totalAccountValue": 100_000.0}
        if path == "/portfolio/positions":
            return {"positions": POSITIONS}
        if path == "/tournaments/t/leaderboard":
            return {"myRank": 5, "total": 100}
        if path == "/exchanges/prices":
            return {"data": [{"exchangeId": eid, **PRICES[eid]} for eid in params["ids"].split(",") if eid in PRICES], "missingIds": []}
        if path.endswith("/orderbook"):
            return {"bids": [{"price": 0.58, "quantity": 1234}], "asks": [{"price": 0.59, "quantity": 99}]}
        if path in ("/orders", "/relationships"):
            return page
        raise AssertionError(f"unexpected path {path}")


class FakeOracle:
    def __init__(self, quotes):
        self.quotes = quotes

    async def get_quotes(self, tokens):
        return {t: self.quotes[t] for t in tokens if t in self.quotes}


def test_gather_inputs_reads_only_and_scopes_prices(tmp_path):
    client = FakeClient()
    reader = SigReader(client, tmp_path / "raw", max_reads_per_min=60_000)
    inp = asyncio.run(R.gather_inputs(reader, FakeOracle(_pm()), FakeOracle({}), MAP, "t", 10.0, with_depth=True))
    prices_call = next(params for path, params in client.calls if path == "/exchanges/prices")
    assert prices_call["tournamentId"] == "tid"
    assert inp.bankroll == 100_000.0 and inp.standing == pytest.approx(0.05)
    # Depth: one book per position above target with a bid (1, 3, 5), not for 7 (no bid) or 8 (no fair).
    books = sorted(path for path, _ in client.calls if path.endswith("/orderbook"))
    assert books == ["/exchanges/1/orderbook", "/exchanges/3/orderbook", "/exchanges/5/orderbook"]
    assert inp.bid_depth == {"1": 1234.0, "3": 1234.0, "5": 1234.0}


def test_script_never_references_order_methods():
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("place_order", "place_batch", "place_multi_leg", "cancel_order", "cancel_all_orders", "OrderRequest", '"POST"', '"DELETE"'):
        assert forbidden not in source, forbidden


@pytest.mark.parametrize("value", ["0", "31"])
def test_read_rate_is_validated(value):
    with pytest.raises(SystemExit) as exit_info:
        R.main(["--max-reads-per-min", value])
    assert exit_info.value.code == 2
