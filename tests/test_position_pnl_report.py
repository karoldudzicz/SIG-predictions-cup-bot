"""scripts/position_pnl_report.py on mocked data - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

from edge_scan import SigReader
from market_mapper import MarketMap, MarketMapping, Race
from markout import unix_time
from oracle_api import OracleQuote
from strategy import OrderJournal, OrderStatus

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "position_pnl_report.py"
_spec = importlib.util.spec_from_file_location("position_pnl_report", SCRIPT)
P = importlib.util.module_from_spec(_spec)
sys.modules["position_pnl_report"] = P  # dataclasses look their module up while it executes
_spec.loader.exec_module(P)


def _m(eid, state, party, token=True):
    return MarketMapping(eid, eid, f"{party} {state}", Race("senate", state), party, f"tok-{eid}" if token else None, None)


MAP = MarketMap({m.sig_exchange_id: m for m in [
    _m("1", "AZ", "D"), _m("2", "GA", "R"), _m("3", "NV", "D"), _m("4", "OH", "D"), _m("5", "ME", "D", token=False),
]})
POSITIONS = [
    {"exchangeId": "1", "quantity": 1_000, "marketValue": 690.0, "costBasis": 600.0},  # old D YES
    {"exchangeId": "2", "quantity": -500, "marketValue": 205.0, "costBasis": 200.0},  # old R NO
    {"exchangeId": "3", "quantity": 200, "marketValue": 58.0, "costBasis": 60.0},  # today's D YES
    {"exchangeId": "4", "quantity": 100, "marketValue": 50.0, "costBasis": 45.0},  # a fill without a known action
    {"exchangeId": "5", "quantity": 50, "marketValue": 25.0, "costBasis": 20.0},  # no fill at all
]
FILLS = [
    {"id": 11, "orderId": 101, "exchangeId": "1", "side": "yes", "quantity": 600, "price": 0.58, "filledAt": "2026-10-02T15:00:00Z"},
    {"id": 12, "orderId": 102, "exchangeId": "1", "side": "yes", "quantity": 400, "price": 0.63, "filledAt": "2026-10-02T16:00:00Z"},
    {"id": 13, "orderId": 103, "exchangeId": "1", "side": "yes", "quantity": -100, "price": 0.66, "filledAt": "2026-10-03T10:00:00Z"},  # a sell
    {"id": 21, "orderId": 201, "exchangeId": "2", "side": "no", "quantity": -500, "price": 0.40, "filledAt": "2026-10-01T18:00:00Z"},  # NO price
    {"id": 31, "orderId": 301, "exchangeId": "3", "side": "yes", "quantity": 200, "price": 0.30, "filledAt": "2026-10-05T14:00:00Z"},
    {"id": 41, "orderId": 999, "exchangeId": "4", "side": "yes", "quantity": 100, "price": 0.45, "filledAt": "2026-10-04T12:00:00Z"},
]
ACTIONS = {"101": "buy", "102": "buy", "103": "sell", "201": "buy", "301": "buy"}
FAIR_AT_FILL = {11: 0.70, 12: 0.70, 21: 0.50}  # YES fair when the bot saw the fill (markout.jsonl)
PRICES = {
    "1": {"bestBid": 0.68, "bestAsk": 0.70},
    "2": {"bestBid": 0.58, "bestAsk": 0.60},  # NO: bid 0.40, ask 0.42
    "3": {"bestBid": 0.28, "bestAsk": 0.30},
    "4": {"bestBid": 0.44, "bestAsk": 0.48},
}


def _q(eid, mid):
    return OracleQuote("polymarket", f"tok-{eid}", mid - 0.005, mid + 0.005, 500.0, 500.0, None, time.time())


PM = {"tok-1": _q("1", 0.71), "tok-2": _q("2", 0.59), "tok-3": _q("3", 0.30), "tok-4": _q("4", 0.47)}


def _inputs(**overrides):
    kwargs = dict(market_map=MAP, positions=POSITIONS, fills=FILLS, prices=PRICES, pm_quotes=PM, ks_quotes={},
                  fair_at_fill=FAIR_AT_FILL, order_actions=ACTIONS,
                  old_until=unix_time(P.DEFAULT_OLD_UNTIL), new_since=unix_time(P.DEFAULT_NEW_SINCE))
    return P.PnlInputs(**{**kwargs, **overrides})


def _run():
    rows, summary = P.build_report(_inputs())
    return {r.exchange_id: r for r in rows}, summary


def test_entry_from_buy_fills_only():
    by_id, _ = _run()
    row = by_id["1"]
    assert row.entry_side == pytest.approx((600 * 0.58 + 400 * 0.63) / 1000) and row.buy_fills == 2  # the sell is left out
    assert row.first_fill == "2026-10-02T15:00:00Z" and row.group == "old"
    assert row.api_avg_cost_side == pytest.approx(0.60)


def test_no_position_is_priced_in_no_terms():
    by_id, _ = _run()
    row = by_id["2"]
    assert (row.side, row.entry_side, row.entry_yes) == ("no", 0.40, pytest.approx(0.60))
    assert (row.bid_side, row.mid_side) == (pytest.approx(0.40), pytest.approx(0.41))
    assert row.fair_side_entry == pytest.approx(0.50) and row.fair_side_now == pytest.approx(0.41)


def test_gaps_pnl_and_drivers():
    by_id, _ = _run()
    d = by_id["1"]  # entry 0.60, fair at entry 0.70 -> gap 0.10; now mid 0.69, fair 0.71 -> gap 0.02
    assert (d.gap_entry, d.gap_now, d.gap_change) == (pytest.approx(0.10), pytest.approx(0.02), pytest.approx(-0.08))
    assert d.gap_closed_share == pytest.approx(0.8)
    assert (d.pnl_mid, d.pnl_bid) == (pytest.approx(90.0), pytest.approx(80.0))
    assert (d.pnl_mid_ticks, d.pnl_bid_ticks) == (pytest.approx(18.0), pytest.approx(16.0))
    assert (d.sig_move, d.oracle_move, d.driver) == (pytest.approx(0.09), pytest.approx(0.01), "SIG moved toward the oracle")
    r = by_id["2"]  # NO: entry 0.40, fair at entry 0.50; now mid 0.41, fair 0.41 -> the oracle came to SIG
    assert r.gap_closed_share == pytest.approx(1.0) and r.driver == "oracle moved toward SIG"
    assert (r.pnl_mid, r.pnl_bid) == (pytest.approx(5.0), pytest.approx(0.0))


def test_groups_notes_and_summary():
    by_id, summary = _run()
    assert by_id["3"].group == "new" and by_id["3"].fair_side_entry is None and by_id["3"].driver == ""
    assert by_id["4"].group == "other" and by_id["5"].group == "unknown"
    assert summary["notes"]["fills_without_action_taken_as_buy"] == 1  # order 999 is not in the journal
    assert summary["notes"]["positions_without_buy_fills"] == 1
    assert summary["notes"]["positions_without_fair_at_entry"] == 3  # 3, 4, 5
    assert summary["notes"]["positions_without_sig_price"] == 1  # 5
    assert summary["by_group"]["old"] == {"positions": 2, "pnl_bid": 80.0, "pnl_mid": 95.0}
    assert summary["by_group"]["new"]["pnl_bid"] == pytest.approx(-4.0)
    assert summary["pnl_bid_total"] == pytest.approx(80.0 + 0.0 - 4.0 + 100 * (0.44 - 0.45))
    assert summary["converged_share"] == 1.0 and summary["positions_with_gap_at_entry"] == 2
    assert summary["median_gap_change"] == pytest.approx(-0.09)
    assert summary["old_positions_drivers"] == {"SIG moved toward the oracle": 1, "oracle moved toward SIG": 1}
    assert [r["exchange_id"] for r in summary["worst_by_bid"]][0] == "3"
    assert [r["exchange_id"] for r in summary["best_by_bid"]][0] == "1"


def test_local_journals(tmp_path):
    lines = [json.dumps({"event": "fill", "fill_id": 11, "fair_at_fill": 0.7}), json.dumps({"event": "markout", "fill_id": 11}),
             json.dumps({"event": "fill", "fill_id": 12, "fair_at_fill": None})]
    assert P.fair_at_fill_from_markout(lines) == {11: 0.7}
    journal = OrderJournal(tmp_path / "j.jsonl")
    journal.record("k", OrderStatus.ACCEPTED, result={"orderId": 101, "action": "buy"})
    assert P.order_actions_from_journal(tmp_path / "j.jsonl", FILLS) == {"101": "buy"}
    assert P.order_actions_from_journal(tmp_path / "missing.jsonl", FILLS) == {}


class FakeClient:
    """GET only, routed by path; fills come in two pages."""

    def __init__(self):
        self.calls = []

    async def _request(self, method, path, params=None, json_body=None):
        assert method == "GET" and json_body is None, f"position_pnl_report must only GET, got {method} {path}"
        self.calls.append((path, dict(params or {})))
        if path == "/tournaments/t":
            return {"id": "tid"}
        if path == "/portfolio/positions":
            return {"positions": POSITIONS}
        if path == "/tournaments/t/portfolio/fills":
            if params.get("cursor") is None:
                return {"data": FILLS[:3], "pagination": {"hasMore": True, "nextCursor": "c2"}}
            return {"data": FILLS[3:], "pagination": {"hasMore": False}}
        if path == "/exchanges/prices":
            return {"data": [{"exchangeId": eid, **PRICES[eid]} for eid in params["ids"].split(",") if eid in PRICES], "missingIds": []}
        raise AssertionError(f"unexpected path {path}")


class FakeOracle:
    def __init__(self, quotes):
        self.quotes = quotes

    async def get_quotes(self, tokens):
        return {t: self.quotes[t] for t in tokens if t in self.quotes}


def test_gather_inputs_reads_only_and_follows_fill_pages(tmp_path):
    client = FakeClient()
    reader = SigReader(client, tmp_path / "raw", max_reads_per_min=60_000)
    inp = asyncio.run(P.gather_inputs(reader, FakeOracle(PM), FakeOracle({}), MAP, "t", 10.0))
    assert len(inp.fills) == len(FILLS) and len(inp.positions) == 5
    prices_call = next(params for path, params in client.calls if path == "/exchanges/prices")
    assert prices_call["tournamentId"] == "tid" and prices_call["ids"] == "1,2,3,4,5"  # held exchanges only


def test_outputs(tmp_path):
    rows, summary = P.build_report(_inputs())
    P.write_outputs(rows, summary, tmp_path / "pnl.csv", tmp_path / "summary.json")
    written = list(csv.DictReader((tmp_path / "pnl.csv").open(encoding="utf-8")))
    assert [r["exchange_id"] for r in written] == ["1", "2", "3", "4", "5"]
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["positions"] == 5


def test_script_never_references_order_methods():
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("place_order", "place_batch", "place_multi_leg", "cancel_order", "cancel_all_orders", "OrderRequest", '"POST"', '"DELETE"'):
        assert forbidden not in source, forbidden


@pytest.mark.parametrize("argv", [["--max-reads-per-min", "31"], ["--old-until", "yesterday"]])
def test_cli_validation(argv):
    with pytest.raises(SystemExit) as exit_info:
        P.main(argv)
    assert exit_info.value.code == 2
