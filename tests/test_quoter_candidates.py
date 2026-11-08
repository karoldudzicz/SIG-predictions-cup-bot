"""scripts/quoter_candidates.py on mocked data - no network.

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

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "quoter_candidates.py"
_spec = importlib.util.spec_from_file_location("quoter_candidates", SCRIPT)
C = importlib.util.module_from_spec(_spec)
sys.modules["quoter_candidates"] = C  # dataclasses look their module up while it executes
_spec.loader.exec_module(C)


def _m(eid, state, party, token=True):
    return MarketMapping(eid, eid, f"{party} {state}", Race("senate", state), party, f"tok-{eid}" if token else None, None)


MAP = MarketMap({m.sig_exchange_id: m for m in [
    _m("1", "AZ", "D"),  # good, spread 4 ticks
    _m("2", "AZ", "R"),  # good, spread 8 ticks
    _m("3", "GA", "D"),  # held
    _m("4", "GA", "R"),  # spread 2 ticks
    _m("5", "NV", "D"),  # price 0.10: outside 0.15-0.85
    _m("6", "NV", "R"),  # one-sided book
    _m("7", "OH", "D"),  # oracle too uncertain
    _m("8", "OH", "R", token=False),  # no fair
    _m("9", "ME", "I"),  # independent
]})
PRICES = {
    "1": {"bestBid": 0.50, "bestAsk": 0.52},
    "2": {"bestBid": 0.40, "bestAsk": 0.44},
    "3": {"bestBid": 0.50, "bestAsk": 0.55},
    "4": {"bestBid": 0.50, "bestAsk": 0.51},
    "5": {"bestBid": 0.08, "bestAsk": 0.12},
    "6": {"bestBid": 0.40, "bestAsk": None},
    "7": {"bestBid": 0.40, "bestAsk": 0.45},
    "8": {"bestBid": 0.40, "bestAsk": 0.45},
    "9": {"bestBid": 0.40, "bestAsk": 0.45},
}


def _q(eid, mid, half_spread=0.005):
    return OracleQuote("polymarket", f"tok-{eid}", mid - half_spread, mid + half_spread, 500.0, 500.0, None, time.time())


PM = {f"tok-{eid}": _q(eid, mid) for eid, mid in {"1": 0.51, "2": 0.42, "3": 0.52, "4": 0.50, "5": 0.10, "6": 0.40, "9": 0.42}.items()}
PM["tok-7"] = _q("7", 0.42, half_spread=0.03)  # u = 0.03 > 0.02
POSITIONS = [{"exchangeId": "3", "quantity": 1_000, "marketValue": 500.0, "costBasis": 480.0}]


def _inputs(**overrides):
    kwargs = dict(market_map=MAP, positions=POSITIONS, bankroll=100_000.0, prices=PRICES, pm_quotes=PM, ks_quotes={})
    return C.CandidateInputs(**{**kwargs, **overrides})


def test_filters_and_order():
    candidates, summary = C.find_candidates(_inputs())
    assert [(c.exchange_id, c.spread_ticks) for c in candidates] == [("2", 8), ("1", 4)]  # widest first
    assert summary["dropped"] == {
        "position held": 1, "spread < 3 ticks": 1, "price outside 0.15-0.85": 1, "one-sided or empty book": 1,
        "u > 0.02": 1, "no fresh fair": 1, "not D / R": 1,
    }
    first = candidates[0]
    assert (first.race, first.party, first.yes_bid, first.yes_ask, first.fair, first.u) == ("senate:AZ", "R", 0.40, 0.44, 0.42, 0.005)
    # Directional room: 15% of 100k minus the held D YES market value (500) on the D side.
    assert (first.free_room_D, first.free_room_R) == (14_500.0, 15_000.0)


def test_no_room_in_one_direction_lists_nothing():
    full_d = POSITIONS + [{"exchangeId": "1", "quantity": 30_000, "marketValue": 15_000.0, "costBasis": 15_000.0}]
    candidates, summary = C.find_candidates(_inputs(positions=full_d))
    assert candidates == [] and summary["free_room"]["D"] <= 0


class FakeClient:
    """GET only, routed by path."""

    def __init__(self, open_orders=(), books=None):
        self.calls = []
        self.open_orders = list(open_orders)
        self.books = books or {}

    async def _request(self, method, path, params=None, json_body=None):
        assert method == "GET" and json_body is None, f"quoter_candidates must only GET, got {method} {path}"
        self.calls.append((path, dict(params or {})))
        if path == "/tournaments/t":
            return {"id": "tid"}
        if path == "/portfolio/pnl":
            return {"totalAccountValue": 100_000.0}
        if path == "/portfolio/positions":
            return {"positions": POSITIONS}
        if path == "/exchanges/prices":
            return {"data": [{"exchangeId": eid, **PRICES[eid]} for eid in params["ids"].split(",")], "missingIds": []}
        if path == "/orders":
            return {"data": self.open_orders, "pagination": {"hasMore": False}}
        if path.endswith("/orderbook"):
            return self.books.get(path, {"bids": [{"price": 0.4, "quantity": 120}], "asks": [{"price": 0.44, "quantity": 80}]})
        raise AssertionError(f"unexpected path {path}")


class FakeOracle:
    def __init__(self, quotes):
        self.quotes = quotes

    async def get_quotes(self, tokens):
        return {t: self.quotes[t] for t in tokens if t in self.quotes}


def test_end_to_end_reads_only_with_depth(tmp_path):
    client = FakeClient()
    reader = SigReader(client, tmp_path / "raw", max_reads_per_min=60_000)

    async def run():
        inp = await C.gather_inputs(reader, FakeOracle(PM), FakeOracle({}), MAP, "t", 10.0)
        candidates, summary = C.find_candidates(inp)
        await C.add_depth(reader, candidates, max_books=1)  # only the widest spread gets a book
        return candidates, summary

    candidates, summary = asyncio.run(run())
    assert next(params for path, params in client.calls if path == "/exchanges/prices")["tournamentId"] == "tid"
    assert [path for path, _ in client.calls if path.endswith("/orderbook")] == ["/exchanges/2/orderbook"]
    assert (candidates[0].bid_depth, candidates[0].ask_depth) == (120.0, 80.0) and candidates[1].bid_depth is None
    C.write_outputs(candidates, summary, tmp_path / "quoter_candidates.csv")
    rows = list(csv.DictReader((tmp_path / "quoter_candidates.csv").open(encoding="utf-8")))
    assert [r["exchange_id"] for r in rows] == ["2", "1"] and set(rows[0]) == set(C.CSV_FIELDS)


def test_script_never_references_order_methods():
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("place_order", "place_batch", "place_multi_leg", "cancel_order", "cancel_all_orders", "OrderRequest", '"POST"', '"DELETE"'):
        assert forbidden not in source, forbidden


@pytest.mark.parametrize("argv", [["--max-reads-per-min", "0"], ["--max-reads-per-min", "31"], ["--max-books", "-1"]])
def test_cli_validation(argv):
    with pytest.raises(SystemExit) as exit_info:
        C.main(argv)
    assert exit_info.value.code == 2


# ---------------------------------------------------------------------- --ignore-room, histogram, own orders


def test_ignore_room_lists_candidates_and_still_reports_the_room():
    full_d = POSITIONS + [{"exchangeId": "1", "quantity": 30_000, "marketValue": 15_000.0, "costBasis": 15_000.0}]
    candidates, summary = C.find_candidates(_inputs(positions=full_d), ignore_room=True)
    assert [c.exchange_id for c in candidates] == ["2"]  # 1 is now held
    assert candidates[0].free_room_D == pytest.approx(-500.0) and summary["room_filter"] is False


def test_spread_histogram_over_every_two_sided_market():
    histogram, widest = C.spread_histogram(_inputs())
    # Spreads in ticks: 1->4, 2->8, 3->10, 4->2, 5->8, 7/8/9->10; 6 is one-sided.
    assert histogram["two_sided_markets"] == 8
    counts = {label: v["count"] for label, v in histogram["histogram"].items() if v["count"]}
    assert counts == {"2": 1, "4": 1, "8+": 6}
    assert histogram["histogram"]["8+"]["share"] == pytest.approx(0.75)
    # Widest first, whatever the filters: held (3), no fair (8) and independent (9) are listed too.
    assert [r.exchange_id for r in widest[:4]] == ["3", "7", "8", "9"]
    no_fair = next(r for r in widest if r.exchange_id == "8")
    assert no_fair.fair is None and no_fair.u is None


def test_our_resting_order_at_the_top_is_taken_out(tmp_path):
    ours = {"id": 9, "exchangeId": "1", "side": "yes", "action": "buy", "priceLimit": 0.50, "quantity": 100, "open": True}
    book = {"bids": [{"price": 0.50, "quantity": 100}, {"price": 0.49, "quantity": 40}], "asks": [{"price": 0.52, "quantity": 70}]}
    client = FakeClient(open_orders=[ours], books={"/exchanges/1/orderbook": book})
    reader = SigReader(client, tmp_path / "raw", max_reads_per_min=60_000)
    inp = asyncio.run(C.gather_inputs(reader, FakeOracle(PM), FakeOracle({}), MAP, "t", 10.0))
    assert inp.prices["1"]["bestBid"] == 0.49 and inp.prices["1"]["ownRemoved"]  # their best bid, not ours
    candidates, _ = C.find_candidates(inp)
    assert next(c for c in candidates if c.exchange_id == "1").spread_ticks == 6
    assert [path for path, _ in client.calls if path.endswith("/orderbook")] == ["/exchanges/1/orderbook"]


def test_histogram_outputs(tmp_path):
    histogram, widest = C.spread_histogram(_inputs())
    C.write_histogram(histogram, widest, tmp_path / "widest.csv", tmp_path / "histogram.json")
    rows = list(csv.DictReader((tmp_path / "widest.csv").open(encoding="utf-8")))
    assert len(rows) == len(widest) and rows[0]["exchange_id"] == "3"
    assert '"8+"' in (tmp_path / "histogram.json").read_text(encoding="utf-8")
