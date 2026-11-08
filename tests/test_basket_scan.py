"""scripts/basket_scan.py with a fake GET-only client - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from edge_scan import SigReader
from market_mapper import MarketMap, MarketMapping, Race

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "basket_scan.py"
_spec = importlib.util.spec_from_file_location("basket_scan", SCRIPT)
B = importlib.util.module_from_spec(_spec)
sys.modules["basket_scan"] = B  # dataclasses look their module up while it executes
_spec.loader.exec_module(B)

MAP = MarketMap(
    {
        "1": MarketMapping("1", "1", "D AZ", Race("senate", "AZ"), "D", None, None),
        "2": MarketMapping("2", "2", "R AZ", Race("senate", "AZ"), "R", None, None),
        "3": MarketMapping("3", "3", "D GA", Race("senate", "GA"), "D", None, None),
        "4": MarketMapping("4", "4", "R GA", Race("senate", "GA"), "R", None, None),
    }
)
RACES = {"senate:AZ": [("D", "1"), ("R", "2")], "senate:GA": [("D", "3"), ("R", "4")]}


def test_observations_only_past_the_threshold():
    prices = {
        "1": {"bestBid": 0.55, "bestAsk": 0.56}, "2": {"bestBid": 0.47, "bestAsk": 0.48},  # bids 1.02 -> edge
        "3": {"bestBid": 0.40, "bestAsk": 0.44}, "4": {"bestBid": 0.50, "bestAsk": 0.555},  # asks 0.995 -> no edge
    }
    found = B.basket_observations(0.0, RACES, prices)
    assert [(o.race, o.kind, o.total, o.excess) for o in found] == [("senate:AZ", "bid_over", 1.02, 0.02)]
    prices["4"]["bestAsk"] = 0.545  # asks 0.985 -> edge
    assert {o.kind for o in B.basket_observations(0.0, RACES, prices)} == {"bid_over", "ask_under"}
    prices["4"]["bestAsk"] = None  # one leg unpriced: that side is not judged
    assert {o.race for o in B.basket_observations(0.0, RACES, prices)} == {"senate:AZ"}


def test_episodes_and_summary():
    obs = B.basket_observations(0.0, RACES, {"1": {"bestBid": 0.56}, "2": {"bestBid": 0.47}})
    tracker = B.EpisodeTracker()
    assert len(tracker.observe(0.0, obs)) == 1  # started
    assert tracker.observe(15.0, obs) == []  # continues
    tracker.observe(30.0, [])  # gone -> closed
    episodes = tracker.finish()
    assert [(e.start, e.last, e.polls) for e in episodes] == [(0.0, 15.0, 2)]
    episodes[0].depth = 120
    summary = B.summarize(episodes, interval_s=15)
    assert summary["episodes"] == 1 and summary["duration_s"]["max"] == 30.0 and summary["depth_shares"]["min"] == 120


class FakeClient:
    """GET only; the second poll has no edge."""

    def __init__(self):
        self.calls = []
        self.polls = 0

    async def _request(self, method, path, params=None, json_body=None):
        assert method == "GET" and json_body is None, f"basket_scan must only GET, got {method} {path}"
        self.calls.append(path)
        if path == "/exchanges/prices":
            self.polls += 1
            bid_d = 0.56 if self.polls == 1 else 0.50
            rows = {"1": (bid_d, 0.57), "2": (0.47, 0.48), "3": (0.40, 0.45), "4": (0.50, 0.56)}
            return {"data": [{"exchangeId": eid, "bestBid": b, "bestAsk": a} for eid, (b, a) in rows.items()]}
        if path.endswith("/orderbook"):
            quantity = {"/exchanges/1/orderbook": 300, "/exchanges/2/orderbook": 80}[path]
            return {"bids": [{"price": 0.5, "quantity": quantity}], "asks": []}
        raise AssertionError(f"unexpected path {path}")


def test_run_scan_end_to_end(tmp_path):
    client = FakeClient()
    reader = SigReader(client, tmp_path / "raw", max_reads_per_min=60_000)

    async def no_sleep(_):
        return None

    summary = asyncio.run(B.run_scan(reader, MAP, polls=2, interval_s=15, out_dir=tmp_path, sleep=no_sleep))
    assert summary["episodes"] == 1 and summary["depth_shares"]["min"] == 80  # thinnest leg
    rows = list(csv.DictReader((tmp_path / "basket.csv").open(encoding="utf-8")))
    assert [(r["race"], r["kind"], r["sum"], r["depth"]) for r in rows] == [("senate:AZ", "bid_over", "1.03", "80.0")]
    assert json.loads((tmp_path / "basket_summary.json").read_text())["episodes"] == 1
    assert client.calls.count("/exchanges/prices") == 2


def test_basket_scan_never_references_order_methods():
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("place_order", "place_batch", "place_multi_leg", "cancel_order", "cancel_all_orders", "OrderRequest", '"POST"', '"DELETE"'):
        assert forbidden not in source, forbidden


@pytest.mark.parametrize("argv", [["--max-reads-per-min", "0"], ["--max-reads-per-min", "150"], ["--interval-s", "0.5"]])
def test_cli_validation(argv):
    with pytest.raises(SystemExit) as exit_info:
        B.main(argv)
    assert exit_info.value.code == 2
