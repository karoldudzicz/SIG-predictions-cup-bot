"""edge_scan.py on synthetic data with a fake GET-only client - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import time
from datetime import datetime, timedelta, timezone

import pytest

import edge_scan
from edge_scan import SigReader, find_holes, hole_trade_stats, parse_candles, parity_series, run_scan
from test_cycle_resilience import MARKET_MAP  # race senate:AZ, D = "1", R = "2"

T0 = datetime(2026, 10, 1, 16, 0, tzinfo=timezone.utc)
NOW = T0 + timedelta(minutes=60)


def _candle(i, close, low=None, high=None):
    return {"time": edge_scan._iso(T0 + timedelta(minutes=i)), "open": close, "close": close,
            "high": high if high is not None else close + 0.01, "low": low if low is not None else close - 0.01,
            "vwap": close, "volume": 100, "tradeCount": 1}


# D: flat at 0.50, a YES-down hole at minute 20 (low 0.35, recovered at 22), a YES-up hole at 30.
D_CANDLES = [_candle(i, 0.50) for i in range(40)]
D_CANDLES[20] = _candle(20, 0.36, low=0.35)
D_CANDLES[21] = _candle(21, 0.42)
D_CANDLES[30] = _candle(30, 0.64, high=0.65)
R_CANDLES = [_candle(i, 0.46) for i in range(40)]  # gap = 0.50 - (1 - 0.46) = -0.04
HOLE_TRADES = [{"price": 0.36, "size": 100}, {"price": 0.45, "size": 50}]  # only the first is >= 0.10 below ref


class FakeClient:
    """Stands in for ExchangeClient; fails the test on anything but GET."""

    def __init__(self):
        self.calls = []

    async def _request(self, method, path, params=None, json_body=None):
        assert method == "GET" and json_body is None, f"edge_scan must only GET, got {method} {path}"
        self.calls.append((path, dict(params or {})))
        exchange_id = path.split("/")[2]
        if path.endswith("/price-history"):
            rows = D_CANDLES if exchange_id == "1" else R_CANDLES
            if params["from"] == edge_scan._iso(T0):  # first page truncated -> continuation
                return {"candles": rows[:25], "to": rows[25]["time"], "coverage": {"complete": False}}
            return {"candles": rows[25:], "to": edge_scan._iso(NOW), "coverage": {"complete": True}}
        if path.endswith("/trades"):
            trades = HOLE_TRADES if params["from"] == edge_scan._iso(T0 + timedelta(minutes=20)) else []
            return {"data": trades, "pagination": {"hasMore": False, "nextCursor": None}}
        raise AssertionError(f"unexpected path {path}")


def test_find_holes_and_recovery():
    holes = find_holes("1", parse_candles(D_CANDLES))
    by_direction = {h.direction: h for h in holes}
    assert set(by_direction) == {"down", "up"}
    down = by_direction["down"]
    assert down.time == T0 + timedelta(minutes=20) and down.ref == 0.50 and down.extreme == 0.35
    assert down.recovery_min == 2.0  # close back within 0.03 of ref at minute 22
    assert by_direction["up"].recovery_min == 1.0


def test_hole_trade_stats_counts_only_trades_beyond_the_threshold():
    down = next(h for h in find_holes("1", parse_candles(D_CANDLES)) if h.direction == "down")
    shares, value = hole_trade_stats(down, HOLE_TRADES)
    assert shares == 100 and value == 14.0  # 100 * |0.50 - 0.36|


def test_parity_gap_series():
    points = parity_series(parse_candles(D_CANDLES), parse_candles(R_CANDLES), NOW)
    assert abs(points[0].gap + 0.04) < 1e-9
    # Last candle at minute 39: closes stay valid for 30 more minutes, but the grid stops at NOW (60).
    assert points[-1].time == NOW - timedelta(minutes=1)


def test_run_scan_end_to_end_and_cache(tmp_path):
    client = FakeClient()
    reader = SigReader(client, tmp_path / "raw", max_reads_per_min=60_000)
    summary = asyncio.run(run_scan(reader, MARKET_MAP, T0, NOW, out_dir=tmp_path))

    assert summary["holes"]["count"] == 2 and summary["holes"]["total_value"] == 14.0
    # The hole minutes 20, 21 and 30 close at 0.36 / 0.42 / 0.64 -> gaps -0.18, -0.12, +0.10;
    # every other minute is the baseline -0.04 (> 0.02, not > 0.05).
    assert summary["parity"]["races"] == 1 and summary["parity"]["minutes_gt_0.05"] == 3
    assert summary["parity"]["minutes_gt_0.02"] == summary["parity"]["minutes"]
    with (tmp_path / "holes.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [(r["exchange_id"], r["race"], r["party"], r["direction"]) for r in rows] == [
        ("1", "senate:AZ", "D", "down"),
        ("1", "senate:AZ", "D", "up"),
    ]
    assert json.loads((tmp_path / "summary.json").read_text())["holes"]["count"] == 2

    # Rerun: truncated candle pages and old trade minutes come from the raw cache.
    first_run_requests = len(client.calls)
    reader2 = SigReader(client, tmp_path / "raw", max_reads_per_min=60_000)
    asyncio.run(run_scan(reader2, MARKET_MAP, T0, NOW, out_dir=tmp_path))
    assert reader2.cache_hits >= 2 + 2  # 2 truncated candle pages + 2 hole-minute trade windows
    assert reader2.requests == len(client.calls) - first_run_requests == 2  # only the two "to now" pages


def test_edge_scan_never_references_order_methods():
    source = open(edge_scan.__file__, encoding="utf-8").read()
    for forbidden in ("place_order", "place_multi_leg", "cancel_order", "cancel_all_orders", "OrderRequest", '"POST"', '"DELETE"'):
        assert forbidden not in source, forbidden


# ---------------------------------------------------------------------- read budget


def test_reads_are_spaced_by_max_reads_per_min(tmp_path):
    client = FakeClient()
    reader = SigReader(client, tmp_path / "raw", max_reads_per_min=600)  # one every 0.1 s

    async def three_reads():
        for minute in range(3):
            params = {"from": edge_scan._iso(T0 + timedelta(minutes=50 + minute)), "to": "x"}
            await reader.get("/exchanges/1/trades", params, cache_key="t", cacheable=lambda b: False)

    started = time.monotonic()
    asyncio.run(three_reads())
    # 3 reads -> at least 2 gaps of 0.1 s; slack for the Windows timer (~15 ms granularity).
    assert time.monotonic() - started >= 0.17
    assert reader.requests == 3


def test_429_pauses_the_scan_and_retries(tmp_path, caplog):
    from exchange_api import ApiError

    class RateLimitedOnce(FakeClient):
        async def _request(self, method, path, params=None, json_body=None):
            if not getattr(self, "limited", False):
                self.limited = True
                raise ApiError(429, "RATE_LIMITED", "Rate limit exceeded.", retry_after=0.2)
            return await super()._request(method, path, params, json_body)

    reader = SigReader(RateLimitedOnce(), tmp_path / "raw", max_reads_per_min=60_000)
    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="edge_scan"):
        body = asyncio.run(reader.get("/exchanges/1/trades", {"from": "a", "to": "b"}, cache_key="t", cacheable=lambda b: False))
    assert body["pagination"]["hasMore"] is False
    assert time.monotonic() - started >= 0.2  # waited Retry-After
    assert reader.rate_limited == 1
    assert any(r.getMessage() == "Rate limited, pausing the scan" for r in caplog.records)


@pytest.mark.parametrize("value", ["0", "-5", "150"])
def test_max_reads_per_min_is_validated(value):
    with pytest.raises(SystemExit) as exit_info:
        edge_scan.main(["--max-reads-per-min", value])
    assert exit_info.value.code == 2
