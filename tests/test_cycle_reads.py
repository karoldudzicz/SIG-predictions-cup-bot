"""Reads per cycle against a routing fake of the SIG API: the real ExchangeClient, 237 markets.
No network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from dataclasses import replace

import pytest

import main
import strategy as S
from market_mapper import MarketMap, MarketMapping, Race
from oracle_api import OracleQuote
from strategy import OrderJournal, Strategy
from test_cycle_resilience import SETTINGS
from test_exchange_http import _client

N_MARKETS = 237
MAP = MarketMap(
    {
        str(i): MarketMapping(str(i), str(i), f"Will the Democratic Party win seat {i}?", Race("house", f"S{i}"), "D", f"tok-{i}", None)
        for i in range(1, N_MARKETS + 1)
    }
)


class _Response:
    def __init__(self, body):
        self.status, self._body, self.headers, self.reason = 200, body, {}, "OK"

    async def json(self, content_type=None):
        return self._body


class _Request:
    def __init__(self, body):
        self._body = body

    async def __aenter__(self):
        return _Response(self._body)

    async def __aexit__(self, *exc_info):
        return False


class RoutingSession:
    """Answers every SIG read the cycle makes; counts requests per path kind."""

    def __init__(self, relationships=()):
        self.paths: list[str] = []
        self.relationships = list(relationships)

    def request(self, method, url, params=None, json=None, timeout=None):
        assert method == "GET", f"DRY_RUN cycle sent {method} {url}"
        path = url.split("/api/v1")[-1]
        self.paths.append(path)
        return _Request(self._body(path, params or {}))

    def _body(self, path, params):
        page = {"data": [], "pagination": {"hasMore": False, "nextCursor": None}}
        if path == "/portfolio/pnl":
            return {"totalAccountValue": 100_000.0}
        if path == "/portfolio/positions":
            return {"positions": []}
        if path == "/tournaments/t":
            return {"id": "tid", "startDate": "2026-10-01T16:00:00Z", "endDate": "2026-11-04T17:00:00Z"}
        if path == "/tournaments/t/leaderboard":
            return {"myRank": 5, "total": 100}
        if path == "/exchanges/prices":
            ids = params["ids"].split(",")
            return {"data": [{"exchangeId": eid, "bestBid": 0.40, "bestAsk": 0.45} for eid in ids], "missingIds": []}
        if path.endswith("/orderbook"):
            return {"bids": [{"price": 0.40, "quantity": 5000}], "asks": [{"price": 0.45, "quantity": 5000}]}
        if path == "/relationships":
            return {**page, "data": self.relationships}
        if path == "/relationships/constraints":
            return {"data": [], "violationsCount": 0}
        if path in ("/orders", "/tournaments/t/portfolio/fills"):
            return page
        raise AssertionError(f"unexpected read {path}")

    def kinds(self) -> Counter:
        return Counter("orderbook" if p.endswith("/orderbook") else p for p in self.paths)

    async def close(self):
        pass


class Oracle:
    """Fair = the SIG mid (no edge) everywhere except `edge` markets (fair 0.60 vs ask 0.45)."""

    def __init__(self, edge=()):
        self.edge = {f"tok-{eid}" for eid in edge}

    async def get_quotes(self, tokens):
        now = time.time()
        return {t: OracleQuote("polymarket", t, (0.595 if t in self.edge else 0.42), (0.605 if t in self.edge else 0.43), 500.0, 500.0, None, now) for t in tokens}


class NoKalshi:
    async def get_quotes(self, tickers):
        return {}


def _cycle_reads(tmp_path, edge=(), cycles=1, relationships=()):
    session = RoutingSession(relationships)
    client = _client(session)
    taker = Strategy(SETTINGS, client, Oracle(edge), NoKalshi(), MAP, OrderJournal(tmp_path / "j.jsonl"))
    reports, per_cycle = [], []
    for _ in range(cycles):
        before = len(session.paths)
        reports.append(asyncio.run(taker.run_cycle()))
        per_cycle.append(Counter("orderbook" if p.endswith("/orderbook") else p for p in session.paths[before:]))
    return reports, per_cycle


def test_idle_cycle_reads_three_bulk_pages_and_the_essentials(tmp_path):
    reports, (first, second) = _cycle_reads(tmp_path, cycles=2)
    assert first["/exchanges/prices"] == 3  # ceil(237 / 100)
    assert first["orderbook"] == 0
    assert first == Counter({
        "/exchanges/prices": 3, "/portfolio/pnl": 1, "/portfolio/positions": 1, "/orders": 1,
        "/tournaments/t": 1, "/relationships": 1, "/tournaments/t/leaderboard": 1, "/relationships/constraints": 1,
    })
    # Steady state: tournament cached, relationships / leaderboard on their 5-minute refresh,
    # constraints skipped (no relationships): 3 bulk + 3 essential.
    assert second == Counter({"/exchanges/prices": 3, "/portfolio/pnl": 1, "/portfolio/positions": 1, "/orders": 1})
    assert reports[1].relationship_violations is None and reports[1].sig_prices == "bulk"


def test_ten_orders_read_ten_books(tmp_path):
    edge = [str(i) for i in range(1, 11)]
    reports, (first,) = _cycle_reads(tmp_path, edge=edge)
    assert first["/exchanges/prices"] == 3 and first["orderbook"] == 10
    assert reports[0].books_read == 10 and reports[0].targets == 10


def test_constraints_stay_on_while_relationships_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "build_clusters", lambda market_map, relationships=(): [])  # shape not under test
    _, cycles = _cycle_reads(tmp_path, cycles=2, relationships=[{"id": "r1"}])
    assert [c["/relationships/constraints"] for c in cycles] == [1, 1]


@pytest.mark.parametrize(("mode", "expected"), [("on", [1, 1]), ("off", [0, 0])])
def test_relationships_check_override(tmp_path, monkeypatch, mode, expected):
    monkeypatch.setattr(S, "RELATIONSHIPS_CHECK_ENABLED", mode)
    _, cycles = _cycle_reads(tmp_path, cycles=2)
    assert [c["/relationships/constraints"] for c in cycles] == expected


def test_min_gap_after_a_long_cycle(monkeypatch):
    bot = main.Bot(replace(SETTINGS, cycle_interval_s=0.01), max_cycles=2)
    delays: list[float] = []

    async def record(seconds):
        delays.append(seconds)

    class SlowStrategy:
        async def run_cycle(self):
            await asyncio.sleep(0.05)  # longer than the 0.01 s interval
            return S.CycleReport()

    monkeypatch.setattr(bot, "_idle", record)
    asyncio.run(bot._loop(SlowStrategy()))
    assert delays == [main.MIN_CYCLE_GAP_S]
