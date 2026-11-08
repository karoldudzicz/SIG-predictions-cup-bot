"""scripts/quote_snapshots.py with a fake GET-only reader - no network.

Run from the project root:  python -m pytest tests/test_quote_snapshots.py -v
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from exchange_api import ApiError
from market_mapper import MarketMap, MarketMapping, Race
from research.reader import ResearchAborted

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "quote_snapshots.py"
_spec = importlib.util.spec_from_file_location("quote_snapshots", SCRIPT)
Q = importlib.util.module_from_spec(_spec)
sys.modules["quote_snapshots"] = Q
_spec.loader.exec_module(Q)

MAP = MarketMap({m.sig_exchange_id: m for m in [MarketMapping("1", "m1", "D RI", Race("senate", "RI"), "D", None, None),
                                                MarketMapping("2", "m2", "R RI", Race("senate", "RI"), "R", None, None)]})
PRICES = {"1": {"exchangeId": "1", "bestBid": 0.05, "bestAsk": 0.06, "latestPrice": 0.055},
          "2": {"exchangeId": "2", "bestBid": 0.40, "bestAsk": 0.45},
          "3": {"exchangeId": "3", "bestBid": 0.90, "bestAsk": 0.92},
          "4": {"exchangeId": "4", "bestBid": None, "bestAsk": None}}  # no quote: no line


def test_plan_bulk_first_then_books_with_the_leftover_budget():
    assert Q.plan(237, 10, 0) == {"chunks": 3, "books_per_poll": 0, "reads_per_poll": 3, "reads_per_hour": 360}
    assert Q.plan(237, 10, 10) == {"chunks": 3, "books_per_poll": 2, "reads_per_poll": 5, "reads_per_hour": 600}  # 5 reads per 30 s
    assert Q.plan(237, 15, 25)["books_per_poll"] == 4
    with pytest.raises(ValueError):
        Q.plan(1000, 10, 0)  # 10 bulk reads per poll do not fit 5


def test_universe_and_snapshot_line():
    tape = [{"path": "/exchanges/9/trades"}, {"path": "/exchanges/1/trades"}, {"path": "/portfolio/positions"}]
    assert Q.universe_ids(MAP, tape) == ["1", "2", "9"]
    line = Q.snapshot_line("2026-10-20T12:00:00.000Z", "1", 0.05, 0.06, 0.055)
    assert line["mid"] == pytest.approx(0.055) and line["spread_ticks"] == 2 and line["source"] == "bulk" and line["bid_size"] is None
    assert Q.snapshot_line("t", "1", None, 0.06)["mid"] is None


class FakeReader:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail
        self.settings = type("S", (), {"tournament_slug": "t"})()

    async def get(self, path, params=None):
        self.calls.append((path, dict(params or {})))
        if self.fail is not None and path == "/exchanges/prices":
            raise self.fail
        if path == "/tournaments/t":
            return {"id": "tid"}
        if path == "/exchanges/prices":
            return {"data": [PRICES[i] for i in params["ids"].split(",") if i in PRICES]}
        if path.endswith("/orderbook"):
            return {"bids": [{"price": 0.05, "quantity": 700}], "asks": [{"price": 0.06, "quantity": 300}]}
        raise AssertionError(path)


def _now():
    return datetime(2026, 10, 20, 23, 59, 59, tzinfo=timezone.utc)


def test_run_polls_bulk_and_cycles_cheap_books(tmp_path):
    reader = FakeReader()
    result = asyncio.run(Q.run(reader, ["1", "2", "3", "4"], books=1, stop=asyncio.Event(), max_polls=2, root=tmp_path, now=_now, poll_s=0.0))
    assert result == {"polls": 2, "lines": 8, "errors": 0}  # 3 quoted + 1 book, twice
    books = [c[0] for c in reader.calls if c[0].endswith("/orderbook")]
    assert books == ["/exchanges/1/orderbook", "/exchanges/3/orderbook"]  # cheap ones, cycling
    lines = [json.loads(l) for l in (tmp_path / "2026-10-20" / "snapshots.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {l["source"] for l in lines} == {"bulk", "book"} and next(l for l in lines if l["source"] == "book")["bid_size"] == 700.0
    assert all(l["exchange_id"] != "4" for l in lines)


def test_stop_event_and_errors(tmp_path):
    stop = asyncio.Event()
    stop.set()
    assert asyncio.run(Q.run(FakeReader(), ["1"], 0, stop, max_polls=5, root=tmp_path, poll_s=0.0))["polls"] == 0
    aborted = asyncio.run(Q.run(FakeReader(fail=ResearchAborted("429")), ["1"], 0, asyncio.Event(), max_polls=5, root=tmp_path, poll_s=0.0))
    assert aborted["polls"] == 0  # repeated 429: the collector stops
    failing = asyncio.run(Q.run(FakeReader(fail=ApiError(503, "DOWN", "x")), ["1"], 0, asyncio.Event(), max_polls=3, root=tmp_path, poll_s=0.0))
    assert failing == {"polls": 3, "lines": 0, "errors": 3}  # a failed poll does not end the run


def test_append_rolls_over_by_utc_date(tmp_path):
    Q.append_lines([Q.snapshot_line("2026-10-20T23:59:59.000Z", "1", 0.1, 0.2)], tmp_path)
    Q.append_lines([Q.snapshot_line("2026-10-21T00:00:29.000Z", "1", 0.1, 0.2)], tmp_path)
    assert (tmp_path / "2026-10-20" / "snapshots.jsonl").exists() and (tmp_path / "2026-10-21" / "snapshots.jsonl").exists()


def test_without_yes_nothing_is_read(monkeypatch, tmp_path, caplog):
    import config

    settings = config.Settings(sig_api_key="", sig_base_url="https://sig.invalid/api/v1", dry_run=True, log_level="INFO",
                               log_dir=tmp_path, tournament_slug="t", oracle_staleness_threshold_s=10.0, cycle_interval_s=15.0)
    monkeypatch.setattr(config, "load_settings", lambda *a, **k: settings)
    monkeypatch.setattr(Q, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(Q.MarketMap, "load", classmethod(lambda cls, *a, **k: MAP))
    monkeypatch.setattr(Q, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(Q, "research_reader", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no reads without --yes")))
    monkeypatch.delenv("RESEARCH_READS_PER_MIN", raising=False)
    with caplog.at_level("INFO"):
        assert Q.main(["--book-markets", "10"]) == 0
    plan = next(r for r in caplog.records if r.getMessage() == "Collector plan")
    assert plan.reads_per_hour == 600 and plan.books_per_poll == 4  # 2 ids -> 1 bulk read, 4 book reads per 30 s
    with pytest.raises(SystemExit):
        Q.main(["--book-markets", "26"])


def test_script_imports_no_order_placement_code():
    import ast

    forbidden = {"strategy", "quoter", "ladder", "main", "fills", "markout"}
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden
            if node.module == "exchange_api":
                assert {a.name for a in node.names} == {"ApiError"}
