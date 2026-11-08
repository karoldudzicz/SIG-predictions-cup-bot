"""scripts/leaderboard_analysis.py on synthetic data with a fake GET-only reader - no network.

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

from market_mapper import MarketMap, MarketMapping, Race

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "leaderboard_analysis.py"
_spec = importlib.util.spec_from_file_location("leaderboard_analysis", SCRIPT)
L = importlib.util.module_from_spec(_spec)
sys.modules["leaderboard_analysis"] = L
_spec.loader.exec_module(L)

ME = "me-uuid-0"


def _m(eid, state, party):
    return MarketMapping(eid, eid, f"{party} {state}", Race("senate", state), party, None, None)


MAP = MarketMap({m.sig_exchange_id: m for m in [_m("1", "AZ", "D"), _m("2", "AZ", "R"), _m("3", "GA", "D"), _m("4", "GA", "R"), _m("5", "NV", "D")]})
PRICES = {
    "1": {"exchangeId": "1", "bestBid": 0.48, "bestAsk": 0.52},  # mid 0.50
    "2": {"exchangeId": "2", "bestBid": 0.40, "bestAsk": 0.44},  # mid 0.42
    "3": {"exchangeId": "3", "bestBid": 0.10, "bestAsk": 0.12},  # mid 0.11
    "4": {"exchangeId": "4", "bestBid": None, "bestAsk": 0.95},  # one-sided
    "5": {"exchangeId": "5", "bestBid": None, "bestAsk": None},  # empty
}


def _lb_rows(period):
    """25 participants; we are rank 3 all-time. pnl_all = 100 * (25 - i) - 1200."""
    rows = []
    for i in range(25):
        pid = ME if i == 2 else f"anon-uuid-{i}"
        pnl_all = 100.0 * (25 - i) - 1200.0
        pnl = {"all": pnl_all, "7d": pnl_all / 2, "1d": pnl_all / 10}[period]
        rows.append({"rank": i + 1, "profileId": pid, "username": f"handle-{i}", "pnl": pnl, "tradesCount": 10 + i})
    if period == "1d":
        rows = [r for r in rows if r["profileId"] != "anon-uuid-24"]  # did not trade today
    return sorted(rows, key=lambda r: -r["pnl"])


class FakeReader:
    def __init__(self, tmp_path, tape=None, books=None, positions=None):
        self.out_dir = tmp_path
        self.calls = []
        self.tape = tape or {}
        self.books = books or {}
        self.positions = positions or []

    def stats(self):
        return {"requests": len(self.calls)}

    async def get(self, path, params=None):
        params = dict(params or {})
        self.calls.append((path, params))
        if path == "/tournaments/t":
            return {"id": "tid", "initialBalance": 10000.0, "myBalance": 9500.0, "memberCount": 25}
        if path == "/account":
            return {"id": ME}
        if path == "/tournaments/t/leaderboard":
            rows = _lb_rows(params["period"])
            page = rows[params["offset"] : params["offset"] + params["limit"]]
            return {"leaderboard": page, "total": len(rows), "myRank": 3}
        if path == "/exchanges/prices":
            return {"data": [PRICES[i] for i in params["ids"].split(",") if i in PRICES]}
        if path == "/orders":
            return {"data": [{"exchangeId": "1", "side": "yes", "action": "buy", "priceLimit": 0.30, "remainingQuantity": 1000, "open": True}],
                    "pagination": {"hasMore": False}}
        if path == "/portfolio/positions":
            return {"positions": self.positions}
        if path == "/portfolio/pnl":
            return {"totalAccountValue": 9500.0 + 42.0, "totalHoldingsValue": 42.0}
        if path.endswith("/trades"):
            return {"data": self.tape.get(path.split("/")[2], []), "pagination": {"hasMore": False}}
        if path.endswith("/orderbook"):
            return self.books.get(path.split("/")[2], {"bids": [], "asks": []})
        raise AssertionError(f"unexpected GET {path}")


# ---------------------------------------------------------------------- pure parts


def test_estimate_reads_counts_every_part():
    # 2 + 3 * 10 pages + 3 price chunks + 1 orders + 5 snapshot + 10 * 2 tape + 237 books
    assert L.estimate_reads(1000, 237, 237, 10) == 2 + 30 + 3 + 1 + 5 + 20 + 237
    assert L.estimate_reads(1000, 237, 20, 10) == 2 + 30 + 3 + 1 + 5 + 20 + 20


def test_analyse_leaderboard_decomposition_and_no_identities():
    boards = {p: {"rows": _lb_rows(p), "total": 25, "my_rank": 3} for p in L.PERIODS}
    summary, top, top_1d = L.analyse_leaderboard(boards, ME)
    assert len(top) == 20 and top[0]["rank"] == 1
    first = top[0]  # pnl_all 1300, 7d 650, 1d 130
    assert (first["early"], first["mid"], first["today"]) == (650.0, 520.0, 130.0)
    assert sum(1 for r in top if r["is_us"]) == 1
    assert summary["all"]["my_pnl"] == 1100.0 and summary["all"]["my_rank"] == 3
    assert summary["all"]["zero_sum"]["participants"] == 25
    # No usernames / profile ids leave the analysis.
    blob = json.dumps([summary, top, top_1d])
    assert "handle-" not in blob and "anon-uuid" not in blob and ME not in blob
    assert summary["top_decomposition"]["missing_parts"] == 0
    assert top_1d[0]["rank_all"] == 1


def test_missing_from_a_period_is_none_not_zero():
    boards = {p: {"rows": _lb_rows(p), "total": 25, "my_rank": 3} for p in L.PERIODS}
    _, top, _ = L.analyse_leaderboard(boards, ME, top_n=25)
    last = top[-1]  # anon-uuid-24 is absent from 1d
    assert last["pnl_1d"] is None and last["today"] is None and last["mid"] is None and last["early"] is not None


def test_contested_first_and_book_scan_order():
    assert L.contested_first(MAP, PRICES) == ["1", "2", "3"]
    assert L.book_scan_order(MAP, PRICES) == ["1", "2", "3", "4"]  # one-sided last, empty skipped


def test_summarise_trades_tick_proxy_and_halves():
    trades = [  # newest first, as the API returns them
        {"id": "4", "createdAt": "2026-10-01T16:40:00Z", "price": 0.49, "size": 3000, "side": "NO"},
        {"id": "3", "createdAt": "2026-10-01T16:30:00Z", "price": 0.52, "size": 10, "side": "YES"},
        {"id": "2", "createdAt": "2026-10-01T16:20:00Z", "price": 0.51, "size": 100, "side": "YES"},
        {"id": "1", "createdAt": "2026-10-01T16:10:00Z", "price": 0.50, "size": 50, "side": "YES"},
    ]
    out = L.summarise_trades(trades)
    assert out["trades"] == 4 and out["shares"] == 3160
    assert out["proxy_buy_share"] == pytest.approx(2 / 3, abs=1e-4) and out["proxy_uncalled"] == 1
    assert out["yes_space_share"] == 0.75 and out["max_size"] == 3000
    assert out["size_buckets"]["3000+"]["count"] == 1
    summary, rows = L.summarise_tape({"1": trades}, MAP)
    assert summary["first_half"]["trades"] == 2 and summary["second_half"]["trades"] == 2
    assert summary["halves_split_at"] == "2026-10-01T16:30:00Z"
    assert rows[0]["race"] == "senate:AZ"


def test_big_levels_exclude_our_own_size():
    book = {"bids": [{"price": 0.48, "quantity": 200}, {"price": 0.30, "quantity": 3500}], "asks": [{"price": 0.52, "quantity": 3000}]}
    own = L.own_orders_by_exchange([{"exchangeId": "1", "side": "yes", "action": "buy", "priceLimit": 0.30, "remainingQuantity": -1000, "open": True}])
    rows = L.big_levels_rows("1", book, own["1"], MAP)
    assert [(r["side"], r["price"], r["quantity"]) for r in rows] == [("ask", 0.52, 3000.0)]  # 3500 - our 1000 < 3000


def test_valuation_check_picks_the_matching_rule():
    positions = [
        {"exchangeId": "2", "quantity": 100, "marketValue": 42.0, "settled": False},  # YES: bid 40, mid 42
        {"exchangeId": "1", "quantity": -50, "marketValue": 25.0, "settled": False},  # NO: bid 24, mid 25
    ]
    # Leaderboard implied value = cash + holdings at mid = 9500 + 42 + 25.
    check = L.valuation_check(10000.0, -433.0, 9500.0, positions, PRICES, 9567.0)
    assert check["holdings"]["bid"] == pytest.approx(64.0) and check["holdings"]["mid"] == pytest.approx(67.0)
    assert check["closest_rule"] in ("cash+mid", "cash+sig_valuation_price", "portfolio_pnl_totalAccountValue")
    assert check["residuals"]["cash+bid"] == pytest.approx(-3.0)
    unpriced = L.valuation_check(10000.0, 0.0, 9500.0, [{"exchangeId": "4", "quantity": 10, "marketValue": 9.0}], PRICES, None)
    assert unpriced["candidates"]["cash+bid"] is None and unpriced["unvalued_positions"]["bid"] == 1
    assert unpriced["closest_rule"] == "cash+sig_valuation_price"
    assert L.valuation_check(10000.0, None, 9500.0, positions, PRICES, None)["closest_rule"] is None


# ---------------------------------------------------------------------- full run


def test_run_end_to_end_reads_only_and_writes_outputs(tmp_path):
    tape = {"1": [{"id": "1", "createdAt": "2026-10-01T16:10:00Z", "price": 0.5, "size": 5, "side": "YES"}]}
    books = {"2": {"bids": [{"price": 0.40, "quantity": 4000}], "asks": [{"price": 0.44, "quantity": 10}]}}
    positions = [{"exchangeId": "2", "quantity": 100, "marketValue": 42.0, "settled": False}]
    reader = FakeReader(tmp_path, tape=tape, books=books, positions=positions)
    summary = asyncio.run(L.run(reader, "t", MAP, max_books=10, tape_exchanges=None))

    # Leaderboard: 25 rows (24 on 1d) at 100 per page -> one page per period.
    lb_calls = [c for c in reader.calls if c[0] == "/tournaments/t/leaderboard"]
    assert [c[1]["period"] for c in lb_calls] == ["all", "7d", "1d", "all"]  # + our row in the snapshot
    assert summary["leaderboard"]["all"]["rows_read"] == 25 and summary["leaderboard"]["1d"]["rows_read"] == 24
    assert summary["valuation"]["leaderboard_pnl_source"] == "re-read in the snapshot"
    assert summary["valuation"]["implied_leaderboard_value"] == 10000.0 + 1100.0
    assert summary["tape"]["markets"] == 3  # only 3 contested markets exist in this map
    assert summary["big_levels"]["levels"] == 1
    trade_calls = [c for c in reader.calls if c[0].endswith("/trades")]
    assert all(c[1]["from"] == L.TAPE_FROM and c[1]["to"] == L.TAPE_TO and c[1]["tournamentId"] == "tid" for c in trade_calls)
    assert len([c for c in reader.calls if c[0].endswith("/orderbook")]) == 4

    for name in ("summary.json", "leaderboard_top20.csv", "leaderboard_top20_1d.csv", "tape_markets.csv", "big_levels.csv", "valuation.csv"):
        assert (tmp_path / name).exists(), name
    with (tmp_path / "big_levels.csv").open(encoding="utf-8") as fh:
        assert list(csv.DictReader(fh))[0]["quantity"] == "4000.0"
    assert "handle-" not in (tmp_path / "summary.json").read_text(encoding="utf-8")


@pytest.fixture
def quiet_main(monkeypatch):
    """main() without .env, logging reconfiguration or the real market map."""
    import config

    settings = config.Settings(sig_api_key="test-key", sig_base_url="https://sig.invalid/api/v1", dry_run=True, log_level="INFO",
                               log_dir=Path("."), tournament_slug="t", oracle_staleness_threshold_s=10.0, cycle_interval_s=15.0)
    monkeypatch.setattr(config, "load_settings", lambda *a, **k: settings)
    monkeypatch.setattr(L, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(L.MarketMap, "load", classmethod(lambda cls, *a, **k: MAP))
    monkeypatch.delenv("RESEARCH_READS_PER_MIN", raising=False)


def test_without_yes_nothing_is_fetched(quiet_main, monkeypatch, caplog):
    def boom(*a, **k):
        raise AssertionError("must not open a reader without --yes")

    monkeypatch.setattr(L, "research_reader", boom)
    with caplog.at_level("INFO"):
        assert L.main([]) == 0
    assert any(r.getMessage() == "Planned SIG reads" for r in caplog.records)


def test_reads_per_min_above_hard_max_is_refused(quiet_main):
    with pytest.raises(SystemExit):
        L.main(["--reads-per-min", "20"])


def test_script_imports_no_order_placement_code():
    import ast

    forbidden = {"strategy", "quoter", "ladder", "main", "fills", "markout"}
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] not in forbidden for a in node.names)
