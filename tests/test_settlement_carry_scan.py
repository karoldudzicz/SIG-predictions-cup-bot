"""scripts/settlement_carry_scan.py with a fake GET-only reader and fake oracles - no network.

Run from the project root:  python -m pytest tests -v

Equity 10,000 and the default RiskLimits: contract 0.02 (200), race worst loss 0.02 (200),
cluster 0.03 (300).
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from market_mapper import MarketMap, MarketMapping, Race
from oracle_api import OracleQuote
from risk import RiskLimits

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "settlement_carry_scan.py"
_spec = importlib.util.spec_from_file_location("settlement_carry_scan", SCRIPT)
S = importlib.util.module_from_spec(_spec)
sys.modules["settlement_carry_scan"] = S
_spec.loader.exec_module(S)


def _m(eid, state, party):
    return MarketMapping(eid, f"m{eid}", f"{party} {state}", Race("senate", state), party, None, None)


MAP = MarketMap({m.sig_exchange_id: m for m in [_m("1", "RI", "D"), _m("2", "RI", "R"), _m("3", "AZ", "D"), _m("4", "AZ", "R")]})
RAW_MAP = {"entries": {
    "1": {"polymarket": {"label": "Jack Reed (D)", "token_id": "pm1"}, "kalshi": {"ticker": "SENATERI-26-D"}},
    "2": {"polymarket": {"label": "Allan Fung (R)", "token_id": "pm2"}},
    "3": {"kalshi": {"ticker": "SENATEAZ-26-D"}},
    "4": {"polymarket": {"label": "Kari Lake", "token_id": "pm4"}},  # no party suffix: dropped
}}
PRICES = {"1": {"exchangeId": "1", "bestBid": 0.88, "bestAsk": 0.90}, "2": {"exchangeId": "2", "bestBid": 0.06, "bestAsk": 0.08},
          "3": {"exchangeId": "3", "bestBid": 0.40, "bestAsk": 0.45}, "4": {"exchangeId": "4", "bestBid": 0.50, "bestAsk": 0.55}}
MIDS = {"1": {"polymarket": 0.96, "kalshi": 0.94}, "2": {"polymarket": 0.04}, "3": {"kalshi": 0.60}, "4": {"polymarket": 0.40}}
BOOKS = {"1": {"bids": [{"price": 0.88, "quantity": 100}], "asks": [{"price": 0.90, "quantity": 1000}, {"price": 0.91, "quantity": 500}, {"price": 0.95, "quantity": 100}]},
         "2": {"bids": [{"price": 0.06, "quantity": 5000}], "asks": [{"price": 0.08, "quantity": 10}]},
         "3": {"bids": [{"price": 0.40, "quantity": 10}], "asks": [{"price": 0.45, "quantity": 100}, {"price": 0.46, "quantity": 1000}]}}
OPEN_ORDERS = [{"exchangeId": "3", "side": "yes", "action": "sell", "priceLimit": 0.46, "remainingQuantity": 200, "open": True}]  # ours: not fillable
POSITIONS = [{"exchangeId": "1", "quantity": 100, "costBasis": 90.0}, {"exchangeId": "4", "quantity": 50, "costBasis": 20.0}]
NOW = datetime(2026, 10, 13, 17, 0, tzinfo=timezone.utc)
CLUSTERS = {"1": "senate:RI", "2": "senate:RI", "3": "senate:AZ", "4": "senate:AZ"}


def _levels():
    books = {eid: S.book_without_own(dict(b), S.own_orders(OPEN_ORDERS).get(eid, [])) for eid, b in BOOKS.items()}
    return {(eid, side): S.stats.side_levels(books[eid], side) for eid in books for side in ("yes", "no")}


def test_evaluate_candidates_drops_and_ranking():
    cands, dropped = S.evaluate(MAP, RAW_MAP, PRICES, MIDS, NOW)
    assert [(c["exchange_id"], c["side"]) for c in cands] == [("3", "yes"), ("1", "yes"), ("2", "no")]  # by yield at haircut 0.02
    assert dropped == {"oracle party missing or different": 1}
    one = next(c for c in cands if c["exchange_id"] == "1")
    assert one["oracle_p_yes"] == 0.95 and one["oracle_gap"] == pytest.approx(0.02) and one["oracle_disagree"] is False
    assert one["edge_0"] == pytest.approx(0.05) and one["edge_0.04"] == pytest.approx(0.01) and one["days_to_settlement"] == 22.0
    two = next(c for c in cands if c["exchange_id"] == "2")
    assert two["price"] == pytest.approx(0.94) and two["oracle_p_side"] == pytest.approx(0.96)


def test_allocation_caps_greedy_sharing_and_flags():
    cands, _ = S.evaluate(MAP, RAW_MAP, PRICES, MIDS, NOW)
    S.allocate(cands, _levels(), POSITIONS, MAP, CLUSTERS, 10_000.0, RiskLimits())
    az, ri_d, ri_r = cands
    # AZ D YES: our 200-share ask at 0.46 is taken out (1000 -> 800); budget = contract room 200;
    # race cap: "none of the listed wins" -> -20 - 0.4589 q >= -200 -> q <= 392.
    assert az["depth_shares"] == 900 and az["contract_room_cost"] == 200 and az["race_room_shares"] == 392
    assert az["allocated_shares"] == 392 and az["allocated_cost"] == pytest.approx(100 * 0.45 + 292 * 0.46)
    assert az["ev_0.02"] == pytest.approx(392 * 0.58 - (45 + 292 * 0.46), abs=0.01)
    assert az["same_race_opposite"] is True and az["held"] is False  # we hold R YES in AZ
    # RI D YES: we already hold 100 @ 0.90 -> contract room 110; race cap leaves 121 shares.
    assert ri_d["held"] is True and ri_d["contract_room_cost"] == 110 and ri_d["allocated_shares"] == 121
    # RI R NO is the same bet as D YES (no opposite flag); after D's allocation the race cap leaves 1 share.
    assert ri_r["same_race_opposite"] is False and ri_r["allocated_shares"] == 1 and ri_r["standalone_shares"] == 117
    tot = S.totals(cands, worst_n=2)
    assert tot["allocated"] == 3 and tot["largest"] == ["3 yes", "1 yes"]
    assert tot["loss_if_largest_all_lose"] == pytest.approx(-(az["allocated_cost"] + ri_d["allocated_cost"]), abs=0.01)
    assert tot["worst_cluster"] == "senate:AZ" and tot["capital_locked"] == pytest.approx(az["allocated_cost"] + ri_d["allocated_cost"] + ri_r["allocated_cost"], abs=0.01)


def test_candidates_without_a_book_get_no_capacity():
    cands, _ = S.evaluate(MAP, RAW_MAP, PRICES, MIDS, NOW)
    S.allocate(cands, {}, POSITIONS, MAP, CLUSTERS, 10_000.0, RiskLimits())
    assert all(c["book_read"] is False and "allocated_shares" not in c for c in cands)
    assert S.totals(cands)["allocated"] == 0


# ---------------------------------------------------------------------- run with fakes


class FakeReader:
    def __init__(self):
        self.out_dir = Path(".")
        self.calls = []
        self.saved = []
        self.raw_log = type("Log", (), {"append": lambda _self, rec: self.saved.append(rec)})()

    async def get(self, path, params=None):
        self.calls.append((path, dict(params or {})))
        if path == "/tournaments/t":
            return {"id": "tid", "initialBalance": 10_000}
        if path == "/exchanges/prices":
            return {"data": [PRICES[i] for i in params["ids"].split(",") if i in PRICES]}
        if path == "/orders":
            return {"data": OPEN_ORDERS, "pagination": {"hasMore": False}}
        if path.endswith("/orderbook"):
            assert params["depth"] == 50
            return BOOKS[path.split("/")[2]]
        if path == "/portfolio/positions":
            return {"positions": POSITIONS}
        raise AssertionError(f"unexpected GET {path}")


class FakeOracle:
    def __init__(self, quotes):
        self.quotes = quotes

    async def get_quotes(self, refs):
        return {r: q for r, q in self.quotes.items() if r in refs}


def _q(ref, mid):
    return OracleQuote("polymarket", ref, mid - 0.01, mid + 0.01, 100, 100, None, 0.0)


def test_run_reads_within_budget_and_falls_back_to_live_positions(monkeypatch):
    monkeypatch.setattr(S, "saved_review", lambda root=None: (None, [], None))
    pm = FakeOracle({"pm1": _q("pm1", 0.96), "pm2": _q("pm2", 0.04), "pm4": _q("pm4", 0.40)})
    ks = FakeOracle({"SENATERI-26-D": _q("k1", 0.94), "SENATEAZ-26-D": _q("k3", 0.60)})
    reader = FakeReader()
    result = asyncio.run(S.run(reader, "t", MAP, RAW_MAP, pm, ks, top_books=2))
    paths = [c[0] for c in reader.calls]
    assert paths.count("/exchanges/prices") == 1 and paths.count("/orders") == 1
    assert sorted(p for p in paths if p.endswith("/orderbook")) == ["/exchanges/1/orderbook", "/exchanges/3/orderbook"]  # top 2 only
    assert "/portfolio/positions" in paths and result["sources"]["positions_from"].startswith("read now")
    assert reader.saved[0]["path"] == S.ORACLE_RECORD_PATH
    assert result["totals"]["books_read"] == 2 and result["totals"]["allocated"] == 2
    md = S.summary_markdown(result["candidates"], result["totals"], result["dropped"], result["sources"])
    assert md.startswith("# Settlement carry scan (HYPOTHETICAL)") and "not the truth" in md


def test_saved_review_reads_positions_and_relationships(tmp_path):
    run = tmp_path / "2026-10-12"
    run.mkdir()
    (run / "raw.jsonl").write_text(
        '{"ts": "t1", "path": "/portfolio/positions", "body": {"positions": [{"exchangeId": "1", "quantity": 5}]}}\n'
        '{"ts": "t2", "path": "/relationships", "params": {"cursor": null}, "body": {"data": [{"id": "r"}]}}\n', encoding="utf-8")
    positions, relationships, source = S.saved_review(tmp_path)
    assert positions == [{"exchangeId": "1", "quantity": 5}] and relationships == [{"id": "r"}] and "t1" in source
    assert S.saved_review(tmp_path / "none") == (None, [], None)


def test_without_yes_nothing_is_fetched(monkeypatch, tmp_path, caplog):
    import config

    settings = config.Settings(sig_api_key="", sig_base_url="https://sig.invalid/api/v1", dry_run=True, log_level="INFO",
                               log_dir=tmp_path, tournament_slug="t", oracle_staleness_threshold_s=10.0, cycle_interval_s=15.0)
    monkeypatch.setattr(config, "load_settings", lambda *a, **k: settings)
    monkeypatch.setattr(S, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(S.MarketMap, "load", classmethod(lambda cls, *a, **k: MAP))
    map_path = tmp_path / "market_map.json"
    map_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(S, "MARKET_MAP_PATH", map_path)
    monkeypatch.setattr(S, "OUT_ROOT", tmp_path / "edge_scan")
    monkeypatch.setattr(S, "research_reader", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no reader without --yes")))
    monkeypatch.delenv("RESEARCH_READS_PER_MIN", raising=False)
    with caplog.at_level("INFO"):
        assert S.main([]) == 0
    planned = next(r for r in caplog.records if r.getMessage() == "Planned SIG reads")
    assert planned.reads == 1 + 1 + 1 + 60 + 1  # tournament, prices, orders, books, positions (none saved)


def test_script_imports_no_order_placement_code():
    import ast

    forbidden = {"strategy", "quoter", "ladder", "main", "fills", "markout"}
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden
            if node.module == "exchange_api":
                assert {a.name for a in node.names} <= {"book_without_own", "yes_terms"}  # pure helpers only
