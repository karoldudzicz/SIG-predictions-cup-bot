"""scripts/opening_analysis.py on synthetic raw / journal data - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from market_mapper import MarketMap, MarketMapping, Race

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "opening_analysis.py"
_spec = importlib.util.spec_from_file_location("opening_analysis", SCRIPT)
O = importlib.util.module_from_spec(_spec)
sys.modules["opening_analysis"] = O
_spec.loader.exec_module(O)

MAP = MarketMap({m.sig_exchange_id: m for m in [
    MarketMapping("1", "1", "D AZ", Race("senate", "AZ"), "D", None, None),
    MarketMapping("2", "2", "R AZ", Race("senate", "AZ"), "R", None, None),
]})


def _trade(tid, ts, price, size, side="YES"):
    return {"id": tid, "createdAt": ts, "price": price, "size": size, "side": side, "volume": size * (price if side == "YES" else 1 - price)}


RAW = [
    {"ts": "x", "path": "/tournaments/midterm-elections", "body": {"id": "tid"}},
    {"ts": "x", "path": "/exchanges/1/trades", "params": {}, "body": {"data": [
        _trade("a", "2026-10-01T16:01:00Z", 0.40, 100), _trade("b", "2026-10-01T16:05:00Z", 0.50, 300),
        _trade("c", "2026-10-01T17:55:00Z", 0.05, 5000, "NO"), _trade("d", "2026-10-01T18:00:00Z", 0.60, 9999),  # d: outside
    ]}},
    {"ts": "x", "path": "/exchanges/1/trades", "params": {}, "body": {"data": [_trade("a", "2026-10-01T16:01:00Z", 0.40, 100)]}},  # duplicate page
    {"ts": "x", "path": "/exchanges/2/trades", "params": {}, "body": {"data": [_trade("e", "2026-10-01T16:30:00Z", 0.55, 50)]}},
    {"ts": "x", "path": "/exchanges/3/trades", "error": {"status": 500}},
]


def test_tape_from_raw_window_dedupe_and_order():
    tapes = O.tape_from_raw(RAW)
    assert [t["id"] for t in tapes["1"]] == ["a", "b", "c"]
    assert [t["id"] for t in tapes["2"]] == ["e"] and "3" not in tapes
    assert O.tournament_id_from_raw(RAW) == "tid"


def test_bucket_rows_vwap_and_extreme_share():
    rows = O.bucket_rows("1", O.tape_from_raw(RAW)["1"], MAP)
    assert len(rows) == 12 and rows[0]["bucket_start"] == "16:00" and rows[11]["bucket_start"] == "17:50"
    assert rows[0]["trades"] == 2 and rows[0]["vwap"] == pytest.approx(0.475) and rows[0]["extreme_share"] == 0.0
    assert rows[11]["vwap"] == pytest.approx(0.05) and rows[11]["extreme_share"] == 1.0
    assert rows[5]["trades"] == 0 and rows[5]["vwap"] is None and rows[5]["extreme_share"] is None


def test_largest_prints_and_notional():
    prints = O.largest_prints(O.tape_from_raw(RAW), MAP, n=2)
    assert [p["size"] for p in prints] == [5000, 300]
    assert prints[0]["size_x_price_yes"] == 250.0 and prints[0]["api_volume"] == pytest.approx(4750.0)  # NO-space print
    assert prints[0]["race"] == "senate:AZ"


def test_drift_uses_first_and_last_non_empty_buckets():
    rows = O.bucket_rows("1", O.tape_from_raw(RAW)["1"], MAP)
    d = O.drift_row("1", rows, 0.08, MAP)
    assert (d["first_bucket"], d["last_bucket"]) == ("16:00", "17:50")
    assert d["last_minus_first"] == pytest.approx(-0.425) and d["mid_minus_last"] == pytest.approx(0.03)
    assert O.drift_row("1", rows, None, MAP)["mid_minus_last"] is None
    empty = O.drift_row("9", O.bucket_rows("9", [], MAP), 0.5, MAP)
    assert empty["first_vwap"] is None and empty["mid_minus_first"] is None


# ---------------------------------------------------------------------- our orders


def test_fair_from_journal_exact_and_fitted_range():
    assert O.parse_edge("edge 0.070") == 0.07 and O.parse_edge("edge -0.010; fitted") == -0.01 and O.parse_edge("flip") is None
    exact = O.fair_from_journal(0.42, "edge 0.070")
    assert exact["fair_side"] == 0.49 and exact["fitted"] is False
    fitted = O.fair_from_journal(0.45, "edge 0.070; fitted to book 300/500")
    assert fitted["fair_side"] is None and (fitted["fair_side_low"], fitted["fair_side_high"]) == (0.47, 0.52)
    assert O.fair_from_journal(None, "edge 0.07")["fair_side"] is None


JOURNAL = [
    {"ts": "2026-10-01T16:05:00.000+00:00", "key": "k1", "status": "pending", "kind": "increase", "reason": "edge 0.060",
     "payload": {"exchangeId": "1", "side": "no", "action": "buy", "quantity": 1000, "price": 0.52}},
    {"ts": "2026-10-01T16:05:01.000+00:00", "key": "k1", "status": "accepted", "kind": "increase", "reason": "edge 0.060",
     "payload": {"exchangeId": "1", "side": "no", "action": "buy", "quantity": 1000, "price": 0.52},
     "result": {"orderId": "o1", "side": "no", "action": "buy", "quantityTraded": -800, "fillPrice": 0.49, "totalCost": 408.0}},  # fillPrice in YES terms here
    {"ts": "2026-10-01T15:59:00.000+00:00", "key": "k0", "status": "dry_run", "kind": "increase", "reason": "edge 0.03", "payload": {"exchangeId": "2", "price": 0.5}},
    {"ts": "2026-10-01T19:00:00.000+00:00", "key": "k9", "status": "accepted", "kind": "increase", "reason": "edge 0.03", "payload": {"exchangeId": "2", "price": 0.5}},
]


def test_journal_orders_window_by_first_entry_and_latest_state():
    orders = O.journal_orders(JOURNAL)
    assert [o["key"] for o in orders] == ["k1"] and orders[0]["status"] == "accepted"


def test_our_fill_rows_fair_vs_fill_in_side_terms():
    rows = O.our_fill_rows(O.journal_orders(JOURNAL), {"o1": 0.43}, MAP)
    r = rows[0]
    assert r["quantity_traded"] == 800 and r["avg_cost_price"] == 0.51 and r["fill_price"] == 0.49
    assert r["fair_side"] == 0.58 and r["fair_yes"] == 0.42  # NO side: fair_yes = 1 - fair_side
    assert r["fill_vs_fair_side"] == pytest.approx(0.07) and r["markout_fair_yes"] == 0.43


def test_markout_fairs_reads_unix_filled_at():
    t = datetime(2026, 10, 1, 16, 30, tzinfo=timezone.utc).timestamp()
    entries = [{"event": "fill", "order_id": "o1", "filled_at": t, "fair_at_fill": 0.4},
               {"event": "fill", "order_id": "o2", "filled_at": t + 3 * 3600, "fair_at_fill": 0.5},
               {"event": "markout", "order_id": "o3", "filled_at": t, "fair_at_fill": 0.6}]
    assert O.markout_fairs(entries) == {"o1": 0.4}


def test_analyse_summary_and_markdown():
    tapes = O.tape_from_raw(RAW)
    result = O.analyse(tapes, {"1": 0.08, "2": 0.5}, O.journal_orders(JOURNAL), {}, MAP)
    s = result["summary"]
    assert s["markets"] == 2 and s["trades"] == 4 and s["drift"]["markets_with_mid"] == 2
    assert s["pooled_by_bucket"][11] == {"bucket_start": "17:50", "trades": 1, "extreme_share": 1.0}
    assert s["our_orders"]["with_fills"] == 1 and s["our_orders"]["fair_exact"] == 1
    md = O.summary_markdown(result)
    assert md.startswith("# Opening tape 2026-10-01T16:00") and "5000 shares" in md


def test_main_offline_without_yes_writes_outputs(tmp_path, monkeypatch, caplog):
    import config

    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(r) for r in RAW) + "\n", encoding="utf-8")
    journal = tmp_path / "order_journal.jsonl"
    journal.write_text("\n".join(json.dumps(r) for r in JOURNAL) + "\n", encoding="utf-8")
    settings = config.Settings(sig_api_key="", sig_base_url="https://sig.invalid/api/v1", dry_run=True, log_level="INFO",
                               log_dir=tmp_path, tournament_slug="t", oracle_staleness_threshold_s=10.0, cycle_interval_s=15.0)
    monkeypatch.setattr(config, "load_settings", lambda *a, **k: settings)
    monkeypatch.setattr(O, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(O.MarketMap, "load", classmethod(lambda cls, *a, **k: MAP))
    monkeypatch.setattr(O, "ORDER_JOURNAL_PATH", journal)
    monkeypatch.setattr(O, "MARKOUT_PATH", tmp_path / "absent.jsonl")
    monkeypatch.setattr(O, "run_dir", lambda script: tmp_path / "out")
    monkeypatch.setattr(O, "research_reader", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no read without --yes")))
    monkeypatch.delenv("RESEARCH_READS_PER_MIN", raising=False)
    (tmp_path / "out").mkdir()
    assert O.main(["--raw", str(raw)]) == 0
    for name in ("buckets.csv", "largest_prints.csv", "drift.csv", "our_fills.csv", "summary.json", "summary.md"):
        assert (tmp_path / "out" / name).exists(), name
    assert json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))["drift"]["markets_with_mid"] == 0


def test_script_imports_no_order_placement_code():
    import ast

    forbidden = {"strategy", "quoter", "ladder", "main", "fills", "markout", "exchange_api"}
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden


def test_entry_falls_back_to_the_canonical_limit():
    """The real journal keeps no totalCost / fillPrice: compare with the canonical limit,
    converted when the engine flipped the side."""
    journal = [
        {"ts": "2026-10-01T16:10:00+00:00", "key": "a", "status": "accepted", "kind": "increase", "reason": "edge 0.050",
         "payload": {"exchangeId": "1", "side": "yes", "action": "buy", "quantity": 100, "price": 0.40},
         "result": {"orderId": "oa", "side": "yes", "action": "buy", "price": 0.40, "quantityTraded": 100, "remainingQuantity": 0}},
        {"ts": "2026-10-01T16:11:00+00:00", "key": "b", "status": "accepted", "kind": "increase", "reason": "edge 0.050",
         "payload": {"exchangeId": "1", "side": "no", "action": "buy", "quantity": 100, "price": 0.60},
         "result": {"orderId": "ob", "side": "yes", "action": "sell", "price": 0.40, "quantityTraded": -100, "remainingQuantity": 0}},
        {"ts": "2026-10-01T16:12:00+00:00", "key": "c", "status": "accepted", "kind": "increase", "reason": "edge 0.050",
         "payload": {"exchangeId": "1", "side": "yes", "action": "buy", "quantity": 100, "price": 0.40},
         "result": {"orderId": "oc", "side": "yes", "action": "buy", "price": 0.40, "quantityTraded": 0, "remainingQuantity": 100}},
    ]
    rows = O.our_fill_rows(O.journal_orders(journal), {}, MAP)
    assert (rows[0]["entry_price"], rows[0]["entry_basis"], rows[0]["fill_vs_fair_side"]) == (0.40, "limit (fill at or better)", 0.05)
    assert rows[1]["entry_price"] == 0.60 and rows[1]["fill_vs_fair_side"] == 0.05  # canonical YES 0.40 -> NO 0.60
    assert rows[2]["entry_price"] is None and rows[2]["fill_vs_fair_side"] is None  # nothing traded at placement
