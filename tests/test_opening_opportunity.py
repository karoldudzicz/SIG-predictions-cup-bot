"""scripts/opening_opportunity.py on synthetic raw data - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from market_mapper import MarketMap, MarketMapping, Race

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "opening_opportunity.py"
_spec = importlib.util.spec_from_file_location("opening_opportunity", SCRIPT)
O = importlib.util.module_from_spec(_spec)
sys.modules["opening_opportunity"] = O
_spec.loader.exec_module(O)


def _m(eid, state, party):
    return MarketMapping(eid, eid, f"{party} {state}", Race("senate", state), party, None, None)


MAP = MarketMap({m.sig_exchange_id: m for m in [_m("1", "AZ", "D"), _m("2", "AZ", "R"), _m("3", "ME", "D"), _m("4", "ME", "R"), _m("5", "ME", "I")]})


def _t(tid, hhmmss, price, size):
    return {"id": tid, "createdAt": f"2026-10-01T{hhmmss}Z", "price": price, "size": size, "side": "YES"}


RAW = [
    {"ts": "x", "path": "/tournaments/t/markets", "body": {"data": [{"id": "m", "title": "Arizona Senate", "exchanges": [{"id": "1"}, {"id": "2"}]}]}},
    # exchange 1: complete history from the start (no `from`), opens at 0.30, closes ~0.50
    {"ts": "x", "path": "/exchanges/1/trades", "params": {"limit": 200}, "pagination": None, "body": {
        "data": [_t("a", "16:01:00", 0.30, 100), _t("b", "16:30:00", 0.05, 10), _t("c", "17:55:00", 0.50, 200), _t("z", "18:05:00", 0.9, 999)],
        "pagination": {"hasMore": False}}},
    # exchange 2: only a later page fetched (hasMore true) -> not covered
    {"ts": "x", "path": "/exchanges/2/trades", "params": {"limit": 200}, "body": {
        "data": [_t("d", "16:01:30", 0.75, 50), _t("e", "17:55:20", 0.52, 20)], "pagination": {"hasMore": True, "nextCursor": "c"}}},
    # exchange 3 / 4 / 5: a three-party race, window-limited fetch (from = 16:00Z) - covered
    {"ts": "x", "path": "/exchanges/3/trades", "params": {"from": "2026-10-01T16:00:00Z"}, "body": {"data": [_t("f", "16:10:00", 0.45, 10)], "pagination": {"hasMore": False}}},
    {"ts": "x", "path": "/exchanges/4/trades", "params": {"from": "2026-10-01T16:00:00Z"}, "body": {"data": [_t("g", "16:10:30", 0.40, 10)], "pagination": {"hasMore": False}}},
    {"ts": "x", "path": "/exchanges/1/trades", "params": {"limit": 200}, "body": {"data": [_t("a", "16:01:00", 0.30, 100)], "pagination": {"hasMore": False}}},  # duplicate
    {"ts": "x", "path": "/exchanges/9/trades", "error": {"status": 500}},
]


def test_window_tapes_dedupe_coverage_and_titles():
    tapes, covered, titles = O.window_tapes(RAW)
    assert [t["id"] for t in tapes["1"]] == ["a", "b", "c"]  # 18:05 is outside, duplicate dropped
    assert covered == {"1", "3", "4"} and titles["1"] == "Arizona Senate"


def test_close_vwap_uses_last_non_empty_bucket():
    tapes, _, _ = O.window_tapes(RAW)
    assert O.close_vwap(tapes["1"]) == (pytest.approx(0.50), "17:50")
    assert O.close_vwap(tapes["3"]) == (pytest.approx(0.45), "16:10")
    assert O.close_vwap([]) == (None, None)


def test_ranking_rows_deviation_volume_and_ranks():
    tapes, covered, titles = O.window_tapes(RAW)
    rows = {r["exchange_id"]: r for r in O.ranking_rows(tapes, covered, titles, {"1": 0.60}, MAP)}
    one = rows["1"]
    # vs close 0.50: 100*0.20 + 10*0.45 + 200*0 = 24.5; vs mid 0.60: 30 + 5.5 + 20 = 55.5
    assert one["dev_vs_close"] == pytest.approx(24.5) and one["dev_vs_mid"] == pytest.approx(55.5)
    assert one["rank_close"] == 1 and one["rank_mid"] == 1 and one["extreme_share"] == pytest.approx(1 / 3, abs=1e-4)
    assert one["dev_vs_close_first_half"] == pytest.approx(24.5) and one["dev_vs_close_second_half"] == pytest.approx(0.0)
    assert rows["2"]["covered"] is False and rows["2"]["dev_vs_mid"] is None and "rank_mid" not in rows["2"]
    assert one["race"] == "senate:AZ" and one["title"] == "Arizona Senate"


def test_race_pairs_two_party_and_independent_flag():
    tapes, _, _ = O.window_tapes(RAW)
    summary, pairs = O.race_pairs(tapes, MAP)
    az = next(s for s in summary if s["race"] == "senate:AZ")
    # D 16:01:00 0.30 <-> R 16:01:30 0.75 = 1.05; D 16:30 has no R within 60 s; D 17:55:00 0.50 <-> R 17:55:20 0.52 = 1.02
    assert az["pairs"] == 2 and az["max_sum"] == pytest.approx(1.05) and az["share_above"] == 1.0 and az["has_independent"] is False
    me = next(s for s in summary if s["race"] == "senate:ME")
    assert me["has_independent"] is True and me["median_sum"] == pytest.approx(0.85) and me["share_below"] == 1.0
    first = next(p for p in pairs if p["race"] == "senate:AZ")
    assert (first["size_d"], first["size_r"], first["gap_s"]) == (100, 50, 30.0)


def test_analyse_summary_labels_proxies():
    result = O.analyse(RAW, {}, MAP)
    s = result["summary"]
    assert s["label"].startswith("PROXY") and s["exchanges_with_window_trades"] == 4 and s["covered"] == 3
    assert s["pairs"]["two_party_pairs"] == 2 and s["pairs"]["two_party_share_away_from_1"] == 1.0
    assert s["top_by_mid"] == []  # offline: no mid
    assert result["extreme_pairs"][0]["excess"] == pytest.approx(-0.15)  # the ME pair (with an Independent) is furthest from 1
    assert O.summary_markdown(result).startswith("# Opening opportunity")


def test_main_offline(monkeypatch, tmp_path):
    import config

    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(r) for r in RAW) + "\n", encoding="utf-8")
    settings = config.Settings(sig_api_key="", sig_base_url="https://sig.invalid/api/v1", dry_run=True, log_level="INFO",
                               log_dir=tmp_path, tournament_slug="t", oracle_staleness_threshold_s=10.0, cycle_interval_s=15.0)
    monkeypatch.setattr(config, "load_settings", lambda *a, **k: settings)
    monkeypatch.setattr(O, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(O.MarketMap, "load", classmethod(lambda cls, *a, **k: MAP))
    monkeypatch.setattr(O, "run_dir", lambda script: tmp_path / "out")
    monkeypatch.setattr(O, "research_reader", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no read without --yes")))
    monkeypatch.delenv("RESEARCH_READS_PER_MIN", raising=False)
    (tmp_path / "out").mkdir()
    assert O.main(["--raw", str(raw)]) == 0
    for name in ("ranking.csv", "pairs_by_race.csv", "extreme_pairs.csv", "summary.json", "summary.md"):
        assert (tmp_path / "out" / name).exists(), name


def test_script_imports_no_order_placement_code():
    import ast

    forbidden = {"strategy", "quoter", "ladder", "main", "fills", "markout", "exchange_api"}
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden
