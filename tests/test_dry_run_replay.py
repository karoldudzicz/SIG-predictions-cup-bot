"""scripts/dry_run_replay.py on a synthetic order journal - no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from market_mapper import MarketMap, MarketMapping, Race

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "dry_run_replay.py"
_spec = importlib.util.spec_from_file_location("dry_run_replay", SCRIPT)
R = importlib.util.module_from_spec(_spec)
sys.modules["dry_run_replay"] = R
_spec.loader.exec_module(R)

MAP = MarketMap({m.sig_exchange_id: m for m in [
    MarketMapping("1", "1", "D AZ", Race("senate", "AZ"), "D", None, None),
    MarketMapping("2", "2", "R GA", Race("senate", "GA"), "R", None, None),
]})
T0 = datetime(2026, 10, 1, 16, 0, tzinfo=timezone.utc)
SINCE = T0


def _entry(minutes, key, status, eid="1", side="yes", action="buy", price=0.40, qty=100, result=None, kind="increase"):
    e = {"ts": (T0 + timedelta(minutes=minutes)).isoformat(), "key": key, "status": status, "kind": kind, "reason": "edge 0.05",
         "payload": {"exchangeId": eid, "side": side, "action": action, "quantity": qty, "price": price}}
    if result is not None:
        e["result"] = result
    return e


# An intent repeated every 2 min for 30 min (16 entries), a different limit once, a NO intent on day 2,
# and two live orders (one canonicalized SELL -> BUY NO, one with nothing traded at placement).
JOURNAL = (
    [_entry(2 * i, f"d{i}", "dry_run") for i in range(16)]
    + [_entry(5, "x", "dry_run", price=0.41)]
    + [_entry(24 * 60 + 5, "n1", "dry_run", eid="2", side="no", price=0.70, qty=50)]
    + [_entry(-30, "old", "dry_run")]  # before the start: ignored
    + [_entry(3 * 24 * 60, "L1", "pending", eid="1", side="yes", action="sell", price=0.45, qty=200),
       _entry(3 * 24 * 60 + 0.01, "L1", "accepted", eid="1", side="yes", action="sell", price=0.45, qty=200,
              result={"orderId": "o1", "side": "no", "action": "buy", "price": 0.55, "quantityTraded": -150, "remainingQuantity": -50}),
       _entry(3 * 24 * 60 + 1, "L2", "accepted", eid="2", price=0.30, qty=10, result={"orderId": "o2", "side": "yes", "action": "buy", "price": 0.30, "quantityTraded": 0, "remainingQuantity": 10})]
)
MIDS = {"1": 0.50, "2": 0.25}


def _orders():
    return R.latest_by_key(JOURNAL, SINCE)


def test_latest_by_key_dates_by_first_entry_and_merges_state():
    orders = {o["key"]: o for o in _orders()}
    assert "old" not in orders and orders["L1"]["status"] == "accepted" and orders["L1"]["_first"] == T0 + timedelta(days=3)


def test_dry_run_clusters_anchored_dedupe_and_hypothetical_pnl():
    rows = R.dry_run_clusters(_orders(), MIDS, MAP, 600)
    at_040 = [r for r in rows if r["exchange_id"] == "1" and r["limit"] == 0.40]
    # 16 entries every 2 min: anchored 10-min windows -> clusters at 0, 10, 20, 30 min (5 + 5 + 5 + 1)
    assert [r["repeats"] for r in at_040] == [5, 5, 5, 1]
    assert all(r["repeats_chained_cluster"] == 16 for r in at_040)  # one chained intent
    assert at_040[0]["pnl"] == pytest.approx(10.0) and at_040[0]["pnl_ticks_per_share"] == pytest.approx(20.0)  # 100 x (0.50 - 0.40)
    other = next(r for r in rows if r["limit"] == 0.41)
    assert other["repeats"] == 1  # a different limit is a different intent
    no = next(r for r in rows if r["exchange_id"] == "2")
    assert no["mark_side"] == 0.75 and no["pnl"] == pytest.approx(2.5) and no["day"] == "2026-10-02" and no["race"] == "senate:GA"


def test_live_rows_use_the_canonical_response():
    rows = {r["ts"][:16]: r for r in R.live_rows(_orders(), MIDS, MAP)}
    canon = rows["2026-10-04T16:00"]
    assert (canon["sent_side"], canon["sent_action"], canon["side"], canon["action"]) == ("yes", "sell", "no", "buy")
    assert canon["quantity_traded"] == 150 and canon["remaining"] == 50
    assert canon["mark_side"] == 0.5 and canon["pnl"] == pytest.approx(150 * (0.50 - 0.55))
    unfilled = rows["2026-10-04T16:01"]
    assert unfilled["quantity_traded"] == 0 and unfilled["pnl"] is None


def test_without_mids_pnl_is_empty_not_zero():
    rows = R.dry_run_clusters(_orders(), {}, MAP, 600)
    assert all(r["pnl"] is None for r in rows)
    groups = R.group_rows("dry_run", rows, "day", "quantity")
    assert all(g["pnl"] is None and g["unvalued"] == g["fills"] for g in groups)


def test_group_rows_and_summary():
    orders = _orders()
    clusters = R.dry_run_clusters(orders, MIDS, MAP, 600)
    live = R.live_rows(orders, MIDS, MAP)
    by_day = {g["group"]: g for g in R.group_rows("dry_run", clusters, "day", "quantity")}
    assert by_day["2026-10-01"]["fills"] == 5 and by_day["2026-10-01"]["shares"] == 500
    assert by_day["2026-10-01"]["pnl"] == pytest.approx(4 * 10.0 + 9.0)
    summary = R.summarise(orders, clusters, live, 600)
    assert summary["dry_run"]["journal_entries"] == 18 and summary["dry_run"]["clusters_anchored"] == 6 and summary["dry_run"]["clusters_chained"] == 3
    assert summary["live"]["accepted"] == 2 and summary["live"]["filled_at_placement"] == 1 and summary["live"]["with_resting_remainder"] == 2
    assert summary["label"].startswith("HYPOTHETICAL")
    assert set(summary["halves"]["dry_run"]) == {"first_half", "second_half"}
    md = R.summary_markdown(summary)
    assert md.startswith("# Dry-run replay (HYPOTHETICAL)") and "expected 202" in md


@pytest.fixture
def quiet_main(monkeypatch, tmp_path):
    import config

    settings = config.Settings(sig_api_key="test-key", sig_base_url="https://sig.invalid/api/v1", dry_run=True, log_level="INFO",
                               log_dir=tmp_path, tournament_slug="t", oracle_staleness_threshold_s=10.0, cycle_interval_s=15.0)
    monkeypatch.setattr(config, "load_settings", lambda *a, **k: settings)
    monkeypatch.setattr(R, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(R.MarketMap, "load", classmethod(lambda cls, *a, **k: MAP))
    monkeypatch.setattr(R, "run_dir", lambda script: tmp_path / "out")
    monkeypatch.setattr(R, "research_reader", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no reader without --yes")))
    monkeypatch.setattr(R, "newest_raw", lambda root=None: None)  # never pick up a real saved read
    monkeypatch.delenv("RESEARCH_READS_PER_MIN", raising=False)
    (tmp_path / "out").mkdir()
    journal = tmp_path / "order_journal.jsonl"
    journal.write_text("\n".join(json.dumps(e) for e in JOURNAL) + "\n", encoding="utf-8")
    monkeypatch.setattr(R, "ORDER_JOURNAL_PATH", journal)
    return tmp_path


def test_main_offline_writes_counts(quiet_main, caplog):
    with caplog.at_level("INFO"):
        assert R.main([]) == 0
    out = quiet_main / "out"
    for name in ("dry_run_clusters.csv", "live_orders.csv", "by_day.csv", "by_race.csv", "summary.json", "summary.md"):
        assert (out / name).exists(), name
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["dry_run"]["clusters_anchored"] == 6 and summary["dry_run"]["pnl"] is None
    assert any(r.getMessage() == "Live accepted count differs from the expected one" for r in caplog.records)


def test_main_with_yes_reads_mids(quiet_main, monkeypatch):
    async def fake_mids(ids, rpm):
        assert ids == ["1", "2"]
        return MIDS

    monkeypatch.setattr(R, "_read_mids", fake_mids)
    assert R.main(["--yes"]) == 0
    summary = json.loads((quiet_main / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["dry_run"]["pnl"] == pytest.approx(4 * 10.0 + 9.0 + 2.5)  # four 0.40 clusters, the 0.41 one, the NO one
    assert summary["live"]["pnl"] == pytest.approx(150 * (0.50 - 0.55))


def test_script_imports_no_order_placement_code():
    import ast

    forbidden = {"strategy", "quoter", "ladder", "main", "fills", "markout", "exchange_api"}
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden


# ---------------------------------------------------------------------- calibration extension


def test_rows_carry_claimed_edge_and_oracle_source():
    rows = R.dry_run_clusters(_orders(), MIDS, MAP, 600)
    assert all(r["edge_claimed"] == 0.05 for r in rows) and all(r["oracle_source"] == "none" for r in rows)
    mapped = MarketMap({"1": MarketMapping("1", "1", "D AZ", Race("senate", "AZ"), "D", "tok", "TICK")})
    assert R.oracle_source(mapped, "1") == "both" and R.oracle_source(mapped, "9") == "none"


def test_mids_from_raw_reuses_the_latest_saved_read():
    records = [
        {"ts": "2026-10-12T10:00:00Z", "path": "/exchanges/prices", "body": {"data": [{"exchangeId": "1", "bestBid": 0.40, "bestAsk": 0.44}]}},
        {"ts": "2026-10-12T10:00:06Z", "path": "/tournaments/t", "body": {"id": "x"}},
        {"ts": "2026-10-12T11:00:00Z", "path": "/exchanges/prices", "body": {"data": [{"exchangeId": "1", "bestBid": 0.50, "bestAsk": 0.52},
                                                                                     {"exchangeId": "2", "bestBid": None, "bestAsk": 0.3}]}},
        {"ts": "2026-10-12T11:00:06Z", "path": "/exchanges/prices", "error": {"status": 500}},
    ]
    mids, mark_ts = R.mids_from_raw(records)
    assert mids == {"1": 0.51, "2": None} and mark_ts == "2026-10-12T11:00:00Z"
    assert R.mids_from_raw([]) == ({}, None)


def _fill(ts, race, shares, pnl, edge, entry=0.40, leg="yes", src="polymarket"):
    return {"ts": ts, "race": race, "cluster": race, "shares": shares, "pnl": pnl, "edge": edge, "entry": entry, "leg": leg, "oracle_source": src}


def test_calibration_variants_halves_and_verdict():
    # Higher claimed edge, WORSE markout: the verdict must say it does not rise.
    fills = [_fill(f"2026-10-0{d}T12:00:00", race, 100, pnl, edge)
             for d in (1, 2) for race, pnl, edge in (("senate:RI", -9.0, 0.08), ("senate:AZ", -5.0, 0.08), ("senate:GA", 1.0, 0.01), ("senate:NV", 2.0, 0.01))]
    rows, verdicts = R.calibration({"dry_run": fills, "live": []}, ["senate:RI"])
    hi_all = next(r for r in rows if (r["variant"], r["sample"], r["half"], r["dimension"], r["group"]) == ("all", "dry_run", "all", "claimed_edge", ">=0.07"))
    assert hi_all["fills"] == 4 and hi_all["races"] == 2 and hi_all["mean_markout_ticks"] == pytest.approx(-14.0)
    hi_ex = next(r for r in rows if (r["variant"], r["sample"], r["half"], r["dimension"], r["group"]) == ("excluding", "dry_run", "all", "claimed_edge", ">=0.07"))
    assert hi_ex["fills"] == 2 and hi_ex["races"] == 1 and hi_ex["t_race_clustered"] is None  # one race: no t
    assert verdicts["all"]["dry_run"]["all"] is False and verdicts["all"]["live"]["all"] is None
    assert {r["dimension"] for r in rows} == {"claimed_edge", "entry_price", "leg", "oracle_source"}
    first = [r for r in rows if r["half"] == "first_half" and r["variant"] == "all" and r["sample"] == "dry_run" and r["dimension"] == "leg" and r["group"] == "yes"]
    assert first[0]["fills"] == 4


def test_race_table_and_concentration():
    fills = [_fill("t1", "a", 100, -50.0, 0.05, entry=0.30), _fill("t2", "a", 300, -10.0, 0.05, entry=0.50),
             _fill("t3", "b", 100, 20.0, 0.05), _fill("t4", "c", 100, -40.0, 0.05)]
    rows, conc = R.race_table({"dry_run": fills})
    assert [r["race"] for r in rows] == ["a", "c", "b"] and rows[0]["mean_entry"] == pytest.approx(0.45) and rows[0]["pnl"] == -60.0
    assert conc["dry_run"]["worst"] == ["a", "c", "b"] and conc["dry_run"]["net_pnl"] == -80.0
    assert rows[0]["share_of_net_loss"] == pytest.approx(0.75)


def test_main_offline_reuses_saved_mids_and_writes_calibration(quiet_main, monkeypatch):
    raw = quiet_main / "raw.jsonl"
    raw.write_text(json.dumps({"ts": "2026-10-12T11:00:00Z", "path": "/exchanges/prices",
                               "body": {"data": [{"exchangeId": "1", "bestBid": 0.49, "bestAsk": 0.51}, {"exchangeId": "2", "bestBid": 0.24, "bestAsk": 0.26}]}}) + "\n",
                   encoding="utf-8")
    assert R.main(["--raw", str(raw)]) == 0
    out = quiet_main / "out"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["mark"]["from"] == str(raw) and summary["mark"]["at"] == "2026-10-12T11:00:00Z"
    assert summary["dry_run"]["pnl"] == pytest.approx(4 * 10.0 + 9.0 + 2.5)  # same as reading the mids now
    assert summary["calibration"]["excluded"] == list(R.DEFAULT_EXCLUDE) and "rows" not in summary["calibration"]
    assert (out / "calibration.csv").exists() and (out / "race_table.csv").exists()
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "## Calibration by claimed edge" in md and "| >=0.07 |" in md
