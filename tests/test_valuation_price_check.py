"""scripts/valuation_price_check.py on a synthetic raw.jsonl - offline, no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "valuation_price_check.py"
_spec = importlib.util.spec_from_file_location("valuation_price_check", SCRIPT)
V = importlib.util.module_from_spec(_spec)
sys.modules["valuation_price_check"] = V
_spec.loader.exec_module(V)


def _rec(ts, path, body):
    return {"ts": ts, "method": "GET", "path": path, "params": {}, "body": body}


def _prices(*rows):
    return {"data": [{"exchangeId": e, "bestBid": b, "bestAsk": a, "latestPrice": l} for e, b, a, l in rows]}


POSITIONS = {"positions": [
    {"exchangeId": "1", "quantity": 1000, "currentPrice": 0.43, "marketValue": 430.0},  # = last
    {"exchangeId": "2", "quantity": -500, "currentPrice": 0.30, "marketValue": 350.0},  # NO, YES-normalised mark = last
    {"exchangeId": "3", "quantity": 200, "currentPrice": 0.20, "marketValue": 40.0},  # only in the earlier prices read
    {"exchangeId": "4", "quantity": 0, "currentPrice": None, "marketValue": 0.0},  # flat: ignored
]}
RECORDS = [
    _rec("2026-10-07T12:00:00.000Z", "/exchanges/prices", _prices(("1", 0.10, 0.20, 0.15), ("3", 0.18, 0.24, 0.20))),
    _rec("2026-10-07T12:05:00.000Z", "/portfolio/positions", POSITIONS),
    _rec("2026-10-07T12:05:06.000Z", "/exchanges/prices", _prices(("1", 0.40, 0.44, 0.43), ("2", 0.28, 0.34, 0.30))),
    _rec("2026-10-07T12:05:12.000Z", "/portfolio/pnl", {"totalAccountValue": 1.0}),
    {"ts": "2026-10-07T12:05:18.000Z", "method": "GET", "path": "/x", "params": {}, "error": {"status": 404}},
]


def test_pairing_prefers_the_snapshot_prices_and_reports_age():
    positions, prices, meta = V.pair_positions_with_prices(RECORDS)
    assert len(positions) == 4 and meta["positions_at"] == "2026-10-07T12:05:00.000Z"
    assert prices["1"]["bestBid"] == 0.40 and prices["1"]["_source"] == "snapshot" and prices["1"]["_age_s"] == 6.0
    assert prices["3"]["_source"] == "earlier" and prices["3"]["_age_s"] == -300.0
    assert "2" in prices


def test_analyse_finds_last_trade_and_yes_convention():
    positions, prices, _ = V.pair_positions_with_prices(RECORDS)
    rows, summary = V.analyse(positions, prices)
    assert [r["exchange_id"] for r in rows] == ["1", "2", "3"]  # largest |marketValue| first, flat dropped
    assert summary["currentPrice_convention"] == "yes"
    by_id = {r["exchange_id"]: r for r in rows}
    assert by_id["1"]["matches"] == "last" and by_id["2"]["matches"] == "last" and by_id["2"]["mark_yes"] == 0.30
    assert by_id["3"]["matches"] == "last" and by_id["3"]["price_source"] == "earlier"  # mid 0.21 != 0.20
    assert summary["by_rule"]["last"] == {"comparable": 3, "matches": 3, "share": 1.0, "median_abs_diff": 0.0}
    assert summary["verdict"] == "INSUFFICIENT DATA" and summary["explained_by"] == []  # < 5 comparable positions


def test_verdict_names_the_rule_with_enough_positions():
    positions = {"positions": [{"exchangeId": str(i), "quantity": 100, "currentPrice": 0.42, "marketValue": 42.0} for i in range(6)]}
    records = [_rec("2026-10-07T12:00:00Z", "/portfolio/positions", positions),
               _rec("2026-10-07T12:00:06Z", "/exchanges/prices", _prices(*[(str(i), 0.40, 0.44, 0.45) for i in range(6)]))]
    rows, summary = V.analyse(*V.pair_positions_with_prices(records)[:2])
    assert summary["explained_by"] == ["mid"] and summary["verdict"] == "mid"
    assert summary["vs_spread"] == {"inside spread": 6}


def test_no_positions_is_insufficient_data():
    rows, summary = V.analyse(*V.pair_positions_with_prices([])[:2])
    assert rows == [] and summary["verdict"] == "INSUFFICIENT DATA"


def test_main_writes_outputs_next_to_raw(tmp_path, monkeypatch):
    import config

    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n", encoding="utf-8")
    settings = config.Settings(sig_api_key="", sig_base_url="https://sig.invalid/api/v1", dry_run=True, log_level="INFO",
                               log_dir=tmp_path, tournament_slug="t", oracle_staleness_threshold_s=10.0, cycle_interval_s=15.0)
    monkeypatch.setattr(config, "load_settings", lambda *a, **k: settings)
    monkeypatch.setattr(V, "setup_logging", lambda *a, **k: None)
    assert V.main(["--raw", str(raw)]) == 0
    out = json.loads((tmp_path / "valuation_price.json").read_text(encoding="utf-8"))
    assert out["positions_compared"] == 3 and out["spec"]["field"].startswith("Position.currentPrice")
    assert (tmp_path / "valuation_price.csv").exists()


def test_offline_script_imports_no_api_client():
    import ast

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    modules = {(n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m.split(".")[0] in {"exchange_api", "strategy", "quoter", "ladder", "aiohttp"} for m in modules)
