"""scripts/basket_arb_report.py on a synthetic probe run - offline, hand-computed cases.

Run from the project root:  python -m pytest tests/test_basket_arb_report.py -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from market_mapper import MarketMap, MarketMapping, Race

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "basket_arb_report.py"
_spec = importlib.util.spec_from_file_location("basket_arb_report", SCRIPT)
B = importlib.util.module_from_spec(_spec)
sys.modules["basket_arb_report"] = B
_spec.loader.exec_module(B)


def _m(eid, state, party):
    return MarketMapping(eid, f"m{eid}", f"{party} {state}", Race("senate", state), party, None, None)


MAP = MarketMap({m.sig_exchange_id: m for m in [_m("1", "RI", "D"), _m("2", "RI", "R"), _m("3", "AZ", "D"), _m("4", "AZ", "R"), _m("5", "NV", "D")]})
T0 = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
# RI bid sums per minute: 1.03, 1.03, 1.02, 0.99, 1.04, 0.90 -> runs of 3 and 1 mispriced observations.
RI = [(0.55, 0.48), (0.55, 0.48), (0.55, 0.47), (0.50, 0.49), (0.56, 0.48), (0.50, 0.40)]


def _iso(t):
    return t.isoformat().replace("+00:00", "Z")


def _records():
    recs = []
    for k, (d, r) in enumerate(RI):
        t = T0 + timedelta(minutes=k)
        recs.append({"ts": _iso(t), "path": "/relationships/constraints", "body": {"data": []}})
        az4 = None if k == 5 else {"exchangeId": "4", "bestBid": 0.50, "bestAsk": 0.52}  # AZ R unquoted in the last sample
        rows = [{"exchangeId": "1", "bestBid": d}, {"exchangeId": "2", "bestBid": r}, {"exchangeId": "3", "bestBid": 0.40}] + ([az4] if az4 else [])
        recs.append({"ts": _iso(t + timedelta(seconds=6)), "path": "/exchanges/prices", "body": {"data": rows}})
    # A failed constraints read: the prices after it must not join the previous sample.
    recs.append({"ts": _iso(T0 + timedelta(minutes=6)), "path": "/relationships/constraints", "error": {"status": 503}})
    recs.append({"ts": _iso(T0 + timedelta(minutes=6, seconds=6)), "path": "/exchanges/prices", "body": {"data": [{"exchangeId": "1", "bestBid": 0.99}]}})
    return recs


def test_samples_from_raw_rebuild_legs():
    samples = B.samples_from_raw(_records())
    assert len(samples) == 6 and samples[0]["prices"]["1"]["bestBid"] == 0.55
    assert samples[-1]["prices"]["1"]["bestBid"] == 0.50  # not overwritten by the orphan prices read
    assert B.interval_from([], samples) == 60.0 and B.interval_from([{"interval_s": 30}], samples) == 30.0


def test_race_legs_and_depth_from_saved_books():
    assert B.race_legs(MAP) == {"senate:AZ": ["3", "4"], "senate:RI": ["1", "2"]}  # NV has one party: no basket
    books = [{"ts": "a", "path": "/exchanges/1/orderbook", "body": {"bids": [{"price": 0.54, "quantity": 900}, {"price": 0.55, "quantity": 500}]}},
             {"ts": "b", "path": "/tournaments/t/markets/m2", "body": {"exchanges": [{"id": "2", "bids": [{"price": 0.48, "quantity": 40}]}]}},
             {"ts": "c", "path": "/exchanges/3/orderbook", "error": {"status": 500}}]
    depth = B.depth_from_books(books)
    assert depth == {"1": {"quantity": 500.0, "price": 0.55, "ts": "a"}, "2": {"quantity": 40.0, "price": 0.48, "ts": "b"}}


def test_race_analysis_hand_computed():
    samples = B.samples_from_raw(_records())
    result = B.analyse(samples, B.race_legs(MAP), 60.0, {}, size=100.0, min_obs=2)
    ri = next(r for r in result["rows"] if r["race"] == "senate:RI")
    assert (ri["observations"], ri["mispriced"], ri["mispriced_share"]) == (6, 4, 0.6667)
    # ticks after 1 tick per leg: (0.03-0.01)/0.005 = 4, 4, 2, 6 -> p10 2.6, median 4, p90 5.4
    assert (ri["ticks_p10"], ri["ticks_median"], ri["ticks_p90"]) == (2.6, 4.0, 5.4)
    assert ri["capital_median"] == pytest.approx(0.97) and ri["break_even_fee_per_leg_median"] == pytest.approx(0.015)
    assert ri["profit_per_capital_median"] == pytest.approx(0.02 / 0.97) and ri["profit_per_capital_haircut_median"] == pytest.approx(0.01 / 0.97)
    assert (ri["runs"], ri["runs_ge_min_obs"], ri["run_obs_max"], ri["run_duration_s_median"]) == (2, 1, 3, 60.0)
    # one run >= 2 obs, executed once at its first prices: (0.03 - 0.01 - 0.01) * 100 = 1.0 over 0.1 h
    assert ri["size_used"] == 100.0 and ri["depth_shares"] is None and ri["profit_per_hour"] == pytest.approx(10.0)
    assert ri["per_day_mispriced_share"] == {"2026-10-09": 0.6667} and ri["rank"] == 1
    az = next(r for r in result["rows"] if r["race"] == "senate:AZ")
    assert az["observations"] == 5 and az["mispriced"] == 0 and az["profit_per_hour"] == 0.0
    total = result["total"]
    assert total["observations"] == 11 and total["mispriced"] == 4 and total["hours"] == 0.1 and total["profit_per_hour"] == pytest.approx(10.0)


def test_depth_caps_the_size():
    samples = B.samples_from_raw(_records())
    depth = {"1": {"quantity": 500.0, "price": 0.55, "ts": "a"}, "2": {"quantity": 40.0, "price": 0.48, "ts": "b"}}
    ri = next(r for r in B.analyse(samples, B.race_legs(MAP), 60.0, depth, size=100.0)["rows"] if r["race"] == "senate:RI")
    assert ri["depth_shares"] == 40.0 and ri["size_used"] == 40.0 and ri["profit_per_hour"] == pytest.approx(4.0)
    assert "not read with the sample" in ri["depth_from"]


def test_main_writes_outputs(monkeypatch, tmp_path):
    import config

    run = tmp_path / "relationship_probe" / "2026-10-09"
    run.mkdir(parents=True)
    (run / "raw.jsonl").write_text("\n".join(json.dumps(r) for r in _records()) + "\n", encoding="utf-8")
    (run / "samples.jsonl").write_text(json.dumps({"ts": "x", "interval_s": 60, "rows": []}) + "\n", encoding="utf-8")
    settings = config.Settings(sig_api_key="", sig_base_url="https://sig.invalid/api/v1", dry_run=True, log_level="INFO",
                               log_dir=tmp_path, tournament_slug="t", oracle_staleness_threshold_s=10.0, cycle_interval_s=15.0)
    monkeypatch.setattr(config, "load_settings", lambda *a, **k: settings)
    monkeypatch.setattr(B, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(B.MarketMap, "load", classmethod(lambda cls, *a, **k: MAP))
    assert B.main([str(run)]) == 0
    md = (run / "analysis" / "summary.md").read_text(encoding="utf-8")
    assert md.startswith("# Basket (constraint) arbitrage") and "HYPOTHETICAL" in md and "No fee is assumed" in md and "| 1 | senate:RI |" in md
    assert (run / "analysis" / "races.csv").exists()
    assert B.main([str(tmp_path / "missing")]) == 2


def test_script_makes_no_reads():
    import ast

    forbidden = {"strategy", "quoter", "ladder", "main", "fills", "markout", "exchange_api", "oracle_api", "aiohttp"}
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden
            assert "research_reader" not in [a.name for a in node.names]
