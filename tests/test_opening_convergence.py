"""scripts/opening_convergence.py on synthetic saved reads - offline, no network.

Run from the project root:  python -m pytest tests -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from market_mapper import MarketMap, MarketMapping, Race

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "opening_convergence.py"
_spec = importlib.util.spec_from_file_location("opening_convergence", SCRIPT)
C = importlib.util.module_from_spec(_spec)
sys.modules["opening_convergence"] = C
_spec.loader.exec_module(C)


def _m(eid, office, state, party, district=None):
    return MarketMapping(eid, f"m{eid}", f"{party} {state}", Race(office, state, district), party, None, None)


MAP = MarketMap({m.sig_exchange_id: m for m in [
    _m("1", "senate", "RI", "D"), _m("2", "senate", "RI", "R"),
    _m("3", "senate", "AZ", "D"),
    _m("1017", "house", "FL", "D", "22"), _m("1018", "house", "FL", "R", "22"),
]})


def _c(day_hour, close, o=None, trades=3, volume=100.0):
    day, hour = day_hour
    return {"time": f"2026-10-0{day}T{hour:02d}:00:00Z", "open": o if o is not None else close, "high": close + 0.01, "low": close - 0.01,
            "close": close, "tradeCount": trades, "volume": volume}


CANDLES = {
    "1": [_c((1, 16), 0.50), _c((1, 18), 0.80), _c((2, 10), 0.93), _c((1, 20), 0.85)],  # opens 0.50, oracle 0.96: converges
    "2": [_c((1, 17), 0.45), _c((1, 19), 0.55)],  # opens 0.45, oracle 0.04: moves AWAY
    "3": [_c((1, 16), 0.52), _c((1, 19), 0.50)],  # oracle 0.50: not mis-initialised
    "1017": [_c((1, 16), 0.48, o=0.40, trades=12, volume=900.0)],
}
RAW_MAP = {"generated_at": "2026-09-29T20:37:39+00:00", "entries": {
    "1": {"check_prices": {"polymarket": 0.97, "kalshi": 0.95}}, "2": {"check_prices": {"polymarket": 0.04}},
    "3": {"check_prices": {"polymarket": 0.5}}, "1017": {"check_prices": {"kalshi": 0.47}}, "1018": {"check_prices": {}},
}}


def test_inputs_from_saved_reads():
    records = [{"ts": "a", "path": "/tournaments/t/markets", "body": {"data": [{"id": "m", "title": "RI Senate", "exchanges": [{"id": "1"}]}]}},
               {"ts": "b", "path": "/exchanges/1/price-history", "body": {"candles": [_c((1, 16), 0.1)]}},
               {"ts": "c", "path": "/exchanges/1/price-history", "body": {"candles": CANDLES["1"]}},  # later read wins
               {"ts": "d", "path": "/exchanges/2/price-history", "error": {"status": 500}}]
    candles, titles = C.candles_from_raw(records)
    assert candles == {"1": CANDLES["1"]} and titles == {"1": "RI Senate"}
    mids, at = C.mids_from_raw([{"ts": "2026-10-12T11:00:00Z", "path": "/exchanges/prices", "body": {"data": [{"exchangeId": "1", "bestBid": 0.94, "bestAsk": 0.96}]}}])
    assert mids == {"1": 0.95} and at == "2026-10-12T11:00:00Z"
    assert C.saved_oracle_mids([{"ts": "x", "path": "oracle:mids", "body": {"mids": {"1": 0.97}}}]) == ({"1": 0.97}, "x")
    assert C.map_oracle_fair(RAW_MAP) == {"1": 0.96, "2": 0.04, "3": 0.5, "1017": 0.47, "1018": None}


def test_exchange_row_closes_and_convergence():
    row = C.exchange_row("1", CANDLES["1"], None, MAP, 0.96, 0.97, 0.95)
    assert row["mis_initialised"] is True and row["first_open"] == 0.50 and row["first_candle"] == "2026-10-01T16:00:00+00:00"
    assert (row["first_high"], row["first_low"], row["first_trades"], row["first_volume"]) == (0.51, 0.49, 3, 100.0)
    # 20:00Z: the 18:00 candle (ended 19:00) is the last finished one; the 20:00 candle has not ended
    assert row["close_2000"] == 0.80 and row["close_2000_candle"].startswith("2026-10-01T18:00")
    assert row["close_oct3"] == 0.93 and row["toward_2000"] == pytest.approx(0.30) and row["gap_at_open"] == pytest.approx(0.46)
    assert row["gap_closed_2000"] == pytest.approx(0.30 / 0.46, abs=1e-3) and row["gap_closed_now"] == pytest.approx(0.45 / 0.46, abs=1e-3)
    assert row["race"] == "senate:RI"


def test_analyse_counts_sums_and_sorts():
    result = C.analyse(CANDLES, {}, MAP, C.map_oracle_fair(RAW_MAP), {}, {"1": 0.95})
    s = result["summary"]
    assert s["mis_initialised"] == 2 and s["mis_initialised_close_based"] == 2 and s["moved_toward"] == 1 and s["moved_away"] == 1
    assert s["coverage"]["candles_starting_at"]["2026-10-01T17:00Z"] == 1 and s["coverage"]["with_2000_close_all"] == 4  # 1017: its 16:00 candle carries forward
    assert s["sum_toward_oracle_2000"] == pytest.approx(0.30 - 0.10) and s["sum_positive_toward_oracle_2000"] == pytest.approx(0.30)
    assert [r["exchange_id"] for r in result["flagged"]] == ["1", "2"]  # sorted by the move toward the oracle
    assert s["label"].startswith("PROXY")


def test_controls_take_the_opening_hour():
    controls = {c["exchange_id"]: c for c in C.control_rows(CANDLES, MAP, C.map_oracle_fair(RAW_MAP), races=("house:FL-22", "house:NH-01"))}
    d = controls["1017"]
    assert (d["open"], d["close"], d["trades"], d["volume"]) == (0.40, 0.48, 12, 900.0) and d["close_minus_oracle"] == pytest.approx(0.01)
    assert d["open_minus_oracle"] == pytest.approx(-0.07)
    assert controls[None]["race"] == "house:NH-01" and controls[None]["candle"] == "race not in the market map"
    assert set(C.CONTROL_RACES) >= {"senate:RI", "senate:DE", "senate:OK", "senate:FL", "house:NH-01"}
    assert controls["1018"]["candle"] is None and controls["1018"]["trades"] == 0  # no trade in the opening hour


def test_main_offline_writes_outputs(monkeypatch, tmp_path):
    import config

    tape = tmp_path / "tape.jsonl"
    tape.write_text("\n".join(json.dumps({"ts": "x", "path": f"/exchanges/{e}/price-history", "body": {"candles": cs}}) for e, cs in CANDLES.items()) + "\n", encoding="utf-8")
    mids = tmp_path / "mids.jsonl"
    mids.write_text(json.dumps({"ts": "2026-10-12T11:00:00Z", "path": "/exchanges/prices", "body": {"data": [{"exchangeId": "1", "bestBid": 0.94, "bestAsk": 0.96}]}}) + "\n",
                    encoding="utf-8")
    map_path = tmp_path / "market_map.json"
    map_path.write_text(json.dumps(RAW_MAP), encoding="utf-8")
    settings = config.Settings(sig_api_key="", sig_base_url="https://sig.invalid/api/v1", dry_run=True, log_level="INFO",
                               log_dir=tmp_path, tournament_slug="t", oracle_staleness_threshold_s=10.0, cycle_interval_s=15.0)
    monkeypatch.setattr(config, "load_settings", lambda *a, **k: settings)
    monkeypatch.setattr(C, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(C.MarketMap, "load", classmethod(lambda cls, *a, **k: MAP))
    monkeypatch.setattr(C, "MARKET_MAP_PATH", map_path)
    monkeypatch.setattr(C, "OUT_ROOT", tmp_path / "edge_scan")  # no saved position_review read
    monkeypatch.setattr(C, "run_dir", lambda script: tmp_path / "out")
    (tmp_path / "out").mkdir()
    assert C.main(["--tape-raw", str(tape), "--mids-raw", str(mids)]) == 0
    md = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    assert md.startswith("# Opening convergence (PROXY)") and "**2**" in md and "not P&L" in md and "house:FL-22" in md
    assert (tmp_path / "out" / "opening_convergence.csv").exists() and (tmp_path / "out" / "controls.csv").exists()


def test_main_without_saved_history_fails_cleanly(monkeypatch, tmp_path):
    import config

    settings = config.Settings(sig_api_key="", sig_base_url="https://sig.invalid/api/v1", dry_run=True, log_level="INFO",
                               log_dir=tmp_path, tournament_slug="t", oracle_staleness_threshold_s=10.0, cycle_interval_s=15.0)
    monkeypatch.setattr(config, "load_settings", lambda *a, **k: settings)
    monkeypatch.setattr(C, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(C, "OUT_ROOT", tmp_path / "edge_scan")
    assert C.main([]) == 2


def test_script_makes_no_reads():
    import ast

    forbidden = {"strategy", "quoter", "ladder", "main", "fills", "markout", "exchange_api", "oracle_api", "aiohttp"}
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden
            assert "research_reader" not in [a.name for a in node.names]


def test_regression_20z_closes_are_counted_over_all_exchanges():
    """The bug behind with_2000_close=0: it counted only FLAGGED exchanges, so with nothing
    flagged it was 0 even though every exchange had a 20:00Z close."""
    candles = {str(i): [_c((1, 16), 0.70), _c((1, 19), 0.72)] for i in range(5)}  # nothing near 0.50: none flagged
    result = C.analyse(candles, {}, MAP, {str(i): 0.95 for i in range(5)}, {}, {})
    s = result["summary"]
    assert s["mis_initialised"] == 0 and s["flagged_with_2000_close"] == 0
    assert s["coverage"]["with_2000_close_all"] == 5 and s["coverage"]["candles_starting_at"]["2026-10-01T19:00Z"] == 5


def test_open_based_definition_differs_from_close_based():
    # Opens at 0.50, trades up to 0.90 within the first hour; oracle 0.96.
    row = C.exchange_row("1", [_c((1, 16), 0.90, o=0.50), _c((1, 19), 0.93)], None, MAP, 0.96, None, None)
    assert row["mis_initialised"] is True and row["mis_initialised_close"] is False
    assert row["move_to_2000"] == pytest.approx(0.43) and row["gap_closed_2000"] == pytest.approx(0.43 / 0.46, abs=1e-3)


def test_epoch_candle_times_are_read():
    from datetime import datetime, timezone

    t16 = int(datetime(2026, 10, 1, 16, tzinfo=timezone.utc).timestamp() * 1000)  # epoch ms
    candles = [{"time": t16, "open": 0.5, "high": 0.5, "low": 0.5, "close": 0.5}, {"time": t16 + 3 * 3_600_000, "open": 0.8, "high": 0.8, "low": 0.8, "close": 0.8}]
    row = C.exchange_row("1", candles, None, MAP, 0.96, None, None)
    assert row["first_open"] == 0.5 and row["close_2000"] == 0.8
